import copy
import datetime as dt
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp

from data_tools.dataloader import get_client_dataloader
from predict import local_validate, minigpt_validate
from train import execute_epoch
from utils.grad_traceback import get_downscale_index
from utils.utils import save_checkpoint

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        self.args = args
        self.global_model = global_model

        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios

        assert len(self.vertical_scale_ratios) == len(self.horizontal_scale_ratios) == len(self.client_split_ratios)

        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.num_levels = len(self.vertical_scale_ratios)
        
        # 架构适配：MiniGPT大模型使用LoRA分层配置，传统模型使用梯度追踪
        self.is_minigpt = 'MiniGPT' in str(type(global_model))
        
        if self.is_minigpt:
            print("=" * 60)
            print("🔧 检测到MiniGPT模型")
            print("=" * 60)
            print("✓ 跳过梯度追踪（避免显存溢出）")
            print("✓ 启用LoRA资源自适应配置")
            print("-" * 60)
            
            # 为不同资源等级定义LoRA配置（Level-Adaptive策略）
            # 配置格式: {'layers': 训练的层列表, 'rank': LoRA秩, 'alpha': LoRA alpha, 
            #           'lr': 学习率, 'grad_clip': 梯度裁剪, 'weight_decay': 权重衰减}
            self.lora_configs = {
                0: {
                    'layers': list(range(8)),  
                    'rank': 8,     
                    'alpha': 16,   
                    'desc': '低资源(前8层)',
                    'lr': 1e-6,           # 极低学习率（8层最不稳定）
                    'grad_clip': 1.5,     # 提高裁剪阈值
                    'weight_decay': 0.02  # 强正则化
                },
                1: {
                    'layers': list(range(16)),
                    'rank': 16,    
                    'alpha': 32,   
                    'desc': '中低资源(前16层)',
                    'lr': 2e-6,
                    'grad_clip': 2.5,
                    'weight_decay': 0.015
                },
                2: {
                    'layers': list(range(24)),
                    'rank': 32,    
                    'alpha': 64,   
                    'desc': '中高资源(前24层)',
                    'lr': 3e-6,
                    'grad_clip': 4.0,
                    'weight_decay': 0.01
                },
                3: {
                    'layers': list(range(32)),
                    'rank': 64,    
                    'alpha': 128,  
                    'desc': '全资源(32层)',
                    'lr': 5e-6,
                    'grad_clip': 5.0,  # 提高到5.0，允许更大梯度
                    'weight_decay': 0.01
                },
            }
            
            for level, config in self.lora_configs.items():
                print(f"  Level {level}: {config['desc']} - rank={config['rank']}, alpha={config['alpha']}")
            
            print("=" * 60)
            self.idx_dicts = None  # 不使用索引字典
        else:
            # 传统模型仍使用梯度追踪
            self.idx_dicts = [get_downscale_index(self.global_model, args, s) for s in self.vertical_scale_ratios]   
            self.lora_configs = None
            self.is_minigpt = False
        
        self.client_groups = client_groups

        self.lora_target_modules = getattr(global_model, "lora_target_modules", [])

        self.use_gpu = args.use_gpu

    def fed_train(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params):

        scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' +
                  '\tlocal_val_acc1' * self.num_levels]
        best_acc1, best_round = 0.0, 0

        # 如果没有提供客户端分组，则根据客户端分组比例随机分配客户端。
        # pre-assignment of levels to clients (needs to be saved for inference)
        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs = np.random.permutation(client_idxs)
            client_groups = []
            s = 0
            for i, ratio in enumerate(self.client_split_ratios):
                # 最后一组包含所有剩余客户端
                if i == len(self.client_split_ratios) - 1:
                    e = len(shuffled_client_idxs)
                else:
                    e = s + int(len(shuffled_client_idxs) * ratio)
                client_groups.append(shuffled_client_idxs[s: e])
                s = e
            self.client_groups = client_groups
            
            # 打印客户端分组信息
            print("=" * 60)
            print("📊 客户端资源等级分配")
            print("=" * 60)
            for level, group in enumerate(client_groups):
                if self.is_minigpt:
                    config = self.lora_configs[level]
                    print(f"Level {level}: {len(group)}个客户端 → {len(config['layers'])}层LoRA, rank={config['rank']}")
                else:
                    print(f"Level {level}: {len(group)}个客户端 → scale={self.vertical_scale_ratios[level]}")
            print("=" * 60)

            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(self.client_groups, f)

        # 进入训练轮次循环，每轮调用 execute_round 方法进行训练和验证。
        # 记录每轮的训练损失、验证损失和准确率，并保存最佳模型。
        # 【优化】初始化best_val_loss用于MiniGPT模型
        best_val_loss = float('inf')
        
        for round_idx in range(args.start_round, self.num_rounds):

            print(f'\n | Global Training Round : {round_idx + 1} |\n')

            train_loss, val_results, local_val_results = \
                self.execute_round(train_set, val_set, user_groups, criterion, args, batch_size,
                                   train_params, round_idx)

            # 每轮结束后强制清理GPU缓存，防止内存累积
            torch.cuda.empty_cache()
            
            val_loss, val_acc1, val_acc5, _, _ = val_results

            scores.append(('{}' + '\t{:.4f}' * int(6 + self.num_levels))
                          .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                                  local_val_results[-1][1], local_val_results[-1][2],
                                  *[l[1] for l in local_val_results[:-1]]))

            # 【优化】MiniGPT模型基于val_loss判断最佳模型（loss越低越好），传统模型基于val_acc1
            if self.is_minigpt:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_round = round_idx
                    print(f'⭐ 最佳验证Loss: {best_val_loss:.4f} (Round {round_idx + 1})')
            else:
                is_best = val_acc1 > best_acc1
                if is_best:
                    best_acc1 = val_acc1
                    best_round = round_idx
                    print('Best var_acc1 {}'.format(best_acc1))

            model_filename = 'checkpoint_%03d.pth.tar' % round_idx
            # 【修复】只保存LoRA参数，避免保存27GB基础模型
            if args.arch == 'MiniGPTv2':
                state_dict_to_save = {
                    k: v for k, v in self.global_model.state_dict().items()
                    if 'lora' in k.lower()
                }
                print(f"💾 保存LoRA checkpoint: {len(state_dict_to_save)}个参数")
            else:
                state_dict_to_save = self.global_model.state_dict()
            
            # 【优化】保存checkpoint状态，包含val_loss信息
            save_checkpoint({
                'round': round_idx,
                'arch': args.arch,
                'state_dict': state_dict_to_save,
                'best_acc1': best_acc1,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss if self.is_minigpt else None,
            }, args, is_best, model_filename, scores)

        return best_acc1, best_round

    # 根据客户端索引返回其对应的复杂度级别。如果客户端不在任何分组中，则返回 -1。
    def get_level(self, client_idx):
        # Return the complexity level of given client, starts with 0
        try:
            level = np.where([client_idx in c for c in self.client_groups])[0][0]
        except:
            # client will be skipped
            level = -1

        return level

    def execute_round(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params, round_idx):
        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)
        client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        # 为每个客户端获取本地数据加载器、复杂度级别、缩放比例和本地模型。
        client_train_loaders = [get_client_dataloader(train_set, user_groups[0][client_idx], args, batch_size) for
                                client_idx in client_idxs]
        levels = [self.get_level(client_idx) for client_idx in client_idxs]
        scales = [self.vertical_scale_ratios[level] for level in levels]
        local_models = [self.get_local_split(levels[i], scales[i]) for i in range(len(client_idxs))]
        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]

        pool_args = [train_set, user_groups, criterion, args, batch_size, train_params, round_idx]
        local_weights = []
        local_losses = []
        local_grad_flags = []

        pool_args.append(None)

        for i, client_idx in enumerate(client_idxs):
            # 传递federator的lora_configs给客户端训练函数
            client_args = pool_args + [local_models[i], client_train_loaders[i], levels[i], scales[i], 
                                       h_scale_ratios[i], client_idx, self.lora_configs]
            result = execute_client_round(client_args)

            if args.use_gpu:
                for k, v in result[0].items():
                    result[0][k] = v.cuda(0)

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(client_idxs)} completely finished')

        train_loss = sum(local_losses) / len(client_idxs)

        # Update the global model
        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model)
        self.global_model.load_state_dict(global_weights)

        # Validation for all clients
        if self.client_split_ratios[-1] == 0:
            level = np.where(self.client_split_ratios)[0].tolist()[-1]
            scale = self.vertical_scale_ratios[level]
            global_model = self.get_local_split(level, scale)
            if self.use_gpu:
                global_model = global_model.cuda()
        else:
            # 对于MiniGPT大模型，直接使用引用，避免deepcopy导致OOM
            if self.is_minigpt:
                global_model = self.global_model
            else:
                global_model = copy.deepcopy(self.global_model)

        # 验证阶段：MiniGPT模型使用简化验证（仅计算Loss）
        if self.is_minigpt:
            print("=" * 60)
            print("📊 MiniGPT模型验证（简化版：仅计算Loss）")
            print("=" * 60)
            
            # 创建验证数据加载器（使用部分验证数据）
            # 这里简化为使用一个客户端的验证数据
            if len(user_groups[1]) > 0:
                val_client_idx = 0  # 使用第一个客户端的验证数据
                val_loader = get_client_dataloader(val_set, user_groups[1][val_client_idx], args, batch_size=1)
                
                try:
                    avg_val_loss = minigpt_validate(global_model, val_loader, criterion, args)
                except Exception as e:
                    print(f"⚠️  验证过程出错: {e}")
                    avg_val_loss = 0.0
            else:
                print("⚠️  没有验证数据，跳过验证")
                avg_val_loss = 0.0
            
            # 返回兼容的格式（使用Loss作为主要指标）
            val_results = (avg_val_loss, 0.0, 0.0, np.array([0.0]), np.array([0.0]))
            local_val_results = [(avg_val_loss, 0.0, 0.0) for _ in range(self.num_levels + 1)]
            print("=" * 60)
        else:
            # 传统CNN模型：正常执行验证
            val_results, local_val_results = local_validate(self, val_set, user_groups[1], criterion, args, 512,
                                                            global_model)

        # 【强化内存清理】防止DataLoader worker进程泄漏
        # 1. 显式关闭所有DataLoader
        for loader in client_train_loaders:
            # 清理DataLoader的迭代器和worker
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except:
                    pass
                del loader._iterator
        
        # 2. 删除所有临时变量
        del client_train_loaders, local_models, local_weights, local_losses, local_grad_flags
        if self.is_minigpt and 'val_loader' in locals():
            del val_loader
        
        # 3. 强制Python垃圾回收
        import gc
        gc.collect()
        
        # 4. 清理GPU缓存
        torch.cuda.empty_cache()
        
        return train_loss, val_results, local_val_results

    def average_weights(self, w, grad_flags, levels, model):
        """
        聚合多个客户端的模型参数
        - 对于MiniGPT：仅聚合LoRA参数（简单平均）
        - 对于传统模型：使用原始的梯度感知聚合逻辑
        """
        # 对于MiniGPT大模型，避免deepcopy整个state_dict（会OOM）
        # 改为直接使用model.state_dict()的引用，然后只修改LoRA参数
        if self.is_minigpt:
            w_avg = model.state_dict()
        else:
            w_avg = copy.deepcopy(model.state_dict())
        
        # 判断是否为LoRA参数
        def is_lora_param(key):
            return "lora" in key or any(module in key for module in self.lora_target_modules)
        
        # MiniGPT模型：简化的LoRA参数聚合
        if self.is_minigpt:
            lora_param_count = 0
            for key in w_avg.keys():
                # 只处理LoRA参数（因为客户端只返回LoRA参数）
                if is_lora_param(key):
                    # 检查是否所有客户端都有这个key
                    if all(key in w_ for w_ in w):
                        lora_param_count += 1
                        # 收集有梯度更新的客户端参数
                        updated_params = [w_[key] for i, w_ in enumerate(w) if grad_flags[i].get(key, False)]
                        
                        if updated_params:
                            # 简单平均
                            w_avg[key] = sum(updated_params) / len(updated_params)
                        # else: 保持全局模型的原始值
                # 非LoRA参数（BatchNorm等）：保持全局模型的原始值
                # 不做任何操作，因为客户端没有返回这些参数
            
            print(f"✓ 聚合完成：{lora_param_count}个LoRA参数已更新")
            return w_avg
        
        # 传统模型：使用原始的梯度追踪聚合逻辑
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                w_avg[key] = w[0][key]
                continue

            if 'running' in key:
                w_avg[key] = sum([w_[key] for w_ in w]) / len(w)
                continue

            if is_lora_param(key):
                tmp = torch.zeros_like(w_avg[key])
                count = torch.zeros_like(tmp)
                for i in range(len(w)):
                    if grad_flags[i][key]:
                        idx = self.idx_dicts[levels[i]][key]
                        idx = self.fix_idx_array(idx, w[i][key].shape)
                        tmp[idx] += w[i][key].flatten()
                        count[idx] += 1
                w_avg[key][count != 0] = tmp[count != 0]
                count[count == 0] = 1
                w_avg[key] = w_avg[key] / count
                
        return w_avg

    # 根据输入的二进制掩码和本地形状返回输出形状。
    def get_idx_shape(self, inp, local_shape):
        # Return the output shape for binary mask input
        # [[1, 1, 0], [1, 1, 0], [0, 0, 0,]] -> [2, 2]
        if any([s == 0 for s in inp.shape]):
            print('Indexing error')
            raise RuntimeError

        if len(local_shape) == 4:
            dim_1 = inp.shape[2] // 2
            dim_2 = inp.shape[3] // 2
            idx_shape = (inp[:, 0, dim_1, dim_2].sum().item(),
                         inp[0, :, dim_1, dim_2].sum().item(), *local_shape[2:])
        elif len(local_shape) == 2:
            idx_shape = (inp[:, 0].sum().item(),
                         inp[0, :].sum().item())
        else:
            idx_shape = (inp.sum(),)

        return idx_shape

    # 修复索引数组，确保其形状与本地形状匹配。
    def fix_idx_array(self, idx_array, local_shape):
        idx_shape = self.get_idx_shape(idx_array, local_shape)
        if all([idx_shape[i] >= local_shape[i] for i in range(len(local_shape))]):
            pass
        else:
            idx_array = idx_array[idx_array.sum(dim=1).argmax()].repeat((idx_array.shape[0], 1))
            idx_shape = self.get_idx_shape(idx_array, local_shape)

        ind_list = [slice(None)] * len(idx_array.shape)
        for i in range(len(local_shape)):

            lim = idx_array.shape[i]
            while idx_shape[i] != local_shape[i]:
                lim -= 1
                ind_list[i] = slice(0, lim)
                idx_shape = self.get_idx_shape(idx_array[tuple(ind_list)], local_shape)

        tmp = torch.zeros_like(idx_array, dtype=bool)
        tmp[tuple(ind_list)] = idx_array[tuple(ind_list)]
        idx_array = tmp

        if len(idx_array.shape) == 4:
            dim_1 = idx_array.shape[2] // 2
            dim_2 = idx_array.shape[3] // 2
            if idx_array.sum(dim=0).sum(dim=0)[0, 0] != idx_array.sum(dim=0).sum(dim=0)[dim_1, dim_2]:
                idx_array = idx_array[:, :, dim_1, dim_2].repeat(idx_array.shape[2], idx_array.shape[3], 1, 1).permute(
                    2, 3, 0, 1)
        return idx_array

    def get_local_split(self, level, scale):
        """
        为不同资源等级的客户端创建本地模型
        - 对于MiniGPT：共享base model，仅配置不同的LoRA参数
        - 对于传统模型：使用原始的参数裁剪逻辑
        """
        # MiniGPT模型：使用参数共享 + LoRA配置
        if self.is_minigpt:
            # 处理异常level（-1表示客户端不在任何分组中）
            if level < 0 or level >= len(self.lora_configs):
                print(f"⚠️  警告：客户端level {level}无效，使用默认配置（Level 0）")
                level = 0
            
            # 直接返回全局模型的引用（共享base model）
            # 在训练时通过冻结/解冻特定LoRA层来实现资源自适应
            # 注意：这里返回的是引用，不是副本！
            print(f"✓ 客户端Level {level}: LoRA配置 - 层数={len(self.lora_configs[level]['layers'])}, rank={self.lora_configs[level]['rank']}")
            return self.global_model
        
        # 传统模型：使用原始的deepcopy + 参数裁剪逻辑
        model = copy.deepcopy(self.global_model)

        if scale == 1:
            return model

        model_kwargs = model.stored_inp_kwargs
        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        else:
            model_kwargs['params']['scale'] = scale

        local_model = type(self.global_model)(**model_kwargs)
        if 'bert' in str(type(local_model)):
            local_model.add_exits(model_kwargs['ee_layer_locations'])

        local_state_dict = local_model.state_dict()

        for n, p in self.global_model.state_dict().items():

            if 'num_batches_tracked' in n:
                local_state_dict[n] = p
                continue

            global_shape = p.shape
            local_shape = local_state_dict[n].shape

            if len(global_shape) != len(local_shape):
                print('Models are not alignable!')
                raise RuntimeError

            idx_array = self.fix_idx_array(self.idx_dicts[level][n], local_shape)
            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)

        return local_model


def execute_client_round(args):
    train_set, user_groups, criterion, args, batch_size, train_params, round_idx, global_model, \
    local_model, client_train_loader, level, scale, h_scale_ratio, client_idx, lora_configs = args

    if args.use_gpu:
        local_model = local_model.cuda()

    # MiniGPT模型：根据Level配置LoRA层的可训练性
    if 'MiniGPT' in str(type(local_model)) and lora_configs is not None:
        # 验证level有效性
        if level < 0 or level >= len(lora_configs):
            print(f"⚠️  客户端{client_idx}: Level {level}无效，使用Level 0")
            level = 0
        
        config = lora_configs[level]
        target_layers = set(config['layers'])
        
        # 第一步：冻结所有LoRA参数
        for name, param in local_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
        
        # 第二步：只解冻目标层的LoRA参数
        trainable_lora_count = 0
        for layer_idx in target_layers:
            # 更宽松的匹配：匹配包含.layers.{layer_idx}.的参数
            layer_pattern = f'.layers.{layer_idx}.'
            for name, param in local_model.named_parameters():
                if layer_pattern in name and 'lora' in name.lower():
                    param.requires_grad = True
                    trainable_lora_count += 1
        
        # 收集可训练参数（只包含LoRA参数）
        trainable_params = [v for k, v in local_model.named_parameters() if v.requires_grad]
        
        # 使用Level-Adaptive训练参数
        level_lr = config.get('lr', 5e-6)
        level_wd = config.get('weight_decay', 0.01)
        level_grad_clip = config.get('grad_clip', 1.0)
        
        print(f"✓ 客户端{client_idx} [Level {level}]: {len(target_layers)}层LoRA ({trainable_lora_count}参数), "
              f"lr={level_lr:.1e}, grad_clip={level_grad_clip}, wd={level_wd}")
        
        # 使用AdamW优化器（更适合Transformer），使用Level对应的学习率
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=level_lr,
                                      weight_decay=level_wd)
        
        # 将grad_clip传递给train_params，供execute_epoch使用
        train_params['grad_clip_norm'] = level_grad_clip
    else:
        # 传统模型：使用原始的SGD优化器
        base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
        exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]

        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': exit_params}],
                                    lr=train_params['lr'],
                                    momentum=train_params['momentum'],
                                    weight_decay=train_params['weight_decay'])

    loss = 0.0
    for epoch in range(train_params['num_epoch']):
        print(f'{client_idx}-{epoch}-{dt.datetime.now()}')
        iter_idx = round_idx
        loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx, epoch,
                             args, train_params, h_scale_ratio, level, global_model)

    print(f'Finished epochs for {client_idx}')
    
    # 只保存LoRA参数，避免保存27GB基础模型参数
    if args.arch == 'MiniGPTv2':
        # MiniGPT: 只保存LoRA参数（约128MB）
        local_weights = {
            k: v.cpu() 
            for k, v in local_model.state_dict(keep_vars=True).items() 
            if 'lora' in k.lower()
        }
        local_grad_flags = {
            k: v.grad is not None 
            for k, v in local_model.state_dict(keep_vars=True).items() 
            if 'lora' in k.lower()
        }
        print(f"✓ 只保存LoRA参数: {len(local_weights)}个参数 (过滤掉基础模型)")
    else:
        # 传统模型: 保存所有参数
        local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
        local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

    # 每个客户端训练完毕后立即释放内存
    # 清理优化器（AdamW会持有大量momentum和variance状态）
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    
    del local_model
    torch.cuda.empty_cache()

    return local_weights, local_grad_flags, loss