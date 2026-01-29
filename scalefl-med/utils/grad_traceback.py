from cmath import e
import copy
import math

import torch
import torch.nn as nn

def filter(x, dim=1, scale=1., target_shape=None, model_name=None, split=0):
    """
    对张量进行垂直分割过滤，保留指定维度的部分元素
    """
    # 创建副本避免修改原张量
    x_filtered = x.clone()

    if target_shape is None:
        if scale == 1:
            return x_filtered
        # 使用四舍五入确保维度计算准确
        out_dim = int(round(scale * x_filtered.shape[dim]))
    else:
        if dim == 1:
            out_dim = target_shape[0]
        else:
            if model_name == 'Linear':
                out_dim = target_shape[0]
            else:
                out_dim = target_shape[-1]

    # 创建完整的索引列表，处理所有维度
    ind_list = []
    for i in range(x_filtered.ndim):
        if i == min(dim, x_filtered.ndim - 1):  # 对目标维度应用过滤
            if split:
                mask = ([False for _ in range(out_dim // split)] +
                        [True for _ in range(x_filtered.shape[i] // split - out_dim // split)]) * split
                # 确保掩码长度正确
                mask = mask[:x_filtered.shape[i]]
                ind_list.append(mask)
            else:
                ind_list.append(slice(out_dim, x_filtered.shape[i]))
        else:
            ind_list.append(slice(None))  # 其他维度保持不变

    x_filtered[tuple(ind_list)] = 0

    return x_filtered


def get_num_gen(gen):
    """获取生成器中的元素数量"""
    return sum(1 for _ in gen)


def is_leaf(model):
    """判断模型是否为叶子节点（无子模块）"""
    return get_num_gen(model.children()) == 0


def get_downscale_index(model, args, scale=1.):
    """
    分析模型参数在垂直分割后的梯度贡献，返回有效参数索引字典
    """
    # 根据模型类型确定维度
    if 'bert' in str(model.__class__):
        dim = 2
    else:
        dim = 1

    def should_filter(x):
        """判断模块是否需要过滤"""
        if hasattr(x, 'old_forward'):
            return is_leaf(x) and 'modify_forward' not in str(x.forward)
        else:
            return is_leaf(x)

    def restore_forward(model):
        """恢复模型的原始前向传播函数"""
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                delattr(child, 'old_forward')  # 清理属性
            else:
                restore_forward(child)

    def modify_forward_recursive(model, local_model, split=1):
        """递归修改模型的前向传播函数，添加过滤逻辑"""
        # 判断是否包含线性层
        include_linear = 'bert' in str(model.__class__) and 'classifiers' not in str(model.__class__)

        for i, child in enumerate(model.children()):
            local_child = list(local_model.children())[i] if hasattr(local_model, 'children') else local_model
            
            if should_filter(child):
                # 创建闭包捕获当前状态
                def create_new_forward(module, local_shape, current_split=split, current_dim=dim):
                    def new_forward(*args, **kwargs):
                        # 调用原始前向传播
                        if hasattr(module, 'old_forward'):
                            raw_output = module.old_forward(*args, **kwargs)
                        else:
                            raw_output = module._old_forward(*args, **kwargs)
                        
                        # 处理输出
                        if isinstance(raw_output, tuple):
                            x_ = raw_output[0]
                            is_tuple = True
                        else:
                            x_ = raw_output
                            is_tuple = False
                        
                        # 应用过滤条件
                        module_name = module._get_name()
                        if include_linear:
                            if any(n in module_name for n in ['Conv', 'BatchNorm', 'LayerNorm', 'Linear', 'Embedding']):
                                x_ = filter(x_, dim=current_dim, scale=scale, target_shape=local_shape,
                                          model_name=module_name, split=current_split)
                        else:
                            if any(n in module_name for n in ['Conv', 'BatchNorm', 'LayerNorm', 'Embedding']):
                                x_ = filter(x_, dim=current_dim, scale=scale, target_shape=local_shape,
                                          split=current_split)
                        
                        # 返回处理后的输出
                        if is_tuple:
                            output_list = list(raw_output)
                            output_list[0] = x_
                            return tuple(output_list)
                        else:
                            return x_
                    
                    return new_forward
                
                # 保存原始前向传播
                if not hasattr(child, 'old_forward'):
                    child.old_forward = child.forward
                
                # 获取本地形状
                if hasattr(local_child, 'weight'):
                    local_shape = local_child.weight.shape
                else:
                    local_shape = None
                
                # 应用新的前向传播
                child.forward = create_new_forward(child, local_shape, split, dim)
            else:
                # 递归处理子模块
                modify_forward_recursive(child, local_child, split)

    # 复制模型初始化参数并禁用dropout
    model_kwargs = copy.deepcopy(model.stored_inp_kwargs)
    
    # 移除不接受的参数
    if 'scale' in model_kwargs:
        del model_kwargs['scale']
    if 'cfg' in model_kwargs and 'scale' in model_kwargs['cfg']:
        del model_kwargs['cfg']['scale']
    if 'config' in model_kwargs and 'scale' in model_kwargs['config']:
        del model_kwargs['config']['scale']
    
    # 禁用dropout
    if 'cfg' in model_kwargs:
        model_kwargs['cfg']['dropout_rate'] = 0
        model_kwargs['cfg']['drop_connect_rate'] = 0
    if 'config' in model_kwargs:
        for attr_name in dir(model_kwargs['config']):
            if 'dropout' in attr_name.lower():
                setattr(model_kwargs['config'], attr_name, 0)

    # 创建副本模型
    copy_model = type(model)(**model_kwargs)
    if 'bert' in str(type(copy_model)) and hasattr(model_kwargs, 'ee_layer_locations'):
        copy_model.add_exits(model_kwargs['ee_layer_locations'])

    # 移动到GPU（如果可用）
    if args.use_gpu and torch.cuda.is_available():
        copy_model = copy_model.cuda()

    # 创建本地模型用于形状参考（在CPU上）
    local_model_kwargs = copy.deepcopy(model_kwargs)
    local_model = type(model)(**local_model_kwargs)
    if 'bert' in str(type(local_model)) and hasattr(local_model_kwargs, 'ee_layer_locations'):
        local_model.add_exits(local_model_kwargs['ee_layer_locations'])

    # 修改前向传播
    if 'bert' in args.arch:
        num_heads = getattr(model_kwargs.get('config', None), 'num_attention_heads', 12)
        modify_forward_recursive(copy_model, local_model, split=num_heads)
    else:
        modify_forward_recursive(copy_model, local_model)

    # 清理本地模型
    del local_model
    torch.cuda.empty_cache()

    # 重新初始化权重以确保梯度多样性
    def reinitialize_weights(m):
        for name, param in m.named_parameters():
            if param.requires_grad:
                # 对注意力层使用更大的初始化
                if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)

    copy_model.apply(reinitialize_weights)

    # 准备输入数据
    if 'bert' in str(model.__class__):
        # BERT类模型输入
        inp = torch.ones((2, 128), dtype=torch.long)  # 使用batch_size=2确保稳定性
    else:
        # 视觉类模型输入
        inp = torch.randn((2, 3, args.image_size, args.image_size))
    
    if args.use_gpu and torch.cuda.is_available():
        inp = inp.cuda()

    # 获取模型的数据类型
    model_dtype = next(copy_model.parameters()).dtype
    print(f"模型数据类型: {model_dtype}")
    
    # 将输入转换为模型的数据类型（如果输入不是整数类型）
    if inp.dtype != model_dtype and inp.dtype != torch.long:
        inp = inp.to(model_dtype)
        print(f"已将输入数据类型转换为: {inp.dtype}")

    # 注册钩子捕获中间层输出
    intermediate_outputs = []
    
    def hook_fn(module, input, output):
        # 只捕获第一个有效输出
        if not intermediate_outputs:
            if isinstance(output, tuple):
                intermediate_outputs.append(output[0].detach().clone())
            else:
                intermediate_outputs.append(output.detach().clone())

    # 注册钩子到关键层
    hooks = []
    for name, module in copy_model.named_modules():
        # 优先选择注意力层的q_proj，如果没有则选择第一个线性层
        if 'q_proj' in name or (len(intermediate_outputs) == 0 and isinstance(module, nn.Linear)):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            if len(hooks) >= 3:  # 注册多个钩子确保捕获到输出
                break

    try:
        # 执行前向传播
        copy_model.eval()  # 设置为评估模式
        
        # 确保启用梯度计算
        with torch.set_grad_enabled(True):
            # 尝试不同的输入方式
            if hasattr(copy_model, 'forward_specific'):
                print("使用 forward_specific 方法")
                preds = copy_model.forward_specific(inp)
            else:
                # 通用前向传播
                try:
                    print("尝试标准 forward 方法")
                    preds = copy_model(inp)
                except Exception as e:
                    print(f"标准前向传播失败: {e}")
                    # 如果标准前向传播失败，尝试其他方式
                    try:
                        if hasattr(copy_model, 'features'):
                            print("使用 features 方法")
                            preds = copy_model.features(inp)
                        elif hasattr(copy_model, 'encoder'):
                            print("使用 encoder 方法")
                            preds = copy_model.encoder(inp)
                        else:
                            # 最后尝试直接调用模块
                            print("使用直接 forward 调用")
                            preds = copy_model.forward(inp)
                    except Exception as e2:
                        print(f"所有前向传播方法均失败: {e2}")
                        # 创建一个虚拟输出
                        for module in copy_model.modules():
                            if isinstance(module, nn.Linear):
                                preds = torch.randn(2, module.out_features, dtype=model_dtype)
                                if args.use_gpu and torch.cuda.is_available():
                                    preds = preds.cuda()
                                print("创建虚拟输出")
                                break

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 如果有钩子捕获的输出，使用它
        if intermediate_outputs:
            print("使用钩子捕获的中间输出")
            preds = intermediate_outputs[0]
        
        # 确保preds的数据类型与模型一致
        if preds.dtype != model_dtype:
            print(f"将输出数据类型从 {preds.dtype} 转换为 {model_dtype}")
            preds = preds.to(model_dtype)
        
        # 确保preds需要梯度
        if not preds.requires_grad:
            print("设置输出需要梯度")
            preds = preds.clone().requires_grad_(True)

        print(f"最终输出形状: {preds.shape}, 数据类型: {preds.dtype}")

        # 创建目标张量（与preds相同数据类型）
        target = torch.randn_like(preds, dtype=preds.dtype)
        if args.use_gpu and torch.cuda.is_available():
            target = target.cuda()

        # 计算损失
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds, target)
        print(f"损失计算成功: {loss.item()}")

        # 反向传播
        copy_model.zero_grad()
        loss.backward()
        print("反向传播成功")

        # 收集梯度信息
        idx_dict = {}
        grad_found = False
        for name, param in copy_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_abs = param.grad.abs()
                # 检查梯度是否全为零
                if grad_abs.sum().item() > 0:
                    grad_found = True
                    # 选择梯度最大的前scale比例的参数
                    if grad_abs.numel() > 0:
                        try:
                            threshold = torch.quantile(grad_abs, 1 - scale)
                            idx_dict[name] = grad_abs > threshold
                        except:
                            # 如果分位数计算失败，使用中位数
                            threshold = torch.median(grad_abs)
                            idx_dict[name] = grad_abs > threshold
                    else:
                        idx_dict[name] = torch.zeros_like(param, dtype=bool)
                else:
                    idx_dict[name] = torch.zeros_like(param, dtype=bool)
            else:
                idx_dict[name] = torch.ones_like(param, dtype=bool)
        
        if not grad_found:
            print("警告: 未找到任何非零梯度")

    except Exception as e:
        print(f"梯度计算过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回保守的索引（全部为True）
        idx_dict = {}
        for name, param in copy_model.named_parameters():
            idx_dict[name] = torch.ones_like(param, dtype=bool)
    
    finally:
        # 恢复原始前向传播并清理
        restore_forward(copy_model)
        del copy_model
        torch.cuda.empty_cache()

    return idx_dict

