# #!/usr/bin/env python3
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import pickle as pkl
#
# import numpy as np
# import torch.backends.cudnn as cudnn
# import torch.optim
# import models
# from args import arg_parser, modify_args
# from config import Config
# from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups
# from fed import Federator
# from models.model_utils import KDLoss
# from predict import validate, local_validate
# from utils.utils import load_checkpoint, measure_flops, load_state_dict, save_user_groups, load_user_groups
#
#
# # 新增：导入 MiniGPT - Med 相关模块
# from models.minigpt4.common.config import Config as MiniGPTConfig
# from models.minigpt4.models.minigpt_v2 import MiniGPTv2
# from models.minigpt4.datasets.datasets.knee_x_ray_dataset import KneeXrayDataset
#
#
# np.set_printoptions(precision=2)
#
# args = arg_parser.parse_args()
# args = modify_args(args)
# torch.manual_seed(args.seed)
#
#
# def main():
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     global args
#
#
#     if not os.path.exists(args.save_path):
#         os.makedirs(args.save_path)
#
#     config = Config()
#
#     if args.ee_locs:
#         config.model_params[args.data][args.arch]['ee_layer_locations'] = args.ee_locs
#
#
#     # 修改：支持 minigpt_v2 模型
#     if args.arch == "MiniGPTv2":
#         minigpt_cfg = MiniGPTConfig(args)
#         # minigpt_cfg.update(config.model_params[args.data][args.arch])
#         # model = MiniGPTv2.from_config()
#         model = MiniGPTv2.from_config(minigpt_cfg).to(device)
#     else:
#         model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]}).to(device)
#
#
#     # args.num_exits = config.model_params[args.data][args.arch]['num_blocks']   # 在minigpt_v2中未设置num_blocks参数，报错，删了试试
#     args.num_exits = config.model_params[args.data][args.arch]
#
#
#
#     # if args.use_gpu:
#     #     model = model.cuda()
#     #     criterion = KDLoss(args).cuda()
#     # else:
#     #     criterion = KDLoss(args)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     criterion = KDLoss(args).to(device)
#
#     if args.resume:
#         checkpoint = load_checkpoint(args, load_best=False)
#         if checkpoint is not None:
#             args.start_round = checkpoint['round'] + 1
#             model.load_state_dict(checkpoint['state_dict'])
#
#     cudnn.benchmark = True
#
#     batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]['batch_size']
#     train_set, val_set, test_set = get_datasets(args)
#     _, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
#     if val_set is None:
#         val_set = val_loader.dataset
#
#
#     train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)
#
#     prev_user_groups = load_user_groups(args)
#     if prev_user_groups is None:
#         if args.resume:
#             print('Could not find user groups')
#             raise RuntimeError
#         user_groups = (train_user_groups, val_user_groups, test_user_groups)
#         save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
#     else:
#         user_groups = prev_user_groups
#
#     if args.evalmode is not None:
#         load_state_dict(args, model)
#         if 'global' in args.evalmode:
#             validate(model, test_loader, criterion, args)
#             return
#         elif 'local' in args.evalmode:
#             train_args = eval('argparse.' + open(os.path.join(args.save_path, 'args.txt')).readlines()[0])
#             if os.path.exists(os.path.join(args.save_path, 'client_groups.pkl')):
#                 client_groups = pkl.load(open(os.path.join(args.save_path, 'client_groups.pkl'), 'rb'))
#             else:
#                 client_groups = []
#             federator = Federator(model, train_args, client_groups)
#             local_validate(federator, test_set, user_groups[1], criterion, args, batch_size)
#             return
#         else:
#             raise NotImplementedError
#
#     with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
#         print(args, file=f)
#
#     federator = Federator(model, args)
#     best_acc1, best_round = federator.fed_train(train_set, val_set, user_groups, criterion, args, batch_size,
#                                                  config.training_params[args.data][args.arch])
#
#     print('Best val_acc1: {:.4f} at round {}'.format(best_acc1, best_round))
#     validate(federator.global_model, test_loader, criterion, args, save=True)
#
#     return
#
#
# if __name__ == '__main__':
#     main()






# !/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups
from fed import Federator
from models.model_utils import KDLoss
from predict import validate, local_validate
from utils.utils import load_checkpoint, measure_flops, load_state_dict, save_user_groups, load_user_groups

# 新增：导入 MiniGPT - Med 相关模块
from models.minigpt4.common.config import Config as MiniGPTConfig
from models.minigpt4.models.minigpt_v2 import MiniGPTv2
from models.minigpt4.datasets.datasets.knee_x_ray_dataset import KneeXrayDataset

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global args


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    config = Config()

    if args.ee_locs:
        config.model_params[args.data][args.arch]['ee_layer_locations'] = args.ee_locs

    # 修改：支持 minigpt_v2 模型
    if args.arch == "MiniGPTv2":
        minigpt_cfg = MiniGPTConfig(args)
        # minigpt_cfg.update(config.model_params[args.data][args.arch])
        # model = MiniGPTv2.from_config()

        print("MiniGPT Config:", minigpt_cfg)
        print("args.image_size:", args.image_size)


        model = MiniGPTv2.from_config(minigpt_cfg).to(device)
    else:
        model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]}).to(device)

    # args.num_exits = config.model_params[args.data][args.arch]['num_blocks']   # 在minigpt_v2中未设置num_blocks参数，报错，删了试试
    # 【修复】根据模型架构设置num_exits
    if args.arch == 'MiniGPTv2':
        args.num_exits = 1  # MiniGPT没有early exit概念，设为1
    else:
        model_config = config.model_params[args.data][args.arch]
        args.num_exits = model_config.get('num_blocks', 1) if isinstance(model_config, dict) else model_config

    # if args.use_gpu:
    #     model = model.cuda()
    #     criterion = KDLoss(args).cuda()
    # else:
    #     criterion = KDLoss(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = KDLoss(args).to(device)

    # if args.resume:
    #     checkpoint = load_checkpoint(args, load_best=False)
    #     if checkpoint is not None:
    #         args.start_round = checkpoint['round'] + 1
    #         model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]['batch_size']

    print(f"当前 batch_size: {batch_size}")

    train_set, val_set, test_set = get_datasets(args)

    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")


    print("开始初始化数据加载器...")
    # train_loader, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
    train_loader, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
    print("数据加载器初始化完成！")
    if val_set is None:
        val_set = val_loader.dataset

    print("尝试加载一个批次的数据...")
    try:
        batch = next(iter(train_loader))
        print(f"成功加载批次！样本数量: {len(batch)}")
        print(f"批次组件的键名: {batch.keys()}")  # 打印所有组件的键
        # 1. 提取图像组件（根据实际键名调整，常见键名：'image'、'vis'、0（元组第一个元素））
        if isinstance(batch, dict):
            images = batch['image']  # 字典格式，通过键获取图像
        else:
            images = batch[0]  # 元组/列表格式，通常第一个元素是图像

        # 2. 打印图像批次的基本信息
        print(f"图片批次形状: {images.shape}")  # 格式：[batch_size, 通道数, 高度, 宽度]
        print(f"图片数据类型: {images.dtype}")  # 通常为 float32
        print(f"图片像素值范围: [{images.min():.4f}, {images.max():.4f}]")  # 确认是否归一化
        print(f"实际 batch_size（图片数量）: {images.shape[0]}")  # 第一个维度即为样本数
    except Exception as e:
        print(f"加载批次失败: {e}")
        raise

    print(f"训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}, 测试集大小: {len(test_set)}")



    print("开始划分用户组...")
    train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)
    print("用户组划分完成！")

    prev_user_groups = load_user_groups(args)
    if prev_user_groups is None:
        if args.resume:
            print('Could not find user groups')
            raise RuntimeError
        user_groups = (train_user_groups, val_user_groups, test_user_groups)
        save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
    else:
        user_groups = prev_user_groups

    if args.evalmode is not None:
        load_state_dict(args, model)
        if 'global' in args.evalmode:
            validate(model, test_loader, criterion, args)
            return
        elif 'local' in args.evalmode:
            train_args = eval('argparse.' + open(os.path.join(args.save_path, 'args.txt')).readlines()[0])
            if os.path.exists(os.path.join(args.save_path, 'client_groups.pkl')):
                client_groups = pkl.load(open(os.path.join(args.save_path, 'client_groups.pkl'), 'rb'))
            else:
                client_groups = []
            federator = Federator(model, train_args, client_groups)
            local_validate(federator, test_set, user_groups[1], criterion, args, batch_size)
            return
        else:
            raise NotImplementedError

    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        print(args, file=f)

    federator = Federator(model, args)
    best_acc1, best_round = federator.fed_train(train_set, val_set, user_groups, criterion, args, batch_size,
                                                config.training_params[args.data][args.arch])

    print('Best val_acc1: {:.4f} at round {}'.format(best_acc1, best_round))
    
    # 【修复】根据模型架构选择测试方法
    if args.arch == 'MiniGPTv2':
        print('\n' + '='*60)
        print('✓ MiniGPT模型训练完成')
        print('='*60)
        print('训练过程中每轮都已进行验证')
        print('Checkpoint已保存至:', args.save_path)
        print('='*60 + '\n')
        
        # 在测试集上运行验证
        print('在测试集上运行验证...')
        from predict import minigpt_validate
        test_loss = minigpt_validate(federator.global_model, test_loader, None, args)
        print(f'\n✓ 测试集验证Loss: {test_loss:.4f}\n')
    else:
        # 传统模型使用标准validate函数
        validate(federator.global_model, test_loader, criterion, args, save=True)

    return


if __name__ == '__main__':
    main()