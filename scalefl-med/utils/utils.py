import glob
import math
import numpy as np
import os
import shutil
import pickle

import models
import torch

from utils.op_counter import measure_model


def save_checkpoint(state, args, is_best, filename, result, keep_last_n=3):
    """
    保存checkpoint，同时保留最近N个checkpoint和最佳模型
    
    Args:
        state: 模型状态字典
        args: 参数
        is_best: 是否是最佳模型
        filename: checkpoint文件名
        result: 结果记录
        keep_last_n: 保留最近N个checkpoint（默认3个）
    """
    print(args)
    result_filename = os.path.join(args.save_path, 'scores.tsv')
    model_dir = os.path.join(args.save_path, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    # 【优化】保留最近N个checkpoint，而不是只保留1个
    prev_checkpoint_list = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_*.pth.tar')))
    if len(prev_checkpoint_list) >= keep_last_n:
        # 删除最旧的checkpoint
        for old_ckpt in prev_checkpoint_list[:-keep_last_n+1]:
            print(f"  🗑️  删除旧checkpoint: {os.path.basename(old_ckpt)}")
            os.remove(old_ckpt)

    torch.save(state, model_filename)

    with open(result_filename, 'a') as f:
        print(result[-1], file=f)

    if is_best:
        shutil.copyfile(model_filename, best_filename)
        print(f"  ⭐ 保存最佳模型: model_best.pth.tar")

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args, load_best=True):
    model_dir = os.path.join(args.save_path, 'save_models')
    if load_best:
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
    else:
        model_filename = glob.glob(os.path.join(model_dir, 'checkpoint*'))[0]

    if os.path.exists(model_filename):
        print("=> loading checkpoint '{}'".format(model_filename))
        state = torch.load(model_filename)
        print("=> loaded checkpoint '{}'".format(model_filename))
    else:
        return None

    return state


def save_user_groups(args, user_groups):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')
    if not os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'wb') as fout:
            pickle.dump(user_groups, fout)


def load_user_groups(args):
    user_groups_filename = os.path.join(args.save_path, 'user_groups.pkl')

    if os.path.exists(user_groups_filename):
        with open(user_groups_filename, 'rb') as fin:
            user_groups = pickle.load(fin)
    else:
        user_groups = None

    return user_groups


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, params):
    if params['lr_type'] == 'multistep':
        lr, decay_rate = params['lr'], params['decay_rate']
        if epoch >= params['decay_epochs'][1]:
            lr *= decay_rate ** 2
        elif epoch >= params['decay_epochs'][0]:
            lr *= decay_rate
    elif params['lr_type'] == 'exp':
        lr = params['lr'] * (np.power(params['decay_rate'], epoch))
    else:
        lr = params['lr']
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr


def measure_flops(args, params):
    model = getattr(models, args.arch)(args, params)
    model.eval()
    n_flops, n_params = measure_model(model, args.image_size[0], args.image_size[1])
    torch.save(n_flops, os.path.join(args.save_path, 'flops.pth'))
    del (model)


def load_state_dict(args, model):
    state_dict = torch.load(args.evaluate_from)['state_dict']
    model.load_state_dict(state_dict)
