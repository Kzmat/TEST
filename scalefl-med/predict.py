#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from data_tools.dataloader import get_client_dataloader
from utils.utils import accuracy, AverageMeter
from utils.utils import load_state_dict


# 为MiniGPT模型提供真实的验证功能
def minigpt_validate(model, val_loader, criterion, args):
    """
    MiniGPT模型的验证：计算验证集上的实际loss
    
    注意：这里计算的是训练loss，完整的VLM评估还需要专门的指标（BLEU, ROUGE, CIDEr等）
    
    返回: avg_loss (float) - 验证集上的平均loss
    """
    print(f"⏳ MiniGPT模型验证...")
    
    model.eval()
    losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # 复用训练时的数据预处理逻辑
            if isinstance(batch, dict):
                images = batch["image"]
                if args.use_gpu:
                    images = images.cuda()
                
                # 获取answer字段（caption）用于计算loss
                answers = batch.get("answer", None)
                
                # 构造验证样本（与训练时相同的格式）
                if answers is not None and len(answers) > 0:
                    samples = {
                        "image": images,
                        "text_input": batch.get("instruction_input", ["Describe this knee X-ray image."] * images.shape[0]),
                        "answer": answers
                    }
                else:
                    # 如果没有标签，跳过这个batch
                    continue
                
                # 前向传播计算loss
                try:
                    outputs = model(samples)
                    
                    # 提取loss（可能是dict或者有.loss属性）
                    if isinstance(outputs, dict):
                        loss = outputs.get("loss", None)
                    elif hasattr(outputs, "loss"):
                        loss = outputs.loss
                    else:
                        # 如果没有loss，跳过
                        continue
                    
                    if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                        losses.append(loss.item())
                    
                    # 打印进度（每10个batch）
                    if (i + 1) % 10 == 0:
                        current_avg = np.mean(losses) if losses else 0.0
                        print(f"  验证进度: [{i+1}/{len(val_loader)}] Loss: {current_avg:.4f}")
                        
                except Exception as e:
                    print(f"  ⚠️  验证batch {i} 失败: {e}")
                    continue
    
    # 计算平均loss
    avg_loss = np.mean(losses) if losses else 0.0
    print(f"✓ 验证完成: 平均Loss = {avg_loss:.4f} (基于 {len(losses)} 个有效batch)")
   
    
    return avg_loss


# 对联邦学习本地模型进行验证
def local_validate(federator, dataset, user_groups, criterion, args, batch_size, model=None, save=False):
    with open('models/eval_configs/minigptv2_eval.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # 初始化 MiniGPTv2 模型
    if model is None:
        model = MiniGPTv2.from_config(cfg)
    
    # if model is None:
    #     load_state_dict(args, federator.global_model)
    #     model = federator.global_model

    if args.use_gpu:
        federator.global_model = federator.global_model.cuda()
        model = model.cuda()

    result_lists = [[] for _ in range(5)]
    local_result_lists = [[[] for _ in range(3)] for _ in range(federator.num_levels + 1)]
    levels = []

    for client_idx in range(args.num_clients):
        client_loader = get_client_dataloader(dataset, user_groups[client_idx], args, batch_size)

        level = federator.get_level(client_idx)
        scale = federator.vertical_scale_ratios[level]
        exit_idx = federator.horizontal_scale_ratios[level]
        local_model = federator.get_local_split(level, scale)
        if args.use_gpu:
            local_model = local_model.cuda()

        if level == federator.num_levels - 1:  # local and global models/results are same for highest level
            results = validate(model, client_loader, criterion, args, client_idx=client_idx, save=False)
            local_results = results
        else:
            local_results = validate([model, local_model], client_loader, criterion, args, client_idx=client_idx,
                                     exit_idx=[0, exit_idx], save=False)
            results = local_results[0]
            local_results = local_results[1]

        for i in range(len(result_lists)):
            result_lists[i].append(results[i])

        for i in range(len(local_result_lists[0])):
            local_result_lists[-1][i].append(local_results[i])

        for i in range(len(local_result_lists[0])):
            local_result_lists[level][i].append(local_results[i])

        levels.append(level)

    results = [sum(result_list) / federator.num_clients for result_list in result_lists]
    local_results = []
    for local_result_list in local_result_lists:
        local_result_list = [sum(l) / len(local_result_list[0]) if len(local_result_list[0]) != 0 else 0 for l in
                             local_result_list]
        local_results.append(local_result_list)

    test_result_filename = os.path.join(args.save_path, 'test_scores.tsv')
    with open(test_result_filename, 'w') as f:
        for j in range(federator.num_levels):
            text = f'Level {j + 1}/{federator.num_levels} * prec@1 {local_results[j][1]:.3f} prec@5 {local_results[j][2]:.3f}'
            print(text)
            if save:
                print(text, file=f)
        top1 = results[1]
        top5 = results[2]
        text = f'All clients * prec@1 {top1:.3f} prec@5 {top5:.3f}'
        print(text)
        if save:
            print(text, file=f)

    return results, local_results


def validate(models, val_loader, criterion, args, client_idx=0, exit_idx=0, save=False):
    # 【修复】MiniGPT模型不支持此验证函数（设计用于early exit的CNN模型）
    if hasattr(args, 'arch') and args.arch == 'MiniGPTv2':
        print("MiniGPT模型不支持此验证函数，请使用minigpt_validate()")
        print("   （此函数设计用于有early exit的CNN模型）")
        return [0.0, 0.0, 0.0], []
    
    if not isinstance(models, list):
        models = [models]

    if not isinstance(exit_idx, list):
        exit_idxs = [exit_idx]
    else:
        exit_idxs = exit_idx

    batch_time = [AverageMeter() for _ in range(len(models))]
    losses = [AverageMeter() for _ in range(len(models))]
    data_time = AverageMeter()

    num_exits = [exit_idx if exit_idx != 0 else args.num_exits for exit_idx in exit_idxs]

    top1 = [[AverageMeter()] * num_exits[i] for i in range(len(models))]
    top5 = [[AverageMeter()] * num_exits[i] for i in range(len(models))]

    print(f'Validation results for Client {client_idx + 1} with Exit {exit_idxs}')

    end = time.time()
    with torch.no_grad():
        for i, (inp, target) in enumerate(val_loader):
            if args.use_gpu:
                target = target.cuda()
                inp = inp.cuda()

            data_time.update(time.time() - end)

            for model_idx, model in enumerate(models):
                output = model(inp, manual_early_exit_index=exit_idxs[model_idx])
                if not isinstance(output, list):
                    output = [output]

                loss = 0.0
                for j in range(len(output)):
                    if j == len(output) - 1:
                        loss += criterion.ce_loss(output[j], target)
                    else:
                        loss += criterion.loss_fn_kd(output[j], target, output[-1])

                for j in range(len(output)):
                    if 'bert' in args.arch:
                        prec1, prec5 = accuracy(output[j], target, topk=(1, 1))
                    else:
                        prec1, prec5 = accuracy(output[j], target, topk=(1, 5))
                    top1[model_idx][j].update(prec1.item(), inp.size(0))
                    top5[model_idx][j].update(prec5.item(), inp.size(0))

                loss /= len(output) * (len(output) + 1) / 2
                losses[model_idx].update(loss.item(), inp.size(0))

                # measure elapsed time
                batch_time[model_idx].update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print(f'Iter: [{i + 1}/{len(val_loader)}]\t\t' +
                          f'Time: {batch_time[model_idx].avg:.3f}\t' +
                          f'Data: {data_time.avg:.3f}\t' +
                          f'Loss: {losses[model_idx].val:.4f}\t' +
                          f'Acc@1: {top1[model_idx][-1].val:.4f}\t' +
                          f'Acc@5: {top5[model_idx][-1].val:.4f}')

    if save:
        test_result_filename = os.path.join(args.save_path, 'test_scores.tsv')
        with open(test_result_filename, 'w') as f:
            for model_idx in range(len(models)):
                text = f'Model index {model_idx} * prec@1 {top1[model_idx][-1].avg:.3f} prec@5 {top5[model_idx][-1].avg:.3f}'
                print(text)
                print(text, file=f)

    if len(models) == 1:
        return losses[0].avg, top1[0][-1].avg, top5[0][-1].avg, np.array([t.avg for t in top1[0]]), np.array(
            [t.avg for t in top5[0]])
    else:
        return [[losses[i].avg, top1[i][-1].avg, top5[i][-1].avg, np.array([t.avg for t in top1[i]]),
                 np.array([t.avg for t in top5[i]])] for i in range(len(models))]
