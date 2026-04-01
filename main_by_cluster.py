# -*- coding: utf-8 -*-
# type: ignore
#!/usr/bin/env python3
"""
BGC-DETR主训练脚本
实现基于DETR架构的生物基因簇检测模型的训练和验证
支持5折交叉验证和分布式训练
"""

import argparse
import json
import time
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import random
from util.misc import (collate_fn, NestedTensor, get_args_parser, n_parameters,
                       init_distributed_mode, get_sha, save_on_master, is_main_process,
                       get_rank, get_world_size)
from data import BalancedClusterDataset
from models import build_model
from engine import train_one_epoch, evaluate
from torch.utils.data import DistributedSampler
import os
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    设置分布式训练环境，禁用非主进程的打印输出

    Args:
        is_master: 是否为主进程
    """
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def main(args):
    """
    主训练函数
    实现5折交叉验证的完整训练流程

    Args:
        args: 命令行参数
    """
    # ===== 设置多进程启动方法 =====
    # 必须在任何CUDA操作之前设置
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        if is_main_process():
            print("已设置多进程启动方法为 'spawn'")
    except RuntimeError:
        if is_main_process():
            print("多进程启动方法已设置，跳过")

    # ===== 分布式训练设置 =====
    # 检查环境变量，判断是否使用分布式训练
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        args.distributed = False

    # 设置当前设备
    # 单卡：使用 GPU 0；多卡分布式：按 local_rank 选择对应 GPU
    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # 初始化分布式环境
    if args.distributed:
        # 使用GLOO后端，它更适合在单个GPU上运行多个进程
        dist.init_process_group(
            backend='gloo',  # 使用GLOO而不是NCCL
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        if rank == 0:
            print(f"| distributed init (rank {rank}): env://")
            print(f"| using GPUs: {world_size} (this process -> local_rank={local_rank})")
            print(f"| world size: {world_size}")
            print(f"| device: {device}")
            print(f"| backend: gloo")

    # 设置随机种子，确保实验可重现
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # 确保所有进程同步
    if args.distributed:
        dist.barrier()

    # 只有主进程打印GPU使用情况
    if rank == 0:
        print("\nGPU使用情况：")
        print(f"使用的GPU: 0")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"总进程数: {world_size}")

    if is_main_process():
        print("git:\n  {}\n".format(get_sha()))

    # 检查冻结权重参数
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    if is_main_process():
        print(args)

    # ===== 参数验证 =====
    # 检查是否指定了训练集或验证集的JSON文件
    if not (args.train_mapping_json or args.val_mapping_json or args.mapping_json):
        raise ValueError(
            "必须指定训练集 (--train_mapping_json) 或验证集 (--val_mapping_json) 或完整的mapping文件 (--mapping_json)")

    # 如果只指定了mapping_json，则使用旧的交叉验证模式
    use_cross_validation = args.mapping_json is not None and (
        args.train_mapping_json is None and args.val_mapping_json is None)

    if use_cross_validation:
        if is_main_process():
            print("使用交叉验证模式 (legacy)")
        # ===== 数据集加载和划分 =====
        # 首先加载完整数据集
        full_dataset = BalancedClusterDataset(
            args.mapping_json,
            args.emb_dir,
            max_tokens=args.max_tokens,
            balance_strategy='none',  # 初始加载时不进行平衡
            binary_mode=args.binary_mode
        )
    else:
        if is_main_process():
            print("使用分离的数据集模式")
        full_dataset = None

    # 检查数据集是否为空（仅在交叉验证模式下）
    if use_cross_validation and full_dataset is not None:
        if len(full_dataset) == 0:
            raise ValueError("加载的数据集为空，请检查数据路径和格式是否正确")

    if is_main_process() and use_cross_validation:
        print(f"原始数据集大小: {len(full_dataset)} 个样本")

    if use_cross_validation:
        # 使用固定的随机种子进行划分
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        # 首先统计每个BGC的预测框数量和类别
        bgc_box_counts = {}
        bgc_classes = {}
        for bgc_id in full_dataset.bgc_ids:
            bgc_box_counts[bgc_id] = len(full_dataset.mapping[bgc_id])
            # 收集这个BGC包含的所有类别
            classes = set()
            for region in full_dataset.mapping[bgc_id]:
                classes.add(region['type'])
            bgc_classes[bgc_id] = classes

        # 按预测框数量对BGC进行排序
        sorted_bgcs = sorted(bgc_box_counts.items(),
                             key=lambda x: x[1], reverse=True)

        # 准备5-fold交叉验证
        n_folds = 5
        folds = [[] for _ in range(n_folds)]

        # 分层抽样：确保每个fold都包含所有类别
        # 首先按类别分组BGC
        class_to_bgcs = {}
        for bgc_id, classes in bgc_classes.items():
            for cls in classes:
                if cls not in class_to_bgcs:
                    class_to_bgcs[cls] = []
                class_to_bgcs[cls].append(bgc_id)

        # 对每个类别，将其BGC平均分配到各个fold
        for cls, bgc_list in class_to_bgcs.items():
            # 随机打乱BGC列表
            random.shuffle(bgc_list)
            # 平均分配到各个fold
            for i, bgc_id in enumerate(bgc_list):
                fold_idx = i % n_folds
                folds[fold_idx].append(bgc_id)

        # 去重（因为一个BGC可能包含多个类别）
        for i in range(n_folds):
            folds[i] = list(set(folds[i]))

        # 打印每个fold的统计信息
        if is_main_process():
            print(f"\n数据集划分统计：")
            print(f"- 原始数据集大小: {len(full_dataset)} 个BGC样本")
            total_boxes = sum(bgc_box_counts.values())
            print(f"- 总预测框数量: {total_boxes} 个")

        for i, fold_bgcs in enumerate(folds):
            fold_boxes = sum(bgc_box_counts[bgc] for bgc in fold_bgcs)
            if is_main_process():
                print(f"\nFold {i+1} 统计：")
                print(f"- BGC样本数: {len(fold_bgcs)}")
                print(
                    f"- 预测框数量: {fold_boxes} ({fold_boxes/total_boxes*100:.1f}%)")

    # 存储所有fold的结果
    all_fold_results = []

    if use_cross_validation:
        # ===== 5折交叉验证训练循环 =====
        fold_range = range(n_folds)
        n_folds_to_run = n_folds
        display_n_folds = n_folds
    else:
        # ===== 直接训练模式 =====
        fold_range = range(1)  # 只运行一次
        n_folds_to_run = 1
        display_n_folds = 1
        folds = [None]  # 占位符
        bgc_box_counts = {}  # 初始化为空字典

    for fold_idx in fold_range:
        if is_main_process():
            print(f"\n{'='*50}")
            print(f"开始训练 Fold {fold_idx + 1}/{display_n_folds}")
            print(f"{'='*50}")

        if use_cross_validation:
            # 准备当前fold的训练集和验证集
            val_bgcs = folds[fold_idx]  # 当前fold作为验证集
            train_bgcs = []
            for i in range(n_folds):
                if i != fold_idx:
                    train_bgcs.extend(folds[i])  # 其他fold作为训练集

            # 创建训练集（使用类别平衡）
            dataset_train = BalancedClusterDataset(
                args.mapping_json,
                args.emb_dir,
                max_tokens=args.max_tokens,
                balance_strategy=args.balance_strategy,
                seq_id_list=train_bgcs,
                binary_mode=args.binary_mode
            )

            # 创建验证集（不使用类别平衡）
            dataset_val = BalancedClusterDataset(
                args.mapping_json,
                args.emb_dir,
                max_tokens=args.max_tokens,
                balance_strategy='none',
                seq_id_list=val_bgcs,
                binary_mode=args.binary_mode
            )

            if is_main_process():
                print(f"\nFold {fold_idx + 1} 数据集划分：")
                print(f"- 训练集BGC数量: {len(train_bgcs)}")
                print(f"- 验证集BGC数量: {len(val_bgcs)}")
        else:
            # 使用分离的数据集模式
            if is_main_process():
                print(f"\n直接训练模式：")

            # 创建训练集
            if args.train_mapping_json:
                # 使用指定的训练集embeddings路径
                train_emb_dir = args.train_emb_dir if args.train_emb_dir else args.emb_dir

                dataset_train = BalancedClusterDataset(
                    args.train_mapping_json,
                    train_emb_dir,
                    max_tokens=args.max_tokens,
                    balance_strategy=args.balance_strategy,
                    binary_mode=args.binary_mode
                )
                if is_main_process():
                    print(f"- 训练集: {args.train_mapping_json}")
                    print(f"- 训练集embeddings: {train_emb_dir}")
                    print(f"- 训练集BGC数量: {len(dataset_train)}")
            else:
                dataset_train = None
                if is_main_process():
                    print("- 未指定训练集，将只进行验证")

            # 创建验证集
            if args.val_mapping_json:
                # 使用指定的验证集embeddings路径
                val_emb_dir = args.val_emb_dir if args.val_emb_dir else args.emb_dir

                dataset_val = BalancedClusterDataset(
                    args.val_mapping_json,
                    val_emb_dir,
                    max_tokens=args.max_tokens,
                    balance_strategy='none',
                    binary_mode=args.binary_mode
                )
                if is_main_process():
                    print(f"- 验证集: {args.val_mapping_json}")
                    print(f"- 验证集embeddings: {val_emb_dir}")
                    print(f"- 验证集BGC数量: {len(dataset_val)}")
            else:
                dataset_val = None
                if is_main_process():
                    print("- 未指定验证集，将只进行训练")

        if is_main_process():
            print(f"\nFold {fold_idx + 1} 数据集大小：")
            if use_cross_validation:
                print(
                    f"- 训练集：{len(dataset_train)} 个BGC样本，{sum(bgc_box_counts[bgc] for bgc in train_bgcs)} 个预测框")
                print(
                    f"- 验证集：{len(dataset_val)} 个BGC样本，{sum(bgc_box_counts[bgc] for bgc in val_bgcs)} 个预测框")
            else:
                if dataset_train:
                    print(f"- 训练集：{len(dataset_train)} 个BGC样本")
                if dataset_val:
                    print(f"- 验证集：{len(dataset_val)} 个BGC样本")
                else:
                    print("- 无可用数据集信息")

        # 从数据集中获取实际的类别数量
        # 优先使用训练集，否则使用验证集
        reference_dataset = dataset_train if dataset_train else dataset_val
        num_classes = len(reference_dataset.label2id)  # 数据集中已经包含了no-object类
        if is_main_process():
            print(f"从数据集获取的类别数量: {num_classes}")
            print(f"类别映射: {reference_dataset.label2id}")
        args.num_classes = num_classes

        # ===== 数据加载器设置 =====
        if dataset_train is not None:
            if args.distributed:
                sampler_train = DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                           collate_fn=collate_fn, num_workers=args.num_workers)
        else:
            data_loader_train = None

        if dataset_val is not None:
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                         drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)
        else:
            data_loader_val = None

        # ===== 模型构建 =====
        # 为每个fold创建新的模型
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        criterion.to(device)

        # ===== 检查模型初始化状态 =====
        if is_main_process() and fold_idx == 0:  # 只在第一个fold检查一次
            print(f"\n🔍 模型初始化状态检查:")
            print(f"  • 分类头权重shape: {model.class_embed.weight.shape}")
            print(
                f"  • 分类头偏置: {[f'{b:.3f}' for b in model.class_embed.bias.tolist()]}")
            print(f"  • num_queries: {model.num_queries}")
            print(f"  • num_classes: {model.num_classes}")
            print(f"  • eos_coef: {args.eos_coef}")
            print(f"  • focal_alpha: {args.focal_alpha}")
            print(f"  • 学习率: {args.lr}")

            # 测试一个小批次的前向传播
            # 注意：backbone 现在支持任意 embed_dim（例如 1280），这里需要与 args.embed_dim 对齐
            print(f"  • 正在测试前向传播...")
            with torch.no_grad():
                # 使用 embed_dim 作为输入通道数，模拟真实的 ESM 嵌入维度
                embed_dim = getattr(args, "embed_dim", args.hidden_dim)
                test_input = torch.randn(2, embed_dim, 128, 1).to(device)
                test_mask = torch.zeros(2, 128, dtype=torch.bool).to(device)
                from util.misc import NestedTensor
                test_nested = NestedTensor(test_input, test_mask)

                test_output = model(test_nested)
                # [2, num_queries, num_classes]
                test_logits = test_output['pred_logits']
                test_probs = test_logits.softmax(-1)
                test_labels = test_logits.argmax(-1)

                print(f"  • 测试输出shape: {test_logits.shape}")
                print(
                    f"  • 预测类别分布: {torch.unique(test_labels, return_counts=True)}")
                print(
                    f"  • 各类别平均概率: {[f'{p:.3f}' for p in test_probs.mean(dim=(0,1)).tolist()]}")

                if len(torch.unique(test_labels)) == 1:
                    print(f"  ⚠️  警告: 模型初始化后只预测一个类别!")
                else:
                    print(f"  ✅ 模型初始化正常，预测多个类别")
            print()

        model_without_ddp = model

        # 如需多卡，应使用分布式多进程模式；当前默认保持单进程单卡，避免与自定义 NestedTensor 的 DataParallel 兼容性问题
        if args.distributed:
            try:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], find_unused_parameters=True)
                model_without_ddp = model.module
                if is_main_process():
                    print(f"Fold {fold_idx + 1} DDP初始化成功")
            except Exception as e:
                if is_main_process():
                    print(f"Fold {fold_idx + 1} DDP初始化失败: {str(e)}")
                    print("降级为单GPU训练模式")
                args.distributed = False
                model_without_ddp = model

        # ===== 优化器和学习率调度器设置 =====
        # 为每个fold创建新的优化器
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
             if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        lr_scheduler.last_epoch = -1  # 重置调度器状态，确保每个fold都从初始学习率开始
        if is_main_process():
            print(f"Fold {fold_idx + 1} 学习率调度器已重置，初始学习率: {args.lr}")

        # ===== 预训练权重加载 =====
        # 加载预训练权重（如果有）
        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

        output_dir = Path(args.output_dir)
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        # ===== 评估模式 =====
        if args.eval:
            if data_loader_val is not None:
                val_stats = evaluate(model, criterion, postprocessors,
                                     data_loader_val, device, args.output_dir,
                                     epoch=0, fold=fold_idx+1,
                                     class_mapping=dataset_val.id2label,
                                     binary_mode=args.binary_mode,
                                     bg_class_index=args.num_classes - 1)
                if args.output_dir:
                    save_on_master(val_stats, output_dir / "validation.pth")
                all_fold_results.append(val_stats)
            else:
                if is_main_process():
                    print("警告: 未指定验证集，跳过评估模式")
            continue

        if data_loader_train is not None:
            if is_main_process():
                print("Start training")
            start_time = time.time()

            # ===== 训练循环 =====
        # 用于跟踪最佳模型
        best_f1 = 0.0
        best_epoch = 0

        # 用于跟踪每save_interval轮中的最佳模型
        interval_best_f1 = 0.0
        interval_best_epoch = 0
        interval_best_model_state = None
        interval_best_optimizer_state = None
        interval_best_lr_scheduler_state = None

        if is_main_process():
            print(f"\n开始训练 Fold {fold_idx + 1}")
            print(f"训练参数：")
            print(f"- 学习率: {args.lr}")
            print(f"- 批次大小: {args.batch_size}")
            print(f"- 保存间隔: {args.save_interval}轮")
            print(f"- 训练轮数: {args.epochs}")

        for epoch in range(args.start_epoch, args.epochs):
            try:
                if args.distributed:
                    sampler_train.set_epoch(epoch)

                # 训练一个epoch
                if is_main_process():
                    print(f"\n开始训练 epoch {epoch}...")
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, fold=fold_idx+1)
                lr_scheduler.step()

                # 评估当前模型
                if is_main_process():
                    print(f"\n开始评估 epoch {epoch}...")
                if args.ema:
                    # 初始化或更新 EMA 权重
                    if 'ema' not in locals():
                        ema = {k: v.detach().clone()
                               for k, v in model_without_ddp.state_dict().items()}
                    else:
                        with torch.no_grad():
                            for k, v in model_without_ddp.state_dict().items():
                                if k in ema:
                                    ema[k].mul_(args.ema_decay).add_(
                                        v.detach(), alpha=1.0 - args.ema_decay)
                                else:
                                    ema[k] = v.detach().clone()
                    # 备份、替换、评估、还原
                    backup = {k: v.detach().clone()
                              for k, v in model_without_ddp.state_dict().items()}
                    model_without_ddp.load_state_dict(ema, strict=False)
                    val_stats = evaluate(model, criterion, postprocessors,
                                         data_loader_val, device, args.output_dir, epoch,
                                         fold=fold_idx+1, class_mapping=dataset_val.id2label,
                                         binary_mode=args.binary_mode,
                                         bg_class_index=args.num_classes - 1)
                    model_without_ddp.load_state_dict(backup, strict=False)
                else:
                    val_stats = evaluate(model, criterion, postprocessors,
                                         data_loader_val, device, args.output_dir, epoch,
                                         fold=fold_idx+1, class_mapping=dataset_val.id2label,
                                         binary_mode=args.binary_mode,
                                         bg_class_index=args.num_classes - 1)
            except Exception as e:
                if is_main_process():
                    print(f"Epoch {epoch} 训练过程中出错: {str(e)}")
                    print("跳过当前epoch，继续下一个")
                continue

            # ===== 模型评估和保存 =====
            # 从验证集指标中直接获取F1分数（只在主进程中进行）
            if is_main_process():
                print(f"\n评估完成，开始计算F1分数...")
                val_f1 = val_stats.get('f1_score', 0.0)
                print(f"\n当前epoch {epoch} 的验证集F1分数: {val_f1:.4f}")

                # 更新全局最佳模型
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch = epoch
                    print(f"发现新的全局最佳F1分数: {best_f1:.4f} (epoch {best_epoch})")
                    # 保存全局最佳模型
                    if args.output_dir:
                        print("正在保存全局最佳模型...")
                        checkpoint_path = output_dir / \
                            f'checkpoint_fold_{fold_idx + 1}.pth'
                        save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'best_f1': best_f1,
                        }, checkpoint_path)
                        print("全局最佳模型保存完成")

                # 更新当前间隔内的最佳模型
                if val_f1 > interval_best_f1:
                    interval_best_f1 = val_f1
                    interval_best_epoch = epoch
                    interval_best_model_state = model_without_ddp.state_dict()
                    interval_best_optimizer_state = optimizer.state_dict()
                    interval_best_lr_scheduler_state = lr_scheduler.state_dict()
                    print(
                        f"更新当前{args.save_interval}轮内的最佳F1分数: {interval_best_f1:.4f} (epoch {interval_best_epoch})")

                # ===== 定期保存检查点 =====
                # 每save_interval轮保存一次历史检查点，保存该间隔内F1分数最高的模型
                if args.output_dir and (epoch + 1) % args.save_interval == 0:
                    print(f"定期保存检查点 (每{args.save_interval}轮，保存该间隔内最佳模型)...")
                    checkpoint_path = output_dir / \
                        f'checkpoint_fold_{fold_idx + 1}_epoch_{epoch}_best_in_interval.pth'

                    # 使用该间隔内最佳模型的状态进行保存
                    save_on_master({
                        'model': interval_best_model_state,
                        'optimizer': interval_best_optimizer_state,
                        'lr_scheduler': interval_best_lr_scheduler_state,
                        'epoch': interval_best_epoch,
                        'args': args,
                        'best_f1': interval_best_f1,
                        'best_epoch': interval_best_epoch,
                        'val_f1': interval_best_f1,
                        'interval_start_epoch': epoch - args.save_interval + 1,
                        'interval_end_epoch': epoch,
                    }, checkpoint_path)
                    print(f"间隔内最佳模型已保存: {checkpoint_path}")
                    print(
                        f"该间隔内最佳F1分数: {interval_best_f1:.4f} (epoch {interval_best_epoch})")

                    # 重置间隔内最佳模型跟踪
                    interval_best_f1 = 0.0
                    interval_best_epoch = 0
                    interval_best_model_state = None
                    interval_best_optimizer_state = None
                    interval_best_lr_scheduler_state = None

                print(f"\n准备开始下一个epoch {epoch + 1}...")

            # ===== 分布式训练同步 =====
            # 同步所有进程，确保所有进程都完成了当前epoch
            if args.distributed:
                try:
                    dist.barrier()
                except Exception as e:
                    if is_main_process():
                        print(f"Epoch {epoch} 同步时出错: {str(e)}")
                    # 如果同步失败，继续训练

        # ===== Fold结果记录 =====
        # 记录当前fold的结果（只在主进程中进行）
        if is_main_process():
            all_fold_results.append({
                'fold': fold_idx + 1,
                'best_epoch': best_epoch,
                'best_f1': best_f1
            })
            print(f"\nFold {fold_idx + 1} 训练完成")
            print(f"- 最佳F1分数: {best_f1:.4f}")
            print(f"- 最佳epoch: {best_epoch}")

        else:
            # 没有训练数据的处理
            if is_main_process():
                print("跳过训练阶段（未指定训练集）")
            best_f1 = 0.0
            best_epoch = 0

        # ===== Fold结果记录 =====
        # 记录当前fold的结果（只在主进程中进行）
        if is_main_process():
            all_fold_results.append({
                'fold': fold_idx + 1,
                'best_epoch': best_epoch,
                'best_f1': best_f1
            })
            if data_loader_train is not None:
                print(f"\nFold {fold_idx + 1} 训练完成")
                print(f"- 最佳F1分数: {best_f1:.4f}")
                print(f"- 最佳epoch: {best_epoch}")
            else:
                print(f"\nFold {fold_idx + 1} 完成（仅验证模式）")

            # 在fold之间进行同步和清理
        if args.distributed:
            try:
                # 确保所有进程都完成了当前fold
                dist.barrier()

                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 重置epoch计数器，为下一个fold做准备
                args.start_epoch = 0

                if is_main_process():
                    print(f"Fold {fold_idx + 1} 清理完成，准备开始下一个fold...")

            except Exception as e:
                if is_main_process():
                    print(f"Fold {fold_idx + 1} 清理时出错: {str(e)}")
                    print("继续下一个fold...")
                # 如果清理失败，继续下一个fold
                # 重置epoch计数器，为下一个fold做准备
                args.start_epoch = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ===== 最终结果统计 =====
    # 计算并打印所有fold的平均结果（只在主进程中进行）
    if is_main_process():
        if use_cross_validation:
            avg_f1 = sum(fold['best_f1']
                         for fold in all_fold_results) / n_folds
            print("\n" + "="*50)
            print("交叉验证最终结果：")
            print("="*50)
            print(f"平均最佳F1分数: {avg_f1:.4f}")
            for fold in all_fold_results:
                print(
                    f"Fold {fold['fold']}: 最佳F1分数 = {fold['best_f1']:.4f} (epoch {fold['best_epoch']})")
        else:
            print("\n" + "="*50)
            print("直接训练最终结果：")
            print("="*50)
            for fold in all_fold_results:
                print(
                    f"最佳F1分数: {fold['best_f1']:.4f} (epoch {fold['best_epoch']})")

    # 保存所有fold的结果
    if args.output_dir:
        if use_cross_validation:
            results_path = output_dir / 'cross_validation_results.json'
        else:
            results_path = output_dir / 'direct_training_results.json'
        save_on_master(all_fold_results, results_path)

    # 计算总时间
    if data_loader_train is not None and 'start_time' in locals():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    else:
        print('Total time: N/A (no training performed)')


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if getattr(args, 'binary_mode', False):
        print("⚠️  binary_mode 开关已启用：目前仅输出提示，后续阶段将逐步加入完整逻辑。")
    main(args)


# ===== 运行命令示例 =====

# CUDA_VISIBLE_DEVICES=0 python main_by_cluster.py \
#   --train_mapping_json data/converted/train/bgc_mapping.json \
#   --train_emb_dir    data/converted/train/embeddings_1280 \
#   --val_mapping_json data/converted/val/bgc_mapping.json \
#   --val_emb_dir      data/converted/val/embeddings_1280 \
#   --embed_dim 1280 \
#   --batch_size 128 \
#   --lr 1e-4 \
#   --lr_backbone 5e-5 \
#   --epochs 300 \
#   --balance_strategy weighted \
#   --output_dir outputs_1280_gpu0
