#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score

# 导入必要的模块
from models.detr import DETR
from models.transformer import build_transformer
from data import BalancedClusterDataset
from util.misc import collate_fn
from engine import evaluate

warnings.filterwarnings("ignore")


def load_model_from_checkpoint(checkpoint_path, args):
    """
    从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        args: 参数配置

    Returns:
        model: 加载的模型
        criterion: 损失函数
        postprocessors: 后处理器
        epoch: 训练轮数
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    try:
        # Try new PyTorch parameters
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Use old method if weights_only not supported
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None, None, None, None

    # 使用检查点中的参数来重建模型，确保参数一致性
    if 'args' in checkpoint:
        training_args = checkpoint['args']
        # 将训练时的关键参数复制到当前args中
        for key in ['num_classes', 'num_queries', 'hidden_dim', 'nheads',
                    'enc_layers', 'dec_layers', 'dim_feedforward', 'dropout',
                    'pre_norm', 'dilation', 'masks', 'aux_loss',
                    'focal_alpha', 'clip_max_norm', 'weight_decay', 'embed_dim',
                    'remove_difficult', 'set_cost_class', 'set_cost_bbox', 'set_cost_giou',
                    'ce_loss_coef', 'bbox_loss_coef', 'giou_loss_coef', 'cds_loss_coef', 'eos_coef']:
            if hasattr(training_args, key):
                setattr(args, key, getattr(training_args, key))

    # 创建模型
    from models import build_model
    model, criterion, postprocessors = build_model(args)

    # 加载模型权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 0)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0

    model.eval()
    print(f"模型加载成功，训练轮数: {epoch}")

    return model, criterion, postprocessors, epoch


def create_validation_dataset(args):
    """
    创建验证数据集，强制使用训练时的类别映射

    Args:
        args: 参数配置

    Returns:
        dataset: 验证数据集
        dataloader: 数据加载器
    """
    print("正在创建验证数据集...")

    # 🔥 强制使用训练时的类别映射
    # 训练时的类别映射（模型期望的）
    fixed_label2id = {
        "NRPS": 0,
        "PKS": 1,
        "other": 2,
        "ribosomal": 3,
        "saccharide": 4,
        "terpene": 5,
        "no-object": 6
    }

    # 🔥 处理验证数据集中的额外类别
    # 将RiPP映射到other类（因为训练时没有RiPP）
    print("⚠️  检测到验证数据集包含训练时没有的类别:")
    print("  RiPP -> 映射到 other (索引2)")

    # 创建数据映射转换函数
    def convert_type_to_train_mapping(type_name):
        if type_name == "RiPP":
            return "other"  # 将RiPP映射到other
        elif type_name in fixed_label2id:
            return type_name  # 保持原有映射
        else:
            print(f"⚠️  未知类别: {type_name} -> 映射到 other")
            return "other"  # 未知类别也映射到other

    print(f"使用固定类别映射: {fixed_label2id}")

    # 创建验证数据集
    val_dataset = BalancedClusterDataset(
        mapping=args.mapping_json,
        emb_dir=args.emb_dir,
        max_tokens=args.max_tokens,
        balance_strategy='none',  # 验证时不需要平衡
    )

    # 🔥 强制覆盖类别映射
    val_dataset.label2id = fixed_label2id
    val_dataset.num_classes = len(fixed_label2id)

    # 🔥 创建id2label映射
    val_dataset.id2label = {v: k for k, v in fixed_label2id.items()}

    print(f"验证集类别分布:")
    val_types = []
    with open(args.mapping_json, 'r') as f:
        mapping_data = json.load(f)
        for bgc_id, regions in mapping_data.items():
            # 空样本计为 no-object
            if not regions:
                val_types.append('no-object')
                continue
            for region in regions:
                val_types.append(region['type'])

    from collections import Counter
    type_counts = Counter(val_types)
    for cls, count in sorted(type_counts.items()):
        print(f"  {cls}: {count}个")

    # 创建数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f"验证集大小: {len(val_dataset)}")
    print(f"Batch数量: {len(val_dataloader)}")
    print(f"类别映射: {val_dataset.label2id}")

    return val_dataset, val_dataloader


def collect_detailed_predictions(model, data_loader, device, class_mapping, output_file, model_info=None, max_tokens=128, sample_level_fp=False):
    """
    收集每个样本的详细预测信息并保存到文件

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        class_mapping: 类别映射字典
        output_file: 输出文件路径
        model_info: 模型信息字典，包含模型名称、checkpoint路径等
        max_tokens: 最大序列长度，用于真实框反归一化
        sample_level_fp: 是否按样本级别计算假阳性（推荐用于负样本处理）
    """
    print(f"正在收集详细预测信息，保存到: {output_file}")

    model.eval()
    detailed_results = {
        'model_info': model_info or {},
        'samples': []
    }

    # 统计变量
    total_samples = 0
    negative_samples = 0
    negative_samples_with_fp = 0
    total_fp_count = 0  # 按样本级别计算的FP总数

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(data_loader, desc="收集预测信息")):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            # 模型前向传播
            outputs = model(samples)

            # 获取预测结果
            # [batch_size, num_queries, num_classes]
            pred_logits = outputs['pred_logits']
            # [batch_size, num_queries, 2]
            pred_boxes = outputs['pred_boxes']

            # 计算预测概率和类别
            pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_scores, pred_labels = pred_probs[..., :-1].max(-1)  # 排除背景类

            batch_size = pred_logits.shape[0]

            for i in range(batch_size):
                sample_result = {
                    'sample_id': batch_idx * data_loader.batch_size + i,
                    'predictions': [],
                    'ground_truth': [],
                    'is_negative_sample': False,
                    'has_positive_prediction': False
                }

                # 获取序列长度用于反归一化
                seq_len = targets[i].get('size', 128)
                if isinstance(seq_len, torch.Tensor):
                    if seq_len.numel() == 1:
                        seq_len = seq_len.item()
                    else:
                        seq_len = seq_len[0].item()

                # 获取max_tokens用于真实框反归一化
                # max_tokens = 128  # 默认值，应该与数据加载器中的max_tokens一致

                # 收集预测框信息
                positive_pred_count = 0  # 统计正样本预测数量
                for j in range(pred_logits.shape[1]):  # num_queries
                    pred_label = pred_labels[i, j].item()
                    pred_score = pred_scores[i, j].item()
                    pred_box = pred_boxes[i, j]

                    # 反归一化坐标
                    start_cds = int(pred_box[0].item() * seq_len)
                    end_cds = int(pred_box[1].item() * seq_len)

                    # 获取类别名称
                    pred_class_name = class_mapping.get(
                        pred_label, f"unknown_{pred_label}")

                    prediction_info = {
                        'query_id': j,
                        'class_id': pred_label,
                        'class_name': pred_class_name,
                        'confidence': pred_score,
                        'start_cds': start_cds,
                        'end_cds': end_cds,
                        'normalized_start': pred_box[0].item(),
                        'normalized_end': pred_box[1].item()
                    }
                    sample_result['predictions'].append(prediction_info)

                    # 统计正样本预测
                    if pred_label != 6:  # 不是no-object
                        positive_pred_count += 1

                # 收集真实框信息
                gt_boxes = targets[i]['boxes']
                gt_labels = targets[i]['labels']

                # 检查是否为负样本（没有真实框或只有no-object框）
                is_negative_sample = True
                for j in range(len(gt_boxes)):
                    gt_label = gt_labels[j].item()
                    gt_box = gt_boxes[j]

                    # 真实框坐标是用max_tokens归一化的，需要用max_tokens反归一化
                    gt_start_cds = int(gt_box[0].item() * max_tokens)
                    gt_end_cds = int(gt_box[1].item() * max_tokens)

                    # 获取类别名称
                    gt_class_name = class_mapping.get(
                        gt_label, f"unknown_{gt_label}")

                    gt_info = {
                        'gt_id': j,
                        'class_id': gt_label,
                        'class_name': gt_class_name,
                        'start_cds': gt_start_cds,
                        'end_cds': gt_end_cds,
                        'normalized_start': gt_box[0].item(),
                        'normalized_end': gt_box[1].item()
                    }
                    sample_result['ground_truth'].append(gt_info)

                    # 检查是否有非no-object的真实框
                    if gt_label != 6:  # 不是no-object
                        is_negative_sample = False

                # 标记负样本和正样本预测
                sample_result['is_negative_sample'] = is_negative_sample
                sample_result['has_positive_prediction'] = positive_pred_count > 0

                # 统计FP
                if is_negative_sample:
                    negative_samples += 1
                    if positive_pred_count > 0:
                        negative_samples_with_fp += 1
                        if sample_level_fp:
                            total_fp_count += 1  # 按样本级别，只加1个FP
                        else:
                            total_fp_count += positive_pred_count  # 按预测框级别

                total_samples += 1
                detailed_results['samples'].append(sample_result)

    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"详细预测信息已保存到: {output_file}")
    print(f"共处理了 {total_samples} 个样本")

    # 打印负样本分析结果
    if negative_samples > 0:
        print(f"\n负样本分析:")
        print(f"  总负样本数: {negative_samples}")
        print(f"  有假阳性的负样本数: {negative_samples_with_fp}")
        print(f"  负样本假阳性率: {negative_samples_with_fp/negative_samples:.2%}")

        if sample_level_fp:
            print(f"   按样本级别计算: 假阳性数 = {total_fp_count}")
            print(f"   推荐方式: 每个负样本最多贡献1个FP")
        else:
            print(f"   按预测框级别计算: 假阳性数 = {total_fp_count}")
            print(f"   建议使用 --sample_level_fp 参数来按样本级别计算假阳性")
    else:
        print("没有发现负样本")


def main():
    parser = argparse.ArgumentParser('DETR Final Validation')

    # 模型参数 - 需要与训练时保持一致
    parser.add_argument('--num_classes', default=7, type=int,
                        help='类别数量(包括no-object类)')  # 修改：使用固定值
    parser.add_argument('--num_queries', default=3, type=int, help='查询数量')
    parser.add_argument('--hidden_dim', default=256, type=int, help='隐藏层维度')
    parser.add_argument('--nheads', default=8, type=int, help='注意力头数')
    parser.add_argument('--enc_layers', default=6, type=int, help='编码器层数')
    parser.add_argument('--dec_layers', default=6, type=int, help='解码器层数')
    parser.add_argument('--dim_feedforward', default=2048,
                        type=int, help='前馈网络维度')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout率')
    parser.add_argument('--aux_loss', action='store_true',
                        default=True, help='使用辅助损失')
    parser.add_argument('--max_tokens', default=128, type=int, help='最大序列长度')

    # 添加其他必需的参数
    parser.add_argument('--backbone', default='esm2_t33_650M_UR50D', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--dilation', action='store_true', default=False)
    parser.add_argument('--ce_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.01,
                        type=float)  # no-object类权重系数
    parser.add_argument('--pre_norm', action='store_true',
                        default=False, help='是否在注意力前进行归一化')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.9, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--remove_difficult',
                        action='store_true', default=False)

    # 数据参数
    parser.add_argument('--mapping_json', default='data/test/bgc_mapping.json',
                        help='验证集BGC映射文件（单文件模式）')
    parser.add_argument('--mapping_dir', default=None,
                        help='验证集BGC映射目录（多文件/九基因组模式）')
    parser.add_argument('--emb_dir', default='data/test/embeddings',
                        help='验证集嵌入文件目录')
    parser.add_argument('--batch_size', default=8, type=int, help='批次大小')
    parser.add_argument('--num_workers', default=2, type=int, help='数据加载工作进程数')

    # 评估参数
    parser.add_argument('--device', default='cuda', help='使用设备')
    parser.add_argument('--output_dir', default='./outputs', help='输出目录')
    parser.add_argument('--validation_output_dir', default='./my_validation_results',
                        help='验证结果输出目录')
    parser.add_argument('--no_confidence_filter', action='store_true',
                        help='禁用置信度过滤，设置IoU阈值为0.1，显示所有预测结果')
    parser.add_argument('--localization_only', action='store_true',
                        help='只计算定位任务，忽略分类，匹配时只考虑IoU不考虑类别')

    # 新增：详细预测信息保存参数
    parser.add_argument('--save_detailed_predictions', action='store_true',
                        help='保存每个样本的详细预测信息到文件')
    parser.add_argument('--detailed_predictions_file', default='detailed_predictions.json',
                        help='详细预测信息保存文件名')
    parser.add_argument('--sample_level_fp', action='store_true',
                        help='按样本级别计算假阳性，而不是按预测框数量')

    args = parser.parse_args()

    print("=== DETR 最终验证 ===")
    print(f"使用设备: {args.device}")
    print(f"验证集映射文件: {args.mapping_json}")
    print(f"验证集嵌入目录: {args.emb_dir}")
    print(f"验证结果输出目录: {args.validation_output_dir}")
    print(f"详细结果保存目录: {args.validation_output_dir}")
    print(f"模型参数: 类别数={args.num_classes}, 查询数={args.num_queries}")
    if args.localization_only:
        print(" 定位任务模式: 已启用 (只计算定位，忽略分类)")
        print(" 匹配策略: 只考虑IoU，不考虑类别")
    if args.no_confidence_filter:
        print(" 置信度过滤: 已禁用 (显示所有预测)")
        print(" IoU阈值: 0.1 (更宽松的匹配条件)")
    else:
        print(" IoU阈值: 0.5 (标准匹配条件)")

    if args.save_detailed_predictions:
        print(" 详细预测信息保存: 已启用")
        print(f" 保存文件: {args.detailed_predictions_file}")

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"实际使用设备: {device}")

    # 如果提供了mapping目录，则启用多基因组评估模式
    if args.mapping_dir is not None:
        if not os.path.isdir(args.mapping_dir):
            print(f"错误: 映射目录不存在: {args.mapping_dir}")
            return
        mapping_files = [
            os.path.join(args.mapping_dir, f)
            for f in sorted(os.listdir(args.mapping_dir))
            if f.endswith('.json')
        ]
        if not mapping_files:
            print(f"错误: 映射目录中未找到json文件: {args.mapping_dir}")
            return
        print(f"检测到{len(mapping_files)}个基因组映射文件:")
        for mf in mapping_files:
            print(f"  - {os.path.basename(mf)}")
        multi_genome_mode = True
    else:
        multi_genome_mode = False
        # 创建验证数据集（单文件模式）
        val_dataset, val_dataloader = create_validation_dataset(args)
        print(f" 使用指定的类别数量: {args.num_classes}")
        print(f" 数据集类别映射: {val_dataset.label2id}")
        print(f" 数据集实际类别数: {val_dataset.num_classes}")

    # 找到所有fold的检查点
    checkpoint_files = []
    for filename in os.listdir(args.output_dir):
        if filename.startswith('checkpoint_fold_') and filename.endswith('.pth'):
            checkpoint_files.append(os.path.join(args.output_dir, filename))

    checkpoint_files.sort()
    print(f"\n找到 {len(checkpoint_files)} 个检查点文件:")
    for i, cp in enumerate(checkpoint_files):
        print(f"  {i+1}. {cp}")

    if not checkpoint_files:
        print("错误: 没有找到检查点文件!")
        return

    # 多基因组模式：按模型（checkpoint）× 基因组文件评估
    if multi_genome_mode:
        for cp_idx, checkpoint_path in enumerate(checkpoint_files):
            model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            model_out_dir = os.path.join(
                args.validation_output_dir, model_name)
            os.makedirs(model_out_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"评估模型: {model_name}")
            print(f"输出目录: {model_out_dir}")
            print(f"{'='*60}")

            # 加载模型
            model, criterion, postprocessors, epoch = load_model_from_checkpoint(
                checkpoint_path, args)
            if model is None or criterion is None or postprocessors is None:
                print(f"跳过无效的检查点: {checkpoint_path}")
                continue
            if epoch is None:
                epoch = 0
            model.to(device)

            # 置信度过滤设置（如果启用）
            original_eval_threshold = None
            original_iou_threshold = None
            original_forward = None
            if args.no_confidence_filter:
                print("⚠️  已禁用置信度过滤，将显示所有预测结果")
                print("⚠️  已设置IoU阈值为0.1")
                import engine
                original_eval_threshold = engine.EVAL_CONFIDENCE_THRESHOLD
                original_iou_threshold = engine.IOU_THRESHOLD
                engine.EVAL_CONFIDENCE_THRESHOLD = 0.0
                engine.IOU_THRESHOLD = 0.1
                from models.detr import PostProcess
                original_forward = PostProcess.forward

                def no_filter_forward(self, outputs, target_sizes):
                    import torch
                    import torch.nn.functional as F
                    logits = outputs['pred_logits']
                    boxes = outputs['pred_boxes']
                    prob = F.softmax(logits, -1)
                    scores, labels = prob[..., :-1].max(-1)
                    if isinstance(target_sizes, (list, tuple)):
                        if len(target_sizes) > 0 and isinstance(target_sizes[0], dict):
                            size_values = []
                            for ts in target_sizes:
                                size_value = ts.get('size', 1000)
                                if isinstance(size_value, torch.Tensor):
                                    size_value = size_value.item() if size_value.numel(
                                    ) == 1 else size_value[0].item()
                                elif not isinstance(size_value, (int, float)):
                                    size_value = 1000
                                size_values.append(size_value)
                            num_cds = torch.tensor(
                                size_values, device=boxes.device)
                        else:
                            num_cds = torch.tensor(
                                target_sizes, device=boxes.device)
                    elif isinstance(target_sizes, dict):
                        size_value = target_sizes.get('size', 1000)
                        if isinstance(size_value, torch.Tensor):
                            size_value = size_value.item() if size_value.numel(
                            ) == 1 else size_value[0].item()
                        elif not isinstance(size_value, (int, float)):
                            size_value = 1000
                        num_cds = torch.tensor(
                            [size_value], device=boxes.device)
                    else:
                        num_cds = target_sizes[:, 0] if target_sizes.dim(
                        ) > 1 else target_sizes
                    if num_cds.dim() == 0:
                        num_cds = num_cds.unsqueeze(0)
                    boxes = boxes * num_cds[:, None, None]
                    boxes = torch.clamp(boxes, min=0)
                    boxes = boxes.round().long()
                    predictions = []
                    for i in range(len(scores)):
                        sample_predictions = []
                        for j in range(len(scores[i])):
                            prediction = {
                                'start_cds': boxes[i, j, 0].item(),
                                'end_cds': boxes[i, j, 1].item(),
                                'class': labels[i, j].item(),
                                'score': scores[i, j].item()
                            }
                            sample_predictions.append(prediction)
                        sample_predictions.sort(
                            key=lambda x: x['score'], reverse=True)
                        predictions.append(sample_predictions)
                    results = []
                    for i in range(len(scores)):
                        results.append(
                            {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]})
                    return {'predictions': predictions, 'results': results}
                PostProcess.forward = no_filter_forward

            # 遍历九个基因组映射文件
            for mf in mapping_files:
                genome_name = os.path.splitext(os.path.basename(mf))[0]
                # 临时设置映射文件
                args.mapping_json = mf
                # 构建数据集/加载器
                val_dataset, val_dataloader = create_validation_dataset(args)

                # 评估（不保存详细预测）
                val_results_dir = os.path.join(model_out_dir, '_internal')
                os.makedirs(val_results_dir, exist_ok=True)
                try:
                    eval_stats = evaluate(
                        model=model,
                        criterion=criterion,
                        postprocessors=postprocessors,
                        data_loader=val_dataloader,
                        device=device,
                        log_dir=val_results_dir,
                        epoch=epoch,
                        fold=1,
                        class_mapping=val_dataset.id2label,
                        localization_only=args.localization_only,
                        sample_level_fp=args.sample_level_fp
                    )
                except Exception as e:
                    print(f"评估基因组{genome_name}时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # 保存简洁指标到模型目录下（一个基因组一个文件）
                out_metrics_path = os.path.join(
                    model_out_dir, f"{genome_name}_metrics.json")
                with open(out_metrics_path, 'w') as f:
                    json.dump(eval_stats, f, indent=2)
                print(
                    f"已保存 {model_name} 在 {genome_name} 上的指标: {out_metrics_path}")

            # 恢复置信度过滤设置
            if args.no_confidence_filter and original_eval_threshold is not None:
                import engine
                engine.EVAL_CONFIDENCE_THRESHOLD = original_eval_threshold
                engine.IOU_THRESHOLD = original_iou_threshold
                if original_forward is not None:
                    from models.detr import PostProcess
                    PostProcess.forward = original_forward

        print("\n验证完成!")
        return

    # 单文件模式：逐个评估每个fold
    all_results = []

    for i, checkpoint_path in enumerate(checkpoint_files):
        print(f"\n{'='*60}")
        print(f"评估 Fold {i+1}: {os.path.basename(checkpoint_path)}")
        print(f"{'='*60}")

        # 加载模型
        model, criterion, postprocessors, epoch = load_model_from_checkpoint(
            checkpoint_path, args)
        if model is None or criterion is None or postprocessors is None:
            print(f"跳过无效的检查点: {checkpoint_path}")
            continue

        # 确保epoch不为None
        if epoch is None:
            epoch = 0

        model.to(device)

        # 如果启用了禁用置信度过滤选项，临时修改全局阈值
        original_eval_threshold = None
        original_forward = None

        if args.no_confidence_filter:
            print("⚠️  已禁用置信度过滤，将显示所有预测结果")
            print("⚠️  已设置IoU阈值为0.1")

            # 临时修改engine中的评估置信度阈值和IoU阈值
            import engine
            original_eval_threshold = engine.EVAL_CONFIDENCE_THRESHOLD
            original_iou_threshold = engine.IOU_THRESHOLD
            engine.EVAL_CONFIDENCE_THRESHOLD = 0.0
            engine.IOU_THRESHOLD = 0.3  # 设置IoU阈值

            # 临时修改后处理器中的置信度阈值
            # 通过monkey patching修改PostProcess的forward方法
            from models.detr import PostProcess
            original_forward = PostProcess.forward

            def no_filter_forward(self, outputs, target_sizes):
                """不使用置信度过滤的后处理版本"""
                import torch
                import torch.nn.functional as F

                logits = outputs['pred_logits']
                boxes = outputs['pred_boxes']

                prob = F.softmax(logits, -1)
                scores, labels = prob[..., :-1].max(-1)

                # 处理target_sizes的各种格式
                if isinstance(target_sizes, (list, tuple)):
                    if len(target_sizes) > 0 and isinstance(target_sizes[0], dict):
                        size_values = []
                        for ts in target_sizes:
                            size_value = ts.get('size', 1000)
                            if isinstance(size_value, torch.Tensor):
                                size_value = size_value.item() if size_value.numel(
                                ) == 1 else size_value[0].item()
                            elif not isinstance(size_value, (int, float)):
                                size_value = 1000
                            size_values.append(size_value)
                        num_cds = torch.tensor(
                            size_values, device=boxes.device)
                    else:
                        num_cds = torch.tensor(
                            target_sizes, device=boxes.device)
                elif isinstance(target_sizes, dict):
                    size_value = target_sizes.get('size', 1000)
                    if isinstance(size_value, torch.Tensor):
                        size_value = size_value.item() if size_value.numel(
                        ) == 1 else size_value[0].item()
                    elif not isinstance(size_value, (int, float)):
                        size_value = 1000
                    num_cds = torch.tensor([size_value], device=boxes.device)
                else:
                    num_cds = target_sizes[:, 0] if target_sizes.dim(
                    ) > 1 else target_sizes

                if num_cds.dim() == 0:
                    num_cds = num_cds.unsqueeze(0)

                boxes = boxes * num_cds[:, None, None]
                boxes = torch.clamp(boxes, min=0)
                boxes = boxes.round().long()

                # 构建预测结果（不使用置信度过滤）
                predictions = []
                for i in range(len(scores)):
                    sample_predictions = []
                    for j in range(len(scores[i])):
                        # 移除置信度过滤，显示所有预测
                        prediction = {
                            'start_cds': boxes[i, j, 0].item(),
                            'end_cds': boxes[i, j, 1].item(),
                            'class': labels[i, j].item(),
                            'score': scores[i, j].item()
                        }
                        sample_predictions.append(prediction)

                    # 按置信度排序
                    sample_predictions.sort(
                        key=lambda x: x['score'], reverse=True)
                    predictions.append(sample_predictions)

                # 保持向后兼容的格式
                results = []
                for i in range(len(scores)):
                    results.append({
                        'scores': scores[i],
                        'labels': labels[i],
                        'boxes': boxes[i]
                    })

                return {'predictions': predictions, 'results': results}

            # 替换方法
            PostProcess.forward = no_filter_forward

        # 创建验证结果输出目录
        os.makedirs(args.validation_output_dir, exist_ok=True)

        # 创建val_results目录用于保存详细的评估结果
        val_results_dir = os.path.join(
            args.validation_output_dir, 'val/val_results')
        os.makedirs(val_results_dir, exist_ok=True)

        # 运行评估
        try:
            eval_stats = evaluate(
                model=model,
                criterion=criterion,
                postprocessors=postprocessors,
                data_loader=val_dataloader,
                device=device,
                log_dir=val_results_dir,  # 使用val_results目录保存详细结果
                epoch=epoch,
                fold=i+1,
                class_mapping=val_dataset.id2label,  # 传递类别映射
                localization_only=args.localization_only,  # 传递定位任务模式参数
                sample_level_fp=args.sample_level_fp  # 传递样本级别FP计算参数
            )

            fold_result = {
                'fold': i + 1,
                'checkpoint': os.path.basename(checkpoint_path),
                'epoch': epoch,
                **eval_stats
            }
            all_results.append(fold_result)

            print(f"\nFold {i+1} 评估结果:")
            if args.localization_only:
                print(
                    f"  Localization Precision: {eval_stats.get('precision', 0):.4f}")
                print(
                    f"  Localization Recall: {eval_stats.get('recall', 0):.4f}")
                print(
                    f"  Localization F1: {eval_stats.get('f1_score', 0):.4f}")
            else:
                print(
                    f"  Classification F1: {eval_stats.get('classification_f1', 0):.4f}")
                print(
                    f"  Classification AUC: {eval_stats.get('classification_auc', 0):.4f}")
                print(
                    f"  Detection Precision: {eval_stats.get('precision', 0):.4f}")
                print(f"  Detection Recall: {eval_stats.get('recall', 0):.4f}")
                print(f"  Detection F1: {eval_stats.get('f1_score', 0):.4f}")

            # 🔥 新增：显示CDS级别指标（如果存在）
            if 'cds_level_metrics' in eval_stats:
                cds_metrics = eval_stats['cds_level_metrics']
                print(
                    f"  CDS Level Precision: {cds_metrics.get('precision', 0):.4f}")
                print(
                    f"  CDS Level Recall: {cds_metrics.get('recall', 0):.4f}")
                print(f"  CDS Level F1: {cds_metrics.get('f1_score', 0):.4f}")
                print(f"  CDS Level AUC: {cds_metrics.get('auc', 0):.4f}")
                print(f"  CDS Level TPR: {cds_metrics.get('tpr', 0):.4f}")
                print(f"  CDS Level FPR: {cds_metrics.get('fpr', 0):.4f}")
                print(
                    f"  CDS Level Stats: TP={cds_metrics.get('true_positive', 0)}, FP={cds_metrics.get('false_positive', 0)}, FN={cds_metrics.get('false_negative', 0)}, TN={cds_metrics.get('true_negative', 0)}")

            # 🔥 新增：保存详细预测信息
            if args.save_detailed_predictions:
                # 从checkpoint路径提取模型信息
                checkpoint_name = os.path.basename(checkpoint_path)
                checkpoint_name_without_ext = os.path.splitext(checkpoint_name)[
                    0]

                # 生成包含训练模型信息的文件名
                detailed_filename = f"detailed_predictions_{checkpoint_name_without_ext}.json"
                detailed_file = os.path.join(
                    args.validation_output_dir,
                    detailed_filename
                )

                collect_detailed_predictions(
                    model=model,
                    data_loader=val_dataloader,
                    device=device,
                    class_mapping=val_dataset.id2label,
                    output_file=detailed_file,
                    model_info={
                        'checkpoint_path': checkpoint_path,
                        'checkpoint_name': checkpoint_name,
                        'epoch': epoch,
                        'model_name': args.backbone,
                        'num_classes': args.num_classes,
                        'num_queries': args.num_queries,
                        'localization_only': args.localization_only,
                        'no_confidence_filter': args.no_confidence_filter,
                        'sample_level_fp': args.sample_level_fp
                    },
                    max_tokens=args.max_tokens,  # 传递max_tokens参数
                    sample_level_fp=args.sample_level_fp  # 传递sample_level_fp参数
                )

        except Exception as e:
            print(f"评估 Fold {i+1} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # 恢复原始设置
            if args.no_confidence_filter and original_eval_threshold is not None:
                import engine
                engine.EVAL_CONFIDENCE_THRESHOLD = original_eval_threshold
                engine.IOU_THRESHOLD = original_iou_threshold

                # 恢复后处理器的原始方法
                if original_forward is not None:
                    from models.detr import PostProcess
                    PostProcess.forward = original_forward

    # 汇总结果
    if all_results:
        print(f"\n{'='*60}")
        print("所有Fold评估结果汇总")
        print(f"{'='*60}")

        if args.localization_only:
            metrics = ['precision', 'recall', 'f1_score']
            metric_names = ['Localization Precision',
                            'Localization Recall', 'Localization F1']
        else:
            metrics = ['classification_f1', 'classification_auc',
                       'precision', 'recall', 'f1_score']
            metric_names = ['Classification F1', 'Classification AUC',
                            'Detection Precision', 'Detection Recall', 'Detection F1']

        for metric, metric_name in zip(metrics, metric_names):
            values = [r.get(metric, 0)
                      for r in all_results if r.get(metric) is not None]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")

        # 🔥 新增：CDS级别指标汇总
        print(f"\n{'='*60}")
        print("CDS级别评估结果汇总")
        print(f"{'='*60}")

        cds_metrics = ['precision', 'recall', 'f1_score', 'auc', 'tpr', 'fpr']
        cds_metric_names = ['CDS Level Precision',
                            'CDS Level Recall', 'CDS Level F1', 'CDS Level AUC', 'CDS Level TPR', 'CDS Level FPR']

        for metric, metric_name in zip(cds_metrics, cds_metric_names):
            values = []
            for r in all_results:
                if 'cds_level_metrics' in r and r['cds_level_metrics'].get(metric) is not None:
                    values.append(r['cds_level_metrics'][metric])

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{metric_name:20s}: 无数据")

        # 保存汇总结果
        results_file = os.path.join(
            args.validation_output_dir, 'final_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n汇总结果已保存到: {results_file}")
        print(
            f"详细评估结果已保存到: {args.validation_output_dir}")

    print("\n验证完成!")


if __name__ == '__main__':
    main()


# cd /share/org/BGI/bgi_suncl/project/BGC-DETR

# python validate_final.py \
#   --mapping_dir data/test/genome_mappings \
#   --emb_dir data/test/embeddings_1280 \
#   --output_dir outputs_1280_gpu0 \
#   --validation_output_dir outputs_1280_gpu0_test_all_with_detail \
#   --localization_only \
#   --sample_level_fp 
#   --save_detailed_predictions
