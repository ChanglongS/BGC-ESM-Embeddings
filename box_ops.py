# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
边界框操作和 GIoU 计算的工具函数模块

该模块提供了用于处理边界框的各种操作函数，专门针对一维序列（如蛋白质序列）的边界框检测任务。
主要功能包括：
1. 边界框格式转换（中心点坐标 ↔ 角点坐标）
2. 一维IoU计算
3. 一维GIoU计算
4. 二维边界框操作（用于兼容性）

输入维度：
- segments1: [N, 2] 张量，表示预测的起始和结束位置
- segments2: [M, 2] 张量，表示目标的起始和结束位置
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    """
    将中心点坐标和宽高转换为左上角和右下角坐标

    转换公式：
    - x0 = x_c - 0.5 * w
    - y0 = y_c - 0.5 * h
    - x1 = x_c + 0.5 * w
    - y1 = y_c + 0.5 * h

    Args:
        x: [N, 4] 张量，格式为 (x_c, y_c, w, h)

    Returns:
        [N, 4] 张量，格式为 (x0, y0, x1, y1)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    将左上角和右下角坐标转换为中心点坐标和宽高

    转换公式：
    - x_c = (x0 + x1) / 2
    - y_c = (y0 + y1) / 2
    - w = x1 - x0
    - h = y1 - y0

    Args:
        x: [N, 4] 张量，格式为 (x0, y0, x1, y1)

    Returns:
        [N, 4] 张量，格式为 (x_c, y_c, w, h)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def segment_iou(segments1, segments2):
    """
    计算一维边界框（线段）的 IoU（交并比）

    一维IoU计算原理：
    1. 计算两个线段的交集长度
    2. 计算两个线段的并集长度
    3. IoU = 交集长度 / 并集长度

    Args:
        segments1: [N, 2] 张量，表示预测的起始和结束位置
        segments2: [M, 2] 张量，表示目标的起始和结束位置

    Returns:
        [N, M] 张量，表示每对线段之间的 IoU
    """
    # 确保输入格式正确
    assert segments1.shape[-1] == 2 and segments2.shape[-1] == 2

    # ===== 1. 计算重叠部分 =====
    # 交集起始位置：两个线段起始位置的最大值
    start = torch.max(segments1[:, None, 0], segments2[:, 0])  # [N, M]

    # 交集结束位置：两个线段结束位置的最小值
    end = torch.min(segments1[:, None, 1], segments2[:, 1])    # [N, M]

    # ===== 2. 计算重叠长度 =====
    # 重叠长度 = 结束位置 - 起始位置，最小为0
    overlap = torch.clamp(end - start, min=0)  # [N, M]

    # ===== 3. 计算每个线段的长度 =====
    # 第一个线段的长度
    length1 = segments1[:, 1] - segments1[:, 0]  # [N]

    # 第二个线段的长度
    length2 = segments2[:, 1] - segments2[:, 0]  # [M]

    # ===== 4. 计算并集长度 =====
    # 并集长度 = 第一个线段长度 + 第二个线段长度 - 重叠长度
    union = length1[:, None] + length2 - overlap  # [N, M]

    # ===== 5. 计算 IoU =====
    # IoU = 重叠长度 / 并集长度，添加小量避免除零
    iou = overlap / (union + 1e-6)
    return iou


def segment_giou(segments1, segments2):
    """
    计算一维边界框（线段）的 GIoU（广义交并比）

    GIoU计算原理：
    1. 计算IoU
    2. 计算最小外接区间
    3. GIoU = IoU - (外接区间长度 - 并集长度) / 外接区间长度

    GIoU的优势：
    - 当两个线段完全重叠时，GIoU = 1
    - 当两个线段完全不重叠时，GIoU = -1
    - 当两个线段部分重叠时，GIoU 在 [-1, 1] 之间
    - 比IoU更能反映线段的相对位置关系

    Args:
        segments1: [N, 2] 张量，表示预测的起始和结束位置
        segments2: [M, 2] 张量，表示目标的起始和结束位置

    Returns:
        [N, M] 张量，表示每对线段之间的 GIoU，范围在[-1, 1]之间
    """
    # ===== 1. 计算重叠部分 =====
    # 交集起始位置
    start = torch.max(segments1[:, None, 0], segments2[:, 0])  # [N, M]

    # 交集结束位置
    end = torch.min(segments1[:, None, 1], segments2[:, 1])    # [N, M]

    # 重叠长度
    overlap = torch.clamp(end - start, min=0)  # [N, M]

    # ===== 2. 计算每个线段的长度 =====
    # 第一个线段的长度
    length1 = segments1[:, 1] - segments1[:, 0]  # [N]

    # 第二个线段的长度
    length2 = segments2[:, 1] - segments2[:, 0]  # [M]

    # ===== 3. 计算并集长度 =====
    # 并集长度 = 第一个线段长度 + 第二个线段长度 - 重叠长度
    union = length1[:, None] + length2 - overlap  # [N, M]

    # ===== 4. 计算最小外接区间 =====
    # 外接区间起始位置：两个线段起始位置的最小值
    min_start = torch.min(segments1[:, None, 0], segments2[:, 0])  # [N, M]

    # 外接区间结束位置：两个线段结束位置的最大值
    max_end = torch.max(segments1[:, None, 1], segments2[:, 1])    # [N, M]

    # 外接区间长度
    outer_length = max_end - min_start  # [N, M]

    # ===== 5. 计算 IoU =====
    # IoU = 重叠长度 / 并集长度
    iou = overlap / (union + 1e-6)

    # ===== 6. 计算 GIoU =====
    # GIoU = IoU - (外接区间长度 - 并集长度) / 外接区间长度
    # 当两个线段完全重叠时，GIoU = 1
    # 当两个线段完全不重叠时，GIoU = -1
    # 当两个线段部分重叠时，GIoU 在 [-1, 1] 之间
    giou = iou - (outer_length - union) / (outer_length + 1e-6)

    # 确保 GIoU 在 [-1, 1] 范围内
    giou = torch.clamp(giou, min=-1.0, max=1.0)

    return giou


def box_iou(boxes1, boxes2):
    """
    计算二维边界框的 IoU 和并集面积

    该函数用于兼容性，处理二维边界框的IoU计算

    Args:
        boxes1: [N, 4] 张量，格式为 (x0, y0, x1, y1)
        boxes2: [M, 4] 张量，格式为 (x0, y0, x1, y1)

    Returns:
        tuple: (iou, union)
            - iou: [N, M] IoU矩阵
            - union: [N, M] 并集面积矩阵
    """
    # 计算每个边界框的面积
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # 计算交集区域
    # 左上角坐标：两个边界框左上角坐标的最大值
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

    # 右下角坐标：两个边界框右下角坐标的最小值
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集区域的宽高
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # 计算交集面积
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # 计算并集面积
    union = area1[:, None] + area2 - inter

    # 计算IoU
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义 IoU (GIoU)
    边界框格式应为 [x0, y0, x1, y1]

    GIoU计算原理：
    1. 计算IoU
    2. 计算最小外接矩形
    3. GIoU = IoU - (外接矩形面积 - 并集面积) / 外接矩形面积

    返回 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)

    Args:
        boxes1: [N, 4] 张量，格式为 (x0, y0, x1, y1)
        boxes2: [M, 4] 张量，格式为 (x0, y0, x1, y1)

    Returns:
        [N, M] GIoU矩阵
    """
    # 检查边界框是否有效
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 计算IoU和并集面积
    iou, union = box_iou(boxes1, boxes2)

    # 计算最小外接矩形
    # 左上角坐标：两个边界框左上角坐标的最小值
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])

    # 右下角坐标：两个边界框右下角坐标的最大值
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算外接矩形的宽高
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # 计算外接矩形面积
    area = wh[:, :, 0] * wh[:, :, 1]

    # 计算GIoU
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """
    计算掩码的边界框

    该函数用于从掩码中提取边界框，通常用于分割任务

    Args:
        masks: [N, H, W] 格式的掩码，其中 N 是掩码数量，(H, W) 是空间维度

    Returns:
        [N, 4] 张量，边界框格式为 xyxy
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    # 获取掩码的空间维度
    h, w = masks.shape[-2:]

    # 创建坐标网格
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    # 计算x坐标的边界
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # 计算y坐标的边界
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # 返回边界框坐标
    return torch.stack([x_min, y_min, x_max, y_max], 1)
