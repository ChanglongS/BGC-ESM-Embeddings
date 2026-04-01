#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 segments_*.csv 生成每个基因组的coverage图片（SVG/PNG）。

CSV要求包含下面字段（可带多余空白，本脚本会strip）：
- segment_id, genome, segment_index, query_id, class_name, confidence, start_cds, end_cds, cds_ids

说明：
- 对每一行预测区间，构建该基因组的coverage数组，并对区间覆盖+1；
- 若能从 cds_ids 中解析到具体的CDS索引（如 101_1_750），按索引逐个+1（更精确，兼容区间是否闭区间差异）；
- 若无法从 cds_ids 解析出索引，则退回按 [start_cds, end_cds) 或 [start_cds, end_cds] 估计：
  * 若 cds_ids 长度与 end-start 相等，则按半开区间 [start, end)
  * 若与 end-start+1 相等，则按闭区间 [start, end]
  * 否则默认闭区间 [start, end]

输出：
- 在 --output_dir 下，为每个输入CSV创建子目录（以文件名去扩展名为名），
  其中包含每个基因组一张 coverage_{model}_{genome}.(svg/png) 图片。

示例：
  python BGC-DETR/scripts/plot_segments_coverage.py \
    --csv /path/to/segments_filtered.csv /path/to/segments_antismash.csv \
    --output_dir /path/to/coverage_plots \
    --image_format svg png --dpi 120
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="从 segments_*.csv 生成coverage图片")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="一个或多个segments_*.csv文件")
    ap.add_argument("--output_dir", required=True, help="输出根目录")
    ap.add_argument("--image_format", nargs="+", default=["svg"],
                    choices=["svg", "png"], help="输出图片格式（可多选）")
    ap.add_argument("--dpi", type=int, default=150, help="PNG分辨率DPI")
    ap.add_argument("--title_suffix", default="", help="图片标题后缀，可为空")
    return ap.parse_args()


_int_re = re.compile(r"^-?\d+$")


def _to_int_safe(s: str, default: int = 0) -> int:
    s = (s or "").strip()
    if _int_re.match(s):
        try:
            return int(s)
        except Exception:
            return default
    try:
        return int(float(s))
    except Exception:
        return default


def _clean_header(headers: List[str]) -> List[str]:
    return [h.strip() for h in headers]


def _parse_row(header: List[str], row_vals: List[str]) -> Dict[str, str]:
    data = {}
    for i, key in enumerate(header):
        val = row_vals[i] if i < len(row_vals) else ""
        data[key] = val.strip()
    return data


def _parse_indices_from_cds_ids(cds_ids_field: str) -> List[int]:
    """尝试从 cds_ids 中解析CDS的数字索引（取每个ID最后一个下划线后的整数）。"""
    if not cds_ids_field:
        return []
    tokens = cds_ids_field.strip().split()
    indices: List[int] = []
    for tok in tokens:
        parts = tok.split("_")
        if not parts:
            continue
        last = parts[-1]
        if _int_re.match(last):
            indices.append(int(last))
        else:
            # 有些ID末尾可能带非纯数字，尝试提取数字尾缀
            m = re.search(r"(\d+)$", last)
            if m:
                indices.append(int(m.group(1)))
    return indices


def build_coverage_from_csv(csv_path: str) -> Dict[str, List[int]]:
    """读取单个CSV，返回每个基因组的coverage数组（1-based索引转换为数组下标时减1）。"""
    genome_to_intervals: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            return {}
        header = _clean_header(raw_header)

        # 兼容常见字段名
        required = [
            "segment_id", "genome", "segment_index", "query_id",
            "class_name", "confidence", "start_cds", "end_cds", "cds_ids"
        ]
        # 如果列顺序不同，也没问题，按名称匹配
        # 若缺少必须列，尝试宽松处理（至少需要 genome/start_cds/end_cds）

        for row in reader:
            if not row:
                continue
            row_data = _parse_row(header, row)

            genome = row_data.get("genome") or row_data.get("genome ") or row_data.get(" genome")
            if genome is None:
                # 尝试从 segment_id 中提取
                genome = (row_data.get("segment_id") or "").strip().split("_")
                genome = genome[1] if len(genome) > 1 else "unknown"
            genome = genome.strip()

            start_cds = _to_int_safe(row_data.get("start_cds", "0"), 0)
            end_cds = _to_int_safe(row_data.get("end_cds", "0"), 0)
            cds_ids = row_data.get("cds_ids", "")

            # 优先从 cds_ids 解析确切索引
            indices = _parse_indices_from_cds_ids(cds_ids)

            # 记录：区间、以及可能的indices（用于更精确的覆盖）
            genome_to_intervals[genome].append((start_cds, end_cds, indices))

    # 构建coverage
    genome_to_coverage: Dict[str, List[int]] = {}
    for genome, items in genome_to_intervals.items():
        # 估计最大长度
        max_end = 0
        for (s, e, idxs) in items:
            if idxs:
                local_max = max(idxs)
                if local_max > max_end:
                    max_end = local_max
            else:
                if e > max_end:
                    max_end = e
        # 1-based索引，数组开到 max_end
        coverage = [0] * max(1, max_end)

        for (s, e, idxs) in items:
            if idxs:
                for idx in idxs:
                    pos = idx - 1  # 1-based -> 0-based
                    if 0 <= pos < len(coverage):
                        coverage[pos] += 1
                continue

            # 退回到区间覆盖
            # 尝试判断闭区间/半开区间（无法判断时默认闭区间）
            length_est = max(0, e - s)
            # 无法知道 cds_ids 数量，默认闭区间
            start = min(s, e)
            end = max(s, e)
            # 默认闭区间
            for pos1 in range(start, end + 1):
                pos0 = pos1 - 1
                if 0 <= pos0 < len(coverage):
                    coverage[pos0] += 1

        genome_to_coverage[genome] = coverage

    return genome_to_coverage


def _model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def plot_and_save(genome_to_coverage: Dict[str, List[int]], out_dir: str, model_name: str,
                  formats: List[str], dpi: int, title_suffix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for genome, cov in sorted(genome_to_coverage.items(), key=lambda kv: kv[0]):
        if not cov:
            continue
        x = list(range(1, len(cov) + 1))
        y = cov

        plt.figure(figsize=(12, 3))
        plt.plot(x, y, color="#1f77b4", linewidth=1)
        plt.fill_between(x, y, color="#1f77b4", alpha=0.2)
        plt.xlabel("CDS index")
        plt.ylabel("Coverage")
        title = f"{model_name} - Genome {genome} coverage"
        if title_suffix:
            title += f" {title_suffix}"
        plt.title(title)
        plt.tight_layout()

        for fmt in formats:
            out_path = os.path.join(out_dir, f"coverage_{model_name}_{genome}.{fmt}")
            if fmt == "png":
                plt.savefig(out_path, dpi=dpi)
            else:
                plt.savefig(out_path)
        plt.close()


def main() -> None:
    args = parse_args()
    root_out = os.path.abspath(args.output_dir)
    os.makedirs(root_out, exist_ok=True)

    for csv_path in args.csv:
        csv_path = os.path.abspath(csv_path)
        if not os.path.isfile(csv_path):
            print(f"跳过不存在的CSV: {csv_path}")
            continue
        model_name = _model_name_from_path(csv_path)
        sub_out = os.path.join(root_out, model_name)
        os.makedirs(sub_out, exist_ok=True)

        print(f"[Load] {csv_path}")
        genome_cov = build_coverage_from_csv(csv_path)
        if not genome_cov:
            print(f"[Warn] {csv_path} 未解析到任何数据")
            continue

        print(f"[Plot] 写入: {sub_out}")
        plot_and_save(genome_cov, sub_out, model_name, args.image_format, args.dpi, args.title_suffix)

        print(f"[OK] {model_name}: 共 {len(genome_cov)} 个基因组完成绘图")


if __name__ == "__main__":
    main()


