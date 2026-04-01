#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给预测结果 CSV 追加核苷酸长度列。

支持两类输入：
1. segments_filtered.csv：根据 cds_ids 回溯整段预测的核苷酸长度
2. cds_filtered.csv：根据 cds_id 回溯单个 CDS 的核苷酸长度

说明：
- 坐标来自同批次 Prodigal 生成的 genome.protein_*.fa 头部：
    >4-4_1_1 # 94 # 2814 # -1 # ...
- 对 segments_filtered.csv：
    * 若所有 CDS 位于同一 contig，则输出从最小 start 到最大 end 的区间长度（含首尾）
    * 若跨 contig，则无法定义单一基因组区间，回退为各 CDS 核苷酸长度求和
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from Bio import SeqIO


HEADER_RE = re.compile(r"^\S+\s+#\s+(\d+)\s+#\s+(\d+)\s+#\s+(-?\d+)\s+#")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为预测结果 CSV 追加核苷酸长度列")
    parser.add_argument("--input_csv", required=True, help="输入 CSV 文件路径")
    parser.add_argument(
        "--proteins_dir",
        default=None,
        help="Prodigal 输出的 proteins 目录；若不提供，则尝试从 input_csv 自动推断",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="输出 CSV 路径；默认在原文件名后追加 _with_nt_length",
    )
    return parser.parse_args()


def infer_proteins_dir(input_csv: Path) -> Optional[Path]:
    # 标准布局：.../GeneOutputs/<run_dir>/filtered/*.csv -> sibling proteins/
    base_dir = input_csv.parent.parent
    candidate = base_dir / "proteins"
    if candidate.is_dir():
        return candidate
    return None


def build_cds_index(proteins_dir: Path) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    fasta_files = sorted(proteins_dir.glob("genome.protein_*.fa"))
    if not fasta_files:
        raise FileNotFoundError(f"未在 proteins 目录中找到 genome.protein_*.fa: {proteins_dir}")

    for fasta_file in fasta_files:
        raw_genome_id = fasta_file.name.replace("genome.protein_", "").replace(".fa", "")
        pred_genome_id = f"PRED_{raw_genome_id}"

        for record in SeqIO.parse(str(fasta_file), "fasta"):
            match = HEADER_RE.match(record.description)
            if not match:
                continue

            start = int(match.group(1))
            end = int(match.group(2))
            strand = int(match.group(3))
            parts = record.id.split("_")
            contig_id = "_".join(parts[:-1]) if len(parts) >= 2 else record.id
            full_cds_id = f"{pred_genome_id}_{record.id}"

            index[full_cds_id] = {
                "start": min(start, end),
                "end": max(start, end),
                "strand": strand,
                "nt_length": abs(end - start) + 1,
                "contig_id": contig_id,
                "genome_id": pred_genome_id,
            }

    if not index:
        raise RuntimeError(f"未能从 proteins 目录解析任何 CDS 坐标: {proteins_dir}")
    return index


def compute_segment_nt_length(cds_ids: List[str], cds_index: Dict[str, dict]) -> str:
    coords = [cds_index[cds_id] for cds_id in cds_ids if cds_id in cds_index]
    if not coords:
        return ""

    contigs = {item["contig_id"] for item in coords}
    if len(contigs) == 1:
        seg_start = min(item["start"] for item in coords)
        seg_end = max(item["end"] for item in coords)
        return str(seg_end - seg_start + 1)

    # 若跨 contig，则无法定义单一的连续核苷酸区间，回退为 CDS 核苷酸长度之和
    total = sum(item["nt_length"] for item in coords)
    return str(total)


def process_csv(input_csv: Path, output_csv: Path, cds_index: Dict[str, dict]) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV 缺少表头: {input_csv}")

        fieldnames = list(reader.fieldnames)
        new_field = "nucleotide_length_bp"
        if new_field not in fieldnames:
            fieldnames.append(new_field)

        rows = []
        for row in reader:
            if "cds_id" in row:
                cds_id = row["cds_id"].strip()
                row[new_field] = str(cds_index[cds_id]["nt_length"]) if cds_id in cds_index else ""
            elif "cds_ids" in row:
                cds_ids = [x for x in row["cds_ids"].strip().split() if x]
                row[new_field] = compute_segment_nt_length(cds_ids, cds_index)
            else:
                raise RuntimeError(
                    f"暂不支持该 CSV 格式，未找到 cds_id 或 cds_ids 列: {input_csv}"
                )
            rows.append(row)

    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    proteins_dir = Path(args.proteins_dir).resolve() if args.proteins_dir else infer_proteins_dir(input_csv)
    if proteins_dir is None or not proteins_dir.is_dir():
        raise FileNotFoundError(
            "无法自动推断 proteins 目录，请显式提供 --proteins_dir"
        )

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
    else:
        output_csv = input_csv.with_name(f"{input_csv.stem}_with_nt_length{input_csv.suffix}")

    cds_index = build_cds_index(proteins_dir)
    process_csv(input_csv, output_csv, cds_index)

    print(f"输入文件: {input_csv}")
    print(f"proteins 目录: {proteins_dir}")
    print(f"输出文件: {output_csv}")


if __name__ == "__main__":
    main()


# python3 /share/org/BGI/bgi_suncl/project/BGC-DETR/scripts/add_nucleotide_length_to_csv.py \
#   --input_csv /share/org/BGI/bgi_suncl/project/GeneOutputs/BGC_DETR_sags_drep_20260318_163042/filtered/segments_filtered.csv



# python3 /share/org/BGI/bgi_suncl/project/BGC-DETR/scripts/add_nucleotide_length_to_csv.py \
#   --input_csv /share/org/BGI/bgi_suncl/project/GeneOutputs/BGC_DETR_sags_drep_20260318_163042/filtered/cds_filtered.csv
