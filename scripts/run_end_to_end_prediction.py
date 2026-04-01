#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端预测流水线：
从核酸FASTA输入 -> Prodigal预测蛋白 -> 生成BGC-DETR输入数据 -> 运行验证得到详细预测 ->
基于置信度阈值导出CSV。中间产物默认写入临时目录并清理。

依赖现有模块：
- run_prodigal_for_genomes.py（调用其函数批量运行Prodigal）
- process_prediction_data_fixed.py（解析protein FASTA为BGC-DETR所需json）
- validate_final.py（评估并保存 detailed_predictions_*.json）
- scripts/export_filtered_predictions.py（将 detailed_predictions + bgc_proteins 导出CSV）

示例：

  python BGC-DETR/scripts/run_end_to_end_prediction.py \
    --input /path/to/genomes_dir_or_fasta \
    --emb_dir /path/to/embeddings \
    --checkpoints_dir /path/to/outputs \
    --final_output_dir /path/to/final_csv \
    --confidence_threshold 0.5 \
    --device cuda --localization_only --sample_level_fp --no_confidence_filter

注：
- 需要系统可访问 prodigal 可执行文件（或允许按 run_prodigal 中策略自动安装）。
- 需要准备好与 bgc_proteins.json 对应的嵌入目录 emb_dir（本脚本不负责生成嵌入）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple


def _add_repo_root_to_syspath() -> str:
    """将仓库根目录加入 sys.path 以便导入相邻模块。"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


REPO_ROOT = _add_repo_root_to_syspath()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端运行：从FASTA到CSV")

    # 输入/输出
    parser.add_argument("--input", required=True,
                        help="输入核酸FASTA文件或目录 (*.fa|*.fna|*.fasta)")
    parser.add_argument("--emb_dir", required=True,
                        help="预先计算的嵌入目录（BalancedClusterDataset使用）")
    parser.add_argument("--checkpoints_dir", required=True,
                        help="包含 checkpoint_fold_*.pth 的目录（validate_final会在此查找模型）")
    parser.add_argument("--final_output_dir", required=True,
                        help="最终导出的CSV输出目录")

    # Prodigal 参数
    parser.add_argument("--mode", choices=["meta", "single"], default="meta",
                        help="Prodigal模式：meta(默认)/single")
    parser.add_argument("--no_rename_contigs", action="store_true",
                        help="不重命名 contig（默认会重命名为 <genome_id>_<idx>）")

    # 生成预测数据参数
    parser.add_argument("--target_length", type=int, default=128,
                        help="分段长度（CDS数量），与模型max_tokens保持一致")

    # 验证/评估参数
    parser.add_argument("--device", default="cuda",
                        help="设备：cuda/cpu")
    parser.add_argument("--no_confidence_filter", action="store_true",
                        help="禁用评估时置信度过滤（显示所有预测，IoU阈值更宽松）")
    parser.add_argument("--localization_only", action="store_true",
                        help="只计算定位，不考虑分类")
    parser.add_argument("--sample_level_fp", action="store_true",
                        help="按样本级别统计假阳性")

    # 导出参数
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="导出CSV时的最小置信度阈值 (默认0.5)")

    # 其它
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="保留中间产物目录以便排查")

    return parser.parse_args()


def run_prodigal_step(input_path: str, proteins_dir: str, mode: str, rename_contigs: bool) -> None:
    """调用 run_prodigal_for_genomes 的逻辑批量生成 protein FASTA。"""
    from run_prodigal_for_genomes import iter_input_fastas, process_one

    os.makedirs(proteins_dir, exist_ok=True)
    inputs = iter_input_fastas(input_path)
    if not inputs:
        raise FileNotFoundError("未找到输入FASTA文件")

    for fa in inputs:
        outp = process_one(fa, proteins_dir, mode, rename_contigs)
        print(f"[Prodigal] OK: {fa} -> {outp}")


def build_prediction_data(proteins_dir: str, target_length: int, out_dir: str) -> Tuple[str, str, str]:
    """解析protein FASTA并生成 bgc_mapping.json / bgc_proteins.json / cds_sequences.json。"""
    from process_prediction_data_fixed import parse_fasta_files, save_prediction_data

    os.makedirs(out_dir, exist_ok=True)
    bgc_mapping, bgc_proteins, cds_sequences = parse_fasta_files(
        proteins_dir=proteins_dir,
        target_length=target_length,
    )
    save_prediction_data(bgc_mapping, bgc_proteins, cds_sequences, out_dir)

    mapping_json = os.path.join(out_dir, "bgc_mapping.json")
    proteins_json = os.path.join(out_dir, "bgc_proteins.json")
    cds_json = os.path.join(out_dir, "cds_sequences.json")
    return mapping_json, proteins_json, cds_json


def run_validate_and_get_best_detail(mapping_json: str,
                                     emb_dir: str,
                                     checkpoints_dir: str,
                                     validation_dir: str,
                                     device: str,
                                     no_confidence_filter: bool,
                                     localization_only: bool,
                                     sample_level_fp: bool) -> Optional[str]:
    """调用 validate_final.py 进行评估并返回“最佳” detailed_predictions_*.json 路径。"""
    validate_script = os.path.join(REPO_ROOT, "validate_final.py")
    if not os.path.isfile(validate_script):
        raise FileNotFoundError(f"未找到验证脚本: {validate_script}")

    os.makedirs(validation_dir, exist_ok=True)

    cmd = [
        sys.executable, validate_script,
        "--mapping_json", os.path.abspath(mapping_json),
        "--emb_dir", os.path.abspath(emb_dir),
        "--output_dir", os.path.abspath(checkpoints_dir),
        "--validation_output_dir", os.path.abspath(validation_dir),
        "--device", device,
        "--save_detailed_predictions",
    ]
    if no_confidence_filter:
        cmd.append("--no_confidence_filter")
    if localization_only:
        cmd.append("--localization_only")
    if sample_level_fp:
        cmd.append("--sample_level_fp")

    print("[Validate] 正在运行:")
    print(" ", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("validate_final.py 运行失败")

    # 读取汇总结果，挑选最佳fold
    results_path = os.path.join(validation_dir, "final_validation_results.json")
    best_checkpoint_noext: Optional[str] = None
    if os.path.isfile(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            # 选择 f1_score 最大的fold（定位或检测）
            best_score = -1.0
            for r in all_results:
                # 优先使用 'f1_score'，若没有则回退到 'classification_f1'
                score = None
                if "f1_score" in r and isinstance(r["f1_score"], (int, float)):
                    score = float(r["f1_score"])
                elif "classification_f1" in r and isinstance(r["classification_f1"], (int, float)):
                    score = float(r["classification_f1"])  # 分类F1
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    # detailed 文件名由 checkpoint 名派生
                    ckpt_name = r.get("checkpoint", "")
                    best_checkpoint_noext = os.path.splitext(ckpt_name)[0]
            if best_checkpoint_noext:
                candidate = os.path.join(validation_dir, f"detailed_predictions_{best_checkpoint_noext}.json")
                if os.path.isfile(candidate):
                    print(f"[Validate] 选择最佳详细结果: {candidate}")
                    return candidate
        except Exception as e:
            print(f"[Validate] 读取或解析汇总结果失败，将回退到最近的详细结果: {e}")

    # 回退：选取 validation_dir 下最近的 detailed_predictions_*.json
    detailed_files = [
        os.path.join(validation_dir, f)
        for f in os.listdir(validation_dir)
        if f.startswith("detailed_predictions_") and f.endswith(".json")
    ]
    if not detailed_files:
        print("[Validate] 未找到 detailed_predictions_*.json")
        return None
    detailed_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f"[Validate] 回退选择: {detailed_files[0]}")
    return detailed_files[0]


def run_export_csv(detailed_json: str,
                   bgc_proteins_json: str,
                   confidence_threshold: float,
                   final_output_dir: str) -> None:
    export_script = os.path.join(REPO_ROOT, "scripts", "export_filtered_predictions.py")
    if not os.path.isfile(export_script):
        raise FileNotFoundError(f"未找到导出脚本: {export_script}")

    os.makedirs(final_output_dir, exist_ok=True)

    cmd = [
        sys.executable, export_script,
        "--detailed_json", os.path.abspath(detailed_json),
        "--bgc_proteins", os.path.abspath(bgc_proteins_json),
        "--output_dir", os.path.abspath(final_output_dir),
        "--confidence_threshold", str(confidence_threshold),
    ]
    print("[Export] 正在运行:")
    print(" ", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("export_filtered_predictions.py 运行失败")


def main() -> None:
    args = parse_args()

    # 绝对路径化
    input_path = os.path.abspath(args.input)
    emb_dir = os.path.abspath(args.emb_dir)
    checkpoints_dir = os.path.abspath(args.checkpoints_dir)
    final_output_dir = os.path.abspath(args.final_output_dir)

    if not (os.path.isdir(emb_dir) or os.path.isfile(emb_dir)):
        raise FileNotFoundError(f"嵌入目录不存在: {emb_dir}")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"检查点目录不存在: {checkpoints_dir}")

    # 临时目录
    tmp_base = tempfile.mkdtemp(prefix="bgcdetr_e2e_") if args.keep_intermediate else None
    try:
        if tmp_base is None:
            with tempfile.TemporaryDirectory(prefix="bgcdetr_e2e_") as td:
                proteins_dir = os.path.join(td, "proteins")
                preddata_dir = os.path.join(td, "prediction_data")
                validation_dir = os.path.join(td, "validation")

                # Step 1: Prodigal
                print("\n==== Step 1/4: 运行 Prodigal 生成 protein FASTA ====")
                run_prodigal_step(
                    input_path=input_path,
                    proteins_dir=proteins_dir,
                    mode=args.mode,
                    rename_contigs=not args.no_rename_contigs,
                )

                # Step 2: 生成预测数据JSON
                print("\n==== Step 2/4: 生成 BGC-DETR 输入数据(JSON) ====")
                mapping_json, bgc_proteins_json, _ = build_prediction_data(
                    proteins_dir=proteins_dir,
                    target_length=args.target_length,
                    out_dir=preddata_dir,
                )

                # Step 3: 运行验证，获得详细预测
                print("\n==== Step 3/4: 运行验证并选择最佳详细预测 ====")
                detailed_json = run_validate_and_get_best_detail(
                    mapping_json=mapping_json,
                    emb_dir=emb_dir,
                    checkpoints_dir=checkpoints_dir,
                    validation_dir=validation_dir,
                    device=args.device,
                    no_confidence_filter=args.no_confidence_filter,
                    localization_only=args.localization_only,
                    sample_level_fp=args.sample_level_fp,
                )
                if not detailed_json:
                    raise RuntimeError("未能生成详细预测结果 detailed_predictions_*.json")

                # Step 4: 导出CSV
                print("\n==== Step 4/4: 导出CSV ====")
                run_export_csv(
                    detailed_json=detailed_json,
                    bgc_proteins_json=bgc_proteins_json,
                    confidence_threshold=args.confidence_threshold,
                    final_output_dir=final_output_dir,
                )
                print(f"\n完成：CSV已写入 {final_output_dir}")
        else:
            # 保留中间目录
            proteins_dir = os.path.join(tmp_base, "proteins")
            preddata_dir = os.path.join(tmp_base, "prediction_data")
            validation_dir = os.path.join(tmp_base, "validation")
            os.makedirs(proteins_dir, exist_ok=True)
            os.makedirs(preddata_dir, exist_ok=True)
            os.makedirs(validation_dir, exist_ok=True)

            print("\n==== Step 1/4: 运行 Prodigal 生成 protein FASTA ====")
            run_prodigal_step(
                input_path=input_path,
                proteins_dir=proteins_dir,
                mode=args.mode,
                rename_contigs=not args.no_rename_contigs,
            )

            print("\n==== Step 2/4: 生成 BGC-DETR 输入数据(JSON) ====")
            mapping_json, bgc_proteins_json, _ = build_prediction_data(
                proteins_dir=proteins_dir,
                target_length=args.target_length,
                out_dir=preddata_dir,
            )

            print("\n==== Step 3/4: 运行验证并选择最佳详细预测 ====")
            detailed_json = run_validate_and_get_best_detail(
                mapping_json=mapping_json,
                emb_dir=emb_dir,
                checkpoints_dir=checkpoints_dir,
                validation_dir=validation_dir,
                device=args.device,
                no_confidence_filter=args.no_confidence_filter,
                localization_only=args.localization_only,
                sample_level_fp=args.sample_level_fp,
            )
            if not detailed_json:
                raise RuntimeError("未能生成详细预测结果 detailed_predictions_*.json")

            print("\n==== Step 4/4: 导出CSV ====")
            run_export_csv(
                detailed_json=detailed_json,
                bgc_proteins_json=bgc_proteins_json,
                confidence_threshold=args.confidence_threshold,
                final_output_dir=final_output_dir,
            )

            print("\n完成：CSV已写入 {}".format(final_output_dir))
            print("保留中间目录: {}".format(tmp_base))
    finally:
        # 如未要求保留，Nothing to do（TemporaryDirectory 已清理）；若要求保留，则不清理。
        pass


if __name__ == "__main__":
    main()


