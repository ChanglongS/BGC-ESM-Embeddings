#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

INPUT_DIR="${PROJECT_ROOT}/sags_drep"
CHECKPOINT_DIR="${REPO_ROOT}/outputs_1280_gpu0"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
GPU_IDS="${GPU_IDS:-0}"
PRODIGAL_MODE="${PRODIGAL_MODE:-meta}"

if [[ $# -ge 1 ]]; then
  OUTPUT_BASE="$1"
else
  OUTPUT_BASE="${PROJECT_ROOT}/GeneOutputs/BGC_DETR_sags_drep_$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "输入目录不存在: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "checkpoint 目录不存在: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${OUTPUT_BASE}/proteins" "${OUTPUT_BASE}/prediction_data" \
         "${OUTPUT_BASE}/embeddings" "${OUTPUT_BASE}/validation" \
         "${OUTPUT_BASE}/filtered"

LATEST_RUN_FILE="${PROJECT_ROOT}/GeneOutputs/BGC_DETR_sags_drep_latest.txt"
printf '%s\n' "${OUTPUT_BASE}" > "${LATEST_RUN_FILE}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "找不到 conda 初始化脚本: ${HOME}/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

conda activate bgcdetr

echo "========== BGC-DETR sags_drep 任务开始 =========="
echo "时间: $(date '+%F %T')"
echo "输入目录: ${INPUT_DIR}"
echo "输出目录: ${OUTPUT_BASE}"
echo "checkpoint 目录: ${CHECKPOINT_DIR}"
echo "GPU IDs: ${GPU_IDS}"

echo
echo "[1/5] Prodigal 预测蛋白序列"
python "${REPO_ROOT}/run_prodigal_for_genomes.py" \
  --input "${INPUT_DIR}" \
  --output_dir "${OUTPUT_BASE}/proteins" \
  --mode "${PRODIGAL_MODE}"

echo
echo "[2/5] 生成 128 CDS 预测片段"
python "${REPO_ROOT}/process_prediction_data_fixed.py" \
  --proteins_dir "${OUTPUT_BASE}/proteins" \
  --output_dir "${OUTPUT_BASE}/prediction_data" \
  --target_length 128

echo
echo "[3/5] 生成 ESM 嵌入"
python "${REPO_ROOT}/scripts/example_usage.py" \
  --bgc_proteins "${OUTPUT_BASE}/prediction_data/bgc_proteins.json" \
  --cds_sequences "${OUTPUT_BASE}/prediction_data/cds_sequences.json" \
  --output_dir "${OUTPUT_BASE}/embeddings" \
  --device cuda \
  --gpu_ids "${GPU_IDS}"

echo
echo "[4/5] 运行 BGC-DETR 预测"
python "${REPO_ROOT}/validate_final.py" \
  --mapping_json "${OUTPUT_BASE}/prediction_data/bgc_mapping.json" \
  --emb_dir "${OUTPUT_BASE}/embeddings" \
  --output_dir "${CHECKPOINT_DIR}" \
  --validation_output_dir "${OUTPUT_BASE}/validation" \
  --device cuda \
  --save_detailed_predictions

echo
echo "[5/5] 导出过滤后的 CSV"
DETAILED_JSON="$(
python - "${OUTPUT_BASE}/validation" <<'PY'
import json
import os
import sys

validation_dir = sys.argv[1]
results_path = os.path.join(validation_dir, "final_validation_results.json")

chosen = None
best_score = float("-inf")

if os.path.isfile(results_path):
    with open(results_path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    for item in results:
        score = item.get("f1_score")
        if score is None:
            score = item.get("classification_f1")
        if isinstance(score, (int, float)) and score > best_score:
            ckpt_name = item.get("checkpoint", "")
            candidate = os.path.join(
                validation_dir,
                f"detailed_predictions_{os.path.splitext(ckpt_name)[0]}.json",
            )
            if os.path.isfile(candidate):
                best_score = float(score)
                chosen = candidate

if chosen is None:
    detailed_files = [
        os.path.join(validation_dir, name)
        for name in os.listdir(validation_dir)
        if name.startswith("detailed_predictions_") and name.endswith(".json")
    ]
    if detailed_files:
        detailed_files.sort(key=os.path.getmtime, reverse=True)
        chosen = detailed_files[0]

if chosen is None:
    raise SystemExit("未找到 detailed_predictions_*.json")

print(chosen)
PY
)"

python "${REPO_ROOT}/scripts/export_filtered_predictions.py" \
  --detailed_json "${DETAILED_JSON}" \
  --bgc_proteins "${OUTPUT_BASE}/prediction_data/bgc_proteins.json" \
  --output_dir "${OUTPUT_BASE}/filtered" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD}"

echo
echo "任务完成: $(date '+%F %T')"
echo "输出目录: ${OUTPUT_BASE}"
echo "详细预测文件: ${DETAILED_JSON}"
echo "结果 CSV 目录: ${OUTPUT_BASE}/filtered"
