#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bgcdetr

REPO=/share/org/BGI/bgi_suncl/project/BGC-DETR
BASE=/share/org/BGI/bgi_suncl/project/GeneOutputs/BGC_DETR_sags_drep_20260318_163042

rm -rf "$BASE/validation" "$BASE/filtered"
mkdir -p "$BASE/validation" "$BASE/filtered"

python "$REPO/process_prediction_data_fixed.py" \
  --proteins_dir "$BASE/proteins" \
  --output_dir "$BASE/prediction_data" \
  --target_length 128

python "$REPO/validate_final.py" \
  --mapping_json "$BASE/prediction_data/bgc_mapping.json" \
  --emb_dir "$BASE/embeddings" \
  --output_dir "$REPO/outputs_1280_gpu0" \
  --validation_output_dir "$BASE/validation" \
  --device cuda \
  --save_detailed_predictions

BEST_JSON=$(python3 - <<'PY'
import json
import os

base = "/share/org/BGI/bgi_suncl/project/GeneOutputs/BGC_DETR_sags_drep_20260318_163042/validation"
results = os.path.join(base, "final_validation_results.json")

best = None
best_score = float("-inf")

with open(results, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    score = item.get("f1_score")
    if score is None:
        score = item.get("classification_f1")
    if isinstance(score, (int, float)) and score > best_score:
        ckpt = os.path.splitext(item["checkpoint"])[0]
        cand = os.path.join(base, f"detailed_predictions_{ckpt}.json")
        if os.path.exists(cand):
            best = cand
            best_score = score

if not best:
    raise SystemExit("未找到 detailed_predictions 文件")

print(best)
PY
)

python "$REPO/scripts/export_filtered_predictions.py" \
  --detailed_json "$BEST_JSON" \
  --bgc_proteins "$BASE/prediction_data/bgc_proteins.json" \
  --output_dir "$BASE/filtered" \
  --confidence_threshold 0.5
