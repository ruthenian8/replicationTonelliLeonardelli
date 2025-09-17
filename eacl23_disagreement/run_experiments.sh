#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHAMP_DIR="/content/machamp"
PARAMS="/content/replicationTonelliLeonardelli/machamp/configs/params.json"
BASE="/content/data/md_agreement"
SPLITS="/content/data/splits"
OUT="$ROOT/results"
CONFIG_DIR="$ROOT/configs"

mkdir -p "$OUT" "$CONFIG_DIR"

# 0) Generate configs per split/mode
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App"         --mode single --out "$CONFIG_DIR/app.single.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0all"   --mode single --out "$CONFIG_DIR/app_a0all.single.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0_SUBJ" --mode single --out "$CONFIG_DIR/app_a0subj.single.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0_MISS" --mode single --out "$CONFIG_DIR/app_a0miss.single.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0_AMB"  --mode single --out "$CONFIG_DIR/app_a0amb.single.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0all"   --mode mtl3   --out "$CONFIG_DIR/app_a0all.mtl3.json"
python3 "$ROOT/gen_config.py" --split_dir "$SPLITS/App_A0all"   --mode mtl6   --out "$CONFIG_DIR/app_a0all.mtl6.json"

cd "$MACHAMP_DIR"
shopt -s globstar
SEEDS=${SEEDS:-"$(seq 1 20)"}

run_train() {
  local config_path="$1"
  local run_prefix="$2"
  for seed in $SEEDS; do
    echo "[TRAIN] ${run_prefix} seed=${seed}"
    python3 train.py \
      --dataset_configs "$config_path" \
      --parameters_config "$PARAMS" \
      --device 0 \
      --name "${run_prefix}_s${seed}"
  done
}

run_train "$CONFIG_DIR/app.single.json"        "EACL23_App"
run_train "$CONFIG_DIR/app_a0all.single.json"  "EACL23_AppA0all"
run_train "$CONFIG_DIR/app_a0subj.single.json" "EACL23_AppA0subj"
run_train "$CONFIG_DIR/app_a0miss.single.json" "EACL23_AppA0miss"
run_train "$CONFIG_DIR/app_a0amb.single.json"  "EACL23_AppA0amb"
run_train "$CONFIG_DIR/app_a0all.mtl3.json"    "EACL23_MTL3"
run_train "$CONFIG_DIR/app_a0all.mtl6.json"    "EACL23_MTL6"

predict_runs() {
  local prefix="$1"
  for seed in $SEEDS; do
    local run_dir
    run_dir=$(ls -d logs/**/"${prefix}_s${seed}" 2>/dev/null | tail -n 1 || true)
    if [[ -z "$run_dir" ]]; then
      echo "Missing run dir for ${prefix} seed ${seed}" >&2
      continue
    fi
    echo "[PRED ] ${run_dir}"
    python3 predict.py \
      "$run_dir/model.pt" \
      "$BASE/test.tsv" \
      "$OUT/${prefix}_s${seed}.pred.txt" \
      --dataset MD \
      --device 0
  done
}

predict_runs "EACL23_App"
predict_runs "EACL23_AppA0all"
predict_runs "EACL23_AppA0subj"
predict_runs "EACL23_AppA0miss"
predict_runs "EACL23_AppA0amb"
predict_runs "EACL23_MTL3"
predict_runs "EACL23_MTL6"

cd "$ROOT"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_App_s*.pred.txt" \
  --group_by subtype \
  --out_json "$OUT/table2_by_subtype.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_App_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table2_by_category.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_App_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table3_App.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_AppA0all_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table3_App_A0all.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_AppA0subj_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table3_App_A0_SUBJ.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_AppA0miss_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table3_App_A0_MISS.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_AppA0amb_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table3_App_A0_AMB.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_MTL3_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table4_MTL3.json"

python3 "$ROOT/score_groups.py" \
  --test_tsv "$BASE/test.tsv" \
  --pred_glob "$OUT/EACL23_MTL6_s*.pred.txt" \
  --group_by category \
  --out_json "$OUT/table4_MTL6.json"
