#!/usr/bin/env bash
set -euo pipefail

# Simple pipeline to evaluate a model against GPT-4 labels on alpaca_golden.
# 1) Run decode.py in alpaca_golden mode to produce paired responses (model vs GPT-4 output).
# 2) Run annotate.py to score and report accuracy (model win rate vs GPT-4 labels).

# Usage: adjust the variables below as needed, then run: bash scripts/eval/pesudo_alpaca_eval.sh

GPU_IDS="0,1"
MODEL="/data2/jty/CoDPO/checkpoints/Qwen2.5-3B-Instruct-DPO"
# MODEL="/data2/jty/models/Qwen2.5-3B-Instruct"
OUTPUT_DIR="results/Qwen2.5-3B-Instruct-DPO"
SEED=42
REWARD_MODEL="/data2/jty/models/ArmoRM"
BATCH_SIZE=8

# Parse CLI overrides: allow KEY=VALUE or --key value (keys mapped to UPPERCASE, dashes -> underscores)
print_usage() {
	cat <<'USAGE'
Usage: ./scripts/eval/pesudo_alpaca_eval.sh [KEY=VALUE ...] [--key value ...]

Examples:
  GPU_IDS="0" ./scripts/eval/pesudo_alpaca_eval.sh
  ./scripts/eval/pesudo_alpaca_eval.sh GPU_IDS=0,1 OUTPUT_DIR=my_results SEED=999
  ./scripts/eval/pesudo_alpaca_eval.sh --gpu-ids 0,1 --output-dir my_results --seed 999
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	print_usage
	exit 0
fi

while [[ $# -gt 0 ]]; do
	arg="$1"
	if [[ "$arg" == *=* ]]; then
		eval "$arg"
		shift
		continue
	fi

	if [[ "$arg" == --* ]]; then
		key="${arg#--}"
		key_norm="$(echo "$key" | tr '[:lower:]-' '[:upper:]_')"
		shift
		if [[ $# -eq 0 ]]; then
			echo "Missing value for --$key"
			exit 1
		fi
		val="$1"
		eval "$key_norm=\"$val\""
		shift
		continue
	fi

	echo "Unknown argument: $arg"
	print_usage
	exit 1
done

mkdir -p "${OUTPUT_DIR}"

echo "[1/2] Decoding with vLLM (alpaca_golden)..."
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/on_policy_data_gen/decode.py \
	--alpaca_golden \
	--model "${MODEL}" \
	--output_dir "${OUTPUT_DIR}" \
	--seed "${SEED}" \
	--max_tokens 2048 \
	--temperature 0.8 \
	--top_p 0.95

GEN_FILE="${OUTPUT_DIR}/output_alpaca_${SEED}.json"

echo "[2/2] Annotating and computing accuracy (model win rate vs GPT-4 labels)..."
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/on_policy_data_gen/annotate.py \
	--generation_file "${GEN_FILE}" \
	--reward_model "${REWARD_MODEL}" \
	--reward_mode golden \
	--batch_size "${BATCH_SIZE}" \
	--output_dir "${OUTPUT_DIR}"

RM_FILE="${OUTPUT_DIR}/output_alpaca_${SEED}_rm.json"
SUMMARY_FILE="${OUTPUT_DIR}/output_alpaca_${SEED}_rm_summary.json"

echo "[post] Averaging RM scores (model vs GPT-4)..."
python scripts/eval/average_rm_scores.py --input "${RM_FILE}" --output "${SUMMARY_FILE}"

echo "Done. Results in ${OUTPUT_DIR}" 
