#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1"

# default parameters
MAX_ROUNDS=3
SEED=42
LR=5e-7
INIT_EPOCHS=3
TRAIN_PER_DEVICE_BS=8
EVAL_PER_DEVICE_BS=8
GRAD_ACC=8
EVAL_STEPS=10
SAVE_STRATEGY="epoch"
BETA=0.1

DATA_ROOT="datasets/ultrafeedback_binarized_annotated"
SPLIT_PREFIX="${DATA_ROOT}/split_"
BASE_MODEL="/data1/Common_LLM_Base/Qwen/Qwen2.5-3B-Instruct/"
REWARD_MODEL="/data2/jty/models/ArmoRM"
OUTPUT_ROOT="self_train_runs"

train_dpo () {
    local DATASET_PATH="$1"
    local MODEL_PATH="$2"
    local OUTPUT_DIR="$3"
    local EPOCHS="$4"

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch --config_file trl/accelerate_configs/zero3.yaml scripts/run_dpo.py \
        --dataset_name "${DATASET_PATH}" \
        --model_name_or_path "${MODEL_PATH}" \
        --learning_rate "${LR}" \
        --beta "${BETA}" \
        --num_train_epochs "${EPOCHS}" \
        --per_device_train_batch_size "${TRAIN_PER_DEVICE_BS}" \
        --per_device_eval_batch_size "${EVAL_PER_DEVICE_BS}" \
        --gradient_accumulation_steps "${GRAD_ACC}" \
        --gradient_checkpointing \
        --eval_strategy steps \
        --eval_steps "${EVAL_STEPS}" \
        --save_strategy "${SAVE_STRATEGY}" \
        --output_dir "${OUTPUT_DIR}" \
        --no_remove_unused_columns
}

# Parse CLI overrides: allow KEY=VALUE or --key value (keys mapped to UPPERCASE, dashes -> underscores)
print_usage() {
    cat <<'USAGE'
Usage: ./self_train.sh [KEY=VALUE ...] [--key value ...]

Examples:
  GPU_IDS="0" ./self_train.sh
  ./self_train.sh GPU_IDS=0,1 MAX_ROUNDS=5 SEED=123
  ./self_train.sh --gpu-ids 0,1 --max-rounds 5
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

# read arguments from command line
while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ "$arg" == *=* ]]; then
        # KEY=VALUE form, set as-is
        eval "$arg"
        shift
        continue
    fi

    if [[ "$arg" == --* ]]; then
        key="${arg#--}"
        # normalize: replace - with _ and uppercase
        key_norm="$(echo "$key" | tr '[:lower:]-' '[:upper:]_')"
        shift
        if [[ $# -eq 0 ]]; then
            echo "Missing value for --$key"
            exit 1
        fi
        val="$1"
        # export so functions/accelerate inherit if needed
        eval "$key_norm=\"$val\""
        shift
        continue
    fi

    echo "Unknown argument: $arg"
    print_usage
    exit 1
done

# Pipeline start
mkdir -p "${OUTPUT_ROOT}"

# Round 0: 在 split_0 上初始 DPO 训练
ROUND0_OUT="${OUTPUT_ROOT}/round0"
train_dpo "${SPLIT_PREFIX}0" "${BASE_MODEL}" "${ROUND0_OUT}" "${INIT_EPOCHS}"
CURRENT_MODEL="${ROUND0_OUT}"
REFERENCE_MODEL="${BASE_MODEL}"

# 后续轮次（最多 3 轮）：生成->自标注->再训练
for ROUND in 1 2 3; do
    [ "${ROUND}" -gt "${MAX_ROUNDS}" ] && break
    SPLIT_DIR="${SPLIT_PREFIX}${ROUND}"
    [ ! -d "${SPLIT_DIR}" ] && { echo "Skip round ${ROUND}: missing ${SPLIT_DIR}"; continue; }

    WORK_DIR="${OUTPUT_ROOT}/round${ROUND}"
    mkdir -p "${WORK_DIR}"

    echo "[${ROUND}] Decode prompts from ${SPLIT_DIR} ..."
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/on_policy_data_gen/decode.py \
        --data_dir "${SPLIT_DIR}" \
        --data_split train \
        --model "${CURRENT_MODEL}" \
        --output_dir "${WORK_DIR}/gen" \
        --seed "${SEED}" \
        --num_samples 2 \
        --max_tokens 1024 \
        --temperature 0.8 \
        --top_p 0.95

    GEN_FILE="${WORK_DIR}/gen/output_${SEED}.json"

    echo "[${ROUND}] Annotate (self-label) ..."
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/on_policy_data_gen/annotate.py \
        --generation_file "${GEN_FILE}" \
        --reward_model "${CURRENT_MODEL}" \
        --ref_model "${BASE_MODEL}" \
        --reward_mode dpo \
        --beta "${BETA}" \
        --batch_size "${EVAL_PER_DEVICE_BS}" \
        --output_dir "${WORK_DIR}/anno" \
        --save_dataset

    ANNO_FILE="${WORK_DIR}/anno/"
    
    echo "[${ROUND}] DPO train on self-labeled data ..."
    ROUND_OUT="${WORK_DIR}/ckpt"
    train_dpo "${ANNO_FILE}" "${CURRENT_MODEL}" "${ROUND_OUT}" "1"
    REFERENCE_MODEL="${CURRENT_MODEL}"
    CURRENT_MODEL="${ROUND_OUT}"
done

echo "Pipeline finished. Latest model: ${CURRENT_MODEL}"
