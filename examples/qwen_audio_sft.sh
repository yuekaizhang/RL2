#!/usr/bin/env bash
# ==============================================================================
# Qwen2-Audio SFT using RL2 Framework with Megatron-Bridge
#
# This script orchestrates the full audio SFT pipeline:
#   1. Convert HuggingFace checkpoint to Megatron format (if needed)
#   2. Run distributed SFT training via RL2
#   3. Export trained Megatron checkpoint back to HuggingFace format
#
# Usage:
#   bash examples/qwen_audio_sft.sh
#
# Environment variables:
#   HF_MODEL     - HuggingFace model path (default: /workspace_yuekai/HF/Qwen2-Audio-7B)
#   NPROC        - Number of GPUs (default: 4)
#   DATASET_NAME - HuggingFace dataset name (default: yuekai/aishell)
#   N_EPOCHS     - Number of training epochs (default: 3)
# ==============================================================================
set -euo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Logging: tee all output (stdout+stderr) to log file ---
LOG_FILE=${LOG_FILE:-${RL2_ROOT}/log.txt}
exec > >(tee "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"
MEGATRON_BRIDGE_ROOT="/workspace_yuekai/asr/Megatron-Bridge"

export PYTHONPATH=${RL2_ROOT}:${MEGATRON_BRIDGE_ROOT}:${PYTHONPATH:-}
python_path=${MEGATRON_BRIDGE_ROOT}/.venv/bin/python

# Verify python exists
if [ ! -f "${python_path}" ]; then
    echo "Error: Python not found at ${python_path}"
    echo "Please ensure Megatron-Bridge venv is set up."
    exit 1
fi

# --- Model Configuration ---
HF_MODEL=${HF_MODEL:-/workspace_yuekai/HF/Qwen2-Audio-7B}
MODEL_NAME=${MODEL_NAME:-qwen2_audio_7b}
MEGATRON_CKPT_DIR=${MEGATRON_CKPT_DIR:-${RL2_ROOT}/megatron_ckpts/${MODEL_NAME}}

# --- Training Configuration ---
NPROC=${NPROC:-4}
N_EPOCHS=${N_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_AUDIO_SECONDS=${MAX_AUDIO_SECONDS:-30}
LR=${LR:-2e-5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_NAME}_sft}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-rl2-qwen2-audio-sft}

# --- Dataset Configuration ---
DATASET_NAME=${DATASET_NAME:-yuekai/aishell}
DATASET_SUBSET=${DATASET_SUBSET:-train}
DATASET_SPLIT=${DATASET_SPLIT:-test}
VAL_SUBSET=${VAL_SUBSET:-dev}
VAL_SPLIT=${VAL_SPLIT:-test}
PROMPT=${PROMPT:-"Detect the language and recognize the speech: <|zh|>"}

# --- Freeze Configuration ---
FREEZE_AUDIO=${FREEZE_AUDIO:-false}
FREEZE_LM=${FREEZE_LM:-false}
FREEZE_PROJECTOR=${FREEZE_PROJECTOR:-false}

# ==============================================================================
# Step 1: Convert HF checkpoint to Megatron format (if not already done)
# ==============================================================================
if [ ! -d "${MEGATRON_CKPT_DIR}/iter_0000000" ]; then
    echo "============================================================"
    echo "  Converting HF model to Megatron format..."
    echo "  HF Model: ${HF_MODEL}"
    echo "  Megatron Path: ${MEGATRON_CKPT_DIR}"
    echo "============================================================"
    ${python_path} ${MEGATRON_BRIDGE_ROOT}/examples/conversion/convert_checkpoints.py import \
        --hf-model ${HF_MODEL} \
        --megatron-path ${MEGATRON_CKPT_DIR}
    echo "Conversion complete."
else
    echo "Megatron checkpoint already exists at ${MEGATRON_CKPT_DIR}, skipping conversion."
fi

# ==============================================================================
# Step 2: Run SFT Training
# ==============================================================================
echo "============================================================"
echo "  Starting Audio SFT Training"
echo "  Model: ${HF_MODEL}"
echo "  Dataset: ${DATASET_NAME}"
echo "  GPUs: ${NPROC}"
echo "  Epochs: ${N_EPOCHS}"
echo "============================================================"

${python_path} -m torch.distributed.run \
    --nproc_per_node=${NPROC} \
    -m RL2.trainer.audio_sft \
    data.train.dataset_name=${DATASET_NAME} \
    data.train.dataset_subset=${DATASET_SUBSET} \
    data.train.dataset_split=${DATASET_SPLIT} \
    data.train.batch_size=${BATCH_SIZE} \
    data.train.max_length=${MAX_LENGTH} \
    data.train.max_audio_seconds=${MAX_AUDIO_SECONDS} \
    "data.train.prompt='${PROMPT}'" \
    data.test.dataset_subset=${VAL_SUBSET} \
    data.test.dataset_split=${VAL_SPLIT} \
    actor.model_name=${HF_MODEL} \
    actor.max_length_per_device=${MAX_LENGTH} \
    actor.freeze_audio_encoder=${FREEZE_AUDIO} \
    actor.freeze_language_model=${FREEZE_LM} \
    actor.freeze_multi_modal_projector=${FREEZE_PROJECTOR} \
    actor.optimizer.lr=${LR} \
    trainer.project=${WANDB_PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_epochs=${N_EPOCHS}

echo "Training complete."

# ==============================================================================
# Step 3: Export Megatron checkpoint to HuggingFace format
# ==============================================================================
# SAVE_DIR="ckpts/${EXPERIMENT_NAME}"
# HF_EXPORT_DIR="${SAVE_DIR}/hf_export"

# if [ -d "${SAVE_DIR}/actor" ]; then
#     echo "============================================================"
#     echo "  Exporting Megatron checkpoint to HuggingFace format..."
#     echo "  Source: ${SAVE_DIR}/actor"
#     echo "  Destination: ${HF_EXPORT_DIR}"
#     echo "============================================================"
#     ${python_path} ${MEGATRON_BRIDGE_ROOT}/examples/models/audio_lm/qwen2_audio/export_hf.py \
#         --megatron-path ${SAVE_DIR}/actor \
#         --hf-path ${HF_EXPORT_DIR} \
#         --hf-model-path ${HF_MODEL}
#     echo "Export complete. HF model saved to: ${HF_EXPORT_DIR}"
# else
#     echo "Warning: No saved model found at ${SAVE_DIR}/actor, skipping HF export."
#     echo "This may happen if save_freq was not set during training."
# fi

echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
