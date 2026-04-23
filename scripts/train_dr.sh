#!/usr/bin/env bash
# Training wrapper for diabetic retinopathy experiments with pre-split data.
# This script runs CausalFund (CaSN) with a set of baseline backbones under
# Setting 3: hospital train/val/test, smartphone val/test (no smartphone train).
# Default hyperparameters are taken from a well-performing ResNet-50 CaSN run:
#   bias=4.0, int_lambda=0.5, target_lambda=0.2, kl_lambda=0.01, int_reg=0.2, lr=1e-5.
#
# Example:
#   scripts/train_dr.sh --gpus 0,1 --epochs 20 --algorithm CaSN

set -euo pipefail

# Ensure local package is discoverable without installation
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Backbones to evaluate (same list as in train_dr_all_models.sh)
MODEL_ARCHES=(
  # resnet18
  # resnet50
  # resnet101
  # vgg16_bn
  # efficientnet_b0
  # efficientnet_b3
  # densenet121
  # vit_b_16
  # mobilenet_v2
  # mobilenet_v3_large
  # shufflenet_v2_x1_0
  # squeezenet1_1
  mobileformer_294m
)

GPUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default arguments for CaSN on DR (Setting 3, smartphone as test domain, no smartphone train)
DEFAULT_ARGS=(
  --data_root ./data/dr_combined
  --train_split train
  --val_split val
  --test_split test
  --class_map "0=0,1=0,2=0,3=1,4=1"
  --algorithm CaSN          # CausalFund (CaSN); override with --algorithm ERM / CaSN_MMD / CaSN_IRM if desired
  --test_domain smartphone  # hospital is training env, smartphone is held-out test env by default
  --data_augmentation none  # use original smartphone images for val/test
  --pretrained
  --lr 1e-5
  --batch_size 32
  --epochs 20
  --bias 4.0
  --int_lambda 0.5
  --target_lambda 0.2
  --kl_lambda 0.01
  --int_reg 0.2
  --output_dir ./results/dr_casn_setting4
  --gpus 0
  --seed 0
)

timestamp="$(date +%Y%m%d_%H%M%S)"

for arch in "${MODEL_ARCHES[@]}"; do
  run_name="dr_casn_${arch}_${timestamp}"
  echo ">>> Starting CausalFund DR training for model_arch=${arch}, run=${run_name}"

  CMD=(
    python scripts/train_with_casn.py
    "${DEFAULT_ARGS[@]}"
    --model_arch "${arch}"
    --run_name "${run_name}"
  )

  # Allow overriding GPUs from the command line
  if [[ -n "${GPUS}" ]]; then
    CMD+=(--gpus "${GPUS}")
  fi

  # Allow overriding any default argument from the command line
  CMD+=("${EXTRA_ARGS[@]}")

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
done
