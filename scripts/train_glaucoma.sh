#!/usr/bin/env bash
# Training wrapper for glaucoma experiments with pre-split data.
# This mirrors `scripts/train_dr.sh`, but targets the glaucoma dataset.
#
# Example:
#   scripts/train_glaucoma.sh --gpus 0,1 --epochs 50 --algorithm CaSN

set -euo pipefail

# Ensure local package is discoverable without installation
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Backbones to evaluate (same list pattern as `train_dr.sh`)
MODEL_ARCHES=(
  # resnet18
  # resnet50
  # resnet101
  vgg16_bn
  # efficientnet_b0
  # efficientnet_b3
  # densenet121
  # vit_b_16
  # mobilenet_v2
  # mobilenet_v3_large
  # shufflenet_v2_x1_0
  squeezenet1_1
  # mobileformer_294m
)

GPUS="1"
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

# Default arguments for glaucoma (pre-split hospital + smartphone dataset)
DEFAULT_ARGS=(
  --data_root ./data/glaucoma
  --train_split train
  --val_split val
  --test_split test
  --class_map "normal=0,glaucoma=1"
  --algorithm ERM           # override with --algorithm CaSN / CaSN_MMD / CaSN_IRM if desired
  --test_domain smartphone  # keep smartphone as the designated test env
  # --hospital_only
  # --include_test_domain_in_train  # include smartphone-train in training (train on hospital + smartphone)
  --data_augmentation severe
  --pretrained
  --lr 5e-5
  --batch_size 32
  --epochs 20
  --bias 3.0
  --int_lambda 1.0
  --kl_lambda 0.01
  --target_lambda 0.1
  --int_reg 0.1
  --output_dir ./results/glaucoma_erm_severe
  --gpus 2
  --seed 0
  --no-freeze_bn
)

timestamp="$(date +%Y%m%d_%H%M%S)"

for arch in "${MODEL_ARCHES[@]}"; do
  run_name="glaucoma_erm_${arch}_${timestamp}"
  echo ">>> Starting glaucoma ERM training for model_arch=${arch}, run=${run_name}"

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

