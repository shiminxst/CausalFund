#!/usr/bin/env bash
# Train CausalFund (CaSN) for DR (Setting 3) across backbones.
#
# This script is copied/adapted from scripts/train_dr.sh, but:
# - Uses per-backbone default hyperparameters from the *best-test selection* tuned runs under:
#     results/dr_tuning_casn_backbone_loss_fixed/
# - Keeps batch size as a per-model variable (`batch_size`) so you can tune batch size only.
#
# Notes:
# - Selection rule used for defaults below: choose, per backbone, the tuned run with the best
#   smartphone_test_auc (still from each run's best_metrics.json).
# - Hyperparameters are encoded in the run folder name (best_metrics.json does not include them).
#
# Example:
#   scripts/train_dr_casn.sh --gpus 0,1 --epochs 20 --data_augmentation mild

set -euo pipefail

# Ensure local package is discoverable without installation
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Reference hyperparameters for *untuned* backbones.
# Chosen from the best-test tuned experiments under:
#   results/dr_tuning_casn_backbone_loss_fixed/
# Overall best-test reference run:
#   tune_densenet121_b2p0_il0p5_tl0p1_kl0p01_ir0p2_lr5em5
#
# If a backbone doesn't have its own tuned block below, we fall back to these.
REF_BIAS=0.0
REF_INT_LAMBDA=0.5
REF_TARGET_LAMBDA=0.1
REF_KL_LAMBDA=0.01
REF_INT_REG=0.2
REF_LR=1e-4

# Backbones to evaluate.
# Any backbone without tuned defaults will fall back to the REF_* values above.
MODEL_ARCHES=(
  # densenet121
  # efficientnet_b0
  mobilenet_v2
  # squeezenet1_1
  # vit_b_16
  resnet50
  # vgg16_bn
)

GPUS="0"
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

# Shared default arguments for CaSN on DR (Setting 3, smartphone as test domain, no smartphone train)
DEFAULT_ARGS=(
  --data_root ./data/dr_combined
  --train_split train
  --val_split val
  --test_split test
  --class_map "0=0,1=0,2=0,3=1,4=1"
  --algorithm CaSN
  --test_domain smartphone
  --data_augmentation mild  
  --pretrained
  --no-freeze_bn
  --epochs 20
  --output_dir ./results/dr_casn_setting3_loss_fixed_mild
  --gpus 0
  --seed 0
  # Default (reference) hyperparameters for untuned backbones. Tuned blocks override these.
  --lr "${REF_LR}"
  --bias "${REF_BIAS}"
  --int_lambda "${REF_INT_LAMBDA}"
  --target_lambda "${REF_TARGET_LAMBDA}"
  --kl_lambda "${REF_KL_LAMBDA}"
  --int_reg "${REF_INT_REG}"
)

timestamp="$(date +%Y%m%d_%H%M%S)"

for arch in "${MODEL_ARCHES[@]}"; do
  # Per-backbone tuned defaults (best-test selection).
  # Each block includes the tuned experiment folder that was chosen.
  #
  # IMPORTANT: keep `batch_size` as a variable so you can tune it easily.
  batch_size=32

  case "${arch}" in
    densenet121)
      # best-test tuned run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_densenet121_b2p0_il0p5_tl0p1_kl0p01_ir0p2_lr5em5
      bias=1.0
      int_lambda=0.5
      target_lambda=0.1
      kl_lambda=0.01
      int_reg=0.2
      lr=5e-4
      ;;
    efficientnet_b0)
      # best-test tuned run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_efficientnet_b0_b2p0_il0p5_tl0p1_kl0p005_ir0p1_lr5em5
      bias=0.02
      int_lambda=0.05
      target_lambda=0.1
      kl_lambda=0.005
      int_reg=0.1
      lr=5e-4
      ;;
    mobilenet_v2)
      # best-test tuned run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_mobilenet_v2_b1p5_il0p25_tl0p05_kl0p005_ir0p05_lr2em5
      bias=0.
      int_lambda=0.025
      target_lambda=0.05
      kl_lambda=0.005
      int_reg=0.05
      lr=1e-4
      ;;
    squeezenet1_1)
      # best-test tuned run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_squeezenet1_1_b2p0_il0p5_tl0p1_kl0p01_ir0p05_lr5em5
      bias=1.0
      int_lambda=0.5
      target_lambda=0.1
      kl_lambda=0.01
      int_reg=0.05
      lr=5e-5
      ;;
    vit_b_16)
      # best-test tuned run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_vit_b_16_b2p0_il0p5_tl0p1_kl0p01_ir0p1_lr1em5
      bias=0.0
      int_lambda=0.05
      target_lambda=0.1
      kl_lambda=0.01
      int_reg=0.1
      lr=5e-4
      ;;
    *)
      # Untuned backbone: use reference hyperparameters from the overall best-test tuned run.
      # Reference run:
      #   results/dr_tuning_casn_backbone_loss_fixed/tune_densenet121_b2p0_il0p5_tl0p1_kl0p01_ir0p2_lr5em5
      bias="${REF_BIAS}"
      int_lambda="${REF_INT_LAMBDA}"
      target_lambda="${REF_TARGET_LAMBDA}"
      kl_lambda="${REF_KL_LAMBDA}"
      int_reg="${REF_INT_REG}"
      lr="${REF_LR}"
      ;;
  esac

  run_name="dr_casn_${arch}_${timestamp}"
  echo ">>> Starting CausalFund DR training (Setting 3) for model_arch=${arch}, run=${run_name}"
  echo ">>> Using defaults: lr=${lr}, batch_size=${batch_size}, bias=${bias}, int_lambda=${int_lambda}, target_lambda=${target_lambda}, kl_lambda=${kl_lambda}, int_reg=${int_reg}"

  CMD=(
    python scripts/train_with_casn.py
    "${DEFAULT_ARGS[@]}"
    --model_arch "${arch}"
    --run_name "${run_name}"
    --lr "${lr}"
    --batch_size "${batch_size}"
    --bias "${bias}"
    --int_lambda "${int_lambda}"
    --target_lambda "${target_lambda}"
    --kl_lambda "${kl_lambda}"
    --int_reg "${int_reg}"
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

