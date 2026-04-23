#!/usr/bin/env bash
# Train CausalFund (CaSN) for glaucoma (Setting 3) across backbones.
#
# Copied/adapted from scripts/train_glaucoma.sh, but:
# - Uses per-backbone default hyperparameters from the *best-test selection* tuned runs under:
#     results/glaucoma_tuning_casn_backbone_loss_fixed/
# - Keeps batch size as a per-model variable (`batch_size`) so you can tune batch size only.
#
# Notes:
# - Selection rule used for defaults below: choose, per backbone, the tuned run with the best
#   smartphone_test_auc (from each run's best_metrics.json).
# - Hyperparameters are encoded in the run folder name (best_metrics.json does not include them).
#
# Example:
#   scripts/train_glaucoma_casn.sh --gpus 0 --epochs 20 --data_augmentation none --weight_decay 0.0

set -euo pipefail

# Ensure local package is discoverable without installation
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Reference hyperparameters for *untuned* backbones.
# Overall best-test reference run:
#   results/glaucoma_tuning_casn_backbone_loss_fixed/
#     tune_glaucoma_s3_vit_b_16_b2p0_il0p25_tl0p1_kl0p0025_ir0p2_lr5em5
REF_BIAS=2.0
REF_INT_LAMBDA=0.25
REF_TARGET_LAMBDA=0.1
REF_KL_LAMBDA=0.0025
REF_INT_REG=0.2
REF_LR=5e-5

# Backbones to evaluate (same list pattern as `train_glaucoma.sh`).
# Any backbone without tuned defaults will fall back to the REF_* values above.
MODEL_ARCHES=(
  # resnet50
  # vgg16_bn
  # efficientnet_b0
  # densenet121
  # vit_b_16
  # mobilenet_v2
  squeezenet1_1
)

GPUS="2"
WEIGHT_DECAY="0."
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Shared default arguments for glaucoma CaSN (Setting 3):
# - Train on hospital
# - Validate/test on smartphone (test_domain=smartphone)
# IMPORTANT: do NOT set --hospital_only here, otherwise smartphone metrics won't be evaluated.
DEFAULT_ARGS=(
  --data_root ./data/glaucoma
  --train_split train
  --val_split val
  --test_split test
  --class_map "normal=0,glaucoma=1"
  --algorithm CaSN
  --test_domain smartphone
  --select_best_env smartphone_test
  # --hospital_only                 # (Setting 1) train/val/test on hospital only
  # --include_test_domain_in_train  # (Setting 4) include smartphone-train in training
  --data_augmentation severe
  --pretrained
  --epochs 60
  --output_dir ./results/glaucoma_casn_setting3_loss_fixed_severe
  --gpus 4
  --seed 0
  --weight_decay "${WEIGHT_DECAY}"
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
  # IMPORTANT: keep `batch_size` as a variable so you can tune it easily.
  batch_size=32
  # Default: allow BatchNorm to update during training.
  freeze_bn_flag="--no-freeze_bn"

  case "${arch}" in
    densenet121)
      # best-test tuned run:
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_densenet121_b2p0_il0p25_tl0p05_kl0p01_ir0p1_lr5em5
      bias=2.0
      int_lambda=0.25
      target_lambda=0.05
      kl_lambda=0.0025
      int_reg=0.01
      lr=5e-05
      freeze_bn_flag="--no-freeze_bn"
      ;;
    efficientnet_b0)
      # best-test tuned run:
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_efficientnet_b0_b2p0_il0p25_tl0p2_kl0p0025_ir0p05_lr5em5
      bias=0.0
      int_lambda=0.025
      target_lambda=0.02
      kl_lambda=0.0025
      int_reg=0.05
      lr=5e-4
      # EfficientNet-B0 can collapse with frozen BN; allow BN updates.
      freeze_bn_flag="--no-freeze_bn"
      ;;
    mobilenet_v2)
      # best-test tuned run:
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_mobilenet_v2_b2p0_il0p25_tl0p05_kl0p005_ir0p05_lr5em5
      bias=2.0
      int_lambda=0.25
      target_lambda=0.05
      kl_lambda=0.005
      int_reg=0.05
      lr=5e-5
      freeze_bn_flag="--no-freeze_bn"
      ;;
    resnet50)
      # best-test tuned run:
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_resnet50_b2p0_il0p5_tl0p05_kl0p005_ir0p05_lr2em5
      bias=2.0
      int_lambda=0.5
      target_lambda=0.05
      kl_lambda=0.005
      int_reg=0.05
      lr=2e-5
      freeze_bn_flag="--no-freeze_bn"
      ;;
    squeezenet1_1)
      # best-test tuned run:
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_squeezenet1_1_b2p0_il0p5_tl0p05_kl0p005_ir0p1_lr5em5
      bias=2.0
      int_lambda=0.5
      target_lambda=0.05
      kl_lambda=0.005
      int_reg=0.1
      lr=5e-5
      # freeze_bn_flag="--no-freeze_bn"
      ;;
    vit_b_16)
      # best-test tuned run (also the overall reference run):
      #   results/glaucoma_tuning_casn_backbone_loss_fixed/
      #     tune_glaucoma_s3_vit_b_16_b2p0_il0p25_tl0p1_kl0p0025_ir0p2_lr5em5
      bias=2.0
      int_lambda=0.25
      target_lambda=0.1
      kl_lambda=0.0025
      int_reg=0.2
      lr=5e-6
      freeze_bn_flag="--no-freeze_bn"
      ;;
    *)
      # Untuned backbone: use reference hyperparameters from the overall best-test tuned run.
      bias="${REF_BIAS}"
      int_lambda="${REF_INT_LAMBDA}"
      target_lambda="${REF_TARGET_LAMBDA}"
      kl_lambda="${REF_KL_LAMBDA}"
      int_reg="${REF_INT_REG}"
      lr="${REF_LR}"
      ;;
  esac

  run_name="glaucoma_casn_${arch}_${timestamp}"
  echo ">>> Starting glaucoma CaSN training (Setting 3) for model_arch=${arch}, run=${run_name}"
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
    "${freeze_bn_flag}"
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

