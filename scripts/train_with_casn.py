#!/usr/bin/env python3
"""
Training script using CaSN/DomainBed framework for domain-invariant learning.

This script uses the CaSN implementation from the downloaded repository.
Compares three approaches:
1. Baseline 1: Smartphone only (use --smartphone_only flag)
2. Baseline 2: Hospital + Smartphone with ERM (use --algorithm ERM)
3. CaSN: Hospital + Smartphone with domain-invariant learning (use --algorithm CaSN)
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import csv

import torch
import numpy as np

# Import from CausalFund (self-contained, no external dependencies)
from torch.utils.data import Dataset, DataLoader

from causalfund.datasets.domain_dataset import create_casn_compatible_dataset
from causalfund.utils import set_seed
from causalfund.utils.metrics import calculate_metrics
from causalfund.algorithms import get_algorithm_class
from causalfund.algorithms.networks import SUPPORTED_BACKBONES
from causalfund.dataloaders import FastDataLoader
from causalfund import hparams as hparams_registry


def parse_class_map(class_map_str: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse class mapping string into a dictionary."""
    if not class_map_str:
        return None
    mapping: Dict[str, int] = {}
    entries = [item.strip() for item in class_map_str.split(',') if item.strip()]
    for entry in entries:
        if '=' not in entry:
            raise ValueError(f"Invalid class_map entry '{entry}'. "
                             f"Expected format 'name=index'.")
        name, value = entry.split('=', 1)
        mapping[name.strip()] = int(value.strip())
    return mapping


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train fundus glaucoma model with CaSN'
    )

    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing domain folders or split subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/experiment',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--train_split',
        type=str,
        default=None,
        help='Optional subdirectory name for training split (e.g., "train")'
    )
    parser.add_argument(
        '--val_split',
        type=str,
        default=None,
        help='Optional subdirectory name for validation split (e.g., "val")'
    )
    parser.add_argument(
        '--test_split',
        type=str,
        default=None,
        help='Optional subdirectory name for test split (e.g., "test")'
    )
    parser.add_argument(
        '--class_map',
        type=str,
        default=None,
        help=('Optional class mapping (e.g., "Non-DR=0,DR=1"); if omitted, '
              'class folders are discovered automatically')
    )

    # Model and training
    parser.add_argument(
        '--algorithm',
        type=str,
        default='CaSN_MMD',
        choices=['ERM', 'CaSN', 'CaSN_MMD', 'CaSN_IRM'],
        help='Training algorithm'
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='resnet50',
        choices=SUPPORTED_BACKBONES,
        help='Backbone architecture for feature extraction'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use ImageNet pretrained weights'
    )

    # Training settings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument(
        '--freeze_bn',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to freeze BatchNorm layers in the backbone (sets BN modules to eval mode). "
            "Freezing BN can improve stability for some backbones, but may hurt adaptation for others "
            "(often EfficientNet). Use --no-freeze_bn to allow BN to update during training."
        )
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--steps_per_epoch',
        type=int,
        default=None,
        help='Optional number of steps per epoch (defaults to min batches across domains)'
    )
    parser.add_argument(
        '--include_test_domain_in_train',
        action='store_true',
        help=(
            "If set, also include the test domain's TRAIN split as part of "
            "training (only meaningful for pre-split datasets)."
        ),
    )
    parser.add_argument('--checkpoint_freq', type=int, default=500)
    parser.add_argument(
        '--holdout_fraction',
        type=float,
        default=0.2,
        help='Fraction for validation when not using pre-split folders'
    )

    # CaSN-specific
    parser.add_argument(
        '--bias',
        type=float,
        default=3.0,
        help='CaSN intervention strength'
    )
    parser.add_argument(
        '--int_lambda',
        type=float,
        default=1.0,
        help='CaSN intervention loss weight'
    )
    parser.add_argument(
        '--kl_lambda',
        type=float,
        default=0.01,
        help='CaSN KL divergence weight'
    )
    parser.add_argument(
        '--int_reg',
        type=float,
        default=0.1,
        help='CaSN intervention regularization weight'
    )
    parser.add_argument(
        '--target_lambda',
        type=float,
        default=0.1,
        help='CaSN target consistency loss weight'
    )
    parser.add_argument(
        '--max_optimization_step',
        type=int,
        default=1,
        help='CaSN max optimization steps'
    )

    # CaSN-MMD specific
    parser.add_argument(
        '--mmd_weight',
        type=float,
        default=1.0,
        help='MMD penalty weight (for CaSN_MMD)'
    )
    parser.add_argument(
        '--mmd_kernel',
        type=str,
        default='gaussian',
        choices=['gaussian', 'linear'],
        help='MMD kernel type (for CaSN_MMD)'
    )
    parser.add_argument(
        '--mmd_gamma',
        type=str,
        default='0.5,1.0,2.0',
        help='Comma-separated gamma values for Gaussian MMD kernel'
    )

    # CaSN-IRM specific
    parser.add_argument(
        '--irm_lambda',
        type=float,
        default=1000.0,
        help='IRM penalty weight after annealing (for CaSN_IRM)'
    )
    parser.add_argument(
        '--irm_penalty_anneal_iters',
        type=int,
        default=500,
        help='Iterations before applying full IRM penalty'
    )

    # Other
    parser.add_argument(
        '--test_domain',
        type=str,
        default='smartphone',
        choices=['hospital', 'smartphone'],
        help='Which domain to use as the held-out test environment'
    )
    parser.add_argument(
        '--smartphone_only',
        action='store_true',
        help='Train only on smartphone data (Baseline 1)'
    )
    parser.add_argument(
        '--hospital_only',
        action='store_true',
        help='Train only on hospital data (Baseline 2)'
    )
    parser.add_argument(
        '--use_quality_aug',
        action='store_true',
        help='Use quality-degraded hospital images as an additional environment'
    )
    parser.add_argument(
        '--data_augmentation',
        type=str,
        default='none',
        choices=['none', 'mild', 'moderate', 'severe'],
        help=('Which smartphone data variant to use for train/val/test: '
              '"none" uses the original smartphone/ folders, while "mild", '
              '"moderate", and "severe" use the corresponding pre-generated '
              'smartphone_{level} folders (e.g., smartphone_mild).')
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu']
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='3',
        help='Optional comma-separated GPU indices, e.g. "3,4"'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Optional name for this training run (defaults to timestamp)'
    )
    parser.add_argument(
        '--skip_checkpoints',
        action='store_true',
        help='Skip saving best and last model checkpoints (useful for sweeps)'
    )

    parser.add_argument(
        '--select_best_env',
        type=str,
        default='avg',
        help=(
            "How to select the 'best' checkpoint. "
            "Use 'avg' to select by mean validation AUC across available validation environments "
            "(current default behavior). "
            "Or pass an environment name such as 'smartphone' or 'hospital' to select by that "
            "environment's validation AUC only. "
            "For an optimistic upper bound, you may pass '<env>_test' (e.g. 'smartphone_test') "
            "to select by that environment's test AUC (NOTE: this uses the test set for model "
            "selection and should not be reported as a fair estimate)."
        )
    )

    return parser.parse_args()


METRIC_NAMES = ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity']
PRIMARY_METRIC = 'auc'
METRIC_DISPLAY = {
    'accuracy': 'acc',
    'auc': 'auc',
    'f1': 'f1',
    'sensitivity': 'sens',
    'specificity': 'spec'
}


def setup_hparams(args):
    """Setup hyperparameters for algorithm."""
    # Get default hparams from DomainBed
    hparams = hparams_registry.default_hparams(args.algorithm, 'RotatedMNIST')
    
    # Override with our settings
    hparams['lr'] = args.lr
    hparams['weight_decay'] = args.weight_decay
    hparams['batch_size'] = args.batch_size
    hparams['model_arch'] = args.model_arch
    hparams['pretrained'] = args.pretrained
    hparams['freeze_bn'] = args.freeze_bn
    
    # CaSN-specific
    if 'CaSN' in args.algorithm:
        hparams['bias'] = args.bias
        hparams['int_lambda'] = args.int_lambda
        hparams['kl_lambda'] = args.kl_lambda
        hparams['int_reg'] = args.int_reg
        hparams['target_lambda'] = args.target_lambda
        hparams['max_optimization_step'] = args.max_optimization_step
        hparams['if_adversarial'] = False
        hparams['prior_type'] = 'conditional'

    if args.algorithm == 'CaSN_MMD':
        hparams['mmd_weight'] = args.mmd_weight
        hparams['mmd_kernel'] = args.mmd_kernel
        gamma_values = [float(val.strip()) for val in args.mmd_gamma.split(',') if val.strip()]
        if gamma_values:
            hparams['mmd_gamma'] = gamma_values

    if args.algorithm == 'CaSN_IRM':
        hparams['irm_lambda'] = args.irm_lambda
        hparams['irm_penalty_anneal_iters'] = args.irm_penalty_anneal_iters
    
    return hparams


def evaluate(algorithm, loader, device):
    """Evaluate algorithm on a dataset and return detailed metrics."""
    algorithm.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    binary_probs = (getattr(algorithm, 'num_classes', 2) == 2)
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = algorithm.predict(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            if binary_probs:
                all_probs.append(probs[:, 1].cpu().numpy())
    
    algorithm.train()
    
    if not all_labels:
        return {name: None for name in METRIC_NAMES}
    
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs) if binary_probs and all_probs else None
    
    metrics = calculate_metrics(predictions, labels, probabilities)
    # Ensure all expected keys exist
    return {name: metrics.get(name) for name in METRIC_NAMES}


def format_metric_summary(env_name: str, split: str, metrics: Dict[str, Optional[float]]) -> str:
    parts = [f"{env_name:20s} {split}"]
    for metric_name in METRIC_NAMES:
        label = METRIC_DISPLAY[metric_name]
        value = metrics.get(metric_name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            parts.append(f"{label}: N/A")
        else:
            parts.append(f"{label}: {value:.4f}")
    return " | ".join(parts)


def main():
    args = parse_args()
    
    # Configure visible GPUs if specified
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
    
    # Create output/run directories
    experiment_root = Path(args.output_dir)
    experiment_root.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        algorithm_tag = args.algorithm.replace(' ', '_').lower()
        model_tag = args.model_arch.replace(' ', '_').lower()
        run_name = f"run_{timestamp}_{algorithm_tag}_{model_tag}"
    run_dir = experiment_root / run_name
    run_dir.mkdir(parents=False, exist_ok=True)
    
    # Setup logging
    log_file_path = run_dir / "train.log"
    log_file = open(log_file_path, "a", buffering=1)
    
    def log(message: str):
        print(message, flush=True)
        log_file.write(message + "\n")
    
    # Save arguments
    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seed
    set_seed(args.seed)
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = args.device
    
    log(f"\n{'='*60}")
    log(f"CausalFund Training")
    log(f"{'='*60}")
    log(f"Run directory: {run_dir}")
    log(f"Algorithm: {args.algorithm}")
    log(f"Test domain: {args.test_domain}")
    log(f"Device: {device}")
    log(f"{'='*60}\n")
    
    # Parse class mapping if provided
    class_map = parse_class_map(args.class_map)
    
    # Load dataset
    log("Loading dataset...")
    smartphone_aug = None
    if getattr(args, "data_augmentation", None):
        if args.data_augmentation.lower() not in ("", "none"):
            smartphone_aug = args.data_augmentation.lower()
    dataset = create_casn_compatible_dataset(
        data_root=args.data_root,
        test_domain=args.test_domain,
        use_quality_augmentation=args.use_quality_aug,
        class_map=class_map,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        smartphone_augmentation=smartphone_aug,
    )
    
    if args.smartphone_only and args.hospital_only:
        raise ValueError("Cannot enable both --smartphone_only and --hospital_only.")
    
    # If smartphone only (Baseline 1), use only smartphone environment
    if args.smartphone_only:
        log("\n[BASELINE 1] Using smartphone data only")
        if 'smartphone' not in dataset.env_names:
            raise ValueError("Smartphone domain not found in dataset.")
        smartphone_idx = dataset.env_names.index('smartphone')
        dataset.datasets = [dataset.datasets[smartphone_idx]]
        dataset.env_names = ["smartphone"]
        dataset.test_envs = []
        if getattr(dataset, 'pre_split', False):
            dataset.train_env_datasets = [dataset.train_env_datasets[smartphone_idx]]
            if dataset.val_env_datasets is not None:
                dataset.val_env_datasets = [dataset.val_env_datasets[smartphone_idx]]
            if dataset.test_env_datasets is not None:
                dataset.test_env_datasets = [dataset.test_env_datasets[smartphone_idx]]
    
    if args.hospital_only:
        log("\n[BASELINE 2] Using hospital data only")
        if 'hospital' not in dataset.env_names:
            raise ValueError("Hospital domain not found in dataset.")
        hospital_idx = dataset.env_names.index('hospital')
        dataset.datasets = [dataset.datasets[hospital_idx]]
        dataset.env_names = ["hospital"]
        dataset.test_envs = []
        if getattr(dataset, 'pre_split', False):
            dataset.train_env_datasets = [dataset.train_env_datasets[hospital_idx]]
            if dataset.val_env_datasets is not None:
                dataset.val_env_datasets = [dataset.val_env_datasets[hospital_idx]]
            if dataset.test_env_datasets is not None:
                dataset.test_env_datasets = [dataset.test_env_datasets[hospital_idx]]

    # Log which smartphone image folders are being used on disk
    if 'smartphone' in getattr(dataset, 'env_names', []):
        try:
            base_root = Path(args.data_root)
            smartphone_dir_name = (
                dataset._get_env_dir_name('smartphone')
                if hasattr(dataset, '_get_env_dir_name')
                else 'smartphone'
            )
            if getattr(dataset, 'pre_split', False):
                if args.train_split:
                    log(f"[DATA] smartphone TRAIN folder: {base_root / args.train_split / smartphone_dir_name}")
                if args.val_split:
                    log(f"[DATA] smartphone VAL folder:   {base_root / args.val_split / smartphone_dir_name}")
                if args.test_split:
                    log(f"[DATA] smartphone TEST folder:  {base_root / args.test_split / smartphone_dir_name}")
            else:
                log(f"[DATA] smartphone folder: {base_root / smartphone_dir_name}")
        except Exception as e:
            log(f"[DATA] Warning: could not determine smartphone folder path ({e})")

    # Setup hyperparameters
    hparams = setup_hparams(args)
    
    log("\nHyperparameters:")
    for key, value in hparams.items():
        log(f"  {key}: {value}")
    
    # Split into train and validation
    in_splits: List[Dataset] = []
    train_env_names: List[str] = []
    eval_entries: List[Tuple[str, Dataset]] = []
    test_entries: List[Tuple[str, Dataset]] = []
    pre_split = getattr(dataset, 'pre_split', False)
    split_stats: Dict[str, Dict[str, int]] = {}
    class_stats: Dict[str, Dict[str, int]] = {}

    def _count_classes(samples: List[Tuple[str, int]], split: str, env_name: str):
        counts = class_stats.setdefault(split, {}).setdefault(env_name, [0] * dataset.num_classes)
        labels = [label for _, label in samples]
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        for lbl, cnt in zip(unique_labels, label_counts):
            counts[lbl] += int(cnt)

    if pre_split:
        for env_idx, env_dataset in enumerate(dataset.datasets):
            env_name = dataset.env_names[env_idx]
            if (env_idx in dataset.test_envs) and (not args.include_test_domain_in_train):
                # Default DomainBed behaviour: do not train on the held-out test domain
                continue
            train_env_names.append(env_name)
            in_splits.append(env_dataset)
            split_stats.setdefault(env_name, {})['train'] = len(env_dataset)
            if hasattr(env_dataset, 'samples'):
                _count_classes(env_dataset.samples, 'train', env_name)

        if dataset.val_env_datasets is not None:
            for env_idx, val_dataset in enumerate(dataset.val_env_datasets):
                if val_dataset is None or len(val_dataset) == 0:
                    continue
                env_name = dataset.env_names[env_idx]
                eval_entries.append((env_name, val_dataset))
                split_stats.setdefault(env_name, {})['val'] = len(val_dataset)
                if hasattr(val_dataset, 'samples'):
                    _count_classes(val_dataset.samples, 'val', env_name)

        if dataset.test_env_datasets is not None:
            for env_idx, test_dataset in enumerate(dataset.test_env_datasets):
                if test_dataset is None or len(test_dataset) == 0:
                    continue
                env_name = dataset.env_names[env_idx]
                test_entries.append((env_name, test_dataset))
                split_stats.setdefault(env_name, {})['test'] = len(test_dataset)
                if hasattr(test_dataset, 'samples'):
                    _count_classes(test_dataset.samples, 'test', env_name)
    else:
        for env_i, env_dataset in enumerate(dataset):
            if (env_i in dataset.test_envs) and (not args.include_test_domain_in_train):
                # Default DomainBed behaviour: do not train on the held-out test domain
                continue

            n = len(env_dataset)
            n_out = int(args.holdout_fraction * n)
            n_in = n - n_out
            
            indices = torch.randperm(n).tolist()
            in_indices = indices[:n_in]
            out_indices = indices[n_in:]
            
            in_splits.append(torch.utils.data.Subset(env_dataset, in_indices))
            out_subset = torch.utils.data.Subset(env_dataset, out_indices)
            eval_entries.append((dataset.env_names[env_i], out_subset))
            
            env_name = dataset.env_names[env_i]
            train_env_names.append(env_name)
            split_stats.setdefault(env_name, {})['train'] = n_in
            split_stats.setdefault(env_name, {})['val'] = n_out
            if hasattr(env_dataset, 'samples'):
                in_samples = [env_dataset.samples[i] for i in in_indices]
                val_samples = [env_dataset.samples[i] for i in out_indices]
                _count_classes(in_samples, 'train', env_name)
                _count_classes(val_samples, 'val', env_name)

    if test_entries and not pre_split:
        # For non pre-split datasets, count test env sizes directly
        for env_idx in dataset.test_envs:
            env_name = dataset.env_names[env_idx]
            test_count = len(dataset.datasets[env_idx])
            split_stats.setdefault(env_name, {})['test'] = test_count
            test_entries.append((env_name, dataset.datasets[env_idx]))
            test_dataset = dataset.datasets[env_idx]
            if hasattr(test_dataset, 'samples'):
                _count_classes(test_dataset.samples, 'test', env_name)
    
    log("\nDataset split summary:")
    if not split_stats:
        log("  (No data available for training/validation.)")
    else:
        for env_name in dataset.env_names:
            stats = split_stats.get(env_name, {})
            train_n = stats.get('train', 0)
            val_n = stats.get('val', 0)
            test_n = stats.get('test', 0)
            log(f"  {env_name:15s} | train: {train_n:5d} | val: {val_n:5d} | test: {test_n:5d}")
    if class_stats:
        log("\nPer-class counts:")
        class_names = getattr(dataset, 'classes', [str(i) for i in range(dataset.num_classes)])
        for split in ['train', 'val', 'test']:
            env_counts = class_stats.get(split, {})
            if not env_counts:
                continue
            log(f"  {split.upper()}:")
            for env_name, counts in env_counts.items():
                count_str = ", ".join(f"{cls}:{cnt}" for cls, cnt in zip(class_names, counts))
                log(f"    {env_name:15s} -> {count_str}")
    
    # Create data loaders
    train_loaders = [
        DataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=args.workers
        )
        for env in in_splits
    ]
    
    eval_pairs = [
        (
            env_name,
            FastDataLoader(
                dataset=env_dataset,
                batch_size=64,
                num_workers=args.workers
            )
        )
        for env_name, env_dataset in eval_entries
    ]

    test_pairs = [
        (
            env_name,
        FastDataLoader(
                dataset=env_dataset,
            batch_size=64,
            num_workers=args.workers
            )
        )
        for env_name, env_dataset in test_entries
    ]
    
    # Initialize algorithm
    log(f"\nInitializing {args.algorithm}...")
    algorithm_class = get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        input_shape=dataset.INPUT_SHAPE,
        num_classes=dataset.num_classes,
        num_domains=len(in_splits),
        hparams=hparams
    )
    algorithm.to(device)
    
    n_params = sum(p.numel() for p in algorithm.parameters())
    n_trainable = sum(p.numel() for p in algorithm.parameters() if p.requires_grad)
    log(f"Total parameters: {n_params:,}")
    log(f"Trainable parameters: {n_trainable:,}")
    
    loader_lengths = [len(loader) for loader in train_loaders if len(loader) > 0]
    if not loader_lengths:
        raise ValueError("Training loaders are empty; check dataset and batch size.")
    inferred_steps = min(loader_lengths)
    steps_per_epoch = args.steps_per_epoch or inferred_steps
    log(f"\nSteps per epoch: {steps_per_epoch}")
    
    metrics_file = (run_dir / "metrics.csv").open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    val_header = [
        f"{env_name}_val_{metric}"
        for env_name, _ in eval_pairs
        for metric in METRIC_NAMES
    ]
    metrics_writer.writerow(["epoch", "train_loss"] + val_header)
    metrics_file.flush()
    
    best_checkpoint_path = run_dir / "best_model.pt"
    best_results_path = run_dir / "best_metrics.json"
    last_checkpoint_path = run_dir / "last_model.pt"
    last_results_path = run_dir / "last_metrics.json"
    test_results_path = run_dir / "test_metrics.json"
    save_checkpoints = not args.skip_checkpoints
    if args.skip_checkpoints:
        log("Checkpoint saving disabled (--skip_checkpoints).")
    
    log(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    best_val_score: Optional[float] = None
    select_env_raw = (args.select_best_env or 'avg').strip().lower()
    # Supported:
    #   - avg
    #   - smartphone / hospital  (select by <env> val AUC)
    #   - smartphone_test / hospital_test (select by <env> test AUC; optimistic upper bound)
    select_split = "val"
    select_env = select_env_raw
    if select_env_raw.endswith("_test"):
        select_split = "test"
        select_env = select_env_raw[: -len("_test")]
    if select_env_raw != 'avg':
        if select_split == "test":
            log(f"[MODEL SELECTION] Selecting best checkpoint by {select_env} test {PRIMARY_METRIC} (UPPER BOUND; uses test set for selection)")
        else:
            log(f"[MODEL SELECTION] Selecting best checkpoint by {select_env} val {PRIMARY_METRIC}")
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_losses: List[float] = []
            train_iters = [iter(loader) for loader in train_loaders]
            
            for _ in range(steps_per_epoch):
                minibatches_device = []
                for loader_idx, iterator in enumerate(train_iters):
                    try:
                        x, y = next(iterator)
                    except StopIteration:
                        train_iters[loader_idx] = iter(train_loaders[loader_idx])
                        x, y = next(train_iters[loader_idx])
                    minibatches_device.append((x.to(device), y.to(device)))
                
                step_vals = algorithm.update(minibatches_device)
                epoch_losses.append(step_vals.get('loss', 0.0))
            
            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            log(f"\nEpoch {epoch}/{args.epochs} | Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation at end of epoch
            results = {'epoch': epoch, 'loss': avg_epoch_loss}
            val_primary_scores: List[float] = []
            if eval_pairs:
                for env_name, eval_loader in eval_pairs:
                    metrics = evaluate(algorithm, eval_loader, device)
                    primary_value = metrics.get(PRIMARY_METRIC)
                    if primary_value is not None and not np.isnan(primary_value):
                        val_primary_scores.append(primary_value)
                    for metric_name in METRIC_NAMES:
                        key = f'{env_name}_val_{metric_name}'
                        results[key] = metrics.get(metric_name)
                    log(format_metric_summary(env_name, 'val', metrics))
            else:
                log("No validation splits available; skipping validation metrics.")
            
            avg_val_primary = float(np.mean(val_primary_scores)) if val_primary_scores else float('nan')
            # Determine model selection score
            selection_score = avg_val_primary
            selection_label = f"avg val {PRIMARY_METRIC}"
            if eval_pairs and select_env_raw != 'avg':
                if select_split == "val":
                    key = f"{select_env}_val_{PRIMARY_METRIC}"
                    cand = results.get(key)
                    if cand is None or (isinstance(cand, float) and np.isnan(cand)):
                        log(f"[MODEL SELECTION] Warning: missing {key}; falling back to avg val {PRIMARY_METRIC}")
                    else:
                        selection_score = float(cand)
                        selection_label = f"{select_env} val {PRIMARY_METRIC}"
                else:
                    # Select by test metric (optimistic upper bound). Compute only the required env's
                    # test metric each epoch to keep overhead low.
                    if not test_pairs:
                        log(f"[MODEL SELECTION] Warning: no test split available; falling back to avg val {PRIMARY_METRIC}")
                    else:
                        test_loader = None
                        for env_name, loader in test_pairs:
                            if env_name == select_env:
                                test_loader = loader
                                break
                        if test_loader is None:
                            log(f"[MODEL SELECTION] Warning: missing {select_env} test loader; falling back to avg val {PRIMARY_METRIC}")
                        else:
                            test_metrics = evaluate(algorithm, test_loader, device)
                            cand = test_metrics.get(PRIMARY_METRIC)
                            if cand is None or (isinstance(cand, float) and np.isnan(cand)):
                                log(f"[MODEL SELECTION] Warning: missing {select_env}_test_{PRIMARY_METRIC}; falling back to avg val {PRIMARY_METRIC}")
                            else:
                                selection_score = float(cand)
                                selection_label = f"{select_env} test {PRIMARY_METRIC}"

            if eval_pairs and (best_val_score is None or (not np.isnan(selection_score) and selection_score > best_val_score)):
                best_val_score = selection_score
                if save_checkpoints:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': algorithm.state_dict(),
                        'results': results,
                        'args': vars(args),
                        'hparams': hparams
                    }, best_checkpoint_path)
                best_results = dict(results)
                if test_pairs:
                    log("[BEST MODEL] Test Evaluation")
                    for env_name, test_loader in test_pairs:
                        test_metrics = evaluate(algorithm, test_loader, device)
                        for metric_name in METRIC_NAMES:
                            key = f'{env_name}_test_{metric_name}'
                            best_results[key] = test_metrics.get(metric_name)
                        log("[BEST MODEL] " + format_metric_summary(env_name, 'test', test_metrics))
                with open(best_results_path, 'w') as f:
                    json.dump(best_results, f, indent=4)
                log(f"✓ Saved best model ({selection_label}: {best_val_score:.4f})")
            
            row = [epoch, avg_epoch_loss]
            for env_name, _ in eval_pairs:
                for metric_name in METRIC_NAMES:
                    key = f'{env_name}_val_{metric_name}'
                    value = results.get(key)
                    row.append(value if value is not None else '')
            metrics_writer.writerow(row)
            metrics_file.flush()
    
        # Final save
        log("\n" + "="*60)
        log("TRAINING COMPLETE")
        log("="*60)
        
        final_results = {'epoch': args.epochs}
        if eval_pairs:
            for env_name, eval_loader in eval_pairs:
                metrics = evaluate(algorithm, eval_loader, device)
                for metric_name in METRIC_NAMES:
                    final_results[f'{env_name}_val_{metric_name}'] = metrics.get(metric_name)
                log(format_metric_summary(env_name, 'val(final)', metrics))
        else:
            log("No validation loaders available for final evaluation.")
        
        if test_pairs:
            log("\nTest Evaluation")
            for env_name, test_loader in test_pairs:
                metrics = evaluate(algorithm, test_loader, device)
                for metric_name in METRIC_NAMES:
                    final_results[f'{env_name}_test_{metric_name}'] = metrics.get(metric_name)
                log(format_metric_summary(env_name, 'test', metrics))
        
        if save_checkpoints:
            torch.save({
                'model_state_dict': algorithm.state_dict(),
                'results': final_results,
                'args': vars(args),
                'hparams': hparams
            }, last_checkpoint_path)
        with open(last_results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        test_metrics = {'epoch': final_results['epoch']}
        has_test_metrics = False
        for key, value in final_results.items():
            if '_test_' in key and value is not None:
                test_metrics[key] = value
                has_test_metrics = True
        if has_test_metrics:
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=4)
            log(f"Test metrics saved to: {test_results_path}")
        
        elapsed = time.time() - start_time
        log(f"\nTotal training time: {elapsed/60:.2f} minutes")
        log(f"Results saved to: {run_dir}")
        if best_val_score is not None:
            log(f"Best validation {PRIMARY_METRIC} (selection score): {best_val_score:.4f}")
        else:
            log(f"Best validation {PRIMARY_METRIC}: N/A (no validation split)")
    finally:
        metrics_file.close()
        log_file.close()


if __name__ == '__main__':
    main()

