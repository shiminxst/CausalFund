# CausalFund

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU training)

## Quick Start

### 1. Prepare Your Data

Organize fundus images in one of the following ways.

**Single-root layout (default)**

```
data/
├── hospital/                # High-quality hospital images
│   ├── healthy/
│   │   ├── img001.jpg
│   │   └── ...
│   └── glaucoma/
│       ├── img001.jpg
│       └── ...
└── smartphone/              # Low-quality smartphone images
    ├── healthy/
    └── glaucoma/
```

**Pre-split layout (e.g., for DR datasets)**

```
data/
├── train/
│   ├── hospital/
│   │   ├── Non-DR/
│   │   └── DR/
│   └── smartphone/
│       ├── Non-DR/
│       └── DR/
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

Use `--train_split train --val_split val --test_split test` to load the second layout.  
Provide `--class_map Non-DR=0,DR=1` (or your own mapping) if class names differ from “healthy/glaucoma”.

### 2. Train Models

```bash
# Baseline ERM
python scripts/train_with_casn.py \
    --data_root ./data \
    --algorithm ERM \
    --output_dir ./results/erm_baseline
