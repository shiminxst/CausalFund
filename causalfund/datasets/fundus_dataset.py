"""
Fundus image dataset loaders for glaucoma detection.

Supports:
- Hospital (high-quality) and smartphone (low-quality) domains
- Binary classification: healthy vs glaucoma
- Flexible data organization (directory structure or CSV)
"""

import os
from typing import Optional, Callable, Tuple, List, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class FundusDataset(Dataset):
    """
    Fundus image dataset for a single domain.
    
    Expected directory structure:
        data_dir/
            ├── healthy/
            │   ├── img001.jpg
            │   └── ...
            └── glaucoma/
                ├── img001.jpg
                └── ...
    
    Args:
        data_dir: Root directory containing 'healthy' and 'glaucoma' subdirectories
        transform: Optional transform to apply to images
        domain: Domain name (e.g., 'hospital', 'smartphone') for logging
        target_transform: Optional transform to apply to labels
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        domain: str = "unknown",
        target_transform: Optional[Callable] = None,
        class_map: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.domain = domain
        self.class_map = self._prepare_class_map(class_map)
        self.label_to_names = self._build_label_to_names()
        self.inverse_class_map = {
            label: "/".join(sorted(names))
            for label, names in self.label_to_names.items()
        }
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
        # Class names
        self.classes = [self.inverse_class_map[i] for i in sorted(self.inverse_class_map.keys())]
        self.num_classes = len(self.inverse_class_map)
        
        print(f"[{domain}] Loaded {len(self.samples)} images: "
              f"{self.get_class_counts()}")
    
    def _prepare_class_map(self, class_map: Optional[Dict[str, int]]) -> Dict[str, int]:
        """Prepare class mapping from names to integer labels."""
        if class_map:
            prepared = {}
            for name, label in class_map.items():
                prepared[str(name)] = int(label)
            return prepared
        
        # Auto-discover classes from subdirectories
        class_dirs = [
            p.name for p in self.data_dir.iterdir()
            if p.is_dir() and not p.name.startswith('.')
        ]
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_dir}.")
        if {'healthy', 'glaucoma'}.issubset(set(class_dirs)):
            return {'healthy': 0, 'glaucoma': 1}
        class_dirs.sort()
        return {name: idx for idx, name in enumerate(class_dirs)}
    
    def _build_label_to_names(self) -> Dict[int, List[str]]:
        label_to_names: Dict[int, List[str]] = {}
        for name, label in self.class_map.items():
            label_to_names.setdefault(label, []).append(name)
        return label_to_names
    
    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        for class_name, label in self.class_map.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    self.samples.append((str(img_path), label))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}. "
                           f"Check directory structure.")
    
    def get_class_counts(self) -> dict:
        """Get number of samples per class."""
        counts = {label: 0 for label in self.class_map.values()}
        for _, label in self.samples:
            counts[label] += 1
        named_counts = {
            self.inverse_class_map[label]: count
            for label, count in counts.items()
        }
        named_counts['total'] = len(self.samples)
        return named_counts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class FundusCSVDataset(Dataset):
    """
    Fundus dataset loaded from CSV file.
    
    CSV format:
        image_path,label,domain
        /path/to/img1.jpg,0,hospital
        /path/to/img2.jpg,1,smartphone
        ...
    
    Where label: 0=healthy, 1=glaucoma
    """
    
    def __init__(
        self,
        csv_path: str,
        domain: Optional[str] = None,
        transform: Optional[Callable] = None,
        root_dir: Optional[str] = None
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.root_dir = root_dir
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter by domain if specified
        if domain is not None:
            self.df = self.df[self.df['domain'] == domain].reset_index(drop=True)
            self.domain = domain
        else:
            self.domain = "mixed"
        
        print(f"[{self.domain}] Loaded {len(self.df)} images from CSV")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Get image path
        img_path = row['image_path']
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        
        # Load image and label
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class FundusDataModule:
    """
    Data module for managing fundus datasets across multiple domains.
    
    Handles:
    - Loading hospital and smartphone datasets
    - Train/val/test splits
    - Data loaders with appropriate batch sizes
    - Domain-specific augmentations
    
    Args:
        data_root: Root directory containing hospital/ and smartphone/ subdirectories
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_fraction: Fraction of data to use for validation
        augment_train: Whether to apply augmentation to training data
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_fraction: float = 0.2,
        augment_train: bool = True,
        image_size: int = 224
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.augment_train = augment_train
        self.image_size = image_size
        
        # Define transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        
        # Datasets (to be populated)
        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}
    
    def _get_train_transform(self) -> transforms.Compose:
        """Get training data augmentation pipeline."""
        if self.augment_train:
            return transforms.Compose([
                transforms.Resize((int(self.image_size * 1.15), 
                                 int(self.image_size * 1.15))),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return self._get_val_transform()
    
    def _get_val_transform(self) -> transforms.Compose:
        """Get validation/test transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def setup(self, domains: List[str] = ['hospital', 'smartphone']):
        """
        Setup datasets for specified domains.
        
        Args:
            domains: List of domain names to load
        """
        for domain in domains:
            domain_dir = self.data_root / domain
            
            if not domain_dir.exists():
                print(f"Warning: {domain_dir} does not exist, skipping...")
                continue
            
            # Load full domain dataset
            full_dataset = FundusDataset(
                data_dir=str(domain_dir),
                transform=None,  # We'll apply transforms in split datasets
                domain=domain
            )
            
            # Split into train and validation
            n = len(full_dataset)
            n_val = int(n * self.val_fraction)
            n_train = n - n_val
            
            indices = torch.randperm(n).tolist()
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            # Create train and val subsets with appropriate transforms
            self.train_datasets[domain] = torch.utils.data.Subset(
                FundusDataset(
                    data_dir=str(domain_dir),
                    transform=self.train_transform,
                    domain=f"{domain}_train"
                ),
                train_indices
            )
            
            self.val_datasets[domain] = torch.utils.data.Subset(
                FundusDataset(
                    data_dir=str(domain_dir),
                    transform=self.val_transform,
                    domain=f"{domain}_val"
                ),
                val_indices
            )
            
            print(f"[{domain}] Train: {n_train}, Val: {n_val}")
    
    def setup_test(self, test_root: Optional[str] = None, 
                   domains: List[str] = ['hospital', 'smartphone']):
        """
        Setup test datasets.
        
        Args:
            test_root: Root directory for test data (if different from training)
            domains: List of domain names to load for testing
        """
        if test_root is None:
            test_root = self.data_root
        else:
            test_root = Path(test_root)
        
        for domain in domains:
            domain_dir = test_root / domain
            
            if not domain_dir.exists():
                print(f"Warning: Test directory {domain_dir} does not exist")
                continue
            
            self.test_datasets[domain] = FundusDataset(
                data_dir=str(domain_dir),
                transform=self.val_transform,
                domain=f"{domain}_test"
            )
    
    def train_dataloader(self, domain: str) -> DataLoader:
        """Get training dataloader for a specific domain."""
        return DataLoader(
            self.train_datasets[domain],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self, domain: str) -> DataLoader:
        """Get validation dataloader for a specific domain."""
        return DataLoader(
            self.val_datasets[domain],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self, domain: str) -> DataLoader:
        """Get test dataloader for a specific domain."""
        return DataLoader(
            self.test_datasets[domain],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_train_loaders(self) -> dict:
        """Get all training dataloaders as a dictionary."""
        return {
            domain: self.train_dataloader(domain)
            for domain in self.train_datasets.keys()
        }
    
    def get_all_val_loaders(self) -> dict:
        """Get all validation dataloaders as a dictionary."""
        return {
            domain: self.val_dataloader(domain)
            for domain in self.val_datasets.keys()
        }

