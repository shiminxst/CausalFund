"""
Domain-aware fundus dataset for domain generalization.

Treats hospital and smartphone as separate "environments" for domain generalization.
Self-contained implementation with no external dependencies.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from causalfund.datasets.fundus_dataset import FundusDataset


class MultipleDomainDataset:
    """
    Base class for multi-domain datasets.
    
    Stores multiple datasets, one per domain/environment.
    """
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 4
    ENVIRONMENTS = None
    INPUT_SHAPE = None
    
    def __getitem__(self, index):
        return self.datasets[index]
    
    def __len__(self):
        return len(self.datasets)


class DomainFundusDataset(MultipleDomainDataset):
    """
    Multi-domain fundus dataset for domain generalization.
    
    Compatible with DomainBed/CaSN training framework.
    Treats hospital and smartphone as separate environments/domains.
    
    Args:
        root: Root directory containing hospital/ and smartphone/ subdirectories
        test_envs: List of environment indices to use for testing
                   (0=hospital, 1=smartphone)
        hparams: Hyperparameters dictionary
        augment: Whether to apply data augmentation to training domains
    """
    
    ENVIRONMENTS = ["hospital", "smartphone"]
    INPUT_SHAPE = (3, 224, 224)
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 4
    
    def __init__(
        self,
        root: str,
        test_envs: List[int],
        hparams: Optional[dict] = None,
        augment: bool = True,
        smartphone_augmentation: Optional[str] = None,
        class_map: Optional[dict] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
    ):
        super().__init__()
        
        self.root = Path(root)
        self.test_envs = test_envs
        self.hparams = hparams or {}
        self.augment = augment
        # Controls which underlying folder to use for the smartphone domain:
        # None / "none" → "smartphone"; "mild" → "smartphone_mild", etc.
        if smartphone_augmentation is None:
            self.smartphone_augmentation: Optional[str] = None
        else:
            aug = smartphone_augmentation.lower()
            self.smartphone_augmentation = None if aug in ("", "none") else aug
        self.class_map = class_map
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.pre_split = self.train_split is not None
        
        self.input_shape = self.INPUT_SHAPE
        self.datasets: List[Dataset] = []
        self.train_env_datasets: List[Dataset] = []
        self.val_env_datasets: Optional[List[Optional[Dataset]]] = None
        self.test_env_datasets: Optional[List[Optional[Dataset]]] = None
        self.env_names: List[str] = list(self.ENVIRONMENTS)
        
        # Define transforms
        self.augment_transform = self._get_augment_transform()
        self.base_transform = self._get_base_transform()
        
        if self.pre_split:
            self._load_pre_split_datasets()
        else:
            self._load_standard_datasets()
        
        self.num_classes = self._infer_num_classes()
        
        print(f"\nDomain Fundus Dataset Summary:")
        print(f"Total environments: {len(self.env_names)}")
        print(f"Test environments: {[self.env_names[i] for i in test_envs if i < len(self.env_names)]}")
        print(f"Training environments: "
              f"{[self.env_names[i] for i in range(len(self.env_names)) if i not in test_envs]}")

    def _get_env_dir_name(self, env_name: str) -> str:
        """Map logical env name to on-disk folder name.

        For hospital, we always use 'hospital'. For smartphone, we may switch
        to 'smartphone_mild', 'smartphone_moderate', or 'smartphone_severe'
        depending on the smartphone_augmentation setting.
        """
        if env_name == "smartphone" and self.smartphone_augmentation:
            return f"{env_name}_{self.smartphone_augmentation}"
        return env_name

    def _resolve_split_dir(self, split: Optional[str], env_name: str) -> Optional[Path]:
        dir_name = self._get_env_dir_name(env_name)
        if split is None:
            return self.root / dir_name
        return self.root / split / dir_name

    def _load_dataset(
        self,
        directory: Optional[Path],
        transform: transforms.Compose,
        domain_label: str,
        allow_missing: bool = False
    ) -> Dataset:
        if directory is None or not directory.exists():
            if not allow_missing:
                print(f"Warning: {directory} does not exist!")
            return torch.utils.data.TensorDataset(
                torch.zeros(0, *self.INPUT_SHAPE),
                torch.zeros(0, dtype=torch.long)
            )
        return FundusDataset(
            data_dir=str(directory),
            transform=transform,
            domain=domain_label,
            class_map=self.class_map
        )

    def _load_standard_datasets(self):
        for env_idx, env_name in enumerate(self.ENVIRONMENTS):
            is_test = env_idx in self.test_envs
            transform = self.base_transform if is_test else self.augment_transform
            env_dir = self.root / self._get_env_dir_name(env_name)
            dataset = self._load_dataset(env_dir, transform, env_name, allow_missing=False)
            self.datasets.append(dataset)

    def _load_pre_split_datasets(self):
        self.val_env_datasets = []
        self.test_env_datasets = []

        for env_idx, env_name in enumerate(self.ENVIRONMENTS):
            # Training dataset
            train_dir = self._resolve_split_dir(self.train_split, env_name)
            transform = self.base_transform if env_idx in self.test_envs else self.augment_transform
            train_dataset = self._load_dataset(train_dir, transform, f"{env_name}_train", allow_missing=False)
            self.train_env_datasets.append(train_dataset)
            self.datasets.append(train_dataset)

            # Validation dataset
            val_dataset = None
            if self.val_split:
                val_dir = self._resolve_split_dir(self.val_split, env_name)
                val_dataset = self._load_dataset(val_dir, self.base_transform, f"{env_name}_val", allow_missing=True)
                if len(val_dataset) == 0:
                    val_dataset = None
            self.val_env_datasets.append(val_dataset)

            # Test dataset
            test_dataset = None
            if self.test_split:
                test_dir = self._resolve_split_dir(self.test_split, env_name)
                test_dataset = self._load_dataset(test_dir, self.base_transform, f"{env_name}_test", allow_missing=True)
                if len(test_dataset) == 0:
                    test_dataset = None
            self.test_env_datasets.append(test_dataset)

    def _infer_num_classes(self) -> int:
        for dataset in self.datasets:
            if hasattr(dataset, "num_classes"):
                return dataset.num_classes
            if len(dataset) > 0:
                _, label = dataset[0]
                return int(label) + 1
        if self.val_env_datasets:
            for dataset in self.val_env_datasets:
                if dataset and len(dataset) > 0:
                    _, label = dataset[0]
                    return int(label) + 1
        return 2
    
    def _get_augment_transform(self) -> transforms.Compose:
        """Get training augmentation transform."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_base_transform(self) -> transforms.Compose:
        """Get test/validation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class DomainFundusDatasetWithAugmentation(DomainFundusDataset):
    """
    Extended version that includes quality-augmented hospital data
    to simulate smartphone images.
    
    Useful when smartphone data is very limited.
    
    Creates a third "pseudo-smartphone" environment by applying
    quality degradation to hospital images.
    """
    
    ENVIRONMENTS = ["hospital", "smartphone", "hospital_degraded"]
    
    def __init__(
        self,
        root: str,
        test_envs: List[int],
        hparams: Optional[dict] = None,
        augment: bool = True,
        quality_degradation_severity: str = 'medium',
        smartphone_augmentation: Optional[str] = None,
        class_map: Optional[dict] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
    ):
        from causalfund.datasets.quality_augmentation import SmartphoneSimulator
        
        self.quality_simulator = SmartphoneSimulator(severity=quality_degradation_severity)
        
        # Initialize parent (loads hospital and smartphone)
        super().__init__(
            root=root,
            test_envs=test_envs,
            hparams=hparams,
            augment=augment,
            smartphone_augmentation=smartphone_augmentation,
            class_map=class_map,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )
        
        # Add quality-degraded hospital images as third environment
        hospital_degraded_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            self.quality_simulator.quality_aug,  # Apply degradation
            transforms.RandomCrop(224) if augment else transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load hospital data again but with degradation transform
        if self.pre_split:
            hospital_dir = self._resolve_split_dir(self.train_split, "hospital")
        else:
            hospital_dir = self.root / "hospital"
        if hospital_dir.exists():
            degraded_dataset = FundusDataset(
                data_dir=str(hospital_dir),
                transform=hospital_degraded_transform,
                domain="hospital_degraded",
                class_map=self.class_map
            )
            if len(self.datasets) < len(self.ENVIRONMENTS):
                self.datasets.append(degraded_dataset)
                if self.pre_split:
                    self.train_env_datasets.append(degraded_dataset)
                    if self.val_env_datasets is not None:
                        self.val_env_datasets.append(None)
                    if self.test_env_datasets is not None:
                        self.test_env_datasets.append(None)
            else:
                self.datasets[-1] = degraded_dataset
                if self.pre_split:
                    self.train_env_datasets[-1] = degraded_dataset
                    if self.val_env_datasets is not None:
                        self.val_env_datasets[-1] = None
                    if self.test_env_datasets is not None:
                        self.test_env_datasets[-1] = None
        
        print(f"\nAdded quality-degraded hospital environment")
        print(f"Total environments: {len(self.datasets)}")


def create_casn_compatible_dataset(
    data_root: str,
    test_domain: str = 'smartphone',
    use_quality_augmentation: bool = False,
    class_map: Optional[Dict[str, int]] = None,
    train_split: Optional[str] = None,
    val_split: Optional[str] = None,
    test_split: Optional[str] = None,
    smartphone_augmentation: Optional[str] = None,
) -> DomainFundusDataset:
    """
    Helper function to create CaSN-compatible dataset.
    
    Args:
        data_root: Root directory with hospital/ and smartphone/ subdirectories
        test_domain: Which domain to use for testing ('hospital' or 'smartphone')
        use_quality_augmentation: Whether to add quality-degraded hospital data
    
    Returns:
        DomainFundusDataset instance ready for CaSN training
    
    Example:
        >>> dataset = create_casn_compatible_dataset(
        ...     data_root="/path/to/fundus_data",
        ...     test_domain='smartphone',
        ...     use_quality_augmentation=True
        ... )
        >>> # Use with CaSN training script
        >>> from domainbed import algorithms
        >>> algorithm = algorithms.CaSN(
        ...     input_shape=dataset.INPUT_SHAPE,
        ...     num_classes=dataset.num_classes,
        ...     num_domains=len(dataset) - len(dataset.test_envs),
        ...     hparams=hparams
        ... )
    """
    # Determine test environment index
    if test_domain == 'hospital':
        test_envs = [0]
    elif test_domain == 'smartphone':
        test_envs = [1]
    else:
        raise ValueError(f"Unknown test_domain: {test_domain}. "
                        f"Choose 'hospital' or 'smartphone'")
    
    # Normalise smartphone augmentation flag
    aug = None
    if smartphone_augmentation is not None:
        aug_lower = smartphone_augmentation.lower()
        if aug_lower not in ("", "none"):
            aug = aug_lower

    # Create appropriate dataset class
    if use_quality_augmentation:
        dataset = DomainFundusDatasetWithAugmentation(
            root=data_root,
            test_envs=test_envs,
            hparams={},
            augment=True,
            quality_degradation_severity='medium',
            smartphone_augmentation=aug,
            class_map=class_map,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )
    else:
        dataset = DomainFundusDataset(
            root=data_root,
            test_envs=test_envs,
            hparams={},
            augment=True,
            smartphone_augmentation=aug,
            class_map=class_map,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    return dataset

