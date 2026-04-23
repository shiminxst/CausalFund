"""
Fast data loaders for training and evaluation.

InfiniteDataLoader: Continuously yields batches for training
FastDataLoader: Standard dataloader with optimizations
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
from itertools import cycle


class _InfiniteSampler(Sampler):
    """
    Sampler that yields indices indefinitely.
    
    Wraps another sampler to create an infinite stream.
    """
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    """
    Dataloader that restarts when the dataset is exhausted.
    
    Useful for training when you want to iterate for a fixed number of steps
    rather than a fixed number of epochs.
    """
    
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()
        
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                replacement=True,
                num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset,
                replacement=True
            )
        
        if weights is None:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler,
                batch_size=batch_size,
                drop_last=True
            )
        else:
            # When using weights, the sampler already provides batch_size samples
            batch_sampler = torch.utils.data.BatchSampler(
                sampler,
                batch_size=batch_size,
                drop_last=False
            )
        
        # Wrap batch sampler to make it infinite
        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True
        ))
    
    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)
    
    def __len__(self):
        raise ValueError("InfiniteDataLoader has no len()")


class FastDataLoader(DataLoader):
    """
    Standard DataLoader with some optimizations.
    
    Pins memory and uses multiple workers for faster data loading.
    """
    
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

