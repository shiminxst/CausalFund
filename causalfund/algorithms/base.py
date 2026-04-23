"""Base algorithm class for domain generalization."""

import torch
import torch.nn as nn


class Algorithm(nn.Module):
    """
    Base class for domain generalization algorithms.
    
    All algorithms must implement:
    - update(minibatches): Training step
    - predict(x): Make predictions
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        Args:
            input_shape: Tuple of (channels, height, width)
            num_classes: Number of output classes
            num_domains: Number of training domains
            hparams: Dictionary of hyperparameters
        """
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams
    
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step.
        
        Args:
            minibatches: List of (x, y) tuples, one per domain
            unlabeled: Optional unlabeled data (for domain adaptation)
        
        Returns:
            Dictionary of metrics (e.g., {'loss': 0.5})
        """
        raise NotImplementedError
    
    def predict(self, x):
        """
        Make predictions on input x.
        
        Args:
            x: Input tensor
        
        Returns:
            Logits tensor
        """
        raise NotImplementedError

