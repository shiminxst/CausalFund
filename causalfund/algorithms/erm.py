"""
Empirical Risk Minimization (ERM) algorithm.

Standard supervised learning baseline that minimizes average loss
across all training domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from causalfund.algorithms.base import Algorithm
from causalfund.algorithms.networks import Featurizer, Classifier


class ERM(Algorithm):
    """
    Empirical Risk Minimization.
    
    Simple baseline that trains on all data from all domains
    without any domain adaptation.
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        # Create feature extractor
        self.featurizer = Featurizer(input_shape, hparams)
        
        # Create classifier
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams.get('nonlinear_classifier', False)
        )
        
        # Complete network
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams.get('lr', 5e-5),
            weight_decay=hparams.get('weight_decay', 0.0)
        )
    
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step.
        
        Args:
            minibatches: List of (x, y) tuples from different domains
        
        Returns:
            Dictionary with loss value
        """
        # Concatenate all data from all domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        # Forward pass
        all_logits = self.predict(all_x)
        
        # Compute loss
        loss = F.cross_entropy(all_logits, all_y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def predict(self, x):
        """Make predictions."""
        return self.network(x)

