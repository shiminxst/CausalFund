"""
CaSN (Causal-Spurious Networks) algorithm.

Learns domain-invariant causal representations by using an intervention
mechanism to modify spurious (domain-specific) features while preserving
causal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from causalfund.algorithms.base import Algorithm
from causalfund.algorithms.networks import CaSNNetwork


class CaSN(Algorithm):
    """
    Causal-Spurious Networks for domain-invariant learning.
    
    Key idea:
    - Learn an intervener that modifies domain-specific features
    - Predictions should be robust to these interventions
    - Forces the model to rely on causal (domain-invariant) features
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CaSN, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        # Create CaSN network
        self.network = CaSNNetwork(input_shape, num_classes, hparams, num_domains)
        
        # Hyperparameters
        self.bias = hparams.get('bias', 3.0)
        self.int_lambda = hparams.get('int_lambda', 1.0)
        self.kl_lambda = hparams.get('kl_lambda', 0.01)
        self.int_reg = hparams.get('int_reg', 0.1)
        self.target_lambda = hparams.get('target_lambda', 0.1)
        self.prior_type = hparams.get('prior_type', 'conditional')
        
        # Two optimizers: one for min, one for max (if adversarial)
        lr = hparams.get('lr', 5e-5)
        weight_decay = hparams.get('weight_decay', 0.0)
        
        # Main optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Update counter
        self.register_buffer('update_count', torch.tensor([0]))
    
    def kl_divergence(self, mean, logvar, prior_mean, prior_logvar):
        """
        Compute KL divergence between two Gaussians.
        
        KL(q || p) where q ~ N(mean, var) and p ~ N(prior_mean, prior_var)
        """
        element_wise = (mean - prior_mean).pow(2)
        kl = element_wise.mean()
        return kl
    
    def get_prior(self, labels, dim):
        """
        Get prior distribution parameters.
        
        Args:
            labels: Class labels
            dim: Dimension of latent space
        
        Returns:
            (prior_mean, prior_var)
        """
        device = labels.device
        
        if self.prior_type == 'conditional':
            # Conditional prior based on class labels
            mean = ((labels.float() - 0) / (self.num_classes - 1)).reshape(-1, 1).repeat(1, dim)
            var = torch.ones(labels.size(0), dim)
        else:
            # Standard normal prior
            mean = torch.zeros(labels.size(0), dim)
            var = torch.ones(labels.size(0), dim)
        
        return mean.to(device), var.to(device)
    
    def intervention_loss(self, intervention):
        """
        Regularize intervention to have controlled magnitude.
        
        Want interventions to be meaningful but not too large.
        """
        return torch.norm(torch.pow(intervention, 2) - self.bias)
    
    def target_consistency_loss(self, y_pred, intervened_y_pred):
        """
        Encourage consistency between predictions on original and intervened features.
        
        The model should make similar predictions even after intervention.
        """
        # NOTE:
        # This term should be minimized (>= 0). A negative MSE makes the total
        # objective unbounded below and can lead to very large negative losses.
        return self.mse(torch.sigmoid(y_pred), torch.sigmoid(intervened_y_pred))
    
    def loss_from_outputs(self, outputs, y):
        """
        Compute CaSN loss given precomputed network outputs.
        """
        mean, logvar, z, intervened_z, y_pred, intervened_y_pred, intervention, z_c = outputs
        
        # 1. Classification loss
        classification_loss = F.cross_entropy(y_pred, y)
        
        # 2. Intervention classification loss
        #
        # We want the predictor to remain correct after intervention, so we
        # minimize the intervened classification loss as well.
        #
        # (If you intend an adversarial min-max game where the intervener
        # maximizes this term while the classifier minimizes it, that requires
        # separate optimization steps/parameter groups; applying a negative CE
        # to all parameters makes the objective unbounded below and can cause
        # the reported loss to become very large negative.)
        intervention_classification_loss = F.cross_entropy(intervened_y_pred, y)
        
        # 3. KL divergence
        prior_mean, prior_var = self.get_prior(y, z.size(1))
        kl_loss_z = self.kl_divergence(z, torch.zeros_like(z), prior_mean, prior_var * 0.0001)
        kl_loss_zc = self.kl_divergence(z_c, torch.zeros_like(z_c), prior_mean, prior_var * 0.0001)
        kl_loss = kl_loss_z + kl_loss_zc
        
        # 4. Intervention regularization
        intervention_reg = self.intervention_loss(intervention)
        
        # 5. Target consistency
        target_consistency = self.target_consistency_loss(y_pred, intervened_y_pred)
        
        total_loss = (
            classification_loss +
            self.int_lambda * intervention_classification_loss +
            self.kl_lambda * kl_loss +
            self.int_reg * intervention_reg +
            self.target_lambda * target_consistency
        )
        
        return total_loss, {
            'classification_loss': classification_loss.item(),
            'intervention_classification_loss': intervention_classification_loss.item(),
            'kl_loss': kl_loss.item(),
            'intervention_reg': intervention_reg.item(),
            'target_consistency': target_consistency.item(),
        }
    
    def compute_loss(self, x, y):
        """
        Compute CaSN loss.
        
        Loss components:
        1. Classification loss on original features
        2. Classification loss on intervened features (negative, want to maximize)
        3. KL divergence (regularization)
        4. Intervention regularization
        5. Target consistency
        """
        outputs = self.network(x, y)
        return self.loss_from_outputs(outputs, y)
    
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step.
        
        Args:
            minibatches: List of (x, y) tuples from different domains
        
        Returns:
            Dictionary with loss values
        """
        # Process each domain separately (CaSN can leverage domain structure)
        total_loss = 0
        loss_dict = {}
        
        for i, (x, y) in enumerate(minibatches):
            # Compute loss for this domain
            loss, losses = self.compute_loss(x, y)
            total_loss += loss
            
            # Accumulate loss components
            for key, val in losses.items():
                if key not in loss_dict:
                    loss_dict[key] = []
                loss_dict[key].append(val)
        
        # Average loss across domains
        total_loss = total_loss / len(minibatches)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update counter
        self.update_count += 1
        
        # Return averaged losses
        result = {'loss': total_loss.item()}
        for key, vals in loss_dict.items():
            result[key] = sum(vals) / len(vals)
        
        return result
    
    def predict(self, x):
        """
        Make predictions.
        
        Uses the original (non-intervened) prediction.
        """
        # Forward pass
        _, _, _, _, y_pred, _, _, _ = self.network(x, labels=None)
        return y_pred


class CaSN_MMD(CaSN):
    """
    CaSN variant with Maximum Mean Discrepancy (MMD) alignment across domains.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.mmd_weight = hparams.get('mmd_weight', 1.0)
        self.mmd_kernel = hparams.get('mmd_kernel', 'gaussian')
        self.mmd_gamma = hparams.get('mmd_gamma', [0.5, 1.0, 2.0])

    def gaussian_kernel(self, x, y):
        """
        Gaussian kernel matrix between x and y.
        """
        dist = torch.cdist(x, y, p=2).pow(2)
        kernel = torch.zeros_like(dist)
        for gamma in self.mmd_gamma:
            kernel = kernel + torch.exp(-gamma * dist)
        return kernel

    def linear_kernel(self, x, y):
        """
        Linear kernel matrix between x and y.
        """
        return x @ y.t()

    def mmd_distance(self, x, y):
        """
        Maximum Mean Discrepancy between two sets of representations.
        """
        if x.size(0) == 0 or y.size(0) == 0:
            return torch.tensor(0.0, device=x.device)

        min_size = min(x.size(0), y.size(0))
        x = x[:min_size]
        y = y[:min_size]

        if self.mmd_kernel == 'gaussian':
            Kxx = self.gaussian_kernel(x, x)
            Kyy = self.gaussian_kernel(y, y)
            Kxy = self.gaussian_kernel(x, y)
        else:
            Kxx = self.linear_kernel(x, x)
            Kyy = self.linear_kernel(y, y)
            Kxy = self.linear_kernel(x, y)

        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    def update(self, minibatches, unlabeled=None):
        """
        Override update to incorporate MMD penalty.
        """
        outputs_per_domain = []
        labels_per_domain = []

        for x, y in minibatches:
            outputs = self.network(x, y)
            outputs_per_domain.append(outputs)
            labels_per_domain.append(y)

        total_loss = 0.0
        loss_dict = {}

        for outputs, y in zip(outputs_per_domain, labels_per_domain):
            loss, losses = self.loss_from_outputs(outputs, y)
            total_loss += loss
            for key, val in losses.items():
                loss_dict.setdefault(key, []).append(val)

        total_loss = total_loss / len(minibatches)

        mmd_penalty = torch.tensor(0.0, device=total_loss.device)
        num_pairs = 0
        representations = [outputs[7] for outputs in outputs_per_domain]  # z_c representations

        if len(representations) > 1:
            for i in range(len(representations)):
                for j in range(i + 1, len(representations)):
                    mmd_penalty = mmd_penalty + self.mmd_distance(representations[i], representations[j])
                    num_pairs += 1
            if num_pairs > 0:
                mmd_penalty = mmd_penalty / num_pairs

        objective = total_loss + self.mmd_weight * mmd_penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        self.update_count += 1

        result = {'loss': total_loss.item(), 'mmd_penalty': mmd_penalty.item()}
        for key, vals in loss_dict.items():
            result[key] = sum(vals) / len(vals)
        return result


class CaSN_IRM(CaSN):
    """
    CaSN variant combined with Invariant Risk Minimization (IRM).
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.irm_lambda = hparams.get('irm_lambda', 1.0)
        self.irm_penalty_anneal_iters = hparams.get('irm_penalty_anneal_iters', 500)

    @staticmethod
    def _irm_penalty(logits, y):
        if logits.size(0) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return (grad_1 * grad_2).sum()

    def update(self, minibatches, unlabeled=None):
        outputs_per_domain = []
        labels_per_domain = []

        for x, y in minibatches:
            outputs = self.network(x, y)
            outputs_per_domain.append(outputs)
            labels_per_domain.append(y)

        total_loss = 0.0
        penalties = []
        loss_dict = {}

        for outputs, y in zip(outputs_per_domain, labels_per_domain):
            loss, losses = self.loss_from_outputs(outputs, y)
            total_loss += loss
            penalties.append(self._irm_penalty(outputs[4], y))
            for key, val in losses.items():
                loss_dict.setdefault(key, []).append(val)

        num_domains = len(minibatches)
        total_loss = total_loss / num_domains
        penalty = torch.stack(penalties).mean() if penalties else torch.tensor(0.0, device=total_loss.device)

        penalty_weight = self.irm_lambda if self.update_count >= self.irm_penalty_anneal_iters else 1.0
        objective = total_loss + penalty_weight * penalty

        if self.update_count == self.irm_penalty_anneal_iters:
            # Reset optimizer to avoid issues with sudden gradient scale changes
            lr = self.hparams.get('lr', 5e-5)
            weight_decay = self.hparams.get('weight_decay', 0.0)
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        self.update_count += 1

        result = {
            'loss': total_loss.item(),
            'irm_penalty': penalty.item()
        }
        for key, vals in loss_dict.items():
            result[key] = sum(vals) / len(vals)
        return result

