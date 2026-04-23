"""Evaluation metrics for fundus classification."""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple, Optional


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        predictions: Predicted class labels
        labels: True class labels
        probabilities: Prediction probabilities (optional, for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, average='binary', zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, average='binary', zero_division=0)
    
    # Sensitivity and Specificity (for binary classification)
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC if probabilities provided
    if probabilities is not None:
        try:
            unique = np.unique(labels)
            # AUC is undefined if only one class present in y_true
            if unique.size < 2:
                metrics['auc'] = None
            else:
                # For binary classification, roc_auc_score expects probabilities/scores
                # for the "positive" class. With labels in {0,1} and using probs[:, 1],
                # this is the standard ROC-AUC computation.
                #
                # Note: roc_auc_score does not accept a `pos_label` argument.
                metrics['auc'] = roc_auc_score(labels, probabilities)
        except Exception as e:
            warnings.warn(f"AUC calculation failed (returning None): {e}")
            metrics['auc'] = None
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run evaluation on
    
    Returns:
        (metrics_dict, predictions, labels, probabilities)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Get predictions
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels, all_probs)
    
    return metrics, all_predictions, all_labels, all_probs


def calculate_domain_gap(
    hospital_metrics: Dict[str, float],
    smartphone_metrics: Dict[str, float],
    metric_name: str = 'accuracy'
) -> float:
    """
    Calculate domain gap between hospital and smartphone performance.
    
    Args:
        hospital_metrics: Metrics on hospital domain
        smartphone_metrics: Metrics on smartphone domain
        metric_name: Which metric to use for gap calculation
    
    Returns:
        Domain gap (positive means hospital performs better)
    """
    return hospital_metrics[metric_name] - smartphone_metrics[metric_name]


def calculate_worst_group_accuracy(
    group_accuracies: Dict[str, float]
) -> float:
    """
    Calculate worst-group accuracy (fairness metric).
    
    Args:
        group_accuracies: Dictionary mapping group names to accuracies
    
    Returns:
        Minimum accuracy across all groups
    """
    return min(group_accuracies.values())

