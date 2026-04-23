"""
Neural network architectures for fundus glaucoma detection.

Supports:
- ResNet (18, 50, 101)
- EfficientNet
- Custom architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, List


class FundusClassifier(nn.Module):
    """
    Base classifier for fundus images.
    
    Args:
        backbone: Feature extraction backbone
        num_classes: Number of output classes (default: 2 for binary classification)
        feature_dim: Dimension of feature vector before classifier
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        feature_dim: int = 512,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)


def get_resnet_backbone(
    architecture: str = 'resnet50',
    pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """
    Get ResNet backbone.
    
    Args:
        architecture: 'resnet18', 'resnet50', or 'resnet101'
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        (backbone_model, feature_dim)
    """
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        feature_dim = 512
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        feature_dim = 2048
    elif architecture == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown ResNet architecture: {architecture}")
    
    # Remove the final classification layer
    backbone = nn.Sequential(*list(model.children())[:-1])
    
    # Add adaptive pooling to ensure consistent output size
    backbone = nn.Sequential(
        backbone,
        nn.Flatten()
    )
    
    return backbone, feature_dim


def get_efficientnet_backbone(
    architecture: str = 'efficientnet_b0',
    pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """
    Get EfficientNet backbone.
    
    Args:
        architecture: 'efficientnet_b0', 'efficientnet_b1', etc.
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        (backbone_model, feature_dim)
    """
    if architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = 1280
    elif architecture == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=pretrained)
        feature_dim = 1280
    elif architecture == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained)
        feature_dim = 1408
    else:
        raise ValueError(f"Unknown EfficientNet architecture: {architecture}")
    
    # Remove classifier
    backbone = model.features
    
    # Add pooling and flatten
    backbone = nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    
    return backbone, feature_dim


def get_model(
    architecture: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> FundusClassifier:
    """
    Get fundus classification model.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate
    
    Returns:
        FundusClassifier model
    
    Example:
        >>> model = get_model('resnet50', num_classes=2, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 2])
    """
    # Get backbone
    if architecture.startswith('resnet'):
        backbone, feature_dim = get_resnet_backbone(architecture, pretrained)
    elif architecture.startswith('efficientnet'):
        backbone, feature_dim = get_efficientnet_backbone(architecture, pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Create classifier
    model = FundusClassifier(
        backbone=backbone,
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout=dropout
    )
    
    return model


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    
    Args:
        models: List of models to ensemble
        num_classes: Number of output classes
    """
    
    def __init__(self, models: List[nn.Module], num_classes: int = 2):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Average logits
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

