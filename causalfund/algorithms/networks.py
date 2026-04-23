"""
Neural network architectures for CausalFund algorithms.

Simplified, self-contained implementations for fundus image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


SUPPORTED_BACKBONES = [
    'resnet18',
    'resnet50',
    'resnet101',
    'vgg16_bn',
    'efficientnet_b0',
    'efficientnet_b3',
    'densenet121',
    'vit_b_16',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'shufflenet_v2_x1_0',
    'squeezenet1_1',
    'mobileformer_294m'
]


def _safe_create_model(arch: str, pretrained: bool):
    """
    Create a torchvision model, falling back to non-pretrained if the
    weights hash check or download fails.
    """
    try:
        return getattr(models, arch)(pretrained=pretrained)
    except RuntimeError as e:
        msg = str(e)
        if "invalid hash value" in msg or "download error" in msg or "checksum" in msg:
            print(f"[WARN] Pretrained weights failed for {arch} ({msg}). Falling back to pretrained=False.")
            return getattr(models, arch)(pretrained=False)
        raise


def _replace_last_linear(sequential_module):
    """Replace the last Linear layer in a Sequential block with Identity."""
    for idx in reversed(range(len(sequential_module))):
        if isinstance(sequential_module[idx], nn.Linear):
            in_features = sequential_module[idx].in_features
            sequential_module[idx] = nn.Identity()
            return in_features
    raise ValueError("Unable to locate Linear layer to replace in classifier.")


def _build_backbone(model_arch: str, pretrained: bool):
    """Construct backbone network and return module plus output dimension."""
    arch = model_arch.lower()
    
    if arch.startswith('resnet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported ResNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        n_outputs = model.fc.in_features
        model.fc = nn.Identity()
        return model, n_outputs
    
    if arch.startswith('vgg'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported VGG architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if not isinstance(model.classifier, nn.Sequential):
            raise ValueError("Unexpected classifier type for VGG backbone.")
        n_outputs = _replace_last_linear(model.classifier)
        return model, n_outputs
    
    if arch.startswith('efficientnet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported EfficientNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if not isinstance(model.classifier, nn.Sequential):
            raise ValueError("Unexpected classifier for EfficientNet backbone.")
        n_outputs = _replace_last_linear(model.classifier)
        return model, n_outputs
    
    if arch.startswith('densenet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported DenseNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if not isinstance(model.classifier, nn.Linear):
            raise ValueError("Unexpected classifier for DenseNet backbone.")
        n_outputs = model.classifier.in_features
        model.classifier = nn.Identity()
        return model, n_outputs
    
    if arch.startswith('vit'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported ViT architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if not hasattr(model.heads, 'head'):
            raise ValueError("Unexpected ViT heads module.")
        n_outputs = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, n_outputs
    
    if arch.startswith('mobilenet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported MobileNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if isinstance(model.classifier, nn.Sequential):
            n_outputs = _replace_last_linear(model.classifier)
        else:
            raise ValueError("Unexpected classifier for MobileNet backbone.")
        return model, n_outputs
    
    if arch.startswith('shufflenet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported ShuffleNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        if not hasattr(model, 'fc'):
            raise ValueError("ShuffleNet backbone missing fc layer.")
        n_outputs = model.fc.in_features
        model.fc = nn.Identity()
        return model, n_outputs
    
    if arch.startswith('squeezenet'):
        if not hasattr(models, arch):
            raise ValueError(f"Unsupported SqueezeNet architecture '{model_arch}'.")
        model = _safe_create_model(arch, pretrained)
        backbone = nn.Sequential(
            model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        n_outputs = 512
        return backbone, n_outputs
    
    if arch.startswith('mobileformer'):
        if timm is None:
            raise ImportError(
                "Mobile-Former backbones require the 'timm' package. "
                "Please install timm>=0.9 to use this architecture."
            )
        backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        if not hasattr(backbone, 'num_features'):
            raise ValueError("Unable to determine feature dimension for Mobile-Former backbone.")
        return backbone, backbone.num_features
    
    raise ValueError(f"Unsupported backbone architecture '{model_arch}'. "
                     f"Available options: {', '.join(SUPPORTED_BACKBONES)}")


class Featurizer(nn.Module):
    """
    Feature extractor backbone (ResNet-based).
    
    Extracts features from fundus images using pretrained ResNet.
    """
    
    def __init__(self, input_shape, hparams):
        super(Featurizer, self).__init__()
        
        model_arch = hparams.get('model_arch', 'resnet50')
        pretrained = hparams.get('pretrained', True)
        if model_arch not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported model_arch '{model_arch}'. "
                f"Valid options: {', '.join(SUPPORTED_BACKBONES)}"
            )
        self.network, self.n_outputs = _build_backbone(model_arch, pretrained)
        
        # Add dropout
        self.dropout = nn.Dropout(hparams.get('resnet_dropout', 0.0))
        
        # Optionally freeze batch norm for stability
        self._freeze_bn = bool(hparams.get('freeze_bn', True))
        if self._freeze_bn:
            self.freeze_bn()
    
    def forward(self, x):
        """Extract features from input."""
        return self.dropout(self.network(x))
    
    def train(self, mode=True):
        """Override train to keep batch norm frozen."""
        super().train(mode)
        if self._freeze_bn:
            self.freeze_bn()
    
    def freeze_bn(self):
        """Freeze batch normalization layers."""
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Classifier(nn.Module):
    """
    Classification head on top of features.
    
    Can be linear or with an additional hidden layer.
    """
    
    def __init__(self, in_features, num_classes, is_nonlinear=False):
        super(Classifier, self).__init__()
        
        if is_nonlinear:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features // 2, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


class Intervener(nn.Module):
    """
    Intervention network for CaSN.
    
    Learns interventions on latent representations to modify
    domain-specific (spurious) features.
    """
    
    def __init__(self, input_dim, num_classes, num_domains):
        super(Intervener, self).__init__()
        
        hidden_dim = input_dim // 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, z, y_onehot):
        """
        Generate intervention based on representation and label.
        
        Args:
            z: Latent representation
            y_onehot: One-hot encoded labels
        
        Returns:
            Intervention delta
        """
        inp = torch.cat([z, y_onehot], dim=1)
        return self.network(inp)


class CaSNNetwork(nn.Module):
    """
    Complete CaSN network with intervener.
    
    Combines feature extraction, intervention, and classification.
    """
    
    def __init__(self, input_shape, num_classes, hparams, num_domains):
        super(CaSNNetwork, self).__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        # Feature extractor
        self.featurizer = Featurizer(input_shape, hparams)
        feature_dim = self.featurizer.n_outputs
        
        # Intervener
        self.intervener = Intervener(feature_dim, num_classes, num_domains)
        
        # Classifiers (for both original and intervened features)
        self.classifier = Classifier(
            feature_dim, 
            num_classes, 
            hparams.get('nonlinear_classifier', False)
        )
        
        # For variational approach
        self.mean_encoder = nn.Linear(feature_dim, feature_dim)
        self.logvar_encoder = nn.Linear(feature_dim, feature_dim)
    
    def encode(self, x):
        """Encode input to mean and log variance."""
        features = self.featurizer(x)
        mean = self.mean_encoder(features)
        logvar = self.logvar_encoder(features)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE-style training."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, labels=None):
        """
        Forward pass through CaSN network.
        
        Returns:
            Tuple of (mean, logvar, z, intervened_z, y_pred, intervened_y_pred, intervention, z_classifier)
        """
        # Encode to latent space
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        
        # Prediction on original features
        y_pred = self.classifier(z)
        
        # Generate intervention
        if labels is not None:
            y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        else:
            # Use predicted labels if true labels not available
            y_onehot = F.softmax(y_pred, dim=1)
        
        intervention = self.intervener(z, y_onehot)
        
        # Apply intervention
        intervened_z = z + intervention
        
        # Prediction on intervened features
        intervened_y_pred = self.classifier(intervened_z)
        
        # For compatibility, also return classifier features
        z_classifier = z
        
        return mean, logvar, z, intervened_z, y_pred, intervened_y_pred, intervention, z_classifier

