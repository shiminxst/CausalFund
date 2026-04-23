"""
Quality degradation augmentation to simulate smartphone-like image quality.

Simulates common issues in smartphone fundus imaging:
- Lower resolution
- Motion blur (hand-held device)
- Sensor noise
- Compression artifacts
- Lighting inconsistencies
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
from typing import Tuple


class GaussianNoise(nn.Module):
    """Add Gaussian noise to simulate sensor noise."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W) in range [0, 1]
        
        Returns:
            Noisy image
        """
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0, 1)


class JPEGCompression(nn.Module):
    """Simulate JPEG compression artifacts."""
    
    def __init__(self, quality_range: Tuple[int, int] = (60, 80)):
        super().__init__()
        self.quality_min, self.quality_max = quality_range
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W)
        
        Returns:
            Compressed image
        """
        # Random quality factor
        quality = np.random.randint(self.quality_min, self.quality_max + 1)
        
        # Convert to PIL
        pil_img = TF.to_pil_image(img)
        
        # Save with compression and reload
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        return TF.to_tensor(compressed_img)


class ResolutionDegradation(nn.Module):
    """Simulate lower resolution by downsampling and upsampling."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.4, 0.7)):
        super().__init__()
        self.scale_min, self.scale_max = scale_range
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W)
        
        Returns:
            Degraded resolution image
        """
        _, h, w = img.shape
        
        # Random downsampling scale
        scale = np.random.uniform(self.scale_min, self.scale_max)
        
        # Downsample
        new_h, new_w = int(h * scale), int(w * scale)
        img_down = TF.resize(img, (new_h, new_w), 
                            interpolation=TF.InterpolationMode.BILINEAR)
        
        # Upsample back to original size (introduces blur)
        img_up = TF.resize(img_down, (h, w), 
                          interpolation=TF.InterpolationMode.BILINEAR)
        
        return img_up


class MotionBlur(nn.Module):
    """Simulate motion blur from hand-held device."""
    
    def __init__(self, kernel_size_range: Tuple[int, int] = (3, 7)):
        super().__init__()
        self.kernel_size_min, self.kernel_size_max = kernel_size_range
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W)
        
        Returns:
            Motion-blurred image
        """
        # Random kernel size (must be odd)
        kernel_size = np.random.randint(
            self.kernel_size_min, 
            self.kernel_size_max + 1
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur as approximation of motion blur
        sigma = np.random.uniform(0.5, 2.0)
        img_pil = TF.to_pil_image(img)
        blurred = TF.gaussian_blur(
            TF.to_tensor(img_pil), 
            kernel_size=kernel_size, 
            sigma=sigma
        )
        
        return blurred


class QualityAugmentation:
    """
    Comprehensive quality degradation augmentation pipeline.
    
    Combines multiple degradation effects to simulate smartphone image quality.
    
    Args:
        blur_prob: Probability of applying blur
        noise_prob: Probability of adding noise
        jpeg_prob: Probability of JPEG compression
        resolution_prob: Probability of resolution degradation
        color_jitter_prob: Probability of color jittering
        apply_all: If True, apply all augmentations; if False, apply randomly
    """
    
    def __init__(
        self,
        blur_prob: float = 0.7,
        noise_prob: float = 0.8,
        jpeg_prob: float = 0.6,
        resolution_prob: float = 0.9,
        color_jitter_prob: float = 0.5,
        apply_all: bool = False
    ):
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.jpeg_prob = jpeg_prob
        self.resolution_prob = resolution_prob
        self.color_jitter_prob = color_jitter_prob
        self.apply_all = apply_all
        
        # Initialize individual augmentations
        self.blur = MotionBlur(kernel_size_range=(3, 7))
        self.noise = GaussianNoise(std=0.05)
        self.jpeg = JPEGCompression(quality_range=(60, 80))
        self.resolution = ResolutionDegradation(scale_range=(0.4, 0.7))
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        )
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply quality degradation to image.
        
        Args:
            img: PIL Image
        
        Returns:
            Degraded PIL Image
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(img)
        
        # Apply degradations (order matters!)
        # 1. Resolution degradation (simulates low-quality camera)
        if self.apply_all or np.random.random() < self.resolution_prob:
            img_tensor = self.resolution(img_tensor)
        
        # 2. Motion blur (hand-held device shake)
        if self.apply_all or np.random.random() < self.blur_prob:
            img_tensor = self.blur(img_tensor)
        
        # 3. Color jittering (inconsistent lighting/white balance)
        if self.apply_all or np.random.random() < self.color_jitter_prob:
            img_pil = TF.to_pil_image(img_tensor)
            img_pil = self.color_jitter(img_pil)
            img_tensor = TF.to_tensor(img_pil)
        
        # 4. Noise (sensor noise)
        if self.apply_all or np.random.random() < self.noise_prob:
            img_tensor = self.noise(img_tensor)
        
        # 5. JPEG compression (image storage/transmission)
        if self.apply_all or np.random.random() < self.jpeg_prob:
            img_tensor = self.jpeg(img_tensor)
        
        # Convert back to PIL
        return TF.to_pil_image(img_tensor)
    
    @staticmethod
    def get_transform(severity: str = 'medium') -> 'QualityAugmentation':
        """
        Get predefined quality augmentation with different severity levels.
        
        Args:
            severity: 'mild', 'medium', or 'severe'
        
        Returns:
            QualityAugmentation instance
        """
        if severity == 'mild':
            return QualityAugmentation(
                blur_prob=0.5,
                noise_prob=0.6,
                jpeg_prob=0.4,
                resolution_prob=0.7,
                color_jitter_prob=0.3
            )
        elif severity == 'medium':
            return QualityAugmentation(
                blur_prob=0.7,
                noise_prob=0.8,
                jpeg_prob=0.6,
                resolution_prob=0.9,
                color_jitter_prob=0.5
            )
        elif severity == 'severe':
            return QualityAugmentation(
                blur_prob=0.9,
                noise_prob=0.9,
                jpeg_prob=0.8,
                resolution_prob=1.0,
                color_jitter_prob=0.7
            )
        else:
            raise ValueError(f"Unknown severity: {severity}. "
                           f"Choose from 'mild', 'medium', 'severe'")


class SmartphoneSimulator:
    """
    High-level interface for simulating smartphone quality from hospital images.
    
    Usage:
        simulator = SmartphoneSimulator(severity='medium')
        augmented_dataset = simulator.augment_dataset(hospital_dataset)
    """
    
    def __init__(self, severity: str = 'medium'):
        self.quality_aug = QualityAugmentation.get_transform(severity)
        self.severity = severity
    
    def augment_image(self, img: Image.Image) -> Image.Image:
        """Augment a single image."""
        return self.quality_aug(img)
    
    def get_augmented_transform(self, base_transform=None) -> T.Compose:
        """
        Get a transform pipeline that includes quality degradation.
        
        Args:
            base_transform: Optional base transform to apply after degradation
        
        Returns:
            Composed transform
        """
        transforms_list = [
            T.Resize((256, 256)),
            self.quality_aug,
        ]
        
        if base_transform:
            transforms_list.append(base_transform)
        else:
            # Default: convert to tensor and normalize
            transforms_list.extend([
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        return T.Compose(transforms_list)

