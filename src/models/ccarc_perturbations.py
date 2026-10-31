"""Geometry-Preserving Photometric Perturbations for CCARC Consistency Learning.

Implements differentiable and geometry-preserving photometric transformations
(brightness, contrast, gamma, saturation, and mild sensor noise/blur) to generate
weakly and strongly augmented image views (x_w, x_s) for road damage detection.

Crucially, zero geometric transformations (no cropping, rotation, scaling, or flips)
are applied, guaranteeing 100% spatial alignment between teacher and student
anchor predictions across all feature map strides.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhotometricPerturbationConfig:
    """Configuration for weak and strong photometric perturbations."""

    # Weak Augmentation Limits (Mild adjustments preserving natural road scene)
    weak_brightness: float = 0.05  # +/- 5%
    weak_contrast: float = 0.05    # +/- 5%
    weak_gamma: float = 0.05       # +/- 5%

    # Strong Augmentation Limits (Wider domain-shift simulation: weather/lighting/sensors)
    strong_brightness_min: float = 0.65
    strong_brightness_max: float = 1.35
    strong_contrast_min: float = 0.65
    strong_contrast_max: float = 1.35
    strong_gamma_min: float = 0.70
    strong_gamma_max: float = 1.40
    strong_noise_std: float = 0.015  # Additive Gaussian sensor noise std
    strong_blur_prob: float = 0.20   # Probability of applying mild 3x3 Gaussian smoothing


class PhotometricPerturbation(nn.Module):
    """Differentiable Photometric Augmentation Module for Road Imagery.

    Operates directly on PyTorch image tensors with values normalized in [0.0, 1.0],
    of shape (C, H, W) or (B, C, H, W).
    """

    def __init__(self, config: Optional[PhotometricPerturbationConfig] = None) -> None:
        """Initialize perturbation module."""
        super().__init__()
        self.config = config or PhotometricPerturbationConfig()

    def _adjust_brightness(self, x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        """Scale brightness: x * factor."""
        # factor shape: (B, 1, 1, 1)
        return torch.clamp(x * factor, 0.0, 1.0)

    def _adjust_contrast(self, x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        """Scale contrast around per-image mean luminance."""
        # x shape: (B, C, H, W)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, 0.0, 1.0)

    def _adjust_gamma(self, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Apply gamma transformation: x ** gamma."""
        return torch.clamp(x + eps, eps, 1.0) ** gamma

    def _apply_gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a mild 3x3 Gaussian smoothing kernel."""
        kernel = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ], device=x.device, dtype=x.dtype) / 16.0
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def generate_weak_view(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate weakly perturbed view (teacher input).

        Args:
            x: Tensor in [0, 1] of shape (B, C, H, W) or (C, H, W).
            generator: Optional torch.Generator for deterministic reproducibility.

        Returns:
            Weakly perturbed image tensor with identical shape and dtype.
        """
        is_batched = x.ndim == 4
        if not is_batched:
            x = x.unsqueeze(0)

        b, _, _, _ = x.shape
        cfg = self.config

        # Mild random factors
        rand_b = torch.empty(b, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            1.0 - cfg.weak_brightness, 1.0 + cfg.weak_brightness, generator=generator
        )
        rand_c = torch.empty(b, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            1.0 - cfg.weak_contrast, 1.0 + cfg.weak_contrast, generator=generator
        )

        out = self._adjust_brightness(x, rand_b)
        out = self._adjust_contrast(out, rand_c)

        return out if is_batched else out.squeeze(0)

    def generate_strong_view(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate strongly perturbed view (student input).

        Applies wider photometric adjustments and sensor noise, simulating
        challenging road surface illumination and capture conditions across countries.

        Args:
            x: Tensor in [0, 1] of shape (B, C, H, W) or (C, H, W).
            generator: Optional torch.Generator for deterministic reproducibility.

        Returns:
            Strongly perturbed image tensor with identical shape and dtype.
        """
        is_batched = x.ndim == 4
        if not is_batched:
            x = x.unsqueeze(0)

        b, _, _, _ = x.shape
        cfg = self.config

        # 1. Random Brightness
        b_factors = torch.empty(b, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            cfg.strong_brightness_min, cfg.strong_brightness_max, generator=generator
        )
        out = self._adjust_brightness(x, b_factors)

        # 2. Random Contrast
        c_factors = torch.empty(b, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            cfg.strong_contrast_min, cfg.strong_contrast_max, generator=generator
        )
        out = self._adjust_contrast(out, c_factors)

        # 3. Random Gamma
        g_factors = torch.empty(b, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
            cfg.strong_gamma_min, cfg.strong_gamma_max, generator=generator
        )
        out = self._adjust_gamma(out, g_factors)

        # 4. Optional Gaussian Blur
        if cfg.strong_blur_prob > 0.0:
            blur_mask = torch.rand(b, device=x.device, generator=generator) < cfg.strong_blur_prob
            if blur_mask.any():
                blurred = self._apply_gaussian_blur(out)
                out[blur_mask] = blurred[blur_mask]

        # 5. Mild Gaussian Sensor Noise
        if cfg.strong_noise_std > 0.0:
            noise = torch.randn(out.shape, device=x.device, dtype=x.dtype, generator=generator) * cfg.strong_noise_std
            out = torch.clamp(out + noise, 0.0, 1.0)

        return out if is_batched else out.squeeze(0)

    def generate_pair(
        self,
        x: torch.Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate paired (weak, strong) views from a clean input image.

        Args:
            x: Input tensor in [0, 1].
            seed: Optional random seed for deterministic paired generation.

        Returns:
            Tuple of (x_weak, x_strong). Both views maintain 100% geometric alignment.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)

        x_weak = self.generate_weak_view(x, generator=gen)
        x_strong = self.generate_strong_view(x, generator=gen)
        return x_weak, x_strong
