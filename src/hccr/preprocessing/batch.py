"""Versioned tensor augmentation of deterministically prepared grayscale batches."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as functional


class TensorBatchAugmenter:
    """Apply one guarded affine warp and optional blur, without PIL pixel parity."""

    backend = "tensor_affine_v1"
    guard_pixels = 2

    def __init__(
        self,
        image_size: int,
        rotation_degrees: float = 7,
        translate_ratio: float = 0.05,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        blur_probability: float = 0,
        seed: int = 7,
    ) -> None:
        if type(image_size) is not int or image_size < 8:
            raise ValueError("image_size must be an integer of at least 8 pixels")
        values = (
            rotation_degrees,
            translate_ratio,
            scale_min,
            scale_max,
            blur_probability,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("augmentation parameters must be finite")
        if rotation_degrees < 0 or not 0 <= translate_ratio <= 1:
            raise ValueError("rotation must be nonnegative and translation in [0, 1]")
        if not 0 < scale_min <= scale_max or not 0 <= blur_probability <= 1:
            raise ValueError("scale bounds or blur probability are invalid")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.image_size = image_size
        self.rotation_degrees = float(rotation_degrees)
        self.translate_ratio = float(translate_ratio)
        self.scale_min, self.scale_max = float(scale_min), float(scale_max)
        self.blur_probability = float(blur_probability)
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Restart this epoch's private CPU parameter stream for replay."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._generator.manual_seed((self.seed + epoch) % (2**63 - 1))

    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "image_size": self.image_size,
            "rotation_degrees": self.rotation_degrees,
            "translate_ratio": self.translate_ratio,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "blur_probability": self.blur_probability,
            "blur_kernel_size": 5,
            "blur_sigma": 0.5,
            "white_fill": 1.0,
            "geometry_policy": (
                "single_bilinear_affine_align_corners_false; ink_threshold=250/255; "
                "pixel_footprint_corner_scale_cap; guard=2px; "
                "translation_clamped_to_guard_feasible_center; identity_bypass"
            ),
            "tag_policy": "sampled_operations_before_geometry_guard",
            "rng": "private_cpu_torch_generator_seed_plus_epoch_mod_2**63_minus_1",
            "seed": self.seed,
        }

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        if (
            images.dtype != torch.float32
            or images.ndim != 4
            or images.shape[1:] != (1, self.image_size, self.image_size)
        ):
            raise ValueError("images must be FP32 [N, 1, image_size, image_size]")
        count = images.shape[0]
        if count == 0:
            return images.clone(), []
        draws = torch.rand(
            (count, 5), generator=self._generator, device="cpu", dtype=torch.float32
        )
        scale = self.scale_min + draws[:, 0] * (self.scale_max - self.scale_min)
        translation = (draws[:, 1:3] * 2 - 1) * self.translate_ratio * self.image_size
        angles = (draws[:, 3] * 2 - 1) * self.rotation_degrees
        blur = draws[:, 4] < self.blur_probability
        tags = []
        for scaling, shifting, angle, blurred in zip(
            scale.tolist(),
            translation.tolist(),
            angles.tolist(),
            blur.tolist(),
            strict=True,
        ):
            applied = []
            if not math.isclose(scaling, 1.0):
                applied.append("scale")
            if any(not math.isclose(value, 0.0) for value in shifting):
                applied.append("translation")
            if not math.isclose(angle, 0.0):
                applied.append("rotation")
            if blurred:
                applied.append("gaussian_blur")
            tags.append(",".join(applied))
        with torch.autocast(device_type=images.device.type, enabled=False):
            if (
                self.rotation_degrees == 0
                and self.translate_ratio == 0
                and self.scale_min == self.scale_max == 1
            ):
                output = images.clone()
            else:
                output = self._warp(
                    images,
                    scale.to(images.device),
                    translation.to(images.device),
                    angles.to(images.device),
                )
            if self.blur_probability:
                axis = torch.arange(-2, 3, device=images.device, dtype=torch.float32)
                kernel = torch.exp(-axis.square() / (2 * 0.5**2))
                kernel = kernel / kernel.sum()
                kernel_2d = (kernel[:, None] * kernel[None, :])[None, None]
                blurred_images = functional.conv2d(output - 1, kernel_2d, padding=2) + 1
                output = torch.where(
                    blur.to(images.device)[:, None, None, None], blurred_images, output
                )
            return output, tags

    def _warp(
        self,
        images: torch.Tensor,
        scale: torch.Tensor,
        translation: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        count, _, height, width = images.shape
        ink = images[:, 0] < (250 / 255)
        columns, rows = ink.any(dim=1), ink.any(dim=2)
        axis = torch.arange(width, device=images.device, dtype=torch.float32)
        x_min = torch.where(columns, axis, width).amin(dim=1) - 0.5
        x_max = torch.where(columns, axis, -1).amax(dim=1) + 0.5
        y_min = torch.where(rows, axis, height).amin(dim=1) - 0.5
        y_max = torch.where(rows, axis, -1).amax(dim=1) + 0.5
        center = (width - 1) / 2
        x = torch.stack((x_min, x_min, x_max, x_max), dim=1) - center
        y = torch.stack((y_min, y_max, y_min, y_max), dim=1) - center
        radians = angles * (math.pi / 180)
        cosine, sine = radians.cos(), radians.sin()
        rotated = torch.stack(
            (
                cosine[:, None] * x + sine[:, None] * y,
                -sine[:, None] * x + cosine[:, None] * y,
            ),
            dim=2,
        )
        limit = width / 2 - self.guard_pixels
        translation = translation.clamp(-limit + 0.5, limit - 0.5)
        shift = translation[:, None, :]
        bounds = torch.where(
            rotated.abs() > 1e-8,
            (limit - rotated.sign() * shift) / rotated.abs().clamp_min(1e-8),
            torch.inf,
        )
        maximum_scale = bounds.amin(dim=(1, 2)) * (1 - 1e-6)
        maximum_scale = torch.where(ink.any(dim=(1, 2)), maximum_scale, torch.inf)
        scale = torch.minimum(scale, maximum_scale).clamp_min(1e-8)
        matrix = torch.zeros((count, 2, 3), device=images.device, dtype=torch.float32)
        matrix[:, 0, 0] = cosine / scale
        matrix[:, 0, 1] = -sine / scale
        matrix[:, 1, 0] = sine / scale
        matrix[:, 1, 1] = cosine / scale
        matrix[:, 0, 2] = (
            -(cosine * translation[:, 0] - sine * translation[:, 1])
            / scale
            * (2 / width)
        )
        matrix[:, 1, 2] = (
            -(sine * translation[:, 0] + cosine * translation[:, 1])
            / scale
            * (2 / height)
        )
        grid = functional.affine_grid(matrix, images.shape, align_corners=False)
        return (
            functional.grid_sample(
                images - 1,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            + 1
        )
