"""Inference-only native Big-LaMa backend.

This file is a substantially modified adaptation of upstream LaMa components;
see ``THIRD_PARTY_NOTICES.md`` and ``LICENSES/Apache-2.0.txt``.

The generator architecture is adapted from ``advimman/lama`` (Apache-2.0):
https://github.com/advimman/lama/blob/main/saicinpainting/training/modules/ffc.py

Only the feed-forward FFC generator is retained. Training, Hydra, OmegaConf,
and PyTorch Lightning dependencies are intentionally excluded.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        spectral_pos_encoding=False,
        use_se=False,
        se_kwargs=None,
        ffc3d=False,
        fft_norm="ortho",
    ):
        super().__init__()
        if use_se:
            raise ValueError("Big-LaMa inference does not use squeeze-excitation")
        self.groups = groups
        extra_channels = 2 if spectral_pos_encoding else 0
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + extra_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            original_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        fft_dimensions = (-3, -2, -1) if self.ffc3d else (-2, -1)
        transformed = torch.fft.rfftn(x, dim=fft_dimensions, norm=self.fft_norm)
        transformed = torch.stack((transformed.real, transformed.imag), dim=-1)
        transformed = transformed.permute(0, 1, 4, 2, 3).contiguous()
        transformed = transformed.view((batch, -1) + transformed.size()[3:])

        if self.spectral_pos_encoding:
            height, width = transformed.shape[-2:]
            vertical = torch.linspace(
                0, 1, height, device=transformed.device, dtype=transformed.dtype
            )[None, None, :, None].expand(batch, 1, height, width)
            horizontal = torch.linspace(
                0, 1, width, device=transformed.device, dtype=transformed.dtype
            )[None, None, None, :].expand(batch, 1, height, width)
            transformed = torch.cat((vertical, horizontal, transformed), dim=1)

        transformed = self.relu(self.bn(self.conv_layer(transformed)))
        transformed = transformed.view(
            (batch, -1, 2) + transformed.size()[2:]
        ).permute(0, 1, 3, 4, 2).contiguous()
        transformed = torch.complex(transformed[..., 0], transformed[..., 1])
        inverse_shape = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(
            transformed,
            s=inverse_shape,
            dim=fft_dimensions,
            norm=self.fft_norm,
        )

        if self.spatial_scale_factor is not None:
            output = F.interpolate(
                output,
                size=original_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )
        return output


class _SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        enable_lfu=False,
        **fourier_kwargs,
    ):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = (
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            if stride == 2
            else nn.Identity()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = _FourierUnit(
            out_channels // 2,
            out_channels // 2,
            groups,
            **fourier_kwargs,
        )
        if enable_lfu:
            self.lfu = _FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )

    def forward(self, x):
        x = self.conv1(self.downsample(x))
        output = self.fu(x)
        local_frequency = 0
        if self.enable_lfu:
            _batch, channels, height, _width = x.shape
            split_size = height // 2
            parts = torch.cat(
                torch.split(x[:, :channels // 4], split_size, dim=-2),
                dim=1,
            ).contiguous()
            parts = torch.cat(
                torch.split(parts, split_size, dim=-1),
                dim=1,
            ).contiguous()
            local_frequency = self.lfu(parts).repeat(1, 1, 2, 2).contiguous()
        return self.conv2(x + output + local_frequency)


class _FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=False,
        padding_type="reflect",
        gated=False,
        **spectral_kwargs,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("FFC stride must be 1 or 2")
        input_global = int(in_channels * ratio_gin)
        input_local = in_channels - input_global
        output_global = int(out_channels * ratio_gout)
        output_local = out_channels - output_global
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = input_global

        local_layer = nn.Identity if input_local == 0 or output_local == 0 else nn.Conv2d
        self.convl2l = local_layer(
            input_local,
            output_local,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        local_to_global = (
            nn.Identity if input_local == 0 or output_global == 0 else nn.Conv2d
        )
        self.convl2g = local_to_global(
            input_local,
            output_global,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        global_to_local = (
            nn.Identity if input_global == 0 or output_local == 0 else nn.Conv2d
        )
        self.convg2l = global_to_local(
            input_global,
            output_local,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        global_layer = (
            nn.Identity
            if input_global == 0 or output_global == 0
            else _SpectralTransform
        )
        self.convg2g = global_layer(
            input_global,
            output_global,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )

        self.gated = gated
        gate_layer = (
            nn.Identity
            if input_global == 0 or output_local == 0 or not gated
            else nn.Conv2d
        )
        self.gate = gate_layer(in_channels, 2, 1)

    def forward(self, x):
        x_local, x_global = x if type(x) is tuple else (x, 0)
        global_to_local_gate = 1
        local_to_global_gate = 1
        if self.gated:
            parts = [x_local]
            if torch.is_tensor(x_global):
                parts.append(x_global)
            gates = torch.sigmoid(self.gate(torch.cat(parts, dim=1)))
            global_to_local_gate, local_to_global_gate = gates.chunk(2, dim=1)

        output_local = 0
        output_global = 0
        if self.ratio_gout != 1:
            output_local = (
                self.convl2l(x_local)
                + self.convg2l(x_global) * global_to_local_gate
            )
        if self.ratio_gout != 0:
            output_global = (
                self.convl2g(x_local) * local_to_global_gate
                + self.convg2g(x_global)
            )
        return output_local, output_global


class _FFCBatchNormActivation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
        padding_type="reflect",
        enable_lfu=False,
        **kwargs,
    ):
        super().__init__()
        self.ffc = _FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        local_norm = nn.Identity if ratio_gout == 1 else norm_layer
        global_norm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = local_norm(out_channels - global_channels)
        self.bn_g = global_norm(global_channels)
        local_activation = nn.Identity if ratio_gout == 1 else activation_layer
        global_activation = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = local_activation(inplace=True)
        self.act_g = global_activation(inplace=True)

    def forward(self, x):
        local, global_part = self.ffc(x)
        return self.act_l(self.bn_l(local)), self.act_g(self.bn_g(global_part))


class _FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dimension,
        padding_type,
        norm_layer,
        activation_layer=nn.ReLU,
        dilation=1,
        inline=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.conv1 = _FFCBatchNormActivation(
            dimension,
            dimension,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.conv2 = _FFCBatchNormActivation(
            dimension,
            dimension,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.inline = inline

    def forward(self, x):
        if self.inline:
            split = self.conv1.ffc.global_in_num
            local, global_part = x[:, :-split], x[:, -split:]
        else:
            local, global_part = x if type(x) is tuple else (x, 0)
        identity_local, identity_global = local, global_part
        local, global_part = self.conv1((local, global_part))
        local, global_part = self.conv2((local, global_part))
        output = identity_local + local, identity_global + global_part
        return torch.cat(output, dim=1) if self.inline else output


class _ConcatTuple(nn.Module):
    def forward(self, x):
        local, global_part = x
        return local if not torch.is_tensor(global_part) else torch.cat(x, dim=1)


class _FFCResNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        initial = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
        downsample = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
        residual = {"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False}
        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            _FFCBatchNormActivation(
                4,
                64,
                kernel_size=7,
                padding=0,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                **initial,
            ),
        ]
        for index in range(3):
            multiplier = 2 ** index
            kwargs = dict(downsample)
            if index == 2:
                kwargs["ratio_gout"] = residual["ratio_gin"]
            model.append(_FFCBatchNormActivation(
                64 * multiplier,
                64 * multiplier * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                **kwargs,
            ))
        for _ in range(18):
            model.append(_FFCResnetBlock(
                512,
                padding_type="reflect",
                activation_layer=nn.ReLU,
                norm_layer=nn.BatchNorm2d,
                **residual,
            ))
        model.append(_ConcatTuple())
        for index in range(3):
            multiplier = 2 ** (3 - index)
            model.extend([
                nn.ConvTranspose2d(
                    64 * multiplier,
                    64 * multiplier // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(64 * multiplier // 2),
                nn.ReLU(True),
            ])
        model.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Sigmoid(),
        ])
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor):
        return self.model(input_tensor)


def _resize_mask_conservative(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(
        (mask > 0).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return resized > 0.0


def _rectangles_overlap(first, second) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _merge_rectangles(rectangles: List[Tuple[int, int, int, int]]):
    groups: List[Tuple[int, int, int, int]] = []
    for rectangle in rectangles:
        pending = rectangle
        merged = True
        while merged:
            merged = False
            retained = []
            for existing in groups:
                if _rectangles_overlap(pending, existing):
                    pending = (
                        min(pending[0], existing[0]),
                        min(pending[1], existing[1]),
                        max(pending[2], existing[2]),
                        max(pending[3], existing[3]),
                    )
                    merged = True
                else:
                    retained.append(existing)
            groups = retained
        groups.append(pending)
    return sorted(groups, key=lambda item: (item[1], item[0]))


class BigLamaInpainter:
    """Official Big-LaMa generator with bounded native-resolution ROI inference."""

    def __init__(
        self,
        generator_path,
        device="auto",
        context_px=128,
        max_side_px=1280,
        max_pixels=1_000_000,
        enable_refinement=False,
        refinement_iterations=15,
        refinement_learning_rate=0.002,
        refinement_min_side=512,
        refinement_max_scales=2,
        refinement_max_pixels=400_000,
    ):
        generator_path = Path(generator_path)
        if not generator_path.is_file():
            raise FileNotFoundError(f"Big-LaMa generator not found: {generator_path}")
        requested_device = str(device).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BIG_LAMA_DEVICE=cuda but CUDA is unavailable")
        self.device = torch.device(requested_device)
        self.context_px = max(0, int(context_px))
        self.max_side_px = max(0, int(max_side_px))
        self.max_pixels = max(0, int(max_pixels))
        self.enable_refinement = bool(enable_refinement)
        self.refinement_iterations = max(1, int(refinement_iterations))
        self.refinement_learning_rate = float(refinement_learning_rate)
        self.refinement_min_side = max(1, int(refinement_min_side))
        self.refinement_max_scales = max(1, int(refinement_max_scales))
        self.refinement_max_pixels = max(0, int(refinement_max_pixels))

        payload = torch.load(generator_path, map_location="cpu", weights_only=True)
        if payload.get("format") != "big-lama-generator-v1":
            raise RuntimeError("Unsupported Big-LaMa generator state format")
        self.model = _FFCResNetGenerator()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.last_inference_stats = {}
        self._last_refinement_scales = 0

    @staticmethod
    def _pad_tensor_to_modulo(tensor: torch.Tensor, modulo: int = 8):
        height, width = tensor.shape[-2:]
        pad_bottom = (-height) % modulo
        pad_right = (-width) % modulo
        if pad_bottom == 0 and pad_right == 0:
            return tensor
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        return F.pad(tensor, (0, pad_right, 0, pad_bottom), mode=mode)

    @staticmethod
    def _gaussian_blur(tensor: torch.Tensor) -> torch.Tensor:
        coordinates = torch.arange(-2, 3, device=tensor.device, dtype=tensor.dtype)
        kernel_1d = torch.exp(-0.5 * coordinates.square())
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(tensor.shape[1], 1, 5, 5)
        padded = F.pad(tensor, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, kernel, groups=tensor.shape[1])

    @classmethod
    def _pyramid_down(cls, tensor: torch.Tensor, size=None) -> torch.Tensor:
        if size is None:
            size = (tensor.shape[-2] // 2, tensor.shape[-1] // 2)
        return F.interpolate(
            cls._gaussian_blur(tensor),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    @classmethod
    def _pyramid_down_mask(cls, mask: torch.Tensor, size=None) -> torch.Tensor:
        if size is None:
            size = (mask.shape[-2] // 2, mask.shape[-1] // 2)
        resized = F.interpolate(
            cls._gaussian_blur(mask),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return (resized >= 1e-8).to(mask.dtype)

    def _feed_forward_tensor(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = image.shape[-2:]
        image = self._pad_tensor_to_modulo(image)
        mask = self._pad_tensor_to_modulo(mask)
        network_input = torch.cat([image * (1.0 - mask), mask], dim=1)
        prediction = self.model(network_input)
        composite = prediction * mask + image * (1.0 - mask)
        return composite[:, :, :original_height, :original_width]

    def _refine_scale(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        lower_resolution_reference: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = image.shape[-2:]
        image = self._pad_tensor_to_modulo(image)
        mask = (self._pad_tensor_to_modulo(mask) >= 1e-8).to(image.dtype)
        network_input = torch.cat([image * (1.0 - mask), mask], dim=1)
        front = self.model.model[:5]
        rear = self.model.model[5:]
        with torch.no_grad():
            local_feature, global_feature = front(network_input)

        if lower_resolution_reference is None:
            with torch.no_grad():
                prediction = rear((local_feature, global_feature))
            composite = prediction * mask + image * (1.0 - mask)
            return composite[:, :, :original_height, :original_width].detach()

        local_feature = local_feature.detach().requires_grad_(True)
        global_feature = global_feature.detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [local_feature, global_feature],
            lr=self.refinement_learning_rate,
        )
        repeated_mask = mask.repeat(1, 3, 1, 1)
        reference = lower_resolution_reference.detach().to(self.device)
        prediction = None
        for iteration in range(self.refinement_iterations):
            optimizer.zero_grad(set_to_none=True)
            prediction = rear((local_feature, global_feature))
            prediction_down = self._pyramid_down(
                prediction[:, :, :original_height, :original_width],
                size=reference.shape[-2:],
            )
            mask_down = F.interpolate(
                mask[:, :, :original_height, :original_width],
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            mask_down = (mask_down >= 1.0 - 1e-8).to(image.dtype)
            mask_down = 1.0 - F.max_pool2d(
                1.0 - mask_down,
                kernel_size=15,
                stride=1,
                padding=7,
            )
            repeated_mask_down = mask_down.repeat(1, 3, 1, 1)

            losses = []
            known = repeated_mask < 1e-8
            if torch.any(known):
                losses.append(torch.mean(torch.abs(
                    prediction[known] - image[known]
                )))
            reference_region = repeated_mask_down >= 1e-8
            if torch.any(reference_region):
                losses.append(torch.mean(torch.abs(
                    prediction_down[reference_region]
                    - reference[reference_region]
                )))
            if not losses:
                break
            loss = sum(losses)
            if iteration + 1 < self.refinement_iterations:
                loss.backward()
                optimizer.step()

        if prediction is None:
            prediction = rear((local_feature, global_feature))
        composite = prediction * repeated_mask + image * (1.0 - repeated_mask)
        return composite[:, :, :original_height, :original_width].detach()

    def _refine_tensor(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        breadth = min(image.shape[-2:])
        scale_count = min(
            1 + int(round(max(
                0.0,
                np.log2(breadth / float(self.refinement_min_side)),
            ))),
            self.refinement_max_scales,
        )
        images = [image]
        masks = [mask]
        for _ in range(scale_count - 1):
            images.append(self._pyramid_down(images[-1]))
            masks.append(self._pyramid_down_mask(masks[-1]))
        images.reverse()
        masks.reverse()
        result = None
        for scale_image, scale_mask in zip(images, masks):
            result = self._refine_scale(scale_image, scale_mask, result)
        self._last_refinement_scales = scale_count
        return result

    def _infer_crop(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask.astype(np.float32))
        ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        if self.enable_refinement:
            composite = self._refine_tensor(image_tensor, mask_tensor)
        else:
            with torch.inference_mode():
                composite = self._feed_forward_tensor(image_tensor, mask_tensor)
        output_rgb = (
            composite[0, :, :height, :width]
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0, 255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    def _fit_budget(self, width: int, height: int) -> float:
        scale = 1.0
        if self.max_side_px > 0 and max(width, height) > self.max_side_px:
            scale = min(scale, self.max_side_px / max(width, height))
        pixel_budget = self.max_pixels
        if self.enable_refinement and self.refinement_max_pixels > 0:
            pixel_budget = (
                min(pixel_budget, self.refinement_max_pixels)
                if pixel_budget > 0
                else self.refinement_max_pixels
            )
        if pixel_budget > 0 and width * height > pixel_budget:
            scale = min(scale, (pixel_budget / (width * height)) ** 0.5)
        return scale

    def infer(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 uint8")
        if mask_u8 is None or mask_u8.shape != image_bgr.shape[:2]:
            raise ValueError("mask_u8 must match the image HxW shape")
        mask = mask_u8 > 0
        if not np.any(mask):
            self.last_inference_stats = {
                "backend": "big_lama",
                "device": str(self.device),
                "regions_run": 0,
            }
            return image_bgr.copy()

        height, width = mask.shape
        component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        rectangles = []
        for label in range(1, component_count):
            x, y, component_width, component_height, _area = stats[label]
            rectangles.append((
                max(0, int(x) - self.context_px),
                max(0, int(y) - self.context_px),
                min(width, int(x + component_width) + self.context_px),
                min(height, int(y + component_height) + self.context_px),
            ))
        regions = _merge_rectangles(rectangles)

        result = image_bgr.copy()
        region_stats = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for x0, y0, x1, y1 in regions:
            crop_image = image_bgr[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            crop_height, crop_width = crop_mask.shape
            scale = self._fit_budget(crop_width, crop_height)
            inference_image = crop_image
            inference_mask = crop_mask
            if scale < 1.0:
                resized_width = max(8, int(round(crop_width * scale)))
                resized_height = max(8, int(round(crop_height * scale)))
                inference_image = cv2.resize(
                    crop_image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
                inference_mask = _resize_mask_conservative(
                    crop_mask,
                    resized_width,
                    resized_height,
                )
            prediction = self._infer_crop(inference_image, inference_mask)
            if prediction.shape[:2] != (crop_height, crop_width):
                prediction = cv2.resize(
                    prediction,
                    (crop_width, crop_height),
                    interpolation=cv2.INTER_CUBIC,
                )
            destination = result[y0:y1, x0:x1]
            destination[crop_mask] = prediction[crop_mask]
            region_stats.append({
                "bounds_xyxy": [x0, y0, x1, y1],
                "source_size": [crop_width, crop_height],
                "inference_size": [
                    int(inference_image.shape[1]),
                    int(inference_image.shape[0]),
                ],
                "scale": float(scale),
            })

        result[~mask] = image_bgr[~mask]
        peak_memory = 0
        if self.device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(self.device))
        self.last_inference_stats = {
            "backend": "big_lama",
            "device": str(self.device),
            "refinement_enabled": self.enable_refinement,
            "refinement_scales": int(self._last_refinement_scales),
            "refinement_iterations": int(self.refinement_iterations),
            "refinement_max_pixels": int(self.refinement_max_pixels),
            "regions_run": len(regions),
            "regions": region_stats,
            "peak_cuda_memory_bytes": peak_memory,
        }
        return result
