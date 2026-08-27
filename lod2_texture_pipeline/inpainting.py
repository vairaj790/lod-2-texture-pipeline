# -*- coding: utf-8 -*-
"""LaMa inpainting helpers for filling missing wall regions."""

import os
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import (
    BIG_LAMA_CONTEXT_PX,
    BIG_LAMA_DEVICE,
    BIG_LAMA_ENABLE_REFINEMENT,
    BIG_LAMA_GENERATOR_PATH,
    BIG_LAMA_MAX_PIXELS,
    BIG_LAMA_MAX_SIDE_PX,
    BIG_LAMA_REFINEMENT_ITERATIONS,
    BIG_LAMA_REFINEMENT_LEARNING_RATE,
    BIG_LAMA_REFINEMENT_MAX_PIXELS,
    BIG_LAMA_REFINEMENT_MAX_SCALES,
    BIG_LAMA_REFINEMENT_MIN_SIDE,
    LAMA_BACKEND,
    LAMA_COMPOSITE_FEATHER_PX,
    LAMA_ENABLE_HIGH_RES_TILING,
    LAMA_MASK_DILATE_PX,
    LAMA_MAX_TILES,
    LAMA_MIN_HOLE_AREA_PX,
    LAMA_MODEL_PATH,
    LAMA_ONNX_INTRA_OP_THREADS,
    LAMA_TILE_DETAIL_MAX_DELTA,
    LAMA_TILE_DETAIL_SIGMA_PX,
    LAMA_TILE_DETAIL_STRENGTH,
    LAMA_TILE_LOW_FREQUENCY_WEIGHT,
    LAMA_TILE_OVERLAP_PX,
)


def _static_dimension(value, fallback: int) -> int:
    return int(value) if isinstance(value, (int, np.integer)) and value > 0 else fallback


def _resize_mask_conservative(mask_u8: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Resize a binary mask without losing thin positive regions."""
    out_w, out_h = size_wh
    src_h, src_w = mask_u8.shape
    if (src_w, src_h) == (out_w, out_h):
        return ((mask_u8 > 0).astype(np.uint8) * 255)

    if out_w < src_w or out_h < src_h:
        resized = cv2.resize(
            (mask_u8 > 0).astype(np.float32),
            (out_w, out_h),
            interpolation=cv2.INTER_AREA,
        )
        return ((resized > 0.0).astype(np.uint8) * 255)

    return cv2.resize(
        ((mask_u8 > 0).astype(np.uint8) * 255),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    )


class OnnxRuntimeLamaInpainter:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"LaMa ONNX model not found: {model_path}\n"
                "Place the model at LAMA_MODEL_PATH or set ENABLE_LAMA_FILL=False."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for LaMa inpainting. "
                "Install it with: pip install onnxruntime"
            ) from exc

        self.model_path = model_path

        # An explicit positive value avoids pthread affinity warnings on some
        # Linux hosts. Zero leaves ONNX Runtime's faster automatic CPU pool in
        # control, which matters when native-resolution refinement uses tiles.
        sess_options = ort.SessionOptions()
        if int(LAMA_ONNX_INTRA_OP_THREADS) > 0:
            sess_options.intra_op_num_threads = int(LAMA_ONNX_INTRA_OP_THREADS)
        sess_options.inter_op_num_threads = 1

        # Suppress ONNX Runtime graph-cleanup warnings during LaMa model loading.
        sess_options.log_severity_level = 3

        available_providers = set(ort.get_available_providers())
        providers = [
            provider
            for provider in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            )
            if provider in available_providers
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) < 2:
            raise RuntimeError("LaMa ONNX model must have at least two inputs: image and mask.")
        if len(outputs) < 1:
            raise RuntimeError("LaMa ONNX model must have at least one output.")

        image_input = next(
            (
                item for item in inputs
                if len(item.shape) == 4 and item.shape[1] == 3
            ),
            inputs[0],
        )
        mask_input = next(
            (
                item for item in inputs
                if item.name != image_input.name
                and len(item.shape) == 4
                and item.shape[1] == 1
            ),
            inputs[1],
        )

        self.image_input_name = image_input.name
        self.mask_input_name = mask_input.name
        self.output_name = outputs[0].name
        self.input_height = _static_dimension(image_input.shape[-2], 512)
        self.input_width = _static_dimension(image_input.shape[-1], 512)
        self.last_inference_stats = {}

    @staticmethod
    def _validate_inputs(image_bgr: np.ndarray, mask_u8: np.ndarray) -> None:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 uint8")
        if mask_u8 is None or mask_u8.ndim != 2:
            raise ValueError("mask_u8 must be HxW uint8")
        if mask_u8.shape != image_bgr.shape[:2]:
            raise ValueError("mask_u8 must match the image HxW shape")

    def _run_model(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Run one model-sized BGR tile and return uint8 BGR."""
        expected_shape = (self.input_height, self.input_width)
        if image_bgr.shape[:2] != expected_shape or mask_u8.shape != expected_shape:
            raise ValueError(
                "LaMa model tile must have shape "
                f"{expected_shape}, got {image_bgr.shape[:2]}"
            )

        image_blob = (
            image_bgr.astype(np.float32).transpose(2, 0, 1)[None, :, :, :]
            / 255.0
        )
        mask_blob = (mask_u8 > 0).astype(np.float32)[None, None, :, :]

        output = self.session.run(
            [self.output_name],
            {
                self.image_input_name: image_blob,
                self.mask_input_name: mask_blob,
            },
        )[0]

        if output.ndim != 4:
            raise RuntimeError(f"Unexpected LaMa output shape: {output.shape}")
        if output.shape[1] == 3:
            output = output[0].transpose(1, 2, 0)
        elif output.shape[-1] == 3:
            output = output[0]
        else:
            raise RuntimeError(f"Unexpected LaMa output shape: {output.shape}")

        output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
        output_min = float(np.min(output))
        output_max = float(np.max(output))
        if output_max <= 1.5:
            if output_min < -0.1:
                output = (output + 1.0) * 127.5
            else:
                output = output * 255.0

        return np.clip(output, 0, 255).astype(np.uint8)

    def _infer_letterboxed(
        self,
        image_bgr: np.ndarray,
        mask_u8: np.ndarray,
    ) -> np.ndarray:
        """Coarse pass that preserves the source aspect ratio."""
        orig_h, orig_w = image_bgr.shape[:2]
        scale = min(self.input_width / orig_w, self.input_height / orig_h)
        resized_w = max(1, min(self.input_width, int(round(orig_w * scale))))
        resized_h = max(1, min(self.input_height, int(round(orig_h * scale))))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_image = cv2.resize(
            image_bgr,
            (resized_w, resized_h),
            interpolation=interpolation,
        )
        resized_mask = _resize_mask_conservative(
            mask_u8,
            (resized_w, resized_h),
        )

        pad_left = (self.input_width - resized_w) // 2
        pad_right = self.input_width - resized_w - pad_left
        pad_top = (self.input_height - resized_h) // 2
        pad_bottom = self.input_height - resized_h - pad_top
        border_mode = (
            cv2.BORDER_REFLECT_101
            if resized_h > 1 and resized_w > 1
            else cv2.BORDER_REPLICATE
        )
        model_image = cv2.copyMakeBorder(
            resized_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode,
        )
        model_mask = cv2.copyMakeBorder(
            resized_mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        prediction = self._run_model(model_image, model_mask)
        prediction = prediction[
            pad_top:pad_top + resized_h,
            pad_left:pad_left + resized_w,
        ]
        if prediction.shape[:2] != (orig_h, orig_w):
            prediction = cv2.resize(
                prediction,
                (orig_w, orig_h),
                interpolation=cv2.INTER_CUBIC,
            )
        return prediction

    @staticmethod
    def _tile_starts(length: int, tile_length: int, overlap: int):
        if length <= tile_length:
            return [0]
        stride = max(1, tile_length - overlap)
        starts = list(range(0, length - tile_length + 1, stride))
        final_start = length - tile_length
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def _infer_high_resolution_tiles(
        self,
        image_bgr: np.ndarray,
        mask_u8: np.ndarray,
        coarse_prediction: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        """Refine masked pixels with overlapping model-sized native tiles."""
        orig_h, orig_w = image_bgr.shape[:2]
        pad_w = max(0, self.input_width - orig_w)
        pad_h = max(0, self.input_height - orig_h)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        seeded = image_bgr.copy()
        seed_idx = mask_u8 > 0
        seeded[seed_idx] = coarse_prediction[seed_idx]
        border_mode = (
            cv2.BORDER_REFLECT_101
            if orig_h > 1 and orig_w > 1
            else cv2.BORDER_REPLICATE
        )
        padded_image = cv2.copyMakeBorder(
            seeded,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode,
        )
        padded_mask = cv2.copyMakeBorder(
            ((mask_u8 > 0).astype(np.uint8) * 255),
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        padded_h, padded_w = padded_mask.shape
        overlap = max(
            0,
            min(
                int(LAMA_TILE_OVERLAP_PX),
                self.input_width - 1,
                self.input_height - 1,
            ),
        )
        y_starts = self._tile_starts(padded_h, self.input_height, overlap)
        x_starts = self._tile_starts(padded_w, self.input_width, overlap)

        candidates = []
        for y0 in y_starts:
            for x0 in x_starts:
                tile_mask = padded_mask[
                    y0:y0 + self.input_height,
                    x0:x0 + self.input_width,
                ]
                masked_pixels = int(np.count_nonzero(tile_mask))
                if masked_pixels > 0:
                    candidates.append((y0, x0, masked_pixels))

        total_candidates = len(candidates)
        max_tiles = max(0, int(LAMA_MAX_TILES))
        if max_tiles and len(candidates) > max_tiles:
            # Keep the tiles carrying the most missing content. Pixels not
            # reached by the cap retain the aspect-preserving coarse result.
            candidates = sorted(
                candidates,
                key=lambda item: (-item[2], item[0], item[1]),
            )[:max_tiles]

        def cosine_taper(length: int, taper: int) -> np.ndarray:
            window = np.ones(length, dtype=np.float32)
            taper = min(max(0, int(taper)), length // 2)
            if taper > 0:
                ramp = np.sin(
                    np.linspace(0.0, np.pi / 2.0, taper, dtype=np.float32)
                ) ** 2
                window[:taper] = ramp
                window[-taper:] = ramp[::-1]
            return window

        taper_px = max(1, overlap // 2) if overlap > 0 else 0
        win_y = cosine_taper(self.input_height, taper_px)
        win_x = cosine_taper(self.input_width, taper_px)
        tile_weight = np.outer(win_y, win_x)

        padded_coarse = cv2.copyMakeBorder(
            coarse_prediction,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        accum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
        weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
        detail_sigma = max(0.0, float(LAMA_TILE_DETAIL_SIGMA_PX))
        detail_strength = float(np.clip(LAMA_TILE_DETAIL_STRENGTH, 0.0, 1.0))
        detail_max_delta = max(0.0, float(LAMA_TILE_DETAIL_MAX_DELTA))
        low_frequency_weight = float(np.clip(
            LAMA_TILE_LOW_FREQUENCY_WEIGHT,
            0.0,
            1.0,
        ))
        for y0, x0, _masked_pixels in candidates:
            tile_image = padded_image[
                y0:y0 + self.input_height,
                x0:x0 + self.input_width,
            ]
            tile_mask = padded_mask[
                y0:y0 + self.input_height,
                x0:x0 + self.input_width,
            ]
            tile_prediction = self._run_model(tile_image, tile_mask)
            tile_mask_float = (tile_mask > 0).astype(np.float32)
            residual = (
                tile_prediction.astype(np.float32)
                - tile_image.astype(np.float32)
            )
            if detail_sigma > 0.0:
                low_frequency_numerator = cv2.GaussianBlur(
                    residual * tile_mask_float[:, :, None],
                    (0, 0),
                    sigmaX=detail_sigma,
                    sigmaY=detail_sigma,
                    borderType=cv2.BORDER_REFLECT_101,
                )
                low_frequency_denominator = cv2.GaussianBlur(
                    tile_mask_float,
                    (0, 0),
                    sigmaX=detail_sigma,
                    sigmaY=detail_sigma,
                    borderType=cv2.BORDER_REFLECT_101,
                )
                low_frequency = low_frequency_numerator / np.maximum(
                    low_frequency_denominator[:, :, None],
                    1e-3,
                )
                residual = (
                    residual - low_frequency
                    + low_frequency_weight * low_frequency
                )
            if detail_max_delta > 0.0:
                np.clip(
                    residual,
                    -detail_max_delta,
                    detail_max_delta,
                    out=residual,
                )
            residual *= detail_strength
            tile_fill = tile_mask > 0
            weights = tile_weight * tile_fill.astype(np.float32)
            accum[
                y0:y0 + self.input_height,
                x0:x0 + self.input_width,
            ] += residual * weights[:, :, None]
            weight_sum[
                y0:y0 + self.input_height,
                x0:x0 + self.input_width,
            ] += weights

        result = padded_coarse.copy()
        refined = weight_sum > 0
        if np.any(refined):
            result[refined] = np.clip(
                result[refined]
                + accum[refined] / weight_sum[refined, None],
                0,
                255,
            )

        result = np.rint(result[
            pad_top:pad_top + orig_h,
            pad_left:pad_left + orig_w,
        ]).astype(np.uint8)
        return result, len(candidates), total_candidates

    def infer(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        image_bgr : HxWx3 uint8
        mask_u8   : HxW uint8, 255 where fill is needed
        returns   : HxWx3 uint8 BGR
        """
        self._validate_inputs(image_bgr, mask_u8)
        coarse = self._infer_letterboxed(image_bgr, mask_u8)
        tiles_run = 0
        tile_candidates = 0
        if (
            bool(LAMA_ENABLE_HIGH_RES_TILING)
            and (image_bgr.shape[0] > self.input_height
                 or image_bgr.shape[1] > self.input_width)
        ):
            result, tiles_run, tile_candidates = self._infer_high_resolution_tiles(
                image_bgr,
                mask_u8,
                coarse,
            )
        else:
            result = coarse

        # Inference callers may use this method directly, so make its contract
        # preserve every known pixel even though the network predicts a full
        # image tensor.
        known = mask_u8 == 0
        result[known] = image_bgr[known]
        self.last_inference_stats = {
            "coarse_mode": "aspect_preserving_letterbox",
            "tiles_run": int(tiles_run),
            "tile_candidates": int(tile_candidates),
            "model_size": [int(self.input_width), int(self.input_height)],
            "tile_blend_mode": "high_frequency_residual",
            "tile_detail_sigma_px": float(LAMA_TILE_DETAIL_SIGMA_PX),
            "tile_detail_strength": float(LAMA_TILE_DETAIL_STRENGTH),
            "tile_detail_max_delta": float(LAMA_TILE_DETAIL_MAX_DELTA),
            "tile_low_frequency_weight": float(
                LAMA_TILE_LOW_FREQUENCY_WEIGHT
            ),
        }
        return result


_LAMA_INPAINTER = None


class _AutoFallbackLamaInpainter:
    """Use native Big-LaMa unless CUDA memory pressure makes ONNX necessary."""

    def __init__(self, primary):
        self.primary = primary
        self.fallback = None
        self.last_inference_stats = {}

    def infer(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        if self.fallback is not None:
            result = self.fallback.infer(image_bgr, mask_u8)
            self.last_inference_stats = {
                **self.fallback.last_inference_stats,
                "auto_fallback_from": "big_lama",
            }
            return result
        try:
            result = self.primary.infer(image_bgr, mask_u8)
            self.last_inference_stats = dict(self.primary.last_inference_stats)
            return result
        except RuntimeError as exc:
            message = str(exc).lower()
            if (
                "out of memory" not in message
                and "cuda_error_out_of_memory" not in message
            ):
                raise
            warnings.warn(
                "Native Big-LaMa ran out of CUDA memory; switching this run "
                "to the ONNX fallback.",
                RuntimeWarning,
            )
            try:
                import torch

                model = getattr(self.primary, "model", None)
                if model is not None:
                    model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.fallback = OnnxRuntimeLamaInpainter(LAMA_MODEL_PATH)
            result = self.fallback.infer(image_bgr, mask_u8)
            self.last_inference_stats = {
                **self.fallback.last_inference_stats,
                "auto_fallback_from": "big_lama",
                "auto_fallback_reason": "cuda_out_of_memory",
            }
            return result


def get_lama_inpainter():
    global _LAMA_INPAINTER
    if _LAMA_INPAINTER is None:
        backend = str(LAMA_BACKEND).strip().lower().replace("-", "_")
        if backend not in {"auto", "big_lama", "onnx"}:
            raise ValueError(
                "LAMA_BACKEND must be one of: auto, big_lama, onnx"
            )
        prefer_big_lama = backend == "big_lama"
        if backend == "auto" and os.path.isfile(BIG_LAMA_GENERATOR_PATH):
            requested_device = str(BIG_LAMA_DEVICE).strip().lower()
            if requested_device == "cpu":
                prefer_big_lama = True
            else:
                try:
                    import torch

                    prefer_big_lama = torch.cuda.is_available()
                except ImportError:
                    prefer_big_lama = False
        if prefer_big_lama:
            try:
                from .big_lama import BigLamaInpainter

                native_inpainter = BigLamaInpainter(
                    BIG_LAMA_GENERATOR_PATH,
                    device=BIG_LAMA_DEVICE,
                    context_px=BIG_LAMA_CONTEXT_PX,
                    max_side_px=BIG_LAMA_MAX_SIDE_PX,
                    max_pixels=BIG_LAMA_MAX_PIXELS,
                    enable_refinement=BIG_LAMA_ENABLE_REFINEMENT,
                    refinement_iterations=BIG_LAMA_REFINEMENT_ITERATIONS,
                    refinement_learning_rate=BIG_LAMA_REFINEMENT_LEARNING_RATE,
                    refinement_min_side=BIG_LAMA_REFINEMENT_MIN_SIDE,
                    refinement_max_scales=BIG_LAMA_REFINEMENT_MAX_SCALES,
                    refinement_max_pixels=BIG_LAMA_REFINEMENT_MAX_PIXELS,
                )
                _LAMA_INPAINTER = (
                    _AutoFallbackLamaInpainter(native_inpainter)
                    if backend == "auto"
                    else native_inpainter
                )
            except Exception as exc:
                if backend == "big_lama":
                    raise
                warnings.warn(
                    "Native Big-LaMa initialization failed; using the ONNX "
                    f"fallback: {exc}",
                    RuntimeWarning,
                )
        if _LAMA_INPAINTER is None:
            _LAMA_INPAINTER = OnnxRuntimeLamaInpainter(LAMA_MODEL_PATH)
    return _LAMA_INPAINTER


def fill_from_nearest_known_pixels(
    image: np.ndarray,
    known_mask: np.ndarray,
    fallback_color=(180, 180, 180),
) -> np.ndarray:
    """Extend known RGB values spatially without introducing black padding."""
    image = np.asarray(image)
    known_mask = np.asarray(known_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if known_mask.shape != image.shape[:2]:
        raise ValueError("known_mask must match image HxW")

    result = image.copy()
    if np.all(known_mask):
        return result
    if not np.any(known_mask):
        result[:, :] = np.asarray(fallback_color, dtype=result.dtype)
        return result

    # DIST_LABEL_PIXEL assigns each zero (known) pixel a unique label and each
    # unknown pixel the label of its nearest zero. This gives an efficient
    # nearest-neighbour facade extension for multi-megapixel rectified images.
    unknown_u8 = (~known_mask).astype(np.uint8)
    _distance, labels = cv2.distanceTransformWithLabels(
        unknown_u8,
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    maximum_label = int(labels.max())
    lookup = np.empty((maximum_label + 1, 3), dtype=image.dtype)
    lookup[:] = np.asarray(fallback_color, dtype=image.dtype)
    lookup[labels[known_mask]] = image[known_mask]
    unknown = ~known_mask
    result[unknown] = lookup[labels[unknown]]
    return result


def bleed_rgb_into_transparency(
    rgba: np.ndarray,
    radius_px: int,
    valid_rgb_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Copy edge RGB under nearby transparent texels while preserving alpha."""
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be HxWx4")
    radius_px = int(radius_px)
    if radius_px <= 0:
        return rgba.copy()

    alpha_known = rgba[:, :, 3] > 0
    if valid_rgb_mask is not None:
        valid_rgb_mask = np.asarray(valid_rgb_mask, dtype=bool)
        if valid_rgb_mask.shape != rgba.shape[:2]:
            raise ValueError("valid_rgb_mask must match rgba HxW")
        alpha_known &= valid_rgb_mask
    if not np.any(alpha_known):
        return rgba.copy()

    extended_rgb = fill_from_nearest_known_pixels(rgba[:, :, :3], alpha_known)
    distance_to_known = cv2.distanceTransform(
        (~alpha_known).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
    )
    gutter = (~alpha_known) & (distance_to_known <= float(radius_px))
    result = rgba.copy()
    result[gutter, :3] = extended_rgb[gutter]
    return result


def remove_small_mask_components(mask_u8: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 1:
        return mask_u8

    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(mask_u8)
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned[labels == lbl] = 255

    return cleaned


def build_wall_region_mask(height: int, width: int, wall_poly_px: np.ndarray) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    poly = np.round(wall_poly_px).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _composite_weights(
    base_hole_mask: np.ndarray,
    inference_mask: np.ndarray,
    feather_px: int,
) -> np.ndarray:
    base_hole = base_hole_mask > 0
    inference = inference_mask > 0
    weights = np.zeros(base_hole.shape, dtype=np.float32)
    if not np.any(inference):
        return weights

    feather_px = int(feather_px)
    if feather_px <= 0:
        weights[inference] = 1.0
    else:
        inward_distance = cv2.distanceTransform(
            inference.astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_5,
        )
        weights[inference] = np.clip(
            inward_distance[inference] / float(feather_px),
            0.0,
            1.0,
        )

    # Original holes may touch the wall/image boundary and therefore have no
    # outward feathering room. They must still be replaced completely.
    weights[base_hole] = 1.0
    return weights


def lama_fill_rectified_wall(
    ortho_rgba: np.ndarray,
    wall_poly_px: np.ndarray,
    debug_mask_path: Optional[str] = None,
    valid_content_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill missing pixels inside the rectified wall polygon using LaMa.

    When ``valid_content_mask`` is supplied, it is authoritative: pixels
    inside the wall but outside that mask are holes even if the RGBA source is
    still opaque there. This is how the full-image pre-fit semantic mask,
    propagated through rectification and geometric adjustment, removes
    occluders before inpainting without a second segmentation inference.

    Returns:
        filled_rgba, hole_mask_u8
    """
    if ortho_rgba is None or ortho_rgba.ndim != 3 or ortho_rgba.shape[2] != 4:
        raise ValueError("ortho_rgba must be HxWx4 RGBA uint8")

    H, W = ortho_rgba.shape[:2]

    wall_region_mask = build_wall_region_mask(H, W, wall_poly_px)
    alpha = ortho_rgba[:, :, 3]

    if valid_content_mask is None:
        valid_wall_content = (wall_region_mask > 0) & (alpha > 0)
    else:
        valid_content_mask = np.asarray(valid_content_mask, dtype=bool)
        if valid_content_mask.shape != (H, W):
            raise ValueError(
                "valid_content_mask must match the HxW rectified image shape"
            )
        valid_wall_content = (
            (wall_region_mask > 0)
            & (alpha > 0)
            & valid_content_mask
        )

    # Keep the true missing region separate from the larger inference mask.
    # The safety ring gives LaMa clean context across antialiased/segmentation
    # boundaries, but it is only feathered during compositing.
    base_hole_mask = np.zeros((H, W), dtype=np.uint8)
    base_hole_mask[(wall_region_mask > 0) & ~valid_wall_content] = 255
    base_hole_mask = remove_small_mask_components(
        base_hole_mask,
        LAMA_MIN_HOLE_AREA_PX,
    )
    inference_mask = base_hole_mask.copy()

    if LAMA_MASK_DILATE_PX > 0:
        k = 2 * LAMA_MASK_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inference_mask = cv2.dilate(inference_mask, kernel, iterations=1)
        inference_mask[wall_region_mask == 0] = 0

    if inference_mask.max() == 0:
        if debug_mask_path is not None:
            Image.fromarray(inference_mask).save(debug_mask_path)
        return ortho_rgba, inference_mask

    rgb = ortho_rgba[:, :, :3].copy()
    known_context = (
        (wall_region_mask > 0)
        & (alpha > 0)
        & (inference_mask == 0)
    )
    model_rgb = fill_from_nearest_known_pixels(rgb, known_context)

    # In particular, do not make the polygon exterior black. A hole touching
    # the wall boundary otherwise sees unmasked black pixels as authoritative
    # context and LaMa continues them into an opaque black band.
    image_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = get_lama_inpainter().infer(image_bgr, inference_mask)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    filled_rgba = ortho_rgba.copy()
    blend_weights = _composite_weights(
        base_hole_mask,
        inference_mask,
        LAMA_COMPOSITE_FEATHER_PX,
    )
    blend_idx = blend_weights > 0
    weights_3 = blend_weights[blend_idx, None]
    blended = (
        ortho_rgba[blend_idx, :3].astype(np.float32) * (1.0 - weights_3)
        + result_rgb[blend_idx].astype(np.float32) * weights_3
    )
    filled_rgba[blend_idx, :3] = np.clip(
        np.rint(blended),
        0,
        255,
    ).astype(np.uint8)
    filled_rgba[base_hole_mask > 0, 3] = 255

    if debug_mask_path is not None:
        Image.fromarray(inference_mask).save(debug_mask_path)

    return filled_rgba, inference_mask
