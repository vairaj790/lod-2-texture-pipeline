# -*- coding: utf-8 -*-
"""LaMa inpainting helpers for filling missing wall regions."""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import LAMA_MASK_DILATE_PX, LAMA_MIN_HOLE_AREA_PX, LAMA_MODEL_PATH


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

        # Explicit thread settings avoid noisy pthread affinity warnings on some Linux systems.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        # Suppress ONNX Runtime graph-cleanup warnings during LaMa model loading.
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) < 2:
            raise RuntimeError("LaMa ONNX model must have at least two inputs: image and mask.")
        if len(outputs) < 1:
            raise RuntimeError("LaMa ONNX model must have at least one output.")

        self.image_input_name = inputs[0].name
        self.mask_input_name = inputs[1].name
        self.output_name = outputs[0].name

    def infer(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """
        image_bgr : HxWx3 uint8
        mask_u8   : HxW uint8, 255 where fill is needed
        returns   : HxWx3 uint8 BGR
        """
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 uint8")
        if mask_u8 is None or mask_u8.ndim != 2:
            raise ValueError("mask_u8 must be HxW uint8")

        orig_h, orig_w = image_bgr.shape[:2]

        image_512 = cv2.resize(
            image_bgr,
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        mask_512 = cv2.resize(
            mask_u8,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,
        )

        # Keep the same input scaling as the old OpenCV-DNN implementation:
        # image in [0, 1], mask in {0, 1}.
        image_blob = image_512.transpose(2, 0, 1)[None, :, :, :] / 255.0
        mask_blob = (mask_512 > 0).astype(np.float32)[None, None, :, :]

        output = self.session.run(
            [self.output_name],
            {
                self.image_input_name: image_blob.astype(np.float32),
                self.mask_input_name: mask_blob.astype(np.float32),
            },
        )[0]

        output = output[0].transpose(1, 2, 0)

        # Some LaMa ONNX exports return [0, 1], some return [0, 255].
        if float(np.nanmax(output)) <= 1.5:
            output = output * 255.0

        result = np.clip(output, 0, 255).astype(np.uint8)

        result = cv2.resize(
            result,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return result


_LAMA_INPAINTER = None


def get_lama_inpainter() -> OnnxRuntimeLamaInpainter:
    global _LAMA_INPAINTER
    if _LAMA_INPAINTER is None:
        _LAMA_INPAINTER = OnnxRuntimeLamaInpainter(LAMA_MODEL_PATH)
    return _LAMA_INPAINTER


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
    still opaque there. This is how the bounded SAM result removes occluders
    before inpainting.

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

    # Fill every wall pixel not certified as valid facade content.
    hole_mask = np.zeros((H, W), dtype=np.uint8)
    hole_mask[(wall_region_mask > 0) & ~valid_wall_content] = 255

    hole_mask = remove_small_mask_components(hole_mask, LAMA_MIN_HOLE_AREA_PX)

    if LAMA_MASK_DILATE_PX > 0:
        k = 2 * LAMA_MASK_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hole_mask = cv2.dilate(hole_mask, kernel, iterations=1)
        hole_mask[wall_region_mask == 0] = 0

    if hole_mask.max() == 0:
        if debug_mask_path is not None:
            Image.fromarray(hole_mask).save(debug_mask_path)
        return ortho_rgba, hole_mask

    rgb = ortho_rgba[:, :, :3].copy()

    if np.any(valid_wall_content):
        median_color = np.median(rgb[valid_wall_content], axis=0).astype(np.uint8)
    else:
        median_color = np.array([180, 180, 180], dtype=np.uint8)

    rgb[hole_mask > 0] = median_color
    rgb[wall_region_mask == 0] = 0

    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result_bgr = get_lama_inpainter().infer(image_bgr, hole_mask)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    filled_rgba = ortho_rgba.copy()
    fill_idx = hole_mask > 0

    filled_rgba[fill_idx, :3] = result_rgb[fill_idx]
    filled_rgba[fill_idx, 3] = 255

    if debug_mask_path is not None:
        Image.fromarray(hole_mask).save(debug_mask_path)

    return filled_rgba, hole_mask
