# -*- coding: utf-8 -*-
"""LaMa inpainting helpers for filling missing wall regions."""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import LAMA_MASK_DILATE_PX, LAMA_MIN_HOLE_AREA_PX, LAMA_MODEL_PATH
class OpenCVLamaInpainter:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa ONNX model not found: {model_path}")

        self.model_path = model_path
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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

        image_blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=0.00392,
            size=(512, 512),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )

        mask_blob = cv2.dnn.blobFromImage(
            mask_u8,
            scalefactor=1.0,
            size=(512, 512),
            mean=(0,),
            swapRB=False,
            crop=False
        )
        mask_blob = (mask_blob > 0).astype(np.float32)

        self.net.setInput(image_blob, "image")
        self.net.setInput(mask_blob, "mask")
        output = self.net.forward()[0]   # 3 x H x W

        result = np.transpose(output, (1, 2, 0)).astype(np.uint8)
        result = cv2.resize(
            result,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        return result

_LAMA_INPAINTER = None

def get_lama_inpainter() -> OpenCVLamaInpainter:
    global _LAMA_INPAINTER
    if _LAMA_INPAINTER is None:
        _LAMA_INPAINTER = OpenCVLamaInpainter(LAMA_MODEL_PATH)
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
    debug_mask_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill missing pixels inside the rectified wall polygon using LaMa.
    Returns:
        filled_rgba, hole_mask_u8
    """
    if ortho_rgba is None or ortho_rgba.ndim != 3 or ortho_rgba.shape[2] != 4:
        raise ValueError("ortho_rgba must be HxWx4 RGBA uint8")

    H, W = ortho_rgba.shape[:2]

    wall_region_mask = build_wall_region_mask(H, W, wall_poly_px)
    alpha = ortho_rgba[:, :, 3]

    # fill only transparent pixels inside the wall polygon
    hole_mask = np.zeros((H, W), dtype=np.uint8)
    hole_mask[(wall_region_mask > 0) & (alpha == 0)] = 255

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

    valid_wall_pixels = (wall_region_mask > 0) & (alpha > 0)
    if np.any(valid_wall_pixels):
        median_color = np.median(rgb[valid_wall_pixels], axis=0).astype(np.uint8)
    else:
        median_color = np.array([180, 180, 180], dtype=np.uint8)

    rgb[(wall_region_mask > 0) & (alpha == 0)] = median_color
    rgb[wall_region_mask == 0] = 0

    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result_bgr = get_lama_inpainter().infer(image_bgr, hole_mask)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    filled_rgba = ortho_rgba.copy()
    fill_idx = hole_mask > 0

    filled_rgba[fill_idx, :3] = result_rgb[fill_idx]
    filled_rgba[fill_idx, 3] = 255

    # IMPORTANT:
    # keep the original alpha everywhere else so the full orthorectified
    # segmentation remains visible outside the projected wall polygon too.

    if debug_mask_path is not None:
        Image.fromarray(hole_mask).save(debug_mask_path)

    return filled_rgba, hole_mask
