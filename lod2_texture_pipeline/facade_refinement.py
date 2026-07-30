# -*- coding: utf-8 -*-
"""Guardrails for post-rectification facade segmentation."""

from typing import Dict

import numpy as np


def constrain_post_rectification_sam_mask(
    sam_mask,
    allowed_wall_mask,
    *,
    minimum_pixels: int,
    minimum_wall_coverage: float,
) -> Dict[str, object]:
    """Clip SAM to the wall and provide a non-empty projection fallback."""
    allowed = np.asarray(allowed_wall_mask, dtype=bool)
    sam = np.asarray(sam_mask, dtype=bool)
    if allowed.ndim != 2:
        raise ValueError("Allowed wall mask must be two-dimensional.")
    if sam.shape != allowed.shape:
        raise ValueError("SAM and allowed wall masks must have the same shape.")

    allowed_pixels = int(allowed.sum())
    clipped = sam & allowed
    clipped_pixels = int(clipped.sum())
    coverage = clipped_pixels / max(allowed_pixels, 1)
    accepted = (
        allowed_pixels > 0
        and clipped_pixels >= max(1, int(minimum_pixels))
        and coverage >= max(0.0, float(minimum_wall_coverage))
    )

    if allowed_pixels <= 0:
        reason = "empty_projection_mask"
    elif clipped_pixels <= 0:
        reason = "empty_sam_mask_inside_projection"
    elif clipped_pixels < max(1, int(minimum_pixels)):
        reason = "insufficient_sam_pixels"
    elif coverage < max(0.0, float(minimum_wall_coverage)):
        reason = "insufficient_wall_coverage"
    else:
        reason = "accepted"

    return {
        "accepted": bool(accepted),
        "reason": reason,
        "raw_sam_pixels": int(sam.sum()),
        "outside_pixels_removed": int((sam & (~allowed)).sum()),
        "allowed_wall_pixels": allowed_pixels,
        "clipped_sam_pixels": clipped_pixels,
        "wall_coverage": float(coverage),
        "clipped_mask": clipped,
        "effective_refinement_mask": clipped if accepted else allowed.copy(),
        "fallback_to_projection_mask": not accepted,
    }
