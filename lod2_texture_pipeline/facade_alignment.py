# -*- coding: utf-8 -*-
"""Select the image-space facade alignment used by downstream texturing."""

from typing import Dict, Optional

import numpy as np


VALID_FACADE_ALIGNMENT_MODES = ("wall_only", "depth_global")


def _as_points(points, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] < 3
        or value.shape[1] != 2
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite Nx2 point array.")
    return value


def _apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Depth-global homography must be a finite 3x3 matrix.")
    if abs(float(np.linalg.det(H))) < 1.0e-12:
        raise ValueError("Depth-global homography is singular.")

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    transformed = homogeneous @ H.T
    denominator = transformed[:, 2]
    if np.any(np.abs(denominator) < 1.0e-12):
        raise ValueError("Depth-global homography maps facade points to infinity.")
    result = transformed[:, :2] / denominator[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Depth-global homography produced invalid facade points.")
    return result


def select_facade_alignment(
    *,
    requested_mode: str,
    wall_only_outline_px,
    wall_only_rect_px,
    raw_outline_px,
    raw_rect_px,
    depth_fit_result: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Return the outline and rectification quad that downstream must use.

    An accepted depth fit moves the raw projection of both geometries with one
    whole-model homography. A missing, rejected, or invalid depth fit falls back
    to the wall-only result so texturing can continue safely.
    """
    mode = str(requested_mode).strip().lower()
    if mode not in VALID_FACADE_ALIGNMENT_MODES:
        valid = ", ".join(VALID_FACADE_ALIGNMENT_MODES)
        raise ValueError(f"Unknown FACADE_ALIGNMENT_MODE {requested_mode!r}; use {valid}.")

    wall_outline = _as_points(wall_only_outline_px, "wall-only outline")
    wall_rect = _as_points(wall_only_rect_px, "wall-only rectification quad")
    result = {
        "requested_mode": mode,
        "effective_mode": "wall_only",
        "fallback": False,
        "fallback_reason": None,
        "outline_px": wall_outline.copy(),
        "rect_px": wall_rect.copy(),
        "homography": np.eye(3, dtype=np.float64),
        "depth_fit_accepted": bool(depth_fit_result and depth_fit_result.get("applied")),
    }
    if mode == "wall_only":
        return result

    if not depth_fit_result:
        result.update({"fallback": True, "fallback_reason": "depth_fit_unavailable"})
        return result
    if not bool(depth_fit_result.get("applied")):
        result.update({
            "fallback": True,
            "fallback_reason": f"depth_fit_not_accepted: {depth_fit_result.get('reason', 'unknown')}",
        })
        return result

    try:
        raw_outline = _as_points(raw_outline_px, "raw outline")
        raw_rect = _as_points(raw_rect_px, "raw rectification quad")
        H = np.asarray(depth_fit_result["homography"], dtype=np.float64)
        selected_outline = _apply_homography(raw_outline, H)
        selected_rect = _apply_homography(raw_rect, H)
    except (KeyError, TypeError, ValueError) as exc:
        result.update({
            "fallback": True,
            "fallback_reason": f"invalid_depth_fit: {exc}",
        })
        return result

    result.update({
        "effective_mode": "depth_global",
        "outline_px": selected_outline,
        "rect_px": selected_rect,
        "homography": H,
    })
    return result


def facade_alignment_metadata(selection: Dict[str, object]) -> Dict[str, object]:
    """Convert a selection result into JSON-safe metadata."""
    return {
        "requested_mode": str(selection["requested_mode"]),
        "effective_mode": str(selection["effective_mode"]),
        "fallback": bool(selection.get("fallback", False)),
        "fallback_reason": selection.get("fallback_reason"),
        "depth_fit_accepted": bool(selection.get("depth_fit_accepted", False)),
        "H_raw_projection_to_selected": np.asarray(
            selection.get("homography", np.eye(3)), dtype=np.float64
        ).astype(float).tolist(),
        "selected_outline_px": np.asarray(
            selection.get("outline_px", []), dtype=np.float64
        ).astype(float).tolist(),
        "selected_rectification_quad_px": np.asarray(
            selection.get("rect_px", []), dtype=np.float64
        ).astype(float).tolist(),
    }
