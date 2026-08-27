# -*- coding: utf-8 -*-
"""Opening-aware residual rectification for one planar facade.

SAM3 window and door masks define independent observed quadrilaterals.  A
single bounded residual homography makes their two line families parallel and
perpendicular to the already rectified wall axes.  Validated left/right wall
edges are soft constraints and are discarded when they disagree with the
opening consensus.  Roof and ground/base geometry is intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares


EPS = 1.0e-9


@dataclass
class OpeningObservation:
    opening_id: str
    role: str
    prompt: str
    sam_score: float
    stability: float
    quality: float
    source_mask: np.ndarray = field(repr=False)
    rectified_mask: np.ndarray = field(repr=False)
    quad: np.ndarray = field(repr=False)
    fit_method: str = "approx_poly_dp"
    mask_area_px: int = 0
    quad_iou: float = 0.0
    rectangularity: float = 0.0
    convexity: float = 0.0
    source_inside_fraction: float = 0.0
    source_bbox_fill: float = 0.0
    axis_error_deg: float = 0.0
    orthogonality_error_deg: float = 0.0
    shape_type: str = "rectangle"

    def json_record(self) -> Dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            "role": self.role,
            "prompt": self.prompt,
            "sam_score": float(self.sam_score),
            "stability": float(self.stability),
            "quality": float(self.quality),
            "fit_method": self.fit_method,
            "mask_area_px": int(self.mask_area_px),
            "quad_iou": float(self.quad_iou),
            "rectangularity": float(self.rectangularity),
            "convexity": float(self.convexity),
            "source_inside_fraction": float(self.source_inside_fraction),
            "source_bbox_fill": float(self.source_bbox_fill),
            "axis_error_deg": float(self.axis_error_deg),
            "orthogonality_error_deg": float(self.orthogonality_error_deg),
            "shape_type": self.shape_type,
            "quad_xy_tl_tr_br_bl": np.asarray(
                self.quad, dtype=np.float64
            ).tolist(),
        }


@dataclass
class SideConstraint:
    side: str
    source_line: np.ndarray
    target_line: np.ndarray
    support_ratio: float
    overlap_ratio: float
    angle_error_deg: float
    base_weight: float
    admitted: bool = False
    rejection_reason: Optional[str] = None
    consensus_angle_difference_deg: Optional[float] = None

    def json_record(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "source_line_xy": np.asarray(
                self.source_line, dtype=np.float64
            ).tolist(),
            "target_line_xy": np.asarray(
                self.target_line, dtype=np.float64
            ).tolist(),
            "support_ratio": float(self.support_ratio),
            "overlap_ratio": float(self.overlap_ratio),
            "angle_error_deg": float(self.angle_error_deg),
            "base_weight": float(self.base_weight),
            "admitted": bool(self.admitted),
            "rejection_reason": self.rejection_reason,
            "consensus_angle_difference_deg": (
                None
                if self.consensus_angle_difference_deg is None
                else float(self.consensus_angle_difference_deg)
            ),
        }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except (TypeError, RuntimeError):
            pass
    return np.asarray(value)


def _sam3_nhw(
    value: Any, name: str, expected_hw: Tuple[int, int]
) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or tuple(array.shape[1:]) != tuple(expected_hw):
        raise ValueError(
            f"{name}: expected N x 1 x H x W or N x H x W at "
            f"{expected_hw}, got {array.shape}"
        )
    return array


def extract_scored_sam3_instances(
    output: Mapping[str, Any],
    *,
    prompt: str,
    role: str,
    expected_hw: Tuple[int, int],
    minimum_score: float,
) -> List[Dict[str, Any]]:
    """Copy scored masks before the processor's mutable state is reused."""
    probabilities = _sam3_nhw(
        output["masks_logits"], "masks_logits", expected_hw
    ).astype(np.float32)
    returned_masks = _sam3_nhw(
        output["masks"], "masks", expected_hw
    ).astype(bool)
    boxes = _to_numpy(output["boxes"]).astype(np.float32).reshape(-1, 4)
    scores = _to_numpy(output["scores"]).astype(np.float32).reshape(-1)
    count = probabilities.shape[0]
    if (
        returned_masks.shape[0] != count
        or boxes.shape != (count, 4)
        or scores.shape != (count,)
    ):
        raise ValueError("SAM3 opening output has inconsistent instance counts.")
    hard_masks = probabilities > 0.5
    if not np.array_equal(hard_masks, returned_masks):
        raise ValueError(
            "SAM3 hard opening masks disagree with resized probabilities."
        )

    height, width = [int(value) for value in expected_hw]
    rows = []
    for index in range(count):
        probability = probabilities[index]
        score = float(scores[index])
        if (
            score < float(minimum_score)
            or not np.isfinite(score)
            or not np.isfinite(probability).all()
        ):
            continue
        mask = hard_masks[index]
        if not mask.any():
            continue
        loose = probability > 0.45
        tight = probability > 0.55
        stability = float(tight.sum() / max(int(loose.sum()), 1))
        raw_box = boxes[index].copy()
        clipped_box = raw_box.copy()
        clipped_box[[0, 2]] = np.clip(
            clipped_box[[0, 2]], 0.0, float(width)
        )
        clipped_box[[1, 3]] = np.clip(
            clipped_box[[1, 3]], 0.0, float(height)
        )
        rows.append({
            "role": str(role),
            "prompt": str(prompt),
            "sam_index": int(index),
            "score": score,
            "stability": stability,
            "box_xyxy_raw": raw_box,
            "box_xyxy_clipped": clipped_box,
            "mask": mask.copy(),
        })
    return rows


def run_opening_sam3_prompts(
    processor,
    image_rgb: Image.Image,
    prompt_library: Mapping[str, Sequence[str]],
    *,
    proposal_threshold: float = 0.20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run all opening prompts from one selected-image embedding."""
    import torch

    image = image_rgb.convert("RGB")
    expected_hw = (int(image.height), int(image.width))
    previous_threshold = float(processor.confidence_threshold)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    prompt_rows = []
    try:
        processor.set_confidence_threshold(float(proposal_threshold))
        with torch.inference_mode():
            state = processor.set_image(image)
        embedding_seconds = float(time.perf_counter() - started)
        for role, prompts in dict(prompt_library).items():
            for prompt in prompts:
                processor.reset_all_prompts(state)
                prompt_started = time.perf_counter()
                with torch.inference_mode():
                    state = processor.set_text_prompt(
                        state=state, prompt=str(prompt)
                    )
                extracted = extract_scored_sam3_instances(
                    state,
                    prompt=str(prompt),
                    role=str(role),
                    expected_hw=expected_hw,
                    minimum_score=float(proposal_threshold),
                )
                rows.extend(extracted)
                prompt_rows.append({
                    "role": str(role),
                    "prompt": str(prompt),
                    "instance_count": int(len(extracted)),
                    "seconds": float(time.perf_counter() - prompt_started),
                })
    finally:
        processor.set_confidence_threshold(previous_threshold)
    return rows, {
        "proposal_threshold": float(proposal_threshold),
        "embedding_seconds": float(locals().get("embedding_seconds", 0.0)),
        "total_seconds": float(time.perf_counter() - started),
        "raw_instance_count": int(len(rows)),
        "prompts": prompt_rows,
        "image_input": "PIL_RGB",
    }


def apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points_xy must end in an x/y coordinate dimension.")
    original_shape = points.shape
    flat = points.reshape(-1, 2)
    homogeneous = np.column_stack(
        [flat, np.ones(len(flat), dtype=np.float64)]
    )
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    denominator = mapped[:, 2]
    if (
        np.any(~np.isfinite(mapped))
        or np.any(np.abs(denominator) < EPS)
    ):
        raise ValueError("Homography maps one or more points to infinity.")
    result = mapped[:, :2] / denominator[:, None]
    return result.reshape(original_shape)


def order_quad_tl_tr_br_bl(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angles)]
    start = int(np.argmin(cyclic[:, 0] + cyclic[:, 1]))
    cyclic = np.roll(cyclic, -start, axis=0)
    if cyclic[1, 0] < cyclic[-1, 0]:
        cyclic = np.vstack([cyclic[0], cyclic[:0:-1]])
    return cyclic


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def _quad_angles(quad: np.ndarray) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float64)
    values = []
    for index in range(4):
        previous = quad[(index - 1) % 4] - quad[index]
        following = quad[(index + 1) % 4] - quad[index]
        denominator = max(
            float(np.linalg.norm(previous) * np.linalg.norm(following)), EPS
        )
        cosine = float(
            np.clip(np.dot(previous, following) / denominator, -1.0, 1.0)
        )
        values.append(math.degrees(math.acos(cosine)))
    return np.asarray(values, dtype=np.float64)


def _filled_quad_mask(shape_hw: Tuple[int, int], quad: np.ndarray) -> np.ndarray:
    mask = np.zeros(tuple(int(value) for value in shape_hw), dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.rint(np.asarray(quad, dtype=np.float64)).astype(np.int32),
        1,
        lineType=cv2.LINE_8,
    )
    return mask > 0


def _quad_side_support_ratios(
    contour_points: np.ndarray, quad: np.ndarray, *, bin_count: int = 12
) -> List[float]:
    points = np.asarray(contour_points, dtype=np.float64).reshape(-1, 2)
    quad = order_quad_tl_tr_br_bl(quad)
    tolerance = max(
        2.0,
        0.025
        * min(max(float(np.ptp(points[:, 0])), 1.0),
              max(float(np.ptp(points[:, 1])), 1.0)),
    )
    ratios = []
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        edge = end - start
        length = max(float(np.linalg.norm(edge)), EPS)
        unit = edge / length
        relative = points - start[None, :]
        along = relative @ unit
        perpendicular = np.abs(
            relative[:, 0] * unit[1] - relative[:, 1] * unit[0]
        )
        near = (
            (perpendicular <= tolerance)
            & (along >= -tolerance)
            & (along <= length + tolerance)
        )
        if not near.any():
            ratios.append(0.0)
            continue
        normalized = np.clip(along[near] / length, 0.0, 1.0 - 1.0e-12)
        bins = np.floor(normalized * int(bin_count)).astype(int)
        ratios.append(float(len(np.unique(bins)) / int(bin_count)))
    return ratios


def fit_unconstrained_quad(
    mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Fit four observed corners without imposing right angles."""
    component = _largest_component(mask)
    info: Dict[str, Any] = {
        "method": None,
        "quad_iou": 0.0,
        "rectangularity": 0.0,
        "convexity": 0.0,
        "mask_area_px": int(component.sum()),
    }
    if not component.any():
        info["reason"] = "empty_mask"
        return None, info
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        info["reason"] = "no_contour"
        return None, info
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if contour_area < 4.0 or hull_area < 4.0:
        info["reason"] = "degenerate_contour"
        return None, info

    perimeter = float(cv2.arcLength(hull, True))
    quad = None
    method = None
    for fraction in np.linspace(0.004, 0.075, 90):
        approximation = cv2.approxPolyDP(
            hull, float(fraction) * perimeter, True
        )
        if len(approximation) != 4:
            continue
        proposed = approximation[:, 0, :].astype(np.float64)
        if not cv2.isContourConvex(
            proposed.astype(np.float32).reshape(-1, 1, 2)
        ):
            continue
        angles = _quad_angles(order_quad_tl_tr_br_bl(proposed))
        if float(angles.min()) < 35.0 or float(angles.max()) > 145.0:
            continue
        quad = proposed
        method = "approx_poly_dp"
        break
    if quad is None:
        # This fallback is useful for a diagnostic box, but callers reject it
        # as orientation evidence because it manufactures right angles.
        quad = cv2.boxPoints(
            cv2.minAreaRect(contour.astype(np.float32))
        ).astype(np.float64)
        method = "min_area_rect_fallback"
    quad = order_quad_tl_tr_br_bl(quad)
    quad_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if quad_area < 4.0:
        info["reason"] = "degenerate_quad"
        return None, info
    quad_mask = _filled_quad_mask(component.shape, quad)
    intersection = int((component & quad_mask).sum())
    union = int((component | quad_mask).sum())
    rectangle = cv2.minAreaRect(contour.astype(np.float32))
    rectangle_area = max(float(rectangle[1][0] * rectangle[1][1]), EPS)
    side_support = _quad_side_support_ratios(contour[:, 0, :], quad)
    info.update({
        "method": method,
        "quad_iou": float(intersection / max(union, 1)),
        "rectangularity": float(contour_area / rectangle_area),
        "convexity": float(contour_area / max(hull_area, EPS)),
        "quad_area_px": float(quad_area),
        "side_support_ratios": side_support,
        "minimum_side_support": float(min(side_support)),
    })
    return quad, info


def _axis_angle_errors(
    quad: np.ndarray,
) -> Tuple[List[float], List[float]]:
    quad = order_quad_tl_tr_br_bl(quad)
    horizontal_pairs = ((quad[0], quad[1]), (quad[3], quad[2]))
    vertical_pairs = ((quad[0], quad[3]), (quad[1], quad[2]))
    horizontal = []
    for start, end in horizontal_pairs:
        delta = end - start
        horizontal.append(
            math.degrees(
                math.atan2(abs(float(delta[1])), max(abs(float(delta[0])), EPS))
            )
        )
    vertical = []
    for start, end in vertical_pairs:
        delta = end - start
        vertical.append(
            math.degrees(
                math.atan2(abs(float(delta[0])), max(abs(float(delta[1])), EPS))
            )
        )
    return horizontal, vertical


def opening_geometry_metrics(quad: np.ndarray) -> Dict[str, float]:
    horizontal, vertical = _axis_angle_errors(quad)
    angles = _quad_angles(order_quad_tl_tr_br_bl(quad))
    axis = horizontal + vertical
    return {
        "horizontal_error_mean_deg": float(np.mean(horizontal)),
        "horizontal_error_max_deg": float(np.max(horizontal)),
        "vertical_error_mean_deg": float(np.mean(vertical)),
        "vertical_error_max_deg": float(np.max(vertical)),
        "axis_error_mean_deg": float(np.mean(axis)),
        "axis_error_max_deg": float(np.max(axis)),
        "orthogonality_error_mean_deg": float(
            np.mean(np.abs(angles - 90.0))
        ),
        "orthogonality_error_max_deg": float(
            np.max(np.abs(angles - 90.0))
        ),
    }


def signed_vertical_angle_deg(line: np.ndarray) -> float:
    start, end = np.asarray(line, dtype=np.float64).reshape(2, 2)
    delta = end - start
    if delta[1] < 0:
        delta = -delta
    return math.degrees(
        math.atan2(float(delta[0]), max(abs(float(delta[1])), EPS))
    )


def split_and_filter_source_instances(
    raw_rows: Sequence[Dict[str, Any]],
    wall_mask: np.ndarray,
    *,
    exclusion_mask: Optional[np.ndarray] = None,
    association_dilation_px: int = 5,
    minimum_sam_score: float = 0.25,
    minimum_stability: float = 0.78,
    maximum_exclusion_fraction: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    wall = np.asarray(wall_mask, dtype=bool)
    exclusion = np.zeros_like(wall, dtype=bool)
    if exclusion_mask is not None:
        candidate_exclusion = np.asarray(exclusion_mask, dtype=bool)
        if candidate_exclusion.shape != wall.shape:
            raise ValueError("Opening exclusion mask must match the wall mask.")
        exclusion = candidate_exclusion
    radius = max(0, int(association_dilation_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    gate = cv2.dilate(wall.astype(np.uint8), kernel, iterations=1) > 0
    wall_area = max(int(wall.sum()), 1)
    minimum_area = max(80, int(round(0.0005 * wall_area)))
    candidates = []
    rejected = []
    for row in raw_rows:
        raw = np.asarray(row["mask"], dtype=bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw.astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            component = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            inside = component & gate
            area = int(inside.sum())
            inside_fraction = float(area / max(component_area, 1))
            excluded_fraction = float(
                (inside & exclusion).sum() / max(area, 1)
            )
            ys, xs = np.where(inside)
            role = str(row["role"])
            maximum_fraction = (
                0.35
                if role == "door" or str(row["prompt"]) == "shop window"
                else 0.15
            )
            reason = None
            fill = 0.0
            if float(row["score"]) < float(minimum_sam_score):
                reason = "low_sam_score"
            elif float(row["stability"]) < float(minimum_stability):
                reason = "unstable_probability_boundary"
            elif inside_fraction < 0.75:
                reason = "mostly_outside_wall"
            elif excluded_fraction > float(maximum_exclusion_fraction):
                reason = "overlaps_foreground_or_exclusion"
            elif area < minimum_area:
                reason = "too_small"
            elif area > maximum_fraction * wall_area:
                reason = "too_large_for_opening"
            elif len(xs) == 0:
                reason = "empty_after_wall_gate"
            else:
                width = int(xs.max() - xs.min() + 1)
                height = int(ys.max() - ys.min() + 1)
                fill = float(area / max(width * height, 1))
                if min(width, height) < 8:
                    reason = "short_bbox_side"
                elif fill < 0.20:
                    reason = "sparse_component"
                elif (
                    component[0, :].any()
                    or component[-1, :].any()
                    or component[:, 0].any()
                    or component[:, -1].any()
                ):
                    reason = "opening_truncated_by_source_border"
            record = {
                **row,
                "component_index": int(label),
                # The wall gate is for association only.  Keeping the complete
                # SAM component avoids manufacturing a slanted/truncated side
                # at the projected wall boundary before the quad is fitted.
                "mask": component.copy(),
                "area_px": area,
                "inside_fraction": inside_fraction,
                "excluded_fraction": excluded_fraction,
                "bbox_fill": fill,
            }
            if reason is not None:
                record["rejection_reason"] = reason
                rejected.append(record)
                continue
            record["quality"] = float(
                float(row["score"])
                * math.sqrt(float(row["stability"]))
                * inside_fraction
                * math.sqrt(fill)
            )
            candidates.append(record)

    kept = []
    for candidate in sorted(
        candidates, key=lambda item: item["quality"], reverse=True
    ):
        duplicate = None
        for prior in kept:
            intersection = int((candidate["mask"] & prior["mask"]).sum())
            overlap = float(
                intersection
                / max(min(candidate["area_px"], prior["area_px"]), 1)
            )
            if overlap > 0.60:
                duplicate = prior["candidate_id"]
                break
        if duplicate is not None:
            rejected.append({
                **candidate,
                "rejection_reason": "cross_prompt_duplicate",
                "duplicate_of": duplicate,
            })
            continue
        candidate = dict(candidate)
        candidate["candidate_id"] = f"candidate_{len(kept):02d}"
        kept.append(candidate)
    return kept, rejected


def build_opening_observations(
    candidates: Sequence[Dict[str, Any]],
    source_to_rectified_h: np.ndarray,
    output_shape_hw: Tuple[int, int],
    wall_mask_rectified: np.ndarray,
) -> Tuple[List[OpeningObservation], List[Dict[str, Any]]]:
    height, width = [int(value) for value in output_shape_hw]
    wall = np.asarray(wall_mask_rectified, dtype=bool)
    wall_gate = cv2.dilate(
        wall.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) > 0
    observations = []
    rejected = []
    for index, candidate in enumerate(candidates):
        rectified = cv2.warpPerspective(
            np.asarray(candidate["mask"], dtype=np.uint8),
            np.asarray(source_to_rectified_h, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        rectified_wall_fraction = float(
            (rectified & wall_gate).sum() / max(int(rectified.sum()), 1)
        )
        rectified_touches_border = bool(
            rectified[0, :].any()
            or rectified[-1, :].any()
            or rectified[:, 0].any()
            or rectified[:, -1].any()
        )
        rectified = cv2.morphologyEx(
            rectified.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ) > 0
        quad, fit = fit_unconstrained_quad(rectified)
        reason = None
        if quad is None:
            reason = fit.get("reason", "quad_fit_failed")
        elif rectified_touches_border:
            reason = "opening_truncated_by_rectified_canvas"
        elif rectified_wall_fraction < 0.75:
            reason = "mostly_outside_rectified_wall"
        elif fit.get("method") != "approx_poly_dp":
            reason = "right_angle_fallback_not_orientation_evidence"
        elif float(fit.get("quad_iou", 0.0)) < 0.47:
            reason = "low_quad_mask_iou"
        elif float(fit.get("rectangularity", 0.0)) < 0.42:
            reason = "low_rectangularity"
        elif float(fit.get("convexity", 0.0)) < 0.68:
            reason = "low_convexity"
        elif float(fit.get("minimum_side_support", 0.0)) < 0.35:
            reason = "insufficient_four_side_boundary_support"
        else:
            geometry = opening_geometry_metrics(quad)
            if geometry["axis_error_max_deg"] > 25.0:
                reason = "quad_not_near_wall_axes"
            elif geometry["orthogonality_error_max_deg"] > 35.0:
                reason = "quad_corner_angles_implausible"
        record = {
            "candidate_id": candidate.get(
                "candidate_id", f"candidate_{index:02d}"
            ),
            "role": str(candidate["role"]),
            "prompt": str(candidate["prompt"]),
            "sam_score": float(candidate["score"]),
            "stability": float(candidate["stability"]),
            "quality": float(candidate["quality"]),
            "rectified_wall_fraction": rectified_wall_fraction,
            "fit": fit,
        }
        if reason is not None:
            record["rejection_reason"] = reason
            rejected.append(record)
            continue
        quad = order_quad_tl_tr_br_bl(quad)
        geometry = opening_geometry_metrics(quad)
        width_estimate = 0.5 * (
            np.linalg.norm(quad[1] - quad[0])
            + np.linalg.norm(quad[2] - quad[3])
        )
        height_estimate = 0.5 * (
            np.linalg.norm(quad[3] - quad[0])
            + np.linalg.norm(quad[2] - quad[1])
        )
        aspect = max(width_estimate, height_estimate) / max(
            min(width_estimate, height_estimate), EPS
        )
        observations.append(OpeningObservation(
            opening_id=f"opening_{len(observations):02d}",
            role=str(candidate["role"]),
            prompt=str(candidate["prompt"]),
            sam_score=float(candidate["score"]),
            stability=float(candidate["stability"]),
            quality=float(candidate["quality"]),
            source_mask=np.asarray(candidate["mask"], dtype=bool),
            rectified_mask=rectified,
            quad=quad,
            fit_method=str(fit["method"]),
            mask_area_px=int(fit["mask_area_px"]),
            quad_iou=float(fit["quad_iou"]),
            rectangularity=float(fit["rectangularity"]),
            convexity=float(fit["convexity"]),
            source_inside_fraction=float(candidate["inside_fraction"]),
            source_bbox_fill=float(candidate["bbox_fill"]),
            axis_error_deg=float(geometry["axis_error_mean_deg"]),
            orthogonality_error_deg=float(
                geometry["orthogonality_error_mean_deg"]
            ),
            shape_type="square" if aspect <= 1.15 else "rectangle",
        ))
    return observations, rejected


def robust_opening_consensus(
    observations: Sequence[OpeningObservation],
    *,
    maximum_residual_deg: float = 8.0,
) -> Tuple[List[OpeningObservation], List[Dict[str, Any]], Dict[str, float]]:
    if not observations:
        return [], [], {
            "median_signed_vertical_deg": 0.0,
            "mad_deg": 0.0,
            "threshold_deg": float(maximum_residual_deg),
        }
    values = np.asarray([
        0.5
        * (
            signed_vertical_angle_deg(
                np.vstack([observation.quad[0], observation.quad[3]])
            )
            + signed_vertical_angle_deg(
                np.vstack([observation.quad[1], observation.quad[2]])
            )
        )
        for observation in observations
    ])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(float(maximum_residual_deg), 3.5 * 1.4826 * mad)
    kept = []
    rejected = []
    for observation, value in zip(observations, values):
        residual = float(abs(value - median))
        if residual <= threshold:
            kept.append(observation)
        else:
            rejected.append({
                **observation.json_record(),
                "rejection_reason": "vertical_family_consensus_outlier",
                "signed_vertical_angle_deg": float(value),
                "consensus_residual_deg": residual,
            })
    return kept, rejected, {
        "median_signed_vertical_deg": median,
        "mad_deg": mad,
        "threshold_deg": float(threshold),
    }


def side_constraints_from_selected_edges(
    selected_edges: Sequence[Mapping[str, Any]],
) -> List[SideConstraint]:
    constraints = []
    for edge in selected_edges:
        if bool(edge.get("is_bottom")) or edge.get("side") not in {
            "left", "right"
        }:
            continue
        selected = edge.get("selected_line")
        if selected is None:
            continue
        source = np.asarray(selected, dtype=np.float64).reshape(2, 2)
        target = np.vstack([
            np.asarray(edge["target_p0"], dtype=np.float64),
            np.asarray(edge["target_p1"], dtype=np.float64),
        ])
        info = dict(edge.get("info") or {})
        target_length = max(float(np.linalg.norm(target[1] - target[0])), EPS)
        support = float(
            info.get(
                "target_union_coverage_ratio",
                np.linalg.norm(source[1] - source[0]) / target_length,
            )
        )
        overlap = float(info.get("best_overlap_ratio") or 0.0)
        angle_error = float(info.get("best_angle_diff_deg") or 180.0)
        constraints.append(SideConstraint(
            side=str(edge["side"]),
            source_line=source,
            target_line=target,
            support_ratio=support,
            overlap_ratio=overlap,
            angle_error_deg=angle_error,
            base_weight=float(
                np.clip(support, 0.0, 1.0)
                * np.clip(overlap, 0.0, 1.0)
            ),
        ))
    return constraints


def admit_consistent_side_constraints(
    constraints: Sequence[SideConstraint],
    opening_consensus: Mapping[str, float],
    *,
    maximum_consensus_difference_deg: float = 5.0,
) -> List[SideConstraint]:
    median = float(opening_consensus.get("median_signed_vertical_deg", 0.0))
    output = []
    for original in constraints:
        constraint = SideConstraint(
            side=original.side,
            source_line=original.source_line.copy(),
            target_line=original.target_line.copy(),
            support_ratio=original.support_ratio,
            overlap_ratio=original.overlap_ratio,
            angle_error_deg=original.angle_error_deg,
            base_weight=original.base_weight,
        )
        difference = float(
            abs(signed_vertical_angle_deg(constraint.source_line) - median)
        )
        constraint.consensus_angle_difference_deg = difference
        if difference > float(maximum_consensus_difference_deg):
            constraint.rejection_reason = "conflicts_with_opening_consensus"
        else:
            constraint.admitted = True
        output.append(constraint)
    return output


def summarize_opening_geometry(
    observations: Sequence[OpeningObservation],
    homography: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    transform = np.eye(3) if homography is None else homography
    rows = []
    for observation in observations:
        quad = apply_homography(observation.quad, transform)
        rows.append({
            "opening_id": observation.opening_id,
            **opening_geometry_metrics(quad),
        })
    if not rows:
        return {
            "opening_count": 0,
            "median_axis_error_deg": None,
            "p90_axis_error_deg": None,
            "median_orthogonality_error_deg": None,
            "p90_orthogonality_error_deg": None,
            "per_opening": [],
        }
    axis = np.asarray([row["axis_error_mean_deg"] for row in rows])
    orthogonal = np.asarray([
        row["orthogonality_error_mean_deg"] for row in rows
    ])
    return {
        "opening_count": int(len(rows)),
        "median_axis_error_deg": float(np.median(axis)),
        "p90_axis_error_deg": float(np.percentile(axis, 90)),
        "median_orthogonality_error_deg": float(np.median(orthogonal)),
        "p90_orthogonality_error_deg": float(
            np.percentile(orthogonal, 90)
        ),
        "per_opening": rows,
    }


def _normalization_matrices(
    width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    to_normalized = np.array(
        [[1.0 / width, 0.0, 0.0],
         [0.0, 1.0 / height, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    to_pixels = np.array(
        [[width, 0.0, 0.0],
         [0.0, height, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return to_normalized, to_pixels


def _normalized_homography(
    parameters: np.ndarray, *, projective: bool
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    affine = np.array([
        [1.0 + parameters[0], parameters[1], parameters[2]],
        [parameters[3], 1.0 + parameters[4], parameters[5]],
        [0.0, 0.0, 1.0],
    ])
    if projective:
        affine[2, :2] = parameters[6:8]
    return affine


def _opening_span_is_projective(
    observations: Sequence[OpeningObservation]
) -> bool:
    if len(observations) < 4:
        return False
    centers = np.asarray([
        observation.quad.mean(axis=0) for observation in observations
    ])
    widths = np.asarray([
        np.ptp(observation.quad[:, 0]) for observation in observations
    ])
    heights = np.asarray([
        np.ptp(observation.quad[:, 1]) for observation in observations
    ])
    return bool(
        np.ptp(centers[:, 0]) > 1.5 * max(float(np.median(widths)), EPS)
        and np.ptp(centers[:, 1]) > 1.5 * max(float(np.median(heights)), EPS)
    )


def _side_samples(
    constraint: SideConstraint, count: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.linspace(0.0, 1.0, int(count))
    source = constraint.source_line[0][None, :] + values[:, None] * (
        constraint.source_line[1] - constraint.source_line[0]
    )[None, :]
    target = constraint.target_line
    direction = target[1] - target[0]
    direction /= max(float(np.linalg.norm(direction)), EPS)
    along = (source - target[0][None, :]) @ direction
    projected = target[0][None, :] + along[:, None] * direction[None, :]
    return source, projected


def _side_metrics(
    constraints: Sequence[SideConstraint], homography: np.ndarray
) -> Dict[str, Any]:
    rows = []
    for constraint in constraints:
        if not constraint.admitted:
            continue
        source, target = _side_samples(constraint, 9)
        transformed = apply_homography(source, homography)
        distances = np.linalg.norm(transformed - target, axis=1)
        transformed_line = apply_homography(
            constraint.source_line, homography
        )
        source_angle = signed_vertical_angle_deg(transformed_line)
        target_angle = signed_vertical_angle_deg(constraint.target_line)
        rows.append({
            "side": constraint.side,
            "mean_distance_px": float(np.mean(distances)),
            "max_distance_px": float(np.max(distances)),
            "angle_error_deg": float(abs(source_angle - target_angle)),
        })
    return {
        "constraint_count": int(len(rows)),
        "mean_distance_px": (
            None
            if not rows
            else float(np.mean([row["mean_distance_px"] for row in rows]))
        ),
        "per_side": rows,
    }


def _homography_safety(
    homography: np.ndarray,
    width: int,
    height: int,
    wall_polygon: np.ndarray,
) -> Dict[str, float]:
    polygon = np.asarray(wall_polygon, dtype=np.float64)
    xs = np.linspace(float(np.min(polygon[:, 0])), float(np.max(polygon[:, 0])), 9)
    ys = np.linspace(float(np.min(polygon[:, 1])), float(np.max(polygon[:, 1])), 9)
    samples = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    transformed = apply_homography(samples, homography)
    displacement = np.linalg.norm(transformed - samples, axis=1)
    matrix = np.asarray(homography, dtype=np.float64)
    denominator = (
        samples[:, 0] * matrix[2, 0]
        + samples[:, 1] * matrix[2, 1]
        + matrix[2, 2]
    )
    determinants = []
    conditions = []
    for point in samples:
        base = apply_homography(point[None, :], matrix)[0]
        dx = apply_homography((point + [1.0, 0.0])[None, :], matrix)[0] - base
        dy = apply_homography((point + [0.0, 1.0])[None, :], matrix)[0] - base
        jacobian = np.column_stack([dx, dy])
        determinants.append(float(np.linalg.det(jacobian)))
        conditions.append(float(np.linalg.cond(jacobian)))
    return {
        "maximum_displacement_px": float(np.max(displacement)),
        "median_displacement_px": float(np.median(displacement)),
        "minimum_homogeneous_denominator": float(np.min(denominator)),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_condition": float(np.max(conditions)),
    }


def estimate_residual_homography(
    observations: Sequence[OpeningObservation],
    constraints: Sequence[SideConstraint],
    image_shape_hw: Tuple[int, int],
    *,
    wall_polygon: np.ndarray,
    allow_projective: bool = True,
    minimum_openings: int = 3,
    maximum_final_p90_axis_error_deg: float = 3.0,
    maximum_final_p90_orthogonality_error_deg: float = 5.0,
    maximum_final_per_opening_axis_error_deg: float = 4.0,
    maximum_final_per_opening_orthogonality_error_deg: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    height, width = [int(value) for value in image_shape_hw]
    identity = np.eye(3, dtype=np.float64)
    before = summarize_opening_geometry(observations, identity)
    projective = bool(
        allow_projective and _opening_span_is_projective(observations)
    )
    parameter_count = 8 if projective else 6
    info: Dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "model": "projective_homography" if projective else "affine",
        "opening_count": int(len(observations)),
        "admitted_side_count": int(sum(item.admitted for item in constraints)),
        "before": before,
    }
    if len(observations) < int(minimum_openings):
        info.update({
            "reason": "insufficient_opening_constraints",
            "homography": identity.tolist(),
            "after": before,
        })
        return identity, info

    to_normalized, to_pixels = _normalization_matrices(width, height)
    diagonal = max(math.hypot(width, height), EPS)
    isotropic_scale = np.array([width / diagonal, height / diagonal])
    opening_quads = [
        apply_homography(observation.quad, to_normalized)
        for observation in observations
    ]
    side_samples = []
    for constraint in constraints:
        if not constraint.admitted:
            continue
        source, target = _side_samples(constraint)
        side_samples.append((
            apply_homography(source, to_normalized),
            apply_homography(target, to_normalized),
            float(constraint.base_weight),
        ))

    def residuals(parameters: np.ndarray) -> np.ndarray:
        normalized_h = _normalized_homography(
            parameters, projective=projective
        )
        values = []
        for observation, source_quad in zip(observations, opening_quads):
            transformed = apply_homography(source_quad, normalized_h)
            isotropic = transformed * isotropic_scale[None, :]
            horizontal = (
                isotropic[1] - isotropic[0],
                isotropic[2] - isotropic[3],
            )
            vertical = (
                isotropic[3] - isotropic[0],
                isotropic[2] - isotropic[1],
            )
            weight = math.sqrt(float(np.clip(observation.quality, 0.05, 1.0)))
            angular = [
                edge[1] / max(float(np.linalg.norm(edge)), EPS)
                for edge in horizontal
            ] + [
                edge[0] / max(float(np.linalg.norm(edge)), EPS)
                for edge in vertical
            ]
            values.extend((0.5 * weight * np.asarray(angular)).tolist())
            if observation.shape_type == "square":
                horizontal_length = 0.5 * sum(
                    float(np.linalg.norm(edge)) for edge in horizontal
                )
                vertical_length = 0.5 * sum(
                    float(np.linalg.norm(edge)) for edge in vertical
                )
                values.append(
                    0.12
                    * weight
                    * math.log(
                        max(horizontal_length, EPS)
                        / max(vertical_length, EPS)
                    )
                )
            center_shift = transformed.mean(axis=0) - source_quad.mean(axis=0)
            values.extend((math.sqrt(0.05) * center_shift).tolist())
        for source, target, evidence_weight in side_samples:
            transformed = apply_homography(source, normalized_h)
            weight = math.sqrt(max(0.45 * evidence_weight, 0.0))
            values.extend((weight * (transformed - target).reshape(-1)).tolist())
        regularization = np.ones(parameter_count)
        if projective:
            regularization[6:] = 1.8
        values.extend(
            (math.sqrt(0.15) * regularization * parameters).tolist()
        )
        return np.asarray(values, dtype=np.float64)

    lower_affine = np.array([-0.12, -0.14, -0.05, -0.14, -0.12, -0.05])
    upper_affine = np.array([0.12, 0.14, 0.05, 0.14, 0.12, 0.05])
    if projective:
        lower = np.concatenate([lower_affine, [-0.12, -0.12]])
        upper = np.concatenate([upper_affine, [0.12, 0.12]])
    else:
        lower, upper = lower_affine, upper_affine
    info["attempted"] = True
    solution = least_squares(
        residuals,
        np.zeros(parameter_count, dtype=np.float64),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=3000,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    normalized_h = _normalized_homography(
        solution.x, projective=projective
    )
    homography = to_pixels @ normalized_h @ to_normalized
    homography /= homography[2, 2]
    after = summarize_opening_geometry(observations, homography)
    sides_before = _side_metrics(constraints, identity)
    sides_after = _side_metrics(constraints, homography)
    safety = _homography_safety(
        homography, width, height, wall_polygon
    )
    before_median = float(before["median_axis_error_deg"] or 0.0)
    after_median = float(after["median_axis_error_deg"] or 0.0)
    before_p90 = float(before["p90_axis_error_deg"] or 0.0)
    after_p90 = float(after["p90_axis_error_deg"] or 0.0)
    side_before = sides_before["mean_distance_px"]
    side_after = sides_after["mean_distance_px"]
    opening_improved = bool(
        after_median < before_median - 0.05
        or after_p90 < before_p90 - 0.05
    )
    side_improved = bool(
        side_before is not None
        and side_after is not None
        and float(side_after) < float(side_before) - 1.0
    )
    maximum_displacement = 0.10 * math.hypot(width, height)
    reason = None
    if not solution.success:
        reason = "optimizer_failed"
    elif not np.isfinite(homography).all():
        reason = "non_finite_homography"
    elif safety["minimum_homogeneous_denominator"] <= 0.35:
        reason = "unsafe_projective_denominator"
    elif safety["minimum_jacobian_determinant"] <= 0.05:
        reason = "foldover_or_near_singular_jacobian"
    elif safety["maximum_jacobian_condition"] > 8.0:
        reason = "excessive_local_anisotropy"
    elif safety["maximum_displacement_px"] > maximum_displacement:
        reason = "excessive_canvas_displacement"
    elif after_p90 > before_p90 + 0.10:
        reason = "opening_p90_regressed"
    elif after_p90 > float(maximum_final_p90_axis_error_deg):
        reason = "final_opening_axis_error_exceeds_limit"
    elif float(
        after.get("p90_orthogonality_error_deg") or 0.0
    ) > float(maximum_final_p90_orthogonality_error_deg):
        reason = "final_opening_orthogonality_error_exceeds_limit"
    elif any(
        float(row["axis_error_mean_deg"])
        > float(maximum_final_per_opening_axis_error_deg)
        for row in after.get("per_opening", [])
    ):
        reason = "individual_opening_axis_error_exceeds_limit"
    elif any(
        float(row["orthogonality_error_mean_deg"])
        > float(maximum_final_per_opening_orthogonality_error_deg)
        for row in after.get("per_opening", [])
    ):
        reason = "individual_opening_orthogonality_error_exceeds_limit"
    elif not opening_improved and not side_improved:
        reason = "no_material_opening_or_side_improvement"
    accepted = reason is None
    selected = homography if accepted else identity
    info.update({
        "accepted": bool(accepted),
        "reason": reason or "accepted",
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "optimizer_cost": float(solution.cost),
        "optimizer_nfev": int(solution.nfev),
        "parameters": solution.x.astype(float).tolist(),
        "candidate_homography": homography.tolist(),
        "homography": selected.tolist(),
        "after": after if accepted else before,
        "candidate_after": after,
        "sides_before": sides_before,
        "sides_after": sides_after if accepted else sides_before,
        "candidate_sides_after": sides_after,
        "safety": safety,
        "maximum_allowed_displacement_px": float(maximum_displacement),
    })
    return selected, info


def _without_masks(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        row = {}
        for key, value in record.items():
            if key == "mask":
                continue
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                row[key] = value.item()
            elif isinstance(value, Mapping):
                row[key] = _without_masks([value])[0]
            else:
                row[key] = value
        output.append(row)
    return output


def estimate_opening_aware_rectification(
    source_rows: Sequence[Dict[str, Any]],
    source_wall_mask: np.ndarray,
    source_to_rectified_h: np.ndarray,
    output_shape_hw: Tuple[int, int],
    wall_mask_rectified: np.ndarray,
    wall_polygon_rectified: np.ndarray,
    selected_side_edges: Sequence[Mapping[str, Any]],
    *,
    source_exclusion_mask: Optional[np.ndarray] = None,
    minimum_sam_score: float = 0.25,
    minimum_stability: float = 0.78,
    minimum_openings: int = 3,
    maximum_side_consensus_deg: float = 5.0,
    allow_projective: bool = True,
    maximum_final_side_angle_deg: float = 2.0,
    maximum_final_side_distance_px: float = 8.0,
    maximum_final_opening_p90_axis_error_deg: float = 3.0,
    maximum_final_opening_p90_orthogonality_error_deg: float = 5.0,
    maximum_final_per_opening_axis_error_deg: float = 4.0,
    maximum_final_per_opening_orthogonality_error_deg: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, Any], List[OpeningObservation]]:
    """Build opening constraints, veto bad sides, and solve one shared warp."""
    identity = np.eye(3, dtype=np.float64)
    candidates, source_rejected = split_and_filter_source_instances(
        source_rows,
        source_wall_mask,
        exclusion_mask=source_exclusion_mask,
        minimum_sam_score=minimum_sam_score,
        minimum_stability=minimum_stability,
    )
    observations, geometry_rejected = build_opening_observations(
        candidates,
        source_to_rectified_h,
        output_shape_hw,
        wall_mask_rectified,
    )
    observations, consensus_rejected, consensus = robust_opening_consensus(
        observations
    )
    raw_constraints = side_constraints_from_selected_edges(selected_side_edges)
    constraints = admit_consistent_side_constraints(
        raw_constraints,
        consensus,
        maximum_consensus_difference_deg=maximum_side_consensus_deg,
    )
    diagnostics: Dict[str, Any] = {
        "enabled": True,
        "raw_sam_instance_count": int(len(source_rows)),
        "source_candidate_count": int(len(candidates)),
        "accepted_opening_count": int(len(observations)),
        "source_rejections": _without_masks(source_rejected),
        "geometry_rejections": _without_masks(geometry_rejected),
        "consensus_rejections": consensus_rejected,
        "opening_consensus": consensus,
        "openings": [observation.json_record() for observation in observations],
        "side_constraints": [item.json_record() for item in constraints],
        "roof_and_base_used": False,
        "shared_axes_only": True,
    }
    if len(observations) < int(minimum_openings):
        diagnostics.update({
            "applied": False,
            "reason": "insufficient_reliable_openings",
            "homography": identity.tolist(),
            "fallback_policy": "hardened_side_only_path_allowed",
        })
        return identity, diagnostics, observations

    joint_h, joint = estimate_residual_homography(
        observations,
        constraints,
        output_shape_hw,
        wall_polygon=wall_polygon_rectified,
        allow_projective=allow_projective,
        minimum_openings=minimum_openings,
        maximum_final_p90_axis_error_deg=(
            maximum_final_opening_p90_axis_error_deg
        ),
        maximum_final_p90_orthogonality_error_deg=(
            maximum_final_opening_p90_orthogonality_error_deg
        ),
        maximum_final_per_opening_axis_error_deg=(
            maximum_final_per_opening_axis_error_deg
        ),
        maximum_final_per_opening_orthogonality_error_deg=(
            maximum_final_per_opening_orthogonality_error_deg
        ),
    )
    admitted = [item for item in constraints if item.admitted]
    final_side_failure = None
    if joint.get("accepted") and admitted:
        final_sides = _side_metrics(admitted, joint_h)
        for row in final_sides["per_side"]:
            if row["angle_error_deg"] > float(maximum_final_side_angle_deg):
                final_side_failure = (
                    f"{row['side']}_final_angle_exceeds_limit"
                )
                break
            if row["max_distance_px"] > float(maximum_final_side_distance_px):
                final_side_failure = (
                    f"{row['side']}_final_distance_exceeds_limit"
                )
                break

    retry_opening_only = bool(
        admitted and (not joint.get("accepted") or final_side_failure is not None)
    )
    selected_h = joint_h
    selected_solver = joint
    selected_mode = "openings_plus_validated_sides" if admitted else "openings_only"
    opening_only = None
    if retry_opening_only:
        opening_only_h, opening_only = estimate_residual_homography(
            observations,
            [],
            output_shape_hw,
            wall_polygon=wall_polygon_rectified,
            allow_projective=allow_projective,
            minimum_openings=minimum_openings,
            maximum_final_p90_axis_error_deg=(
                maximum_final_opening_p90_axis_error_deg
            ),
            maximum_final_p90_orthogonality_error_deg=(
                maximum_final_opening_p90_orthogonality_error_deg
            ),
            maximum_final_per_opening_axis_error_deg=(
                maximum_final_per_opening_axis_error_deg
            ),
            maximum_final_per_opening_orthogonality_error_deg=(
                maximum_final_per_opening_orthogonality_error_deg
            ),
        )
        selected_h = opening_only_h
        selected_solver = opening_only
        selected_mode = "openings_only_after_side_rejection"
        for constraint in constraints:
            if constraint.admitted:
                constraint.admitted = False
                constraint.rejection_reason = (
                    final_side_failure
                    or "joint_solution_failed_acceptance_guards"
                )
    applied = bool(selected_solver.get("accepted"))
    if not applied:
        selected_h = identity
        selected_mode = "identity_after_opening_solver_guard"
    diagnostics.update({
        "applied": applied,
        "reason": str(selected_solver.get("reason", "unknown")),
        "mode": selected_mode,
        "homography": selected_h.tolist(),
        "joint_solver": joint,
        "opening_only_retry": opening_only,
        "final_side_failure": final_side_failure,
        "fallback_policy": (
            "identity_not_conflicting_legacy_side_warp"
            if not applied else None
        ),
        "side_constraints": [item.json_record() for item in constraints],
    })
    return selected_h, diagnostics, observations


def warp_observation_masks(
    observations: Sequence[OpeningObservation],
    homography: np.ndarray,
    output_shape_hw: Tuple[int, int],
) -> List[np.ndarray]:
    height, width = [int(value) for value in output_shape_hw]
    return [
        cv2.warpPerspective(
            observation.rectified_mask.astype(np.uint8),
            np.asarray(homography, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        for observation in observations
    ]
