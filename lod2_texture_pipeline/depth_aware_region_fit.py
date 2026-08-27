# -*- coding: utf-8 -*-
"""Depth-aware alignment of a projected facade to a segmentation region.

The depth map supplies model visibility and connected-model boundary context.
The segmentation mask is the independent image observation that drives the
alignment. The fitted transform is global in image space, so it can be applied
to the wall projection, rectification quad, and full-model depth together.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.optimize import minimize

from .diagnostic_overlay_style import (
    ACCEPTED_MODEL_LINE,
    RAW_MODEL_LINE,
    REJECTED_MODEL_LINE,
    draw_legend,
    draw_styled_line,
    model_projection_legend,
)


@dataclass(frozen=True)
class DepthAwareRegionFitConfig:
    allow_rotation: bool = True
    max_working_dimension_px: int = 360
    max_translation_px: float = 100.0
    scale_min: float = 0.80
    scale_max: float = 1.20
    max_rotation_deg: float = 5.0
    target_component_search_margin_px: float = 120.0

    minimum_model_area_px: int = 350
    minimum_target_area_px: int = 350
    maximum_target_canvas_fraction: float = 0.94
    maximum_samples_per_set: int = 3500

    boundary_sigma_px: float = 8.0
    context_edge_sigma_px: float = 7.0
    context_edge_weight: float = 0.06
    transform_prior_weight: float = 0.06

    minimum_score_improvement: float = 0.035
    minimum_iou_improvement: float = 0.025
    minimum_boundary_improvement: float = 0.035
    minimum_final_iou: float = 0.30
    minimum_final_precision: float = 0.55
    minimum_mean_vertex_displacement_px: float = 2.0
    optimizer_seed_count: int = 3
    optimizer_max_evaluations: int = 260


def visible_group_mask_from_depth(
    full_model_depth: np.ndarray,
    group_depth: np.ndarray,
    absolute_tolerance_m: float = 0.08,
    relative_tolerance: float = 0.005,
) -> np.ndarray:
    """Return pixels where the group is the visible full-model surface."""
    full = np.asarray(full_model_depth, dtype=np.float32)
    group = np.asarray(group_depth, dtype=np.float32)
    if full.shape != group.shape or full.ndim != 2:
        raise ValueError("Full-model and group depth maps must have the same 2D shape.")

    valid = np.isfinite(full) & (full > 0) & np.isfinite(group) & (group > 0)
    tolerance = np.maximum(
        float(absolute_tolerance_m),
        np.abs(full) * float(relative_tolerance),
    )
    return valid & (np.abs(group - full) <= tolerance)


def depth_discontinuity_edges(
    depth_map: np.ndarray,
    absolute_jump_m: float = 0.30,
    relative_jump: float = 0.025,
) -> np.ndarray:
    """Extract silhouettes and true depth jumps without marking sloped planes."""
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth edge extraction expects a 2D depth map.")

    valid = np.isfinite(depth) & (depth > 0)
    valid_u8 = valid.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(valid_u8, cv2.MORPH_GRADIENT, kernel) > 0

    for axis in (0, 1):
        a = np.take(depth, indices=range(depth.shape[axis] - 1), axis=axis)
        b = np.take(depth, indices=range(1, depth.shape[axis]), axis=axis)
        va = np.isfinite(a) & (a > 0)
        vb = np.isfinite(b) & (b > 0)
        threshold = np.maximum(
            float(absolute_jump_m),
            np.minimum(np.abs(a), np.abs(b)) * float(relative_jump),
        )
        jump = va & vb & (np.abs(a - b) > threshold)
        if axis == 0:
            edges[:-1, :] |= jump
            edges[1:, :] |= jump
        else:
            edges[:, :-1] |= jump
            edges[:, 1:] |= jump

    if edges.shape[0] > 2 and edges.shape[1] > 2:
        edges[[0, -1], :] = False
        edges[:, [0, -1]] = False
    return edges


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8)
    eroded = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    boundary = (mask_u8 > 0) & (eroded == 0)
    if boundary.shape[0] > 2 and boundary.shape[1] > 2:
        boundary[[0, -1], :] = False
        boundary[:, [0, -1]] = False
    return boundary


def _sample_mask_points(mask: np.ndarray, maximum: int) -> np.ndarray:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64)
    points = np.column_stack([xs, ys]).astype(np.float64)
    if len(points) <= int(maximum):
        return points
    indices = np.linspace(0, len(points) - 1, int(maximum), dtype=np.int64)
    return points[indices]


def _distance_to_boundary(boundary: np.ndarray) -> np.ndarray:
    boundary_u8 = np.asarray(boundary, dtype=bool).astype(np.uint8)
    if not boundary_u8.any():
        return np.full(boundary_u8.shape, 1.0e4, dtype=np.float32)
    return cv2.distanceTransform(1 - boundary_u8, cv2.DIST_L2, 3).astype(np.float32)


def _sample_image(values: np.ndarray, points_xy: np.ndarray, border_value: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.empty((0,), dtype=np.float32)
    sampled = cv2.remap(
        np.asarray(values, dtype=np.float32),
        points_xy[:, 0].astype(np.float32).reshape(-1, 1),
        points_xy[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )
    return sampled.reshape(-1)


def _transform_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _similarity_homography(parameters: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    log_scale, rotation_deg, tx, ty = [float(v) for v in parameters]
    scale = math.exp(log_scale)
    angle = math.radians(rotation_deg)
    linear = scale * np.array(
        [[math.cos(angle), -math.sin(angle)],
         [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    center = np.asarray(center_xy, dtype=np.float64)
    offset = center + np.array([tx, ty], dtype=np.float64) - linear @ center
    return np.array(
        [[linear[0, 0], linear[0, 1], offset[0]],
         [linear[1, 0], linear[1, 1], offset[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _select_local_target_components(
    target_mask: np.ndarray,
    model_mask: np.ndarray,
    search_margin_px: float,
) -> np.ndarray:
    target = np.asarray(target_mask, dtype=bool)
    model = np.asarray(model_mask, dtype=bool)
    if not target.any() or not model.any():
        return np.zeros_like(target, dtype=bool)

    distance_to_model = cv2.distanceTransform(
        (~model).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        target.astype(np.uint8),
        connectivity=8,
    )
    margin = float(max(search_margin_px, 1.0))
    model_area = int(model.sum())
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        overlap = int((component & model).sum())
        minimum_direct_overlap = max(8, int(round(0.002 * min(model_area, area))))
        components.append({
            "mask": component,
            "overlap": overlap,
            "direct": overlap >= minimum_direct_overlap,
            "distance": float(np.min(distance_to_model[component])),
        })

    selected = np.zeros_like(target, dtype=bool)
    direct = [row for row in components if row["direct"]]
    if direct:
        for row in direct:
            selected |= row["mask"]
        return selected

    nearby = [row for row in components if row["distance"] <= margin]
    if not nearby:
        return selected
    best_distance = min(row["distance"] for row in nearby)
    for row in nearby:
        if row["distance"] <= best_distance + 3.0:
            selected |= row["mask"]
    return selected


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(width), int(height)),
        interpolation=cv2.INTER_NEAREST,
    ) > 0


def _image_edge_distance(image_bgr: Optional[np.ndarray], width: int, height: int):
    if image_bgr is None:
        return None
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 135)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3).astype(np.float32)


def _binary_overlap_metrics(candidate: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    candidate = np.asarray(candidate, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int((candidate & target).sum())
    union = int((candidate | target).sum())
    candidate_area = int(candidate.sum())
    target_area = int(target.sum())
    precision = float(intersection / max(candidate_area, 1))
    recall = float(intersection / max(target_area, 1))
    return {
        "intersection_px": intersection,
        "union_px": union,
        "candidate_area_px": candidate_area,
        "target_area_px": target_area,
        "iou": float(intersection / max(union, 1)),
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-9)),
    }


def _boundary_agreement(candidate: np.ndarray, target: np.ndarray, sigma_px: float) -> float:
    candidate_boundary = _mask_boundary(candidate)
    target_boundary = _mask_boundary(target)
    candidate_points = _sample_mask_points(candidate_boundary, 5000)
    target_points = _sample_mask_points(target_boundary, 5000)
    if len(candidate_points) == 0 or len(target_points) == 0:
        return 0.0
    target_distance = _distance_to_boundary(target_boundary)
    candidate_distance = _distance_to_boundary(candidate_boundary)
    forward = _sample_image(target_distance, candidate_points, float(sigma_px) * 8.0)
    reverse = _sample_image(candidate_distance, target_points, float(sigma_px) * 8.0)
    sigma = max(float(sigma_px), 1.0)
    return float(0.5 * (np.mean(np.exp(-forward / sigma)) + np.mean(np.exp(-reverse / sigma))))


def fit_depth_aware_segmentation_region(
    segmentation_mask: np.ndarray,
    projected_group_mask: np.ndarray,
    outline_points_px: np.ndarray,
    *,
    image_bgr: Optional[np.ndarray] = None,
    full_model_depth: Optional[np.ndarray] = None,
    config: Optional[DepthAwareRegionFitConfig] = None,
) -> Dict[str, object]:
    """Fit a visible projected wall region to a filled segmentation mask."""
    config = config or DepthAwareRegionFitConfig()
    target_full = np.asarray(segmentation_mask, dtype=bool)
    source_full = np.asarray(projected_group_mask, dtype=bool)
    outline = np.asarray(outline_points_px, dtype=np.float64)

    if target_full.ndim != 2 or source_full.shape != target_full.shape:
        raise ValueError("Segmentation and projected group masks must have the same 2D shape.")
    if outline.ndim != 2 or outline.shape[0] < 3 or outline.shape[1] != 2:
        raise ValueError("Depth-aware region fitting expects an Nx2 facade outline.")
    if not np.isfinite(outline).all():
        raise ValueError("Facade outline contains non-finite coordinates.")

    height_full, width_full = target_full.shape
    source_area_full = int(source_full.sum())
    target_area_full = int(target_full.sum())
    identity = np.eye(3, dtype=np.float64)

    def rejected(reason, target_mask=None):
        target_used = target_full if target_mask is None else np.asarray(target_mask, dtype=bool)
        metrics = _binary_overlap_metrics(source_full, target_used)
        return {
            "applied": False,
            "reason": str(reason),
            "homography": identity,
            "original_points": outline,
            "candidate_points": outline.copy(),
            "fitted_points": outline.copy(),
            "target_mask": target_used,
            "original_mask": source_full,
            "candidate_mask": source_full.copy(),
            "fitted_mask": source_full.copy(),
            "transform": {"scale": 1.0, "rotation_deg": 0.0, "tx": 0.0, "ty": 0.0},
            "score_before": 0.0,
            "score_after": 0.0,
            "score_improvement": 0.0,
            "mean_vertex_displacement_px": 0.0,
            "metrics_before": metrics,
            "metrics_after": metrics.copy(),
        }

    if source_area_full < int(config.minimum_model_area_px):
        return rejected("projected_visible_group_too_small")
    if target_area_full < int(config.minimum_target_area_px):
        return rejected("segmentation_region_too_small")
    if target_area_full / max(height_full * width_full, 1) > float(config.maximum_target_canvas_fraction):
        return rejected("segmentation_covers_most_of_canvas")

    work_scale = min(
        1.0,
        float(config.max_working_dimension_px) / max(height_full, width_full, 1),
    )
    width = max(8, int(round(width_full * work_scale)))
    height = max(8, int(round(height_full * work_scale)))
    source = _resize_mask(source_full, width, height)
    target_raw = _resize_mask(target_full, width, height)
    target = _select_local_target_components(
        target_raw,
        source,
        float(config.target_component_search_margin_px) * work_scale,
    )
    if int(target.sum()) < max(20, int(config.minimum_target_area_px * work_scale * work_scale)):
        return rejected("no_local_segmentation_component_near_projection")

    source_boundary = _mask_boundary(source)
    target_boundary = _mask_boundary(target)
    source_boundary_points = _sample_mask_points(source_boundary, config.maximum_samples_per_set)
    target_boundary_points = _sample_mask_points(target_boundary, config.maximum_samples_per_set)
    source_inside_points = _sample_mask_points(source, config.maximum_samples_per_set)
    target_inside_points = _sample_mask_points(target, config.maximum_samples_per_set)
    if min(len(source_boundary_points), len(target_boundary_points)) < 8:
        return rejected("insufficient_region_boundary_support", _resize_mask(target, width_full, height_full))

    source_soft = cv2.GaussianBlur(source.astype(np.float32), (0, 0), 0.8)
    target_soft = cv2.GaussianBlur(target.astype(np.float32), (0, 0), 0.8)
    source_boundary_distance = _distance_to_boundary(source_boundary)
    target_boundary_distance = _distance_to_boundary(target_boundary)

    image_edge_distance = _image_edge_distance(image_bgr, width, height)
    context_points = np.empty((0, 2), dtype=np.float64)
    if full_model_depth is not None and image_edge_distance is not None:
        full_depth = np.asarray(full_model_depth, dtype=np.float32)
        if full_depth.shape == target_full.shape:
            context_edges = depth_discontinuity_edges(full_depth)
            context_edges = _resize_mask(context_edges, width, height)
            context_points = _sample_mask_points(context_edges, config.maximum_samples_per_set)

    source_centroid = source_inside_points.mean(axis=0)
    target_centroid = target_inside_points.mean(axis=0)
    sigma_boundary = max(float(config.boundary_sigma_px) * work_scale, 1.0)
    sigma_context = max(float(config.context_edge_sigma_px) * work_scale, 1.0)
    translation_limit = min(
        float(config.max_translation_px) * work_scale,
        0.40 * max(width, height),
    )
    rotation_limit = float(config.max_rotation_deg) if config.allow_rotation else 1.0e-6

    bounds = [
        (math.log(float(config.scale_min)), math.log(float(config.scale_max))),
        (-rotation_limit, rotation_limit),
        (-translation_limit, translation_limit),
        (-translation_limit, translation_limit),
    ]

    def evaluate(parameters):
        H = _similarity_homography(parameters, source_centroid)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return -1.0e6, {}

        transformed_source_boundary = _transform_points(source_boundary_points, H)
        source_boundary_distances = _sample_image(
            target_boundary_distance,
            transformed_source_boundary,
            sigma_boundary * 8.0,
        )
        transformed_target_boundary = _transform_points(target_boundary_points, H_inv)
        target_boundary_distances = _sample_image(
            source_boundary_distance,
            transformed_target_boundary,
            sigma_boundary * 8.0,
        )
        forward_boundary = float(np.mean(np.exp(-source_boundary_distances / sigma_boundary)))
        reverse_boundary = float(np.mean(np.exp(-target_boundary_distances / sigma_boundary)))
        boundary_score = 0.60 * forward_boundary + 0.40 * reverse_boundary

        transformed_source_inside = _transform_points(source_inside_points, H)
        precision = float(np.mean(_sample_image(target_soft, transformed_source_inside, 0.0)))
        transformed_target_inside = _transform_points(target_inside_points, H_inv)
        recall = float(np.mean(_sample_image(source_soft, transformed_target_inside, 0.0)))

        context_score = 0.0
        if len(context_points) > 0 and image_edge_distance is not None:
            transformed_context = _transform_points(context_points, H)
            distances = _sample_image(
                image_edge_distance,
                transformed_context,
                sigma_context * 8.0,
            )
            context_score = float(np.mean(np.exp(-distances / sigma_context)))

        log_scale, rotation_deg, tx, ty = [float(v) for v in parameters]
        prior_energy = (
            (log_scale / max(math.log(1.14), 1.0e-6)) ** 2
            + (rotation_deg / max(float(config.max_rotation_deg), 1.0)) ** 2
            + (tx / max(translation_limit * 0.75, 1.0)) ** 2
            + (ty / max(translation_limit * 0.75, 1.0)) ** 2
        )
        prior_score = float(math.exp(-0.5 * prior_energy))
        region_score = 0.42 * boundary_score + 0.40 * precision + 0.18 * recall
        total_score = (
            region_score
            + float(config.context_edge_weight) * context_score
            + float(config.transform_prior_weight) * prior_score
        )
        return float(total_score), {
            "boundary_score": float(boundary_score),
            "forward_boundary_score": float(forward_boundary),
            "reverse_boundary_score": float(reverse_boundary),
            "region_precision": float(precision),
            "region_recall": float(recall),
            "context_depth_edge_score": float(context_score),
            "transform_prior_score": float(prior_score),
        }

    identity_parameters = np.zeros(4, dtype=np.float64)
    score_before, diagnostics_before = evaluate(identity_parameters)
    centroid_delta = np.clip(
        target_centroid - source_centroid,
        -translation_limit,
        translation_limit,
    )
    area_scale = float(np.clip(
        math.sqrt(float(target.sum()) / max(float(source.sum()), 1.0)),
        float(config.scale_min),
        float(config.scale_max),
    ))

    tx_candidates = np.unique(np.clip(
        [0.0, centroid_delta[0], centroid_delta[0] - 0.35 * translation_limit, centroid_delta[0] + 0.35 * translation_limit],
        -translation_limit,
        translation_limit,
    ))
    ty_candidates = np.unique(np.clip(
        [0.0, centroid_delta[1], centroid_delta[1] - 0.35 * translation_limit, centroid_delta[1] + 0.35 * translation_limit],
        -translation_limit,
        translation_limit,
    ))
    scales = np.unique(np.clip(
        [1.0, area_scale, 0.90, 1.10],
        float(config.scale_min),
        float(config.scale_max),
    ))
    rotations = np.array([0.0], dtype=np.float64)
    if config.allow_rotation:
        rotations = np.array([0.0, -0.5 * rotation_limit, 0.5 * rotation_limit], dtype=np.float64)

    ranked_seeds = []
    for scale in scales:
        for rotation in rotations:
            for tx in tx_candidates:
                for ty in ty_candidates:
                    seed = np.array([math.log(float(scale)), rotation, tx, ty], dtype=np.float64)
                    score, _diagnostics = evaluate(seed)
                    ranked_seeds.append((float(score), seed))
    ranked_seeds.sort(key=lambda item: item[0], reverse=True)

    best_parameters = identity_parameters.copy()
    best_score = float(score_before)
    best_diagnostics = diagnostics_before
    for _seed_score, seed in ranked_seeds[:max(1, int(config.optimizer_seed_count))]:
        result = minimize(
            lambda values: -evaluate(values)[0],
            seed,
            method="Powell",
            bounds=bounds,
            options={
                "maxfev": int(config.optimizer_max_evaluations),
                "xtol": 0.05,
                "ftol": 1.0e-4,
            },
        )
        candidate_parameters = np.asarray(result.x, dtype=np.float64)
        candidate_score, candidate_diagnostics = evaluate(candidate_parameters)
        if candidate_score > best_score:
            best_parameters = candidate_parameters
            best_score = float(candidate_score)
            best_diagnostics = candidate_diagnostics

    H_work = _similarity_homography(best_parameters, source_centroid)
    S_full_to_work = np.array(
        [[work_scale, 0.0, 0.0], [0.0, work_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_full = np.linalg.inv(S_full_to_work) @ H_work @ S_full_to_work
    candidate_mask = cv2.warpPerspective(
        source_full.astype(np.uint8),
        H_full,
        (width_full, height_full),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    target_local_full = _resize_mask(target, width_full, height_full)
    metrics_before = _binary_overlap_metrics(source_full, target_local_full)
    metrics_after = _binary_overlap_metrics(candidate_mask, target_local_full)
    metrics_before["boundary_agreement"] = _boundary_agreement(
        source_full,
        target_local_full,
        config.boundary_sigma_px,
    )
    metrics_after["boundary_agreement"] = _boundary_agreement(
        candidate_mask,
        target_local_full,
        config.boundary_sigma_px,
    )

    candidate_points = _transform_points(outline, H_full)
    displacement = float(np.mean(np.linalg.norm(candidate_points - outline, axis=1)))
    score_improvement = float(best_score - score_before)
    iou_improvement = float(metrics_after["iou"] - metrics_before["iou"])
    boundary_improvement = float(
        metrics_after["boundary_agreement"] - metrics_before["boundary_agreement"]
    )

    quality_improved = bool(
        iou_improvement >= float(config.minimum_iou_improvement)
        or boundary_improvement >= float(config.minimum_boundary_improvement)
    )
    applied = bool(
        score_improvement >= float(config.minimum_score_improvement)
        and quality_improved
        and metrics_after["iou"] >= float(config.minimum_final_iou)
        and metrics_after["precision"] >= float(config.minimum_final_precision)
        and displacement >= float(config.minimum_mean_vertex_displacement_px)
    )
    if applied:
        reason = "accepted_region_and_depth_context_improvement"
        fitted_points = candidate_points
        fitted_mask = candidate_mask
        accepted_H = H_full
    else:
        if score_improvement < float(config.minimum_score_improvement):
            reason = "insufficient_score_improvement"
        elif not quality_improved:
            reason = "region_overlap_or_boundary_did_not_improve"
        elif metrics_after["iou"] < float(config.minimum_final_iou):
            reason = "final_region_iou_too_low"
        elif metrics_after["precision"] < float(config.minimum_final_precision):
            reason = "final_region_precision_too_low"
        else:
            reason = "movement_below_materiality_threshold"
        fitted_points = outline.copy()
        fitted_mask = source_full.copy()
        accepted_H = identity

    return {
        "applied": applied,
        "reason": reason,
        "homography": accepted_H,
        "candidate_homography": H_full,
        "original_points": outline,
        "candidate_points": candidate_points,
        "fitted_points": fitted_points,
        "target_mask": target_local_full,
        "original_mask": source_full,
        "candidate_mask": candidate_mask,
        "fitted_mask": fitted_mask,
        "transform": {
            "scale": float(math.exp(best_parameters[0])),
            "rotation_deg": float(best_parameters[1]),
            "tx": float(best_parameters[2] / max(work_scale, 1.0e-9)),
            "ty": float(best_parameters[3] / max(work_scale, 1.0e-9)),
            "center_x": float(source_centroid[0] / max(work_scale, 1.0e-9)),
            "center_y": float(source_centroid[1] / max(work_scale, 1.0e-9)),
        },
        "score_before": float(score_before),
        "score_after": float(best_score),
        "score_improvement": score_improvement,
        "mean_vertex_displacement_px": displacement,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "diagnostics_before": diagnostics_before,
        "diagnostics_after": best_diagnostics,
        "working_scale": float(work_scale),
        "context_depth_edge_count": int(len(context_points)),
    }


def depth_aware_region_fit_metadata(result: Optional[Dict[str, object]]):
    if not result:
        return None

    def scalar_dict(values):
        out = {}
        for key, value in dict(values or {}).items():
            if isinstance(value, (int, np.integer)):
                out[str(key)] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[str(key)] = float(value)
            else:
                out[str(key)] = value
        return out

    transform = scalar_dict(result.get("transform", {}))
    return {
        "method": "visible_model_region_to_segmentation_similarity",
        "depth_role": "full-model visibility and weak connected-model depth-edge context",
        "segmentation_role": "independent filled image-region observation",
        "whole_model_transform": True,
        "applied": bool(result.get("applied", False)),
        "reason": str(result.get("reason", "unknown")),
        "scale": float(transform.get("scale", 1.0)),
        "rotation_deg": float(transform.get("rotation_deg", 0.0)),
        "tx_px": float(transform.get("tx", 0.0)),
        "ty_px": float(transform.get("ty", 0.0)),
        "transform_center_px": [
            float(transform.get("center_x", 0.0)),
            float(transform.get("center_y", 0.0)),
        ],
        "score_before": float(result.get("score_before", 0.0)),
        "score_after": float(result.get("score_after", 0.0)),
        "score_improvement": float(result.get("score_improvement", 0.0)),
        "mean_vertex_displacement_px": float(result.get("mean_vertex_displacement_px", 0.0)),
        "H_pre_segmentation_fit_to_region_fit": np.asarray(
            result.get("homography", np.eye(3)),
            dtype=np.float64,
        ).astype(float).tolist(),
        "candidate_H": np.asarray(
            result.get("candidate_homography", np.eye(3)),
            dtype=np.float64,
        ).astype(float).tolist(),
        "outline_before_px": np.asarray(result.get("original_points", [])).astype(float).tolist(),
        "outline_candidate_px": np.asarray(result.get("candidate_points", [])).astype(float).tolist(),
        "outline_after_px": np.asarray(result.get("fitted_points", [])).astype(float).tolist(),
        "metrics_before": scalar_dict(result.get("metrics_before", {})),
        "metrics_after": scalar_dict(result.get("metrics_after", {})),
        "diagnostics_before": scalar_dict(result.get("diagnostics_before", {})),
        "diagnostics_after": scalar_dict(result.get("diagnostics_after", {})),
        "working_scale": float(result.get("working_scale", 1.0)),
        "context_depth_edge_count": int(result.get("context_depth_edge_count", 0)),
    }


def create_depth_aware_region_fit_overlay(
    image_bgr: np.ndarray,
    result: Dict[str, object],
) -> np.ndarray:
    """Draw segmentation, pre-fit outline, and accepted/candidate region fit."""
    image = np.asarray(image_bgr, dtype=np.uint8).copy()
    target = np.asarray(result.get("target_mask", np.zeros(image.shape[:2])), dtype=bool)
    tint = image.copy()
    tint[target] = np.array([70, 190, 70], dtype=np.uint8)
    image = cv2.addWeighted(image, 0.78, tint, 0.22, 0.0)

    target_contours, _ = cv2.findContours(
        target.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(image, target_contours, -1, (0, 210, 0), 2, cv2.LINE_AA)

    original = np.asarray(result["original_points"], dtype=np.float64)
    shown = np.asarray(
        result["fitted_points"] if result.get("applied") else result["candidate_points"],
        dtype=np.float64,
    )

    def draw_closed_outline(points: np.ndarray, style) -> None:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(values) < 2:
            return
        for index in range(len(values)):
            draw_styled_line(
                image,
                values[index],
                values[(index + 1) % len(values)],
                style,
                color_space="bgr",
            )

    draw_closed_outline(original, RAW_MODEL_LINE)
    shown_style = ACCEPTED_MODEL_LINE if result.get("applied") else REJECTED_MODEL_LINE
    draw_closed_outline(shown, shown_style)

    transform = result.get("transform", {})
    status = "accepted" if result.get("applied") else f"rejected: {result.get('reason')}"
    rows = [
        "evidence: green=segmentation region",
        model_projection_legend(
            fitted=bool(result.get("applied")),
            rejected=not bool(result.get("applied")),
        ),
        (
            f"{status} | scale={float(transform.get('scale', 1.0)):.4f} "
            f"tx={float(transform.get('tx', 0.0)):.1f}px "
            f"ty={float(transform.get('ty', 0.0)):.1f}px "
            f"score gain={float(result.get('score_improvement', 0.0)):.4f}"
        ),
    ]
    draw_legend(image, rows, color_space="bgr")
    return image
