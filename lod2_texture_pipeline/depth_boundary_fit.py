# -*- coding: utf-8 -*-
"""Independent whole-model depth-silhouette fitting.

The geometric source remains the rendered whole-model depth map. Optional
projection-local image semantics suppress occluder edges and add class-aware
boundary likelihoods without replacing the model shape or its search prior.
The same global image-space transform is then applied to the raw facade
projection.
"""

from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .diagnostic_overlay_style import (
    ACCEPTED_MODEL_LINE,
    RAW_MODEL_LINE,
    REJECTED_MODEL_LINE,
    SAM_BASE_GUIDE_LINE,
    SAM_ROOF_GUIDE_LINE,
    SAM_WALL_GUIDE_LINE,
    WALL_ONLY_MODEL_LINE,
    OverlayLineStyle,
    draw_legend,
    draw_styled_line,
    model_projection_legend,
)
from .wireframe_fit import (
    WireframeFitConfig,
    apply_homography,
    fit_wireframe_to_image,
    similarity_homography,
    visible_segments_from_points,
    wireframe_fit_metadata,
)


SEMANTIC_BOUNDARY_CLASSES = ("roof", "wall", "base")


def filter_image_border_wrapper_segments(
    points: np.ndarray,
    segment_indices: Sequence[Tuple[int, int]],
    image_shape_hw: Tuple[int, int],
    *,
    epsilon_px: float = 0.5,
) -> Tuple[Sequence[Tuple[int, int]], Sequence[int]]:
    """Conservatively remove legacy contour segments joining frame points.

    New depth geometry removes frame runs before simplification and should not
    need this fallback. It remains useful for older serialized results, where a
    right-to-bottom corner wrapper may already have collapsed to one diagonal.
    """
    model_points = np.asarray(points, dtype=np.float64)
    if model_points.ndim != 2 or model_points.shape[1] != 2:
        raise ValueError("Model boundary points must be an Nx2 array.")
    height, width = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    epsilon = max(0.0, float(epsilon_px))

    def on_image_frame(point):
        return bool(
            abs(float(point[0])) <= epsilon
            or abs(float(point[0]) - (width - 1)) <= epsilon
            or abs(float(point[1])) <= epsilon
            or abs(float(point[1]) - (height - 1)) <= epsilon
        )

    retained = []
    excluded_source_indices = []
    for source_index, pair in enumerate(segment_indices):
        if len(pair) != 2:
            continue
        index0, index1 = int(pair[0]), int(pair[1])
        if not (
            0 <= index0 < len(model_points)
            and 0 <= index1 < len(model_points)
            and index0 != index1
        ):
            continue
        if (
            on_image_frame(model_points[index0])
            and on_image_frame(model_points[index1])
        ):
            excluded_source_indices.append(int(source_index))
            continue
        retained.append((index0, index1))
    return retained, excluded_source_indices


def extract_depth_silhouette_geometry(
    depth_map: np.ndarray,
    *,
    minimum_area_px: int = 350,
    minimum_component_fraction: float = 0.02,
    contour_epsilon_px: float = 1.5,
    maximum_points: int = 240,
) -> Dict[str, object]:
    """Return external depth-mask contours as one point/segment collection."""
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth silhouette extraction expects a 2D depth map.")

    mask = np.isfinite(depth) & (depth > 0)
    if int(mask.sum()) < int(minimum_area_px):
        raise ValueError("Whole-model depth silhouette is too small.")

    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [c for c in contours if len(c) >= 3]
    if not contours:
        raise ValueError("Whole-model depth map has no external silhouette.")

    areas = np.asarray([abs(float(cv2.contourArea(c))) for c in contours], dtype=np.float64)
    largest_area = float(np.max(areas)) if len(areas) else 0.0
    minimum_component_area = max(
        8.0,
        largest_area * float(minimum_component_fraction),
    )
    kept = [
        contour
        for contour, area in sorted(
            zip(contours, areas),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        if float(area) >= minimum_component_area
    ]
    if not kept:
        kept = [contours[int(np.argmax(areas))]]

    height, width = mask.shape
    approximation_epsilon = max(float(contour_epsilon_px), 0.25)
    simplified_runs = []
    removed_frame_edge_count = 0
    retained_component_indices = set()

    def point_on_exact_frame(point):
        x, y = int(point[0]), int(point[1])
        return bool(x == 0 or x == width - 1 or y == 0 or y == height - 1)

    for component_index, contour in enumerate(kept):
        dense_points = np.asarray(contour[:, 0, :], dtype=np.int32)
        point_count = int(len(dense_points))
        if point_count < 3:
            continue
        frame_points = np.asarray(
            [point_on_exact_frame(point) for point in dense_points],
            dtype=bool,
        )
        frame_edges = frame_points & np.roll(frame_points, -1)
        removed_frame_edge_count += int(frame_edges.sum())

        if not frame_edges.any():
            approx = cv2.approxPolyDP(
                dense_points.reshape(-1, 1, 2),
                epsilon=approximation_epsilon,
                closed=True,
            )
            points = np.asarray(approx[:, 0, :], dtype=np.float64)
            if len(points) >= 3:
                simplified_runs.append({
                    "points": points,
                    "closed": True,
                    "component_index": int(component_index),
                })
                retained_component_indices.add(int(component_index))
            continue

        retained_edges = ~frame_edges
        run_starts = [
            index
            for index in range(point_count)
            if retained_edges[index] and not retained_edges[(index - 1) % point_count]
        ]
        for start in run_starts:
            run_indices = [int(start)]
            edge_index = int(start)
            while retained_edges[edge_index]:
                run_indices.append(int((edge_index + 1) % point_count))
                edge_index = int((edge_index + 1) % point_count)
                if edge_index == start:
                    break
            run_points = dense_points[run_indices]
            if len(run_points) < 2:
                continue
            approx = cv2.approxPolyDP(
                run_points.reshape(-1, 1, 2),
                epsilon=approximation_epsilon,
                closed=False,
            )
            points = np.asarray(approx[:, 0, :], dtype=np.float64)
            if (
                len(points) >= 2
                and float(np.linalg.norm(points[-1] - points[0])) > 0.0
            ):
                simplified_runs.append({
                    "points": points,
                    "closed": False,
                    "component_index": int(component_index),
                })
                retained_component_indices.add(int(component_index))

    if not simplified_runs:
        raise ValueError(
            "Whole-model depth silhouette contains only image-frame closure."
        )

    total_points = sum(len(run["points"]) for run in simplified_runs)
    if total_points > int(maximum_points):
        budgets = []
        minimum_budgets = [
            3 if run["closed"] else 2 for run in simplified_runs
        ]
        remaining = max(int(maximum_points), int(sum(minimum_budgets)))
        remaining_points = int(total_points)
        for index, run in enumerate(simplified_runs):
            points = run["points"]
            minimum_budget = minimum_budgets[index]
            minimum_remaining = int(sum(minimum_budgets[index + 1:]))
            budget = max(
                minimum_budget,
                int(round(remaining * len(points) / max(remaining_points, 1))),
            )
            budget = min(
                budget,
                len(points),
                remaining - minimum_remaining,
            )
            budgets.append(budget)
            remaining -= budget
            remaining_points -= len(points)
        for run, budget in zip(simplified_runs, budgets):
            points = run["points"]
            run["points"] = points[
                np.linspace(0, len(points) - 1, budget, dtype=np.int64)
            ]

    all_points = []
    segment_indices = []
    contour_ranges = []
    offset = 0
    for run in simplified_runs:
        points = run["points"]
        count = len(points)
        all_points.append(points)
        segment_indices.extend(
            (offset + index, offset + index + 1)
            for index in range(count - 1)
        )
        if run["closed"]:
            segment_indices.append((offset + count - 1, offset))
        contour_ranges.append([int(offset), int(offset + count)])
        offset += count

    stacked_points = np.vstack(all_points).astype(np.float64)

    return {
        "mask": mask,
        "points": stacked_points,
        "segment_indices": segment_indices,
        "frame_wrappers_filtered": True,
        "image_border_wrapper_segment_count": int(removed_frame_edge_count),
        "boundary_run_count": int(len(simplified_runs)),
        "contour_ranges": contour_ranges,
        "component_count": int(len(retained_component_indices)),
        "point_count": int(offset),
        "area_px": int(mask.sum()),
    }


def _clip_camera_edge_to_near(camera0, camera1, near_m):
    camera0 = np.asarray(camera0, dtype=np.float64).copy()
    camera1 = np.asarray(camera1, dtype=np.float64).copy()
    depth0 = float(camera0[2])
    depth1 = float(camera1[2])
    if depth0 < near_m and depth1 < near_m:
        return None
    if depth0 < near_m:
        denominator = depth1 - depth0
        if abs(denominator) < 1.0e-12:
            return None
        amount = float(np.clip((near_m - depth0) / denominator, 0.0, 1.0))
        camera0 = camera0 + amount * (camera1 - camera0)
    if depth1 < near_m:
        denominator = float(camera0[2]) - depth1
        if abs(denominator) < 1.0e-12:
            return None
        amount = float(np.clip((near_m - depth1) / denominator, 0.0, 1.0))
        camera1 = camera1 + amount * (camera0 - camera1)
    return camera0, camera1


def _project_camera_points(camera_points, K, image_to_output_H):
    camera_points = np.asarray(camera_points, dtype=np.float64)
    depth = camera_points[:, 2]
    uv = np.empty((len(camera_points), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (camera_points[:, 0] / depth) + K[0, 2]
    uv[:, 1] = K[1, 1] * (-camera_points[:, 1] / depth) + K[1, 2]
    return apply_homography(uv, image_to_output_H)


def _fill_short_false_runs(values, maximum_gap):
    values = np.asarray(values, dtype=bool).copy()
    maximum_gap = max(0, int(maximum_gap))
    if maximum_gap == 0 or len(values) < 3:
        return values
    index = 1
    while index < len(values) - 1:
        if values[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not values[index]:
            index += 1
        if (
            start > 0
            and index < len(values)
            and index - start <= maximum_gap
            and values[start - 1]
            and values[index]
        ):
            values[start:index] = True
    return values


def _true_runs(values):
    values = np.asarray(values, dtype=bool)
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(values) - 1):
            end = index if value and index == len(values) - 1 else index - 1
            if end > start:
                yield int(start), int(end)
            start = None


def _depth_matches_nearby(
    depth_map,
    point_xy,
    expected_depth,
    *,
    search_radius_px,
    absolute_tolerance_m,
    relative_tolerance,
):
    height, width = depth_map.shape
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    radius = max(0, int(search_radius_px))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    nearby = np.asarray(depth_map[y0:y1, x0:x1], dtype=np.float64)
    nearby = nearby[np.isfinite(nearby) & (nearby > 0.0)]
    if len(nearby) == 0:
        return False
    tolerance = max(
        float(absolute_tolerance_m),
        abs(float(expected_depth)) * float(relative_tolerance),
    )
    return bool(float(np.min(np.abs(nearby - float(expected_depth)))) <= tolerance)


def project_semantic_model_boundary_edges(
    *,
    model_edges_xyz_by_class: Mapping[str, Sequence],
    K: np.ndarray,
    R_wc: np.ndarray,
    C: np.ndarray,
    full_model_depth: np.ndarray,
    image_to_output_H: Optional[np.ndarray] = None,
    near_m: float = 0.05,
    sample_step_px: float = 2.0,
    silhouette_tolerance_px: float = 4.0,
    depth_search_radius_px: int = 2,
    depth_tolerance_m: float = 0.35,
    depth_relative_tolerance: float = 0.03,
    maximum_visibility_gap_samples: int = 2,
    minimum_visible_run_px: float = 8.0,
) -> Dict[str, object]:
    """Project labeled model edges and keep only visible silhouette portions."""
    depth_map = np.asarray(full_model_depth, dtype=np.float32)
    if depth_map.ndim != 2:
        raise ValueError("Semantic boundary projection expects a 2D model depth map.")
    K = np.asarray(K, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64).reshape(3)
    output_H = np.asarray(
        np.eye(3) if image_to_output_H is None else image_to_output_H,
        dtype=np.float64,
    )

    model_mask = np.isfinite(depth_map) & (depth_map > 0.0)
    if not model_mask.any():
        raise ValueError("Semantic boundary projection needs a non-empty model depth map.")
    eroded = cv2.erode(model_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    silhouette_boundary = model_mask & ~eroded
    distance_to_silhouette = cv2.distanceTransform(
        (~silhouette_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    points = []
    segments = []
    segment_classes = []
    input_edge_counts = {}
    projected_edge_counts = {}
    visible_segment_counts = {}
    visible_length_px = {}

    height, width = depth_map.shape
    for edge_class in SEMANTIC_BOUNDARY_CLASSES:
        raw_edges = np.asarray(
            model_edges_xyz_by_class.get(edge_class, []),
            dtype=np.float64,
        )
        if raw_edges.size == 0:
            raw_edges = np.empty((0, 2, 3), dtype=np.float64)
        if raw_edges.ndim != 3 or raw_edges.shape[1:] != (2, 3):
            raise ValueError(
                f"Semantic '{edge_class}' edges must have shape Nx2x3."
            )
        input_edge_counts[edge_class] = int(len(raw_edges))
        projected_count = 0
        visible_count = 0
        visible_length = 0.0

        for world_edge in raw_edges:
            camera_edge = (R_wc @ (world_edge - C).T).T
            clipped = _clip_camera_edge_to_near(
                camera_edge[0],
                camera_edge[1],
                float(near_m),
            )
            if clipped is None:
                continue
            camera0, camera1 = clipped
            endpoints = _project_camera_points(
                np.vstack([camera0, camera1]),
                K,
                output_H,
            )
            if not np.isfinite(endpoints).all():
                continue
            projected_length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            if projected_length < 1.0:
                continue
            projected_count += 1
            sample_count = max(
                2,
                int(projected_length / max(float(sample_step_px), 0.5)) + 1,
            )
            amounts = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
            camera_samples = (
                camera0[None, :] * (1.0 - amounts[:, None])
                + camera1[None, :] * amounts[:, None]
            )
            uv_samples = _project_camera_points(camera_samples, K, output_H)
            expected_depths = camera_samples[:, 2]

            valid = np.zeros((sample_count,), dtype=bool)
            for sample_index, (point_xy, expected_depth) in enumerate(
                zip(uv_samples, expected_depths)
            ):
                if not np.isfinite(point_xy).all() or not np.isfinite(expected_depth):
                    continue
                x = int(round(float(point_xy[0])))
                y = int(round(float(point_xy[1])))
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if float(distance_to_silhouette[y, x]) > float(silhouette_tolerance_px):
                    continue
                valid[sample_index] = _depth_matches_nearby(
                    depth_map,
                    point_xy,
                    expected_depth,
                    search_radius_px=depth_search_radius_px,
                    absolute_tolerance_m=depth_tolerance_m,
                    relative_tolerance=depth_relative_tolerance,
                )
            valid = _fill_short_false_runs(valid, maximum_visibility_gap_samples)

            for run_start, run_end in _true_runs(valid):
                start = uv_samples[run_start]
                end = uv_samples[run_end]
                run_length = float(np.linalg.norm(end - start))
                if run_length < float(minimum_visible_run_px):
                    continue
                point_offset = len(points)
                points.extend([start, end])
                segments.append((point_offset, point_offset + 1))
                segment_classes.append(edge_class)
                visible_count += 1
                visible_length += run_length

        projected_edge_counts[edge_class] = int(projected_count)
        visible_segment_counts[edge_class] = int(visible_count)
        visible_length_px[edge_class] = float(visible_length)

    return {
        "points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
        "segment_indices": segments,
        "segment_classes": segment_classes,
        "input_edge_counts": input_edge_counts,
        "projected_edge_counts": projected_edge_counts,
        "visible_segment_counts": visible_segment_counts,
        "visible_length_px_by_class": visible_length_px,
        "silhouette_tolerance_px": float(silhouette_tolerance_px),
        "depth_tolerance_m": float(depth_tolerance_m),
        "depth_relative_tolerance": float(depth_relative_tolerance),
    }


def fit_depth_silhouette_to_image(
    *,
    image_bgr: np.ndarray,
    full_model_depth: np.ndarray,
    raw_wall_outline_px: np.ndarray,
    wall_local_fit_outline_px: np.ndarray,
    fit_config: WireframeFitConfig,
    minimum_area_px: int = 350,
    minimum_component_fraction: float = 0.02,
    contour_epsilon_px: float = 1.5,
    maximum_points: int = 240,
    semantic_boundary_geometry: Optional[Dict[str, object]] = None,
    semantic_class_weights: Optional[Mapping[str, float]] = None,
    valid_image_evidence_mask: Optional[np.ndarray] = None,
    semantic_valid_image_evidence_mask: Optional[np.ndarray] = None,
    semantic_image_boundary_maps: Optional[Mapping[str, np.ndarray]] = None,
    semantic_image_guidance_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Fit weighted model boundaries and derive a global-context wall fit."""
    geometry = extract_depth_silhouette_geometry(
        full_model_depth,
        minimum_area_px=minimum_area_px,
        minimum_component_fraction=minimum_component_fraction,
        contour_epsilon_px=contour_epsilon_px,
        maximum_points=maximum_points,
    )
    semantic_class_weights = {
        "roof": 3.0,
        "wall": 2.0,
        "base": 0.35,
        **dict(semantic_class_weights or {}),
    }
    semantic_points = np.asarray(
        (semantic_boundary_geometry or {}).get("points", []),
        dtype=np.float64,
    ).reshape(-1, 2)
    semantic_segments = list(
        (semantic_boundary_geometry or {}).get("segment_indices", [])
    )
    semantic_classes = list(
        (semantic_boundary_geometry or {}).get("segment_classes", [])
    )
    use_semantic_guides = bool(
        len(semantic_points) >= 2
        and len(semantic_segments) > 0
        and len(semantic_segments) == len(semantic_classes)
    )
    fit_points = semantic_points if use_semantic_guides else geometry["points"]
    fit_segments = semantic_segments if use_semantic_guides else geometry["segment_indices"]
    if not use_semantic_guides and not fit_segments:
        raise ValueError(
            "Whole-model depth silhouette has no non-frame boundary segments."
        )
    class_lengths = {edge_class: 0.0 for edge_class in SEMANTIC_BOUNDARY_CLASSES}
    if use_semantic_guides:
        for (index0, index1), label in zip(semantic_segments, semantic_classes):
            class_lengths.setdefault(label, 0.0)
            class_lengths[label] += float(np.linalg.norm(
                semantic_points[int(index1)] - semantic_points[int(index0)]
            ))
        # Normalize within each semantic class. This makes the configured
        # priorities control the class contribution instead of allowing one
        # long, unreliable base edge to dominate through sample count alone.
        fit_segment_weights = [
            float(semantic_class_weights.get(label, 1.0))
            / max(float(class_lengths.get(label, 0.0)), 1.0)
            for label in semantic_classes
        ]
    else:
        fit_segment_weights = None
    fit_segment_classes = (
        semantic_classes
        if use_semantic_guides
        else ["silhouette"] * len(fit_segments)
    )
    semantic_image_guidance_metadata = dict(
        semantic_image_guidance_metadata or {}
    )
    fit = fit_wireframe_to_image(
        np.asarray(image_bgr, dtype=np.uint8),
        fit_points,
        config=fit_config,
        segment_indices=fit_segments,
        segment_weights=fit_segment_weights,
        valid_evidence_mask=valid_image_evidence_mask,
        semantic_valid_evidence_mask=semantic_valid_image_evidence_mask,
        segment_classes=fit_segment_classes,
        semantic_boundary_maps=semantic_image_boundary_maps,
    )

    transform = fit["transform"]
    center = np.array(
        [transform["transform_center_x"], transform["transform_center_y"]],
        dtype=np.float64,
    )
    candidate_H = similarity_homography(
        transform["scale"],
        transform["rotation_deg"],
        transform["tx"],
        transform["ty"],
        center,
    )
    accepted_H = np.asarray(fit["homography"], dtype=np.float64)
    fit_original_points = np.asarray(fit["original_points"], dtype=np.float64)
    fit_candidate_points = np.asarray(fit["candidate_points"], dtype=np.float64)
    fit_fitted_points = np.asarray(fit["fitted_points"], dtype=np.float64)
    silhouette_points = np.asarray(geometry["points"], dtype=np.float64)
    silhouette_candidate = apply_homography(silhouette_points, candidate_H)
    silhouette_fitted = apply_homography(silhouette_points, accepted_H)
    raw_wall = np.asarray(raw_wall_outline_px, dtype=np.float64)
    local_wall = np.asarray(wall_local_fit_outline_px, dtype=np.float64)
    if raw_wall.ndim != 2 or raw_wall.shape[1] != 2 or not np.isfinite(raw_wall).all():
        raise ValueError("Raw wall projection must be a finite Nx2 array.")
    if local_wall.ndim != 2 or local_wall.shape[1] != 2 or not np.isfinite(local_wall).all():
        raise ValueError("Wall-local fit projection must be a finite Nx2 array.")

    fit.update({
        "raw_depth_mask": geometry["mask"],
        "original_points": silhouette_points,
        "candidate_points": silhouette_candidate,
        "fitted_points": silhouette_fitted,
        "segment_indices": geometry["segment_indices"],
        "segment_classes": [
            "silhouette"
            for _segment in geometry["segment_indices"]
        ],
        "segment_weights": np.ones(
            (len(geometry["segment_indices"]),), dtype=np.float64,
        ),
        "candidate_homography": candidate_H,
        "raw_wall_outline_px": raw_wall,
        "wall_local_fit_outline_px": local_wall,
        "depth_global_candidate_wall_outline_px": apply_homography(raw_wall, candidate_H),
        "depth_global_fitted_wall_outline_px": apply_homography(raw_wall, accepted_H),
        "depth_component_count": geometry["component_count"],
        "depth_boundary_point_count": geometry["point_count"],
        "depth_silhouette_area_px": geometry["area_px"],
        "depth_image_border_wrapper_segment_count": int(
            geometry.get("image_border_wrapper_segment_count", 0)
        ),
        "depth_frame_wrappers_filtered": bool(
            geometry.get("frame_wrappers_filtered", False)
        ),
        "fit_geometry_source": (
            "visible_semantic_projected_edges"
            if use_semantic_guides
            else "depth_silhouette_fallback"
        ),
        "fit_original_points": fit_original_points,
        "fit_candidate_points": fit_candidate_points,
        "fit_fitted_points": fit_fitted_points,
        "fit_segment_indices": fit_segments,
        "fit_segment_classes": fit_segment_classes,
        "fit_segment_weights": np.asarray(
            fit_segment_weights
            if fit_segment_weights is not None
            else np.ones((len(fit_segments),), dtype=np.float64),
            dtype=np.float64,
        ),
        "semantic_segment_classes": semantic_classes if use_semantic_guides else [],
        "semantic_class_weights": {
            key: float(value) for key, value in semantic_class_weights.items()
        },
        "semantic_visible_length_px_by_class": {
            key: float(value) for key, value in class_lengths.items()
        },
        "semantic_boundary_diagnostics": dict(semantic_boundary_geometry or {}),
        "semantic_image_guidance": semantic_image_guidance_metadata,
        "semantic_image_guidance_used": bool(
            semantic_image_guidance_metadata.get("used_for_fitting", False)
            or fit.get("semantic_guidance_active", False)
        ),
        "semantic_image_segmentation_used": bool(
            semantic_image_guidance_metadata.get(
                "segmentation_available",
                False,
            )
            and semantic_image_guidance_metadata.get(
                "uses_semantic_guidance",
                False,
            )
        ),
        "semantic_image_boundary_map_classes": list(
            fit.get("semantic_boundary_map_classes", [])
        ),
        "valid_image_evidence_pixel_count": int(
            fit.get("valid_evidence_pixel_count", image_bgr.shape[0] * image_bgr.shape[1])
        ),
        "excluded_image_evidence_pixel_count": int(
            fit.get("excluded_evidence_pixel_count", 0)
        ),
    })
    return fit


def depth_boundary_fit_metadata(result: Optional[Dict[str, object]]):
    if not result:
        return None
    metadata = wireframe_fit_metadata(result) or {}
    semantic_diagnostics = dict(result.get("semantic_boundary_diagnostics", {}))
    semantic_diagnostics = {
        key: value
        for key, value in semantic_diagnostics.items()
        if key not in {"points", "segment_indices", "segment_classes"}
    }
    metadata.update({
        "method": str(result.get(
            "fit_geometry_source",
            "depth_silhouette_fallback",
        )),
        "uses_segmentation": bool(
            result.get("semantic_image_segmentation_used", False)
        ),
        "downstream_authoritative": False,
        "whole_model_transform": True,
        "fit_geometry_source": str(result.get(
            "fit_geometry_source",
            "depth_silhouette_fallback",
        )),
        "depth_component_count": int(result.get("depth_component_count", 0)),
        "depth_boundary_point_count": int(result.get("depth_boundary_point_count", 0)),
        "depth_silhouette_area_px": int(result.get("depth_silhouette_area_px", 0)),
        "depth_frame_wrappers_filtered": bool(
            result.get("depth_frame_wrappers_filtered", False)
        ),
        "depth_image_border_wrapper_segment_count": int(
            result.get("depth_image_border_wrapper_segment_count", 0)
        ),
        "semantic_segment_count": int(len(result.get("semantic_segment_classes", []))),
        "semantic_segment_count_by_class": {
            edge_class: int(list(result.get("semantic_segment_classes", [])).count(edge_class))
            for edge_class in SEMANTIC_BOUNDARY_CLASSES
        },
        "semantic_class_weights": dict(result.get("semantic_class_weights", {})),
        "semantic_boundary_diagnostics": semantic_diagnostics,
        "semantic_image_guidance": dict(
            result.get("semantic_image_guidance", {})
        ),
        "semantic_image_boundary_map_classes": list(
            result.get("semantic_image_boundary_map_classes", [])
        ),
        "semantic_image_guidance_used": bool(
            result.get("semantic_image_guidance_used", False)
        ),
        "semantic_image_segmentation_used": bool(
            result.get("semantic_image_segmentation_used", False)
        ),
        "selected_source_osm_refit": bool(
            result.get("selected_source_osm_refit", False)
        ),
        "selected_source_osm_refit_numerical_fit_applied": bool(
            result.get("selected_source_osm_refit_numerical_fit_applied", False)
        ),
        "selected_source_osm_refit_identity_fallback": bool(
            result.get("selected_source_osm_refit_identity_fallback", False)
        ),
        "valid_image_evidence_pixel_count": int(
            result.get("valid_image_evidence_pixel_count", 0)
        ),
        "excluded_image_evidence_pixel_count": int(
            result.get("excluded_image_evidence_pixel_count", 0)
        ),
        "osm_excluded_image_evidence_pixel_count": int(
            result.get("osm_excluded_image_evidence_pixel_count", 0)
        ),
        "semantic_or_locality_excluded_image_evidence_pixel_count": int(
            result.get(
                "semantic_or_locality_excluded_image_evidence_pixel_count",
                0,
            )
        ),
        "excluded_image_evidence_column_count": int(
            result.get("excluded_image_evidence_column_count", 0)
        ),
        "H_candidate": np.asarray(
            result.get("candidate_homography", np.eye(3)), dtype=np.float64
        ).astype(float).tolist(),
        "raw_wall_outline_px": np.asarray(
            result.get("raw_wall_outline_px", []), dtype=np.float64
        ).astype(float).tolist(),
        "wall_local_fit_outline_px": np.asarray(
            result.get("wall_local_fit_outline_px", []), dtype=np.float64
        ).astype(float).tolist(),
        "depth_global_candidate_wall_outline_px": np.asarray(
            result.get("depth_global_candidate_wall_outline_px", []), dtype=np.float64
        ).astype(float).tolist(),
        "depth_global_fitted_wall_outline_px": np.asarray(
            result.get("depth_global_fitted_wall_outline_px", []), dtype=np.float64
        ).astype(float).tolist(),
    })
    recovery = result.get("background_aware_recovery")
    if isinstance(recovery, Mapping):
        metadata["background_aware_recovery"] = dict(recovery)
    return metadata


def _draw_segments(
    image: np.ndarray,
    points: np.ndarray,
    segment_indices: Sequence[Tuple[int, int]],
    config: WireframeFitConfig,
    style: OverlayLineStyle,
):
    segments = visible_segments_from_points(
        np.asarray(points, dtype=np.float64),
        segment_indices,
        image.shape[:2],
        config,
    )
    for segment in segments:
        draw_styled_line(
            image,
            segment.start,
            segment.end,
            style,
            color_space="bgr",
        )


def _style_with_legacy_overrides(
    style: OverlayLineStyle,
    *,
    line_thickness_px: Optional[int],
    dash_length_px: float,
    dash_gap_px: float,
) -> OverlayLineStyle:
    """Keep the public overlay sizing knobs while sharing visual semantics."""
    return replace(
        style,
        width_px=(
            style.width_px
            if line_thickness_px is None
            else max(1, int(line_thickness_px))
        ),
        dash_length_px=max(float(dash_length_px), 1.0),
        dash_gap_px=max(float(dash_gap_px), 0.0),
    )


def create_depth_boundary_fit_overlay(
    image_bgr: np.ndarray,
    result: Dict[str, object],
    fit_config: WireframeFitConfig,
    *,
    line_thickness_px: Optional[int] = None,
    dash_length_px: float = ACCEPTED_MODEL_LINE.dash_length_px,
    dash_gap_px: float = ACCEPTED_MODEL_LINE.dash_gap_px,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8).copy()
    if result.get("depth_frame_wrappers_filtered", False):
        segments = list(result["segment_indices"])
    else:
        segments, _excluded_border_segments = (
            filter_image_border_wrapper_segments(
                result["original_points"],
                result["segment_indices"],
                image.shape[:2],
                epsilon_px=float(fit_config.image_border_epsilon_px),
            )
        )
    raw_style = _style_with_legacy_overrides(
        RAW_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    semantic_points = np.asarray(
        result.get("fit_original_points", []),
        dtype=np.float64,
    ).reshape(-1, 2)
    semantic_segments = list(result.get("fit_segment_indices", []))
    semantic_classes = list(result.get("semantic_segment_classes", []))
    semantic_styles = {
        "roof": SAM_ROOF_GUIDE_LINE,
        "wall": SAM_WALL_GUIDE_LINE,
        "base": SAM_BASE_GUIDE_LINE,
    }
    if (
        result.get("fit_geometry_source") == "visible_semantic_projected_edges"
        and len(semantic_segments) == len(semantic_classes)
    ):
        for edge_class in SEMANTIC_BOUNDARY_CLASSES:
            class_segments = [
                segment
                for segment, label in zip(semantic_segments, semantic_classes)
                if label == edge_class
            ]
            if class_segments:
                _draw_segments(
                    image,
                    semantic_points,
                    class_segments,
                    fit_config,
                    semantic_styles[edge_class],
                )
    _draw_segments(
        image,
        result["original_points"],
        segments,
        fit_config,
        raw_style,
    )
    shown = result["fitted_points"] if result.get("applied") else result["candidate_points"]
    shown_style = _style_with_legacy_overrides(
        ACCEPTED_MODEL_LINE if result.get("applied") else REJECTED_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    _draw_segments(
        image,
        shown,
        segments,
        fit_config,
        shown_style,
    )
    status = "accepted" if result.get("applied") else f"candidate only: {result.get('reason')}"
    transform = result.get("transform", {})
    guide_label = (
        "fit guides: yellow roof (3.0x) | green wall (2.0x) | gray base (0.35x)"
        if result.get("fit_geometry_source") == "visible_semantic_projected_edges"
        else "fit guides unavailable: using uniform depth-silhouette boundary"
    )
    weights = dict(result.get("semantic_class_weights", {}))
    if result.get("fit_geometry_source") == "visible_semantic_projected_edges":
        guide_label = (
            "fit guides: yellow roof "
            f"({float(weights.get('roof', 3.0)):.2g}x) | green wall "
            f"({float(weights.get('wall', 2.0)):.2g}x) | gray base "
            f"({float(weights.get('base', 0.35)):.2g}x)"
        )
    image_semantic_label = None
    if result.get("semantic_image_guidance_used", False):
        guidance = dict(result.get("semantic_image_guidance", {}))
        roles = dict(guidance.get("roles", {}))
        if result.get("semantic_image_segmentation_used", False):
            image_semantic_label = (
                "SAM3 pre-fit guidance: "
                f"target={int(dict(roles.get('building', {})).get('selected_pixels', 0))}px | "
                f"sky={int(dict(roles.get('sky', {})).get('selected_pixels', 0))}px | "
                f"occluder={int(guidance.get('excluded_occluder_pixels', 0))}px | "
                f"semantic score={float(result.get('semantic_boundary_score_after', 0.0)):.3f}"
            )
        else:
            image_semantic_label = (
                "projection-local evidence guard active; "
                "SAM3 semantic masks unavailable or not associated"
            )
    outline_label = model_projection_legend(
        fitted=bool(result.get("applied")),
        rejected=not bool(result.get("applied")),
    )
    rows = [
        outline_label,
        guide_label,
        *([image_semantic_label] if image_semantic_label else []),
        *(
            ["checkerboard side crop: OSM obstruction excluded from selected-source refit"]
            if result.get("selected_source_osm_refit")
            else []
        ),
        (
            f"{status} | scale={float(transform.get('scale', 1.0)):.4f} "
            f"tx={float(transform.get('tx', 0.0)):.1f}px "
            f"ty={float(transform.get('ty', 0.0)):.1f}px "
            f"gain={float(result.get('score_improvement', 0.0)):.4f}"
        ),
    ]
    draw_legend(image, rows, color_space="bgr")
    return image


def create_depth_silhouette_shift_overlay(
    result: Dict[str, object],
    fit_config: WireframeFitConfig,
    *,
    line_thickness_px: Optional[int] = None,
    dash_length_px: float = ACCEPTED_MODEL_LINE.dash_length_px,
    dash_gap_px: float = ACCEPTED_MODEL_LINE.dash_gap_px,
) -> np.ndarray:
    """Show the raw silhouette mask with its fitted boundary displacement."""
    raw_mask = np.asarray(result["raw_depth_mask"], dtype=bool)
    if raw_mask.ndim != 2:
        raise ValueError("Raw depth silhouette mask must be two-dimensional.")

    image = np.repeat(
        (raw_mask.astype(np.uint8) * 255)[:, :, None],
        3,
        axis=2,
    )
    shown = result["fitted_points"] if result.get("applied") else result["candidate_points"]
    if result.get("depth_frame_wrappers_filtered", False):
        segments = list(result["segment_indices"])
    else:
        segments, _excluded_border_segments = (
            filter_image_border_wrapper_segments(
                result["original_points"],
                result["segment_indices"],
                raw_mask.shape,
                epsilon_px=float(fit_config.image_border_epsilon_px),
            )
        )
    raw_style = _style_with_legacy_overrides(
        RAW_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    shown_style = _style_with_legacy_overrides(
        ACCEPTED_MODEL_LINE if result.get("applied") else REJECTED_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    _draw_segments(
        image,
        result["original_points"],
        segments,
        fit_config,
        raw_style,
    )
    _draw_segments(
        image,
        shown,
        segments,
        fit_config,
        shown_style,
    )
    draw_legend(
        image,
        [
            model_projection_legend(
                fitted=bool(result.get("applied")),
                rejected=not bool(result.get("applied")),
            ),
            "white fill=raw whole-model depth silhouette",
        ],
        color_space="bgr",
    )
    return image


def create_wall_fit_comparison_overlay(
    image_bgr: np.ndarray,
    result: Dict[str, object],
    *,
    selected_mode: str = "wall_only",
    line_thickness_px: Optional[int] = None,
    dash_length_px: float = ACCEPTED_MODEL_LINE.dash_length_px,
    dash_gap_px: float = ACCEPTED_MODEL_LINE.dash_gap_px,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8).copy()
    wall_local = np.round(result["wall_local_fit_outline_px"]).astype(np.int32)
    depth_wall = np.asarray(
        result["depth_global_fitted_wall_outline_px"]
        if result.get("applied")
        else result["depth_global_candidate_wall_outline_px"],
        dtype=np.float64,
    )
    depth_wall = np.round(depth_wall).astype(np.int32)
    wall_style = _style_with_legacy_overrides(
        WALL_ONLY_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    wall_segments = [
        (index, (index + 1) % len(wall_local))
        for index in range(len(wall_local))
    ]
    comparison_fit_config = WireframeFitConfig(
        minimum_visible_segment_length_px=0.0,
        ignore_segments_on_image_border=False,
    )
    _draw_segments(
        image,
        wall_local,
        wall_segments,
        comparison_fit_config,
        wall_style,
    )
    depth_style = _style_with_legacy_overrides(
        ACCEPTED_MODEL_LINE if result.get("applied") else REJECTED_MODEL_LINE,
        line_thickness_px=line_thickness_px,
        dash_length_px=dash_length_px,
        dash_gap_px=dash_gap_px,
    )
    depth_segments = [
        (index, (index + 1) % len(depth_wall))
        for index in range(len(depth_wall))
    ]
    _draw_segments(
        image,
        depth_wall,
        depth_segments,
        comparison_fit_config,
        depth_style,
    )
    selected_mode = str(selected_mode).strip().lower()
    status = "accepted global fit" if result.get("applied") else "global candidate not accepted"
    downstream_label = (
        "depth-global"
        if selected_mode == "depth_global" and result.get("applied")
        else "wall-only"
    )
    rows = [
        (
            "model: solid blue=wall-only fit | dashed magenta=accepted "
            "depth-global fit"
            if result.get("applied")
            else "model: solid blue=wall-only fit | dashed orange=rejected "
            "depth-global candidate"
        ),
        f"{status} | downstream alignment: {downstream_label}",
    ]
    draw_legend(image, rows, color_space="bgr")
    return image
