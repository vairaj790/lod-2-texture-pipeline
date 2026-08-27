# -*- coding: utf-8 -*-
"""Image-space fitting for projected facade-group outlines.

The fitter preserves the complete outline shape and searches one global
similarity transform. It operates directly on arrays so production callers do
not need intermediate projection JSON files.
"""

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .diagnostic_overlay_style import (
    ACCEPTED_MODEL_LINE,
    RAW_MODEL_LINE,
    REJECTED_MODEL_LINE,
    OverlayLineStyle,
    draw_legend,
    draw_styled_line,
    model_projection_legend,
)


@dataclass(frozen=True)
class WireframeFitConfig:
    allow_rotation: bool = False

    coarse_scale_min: float = 0.88
    coarse_scale_max: float = 1.12
    coarse_scale_step: float = 0.025
    coarse_rotation_min_deg: float = -6.0
    coarse_rotation_max_deg: float = 6.0
    coarse_rotation_step_deg: float = 1.0
    coarse_tx_min: float = -70.0
    coarse_tx_max: float = 70.0
    coarse_tx_step: float = 8.0
    coarse_ty_min: float = -80.0
    coarse_ty_max: float = 80.0
    coarse_ty_step: float = 8.0

    fine_scale_radius: float = 0.030
    fine_scale_step: float = 0.005
    fine_rotation_radius_deg: float = 1.0
    fine_rotation_step_deg: float = 0.25
    fine_tx_radius: float = 8.0
    fine_tx_step: float = 1.5
    fine_ty_radius: float = 8.0
    fine_ty_step: float = 1.5

    boundary_sample_step_px: float = 2.5
    minimum_visible_segment_length_px: float = 10.0
    image_border_epsilon_px: float = 1.5
    ignore_segments_on_image_border: bool = True

    canny_low_threshold: int = 40
    canny_high_threshold: int = 125
    lsd_min_line_length_px: float = 28.0
    lsd_line_draw_thickness: int = 2
    edge_distance_sigma_px: float = 5.0
    line_distance_sigma_px: float = 5.0
    semantic_boundary_sigma_px: float = 6.0

    sky_hue_min: int = 80
    sky_hue_max: int = 135
    sky_saturation_min: int = 30
    sky_value_min: int = 120

    weight_edge_distance: float = 1.70
    weight_long_line_distance: float = 1.00
    weight_gradient: float = 0.35
    weight_semantic_boundary: float = 1.35
    weight_visible_length: float = 0.45
    weight_scale_prior: float = 0.18
    weight_translation_prior: float = 0.14
    weight_rotation_prior: float = 0.08

    scale_prior_log_sigma: float = 0.16
    translation_prior_sigma_x: float = 90.0
    translation_prior_sigma_y: float = 85.0
    rotation_prior_sigma_deg: float = 7.0

    # Optional hard anchor envelope.  The allowed translation is the smaller
    # of the absolute cap and a reference-span fraction, with a small floor for
    # low-resolution/distant targets. ``None`` keeps legacy behavior.
    maximum_translation_x_px: Optional[float] = None
    maximum_translation_y_px: Optional[float] = None
    maximum_translation_norm_px: Optional[float] = None
    maximum_translation_norm_fraction: Optional[float] = None
    maximum_translation_fraction_x: Optional[float] = None
    maximum_translation_fraction_y: Optional[float] = None
    minimum_translation_allowance_x_px: float = 0.0
    minimum_translation_allowance_y_px: float = 0.0
    maximum_mean_displacement_px: Optional[float] = None
    maximum_mean_displacement_fraction: Optional[float] = None
    minimum_anchor_iou: float = 0.0

    # Production acceptance gate. The standalone experiment always returned
    # the numerical winner; production only applies a useful improvement.
    minimum_score_improvement: float = 0.025
    minimum_mean_vertex_displacement_px: float = 4.0
    maximum_edge_score_drop: float = 0.02
    maximum_line_score_drop: float = 0.04
    maximum_semantic_score_drop: float = 0.03
    # With masked image evidence, a candidate must retain most of the boundary
    # samples available at the starting pose. Otherwise it can obtain a high
    # average score simply by moving difficult edges into the excluded region.
    minimum_evidence_retention_ratio: float = 0.80
    minimum_masked_evidence_sample_count: int = 8


@dataclass(frozen=True)
class VisibleSegment:
    start: np.ndarray
    end: np.ndarray
    length: float
    source_index0: int
    source_index1: int
    source_segment_index: int
    ignored_reason: Optional[str] = None


def make_production_fit_config(**overrides) -> WireframeFitConfig:
    config = WireframeFitConfig()
    if overrides:
        config = replace(config, **overrides)
    return config


def _make_range(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Wireframe fit range step must be positive.")
    count = int(math.floor((stop - start) / step)) + 1
    values = start + np.arange(count + 1, dtype=np.float32) * step
    return values[values <= stop + 1e-6]


def _scaled_search_config(config: WireframeFitConfig, image_shape_hw) -> WireframeFitConfig:
    height, width = image_shape_hw
    factor = float(np.clip(max(height, width) / 640.0, 0.75, 3.0))
    if abs(factor - 1.0) < 1e-6:
        return config
    return replace(
        config,
        coarse_tx_min=config.coarse_tx_min * factor,
        coarse_tx_max=config.coarse_tx_max * factor,
        coarse_tx_step=config.coarse_tx_step * factor,
        coarse_ty_min=config.coarse_ty_min * factor,
        coarse_ty_max=config.coarse_ty_max * factor,
        coarse_ty_step=config.coarse_ty_step * factor,
        fine_tx_radius=config.fine_tx_radius * factor,
        fine_tx_step=config.fine_tx_step * factor,
        fine_ty_radius=config.fine_ty_radius * factor,
        fine_ty_step=config.fine_ty_step * factor,
        minimum_visible_segment_length_px=config.minimum_visible_segment_length_px * factor,
        boundary_sample_step_px=config.boundary_sample_step_px * factor,
        lsd_min_line_length_px=config.lsd_min_line_length_px * factor,
        edge_distance_sigma_px=config.edge_distance_sigma_px * factor,
        line_distance_sigma_px=config.line_distance_sigma_px * factor,
        semantic_boundary_sigma_px=config.semantic_boundary_sigma_px * factor,
        translation_prior_sigma_x=config.translation_prior_sigma_x * factor,
        translation_prior_sigma_y=config.translation_prior_sigma_y * factor,
        maximum_translation_x_px=(
            None
            if config.maximum_translation_x_px is None
            else config.maximum_translation_x_px * factor
        ),
        maximum_translation_y_px=(
            None
            if config.maximum_translation_y_px is None
            else config.maximum_translation_y_px * factor
        ),
        maximum_translation_norm_px=(
            None
            if config.maximum_translation_norm_px is None
            else config.maximum_translation_norm_px * factor
        ),
        minimum_translation_allowance_x_px=(
            config.minimum_translation_allowance_x_px * factor
        ),
        minimum_translation_allowance_y_px=(
            config.minimum_translation_allowance_y_px * factor
        ),
        maximum_mean_displacement_px=(
            None
            if config.maximum_mean_displacement_px is None
            else config.maximum_mean_displacement_px * factor
        ),
        minimum_mean_vertex_displacement_px=(
            config.minimum_mean_vertex_displacement_px * max(factor, 1.0)
        ),
    )


def _segment_indices(point_count: int) -> List[Tuple[int, int]]:
    if point_count < 3:
        raise ValueError("A facade wireframe needs at least three points.")
    return [(i, i + 1) for i in range(point_count - 1)] + [(point_count - 1, 0)]


def _validated_segment_indices(point_count, segment_indices):
    if segment_indices is None:
        return _segment_indices(point_count)
    validated = []
    for pair in segment_indices:
        if len(pair) != 2:
            raise ValueError("Each wireframe segment must contain exactly two point indices.")
        index0, index1 = int(pair[0]), int(pair[1])
        if not (0 <= index0 < point_count and 0 <= index1 < point_count):
            raise ValueError("Wireframe segment index is outside the point array.")
        if index0 == index1:
            continue
        validated.append((index0, index1))
    if not validated:
        raise ValueError("At least one valid wireframe segment is required.")
    return validated


def _clip_segment_to_image(p0, p1, width, height):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    p_values = (-dx, dx, -dy, dy)
    q_values = (x0, float(width - 1) - x0, y0, float(height - 1) - y0)
    u1, u2 = 0.0, 1.0
    for p_value, q_value in zip(p_values, q_values):
        if abs(p_value) < 1e-12:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > u2:
                return None
            u1 = max(u1, ratio)
        else:
            if ratio < u1:
                return None
            u2 = min(u2, ratio)
    start = np.array([x0 + u1 * dx, y0 + u1 * dy], dtype=np.float32)
    end = np.array([x0 + u2 * dx, y0 + u2 * dy], dtype=np.float32)
    return start, end


def _same_image_border(p0, p1, width, height, eps):
    return bool(
        (abs(float(p0[0])) <= eps and abs(float(p1[0])) <= eps)
        or (abs(float(p0[0]) - (width - 1)) <= eps and abs(float(p1[0]) - (width - 1)) <= eps)
        or (abs(float(p0[1])) <= eps and abs(float(p1[1])) <= eps)
        or (abs(float(p0[1]) - (height - 1)) <= eps and abs(float(p1[1]) - (height - 1)) <= eps)
    )


def visible_segments_from_points(
    points: np.ndarray,
    segment_indices: Sequence[Tuple[int, int]],
    image_shape_hw: Tuple[int, int],
    config: WireframeFitConfig,
    include_ignored: bool = False,
) -> List[VisibleSegment]:
    height, width = image_shape_hw
    segments = []
    for source_segment_index, (index0, index1) in enumerate(segment_indices):
        clipped = _clip_segment_to_image(points[index0], points[index1], width, height)
        if clipped is None:
            continue
        start, end = clipped
        length = float(np.linalg.norm(end - start))
        reason = None
        if length < config.minimum_visible_segment_length_px:
            reason = "too_short_after_clipping"
        elif config.ignore_segments_on_image_border and _same_image_border(
            start, end, width, height, config.image_border_epsilon_px
        ):
            reason = "image_border_wrapper_segment"
        if reason is None or include_ignored:
            segments.append(VisibleSegment(
                start,
                end,
                length,
                int(index0),
                int(index1),
                int(source_segment_index),
                reason,
            ))
    return segments


def _sample_segment(start, end, step_px):
    length = float(np.linalg.norm(end - start))
    count = max(2, int(length / max(step_px, 0.5)) + 1)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]


def _sample_visible_segments(segments, step_px):
    samples = [
        _sample_segment(segment.start, segment.end, step_px)
        for segment in segments
        if segment.ignored_reason is None
    ]
    return np.vstack(samples).astype(np.float32) if samples else np.empty((0, 2), np.float32)


def _sample_visible_segments_with_weights(segments, step_px, segment_weights):
    samples, weights, _source_segment_indices = (
        _sample_visible_segments_with_weights_and_indices(
            segments,
            step_px,
            segment_weights,
        )
    )
    return samples, weights


def _sample_visible_segments_with_weights_and_indices(
    segments,
    step_px,
    segment_weights,
):
    samples = []
    weights = []
    source_segment_indices = []
    for segment in segments:
        if segment.ignored_reason is not None:
            continue
        segment_samples = _sample_segment(segment.start, segment.end, step_px)
        samples.append(segment_samples)
        weights.append(np.full(
            (len(segment_samples),),
            float(segment_weights[segment.source_segment_index]),
            dtype=np.float32,
        ))
        source_segment_indices.append(np.full(
            (len(segment_samples),),
            int(segment.source_segment_index),
            dtype=np.int32,
        ))
    if not samples:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int32),
        )
    return (
        np.vstack(samples).astype(np.float32),
        np.concatenate(weights),
        np.concatenate(source_segment_indices),
    )


def _visible_length(segments):
    return float(sum(s.length for s in segments if s.ignored_reason is None))


def _weighted_visible_length(segments, segment_weights):
    return float(sum(
        segment.length * float(segment_weights[segment.source_segment_index])
        for segment in segments
        if segment.ignored_reason is None
    ))


def _validated_segment_weights(segment_count, segment_weights):
    if segment_weights is None:
        return np.ones((int(segment_count),), dtype=np.float64)
    weights = np.asarray(segment_weights, dtype=np.float64).reshape(-1)
    if len(weights) != int(segment_count):
        raise ValueError("Segment weights must match the number of wireframe segments.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Segment weights must be finite and non-negative.")
    if not np.any(weights > 0.0):
        raise ValueError("At least one wireframe segment weight must be positive.")
    return weights


def transform_points_similarity(points, scale, rotation_deg, tx, ty, center):
    angle = math.radians(float(rotation_deg))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    transformed = ((np.asarray(points, dtype=np.float64) - center) * float(scale)) @ rotation.T
    return transformed + center + np.array([float(tx), float(ty)], dtype=np.float64)


def similarity_homography(scale, rotation_deg, tx, ty, center):
    angle = math.radians(float(rotation_deg))
    linear = float(scale) * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    center = np.asarray(center, dtype=np.float64)
    offset = center + np.array([float(tx), float(ty)]) - linear @ center
    return np.array(
        [[linear[0, 0], linear[0, 1], offset[0]],
         [linear[1, 0], linear[1, 1], offset[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def apply_homography(points, homography):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones((len(points),), dtype=np.float64)])
    mapped = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def _choose_transform_center(
    points,
    segment_indices,
    image_shape_hw,
    config,
    segment_weights,
):
    visible = visible_segments_from_points(points, segment_indices, image_shape_hw, config)
    samples, weights = _sample_visible_segments_with_weights(
        visible,
        5.0,
        segment_weights,
    )
    if len(samples) and float(weights.sum()) > 0.0:
        return np.average(samples, axis=0, weights=weights).astype(np.float64)
    return points.mean(axis=0)


def _create_score_maps(image_bgr, config, valid_evidence_mask=None):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, config.canny_low_threshold, config.canny_high_threshold)
    line_map = np.zeros_like(canny)
    detected = cv2.createLineSegmentDetector(0).detect(gray)[0]
    if detected is not None:
        for line in detected[:, 0, :]:
            x1, y1, x2, y2 = [float(value) for value in line]
            if math.hypot(x2 - x1, y2 - y1) < config.lsd_min_line_length_px:
                continue
            cv2.line(
                line_map,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                255,
                config.lsd_line_draw_thickness,
                cv2.LINE_AA,
            )
    valid_mask = None
    if valid_evidence_mask is not None:
        valid_mask = np.asarray(valid_evidence_mask, dtype=bool)
        if valid_mask.shape != gray.shape:
            raise ValueError(
                "Wireframe fitting evidence mask must match the image height and width."
            )
        canny[~valid_mask] = 0
        line_map[~valid_mask] = 0
    combined = cv2.bitwise_or(canny, line_map)
    distance_to_edge = cv2.distanceTransform(255 - combined, cv2.DIST_L2, 3)
    distance_to_line = cv2.distanceTransform(255 - line_map, cv2.DIST_L2, 3)
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(cv2.magnitude(gx, gy), (3, 3), 0)
    gradient = cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)
    if valid_mask is not None:
        gradient[~valid_mask] = 0.0
    score_maps = np.dstack([distance_to_edge, distance_to_line, gradient]).astype(np.float32)
    return canny, line_map, score_maps


def _validated_segment_classes(segment_count, segment_classes):
    if segment_classes is None:
        return ["silhouette"] * int(segment_count)
    classes = [str(value).strip().lower() for value in segment_classes]
    if len(classes) != int(segment_count):
        raise ValueError(
            "Segment classes must match the number of wireframe segments."
        )
    return classes


def _create_semantic_boundary_score_maps(
    semantic_boundary_maps: Optional[Mapping[str, np.ndarray]],
    image_shape_hw,
    config,
):
    if not semantic_boundary_maps:
        return {}
    height, width = image_shape_hw
    score_maps = {}
    for raw_label, raw_map in semantic_boundary_maps.items():
        label = str(raw_label).strip().lower()
        if not label:
            continue
        values = np.asarray(raw_map)
        if values.shape != (height, width):
            raise ValueError(
                f"Semantic boundary map '{label}' must match the image height and width."
            )
        if values.dtype == bool or np.issubdtype(values.dtype, np.integer):
            boundary = values.astype(bool)
            if not boundary.any():
                continue
            distance = cv2.distanceTransform(
                (~boundary).astype(np.uint8),
                cv2.DIST_L2,
                3,
            )
            score = np.exp(
                -distance / max(float(config.semantic_boundary_sigma_px), 1.0e-6)
            )
        else:
            score = np.asarray(values, dtype=np.float32)
            if not np.isfinite(score).all():
                raise ValueError(
                    f"Semantic boundary map '{label}' contains non-finite values."
                )
            score = np.clip(score, 0.0, 1.0)
            if not np.any(score > 0.0):
                continue
        score_maps[label] = np.asarray(score, dtype=np.float32)
    return score_maps


def _prepare_semantic_boundary_sampling(
    semantic_score_maps,
    segment_classes,
):
    if not semantic_score_maps:
        return None, np.full((len(segment_classes),), -1, dtype=np.int16)
    labels = sorted(semantic_score_maps)
    label_to_channel = {
        label: index for index, label in enumerate(labels)
    }
    score_stack = np.stack(
        [semantic_score_maps[label] for label in labels],
        axis=2,
    ).astype(np.float32)
    fallback_channel = label_to_channel.get("silhouette", -1)
    segment_channels = np.asarray([
        label_to_channel.get(label, fallback_channel)
        for label in segment_classes
    ], dtype=np.int16)
    return score_stack, segment_channels


def _sample_semantic_boundary_scores(
    semantic_score_stack,
    points_xy,
    source_segment_indices,
    segment_semantic_channels,
):
    scores = np.zeros((len(points_xy),), dtype=np.float32)
    available = np.zeros((len(points_xy),), dtype=bool)
    if semantic_score_stack is None or len(points_xy) == 0:
        return scores, available

    source_segment_indices = np.asarray(source_segment_indices, dtype=np.int32)
    sample_channels = np.asarray(segment_semantic_channels, dtype=np.int16)[
        source_segment_indices
    ]
    sampled = cv2.remap(
        semantic_score_stack,
        points_xy[:, 0].astype(np.float32).reshape(1, -1),
        points_xy[:, 1].astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    channel_count = int(semantic_score_stack.shape[2])
    sampled = np.asarray(sampled, dtype=np.float32).reshape(
        -1,
        channel_count,
    )
    available = sample_channels >= 0
    sample_indices = np.flatnonzero(available)
    if len(sample_indices) > 0:
        scores[sample_indices] = sampled[
            sample_indices,
            sample_channels[sample_indices],
        ]
    return scores, available


def _sample_score_maps(score_maps, points_xy):
    if len(points_xy) == 0:
        return np.empty((0, 3), dtype=np.float32)
    values = cv2.remap(
        score_maps,
        points_xy[:, 0].astype(np.float32).reshape(1, -1),
        points_xy[:, 1].astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return values.reshape(-1, 3).astype(np.float32)


def _sample_valid_evidence(valid_evidence_mask, points_xy):
    if valid_evidence_mask is None:
        return np.ones((len(points_xy),), dtype=bool)
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=bool)
    values = cv2.remap(
        np.asarray(valid_evidence_mask, dtype=np.uint8),
        points_xy[:, 0].astype(np.float32).reshape(1, -1),
        points_xy[:, 1].astype(np.float32).reshape(1, -1),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return values.reshape(-1) > 0


def semantic_boundary_alignment_score(
    points_px: np.ndarray,
    segment_indices: Sequence[Tuple[int, int]],
    segment_classes: Sequence[str],
    segment_weights: Sequence[float],
    semantic_boundary_maps: Mapping[str, np.ndarray],
    image_shape_hw: Tuple[int, int],
    *,
    config: Optional[WireframeFitConfig] = None,
    included_classes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Evaluate model segments against semantic guides on one common mask.

    This diagnostic is intentionally independent of raw-image evidence masks.
    It lets the pipeline compare an incumbent and a background-aware challenger
    against the exact same complete roof/sky guide before choosing either fit.
    """
    points = np.asarray(points_px, dtype=np.float64)
    shape = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Semantic alignment points must be an Nx2 array.")
    segments = _validated_segment_indices(len(points), segment_indices)
    classes = _validated_segment_classes(len(segments), segment_classes)
    weights = _validated_segment_weights(len(segments), segment_weights)
    selected_classes = (
        None
        if included_classes is None
        else {str(value).strip().lower() for value in included_classes}
    )
    if selected_classes is not None:
        keep = np.asarray(
            [label in selected_classes for label in classes],
            dtype=bool,
        )
        if not keep.any():
            return {"score": 0.0, "sample_count": 0, "classes": []}
        weights = weights.copy()
        weights[~keep] = 0.0

    score_config = _scaled_search_config(
        config or WireframeFitConfig(),
        shape,
    )
    semantic_score_maps = _create_semantic_boundary_score_maps(
        semantic_boundary_maps,
        shape,
        score_config,
    )
    score_stack, segment_channels = _prepare_semantic_boundary_sampling(
        semantic_score_maps,
        classes,
    )
    visible = visible_segments_from_points(
        points,
        segments,
        shape,
        score_config,
    )
    samples, sample_weights, source_indices = (
        _sample_visible_segments_with_weights_and_indices(
            visible,
            score_config.boundary_sample_step_px,
            weights,
        )
    )
    semantic_values, semantic_available = _sample_semantic_boundary_scores(
        score_stack,
        samples,
        source_indices,
        segment_channels,
    )
    usable = semantic_available & (sample_weights > 0.0)
    if not usable.any() or float(sample_weights[usable].sum()) <= 0.0:
        return {
            "score": 0.0,
            "sample_count": 0,
            "classes": sorted(selected_classes or set(classes)),
        }
    return {
        "score": float(np.average(
            semantic_values[usable],
            weights=sample_weights[usable],
        )),
        "sample_count": int(usable.sum()),
        "classes": sorted(selected_classes or set(classes)),
    }


def _anchored_translation_limits(points, config):
    points = np.asarray(points, dtype=np.float64)
    spans = np.ptp(points, axis=0) if len(points) else np.zeros((2,))

    def axis_limit(absolute_cap, fraction, minimum_allowance, span):
        candidates = []
        if absolute_cap is not None:
            candidates.append(float(max(0.0, absolute_cap)))
        if fraction is not None:
            relative = max(
                float(max(0.0, minimum_allowance)),
                float(max(0.0, fraction)) * float(max(0.0, span)),
            )
            candidates.append(relative)
        return min(candidates) if candidates else float("inf")

    reference_extent = float(max(spans[0], spans[1], 1.0))
    norm_candidates = []
    if config.maximum_translation_norm_px is not None:
        norm_candidates.append(float(max(0.0, config.maximum_translation_norm_px)))
    if config.maximum_translation_norm_fraction is not None:
        norm_candidates.append(
            float(max(0.0, config.maximum_translation_norm_fraction))
            * reference_extent
        )
    norm_limit = min(norm_candidates) if norm_candidates else float("inf")
    return (
        axis_limit(
            config.maximum_translation_x_px,
            config.maximum_translation_fraction_x,
            config.minimum_translation_allowance_x_px,
            spans[0],
        ),
        axis_limit(
            config.maximum_translation_y_px,
            config.maximum_translation_fraction_y,
            config.minimum_translation_allowance_y_px,
            spans[1],
        ),
        norm_limit,
        reference_extent,
    )


def _convex_anchor_iou(points0, points1, image_shape_hw):
    points0 = np.asarray(points0, dtype=np.float64)
    points1 = np.asarray(points1, dtype=np.float64)
    if len(points0) < 3 or len(points1) < 3:
        return 1.0
    height, width = image_shape_hw
    mask0 = np.zeros((height, width), dtype=np.uint8)
    mask1 = np.zeros((height, width), dtype=np.uint8)
    hull0 = cv2.convexHull(points0.astype(np.float32)).reshape(-1, 2)
    hull1 = cv2.convexHull(points1.astype(np.float32)).reshape(-1, 2)
    cv2.fillPoly(mask0, [np.round(hull0).astype(np.int32)], 1)
    cv2.fillPoly(mask1, [np.round(hull1).astype(np.int32)], 1)
    intersection = int(np.logical_and(mask0, mask1).sum())
    union = int(np.logical_or(mask0, mask1).sum())
    return float(intersection / union) if union > 0 else 1.0


def _score_candidate(
    candidate_points,
    segment_indices,
    scale,
    rotation_deg,
    tx,
    ty,
    image_shape_hw,
    reference_visible_length,
    score_maps,
    config,
    segment_weights,
    valid_evidence_mask,
    semantic_score_stack,
    segment_semantic_channels,
    semantic_valid_evidence_mask=None,
):
    def rejected_diagnostics(visible_length, excluded_sample_count=0):
        return {
            "rejected": 1.0,
            "visible_length": float(visible_length),
            "weighted_visible_length": 0.0,
            "visible_length_score": 0.0,
            "edge_distance_score": 0.0,
            "long_line_score": 0.0,
            "gradient_score": 0.0,
            "semantic_boundary_score": 0.0,
            "scale_prior": 0.0,
            "translation_prior": 0.0,
            "rotation_prior": 0.0,
            "visible_segment_count": 0,
            "evidence_sample_count": 0,
            "excluded_evidence_sample_count": int(excluded_sample_count),
            "semantic_evidence_sample_count": 0,
            "excluded_semantic_evidence_sample_count": 0,
        }

    visible = visible_segments_from_points(candidate_points, segment_indices, image_shape_hw, config)
    visible_length = _visible_length(visible)
    if visible_length < config.minimum_visible_segment_length_px:
        return -1.0e9, rejected_diagnostics(visible_length)
    weighted_visible_length = _weighted_visible_length(visible, segment_weights)
    (
        samples,
        sample_weights,
        source_segment_indices,
    ) = _sample_visible_segments_with_weights_and_indices(
        visible,
        config.boundary_sample_step_px,
        segment_weights,
    )
    values = _sample_score_maps(score_maps, samples)
    semantic_values, semantic_available = _sample_semantic_boundary_scores(
        semantic_score_stack,
        samples,
        source_segment_indices,
        segment_semantic_channels,
    )
    evidence_samples = _sample_valid_evidence(valid_evidence_mask, samples)
    excluded_sample_count = int(len(evidence_samples) - evidence_samples.sum())
    raw_values = values[evidence_samples]
    raw_sample_weights = sample_weights[evidence_samples]
    if len(raw_values) == 0 or float(raw_sample_weights.sum()) <= 0.0:
        return -1.0e9, rejected_diagnostics(
            visible_length,
            excluded_sample_count,
        )

    # Preserve the historical coupling unless a caller explicitly supplies a
    # semantic validity mask.  The background-aware fit uses an independent
    # mask so foreground pixels cannot contribute Canny/LSD evidence while the
    # complete SAM roof/sky guide remains available as semantic evidence.
    effective_semantic_valid_mask = (
        valid_evidence_mask
        if semantic_valid_evidence_mask is None
        else semantic_valid_evidence_mask
    )
    semantic_evidence_samples = _sample_valid_evidence(
        effective_semantic_valid_mask,
        samples,
    )
    excluded_semantic_sample_count = int(
        len(semantic_evidence_samples) - semantic_evidence_samples.sum()
    )

    edge_score = float(np.average(
        np.exp(-raw_values[:, 0] / config.edge_distance_sigma_px),
        weights=raw_sample_weights,
    ))
    line_score = float(np.average(
        np.exp(-raw_values[:, 1] / config.line_distance_sigma_px),
        weights=raw_sample_weights,
    ))
    gradient_score = float(np.average(
        raw_values[:, 2],
        weights=raw_sample_weights,
    ))
    semantic_available = (
        semantic_available
        & semantic_evidence_samples
        & (sample_weights > 0.0)
    )
    semantic_evidence_sample_count = int(semantic_available.sum())
    semantic_weight_sum = float(sample_weights[semantic_available].sum())
    semantic_score = (
        float(np.average(
            semantic_values[semantic_available],
            weights=sample_weights[semantic_available],
        ))
        if semantic_evidence_sample_count > 0 and semantic_weight_sum > 0.0
        else 0.0
    )
    length_score = float(min(
        weighted_visible_length / max(reference_visible_length, 1.0),
        1.25,
    ))
    scale_prior = float(math.exp(-((math.log(max(float(scale), 1e-6)) / config.scale_prior_log_sigma) ** 2)))
    translation_prior = float(math.exp(-(
        (float(tx) / config.translation_prior_sigma_x) ** 2
        + (float(ty) / config.translation_prior_sigma_y) ** 2
    )))
    rotation_prior = float(math.exp(-((float(rotation_deg) / config.rotation_prior_sigma_deg) ** 2)))
    score = (
        config.weight_edge_distance * edge_score
        + config.weight_long_line_distance * line_score
        + config.weight_gradient * gradient_score
        + config.weight_semantic_boundary * semantic_score
        + config.weight_visible_length * length_score
        + config.weight_scale_prior * scale_prior
        + config.weight_translation_prior * translation_prior
        + config.weight_rotation_prior * rotation_prior
    )
    diagnostics = {
        "score": float(score),
        "edge_distance_score": edge_score,
        "long_line_score": line_score,
        "gradient_score": gradient_score,
        "semantic_boundary_score": semantic_score,
        "visible_length": visible_length,
        "weighted_visible_length": weighted_visible_length,
        "visible_length_score": length_score,
        "scale_prior": scale_prior,
        "translation_prior": translation_prior,
        "rotation_prior": rotation_prior,
        "visible_segment_count": int(len(visible)),
        "evidence_sample_count": int(len(raw_values)),
        "excluded_evidence_sample_count": excluded_sample_count,
        "semantic_evidence_sample_count": semantic_evidence_sample_count,
        "excluded_semantic_evidence_sample_count": (
            excluded_semantic_sample_count
        ),
    }
    return float(score), diagnostics


def _search_best_transform(
    points,
    segment_indices,
    image_shape_hw,
    score_maps,
    config,
    segment_weights,
    valid_evidence_mask,
    semantic_score_stack,
    segment_semantic_channels,
    semantic_valid_evidence_mask=None,
):
    center = _choose_transform_center(
        points,
        segment_indices,
        image_shape_hw,
        config,
        segment_weights,
    )
    reference = visible_segments_from_points(points, segment_indices, image_shape_hw, config)
    reference_length = _weighted_visible_length(reference, segment_weights)
    if reference_length <= 0.0:
        raise RuntimeError("No real wireframe segments are visible in the image.")
    (
        maximum_tx,
        maximum_ty,
        maximum_translation_norm,
        reference_extent,
    ) = _anchored_translation_limits(points, config)

    identity_score, identity_diagnostics = _score_candidate(
        points, segment_indices, 1.0, 0.0, 0.0, 0.0,
        image_shape_hw, reference_length, score_maps, config, segment_weights,
        valid_evidence_mask,
        semantic_score_stack,
        segment_semantic_channels,
        semantic_valid_evidence_mask,
    )
    identity_evidence_sample_count = int(
        identity_diagnostics.get("evidence_sample_count", 0)
    )
    identity_diagnostics["evidence_retention_ratio"] = 1.0
    best = {
        "scale": 1.0,
        "rotation_deg": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "score": float(identity_score),
        **identity_diagnostics,
    }

    def evaluate(scale_values, rotation_values, tx_values, ty_values, current):
        best_result = current
        for scale in scale_values:
            for rotation in rotation_values:
                for tx in tx_values:
                    for ty in ty_values:
                        if abs(float(tx)) > maximum_tx + 1.0e-6:
                            continue
                        if abs(float(ty)) > maximum_ty + 1.0e-6:
                            continue
                        if math.hypot(float(tx), float(ty)) > (
                            maximum_translation_norm + 1.0e-6
                        ):
                            continue
                        candidate = transform_points_similarity(points, scale, rotation, tx, ty, center)
                        score, diagnostics = _score_candidate(
                            candidate, segment_indices, scale, rotation, tx, ty,
                            image_shape_hw, reference_length, score_maps, config,
                            segment_weights,
                            valid_evidence_mask,
                            semantic_score_stack,
                            segment_semantic_channels,
                            semantic_valid_evidence_mask,
                        )
                        if valid_evidence_mask is not None:
                            evidence_retention_ratio = float(
                                diagnostics.get("evidence_sample_count", 0)
                                / max(identity_evidence_sample_count, 1)
                            )
                            diagnostics["evidence_retention_ratio"] = (
                                evidence_retention_ratio
                            )
                            if evidence_retention_ratio < float(
                                config.minimum_evidence_retention_ratio
                            ):
                                continue
                            if int(diagnostics.get(
                                "evidence_sample_count",
                                0,
                            )) < int(config.minimum_masked_evidence_sample_count):
                                continue
                        if diagnostics.get(
                            "edge_distance_score",
                            0.0,
                        ) < (
                            identity_diagnostics.get(
                                "edge_distance_score",
                                0.0,
                            )
                            - config.maximum_edge_score_drop
                        ):
                            continue
                        if diagnostics.get(
                            "long_line_score",
                            0.0,
                        ) < (
                            identity_diagnostics.get(
                                "long_line_score",
                                0.0,
                            )
                            - config.maximum_line_score_drop
                        ):
                            continue
                        if (
                            identity_diagnostics.get(
                                "semantic_evidence_sample_count",
                                0,
                            ) > 0
                            and diagnostics.get(
                                "semantic_boundary_score",
                                0.0,
                            ) < (
                                identity_diagnostics.get(
                                    "semantic_boundary_score",
                                    0.0,
                                )
                                - config.maximum_semantic_score_drop
                            )
                        ):
                            continue
                        if score > best_result["score"]:
                            best_result = {
                                "scale": float(scale),
                                "rotation_deg": float(rotation),
                                "tx": float(tx),
                                "ty": float(ty),
                                "score": float(score),
                                **diagnostics,
                            }
        return best_result

    coarse_rotations: Iterable[float] = (
        _make_range(config.coarse_rotation_min_deg, config.coarse_rotation_max_deg, config.coarse_rotation_step_deg)
        if config.allow_rotation else np.array([0.0], dtype=np.float32)
    )
    best = evaluate(
        _make_range(config.coarse_scale_min, config.coarse_scale_max, config.coarse_scale_step),
        coarse_rotations,
        _make_range(config.coarse_tx_min, config.coarse_tx_max, config.coarse_tx_step),
        _make_range(config.coarse_ty_min, config.coarse_ty_max, config.coarse_ty_step),
        best,
    )
    fine_rotations: Iterable[float] = (
        _make_range(
            best["rotation_deg"] - config.fine_rotation_radius_deg,
            best["rotation_deg"] + config.fine_rotation_radius_deg,
            config.fine_rotation_step_deg,
        ) if config.allow_rotation else np.array([0.0], dtype=np.float32)
    )
    best = evaluate(
        _make_range(
            max(config.coarse_scale_min, best["scale"] - config.fine_scale_radius),
            min(config.coarse_scale_max, best["scale"] + config.fine_scale_radius),
            config.fine_scale_step,
        ),
        fine_rotations,
        _make_range(
            max(config.coarse_tx_min, best["tx"] - config.fine_tx_radius),
            min(config.coarse_tx_max, best["tx"] + config.fine_tx_radius),
            config.fine_tx_step,
        ),
        _make_range(
            max(config.coarse_ty_min, best["ty"] - config.fine_ty_radius),
            min(config.coarse_ty_max, best["ty"] + config.fine_ty_radius),
            config.fine_ty_step,
        ),
        best,
    )
    best["transform_center_x"] = float(center[0])
    best["transform_center_y"] = float(center[1])
    best["reference_visible_length"] = float(reference_length)
    best["maximum_translation_x_px"] = (
        float(maximum_tx) if math.isfinite(maximum_tx) else None
    )
    best["maximum_translation_y_px"] = (
        float(maximum_ty) if math.isfinite(maximum_ty) else None
    )
    best["maximum_translation_norm_px"] = (
        float(maximum_translation_norm)
        if math.isfinite(maximum_translation_norm) else None
    )
    best["reference_extent_px"] = float(reference_extent)
    return best, identity_diagnostics, center, segment_indices


def fit_wireframe_to_image(
    image_bgr: np.ndarray,
    outline_points_px: np.ndarray,
    config: Optional[WireframeFitConfig] = None,
    segment_indices: Optional[Sequence[Tuple[int, int]]] = None,
    segment_weights: Optional[Sequence[float]] = None,
    valid_evidence_mask: Optional[np.ndarray] = None,
    semantic_valid_evidence_mask: Optional[np.ndarray] = None,
    segment_classes: Optional[Sequence[str]] = None,
    semantic_boundary_maps: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Fit one projected facade outline and return accepted image geometry."""
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    points = np.asarray(outline_points_px, dtype=np.float64)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Wireframe fitting expects a BGR image with three channels.")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("Wireframe fitting expects an Nx2 point array with at least two points.")
    if not np.isfinite(points).all():
        raise ValueError("Wireframe outline contains non-finite coordinates.")

    search_config = _scaled_search_config(config or WireframeFitConfig(), image_bgr.shape[:2])
    segment_indices = _validated_segment_indices(len(points), segment_indices)
    segment_weights = _validated_segment_weights(len(segment_indices), segment_weights)
    segment_classes = _validated_segment_classes(
        len(segment_indices),
        segment_classes,
    )
    if valid_evidence_mask is not None:
        valid_evidence_mask = np.asarray(valid_evidence_mask, dtype=bool)
        if valid_evidence_mask.shape != image_bgr.shape[:2]:
            raise ValueError(
                "Wireframe fitting evidence mask must match the image height and width."
            )
    if semantic_valid_evidence_mask is not None:
        semantic_valid_evidence_mask = np.asarray(
            semantic_valid_evidence_mask,
            dtype=bool,
        )
        if semantic_valid_evidence_mask.shape != image_bgr.shape[:2]:
            raise ValueError(
                "Semantic fitting evidence mask must match the image height and width."
            )
    canny, line_map, score_maps = _create_score_maps(
        image_bgr,
        search_config,
        valid_evidence_mask,
    )
    semantic_score_maps = _create_semantic_boundary_score_maps(
        semantic_boundary_maps,
        image_bgr.shape[:2],
        search_config,
    )
    (
        semantic_score_stack,
        segment_semantic_channels,
    ) = _prepare_semantic_boundary_sampling(
        semantic_score_maps,
        segment_classes,
    )
    best, identity_diagnostics, center, segment_indices = _search_best_transform(
        points,
        segment_indices,
        image_bgr.shape[:2],
        score_maps,
        search_config,
        segment_weights,
        valid_evidence_mask,
        semantic_score_stack,
        segment_semantic_channels,
        semantic_valid_evidence_mask,
    )
    candidate_points = transform_points_similarity(
        points, best["scale"], best["rotation_deg"], best["tx"], best["ty"], center,
    )
    score_improvement = float(best["score"] - identity_diagnostics["score"])
    mean_vertex_displacement = float(
        np.mean(np.linalg.norm(candidate_points - points, axis=1))
    )
    reference_extent = float(max(
        np.ptp(points[:, 0]),
        np.ptp(points[:, 1]),
        1.0,
    ))
    translation_norm = float(math.hypot(best["tx"], best["ty"]))
    displacement_fraction = float(mean_vertex_displacement / reference_extent)
    displacement_limits = []
    if search_config.maximum_mean_displacement_px is not None:
        displacement_limits.append(float(
            search_config.maximum_mean_displacement_px
        ))
    if search_config.maximum_mean_displacement_fraction is not None:
        displacement_limits.append(
            float(search_config.maximum_mean_displacement_fraction)
            * reference_extent
        )
    maximum_mean_displacement = (
        min(displacement_limits) if displacement_limits else float("inf")
    )
    motion_ok = bool(
        mean_vertex_displacement <= maximum_mean_displacement + 1.0e-6
    )
    anchor_iou = _convex_anchor_iou(
        points,
        candidate_points,
        image_bgr.shape[:2],
    )
    anchor_ok = bool(anchor_iou + 1.0e-9 >= search_config.minimum_anchor_iou)
    edge_ok = best["edge_distance_score"] >= (
        identity_diagnostics["edge_distance_score"] - search_config.maximum_edge_score_drop
    )
    line_ok = best["long_line_score"] >= (
        identity_diagnostics["long_line_score"] - search_config.maximum_line_score_drop
    )
    semantic_guidance_active = bool(
        identity_diagnostics.get("semantic_evidence_sample_count", 0) > 0
        or best.get("semantic_evidence_sample_count", 0) > 0
    )
    semantic_ok = bool(
        not semantic_guidance_active
        or best.get("semantic_boundary_score", 0.0)
        >= (
            identity_diagnostics.get("semantic_boundary_score", 0.0)
            - search_config.maximum_semantic_score_drop
        )
    )
    evidence_ok = bool(
        valid_evidence_mask is None
        or (
            int(identity_diagnostics.get("evidence_sample_count", 0))
            >= int(search_config.minimum_masked_evidence_sample_count)
            and int(best.get("evidence_sample_count", 0))
            >= int(search_config.minimum_masked_evidence_sample_count)
        )
    )
    applied = bool(
        score_improvement >= search_config.minimum_score_improvement
        and mean_vertex_displacement >= search_config.minimum_mean_vertex_displacement_px
        and edge_ok
        and line_ok
        and semantic_ok
        and evidence_ok
        and motion_ok
        and anchor_ok
    )
    if applied:
        reason = "accepted_score_improvement"
        fitted_points = candidate_points
        homography = similarity_homography(
            best["scale"], best["rotation_deg"], best["tx"], best["ty"], center,
        )
    else:
        reason = (
            "motion_exceeds_original_projection_anchor"
            if not motion_ok
            else "original_projection_anchor_iou_below_threshold"
            if not anchor_ok
            else "insufficient_unmasked_boundary_evidence"
            if not evidence_ok
            else "semantic_boundary_score_drop"
            if not semantic_ok
            else "movement_below_materiality_threshold"
            if mean_vertex_displacement < search_config.minimum_mean_vertex_displacement_px
            else "insufficient_score_improvement"
        )
        fitted_points = points.copy()
        homography = np.eye(3, dtype=np.float64)

    return {
        "applied": applied,
        "reason": reason,
        "original_points": points,
        "candidate_points": candidate_points,
        "fitted_points": fitted_points,
        "homography": homography,
        "segment_indices": segment_indices,
        "segment_weights": segment_weights,
        "segment_classes": segment_classes,
        "transform": best,
        "score_before": float(identity_diagnostics["score"]),
        "score_after": float(best["score"]),
        "score_improvement": score_improvement,
        "mean_vertex_displacement_px": mean_vertex_displacement,
        "mean_vertex_displacement_fraction": displacement_fraction,
        "maximum_mean_vertex_displacement_px": (
            float(maximum_mean_displacement)
            if math.isfinite(maximum_mean_displacement) else None
        ),
        "translation_norm_px": translation_norm,
        "anchor_iou": anchor_iou,
        "anchor_motion_gate_passed": motion_ok,
        "anchor_iou_gate_passed": anchor_ok,
        "diagnostics_before": identity_diagnostics,
        "canny": canny,
        "line_map": line_map,
        "semantic_guidance_active": semantic_guidance_active,
        "semantic_boundary_map_classes": sorted(semantic_score_maps),
        "semantic_boundary_score_before": float(
            identity_diagnostics.get("semantic_boundary_score", 0.0)
        ),
        "semantic_boundary_score_after": float(
            best.get("semantic_boundary_score", 0.0)
        ),
        "masked_evidence_gate_passed": evidence_ok,
        "valid_evidence_pixel_count": (
            int(valid_evidence_mask.sum())
            if valid_evidence_mask is not None
            else int(image_bgr.shape[0] * image_bgr.shape[1])
        ),
        "excluded_evidence_pixel_count": (
            int(valid_evidence_mask.size - valid_evidence_mask.sum())
            if valid_evidence_mask is not None
            else 0
        ),
        "semantic_valid_evidence_pixel_count": (
            int(semantic_valid_evidence_mask.sum())
            if semantic_valid_evidence_mask is not None
            else (
                int(valid_evidence_mask.sum())
                if valid_evidence_mask is not None
                else int(image_bgr.shape[0] * image_bgr.shape[1])
            )
        ),
        "semantic_evidence_mask_independent": bool(
            semantic_valid_evidence_mask is not None
        ),
    }


def wireframe_fit_metadata(result: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not result:
        return None
    transform = dict(result.get("transform", {}))
    return {
        "model": "uniform_scale_rotation_translation",
        "shape_preserved": True,
        "applied": bool(result.get("applied", False)),
        "reason": str(result.get("reason", "unknown")),
        "scale": float(transform.get("scale", 1.0)),
        "rotation_deg": float(transform.get("rotation_deg", 0.0)),
        "tx_px": float(transform.get("tx", 0.0)),
        "ty_px": float(transform.get("ty", 0.0)),
        "transform_center_px": [
            float(transform.get("transform_center_x", 0.0)),
            float(transform.get("transform_center_y", 0.0)),
        ],
        "score_before": float(result.get("score_before", 0.0)),
        "score_after": float(result.get("score_after", 0.0)),
        "score_improvement": float(result.get("score_improvement", 0.0)),
        "mean_vertex_displacement_px": float(result.get("mean_vertex_displacement_px", 0.0)),
        "mean_vertex_displacement_fraction": float(
            result.get("mean_vertex_displacement_fraction", 0.0)
        ),
        "maximum_mean_vertex_displacement_px": result.get(
            "maximum_mean_vertex_displacement_px"
        ),
        "translation_norm_px": float(result.get("translation_norm_px", 0.0)),
        "anchor_iou": float(result.get("anchor_iou", 1.0)),
        "anchor_motion_gate_passed": bool(
            result.get("anchor_motion_gate_passed", True)
        ),
        "anchor_iou_gate_passed": bool(
            result.get("anchor_iou_gate_passed", True)
        ),
        "semantic_guidance_active": bool(
            result.get("semantic_guidance_active", False)
        ),
        "semantic_boundary_map_classes": list(
            result.get("semantic_boundary_map_classes", [])
        ),
        "semantic_boundary_score_before": float(
            result.get("semantic_boundary_score_before", 0.0)
        ),
        "semantic_boundary_score_after": float(
            result.get("semantic_boundary_score_after", 0.0)
        ),
        "masked_evidence_gate_passed": bool(
            result.get("masked_evidence_gate_passed", True)
        ),
        "valid_evidence_pixel_count": int(
            result.get("valid_evidence_pixel_count", 0)
        ),
        "semantic_valid_evidence_pixel_count": int(
            result.get("semantic_valid_evidence_pixel_count", 0)
        ),
        "semantic_evidence_mask_independent": bool(
            result.get("semantic_evidence_mask_independent", False)
        ),
        "H_original_to_fitted": np.asarray(result.get("homography", np.eye(3))).astype(float).tolist(),
        "original_outline_px": np.asarray(result.get("original_points", [])).astype(float).tolist(),
        "fitted_outline_px": np.asarray(result.get("fitted_points", [])).astype(float).tolist(),
        "candidate_outline_px": np.asarray(result.get("candidate_points", [])).astype(float).tolist(),
        "diagnostics_before": {
            key: float(value) if isinstance(value, (float, np.floating)) else int(value)
            for key, value in dict(result.get("diagnostics_before", {})).items()
        },
        "diagnostics_after": {
            key: float(value) if isinstance(value, (float, np.floating)) else int(value)
            for key, value in transform.items()
            if key not in {"transform_center_x", "transform_center_y"}
        },
    }


def _draw_segments(
    image,
    points,
    segment_indices,
    config,
    style: OverlayLineStyle,
):
    segments = visible_segments_from_points(
        np.asarray(points, dtype=np.float64), segment_indices, image.shape[:2], config,
    )
    for segment in segments:
        draw_styled_line(
            image,
            segment.start,
            segment.end,
            style,
            color_space="bgr",
        )


def create_wireframe_fit_overlay(image_bgr, result, config=None):
    """Draw the original projection and accepted/rejected fit consistently."""
    image = np.asarray(image_bgr, dtype=np.uint8).copy()
    config = _scaled_search_config(config or WireframeFitConfig(), image.shape[:2])
    segment_indices = result["segment_indices"]
    _draw_segments(
        image,
        result["original_points"],
        segment_indices,
        config,
        RAW_MODEL_LINE,
    )
    shown_points = result["fitted_points"] if result.get("applied") else result["candidate_points"]
    shown_style = ACCEPTED_MODEL_LINE if result.get("applied") else REJECTED_MODEL_LINE
    _draw_segments(image, shown_points, segment_indices, config, shown_style)
    transform = result["transform"]
    status = result.get(
        "status_label",
        "accepted" if result.get("applied") else "rejected candidate",
    )
    rows = [
        model_projection_legend(
            fitted=bool(result.get("applied")),
            rejected=not bool(result.get("applied")),
        ),
        (
            f"{status} | scale={transform['scale']:.4f} "
            f"tx={transform['tx']:.1f}px ty={transform['ty']:.1f}px "
            f"score gain={result['score_improvement']:.4f}"
        ),
    ]
    draw_legend(image, rows, color_space="bgr")
    return image
