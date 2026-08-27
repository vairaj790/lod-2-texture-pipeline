# -*- coding: utf-8 -*-
"""Projection-local semantic guidance for whole-model image-space fitting.

This module deliberately does not run a segmentation model.  It consumes the
boolean instance stacks produced by one and turns them into conservative,
canvas-aligned evidence for the global depth fit.

The public mask contract is:

* ``raw_projection_mask`` is a two-dimensional boolean mask in the processing
  image canvas.
* Role values may be ``H x W``, ``N x H x W``, or ``N x 1 x H x W`` arrays.
* Returned masks are two-dimensional boolean arrays with the same ``H x W``.
* Returned metadata contains no arrays and is safe to serialize as JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .diagnostic_overlay_style import (
    BACKGROUND_AWARE_SEMANTIC_LEGEND_ROWS,
    OUTSIDE_SEARCH_DIM_FACTOR,
    RAW_MODEL_LINE,
    SAM_BACKGROUND_CONTEXT_FILL_RGB,
    SAM_INFERRED_ROOF_BRIDGE_RGB,
    SAM_PROMPTED_OCCLUDER_FILL_RGB,
    SAM_SUPPRESSED_ROOF_GUIDE_RGB,
    SAM_TARGET_BUILDING_FILL_RGB,
    SAM_TARGET_ROOF_FILL_RGB,
    SEARCH_LEGEND_ROW,
    SEMANTIC_FILL_ALPHA,
    SEMANTIC_GUIDE_LINES,
    SEMANTIC_LEGEND_ROWS,
    STRICT_ROOF_AUDIT_LEGEND_ROW,
    OverlayLineStyle,
    draw_legend as draw_overlay_legend,
    draw_styled_line,
    model_projection_legend,
)


CANONICAL_PREFIT_ROLES: Tuple[str, ...] = (
    "building",
    "roof",
    "sky",
    "ground",
    "vegetation",
    "occluder",
    "generic_occluder",
)

_ROLE_ALIASES = {
    "building": "building",
    "buildings": "building",
    "facade": "building",
    "facades": "building",
    "building_facade": "building",
    "building_facades": "building",
    "building_wall": "building",
    "building_walls": "building",
    "house": "building",
    "roof": "roof",
    "roofs": "roof",
    "eave": "roof",
    "eaves": "roof",
    "sky": "sky",
    "clear_sky": "sky",
    "ground": "ground",
    "terrain": "ground",
    "road": "ground",
    "roads": "ground",
    "sidewalk": "ground",
    "pavement": "ground",
    "vegetation": "vegetation",
    "tree": "vegetation",
    "trees": "vegetation",
    "foliage": "vegetation",
    "bush": "vegetation",
    "bushes": "vegetation",
    "shrub": "vegetation",
    "shrubs": "vegetation",
    "occluder": "occluder",
    "occluders": "occluder",
    "occlusion": "occluder",
    "car": "occluder",
    "cars": "occluder",
    "vehicle": "occluder",
    "vehicles": "occluder",
    "person": "occluder",
    "people": "occluder",
    "pole": "occluder",
    "poles": "occluder",
    "street_furniture": "occluder",
    "generic_occluder": "generic_occluder",
    "generic_occluders": "generic_occluder",
    "foreground_object": "generic_occluder",
    "foreground_objects": "generic_occluder",
    "object_in_front_of_building": "generic_occluder",
}


@dataclass(frozen=True)
class PrefitSemanticGuidanceConfig:
    """Controls target association, locality, and semantic boundary creation."""

    search_dilation_px: int = 96
    minimum_instance_area_px: int = 80
    target_association_distance_px: float = 48.0
    target_min_overlap_pixels: int = 12
    target_min_overlap_fraction: float = 0.01
    target_min_local_fraction: float = 0.20
    target_relative_score_threshold: float = 0.30
    target_min_new_projection_pixels: int = 8
    target_max_instances_per_role: int = 8
    occluder_dilation_px: int = 3
    generic_non_target_enabled: bool = True
    generic_non_target_min_target_coverage: float = 0.20
    generic_non_target_projection_inset_px: int = 3
    generic_non_target_target_dilation_px: int = 2
    generic_non_target_min_component_area_px: int = 80
    generic_non_target_max_component_fraction: float = 0.20
    generic_non_target_max_total_fraction: float = 0.45
    generic_non_target_max_target_overlap_fraction: float = 0.15
    context_adjacency_px: int = 4
    envelope_tolerance_px: int = 2
    boundary_thickness_px: int = 2
    image_border_exclusion_px: int = 2
    # Opt-in roof guidance that is tied to the projected model top and to the
    # upper part of the selected building.  Keeping one explicit switch makes
    # the legacy association and boundary construction the default.
    strict_roof_guidance_enabled: bool = False
    strict_roof_projected_band_radius_px: int = 18
    strict_roof_upper_building_fraction: float = 0.48
    strict_roof_attachment_radius_px: int = 8
    strict_roof_min_band_pixels: int = 12
    strict_roof_min_band_span_fraction: float = 0.03
    strict_roof_min_attachment_pixels: int = 12
    strict_roof_max_explicit_foreground_fraction: float = 0.35
    strict_roof_context_radius_px: int = 3
    strict_roof_foreground_guard_radius_px: int = 4
    strict_roof_vegetation_projection_inset_px: int = 2
    strict_roof_vegetation_inside_offset_px: int = 8
    strict_roof_min_guide_component_pixels: int = 5
    strict_roof_bridge_enabled: bool = True
    strict_roof_bridge_min_endpoint_run_px: int = 3
    strict_roof_bridge_max_gap_px: int = 64
    strict_roof_bridge_domain_dilation_px: int = 2

    def __post_init__(self):
        integer_nonnegative = (
            "search_dilation_px",
            "minimum_instance_area_px",
            "target_min_overlap_pixels",
            "target_min_new_projection_pixels",
            "target_max_instances_per_role",
            "occluder_dilation_px",
            "generic_non_target_projection_inset_px",
            "generic_non_target_target_dilation_px",
            "generic_non_target_min_component_area_px",
            "context_adjacency_px",
            "envelope_tolerance_px",
            "boundary_thickness_px",
            "image_border_exclusion_px",
            "strict_roof_projected_band_radius_px",
            "strict_roof_attachment_radius_px",
            "strict_roof_min_band_pixels",
            "strict_roof_min_attachment_pixels",
            "strict_roof_context_radius_px",
            "strict_roof_foreground_guard_radius_px",
            "strict_roof_vegetation_projection_inset_px",
            "strict_roof_vegetation_inside_offset_px",
            "strict_roof_min_guide_component_pixels",
            "strict_roof_bridge_min_endpoint_run_px",
            "strict_roof_bridge_max_gap_px",
            "strict_roof_bridge_domain_dilation_px",
        )
        for name in integer_nonnegative:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.target_max_instances_per_role) < 1:
            raise ValueError("target_max_instances_per_role must be at least one.")
        if int(self.strict_roof_min_guide_component_pixels) < 1:
            raise ValueError(
                "strict_roof_min_guide_component_pixels must be at least one."
            )
        if int(self.strict_roof_bridge_min_endpoint_run_px) < 1:
            raise ValueError(
                "strict_roof_bridge_min_endpoint_run_px must be at least one."
            )
        if float(self.target_association_distance_px) < 0.0:
            raise ValueError("target_association_distance_px must be non-negative.")
        for name in (
            "target_min_overlap_fraction",
            "target_min_local_fraction",
            "target_relative_score_threshold",
            "generic_non_target_min_target_coverage",
            "generic_non_target_max_component_fraction",
            "generic_non_target_max_total_fraction",
            "generic_non_target_max_target_overlap_fraction",
            "strict_roof_upper_building_fraction",
            "strict_roof_min_band_span_fraction",
            "strict_roof_max_explicit_foreground_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


def _canonical_role_name(role: object) -> Optional[str]:
    normalized = str(role).strip().lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(normalized)


def _as_numpy(value):
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


def _normalize_mask_stack(
    value,
    expected_shape: Tuple[int, int],
    minimum_instance_area_px: int,
) -> Tuple[np.ndarray, Sequence[int], Dict[str, object]]:
    """Normalize one role input without allowing a bad stack to abort fitting."""
    height, width = expected_shape
    empty = np.zeros((0, height, width), dtype=bool)
    info: Dict[str, object] = {
        "status": "empty",
        "input_instances": 0,
        "nonempty_instances": 0,
        "retained_instances": 0,
        "discarded_small_instances": 0,
        "minimum_instance_area_px": int(minimum_instance_area_px),
        "input_shape": None,
    }
    if value is None:
        info["reason"] = "missing"
        return empty, [], info

    try:
        array = _as_numpy(value)
    except Exception as exc:
        info.update({
            "status": "ignored",
            "reason": f"array_conversion_failed: {type(exc).__name__}",
        })
        return empty, [], info

    info["input_shape"] = [int(v) for v in array.shape]
    if array.size == 0:
        info["reason"] = "empty_array"
        return empty, [], info

    if array.ndim == 4:
        if array.shape[1] == 1:
            array = array[:, 0, :, :]
        elif array.shape[-1] == 1:
            array = array[:, :, :, 0]
        else:
            info.update({
                "status": "ignored",
                "reason": "four_dimensional_stack_requires_singleton_channel",
            })
            return empty, [], info
    elif array.ndim == 3 and array.shape[:2] == expected_shape and array.shape[2] == 1:
        array = array[:, :, 0][None, :, :]
    elif array.ndim == 2:
        array = array[None, :, :]

    if array.ndim != 3 or tuple(array.shape[1:]) != expected_shape:
        info.update({
            "status": "ignored",
            "reason": (
                f"shape_mismatch_expected_Nx{height}x{width}"
            ),
        })
        return empty, [], info

    info["input_instances"] = int(array.shape[0])
    if array.dtype == np.bool_:
        boolean = array.copy()
    else:
        finite = np.isfinite(array)
        boolean = finite & (array > 0.5)

    nonempty_count = 0
    kept_masks = []
    kept_indices = []
    discarded_small = 0
    minimum_area = int(max(0, minimum_instance_area_px))
    for index, mask in enumerate(boolean):
        area = int(mask.sum())
        if area <= 0:
            continue
        nonempty_count += 1
        if area < minimum_area:
            discarded_small += 1
            continue
        kept_masks.append(np.asarray(mask, dtype=bool))
        kept_indices.append(int(index))
    info.update({
        "nonempty_instances": int(nonempty_count),
        "retained_instances": int(len(kept_masks)),
        "discarded_small_instances": int(discarded_small),
    })
    if not kept_masks:
        info["reason"] = (
            "all_instances_below_minimum_area"
            if nonempty_count > 0 and discarded_small == nonempty_count
            else "no_nonempty_instances"
        )
        return empty, [], info

    info.update({"status": "accepted", "reason": "accepted"})
    return np.stack(kept_masks, axis=0), kept_indices, info


def _collect_role_stacks(
    role_mask_stacks: Optional[Mapping[str, object]],
    expected_shape: Tuple[int, int],
    minimum_instance_area_px: int,
):
    height, width = expected_shape
    masks_by_role = {
        role: [] for role in CANONICAL_PREFIT_ROLES
    }
    refs_by_role = {
        role: [] for role in CANONICAL_PREFIT_ROLES
    }
    input_metadata: Dict[str, Dict[str, object]] = {}
    unknown_roles = []
    role_input_summary = {
        role: {
            "input_instances": 0,
            "nonempty_instances": 0,
            "retained_instances": 0,
            "discarded_small_instances": 0,
        }
        for role in CANONICAL_PREFIT_ROLES
    }

    for source_role, value in dict(role_mask_stacks or {}).items():
        source_name = str(source_role)
        canonical = _canonical_role_name(source_name)
        if canonical is None:
            unknown_roles.append(source_name)
            input_metadata[source_name] = {
                "status": "ignored",
                "reason": "unknown_role",
                "canonical_role": None,
            }
            continue
        stack, source_indices, info = _normalize_mask_stack(
            value,
            expected_shape,
            int(minimum_instance_area_px),
        )
        info = {**info, "canonical_role": canonical}
        input_metadata[source_name] = info
        for count_name in (
            "input_instances",
            "nonempty_instances",
            "retained_instances",
            "discarded_small_instances",
        ):
            role_input_summary[canonical][count_name] += int(
                info.get(count_name, 0)
            )
        for local_index, source_index in enumerate(source_indices):
            masks_by_role[canonical].append(stack[local_index])
            refs_by_role[canonical].append({
                "source_role": source_name,
                "source_index": int(source_index),
            })

    stacked = {}
    for role in CANONICAL_PREFIT_ROLES:
        if masks_by_role[role]:
            stacked[role] = np.stack(masks_by_role[role], axis=0)
        else:
            stacked[role] = np.zeros((0, height, width), dtype=bool)
    return (
        stacked,
        refs_by_role,
        input_metadata,
        sorted(unknown_roles),
        role_input_summary,
    )


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    radius = int(max(0, radius_px))
    if radius == 0 or not mask.any():
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _erode_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius == 0 or not mask.any():
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def _generic_non_target_residual(
    projection: np.ndarray,
    target: np.ndarray,
    config: PrefitSemanticGuidanceConfig,
    generic_proposal_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Find conservative, category-independent non-target islands.

    The residual can suppress misleading image edges, but only when the
    projection-associated target mask already covers enough of the model.
    Large components and large total residuals are rejected because they are
    more likely to indicate projection/segmentation mismatch than foreground.
    """
    projection = np.asarray(projection, dtype=bool)
    target = np.asarray(target, dtype=bool)
    empty = np.zeros_like(projection)
    projection_pixels = int(projection.sum())
    overlap_pixels = int((projection & target).sum())
    target_coverage = overlap_pixels / max(projection_pixels, 1)
    target_pixels = overlap_pixels
    generic_proposal = np.zeros_like(projection)
    if generic_proposal_mask is not None:
        generic_proposal = np.asarray(generic_proposal_mask, dtype=bool)
        if generic_proposal.shape != projection.shape:
            raise ValueError(
                "generic_proposal_mask must match the projection shape."
            )
    info = {
        "enabled": bool(config.generic_non_target_enabled),
        "reason": "not_run",
        "target_projection_overlap_pixels": overlap_pixels,
        "target_projection_coverage": float(target_coverage),
        "raw_residual_pixels": 0,
        "raw_target_complement_pixels": 0,
        "raw_generic_proposal_pixels": 0,
        "generic_proposal_overlap_target_pixels": 0,
        "candidate_components": 0,
        "selected_components": 0,
        "selected_pixels": 0,
        "rejected_small_components": 0,
        "rejected_large_components": 0,
        "rejected_target_overlap_components": 0,
        "fallback_used": False,
    }
    if not bool(config.generic_non_target_enabled):
        info["reason"] = "disabled"
        return empty, info
    if projection_pixels <= 0:
        info["reason"] = "empty_projection"
        return empty, info
    if not target.any():
        info["reason"] = "no_associated_target_mask"
        info["fallback_used"] = True
        return empty, info
    if target_coverage < float(config.generic_non_target_min_target_coverage):
        info["reason"] = "insufficient_target_projection_coverage"
        info["fallback_used"] = True
        return empty, info

    projection_core = _erode_mask(
        projection,
        int(config.generic_non_target_projection_inset_px),
    )
    if not projection_core.any():
        projection_core = projection.copy()
    target_support = _dilate_mask(
        target,
        int(config.generic_non_target_target_dilation_px),
    )
    target_complement = projection_core & (~target_support)
    proposal_inside_projection = projection_core & generic_proposal
    residual = target_complement | proposal_inside_projection
    info["raw_target_complement_pixels"] = int(target_complement.sum())
    info["raw_generic_proposal_pixels"] = int(
        proposal_inside_projection.sum()
    )
    info["generic_proposal_overlap_target_pixels"] = int(
        (proposal_inside_projection & target).sum()
    )
    info["raw_residual_pixels"] = int(residual.sum())
    if not residual.any():
        info["reason"] = "empty_non_target_residual"
        return empty, info

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8),
        connectivity=8,
    )
    selected = np.zeros_like(residual)
    minimum_area = max(1, int(config.generic_non_target_min_component_area_px))
    maximum_component_pixels = int(math.floor(
        float(config.generic_non_target_max_component_fraction)
        * projection_pixels
    ))
    maximum_target_overlap_fraction = float(
        config.generic_non_target_max_target_overlap_fraction
    )
    info["candidate_components"] = max(0, int(count - 1))
    selected_components = 0
    rejected_small = 0
    rejected_large = 0
    rejected_target_overlap = 0
    component_rows = []
    for label in range(1, count):
        component = labels == label
        area = int(component.sum())
        proposal_pixels = int(
            (component & proposal_inside_projection).sum()
        )
        target_overlap = int((component & target).sum())
        target_overlap_fraction = float(
            target_overlap / max(target_pixels, 1)
        )
        if area < minimum_area:
            rejected_small += 1
            decision = "rejected_small"
        elif area > maximum_component_pixels:
            rejected_large += 1
            decision = "rejected_large_projection_mismatch_guard"
        elif (
            proposal_pixels > 0
            and target_overlap_fraction > maximum_target_overlap_fraction
        ):
            rejected_target_overlap += 1
            decision = "rejected_generic_proposal_target_overlap_guard"
        else:
            selected |= component
            selected_components += 1
            decision = "selected_uncertain_non_target"
        component_rows.append({
            "area_px": area,
            "projection_fraction": float(area / max(projection_pixels, 1)),
            "generic_proposal_pixels": proposal_pixels,
            "target_overlap_pixels": target_overlap,
            "target_overlap_fraction": target_overlap_fraction,
            "decision": decision,
        })

    selected_pixels = int(selected.sum())
    maximum_total_pixels = int(round(
        float(config.generic_non_target_max_total_fraction)
        * projection_pixels
    ))
    if selected_pixels > maximum_total_pixels:
        info.update({
            "reason": "total_residual_fraction_exceeded_guard",
            "selected_components_before_guard": int(selected_components),
            "selected_pixels_before_guard": selected_pixels,
            "fallback_used": True,
            "components": component_rows,
            "rejected_small_components": int(rejected_small),
            "rejected_large_components": int(rejected_large),
            "rejected_target_overlap_components": int(
                rejected_target_overlap
            ),
        })
        return empty, info

    info.update({
        "reason": (
            "selected_projection_local_non_target_components"
            if selected_components > 0
            else "no_component_passed_safeguards"
        ),
        "selected_components": int(selected_components),
        "selected_pixels": selected_pixels,
        "rejected_small_components": int(rejected_small),
        "rejected_large_components": int(rejected_large),
        "rejected_target_overlap_components": int(rejected_target_overlap),
        "components": component_rows,
    })
    return selected, info


def _mask_union(stack: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if stack.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    return np.asarray(stack, dtype=bool).any(axis=0)


def _local_search_mask(
    projection: np.ndarray,
    radius_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    outside_projection = (~projection).astype(np.uint8)
    distance = cv2.distanceTransform(outside_projection, cv2.DIST_L2, 3)
    radius = float(max(0, radius_px))
    search = projection | (distance <= radius)
    return search, distance.astype(np.float32)


def _partition_context_by_target_and_sky(
    context_mask: np.ndarray,
    target_mask: np.ndarray,
    sky_mask: np.ndarray,
    projection_mask: np.ndarray,
    adjacency_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split contextual pixels into foreground, background, and unresolved.

    A contextual pixel is foreground when its nearest local semantic neighbour
    is the selected building/roof, and background when that neighbour is sky.
    Pixels without either neighbour remain conservatively foreground when they
    are projection-local; the returned unresolved mask records that fallback.
    This is deliberately spatial rather than prompt-name based: the same tree
    can be foreground over the facade and background where it crosses the sky.
    """
    context = np.asarray(context_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    sky = np.asarray(sky_mask, dtype=bool)
    projection = np.asarray(projection_mask, dtype=bool)
    if not (
        context.shape == target.shape == sky.shape == projection.shape
    ):
        raise ValueError("Context, target, sky, and projection masks must match.")
    if not context.any():
        empty = np.zeros_like(context)
        return empty.copy(), empty.copy(), empty.copy()

    radius = max(0, int(adjacency_px))
    distance_to_target = cv2.distanceTransform(
        (~target).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    distance_to_sky = cv2.distanceTransform(
        (~sky).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    target_neighbour = target | (distance_to_target <= float(radius))
    sky_neighbour = sky | (distance_to_sky <= float(radius))

    foreground = np.zeros_like(context)
    background = np.zeros_like(context)
    component_count, component_labels = cv2.connectedComponents(
        context.astype(np.uint8),
        connectivity=8,
    )
    for component_index in range(1, int(component_count)):
        component = component_labels == component_index
        touches_target = bool((component & target_neighbour).any())
        touches_sky = bool((component & sky_neighbour).any())
        if touches_target and touches_sky:
            # Split an object that spans both building and sky spatially.  The
            # building-side pixels are foreground; the sky-side pixels are
            # background context.  A distance tie stays foreground.
            foreground |= component & (distance_to_target <= distance_to_sky)
            background |= component & (distance_to_sky < distance_to_target)
        elif touches_target:
            foreground |= component
        elif touches_sky:
            background |= component
        else:
            # With neither semantic neighbour there is no positive evidence
            # that the component is behind the building, so retain the
            # conservative foreground interpretation.
            foreground |= component

    unresolved = context & (~foreground) & (~background)
    foreground |= unresolved
    return foreground, background, unresolved


def _associate_target_instances(
    *,
    role: str,
    stack: np.ndarray,
    instance_refs: Sequence[Mapping[str, object]],
    projection: np.ndarray,
    search_mask: np.ndarray,
    distance_to_projection: np.ndarray,
    config: PrefitSemanticGuidanceConfig,
) -> Tuple[np.ndarray, Sequence[int], Sequence[Dict[str, object]], Dict[str, object]]:
    """Select one target anchor and only useful, connected supplementary masks."""
    height, width = projection.shape
    empty = np.zeros((height, width), dtype=bool)
    if stack.shape[0] == 0:
        return empty, [], [], {
            "role": role,
            "raw_instances": 0,
            "eligible_instances": 0,
            "selected_instances": 0,
            "selected_indices": [],
            "selected_pixels": 0,
            "reason": "no_instances",
        }

    projection_area = max(int(projection.sum()), 1)
    candidates = []
    for index, mask in enumerate(stack):
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        local_mask = mask & search_mask
        local_area = int(local_mask.sum())
        overlap = int((mask & projection).sum())
        overlap_fraction = float(overlap / max(min(area, projection_area), 1))
        projection_coverage = float(overlap / projection_area)
        local_fraction = float(local_area / max(area, 1))
        minimum_distance = (
            float(np.min(distance_to_projection[local_mask]))
            if local_area > 0
            else float("inf")
        )
        meaningful_overlap = bool(
            overlap > 0
            and (
                overlap >= int(config.target_min_overlap_pixels)
                or overlap_fraction >= float(config.target_min_overlap_fraction)
            )
        )
        near_projection = bool(
            local_area > 0
            and minimum_distance <= float(config.target_association_distance_px)
            and local_fraction >= float(config.target_min_local_fraction)
        )
        eligible = bool(meaningful_overlap or near_projection)
        distance_scale = max(float(config.target_association_distance_px), 1.0)
        proximity_score = (
            math.exp(-minimum_distance / distance_scale)
            if math.isfinite(minimum_distance)
            else 0.0
        )
        score = float(
            6.0 * min(overlap_fraction, 1.0)
            + 2.0 * min(projection_coverage, 1.0)
            + 1.25 * min(local_fraction, 1.0)
            + proximity_score
        )
        candidates.append({
            "index": int(index),
            "reference": dict(instance_refs[index]),
            "mask": local_mask,
            "area_px": area,
            "local_area_px": local_area,
            "overlap_px": overlap,
            "overlap_fraction": overlap_fraction,
            "projection_coverage": projection_coverage,
            "local_fraction": local_fraction,
            "minimum_distance_px": minimum_distance,
            "meaningful_overlap": meaningful_overlap,
            "near_projection": near_projection,
            "eligible": eligible,
            "score": score,
        })

    eligible = sorted(
        (row for row in candidates if row["eligible"]),
        key=lambda row: (
            row["score"],
            row["overlap_px"],
            row["local_area_px"],
        ),
        reverse=True,
    )
    if not eligible:
        return empty, [], [], {
            "role": role,
            "raw_instances": int(stack.shape[0]),
            "eligible_instances": 0,
            "selected_instances": 0,
            "selected_indices": [],
            "selected_pixels": 0,
            "reason": "no_instance_near_or_overlapping_projection",
        }

    selected_rows = [eligible[0]]
    selected_mask = eligible[0]["mask"].copy()
    covered_projection = selected_mask & projection
    best_score = max(float(eligible[0]["score"]), 1.0e-9)
    connection_radius = max(
        1,
        int(round(float(config.target_association_distance_px) * 0.25)),
    )

    for row in eligible[1:]:
        if len(selected_rows) >= int(config.target_max_instances_per_role):
            break
        if float(row["score"]) < (
            best_score * float(config.target_relative_score_threshold)
        ):
            continue
        new_projection_pixels = int(
            (row["mask"] & projection & (~covered_projection)).sum()
        )
        connected = bool(
            (
                row["mask"]
                & _dilate_mask(selected_mask, connection_radius)
            ).any()
        )
        supplements_projection = bool(
            new_projection_pixels
            >= int(config.target_min_new_projection_pixels)
        )
        if not supplements_projection and not connected:
            continue
        selected_rows.append(row)
        selected_mask |= row["mask"]
        covered_projection |= row["mask"] & projection

    selected_indices = [int(row["index"]) for row in selected_rows]
    selected_details = []
    for row in selected_rows:
        selected_details.append({
            "index": int(row["index"]),
            "source_role": str(row["reference"]["source_role"]),
            "source_index": int(row["reference"]["source_index"]),
            "area_px": int(row["area_px"]),
            "local_area_px": int(row["local_area_px"]),
            "overlap_px": int(row["overlap_px"]),
            "overlap_fraction": float(row["overlap_fraction"]),
            "projection_coverage": float(row["projection_coverage"]),
            "local_fraction": float(row["local_fraction"]),
            "minimum_distance_px": float(row["minimum_distance_px"]),
            "score": float(row["score"]),
        })
    summary = {
        "role": role,
        "raw_instances": int(stack.shape[0]),
        "eligible_instances": int(len(eligible)),
        "selected_instances": int(len(selected_rows)),
        "selected_indices": selected_indices,
        "selected_pixels": int(selected_mask.sum()),
        "selected_projection_overlap_pixels": int(
            (selected_mask & projection).sum()
        ),
        "reason": "selected_projection_local_instances",
    }
    return selected_mask, selected_indices, selected_details, summary


def _projected_top_geometry(
    projection: np.ndarray,
    band_radius_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the model's top envelope, a narrow band, and top y per column."""
    projection = np.asarray(projection, dtype=bool)
    _height, width = projection.shape
    top_y = np.full(width, -1, dtype=np.int32)
    top_envelope = np.zeros_like(projection)
    for x in np.flatnonzero(projection.any(axis=0)):
        ys = np.flatnonzero(projection[:, int(x)])
        if ys.size <= 0:
            continue
        top_y[int(x)] = int(ys[0])
        top_envelope[int(ys[0]), int(x)] = True
    top_band = _dilate_mask(top_envelope, int(band_radius_px))
    return top_envelope, top_band, top_y


def _upper_building_region(
    building: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Keep the upper fraction independently in each occupied image column."""
    building = np.asarray(building, dtype=bool)
    height, _width = building.shape
    upper = np.zeros_like(building)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    for x in np.flatnonzero(building.any(axis=0)):
        ys = np.flatnonzero(building[:, int(x)])
        if ys.size <= 0:
            continue
        limit = int(round(
            float(ys[0]) + fraction * float(ys[-1] - ys[0])
        ))
        limit = min(height, limit + 1)
        upper[:limit, int(x)] = building[:limit, int(x)]
    return upper


def _horizontal_span_fraction(
    mask: np.ndarray,
    reference: np.ndarray,
) -> float:
    xs = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=0))
    reference_xs = np.flatnonzero(
        np.asarray(reference, dtype=bool).any(axis=0)
    )
    if xs.size <= 0 or reference_xs.size <= 0:
        return 0.0
    span = int(xs[-1]) - int(xs[0]) + 1
    reference_span = int(reference_xs[-1]) - int(reference_xs[0]) + 1
    return float(span / max(reference_span, 1))


def _remove_small_components(
    mask: np.ndarray,
    minimum_pixels: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    minimum = max(1, int(minimum_pixels))
    if minimum <= 1 or not mask.any():
        return mask.copy()
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    kept = np.zeros_like(mask)
    for label in range(1, int(count)):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            kept |= labels == label
    return kept


def _associate_strict_roof_instances(
    *,
    stack: np.ndarray,
    instance_refs: Sequence[Mapping[str, object]],
    projection: np.ndarray,
    search_mask: np.ndarray,
    selected_building: np.ndarray,
    explicit_foreground: np.ndarray,
    config: PrefitSemanticGuidanceConfig,
):
    """Select fit roofs only when model-top and upper-building tests agree."""
    shape = projection.shape
    empty = np.zeros(shape, dtype=bool)
    top_envelope, top_band, top_y = _projected_top_geometry(
        projection,
        int(config.strict_roof_projected_band_radius_px),
    )
    upper_building = _upper_building_region(
        selected_building,
        float(config.strict_roof_upper_building_fraction),
    )
    attachment_region = _dilate_mask(
        upper_building,
        int(config.strict_roof_attachment_radius_px),
    )
    geometry = {
        "projected_top_envelope_mask": top_envelope,
        "projected_top_band_mask": top_band,
        "projected_top_y": top_y,
        "upper_building_region_mask": upper_building,
        "upper_building_attachment_mask": attachment_region,
    }
    if stack.shape[0] == 0:
        summary = {
            "role": "roof",
            "raw_instances": 0,
            "eligible_instances": 0,
            "selected_instances": 0,
            "selected_indices": [],
            "selected_pixels": 0,
            "reason": "no_instances",
            "strict_gate_enabled": True,
        }
        return empty, [], [], summary, geometry, []

    candidates = []
    for index, mask in enumerate(stack):
        mask = np.asarray(mask, dtype=bool)
        local_mask = mask & search_mask
        area = int(mask.sum())
        local_area = int(local_mask.sum())
        band_overlap = local_mask & top_band
        band_pixels = int(band_overlap.sum())
        band_span_fraction = _horizontal_span_fraction(
            band_overlap,
            projection,
        )
        attachment_pixels = int((local_mask & attachment_region).sum())
        explicit_overlap = int((local_mask & explicit_foreground).sum())
        explicit_fraction = float(explicit_overlap / max(local_area, 1))
        rejection_reasons = []
        if area < int(config.minimum_instance_area_px) or local_area <= 0:
            rejection_reasons.append("too_small_or_outside_local_search")
        if band_pixels < int(config.strict_roof_min_band_pixels):
            rejection_reasons.append("outside_projected_top_band")
        if band_span_fraction < float(
            config.strict_roof_min_band_span_fraction
        ):
            rejection_reasons.append("insufficient_projected_band_span")
        if attachment_pixels < int(config.strict_roof_min_attachment_pixels):
            rejection_reasons.append("not_attached_to_upper_building")
        if explicit_fraction > float(
            config.strict_roof_max_explicit_foreground_fraction
        ):
            rejection_reasons.append("explicit_foreground_conflict")
        candidates.append({
            "index": int(index),
            "reference": dict(instance_refs[index]),
            "mask": local_mask,
            "area_px": area,
            "local_area_px": local_area,
            "band_pixels": band_pixels,
            "band_span_fraction": band_span_fraction,
            "upper_building_attachment_pixels": attachment_pixels,
            "explicit_foreground_overlap_pixels": explicit_overlap,
            "explicit_foreground_fraction": explicit_fraction,
            "rejection_reasons": rejection_reasons,
        })

    passing = sorted(
        (row for row in candidates if not row["rejection_reasons"]),
        key=lambda row: (
            row["band_pixels"],
            row["upper_building_attachment_pixels"],
            row["local_area_px"],
        ),
        reverse=True,
    )
    selected_rows = passing[: int(config.target_max_instances_per_role)]
    selected_indices = [int(row["index"]) for row in selected_rows]
    selected_index_set = set(selected_indices)
    selected_mask = empty.copy()
    for row in selected_rows:
        selected_mask |= row["mask"]

    instance_decisions = []
    for row in candidates:
        reasons = list(row["rejection_reasons"])
        gate_passed = not reasons
        accepted = int(row["index"]) in selected_index_set
        if gate_passed and not accepted:
            reasons.append("target_instance_limit_exceeded")
        reference = row["reference"]
        instance_decisions.append({
            "index": int(row["index"]),
            "source_role": str(reference.get("source_role", "roof")),
            "source_index": int(reference.get("source_index", row["index"])),
            "area_px": int(row["area_px"]),
            "local_area_px": int(row["local_area_px"]),
            "band_pixels": int(row["band_pixels"]),
            "band_span_fraction": float(row["band_span_fraction"]),
            "upper_building_attachment_pixels": int(
                row["upper_building_attachment_pixels"]
            ),
            "explicit_foreground_overlap_pixels": int(
                row["explicit_foreground_overlap_pixels"]
            ),
            "explicit_foreground_fraction": float(
                row["explicit_foreground_fraction"]
            ),
            "strict_gate_passed": bool(gate_passed),
            "accepted_for_fit": bool(accepted),
            "rejection_reasons": reasons,
        })

    selected_details = [
        dict(instance_decisions[int(row["index"])])
        for row in selected_rows
    ]
    summary = {
        "role": "roof",
        "raw_instances": int(stack.shape[0]),
        "eligible_instances": int(len(passing)),
        "selected_instances": int(len(selected_rows)),
        "selected_indices": selected_indices,
        "selected_pixels": int(selected_mask.sum()),
        "selected_projection_overlap_pixels": int(
            (selected_mask & projection).sum()
        ),
        "reason": (
            "selected_strict_projection_top_roof_instances"
            if selected_rows
            else "no_instance_passed_strict_roof_gate"
        ),
        "strict_gate_enabled": True,
    }
    return (
        selected_mask,
        selected_indices,
        selected_details,
        summary,
        geometry,
        instance_decisions,
    )


def _boolean_x_runs(values: np.ndarray) -> Sequence[Tuple[int, int]]:
    values = np.asarray(values, dtype=bool).reshape(-1)
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
    ]


def _build_strict_roof_bridge(
    observed_seed: np.ndarray,
    projected_top_y: np.ndarray,
    projected_top_band: np.ndarray,
    foreground_guard: np.ndarray,
    search_mask: np.ndarray,
    config: PrefitSemanticGuidanceConfig,
) -> np.ndarray:
    """Infer only short, foreground-explained gaps bracketed at both ends."""
    observed_seed = np.asarray(observed_seed, dtype=bool)
    height, width = observed_seed.shape
    bridge = np.zeros_like(observed_seed)
    if not bool(config.strict_roof_bridge_enabled) or not observed_seed.any():
        return bridge

    occupied_columns = observed_seed.any(axis=0)
    minimum_run = int(config.strict_roof_bridge_min_endpoint_run_px)
    runs = [
        run for run in _boolean_x_runs(occupied_columns)
        if run[1] - run[0] + 1 >= minimum_run
    ]
    bridge_domain = (
        _dilate_mask(
            foreground_guard,
            int(config.strict_roof_bridge_domain_dilation_px),
        )
        & projected_top_band
        & search_mask
    )
    for left, right in zip(runs, runs[1:]):
        gap_start = int(left[1] + 1)
        gap_end = int(right[0] - 1)
        gap_width = gap_end - gap_start + 1
        if (
            gap_width <= 0
            or gap_width > int(config.strict_roof_bridge_max_gap_px)
        ):
            continue
        gap_domain = bridge_domain[:, gap_start:gap_end + 1]
        # Every missing column must be explained by the foreground guard.  A
        # merely nearby obstruction is not enough to infer a roof continuation.
        if gap_domain.size <= 0 or not gap_domain.any(axis=0).all():
            continue

        left_slice_start = max(int(left[0]), int(left[1]) - 2)
        right_slice_end = min(width, int(right[0]) + 3)
        left_points = np.argwhere(
            observed_seed[:, left_slice_start:int(left[1]) + 1]
        )
        right_points = np.argwhere(
            observed_seed[:, int(right[0]):right_slice_end]
        )
        if left_points.size <= 0 or right_points.size <= 0:
            continue
        left_y = float(np.median(left_points[:, 0]))
        right_y = float(np.median(right_points[:, 0]))
        left_top = int(projected_top_y[int(left[1])])
        right_top = int(projected_top_y[int(right[0])])
        if left_top < 0 or right_top < 0:
            continue
        left_delta = left_y - float(left_top)
        right_delta = right_y - float(right_top)

        for offset, x in enumerate(range(gap_start, gap_end + 1)):
            top_y = int(projected_top_y[int(x)])
            if top_y < 0:
                continue
            alpha = float((offset + 1) / (gap_width + 1))
            delta = (1.0 - alpha) * left_delta + alpha * right_delta
            y = int(round(float(top_y) + delta))
            if 0 <= y < height and bridge_domain[y, int(x)]:
                bridge[y, int(x)] = True
    return bridge


def _build_strict_roof_interface(
    *,
    selected_roof: np.ndarray,
    projection: np.ndarray,
    search_mask: np.ndarray,
    interior_image: np.ndarray,
    sky: np.ndarray,
    vegetation: np.ndarray,
    explicit_foreground: np.ndarray,
    additional_foreground: np.ndarray,
    geometry: Mapping[str, np.ndarray],
    config: PrefitSemanticGuidanceConfig,
):
    """Build roof-to-background evidence while excluding foreground contours."""
    top_envelope = np.asarray(
        geometry["projected_top_envelope_mask"], dtype=bool
    )
    top_band = np.asarray(
        geometry["projected_top_band_mask"], dtype=bool
    )
    top_y = np.asarray(geometry["projected_top_y"], dtype=np.int32)
    upper_building = np.asarray(
        geometry["upper_building_region_mask"], dtype=bool
    )
    upper_attachment = np.asarray(
        geometry["upper_building_attachment_mask"], dtype=bool
    )

    inside_projection = _erode_mask(
        projection,
        int(config.strict_roof_vegetation_projection_inset_px),
    )
    image_rows = np.arange(projection.shape[0], dtype=np.int32)[:, None]
    valid_columns = top_y >= 0
    deep_threshold = np.where(
        valid_columns,
        top_y + int(config.strict_roof_vegetation_inside_offset_px),
        projection.shape[0] + 1,
    )[None, :]
    deep_inside = inside_projection & (image_rows >= deep_threshold)
    foreground_vegetation = vegetation & deep_inside
    foreground = (
        explicit_foreground
        | additional_foreground
        | foreground_vegetation
    ) & search_mask
    foreground_guard = _dilate_mask(
        foreground,
        int(config.strict_roof_foreground_guard_radius_px),
    ) & search_mask

    context_radius = int(config.strict_roof_context_radius_px)
    projection_exterior_side = ~_erode_mask(projection, context_radius)
    background_context = (
        sky
        | (
            vegetation
            & projection_exterior_side
            & (~explicit_foreground)
        )
    ) & search_mask
    adjacent_background = _dilate_mask(
        background_context,
        context_radius,
    ) & search_mask

    # Match the approved experiment's elliptical one-pixel erosion.  The
    # legacy guide builder uses a rectangular erosion, which makes diagonal
    # roof contours materially thicker and would change the accepted replay.
    roof_boundary = selected_roof & (~_erode_mask(selected_roof, 1))
    boundary_before_foreground = (
        roof_boundary
        & top_band
        & upper_attachment
        & adjacent_background
        & search_mask
        & interior_image
    )
    observed_seed = boundary_before_foreground & (~foreground_guard)
    observed_seed = _remove_small_components(
        observed_seed,
        int(config.strict_roof_min_guide_component_pixels),
    )
    discarded_small = (
        boundary_before_foreground
        & (~foreground_guard)
        & (~observed_seed)
    )
    consumed = (
        _thicken_boundary(
            observed_seed,
            int(config.boundary_thickness_px),
        )
        & search_mask
        & interior_image
        & (~foreground_guard)
    )
    suppressed_by_foreground = (
        boundary_before_foreground & foreground_guard
    )
    inferred_bridge = _build_strict_roof_bridge(
        observed_seed,
        top_y,
        top_band,
        foreground_guard,
        search_mask,
        config,
    )
    # The bridge is a visualization-only inference.  Removing even accidental
    # overlap with thickened observed evidence makes that contract explicit.
    inferred_bridge &= interior_image & (~consumed)

    diagnostic_masks = {
        "projected_top_envelope_mask": top_envelope.copy(),
        "projected_top_band_mask": top_band.copy(),
        "upper_building_region_mask": upper_building.copy(),
        "upper_building_attachment_mask": upper_attachment.copy(),
        "projection_exterior_side_mask": projection_exterior_side & search_mask,
        "background_context_mask": background_context,
        "foreground_vegetation_mask": foreground_vegetation,
        "foreground_guard_mask": foreground_guard,
        "roof_boundary_before_foreground_mask": boundary_before_foreground,
        "roof_observed_seed_mask": observed_seed,
        "roof_consumed_boundary_mask": consumed,
        "roof_suppressed_by_foreground_mask": suppressed_by_foreground,
        "roof_discarded_small_component_mask": discarded_small,
        "roof_inferred_bridge_not_consumed_mask": inferred_bridge,
    }
    info = {
        "fit_consumed_roof_boundary_pixels": int(consumed.sum()),
        "foreground_suppressed_seed_pixels": int(
            suppressed_by_foreground.sum()
        ),
        "discarded_small_seed_pixels": int(discarded_small.sum()),
        "inferred_not_consumed_bridge_pixels": int(inferred_bridge.sum()),
        "bridge_is_fitting_evidence": False,
        "background_context_pixels": int(background_context.sum()),
        "foreground_guard_pixels": int(foreground_guard.sum()),
    }
    return consumed, diagnostic_masks, info


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ) > 0
    return mask & (~eroded)


def _top_bottom_envelopes(
    target_mask: np.ndarray,
    tolerance_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify the outer top and bottom envelopes without assuming a rectangle."""
    target = np.asarray(target_mask, dtype=bool)
    height, width = target.shape
    top = np.zeros_like(target)
    bottom = np.zeros_like(target)
    tolerance = int(max(0, tolerance_px))
    for x in np.flatnonzero(target.any(axis=0)):
        ys = np.flatnonzero(target[:, int(x)])
        if len(ys) == 0:
            continue
        y_top = int(ys[0])
        y_bottom = int(ys[-1])
        top[y_top:min(height, y_top + tolerance + 1), int(x)] = True
        bottom[max(0, y_bottom - tolerance):y_bottom + 1, int(x)] = True
    return top, bottom


def _thicken_boundary(mask: np.ndarray, thickness_px: int) -> np.ndarray:
    thickness = int(max(0, thickness_px))
    if thickness <= 1:
        return np.asarray(mask, dtype=bool).copy()
    return _dilate_mask(mask, thickness - 1)


def _interior_image_mask(
    shape: Tuple[int, int],
    border_exclusion_px: int,
) -> np.ndarray:
    height, width = shape
    interior = np.ones((height, width), dtype=bool)
    margin = int(max(0, border_exclusion_px))
    if margin <= 0:
        return interior
    margin_y = min(margin, max(1, height // 2))
    margin_x = min(margin, max(1, width // 2))
    interior[:margin_y, :] = False
    interior[height - margin_y:, :] = False
    interior[:, :margin_x] = False
    interior[:, width - margin_x:] = False
    return interior


def build_prefit_semantic_guidance(
    role_mask_stacks: Optional[Mapping[str, object]],
    raw_projection_mask: np.ndarray,
    *,
    config: Optional[PrefitSemanticGuidanceConfig] = None,
    external_exclusion_mask: Optional[np.ndarray] = None,
    downstream_roof_mask_stack: Optional[object] = None,
) -> Dict[str, object]:
    """Build conservative semantic evidence localized to a raw model projection.

    ``building`` and ``roof`` instances are associated individually with the
    projection.  Context roles are localized to its dilated search region.
    ``external_exclusion_mask`` is removed before target association, so an
    OSM-known neighbouring building cannot become the semantic target merely
    because it overlaps the projected model in image space.
    ``downstream_roof_mask_stack`` is deliberately kept outside the canonical
    fit roles.  When provided, its localized union remains available through
    the compatibility ``roof_prompt_mask`` used by post-Hough cleanup, while
    canonical ``roof`` instances alone can affect fitting and visibility.
    Sky and ground provide positive boundary context. Vegetation, prompted
    occluders, and conservative projection-local non-target residuals suppress
    image evidence that could pull the model toward foreground objects.
    """
    config = config or PrefitSemanticGuidanceConfig()
    projection = np.asarray(raw_projection_mask, dtype=bool)
    if projection.ndim != 2:
        raise ValueError("raw_projection_mask must be a two-dimensional mask.")
    height, width = projection.shape
    if height == 0 or width == 0:
        raise ValueError("raw_projection_mask must have non-zero height and width.")
    if external_exclusion_mask is None:
        external_exclusion = np.zeros_like(projection)
    else:
        external_exclusion = np.asarray(external_exclusion_mask, dtype=bool)
        if external_exclusion.shape != projection.shape:
            raise ValueError(
                "external_exclusion_mask must match raw_projection_mask."
            )

    (
        role_stacks,
        role_refs,
        input_metadata,
        unknown_roles,
        role_input_summary,
    ) = _collect_role_stacks(
        role_mask_stacks,
        projection.shape,
        int(config.minimum_instance_area_px),
    )
    downstream_roof_stack_supplied = downstream_roof_mask_stack is not None
    (
        normalized_downstream_roof_stack,
        _downstream_roof_source_indices,
        downstream_roof_input_metadata,
    ) = _normalize_mask_stack(
        downstream_roof_mask_stack,
        projection.shape,
        int(config.minimum_instance_area_px),
    )

    if not projection.any():
        zero = np.zeros_like(projection)
        one = np.ones_like(projection)
        metadata = {
            "version": 3,
            "shape_hw": [int(height), int(width)],
            "uses_semantic_guidance": False,
            "fallback_used": True,
            "reason": "empty_projection_mask",
            "projection_pixels": 0,
            "association_projection_pixels": 0,
            "external_exclusion_pixels": int(external_exclusion.sum()),
            "search_pixels": 0,
            "valid_evidence_pixels": int(one.sum()),
            "excluded_occluder_pixels": 0,
            "hard_occluder_pixels": 0,
            "generic_non_target_pixels": 0,
            "target_pixels": 0,
            "generic_non_target": {
                "enabled": bool(config.generic_non_target_enabled),
                "reason": "empty_projection_mask",
                "selected_pixels": 0,
            },
            "roles": {
                role: {
                    **role_input_summary[role],
                    "raw_instances": int(role_stacks[role].shape[0]),
                    "selected_instances": 0,
                    "selected_pixels": 0,
                }
                for role in CANONICAL_PREFIT_ROLES
            },
            "selected_target_instances": {"building": [], "roof": []},
            "boundary_pixels_by_class": {
                role: 0 for role in ("roof", "wall", "base", "silhouette")
            },
            "diagnostic_boundary_pixels_by_class": {
                role: 0 for role in ("roof", "wall", "base", "silhouette")
            },
            "unknown_roles": unknown_roles,
            "inputs": input_metadata,
            "roof_prompt_stage_separation": {
                "fit_role": "roof",
                "downstream_stack_supplied": bool(
                    downstream_roof_stack_supplied
                ),
                "roof_prompt_mask_source": (
                    "downstream_roof_mask_stack"
                    if downstream_roof_stack_supplied
                    else "canonical_roof_compatibility_fallback"
                ),
                "downstream_input": downstream_roof_input_metadata,
                "fit_roof_prompt_pixels": 0,
                "downstream_roof_prompt_pixels": 0,
            },
            "config": asdict(config),
        }
        if bool(config.strict_roof_guidance_enabled):
            metadata["strict_roof_guidance"] = {
                "enabled": True,
                "reason": "empty_projection_mask",
                "roof_instances": [],
                "raw_roof_prompt_pixels": 0,
                "fit_selected_roof_pixels": 0,
                "fit_selected_roof_instance_count": 0,
                "fit_consumed_roof_boundary_pixels": 0,
                "foreground_suppressed_seed_pixels": 0,
                "discarded_small_seed_pixels": 0,
                "inferred_not_consumed_bridge_pixels": 0,
                "raw_roof_prompt_mask_preserved": True,
                "bridge_is_fitting_evidence": False,
            }
        result = {
            "raw_projection_mask": projection.copy(),
            "external_exclusion_mask": external_exclusion.copy(),
            "local_search_mask": zero,
            "valid_evidence_mask": one,
            "selected_building_mask": zero.copy(),
            "selected_roof_mask": zero.copy(),
            "fit_roof_prompt_mask": zero.copy(),
            "downstream_roof_prompt_mask": zero.copy(),
            "roof_prompt_mask": zero.copy(),
            "target_semantic_mask": zero.copy(),
            "sky_mask": zero.copy(),
            "ground_mask": zero.copy(),
            "vegetation_mask": zero.copy(),
            "hard_occluder_mask": zero.copy(),
            "foreground_hard_context_mask": zero.copy(),
            "background_hard_context_mask": zero.copy(),
            "generic_non_target_mask": zero.copy(),
            "foreground_generic_non_target_mask": zero.copy(),
            "background_generic_non_target_mask": zero.copy(),
            "generic_occluder_proposal_mask": zero.copy(),
            "occluder_mask": zero.copy(),
            "excluded_evidence_mask": zero.copy(),
            "foreground_mask": zero.copy(),
            "background_mask": zero.copy(),
            "foreground_excluded_evidence_mask": zero.copy(),
            "background_aware_valid_evidence_mask": one.copy(),
            "semantic_valid_evidence_mask": one.copy(),
            "boundary_maps": {
                role: zero.copy()
                for role in ("roof", "wall", "base", "silhouette")
            },
            "background_aware_boundary_maps": {
                role: zero.copy()
                for role in ("roof", "wall", "base", "silhouette")
            },
            "diagnostic_boundary_maps": {
                role: zero.copy()
                for role in ("roof", "wall", "base", "silhouette")
            },
            "metadata": metadata,
        }
        if bool(config.strict_roof_guidance_enabled):
            result["strict_roof_diagnostic_masks"] = {
                name: zero.copy()
                for name in (
                    "projected_top_envelope_mask",
                    "projected_top_band_mask",
                    "upper_building_region_mask",
                    "upper_building_attachment_mask",
                    "projection_exterior_side_mask",
                    "background_context_mask",
                    "foreground_vegetation_mask",
                    "foreground_guard_mask",
                    "roof_boundary_before_foreground_mask",
                    "roof_observed_seed_mask",
                    "roof_consumed_boundary_mask",
                    "roof_suppressed_by_foreground_mask",
                    "roof_discarded_small_component_mask",
                    "roof_inferred_bridge_not_consumed_mask",
                    "roof_suppressed_legacy_boundary_mask",
                )
            }
        return result

    association_projection = projection & (~external_exclusion)
    search_mask, _distance_to_projection = _local_search_mask(
        projection,
        int(config.search_dilation_px),
    )
    # Exclude the complete rendered blocker footprint, not only the pixels
    # where it currently overlaps the target.  Otherwise an external-building
    # edge immediately beside the raw projection can still attract a fit.
    search_mask &= ~external_exclusion
    distance_to_association_projection = cv2.distanceTransform(
        (~association_projection).astype(np.uint8),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    selected_building, _, building_details, building_summary = (
        _associate_target_instances(
            role="building",
            stack=role_stacks["building"],
            instance_refs=role_refs["building"],
            projection=association_projection,
            search_mask=search_mask,
            distance_to_projection=distance_to_association_projection,
            config=config,
        )
    )
    strict_roof_geometry = None
    strict_roof_instance_decisions = []
    legacy_selected_roof_for_diagnostics = None
    if bool(config.strict_roof_guidance_enabled):
        (
            legacy_selected_roof_for_diagnostics,
            _,
            _,
            _,
        ) = _associate_target_instances(
            role="roof",
            stack=role_stacks["roof"],
            instance_refs=role_refs["roof"],
            projection=association_projection,
            search_mask=search_mask,
            distance_to_projection=distance_to_association_projection,
            config=config,
        )
        explicit_foreground_for_gate = (
            _mask_union(role_stacks["occluder"], projection.shape)
            & search_mask
        )
        (
            selected_roof,
            _,
            roof_details,
            roof_summary,
            strict_roof_geometry,
            strict_roof_instance_decisions,
        ) = _associate_strict_roof_instances(
            stack=role_stacks["roof"],
            instance_refs=role_refs["roof"],
            projection=association_projection,
            search_mask=search_mask,
            selected_building=selected_building,
            explicit_foreground=explicit_foreground_for_gate,
            config=config,
        )
    else:
        selected_roof, _, roof_details, roof_summary = (
            _associate_target_instances(
                role="roof",
                stack=role_stacks["roof"],
                instance_refs=role_refs["roof"],
                projection=association_projection,
                search_mask=search_mask,
                distance_to_projection=distance_to_association_projection,
                config=config,
            )
        )
    fit_roof_prompt_mask = (
        _mask_union(role_stacks["roof"], projection.shape) & search_mask
    )
    downstream_roof_prompt_mask = (
        _mask_union(normalized_downstream_roof_stack, projection.shape)
        & search_mask
        if downstream_roof_stack_supplied
        else fit_roof_prompt_mask.copy()
    )
    # Backward-compatible downstream contract.  This key has always been the
    # broad, unselected roof union preferred by facade/post-Hough cleanup.
    roof_prompt_mask = downstream_roof_prompt_mask.copy()

    localized_roles = {}
    for role in (
        "sky",
        "ground",
        "vegetation",
        "occluder",
        "generic_occluder",
    ):
        localized_roles[role] = (
            _mask_union(role_stacks[role], projection.shape) & search_mask
        )

    target = (selected_building | selected_roof) & search_mask
    vegetation = localized_roles["vegetation"]
    # These masks come only from the deliberately specific foreground prompt
    # library (vehicle/person/sign/pole/etc.). Independent SAM prompts can
    # overlap, so letting the broad building prompt win here would preserve a
    # detected signboard merely because the building mask also covered it.
    explicit_occluder = localized_roles["occluder"]
    hard_occluder = vegetation | explicit_occluder
    generic_non_target, generic_non_target_info = (
        _generic_non_target_residual(
            association_projection,
            target,
            config,
            generic_proposal_mask=localized_roles["generic_occluder"],
        )
    )
    generic_non_target &= ~hard_occluder
    generic_non_target_info[
        "selected_pixels_after_hard_exclusion_deduplication"
    ] = int(generic_non_target.sum())
    raw_occluder = hard_occluder | generic_non_target
    excluded_evidence = _dilate_mask(
        raw_occluder,
        int(config.occluder_dilation_px),
    ) & search_mask
    interior_image = _interior_image_mask(
        projection.shape,
        int(config.image_border_exclusion_px),
    )
    valid_evidence = search_mask & (~excluded_evidence) & interior_image

    # Keep the legacy evidence above as the stable incumbent.  The new masks
    # form a second, background-aware interpretation used by the guarded fit
    # challenger in the pipeline.  This makes the rule general while allowing
    # already-correct incumbent transforms to remain byte-for-byte unchanged.
    (
        foreground_hard_context,
        background_hard_context,
        _unresolved_hard_context,
    ) = _partition_context_by_target_and_sky(
        hard_occluder,
        target,
        localized_roles["sky"],
        association_projection,
        int(config.context_adjacency_px),
    )
    (
        foreground_generic_non_target,
        background_generic_non_target,
        _unresolved_generic_non_target,
    ) = _partition_context_by_target_and_sky(
        generic_non_target,
        target,
        localized_roles["sky"],
        association_projection,
        int(config.context_adjacency_px),
    )
    foreground_mask = (
        foreground_hard_context | foreground_generic_non_target
    ) & search_mask
    background_mask = (
        localized_roles["sky"]
        | background_hard_context
        | background_generic_non_target
    ) & search_mask & (~foreground_mask)
    foreground_excluded_evidence = _dilate_mask(
        foreground_mask,
        int(config.occluder_dilation_px),
    ) & search_mask
    background_aware_valid_evidence = (
        search_mask & (~foreground_excluded_evidence) & interior_image
    )
    # Semantic guides are produced from SAM target/context boundaries. Their
    # own pixels are trusted independently from raw Canny/LSD evidence so the
    # complete roof/sky line remains available even through a foreground fill.
    semantic_valid_evidence = search_mask & interior_image

    thin_silhouette = _inner_boundary(target)
    top_envelope, bottom_envelope = _top_bottom_envelopes(
        target,
        int(config.envelope_tolerance_px),
    )
    adjacency = int(config.context_adjacency_px)
    roof_context = (
        _dilate_mask(selected_roof, adjacency)
        | _dilate_mask(localized_roles["sky"], adjacency)
        | top_envelope
    )
    base_context = (
        _dilate_mask(localized_roles["ground"], adjacency)
        | bottom_envelope
    )

    roof_seed = thin_silhouette & roof_context
    base_seed = thin_silhouette & base_context & (~roof_seed)
    wall_seed = thin_silhouette & (~roof_seed) & (~base_seed)
    seeds = {
        "roof": roof_seed,
        "wall": wall_seed,
        "base": base_seed,
        "silhouette": thin_silhouette,
    }
    # Fitting must still ignore excluded evidence. Diagnostics deliberately
    # keep a second, unmasked copy so the guide lines remain readable through
    # translucent exclusion fills and later model-projection lines.
    diagnostic_boundary_maps = {
        role: (
            _thicken_boundary(seed, int(config.boundary_thickness_px))
            & search_mask
            & interior_image
        )
        for role, seed in seeds.items()
    }
    boundary_maps = {
        role: mask & (~excluded_evidence)
        for role, mask in diagnostic_boundary_maps.items()
    }
    background_aware_boundary_maps = {
        # The user-facing yellow roof/sky guide is the one semantic boundary
        # that must remain complete.  Other semantic classes still obey the
        # foreground do-not-disturb mask, so trees/vehicles cannot attract a
        # wall or base edge merely because their diagnostic line is visible.
        role: (
            mask.copy()
            if role == "roof"
            else mask & (~foreground_excluded_evidence)
        )
        for role, mask in diagnostic_boundary_maps.items()
    }

    strict_roof_diagnostic_masks = None
    strict_roof_info = None
    if bool(config.strict_roof_guidance_enabled):
        if strict_roof_geometry is None:
            raise AssertionError("Strict roof geometry was not initialized.")
        (
            strict_roof_boundary,
            strict_roof_diagnostic_masks,
            strict_roof_info,
        ) = _build_strict_roof_interface(
            selected_roof=selected_roof,
            projection=association_projection,
            search_mask=search_mask,
            interior_image=interior_image,
            sky=localized_roles["sky"],
            vegetation=vegetation,
            explicit_foreground=explicit_occluder,
            additional_foreground=foreground_generic_non_target,
            geometry=strict_roof_geometry,
            config=config,
        )
        legacy_target = (
            selected_building | legacy_selected_roof_for_diagnostics
        ) & search_mask
        legacy_thin_silhouette = _inner_boundary(legacy_target)
        legacy_top_envelope, _ = _top_bottom_envelopes(
            legacy_target,
            int(config.envelope_tolerance_px),
        )
        legacy_roof_context = (
            _dilate_mask(legacy_selected_roof_for_diagnostics, adjacency)
            | _dilate_mask(localized_roles["sky"], adjacency)
            | legacy_top_envelope
        )
        legacy_diagnostic_roof_boundary = (
            _thicken_boundary(
                legacy_thin_silhouette & legacy_roof_context,
                int(config.boundary_thickness_px),
            )
            & search_mask
            & interior_image
        )
        strict_roof_diagnostic_masks[
            "roof_suppressed_legacy_boundary_mask"
        ] = legacy_diagnostic_roof_boundary & (~strict_roof_boundary)
        # Strict mode has one consumed roof interface.  The same foreground-safe
        # mask is exposed through all existing map families; no inferred bridge
        # is inserted into any of them.
        diagnostic_boundary_maps["roof"] = strict_roof_boundary.copy()
        boundary_maps["roof"] = strict_roof_boundary.copy()
        background_aware_boundary_maps["roof"] = (
            strict_roof_boundary.copy()
        )
        inferred_bridge = strict_roof_diagnostic_masks[
            "roof_inferred_bridge_not_consumed_mask"
        ]
        for map_family in (
            diagnostic_boundary_maps,
            boundary_maps,
            background_aware_boundary_maps,
        ):
            if bool((map_family["roof"] & inferred_bridge).any()):
                raise AssertionError(
                    "A diagnostic-only roof bridge leaked into boundary maps."
                )

    role_summaries = {}
    for role in CANONICAL_PREFIT_ROLES:
        if role == "building":
            role_summaries[role] = {
                **role_input_summary[role],
                **building_summary,
            }
        elif role == "roof":
            role_summaries[role] = {
                **role_input_summary[role],
                **roof_summary,
            }
        else:
            localized = localized_roles[role]
            localized_instance_count = int(sum(
                bool((instance & search_mask).any())
                for instance in role_stacks[role]
            ))
            role_summaries[role] = {
                **role_input_summary[role],
                "role": role,
                "raw_instances": int(role_stacks[role].shape[0]),
                "selected_instances": localized_instance_count,
                "selected_pixels": int(localized.sum()),
                "reason": (
                    "localized_to_projection_search_region"
                    if localized_instance_count > 0
                    else "no_instance_intersects_projection_search_region"
                ),
            }

    has_target = bool(target.any())
    has_occluder = bool(excluded_evidence.any())
    uses_guidance = bool(has_target or has_occluder)
    if has_target:
        reason = "semantic_target_and_context_available"
    elif has_occluder:
        reason = "occluder_only_guidance_no_target_instance"
    else:
        reason = "no_projection_local_semantic_instances"
    metadata = {
        "version": 3,
        "shape_hw": [int(height), int(width)],
        "uses_semantic_guidance": uses_guidance,
        "fallback_used": not has_target,
        "reason": reason,
        "projection_pixels": int(projection.sum()),
        "association_projection_pixels": int(association_projection.sum()),
        "external_exclusion_pixels": int(external_exclusion.sum()),
        "search_pixels": int(search_mask.sum()),
        "valid_evidence_pixels": int(valid_evidence.sum()),
        "excluded_occluder_pixels": int(excluded_evidence.sum()),
        "hard_occluder_pixels": int(hard_occluder.sum()),
        "generic_non_target_pixels": int(generic_non_target.sum()),
        "foreground_pixels": int(foreground_mask.sum()),
        "background_pixels": int(background_mask.sum()),
        "foreground_hard_context_pixels": int(
            foreground_hard_context.sum()
        ),
        "background_hard_context_pixels": int(
            background_hard_context.sum()
        ),
        "foreground_generic_non_target_pixels": int(
            foreground_generic_non_target.sum()
        ),
        "background_generic_non_target_pixels": int(
            background_generic_non_target.sum()
        ),
        "foreground_excluded_evidence_pixels": int(
            foreground_excluded_evidence.sum()
        ),
        "background_aware_valid_evidence_pixels": int(
            background_aware_valid_evidence.sum()
        ),
        "generic_non_target": generic_non_target_info,
        "target_pixels": int(target.sum()),
        "roles": role_summaries,
        "selected_target_instances": {
            "building": list(building_details),
            "roof": list(roof_details),
        },
        "boundary_pixels_by_class": {
            role: int(mask.sum()) for role, mask in boundary_maps.items()
        },
        "diagnostic_boundary_pixels_by_class": {
            role: int(mask.sum())
            for role, mask in diagnostic_boundary_maps.items()
        },
        "boundary_classes": ["roof", "wall", "base", "silhouette"],
        "unknown_roles": unknown_roles,
        "inputs": input_metadata,
        "roof_prompt_stage_separation": {
            "fit_role": "roof",
            "downstream_stack_supplied": bool(
                downstream_roof_stack_supplied
            ),
            "roof_prompt_mask_source": (
                "downstream_roof_mask_stack"
                if downstream_roof_stack_supplied
                else "canonical_roof_compatibility_fallback"
            ),
            "downstream_input": downstream_roof_input_metadata,
            "fit_roof_prompt_pixels": int(fit_roof_prompt_mask.sum()),
            "downstream_roof_prompt_pixels": int(
                downstream_roof_prompt_mask.sum()
            ),
        },
        "config": asdict(config),
    }
    if bool(config.strict_roof_guidance_enabled):
        expected_fit_roof = (
            _mask_union(role_stacks["roof"], projection.shape) & search_mask
        )
        if not np.array_equal(fit_roof_prompt_mask, expected_fit_roof):
            raise AssertionError(
                "Strict association unexpectedly changed fit-prompt roof evidence."
            )
        expected_downstream_roof = (
            _mask_union(normalized_downstream_roof_stack, projection.shape)
            & search_mask
            if downstream_roof_stack_supplied
            else expected_fit_roof
        )
        if not np.array_equal(roof_prompt_mask, expected_downstream_roof):
            raise AssertionError(
                "Strict fitting unexpectedly changed downstream roof evidence."
            )
        metadata["strict_roof_guidance"] = {
            "enabled": True,
            "roof_instances": list(strict_roof_instance_decisions),
            "fit_roof_prompt_pixels": int(fit_roof_prompt_mask.sum()),
            "raw_roof_prompt_pixels": int(roof_prompt_mask.sum()),
            "downstream_roof_prompt_pixels": int(roof_prompt_mask.sum()),
            "fit_selected_roof_pixels": int(selected_roof.sum()),
            "fit_selected_roof_instance_count": int(len(roof_details)),
            "raw_roof_prompt_mask_preserved": True,
            **dict(strict_roof_info or {}),
        }

    result = {
        "raw_projection_mask": projection.copy(),
        "external_exclusion_mask": external_exclusion.copy(),
        "local_search_mask": search_mask,
        "valid_evidence_mask": valid_evidence,
        "selected_building_mask": selected_building,
        "selected_roof_mask": selected_roof,
        "fit_roof_prompt_mask": fit_roof_prompt_mask,
        "downstream_roof_prompt_mask": downstream_roof_prompt_mask,
        "roof_prompt_mask": roof_prompt_mask,
        "target_semantic_mask": target,
        "sky_mask": localized_roles["sky"],
        "ground_mask": localized_roles["ground"],
        "vegetation_mask": vegetation,
        "hard_occluder_mask": hard_occluder,
        "foreground_hard_context_mask": foreground_hard_context,
        "background_hard_context_mask": background_hard_context,
        "generic_non_target_mask": generic_non_target,
        "foreground_generic_non_target_mask": (
            foreground_generic_non_target
        ),
        "background_generic_non_target_mask": (
            background_generic_non_target
        ),
        "generic_occluder_proposal_mask": localized_roles[
            "generic_occluder"
        ],
        "occluder_mask": raw_occluder,
        "excluded_evidence_mask": excluded_evidence,
        "foreground_mask": foreground_mask,
        "background_mask": background_mask,
        "foreground_excluded_evidence_mask": (
            foreground_excluded_evidence
        ),
        "background_aware_valid_evidence_mask": (
            background_aware_valid_evidence
        ),
        "semantic_valid_evidence_mask": semantic_valid_evidence,
        "boundary_maps": boundary_maps,
        "background_aware_boundary_maps": background_aware_boundary_maps,
        "diagnostic_boundary_maps": diagnostic_boundary_maps,
        "metadata": metadata,
    }
    if strict_roof_diagnostic_masks is not None:
        result["strict_roof_diagnostic_masks"] = (
            strict_roof_diagnostic_masks
        )
    return result


def assess_prefit_candidate_visibility(
    guidance: Mapping[str, object],
    *,
    minimum_target_projection_pixels: int = 250,
    minimum_target_support_fraction: float = 0.10,
    maximum_occluder_fraction: float = 0.80,
    low_support_occluder_fraction: float = 0.60,
    minimum_largest_visible_component_fraction: float = 0.05,
    maximum_whole_model_target_area_ratio: float = 6.0,
    reject_when_target_semantics_absent: bool = True,
) -> Dict[str, object]:
    """Assess whether one target wall has enough real image support to fit.

    This gate is deliberately evaluated on target-wall guidance, rather than on
    the much larger whole-model search region.  A missing SAM result is treated
    as an unavailable signal and falls back to geometric/OSM selection; a
    successful segmentation with no target building or roof in the anchored
    wall region is a high-confidence rejection.
    """
    metadata = dict(guidance.get("metadata", {}))

    projection = np.asarray(guidance["raw_projection_mask"], dtype=bool)
    shape = projection.shape

    def mask(name: str) -> np.ndarray:
        value = guidance.get(name)
        if value is None:
            return np.zeros(shape, dtype=bool)
        result = np.asarray(value, dtype=bool)
        if result.shape != shape:
            raise ValueError(
                f"Candidate visibility mask {name!r} does not match projection."
            )
        return result

    external = mask("external_exclusion_mask")
    target_semantics = mask("target_semantic_mask")
    # Include conservative generic residuals here. They are intentionally not
    # allowed to relax an OSM or explicit tree/vehicle/structural exclusion.
    occluder = mask("occluder_mask")
    usable_projection = projection & (~external)
    projection_pixels = int(projection.sum())
    usable_pixels = int(usable_projection.sum())
    target_on_wall = target_semantics & usable_projection
    visible_target_support = target_on_wall & (~occluder)
    target_support_pixels = int(visible_target_support.sum())
    target_semantic_pixels = int(target_on_wall.sum())
    occluder_pixels = int((occluder & usable_projection).sum())
    visible_geometric = usable_projection & (~occluder)
    visible_pixels = int(visible_geometric.sum())

    denominator = max(usable_pixels, 1)
    target_support_fraction = float(target_support_pixels / denominator)
    target_semantic_fraction = float(target_semantic_pixels / denominator)
    occluder_fraction = float(occluder_pixels / denominator)
    combined_visible_fraction = float(visible_pixels / denominator)
    largest_component_fraction = 0.0
    if visible_pixels > 0:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            visible_geometric.astype(np.uint8),
            connectivity=8,
        )
        if count > 1:
            largest_component_fraction = float(
                int(stats[1:, cv2.CC_STAT_AREA].max()) / denominator
            )

    segmentation_available = bool(metadata.get("segmentation_available", False))
    legacy_whole_model_target_area_ratio = float(guidance.get(
        "whole_model_target_area_ratio",
        metadata.get("whole_model_target_area_ratio", 0.0),
    ))
    whole_model_projection_pixels = int(guidance.get(
        "whole_model_projection_pixels",
        metadata.get("whole_model_projection_pixels", 0),
    ))
    selected_target_instances = metadata.get("selected_target_instances", {})
    selected_building_instances = (
        selected_target_instances.get("building", [])
        if isinstance(selected_target_instances, Mapping)
        else []
    )
    selected_building_instance_areas = []
    for instance in selected_building_instances:
        if not isinstance(instance, Mapping):
            continue
        try:
            area = float(instance.get("area_px", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(area) and area > 0.0:
            selected_building_instance_areas.append(area)
    selected_building_instance_area_px = float(
        max(selected_building_instance_areas, default=0.0)
    )
    if (
        selected_building_instance_area_px > 0.0
        and whole_model_projection_pixels > 0
    ):
        # ``target_semantic_mask`` is clipped to the local search region.  Its
        # pixel count therefore makes a foreground facade look target-sized.
        # SAM's instance metadata retains the complete mask area and exposes
        # the identity mismatch even when the target-wall projection is tiny.
        whole_model_target_area_ratio = float(
            selected_building_instance_area_px
            / whole_model_projection_pixels
        )
        whole_model_target_area_ratio_source = (
            "selected_target_instances.building.max_area_px"
        )
    else:
        whole_model_target_area_ratio = legacy_whole_model_target_area_ratio
        whole_model_target_area_ratio_source = "clipped_target_mask_fallback"
    accepted = True
    fallback_used = False
    reason = "semantic_target_visibility_sufficient"
    if not segmentation_available:
        fallback_used = True
        reason = "semantic_segmentation_unavailable_geometry_fallback"
    elif (
        math.isfinite(whole_model_target_area_ratio)
        and whole_model_target_area_ratio
        > float(maximum_whole_model_target_area_ratio)
    ):
        accepted = False
        reason = "semantic_building_match_far_larger_than_projected_model"
    elif projection_pixels < int(minimum_target_projection_pixels):
        fallback_used = True
        reason = "target_projection_too_small_for_semantic_rejection"
    elif usable_pixels <= 0:
        accepted = False
        reason = "no_target_projection_remaining_after_external_exclusion"
    elif (
        reject_when_target_semantics_absent
        and target_semantic_pixels <= 0
    ):
        accepted = False
        reason = "no_building_or_roof_in_anchored_target_search"
    elif occluder_fraction >= float(maximum_occluder_fraction):
        accepted = False
        reason = "target_wall_almost_fully_occluded_by_non_osm_elements"
    elif (
        target_support_fraction < float(minimum_target_support_fraction)
        and occluder_fraction >= float(low_support_occluder_fraction)
    ):
        accepted = False
        reason = "insufficient_target_support_behind_non_osm_occluders"
    elif target_support_fraction < float(minimum_target_support_fraction):
        accepted = False
        reason = "insufficient_building_or_roof_support_in_target_projection"
    elif (
        combined_visible_fraction > 0.0
        and largest_component_fraction
        < float(minimum_largest_visible_component_fraction)
    ):
        accepted = False
        reason = "remaining_target_visibility_is_only_scattered_fragments"

    return {
        "accepted": bool(accepted),
        "fallback_used": bool(fallback_used),
        "reason": reason,
        "segmentation_available": segmentation_available,
        "whole_model_target_area_ratio": whole_model_target_area_ratio,
        "whole_model_target_area_ratio_source": (
            whole_model_target_area_ratio_source
        ),
        "whole_model_projection_pixels": whole_model_projection_pixels,
        "selected_building_instance_area_px": (
            selected_building_instance_area_px
        ),
        "projection_pixels": projection_pixels,
        "usable_projection_pixels": usable_pixels,
        "target_semantic_pixels": target_semantic_pixels,
        "target_support_pixels": target_support_pixels,
        "occluder_pixels": occluder_pixels,
        "target_semantic_fraction": target_semantic_fraction,
        "target_support_fraction": target_support_fraction,
        "non_osm_occluder_fraction": occluder_fraction,
        "combined_visible_fraction": combined_visible_fraction,
        "largest_visible_component_fraction": largest_component_fraction,
        "thresholds": {
            "minimum_target_projection_pixels": int(
                minimum_target_projection_pixels
            ),
            "minimum_target_support_fraction": float(
                minimum_target_support_fraction
            ),
            "maximum_occluder_fraction": float(maximum_occluder_fraction),
            "low_support_occluder_fraction": float(
                low_support_occluder_fraction
            ),
            "minimum_largest_visible_component_fraction": float(
                minimum_largest_visible_component_fraction
            ),
            "maximum_whole_model_target_area_ratio": float(
                maximum_whole_model_target_area_ratio
            ),
            "reject_when_target_semantics_absent": bool(
                reject_when_target_semantics_absent
            ),
        },
    }


def draw_prefit_semantic_guides(
    image_rgb: np.ndarray,
    guidance: Mapping[str, object],
    *,
    line_thickness_px: Optional[int] = None,
) -> np.ndarray:
    """Draw diagnostic SAM3 guide lines on an RGB image.

    Fitting continues to consume ``boundary_maps``, whose pixels are removed
    wherever evidence is excluded.  Diagnostics prefer the parallel unmasked
    maps so the guides stay readable through translucent exclusion fills.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image_rgb must be an HxWx3 or HxWx4 image.")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0)
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.copy()

    shape = image.shape[:2]
    boundary_source = guidance.get("diagnostic_boundary_maps")
    if not isinstance(boundary_source, Mapping):
        boundary_source = guidance.get("boundary_maps", {})
    if not isinstance(boundary_source, Mapping):
        boundary_source = {}

    strict_diagnostics = guidance.get("strict_roof_diagnostic_masks")
    if isinstance(strict_diagnostics, Mapping):
        yy, xx = np.indices(shape)
        dot_pattern = ((xx + yy) % 8) < 3
        for name, color in (
            (
                "roof_suppressed_legacy_boundary_mask",
                SAM_SUPPRESSED_ROOF_GUIDE_RGB,
            ),
            (
                "roof_inferred_bridge_not_consumed_mask",
                SAM_INFERRED_ROOF_BRIDGE_RGB,
            ),
        ):
            value = strict_diagnostics.get(name)
            if value is None:
                continue
            mask = np.asarray(value, dtype=bool)
            if mask.shape != shape:
                raise ValueError(
                    f"Strict roof diagnostic {name!r} does not match image shape."
                )
            image[mask & dot_pattern] = np.asarray(color, dtype=np.uint8)

    for boundary_class in ("silhouette", "roof", "wall", "base"):
        value = boundary_source.get(boundary_class)
        if value is None:
            continue
        mask = np.asarray(value, dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"Guidance boundary {boundary_class!r} does not match image shape."
            )
        boundary_style = SEMANTIC_GUIDE_LINES[boundary_class]
        boundary_width = (
            boundary_style.width_px
            if line_thickness_px is None
            else max(1, int(line_thickness_px))
        )
        if boundary_width > 1:
            mask = _dilate_mask(mask, boundary_width - 1)
        image[mask] = np.asarray(boundary_style.rgb, dtype=np.uint8)
    return image


def create_prefit_semantic_guidance_overlay(
    image_rgb: np.ndarray,
    guidance: Mapping[str, object],
    *,
    fill_alpha: Optional[float] = None,
    line_thickness_px: Optional[int] = None,
    draw_raw_projection_outline: bool = True,
    draw_legend: bool = True,
) -> np.ndarray:
    """Return an RGB debug overlay for one guidance result."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image_rgb must be an HxWx3 or HxWx4 image.")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0)
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = image.copy()

    shape = image.shape[:2]

    def guidance_mask(name, *, boundary=False):
        source = guidance.get("boundary_maps", {}) if boundary else guidance
        value = source.get(name) if isinstance(source, Mapping) else None
        if value is None:
            return np.zeros(shape, dtype=bool)
        mask = np.asarray(value, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"Guidance mask {name!r} does not match image shape.")
        return mask

    overlay = image.copy()
    search = guidance_mask("local_search_mask")
    if search.any():
        overlay[~search] = np.clip(
            overlay[~search].astype(np.float32) * OUTSIDE_SEARCH_DIM_FACTOR,
            0.0,
            255.0,
        ).astype(np.uint8)

    alpha = float(np.clip(
        SEMANTIC_FILL_ALPHA if fill_alpha is None else fill_alpha,
        0.0,
        1.0,
    ))

    def blend(mask, color):
        if not mask.any() or alpha <= 0.0:
            return
        source = overlay[mask].astype(np.float32)
        target = np.asarray(color, dtype=np.float32)
        overlay[mask] = np.clip(
            source * (1.0 - alpha) + target * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)

    background_aware_active = bool(guidance.get("background_aware_active", False))
    if background_aware_active:
        # Draw trusted context beneath target fills and guide lines.  This is a
        # diagnostic only; the masks consumed by the fitter are unchanged.
        blend(guidance_mask("background_mask"), SAM_BACKGROUND_CONTEXT_FILL_RGB)
    blend(guidance_mask("selected_building_mask"), SAM_TARGET_BUILDING_FILL_RGB)
    blend(guidance_mask("selected_roof_mask"), SAM_TARGET_ROOF_FILL_RGB)
    # G remains in ``excluded_evidence_mask`` and therefore still suppresses
    # fitting evidence.  It is intentionally absent from diagnostics.  Pink
    # shows only explicit/vegetation exclusions plus their real safety border.
    hard_occluder = guidance_mask(
        "foreground_mask" if background_aware_active else "hard_occluder_mask"
    )
    if "hard_occluder_mask" not in guidance:
        hard_occluder = (
            guidance_mask("vegetation_mask")
            | (
                guidance_mask("occluder_mask")
                & (~guidance_mask("generic_non_target_mask"))
            )
        )
    metadata = guidance.get("metadata", {})
    metadata_config = (
        metadata.get("config", {}) if isinstance(metadata, Mapping) else {}
    )
    dilation_px = int(
        metadata_config.get("occluder_dilation_px", 3)
        if isinstance(metadata_config, Mapping)
        else 3
    )
    prompted_exclusion_display = _dilate_mask(
        hard_occluder,
        max(0, dilation_px),
    )
    if "local_search_mask" in guidance:
        prompted_exclusion_display &= search
    blend(prompted_exclusion_display, SAM_PROMPTED_OCCLUDER_FILL_RGB)
    blend(hard_occluder, SAM_PROMPTED_OCCLUDER_FILL_RGB)
    strict_diagnostics = guidance.get("strict_roof_diagnostic_masks")
    if isinstance(strict_diagnostics, Mapping):
        strict_guard_value = strict_diagnostics.get("foreground_guard_mask")
        if strict_guard_value is not None:
            strict_guard = np.asarray(strict_guard_value, dtype=bool)
            if strict_guard.shape != shape:
                raise ValueError(
                    "Strict roof foreground guard does not match image shape."
                )
            blend(strict_guard, SAM_PROMPTED_OCCLUDER_FILL_RGB)

    # Guidance sits below model geometry in combined diagnostic overlays.
    overlay = draw_prefit_semantic_guides(
        overlay,
        guidance,
        line_thickness_px=line_thickness_px,
    )

    projection = guidance.get("raw_projection_mask")
    if draw_raw_projection_outline and projection is not None:
        projection = np.asarray(projection, dtype=bool)
        if projection.shape != shape:
            raise ValueError("raw_projection_mask in guidance does not match image.")
        contours, _ = cv2.findContours(
            projection.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        height, width = shape

        def on_exact_image_frame(point):
            return bool(
                int(point[0]) == 0
                or int(point[0]) == width - 1
                or int(point[1]) == 0
                or int(point[1]) == height - 1
            )

        raw_style = RAW_MODEL_LINE
        if line_thickness_px is not None:
            raw_style = OverlayLineStyle(
                rgb=RAW_MODEL_LINE.rgb,
                width_px=max(1, int(line_thickness_px)),
            )
        for contour in contours:
            # Work on the dense raster contour. Any unit contour edge whose
            # two endpoints touch the viewport is a clipping closure, not a
            # projected model edge. Removing it before simplification also
            # catches right-to-bottom corner bridges.
            points = np.asarray(contour[:, 0, :], dtype=np.int32)
            if len(points) < 2:
                continue
            for index in range(len(points)):
                point0 = points[index]
                point1 = points[(index + 1) % len(points)]
                if (
                    on_exact_image_frame(point0)
                    and on_exact_image_frame(point1)
                ):
                    continue
                draw_styled_line(
                    overlay,
                    point0,
                    point1,
                    raw_style,
                    color_space="rgb",
                )

    if draw_legend:
        draw_overlay_legend(
            overlay,
            (
                *(
                    BACKGROUND_AWARE_SEMANTIC_LEGEND_ROWS
                    if background_aware_active
                    else SEMANTIC_LEGEND_ROWS
                ),
                *(
                    (STRICT_ROOF_AUDIT_LEGEND_ROW,)
                    if isinstance(strict_diagnostics, Mapping)
                    else ()
                ),
                model_projection_legend(fitted=False),
                SEARCH_LEGEND_ROW,
            ),
            color_space="rgb",
        )
    return overlay
