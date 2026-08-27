# -*- coding: utf-8 -*-
"""Reuse full-image semantic evidence for facade texture extraction."""

from typing import Dict, Mapping, Optional

import cv2
import numpy as np


def _mask_from_guidance(
    guidance: Optional[Mapping[str, object]],
    name: str,
    shape,
) -> np.ndarray:
    if not isinstance(guidance, Mapping):
        return np.zeros(shape, dtype=bool)
    value = guidance.get(name)
    if value is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(
            f"Semantic guidance mask {name!r} has shape {mask.shape}, "
            f"expected {shape}."
        )
    return mask


def _fill_small_enclosed_holes(
    mask: np.ndarray,
    allowed: np.ndarray,
    protected_exclusion: np.ndarray,
    maximum_area_px: int,
) -> tuple[np.ndarray, int, int]:
    """Restore small facade details omitted inside an otherwise solid target."""
    maximum_area = max(0, int(maximum_area_px))
    if maximum_area <= 0:
        return mask.copy(), 0, 0

    allowed_u8 = allowed.astype(np.uint8)
    eroded = cv2.erode(
        allowed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    allowed_boundary = allowed & (~eroded)
    holes = allowed & (~mask)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8),
        connectivity=8,
    )
    result = mask.copy()
    filled_components = 0
    filled_pixels = 0
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > maximum_area:
            continue
        if bool((component & allowed_boundary).any()):
            continue
        if bool((component & protected_exclusion).any()):
            continue
        result[component] = True
        filled_components += 1
        filled_pixels += area
    return result, filled_components, filled_pixels


def _vertical_wall_boundary_envelopes(
    wall_mask: np.ndarray,
    thickness_px: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return narrow top and bottom wall-envelope bands in each column."""
    wall = np.asarray(wall_mask, dtype=bool)
    thickness = max(1, int(thickness_px))
    top = np.zeros_like(wall)
    bottom = np.zeros_like(wall)
    for x in np.flatnonzero(wall.any(axis=0)):
        ys = np.flatnonzero(wall[:, x])
        top[ys[ys <= int(ys[0]) + thickness - 1], int(x)] = True
        bottom[ys[ys >= int(ys[-1]) - thickness + 1], int(x)] = True
    return top, bottom


def build_post_hough_roof_structure_removal(
    wall_mask,
    roof_mask,
    *,
    enabled: bool = True,
    connection_tolerance_px: int = 3,
    boundary_seed_px: int = 2,
    minimum_divider_component_area_px: int = 32,
    minimum_partition_area_px: int = 80,
    minimum_partition_fraction: float = 0.03,
) -> Dict[str, object]:
    """Mark segmented roofs and any lower structure separated by a roof.

    Split detection uses the complete rectified wall polygon, rather than its
    currently valid texture pixels. Existing facade holes therefore cannot
    turn an isolated roof into a false divider. A roof is a divider only when
    removing a small tolerance band around it leaves significant, disconnected
    wall regions attached to the top and bottom envelopes with no remaining
    top-to-bottom path.
    """
    wall = np.asarray(wall_mask, dtype=bool)
    roof = np.asarray(roof_mask, dtype=bool)
    if wall.ndim != 2:
        raise ValueError("wall_mask must be two-dimensional.")
    if roof.shape != wall.shape:
        raise ValueError("roof_mask must match wall_mask.")

    roof = roof & wall
    removal = roof.copy() if enabled else np.zeros_like(wall)
    below_roof = np.zeros_like(wall)
    wall_pixels = int(wall.sum())
    roof_pixels = int(roof.sum())
    tolerance = max(0, int(connection_tolerance_px))
    boundary_seed = max(1, int(boundary_seed_px))
    minimum_roof_area = max(1, int(minimum_divider_component_area_px))
    minimum_partition_fraction = max(
        0.0,
        float(minimum_partition_fraction),
    )
    required_partition_area = max(
        1,
        int(minimum_partition_area_px),
        int(np.ceil(wall_pixels * minimum_partition_fraction)),
    )

    result = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled" if not enabled else "no_roof_evidence",
        "wall_pixels": wall_pixels,
        "roof_pixels": roof_pixels,
        "removed_pixels": 0,
        "removed_roof_pixels": roof_pixels if enabled else 0,
        "removed_below_roof_pixels": 0,
        "roof_component_count": 0,
        "divider_component_count": 0,
        "roof_only_component_count": 0,
        "connection_tolerance_px": tolerance,
        "boundary_seed_px": boundary_seed,
        "minimum_divider_component_area_px": minimum_roof_area,
        "minimum_partition_area_px": required_partition_area,
        "minimum_partition_fraction": minimum_partition_fraction,
        "components": [],
        "roof_mask": roof,
        "below_roof_mask": below_roof,
        "removal_mask": removal,
    }
    if not enabled or wall_pixels <= 0 or roof_pixels <= 0:
        if enabled and wall_pixels <= 0:
            result["reason"] = "empty_wall"
        return result

    if tolerance > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * tolerance + 1, 2 * tolerance + 1),
        )
        analysis_roof = cv2.dilate(
            roof.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0
        analysis_roof &= wall
    else:
        analysis_roof = roof.copy()

    top_envelope, bottom_envelope = _vertical_wall_boundary_envelopes(
        wall,
        boundary_seed,
    )
    roof_count, roof_labels, roof_stats, _ = cv2.connectedComponentsWithStats(
        analysis_roof.astype(np.uint8),
        connectivity=8,
    )
    component_records = []
    divider_count = 0

    for roof_label in range(1, roof_count):
        component = roof_labels == roof_label
        component_area = int(roof_stats[roof_label, cv2.CC_STAT_AREA])
        record = {
            "component_index": int(roof_label - 1),
            "analysis_area_px": component_area,
            "raw_roof_area_px": int((component & roof).sum()),
            "divider": False,
            "reason": "component_too_small_for_divider_test",
            "upper_partition_pixels": 0,
            "lower_partition_pixels": 0,
            "top_to_bottom_bridge_pixels": 0,
        }
        if component_area < minimum_roof_area:
            component_records.append(record)
            continue

        remaining = wall & (~component)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            remaining.astype(np.uint8),
            connectivity=8,
        )
        upper_labels = []
        lower_labels = []
        bridge_area = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < required_partition_area:
                continue
            region = labels == label
            touches_top = bool((region & top_envelope).any())
            touches_bottom = bool((region & bottom_envelope).any())
            if touches_top and touches_bottom:
                bridge_area += area
            elif touches_top:
                upper_labels.append(label)
            elif touches_bottom:
                lower_labels.append(label)

        upper_area = int(sum(
            int(stats[label, cv2.CC_STAT_AREA]) for label in upper_labels
        ))
        lower_area = int(sum(
            int(stats[label, cv2.CC_STAT_AREA]) for label in lower_labels
        ))
        record.update({
            "upper_partition_pixels": upper_area,
            "lower_partition_pixels": lower_area,
            "top_to_bottom_bridge_pixels": int(bridge_area),
        })

        is_divider = bool(
            upper_labels
            and lower_labels
            and bridge_area == 0
        )
        if not is_divider:
            record["reason"] = (
                "top_to_bottom_path_remains"
                if bridge_area > 0
                else "no_significant_upper_and_lower_partition"
            )
            component_records.append(record)
            continue

        lower_region = np.isin(labels, np.asarray(lower_labels, dtype=np.int32))
        below_roof |= lower_region
        removal |= component | lower_region
        divider_count += 1
        record["divider"] = True
        record["reason"] = "roof_separates_top_and_bottom_wall_partitions"
        component_records.append(record)

    result.update({
        "applied": bool(removal.any()),
        "reason": (
            "divider_and_roof_regions_removed"
            if divider_count > 0
            else "roof_regions_removed_without_lower_partition"
        ),
        "removed_pixels": int(removal.sum()),
        "removed_below_roof_pixels": int(below_roof.sum()),
        "roof_component_count": int(max(0, roof_count - 1)),
        "divider_component_count": int(divider_count),
        "roof_only_component_count": int(max(0, roof_count - 1 - divider_count)),
        "components": component_records,
        "below_roof_mask": below_roof,
        "removal_mask": removal,
    })
    return result


def build_reused_prefit_facade_mask(
    guidance: Optional[Mapping[str, object]],
    fitted_projection_mask,
    *,
    external_exclusion_mask=None,
    enabled: bool = True,
    minimum_pixels: int = 250,
    minimum_wall_coverage: float = 0.35,
    closing_radius_px: int = 2,
    maximum_hole_area_px: int = 900,
    maximum_hard_exclusion_fraction: float = 0.85,
) -> Dict[str, object]:
    """Create an authoritative source-canvas facade mask without another model run.

    The target building/roof mask comes from the full-image SAM3 pass used by
    global fitting. Prompted foreground and vegetation are hard exclusions.
    If the selected target is too incomplete, the fitted projection remains
    the safe fallback while known exclusions are still removed.
    """
    projection = np.asarray(fitted_projection_mask, dtype=bool)
    if projection.ndim != 2:
        raise ValueError("fitted_projection_mask must be two-dimensional.")
    shape = projection.shape

    external = np.zeros(shape, dtype=bool)
    if external_exclusion_mask is not None:
        external = np.asarray(external_exclusion_mask, dtype=bool)
        if external.shape != shape:
            raise ValueError(
                "external_exclusion_mask must match fitted_projection_mask."
            )

    selected_building = _mask_from_guidance(
        guidance,
        "selected_building_mask",
        shape,
    )
    selected_roof = _mask_from_guidance(
        guidance,
        "selected_roof_mask",
        shape,
    )
    roof_evidence = _mask_from_guidance(
        guidance,
        "downstream_roof_prompt_mask",
        shape,
    )
    roof_evidence_source = "downstream_roof_prompt_mask"
    if not roof_evidence.any():
        roof_evidence = _mask_from_guidance(
            guidance,
            "roof_prompt_mask",
            shape,
        )
        roof_evidence_source = "roof_prompt_mask_compatibility_fallback"
    if not roof_evidence.any():
        roof_evidence = selected_roof
        roof_evidence_source = "selected_roof_mask_fallback"
    target = _mask_from_guidance(guidance, "target_semantic_mask", shape)
    if not target.any():
        target = selected_building | selected_roof

    hard_occluder = _mask_from_guidance(
        guidance,
        "hard_occluder_mask",
        shape,
    )
    if not (
        isinstance(guidance, Mapping)
        and "hard_occluder_mask" in guidance
    ):
        # Compatibility with guidance produced before hard/generic exclusions
        # were separated.
        hard_occluder = _mask_from_guidance(
            guidance,
            "occluder_mask",
            shape,
        )
    generic_non_target = _mask_from_guidance(
        guidance,
        "generic_non_target_mask",
        shape,
    )

    available_projection = projection & (~external)
    selected_roof_inside_projection = roof_evidence & available_projection
    # Generic residuals are deliberately not promoted to ``hard_occluder``:
    # they are category-independent and therefore less certain.  They still
    # describe pixels that the reusable target mask explicitly rejected, so
    # morphology/hole restoration must not silently paint them back into the
    # facade (the exact failure seen with an unprompted signboard).
    protected_exclusion = hard_occluder | generic_non_target | external
    candidate = (
        target
        & available_projection
        & (~hard_occluder)
        & (~generic_non_target)
    )

    radius = max(0, int(closing_radius_px))
    if radius > 0 and candidate.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ) > 0
        candidate &= (
            available_projection
            & (~hard_occluder)
            & (~generic_non_target)
        )

    (
        candidate,
        restored_hole_components,
        restored_hole_pixels,
    ) = _fill_small_enclosed_holes(
        candidate,
        available_projection,
        protected_exclusion,
        maximum_hole_area_px,
    )

    projection_pixels = int(available_projection.sum())
    hard_inside_available = hard_occluder & available_projection
    hard_exclusion_fraction = float(
        hard_inside_available.sum() / max(projection_pixels, 1)
    )
    maximum_hard_fraction = float(maximum_hard_exclusion_fraction)
    if not 0.0 <= maximum_hard_fraction <= 1.0:
        raise ValueError(
            "maximum_hard_exclusion_fraction must be in [0, 1]."
        )
    candidate_pixels = int(candidate.sum())
    coverage = candidate_pixels / max(projection_pixels, 1)
    accepted = bool(
        enabled
        and projection_pixels > 0
        and candidate_pixels >= max(1, int(minimum_pixels))
        and coverage >= max(0.0, float(minimum_wall_coverage))
    )

    if not bool(enabled):
        reason = "disabled"
    elif projection_pixels <= 0:
        reason = "empty_fitted_projection"
    elif not target.any():
        reason = "no_reusable_prefit_target"
    elif candidate_pixels < max(1, int(minimum_pixels)):
        reason = "insufficient_reused_target_pixels"
    elif coverage < max(0.0, float(minimum_wall_coverage)):
        reason = "insufficient_reused_target_coverage"
    else:
        reason = "accepted_reused_full_image_semantic_target"

    hard_exclusion_guard_applied = bool(
        projection_pixels > 0
        and hard_exclusion_fraction > maximum_hard_fraction
    )
    fallback_hard_occluder = (
        np.zeros(shape, dtype=bool)
        if hard_exclusion_guard_applied
        else hard_occluder
    )
    fallback = available_projection & (~fallback_hard_occluder)
    effective = candidate if accepted else fallback
    excluded_inside_projection = projection & (~effective)
    return {
        "accepted": accepted,
        "reason": reason,
        "fallback_to_fitted_projection": not accepted,
        "projection_pixels": int(projection.sum()),
        "available_projection_pixels": projection_pixels,
        "target_pixels": int(target.sum()),
        "candidate_pixels": candidate_pixels,
        "candidate_wall_coverage": float(coverage),
        "hard_occluder_pixels": int((hard_occluder & projection).sum()),
        "hard_occluder_available_projection_fraction": (
            hard_exclusion_fraction
        ),
        "maximum_hard_exclusion_fraction": maximum_hard_fraction,
        "hard_exclusion_guard_applied_in_projection_fallback": (
            hard_exclusion_guard_applied
        ),
        "fallback_hard_occluder_pixels": int(
            (fallback_hard_occluder & projection).sum()
        ),
        "generic_non_target_pixels": int(
            (generic_non_target & projection).sum()
        ),
        "external_exclusion_pixels": int((external & projection).sum()),
        "restored_small_hole_components": int(restored_hole_components),
        "restored_small_hole_pixels": int(restored_hole_pixels),
        "second_segmentation_inference_run": False,
        "coordinate_space": "selected_source_full_image",
        "roof_evidence_source": roof_evidence_source,
        "selected_roof_mask": selected_roof_inside_projection,
        "semantic_candidate_mask": candidate,
        "hard_occluder_mask": hard_occluder,
        "generic_non_target_mask": generic_non_target,
        "effective_content_mask": effective,
        "excluded_inside_projection_mask": excluded_inside_projection,
    }
