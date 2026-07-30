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


CANONICAL_PREFIT_ROLES: Tuple[str, ...] = (
    "building",
    "roof",
    "sky",
    "ground",
    "vegetation",
    "occluder",
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
    occluder_dilation_px: int = 5
    context_adjacency_px: int = 4
    envelope_tolerance_px: int = 2
    boundary_thickness_px: int = 2
    image_border_exclusion_px: int = 2

    def __post_init__(self):
        integer_nonnegative = (
            "search_dilation_px",
            "minimum_instance_area_px",
            "target_min_overlap_pixels",
            "target_min_new_projection_pixels",
            "target_max_instances_per_role",
            "occluder_dilation_px",
            "context_adjacency_px",
            "envelope_tolerance_px",
            "boundary_thickness_px",
            "image_border_exclusion_px",
        )
        for name in integer_nonnegative:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.target_max_instances_per_role) < 1:
            raise ValueError("target_max_instances_per_role must be at least one.")
        if float(self.target_association_distance_px) < 0.0:
            raise ValueError("target_association_distance_px must be non-negative.")
        for name in (
            "target_min_overlap_fraction",
            "target_min_local_fraction",
            "target_relative_score_threshold",
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
) -> Dict[str, object]:
    """Build conservative semantic evidence localized to a raw model projection.

    ``building`` and ``roof`` instances are associated individually with the
    projection.  Context roles are localized to its dilated search region.
    Sky and ground provide positive boundary context; only vegetation and
    explicit occluders remove evidence.
    """
    config = config or PrefitSemanticGuidanceConfig()
    projection = np.asarray(raw_projection_mask, dtype=bool)
    if projection.ndim != 2:
        raise ValueError("raw_projection_mask must be a two-dimensional mask.")
    height, width = projection.shape
    if height == 0 or width == 0:
        raise ValueError("raw_projection_mask must have non-zero height and width.")

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

    if not projection.any():
        zero = np.zeros_like(projection)
        one = np.ones_like(projection)
        metadata = {
            "version": 1,
            "shape_hw": [int(height), int(width)],
            "uses_semantic_guidance": False,
            "fallback_used": True,
            "reason": "empty_projection_mask",
            "projection_pixels": 0,
            "search_pixels": 0,
            "valid_evidence_pixels": int(one.sum()),
            "excluded_occluder_pixels": 0,
            "target_pixels": 0,
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
            "unknown_roles": unknown_roles,
            "inputs": input_metadata,
            "config": asdict(config),
        }
        return {
            "raw_projection_mask": projection.copy(),
            "local_search_mask": zero,
            "valid_evidence_mask": one,
            "selected_building_mask": zero.copy(),
            "selected_roof_mask": zero.copy(),
            "target_semantic_mask": zero.copy(),
            "sky_mask": zero.copy(),
            "ground_mask": zero.copy(),
            "vegetation_mask": zero.copy(),
            "occluder_mask": zero.copy(),
            "excluded_evidence_mask": zero.copy(),
            "boundary_maps": {
                role: zero.copy()
                for role in ("roof", "wall", "base", "silhouette")
            },
            "metadata": metadata,
        }

    search_mask, distance_to_projection = _local_search_mask(
        projection,
        int(config.search_dilation_px),
    )
    selected_building, _, building_details, building_summary = (
        _associate_target_instances(
            role="building",
            stack=role_stacks["building"],
            instance_refs=role_refs["building"],
            projection=projection,
            search_mask=search_mask,
            distance_to_projection=distance_to_projection,
            config=config,
        )
    )
    selected_roof, _, roof_details, roof_summary = _associate_target_instances(
        role="roof",
        stack=role_stacks["roof"],
        instance_refs=role_refs["roof"],
        projection=projection,
        search_mask=search_mask,
        distance_to_projection=distance_to_projection,
        config=config,
    )

    localized_roles = {}
    for role in ("sky", "ground", "vegetation", "occluder"):
        localized_roles[role] = (
            _mask_union(role_stacks[role], projection.shape) & search_mask
        )

    vegetation = localized_roles["vegetation"]
    explicit_occluder = localized_roles["occluder"]
    raw_occluder = vegetation | explicit_occluder
    excluded_evidence = _dilate_mask(
        raw_occluder,
        int(config.occluder_dilation_px),
    ) & search_mask
    interior_image = _interior_image_mask(
        projection.shape,
        int(config.image_border_exclusion_px),
    )
    valid_evidence = search_mask & (~excluded_evidence) & interior_image

    target = (selected_building | selected_roof) & search_mask
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
    boundary_maps = {
        role: (
            _thicken_boundary(seed, int(config.boundary_thickness_px))
            & valid_evidence
        )
        for role, seed in seeds.items()
    }

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
        "version": 1,
        "shape_hw": [int(height), int(width)],
        "uses_semantic_guidance": uses_guidance,
        "fallback_used": not has_target,
        "reason": reason,
        "projection_pixels": int(projection.sum()),
        "search_pixels": int(search_mask.sum()),
        "valid_evidence_pixels": int(valid_evidence.sum()),
        "excluded_occluder_pixels": int(excluded_evidence.sum()),
        "target_pixels": int(target.sum()),
        "roles": role_summaries,
        "selected_target_instances": {
            "building": list(building_details),
            "roof": list(roof_details),
        },
        "boundary_pixels_by_class": {
            role: int(mask.sum()) for role, mask in boundary_maps.items()
        },
        "boundary_classes": ["roof", "wall", "base", "silhouette"],
        "unknown_roles": unknown_roles,
        "inputs": input_metadata,
        "config": asdict(config),
    }

    return {
        "raw_projection_mask": projection.copy(),
        "local_search_mask": search_mask,
        "valid_evidence_mask": valid_evidence,
        "selected_building_mask": selected_building,
        "selected_roof_mask": selected_roof,
        "target_semantic_mask": target,
        "sky_mask": localized_roles["sky"],
        "ground_mask": localized_roles["ground"],
        "vegetation_mask": vegetation,
        "occluder_mask": raw_occluder,
        "excluded_evidence_mask": excluded_evidence,
        "boundary_maps": boundary_maps,
        "metadata": metadata,
    }


def create_prefit_semantic_guidance_overlay(
    image_rgb: np.ndarray,
    guidance: Mapping[str, object],
    *,
    fill_alpha: float = 0.28,
    line_thickness_px: int = 2,
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
            overlay[~search].astype(np.float32) * 0.45,
            0.0,
            255.0,
        ).astype(np.uint8)

    alpha = float(np.clip(fill_alpha, 0.0, 1.0))

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

    blend(guidance_mask("selected_building_mask"), (0, 190, 240))
    blend(guidance_mask("selected_roof_mask"), (255, 210, 0))
    blend(guidance_mask("excluded_evidence_mask"), (255, 0, 170))

    boundary_colors = {
        "silhouette": (30, 150, 255),
        "roof": (255, 235, 0),
        "wall": (0, 230, 90),
        "base": (185, 185, 185),
    }
    for boundary_class in ("silhouette", "roof", "wall", "base"):
        mask = guidance_mask(boundary_class, boundary=True)
        if int(line_thickness_px) > 1:
            mask = _dilate_mask(mask, int(line_thickness_px) - 1)
        overlay[mask] = np.asarray(
            boundary_colors[boundary_class],
            dtype=np.uint8,
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
                cv2.line(
                    overlay,
                    tuple(point0.tolist()),
                    tuple(point1.tolist()),
                    (100, 120, 255),
                    max(1, int(line_thickness_px)),
                    lineType=cv2.LINE_AA,
                )

    if draw_legend:
        legend_rows = [
            "SAM3 fills: cyan target building | yellow roof | pink excluded occluder",
            "SAM3 guides: yellow roof | green wall | gray base | orange silhouette",
            "violet: raw model anchor | darkened: outside projection-local search",
        ]
        for index, row in enumerate(legend_rows):
            origin = (8, 20 + index * 19)
            cv2.putText(
                overlay,
                row,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                row,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return overlay
