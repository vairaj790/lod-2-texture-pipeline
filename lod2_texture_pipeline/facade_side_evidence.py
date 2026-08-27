# -*- coding: utf-8 -*-
"""Side-specific facade evidence before semantic projection clipping."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .projection import project_points_world_to_image, render_model_depth_map
from .wireframe_fit import apply_homography


def cyclic_group_neighbors(
    group_records: Sequence[Mapping[str, Any]],
    loop_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Optional[Mapping[str, Any]]]:
    """Return topology-start/end neighbors, including wrapped groups."""
    if not group_records or not loop_records:
        return {"start": None, "end": None}
    by_index = {
        int(record["loop_index"]): record
        for record in loop_records
        if record.get("loop_index") is not None
    }
    count = len(loop_records)
    group_indices = {int(record["loop_index"]) for record in group_records}
    first = int(group_records[0]["loop_index"])
    last = int(group_records[-1]["loop_index"])
    start_index = (first - 1) % count
    end_index = (last + 1) % count
    return {
        "start": None if start_index in group_indices else by_index.get(start_index),
        "end": None if end_index in group_indices else by_index.get(end_index),
    }


def _line_band(
    shape_hw: Tuple[int, int], line_xy: np.ndarray, radius_px: int
) -> np.ndarray:
    height, width = [int(value) for value in shape_hw]
    seed = np.zeros((height, width), dtype=np.uint8)
    line = np.rint(np.asarray(line_xy, dtype=np.float64)).astype(np.int32)
    cv2.line(seed, tuple(line[0]), tuple(line[1]), 1, thickness=1)
    radius = max(0, int(radius_px))
    if radius <= 0:
        return seed > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(seed, kernel, iterations=1) > 0


def _polygon_mask(shape_hw: Tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    height, width = [int(value) for value in shape_hw]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2:
        cv2.fillPoly(
            mask,
            [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)],
            1,
        )
    return mask > 0


def _mask_from_guidance(
    guidance: Optional[Mapping[str, Any]],
    key: str,
    shape_hw: Tuple[int, int],
) -> np.ndarray:
    if not isinstance(guidance, Mapping) or key not in guidance:
        return np.zeros(shape_hw, dtype=bool)
    value = np.asarray(guidance[key], dtype=bool)
    return value.copy() if value.shape == tuple(shape_hw) else np.zeros(
        shape_hw, dtype=bool
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)
    eroded = cv2.erode(
        binary,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return (binary > 0) & (eroded == 0)


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius == 0 or not mask.any():
        return mask.copy()
    return cv2.dilate(
        mask.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        ),
        iterations=1,
    ) > 0


def _edge_bin_coverage(
    mask: np.ndarray,
    line_xy: np.ndarray,
    *,
    bin_count: int = 40,
    patch_radius_px: int = 6,
    minimum_patch_fraction: float = 0.20,
) -> float:
    mask = np.asarray(mask, dtype=bool)
    line = np.asarray(line_xy, dtype=np.float64).reshape(2, 2)
    values = np.linspace(0.05, 0.95, max(2, int(bin_count)))
    points = line[0][None, :] + values[:, None] * (
        line[1] - line[0]
    )[None, :]
    covered = 0
    valid_samples = 0
    radius = max(1, int(patch_radius_px))
    for x_value, y_value in points:
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
            continue
        x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        valid_samples += 1
        patch = mask[y0:y1, x0:x1]
        if patch.size and float(patch.mean()) >= float(minimum_patch_fraction):
            covered += 1
    return float(covered / max(valid_samples, 1))


def _front_facing_distance(
    record: Mapping[str, Any], camera_xyz: np.ndarray
) -> float:
    normal = np.asarray(record.get("normal", np.zeros(3)), dtype=np.float64)
    center = np.asarray(record.get("center", np.zeros(3)), dtype=np.float64)
    camera = np.asarray(camera_xyz, dtype=np.float64)
    if normal.size < 2 or center.size < 2 or camera.size < 2:
        return float("-inf")
    normal_xy = normal[:2]
    norm = float(np.linalg.norm(normal_xy))
    if norm < 1.0e-9:
        return float("-inf")
    return float(np.dot(normal_xy / norm, camera[:2] - center[:2]))


def _project_line(
    line_xyz: np.ndarray,
    K: np.ndarray,
    R_wc: np.ndarray,
    camera_xyz: np.ndarray,
    model_to_selected_h: np.ndarray,
) -> Optional[np.ndarray]:
    raw, visible = project_points_world_to_image(
        np.asarray(line_xyz, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(R_wc, dtype=np.float64),
        np.asarray(camera_xyz, dtype=np.float64),
        clip_behind=True,
    )
    if len(raw) != 2 or int(np.asarray(visible).sum()) != 2:
        return None
    try:
        return apply_homography(
            raw, np.asarray(model_to_selected_h, dtype=np.float64)
        )
    except ValueError:
        return None


def build_adjacent_wall_contexts(
    *,
    group_records: Sequence[Mapping[str, Any]],
    loop_records: Sequence[Mapping[str, Any]],
    mesh_by_name: Mapping[str, Any],
    meshes_named: Sequence[Tuple[str, Any]],
    K: np.ndarray,
    R_wc: np.ndarray,
    camera_xyz: np.ndarray,
    raw_image_size_wh: Tuple[int, int],
    selected_image_size_wh: Tuple[int, int],
    model_to_selected_h: np.ndarray,
    raw_full_depth: Optional[np.ndarray] = None,
    side_band_px: int = 48,
    minimum_visible_fraction: float = 0.08,
) -> Dict[str, Dict[str, Any]]:
    """Classify the two neighboring 3D walls using facing and a z-buffer."""
    neighbors = cyclic_group_neighbors(group_records, loop_records)
    endpoint_lines_xyz = {
        "start": np.vstack([
            np.asarray(group_records[0]["wall_quad"])[0],
            np.asarray(group_records[0]["wall_quad"])[3],
        ]),
        "end": np.vstack([
            np.asarray(group_records[-1]["wall_quad"])[1],
            np.asarray(group_records[-1]["wall_quad"])[2],
        ]),
    }
    raw_width, raw_height = [int(value) for value in raw_image_size_wh]
    selected_width, selected_height = [
        int(value) for value in selected_image_size_wh
    ]
    raw_shape = (raw_height, raw_width)
    selected_shape = (selected_height, selected_width)
    full_depth = None
    if raw_full_depth is not None:
        candidate = np.asarray(raw_full_depth, dtype=np.float32)
        if candidate.shape == raw_shape:
            full_depth = candidate
    if full_depth is None:
        full_depth = render_model_depth_map(
            meshes_named,
            K,
            R_wc,
            camera_xyz,
            (raw_width, raw_height),
        )

    contexts = {}
    for endpoint, neighbor in neighbors.items():
        selected_line = _project_line(
            endpoint_lines_xyz[endpoint],
            K,
            R_wc,
            camera_xyz,
            model_to_selected_h,
        )
        row: Dict[str, Any] = {
            "group_endpoint": endpoint,
            "target_line_px": selected_line,
            "neighbor_global_index": (
                None if neighbor is None else int(neighbor["global_index"])
            ),
            "neighbor_mesh_name": (
                None if neighbor is None else neighbor.get("mesh_name")
            ),
            "adjacent_front_facing": False,
            "front_facing_distance_m": None,
            "adjacent_visible": False,
            "visible_pixel_count": 0,
            "visible_fraction": 0.0,
            "visible_edge_span_ratio": 0.0,
            "adjacent_projection_mask": np.zeros(selected_shape, dtype=bool),
            "adjacent_visible_mask": np.zeros(selected_shape, dtype=bool),
            "view_type": "no_adjacent" if neighbor is None else "adjacent_hidden",
        }
        if neighbor is None or selected_line is None:
            contexts[endpoint] = row
            continue
        signed_distance = _front_facing_distance(neighbor, camera_xyz)
        front_facing = bool(signed_distance > 0.05)
        row.update({
            "front_facing_distance_m": float(signed_distance),
            "adjacent_front_facing": front_facing,
        })
        mesh_name = neighbor.get("mesh_name")
        adjacent_mesh = mesh_by_name.get(str(mesh_name)) if mesh_name else None
        if adjacent_mesh is None:
            row["view_type"] = "unknown"
            contexts[endpoint] = row
            continue
        adjacent_depth = render_model_depth_map(
            [(str(mesh_name), adjacent_mesh)],
            K,
            R_wc,
            camera_xyz,
            (raw_width, raw_height),
        )
        adjacent_valid = np.isfinite(adjacent_depth) & (adjacent_depth > 0.0)
        full_valid = np.isfinite(full_depth) & (full_depth > 0.0)
        tolerance = np.maximum(0.05, np.abs(full_depth) * 0.01)
        visible_raw = (
            adjacent_valid
            & full_valid
            & (np.abs(adjacent_depth - full_depth) <= tolerance)
        )
        projection_selected = cv2.warpPerspective(
            adjacent_valid.astype(np.uint8),
            np.asarray(model_to_selected_h, dtype=np.float64),
            (selected_width, selected_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        visible_selected = cv2.warpPerspective(
            visible_raw.astype(np.uint8),
            np.asarray(model_to_selected_h, dtype=np.float64),
            (selected_width, selected_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        strip = _line_band(selected_shape, selected_line, side_band_px)
        projected_strip = projection_selected & strip
        visible_strip = visible_selected & strip
        visible_pixels = int(visible_strip.sum())
        visible_fraction = float(
            visible_pixels / max(int(projected_strip.sum()), 1)
        )
        span = _edge_bin_coverage(
            visible_strip,
            selected_line,
            patch_radius_px=6,
            minimum_patch_fraction=0.05,
        )
        is_visible = bool(
            front_facing
            and visible_pixels >= 24
            and visible_fraction >= float(minimum_visible_fraction)
            and span >= 0.20
        )
        row.update({
            "adjacent_visible": is_visible,
            "visible_pixel_count": visible_pixels,
            "visible_fraction": visible_fraction,
            "visible_edge_span_ratio": float(span),
            "adjacent_projection_mask": projection_selected,
            "adjacent_visible_mask": visible_selected,
            "view_type": "adjacent_visible" if is_visible else "adjacent_hidden",
        })
        contexts[endpoint] = row

    ordered = sorted(
        [
            item for item in contexts.items()
            if item[1]["target_line_px"] is not None
        ],
        key=lambda item: float(np.mean(item[1]["target_line_px"][:, 0])),
    )
    output = {}
    if len(ordered) >= 2:
        output["left"] = ordered[0][1]
        output["right"] = ordered[-1][1]
    elif len(ordered) == 1:
        # Do not silently call a lone visible endpoint "left".  Compare it to
        # the projected group centre; if that reference cannot be projected,
        # leave the side unclassified rather than constrain the wrong edge.
        group_points = np.vstack([
            np.asarray(record["wall_quad"], dtype=np.float64)
            for record in group_records
        ])
        center_xyz = np.mean(group_points, axis=0, keepdims=True)
        raw_center, center_visible = project_points_world_to_image(
            center_xyz, K, R_wc, camera_xyz, clip_behind=True
        )
        if len(raw_center) == 1 and bool(np.asarray(center_visible)[0]):
            selected_center = apply_homography(
                raw_center, np.asarray(model_to_selected_h, dtype=np.float64)
            )[0]
            line_x = float(np.mean(ordered[0][1]["target_line_px"][:, 0]))
            side = "left" if line_x < float(selected_center[0]) else "right"
            output[side] = ordered[0][1]
    return output


def analyze_source_side_evidence(
    *,
    target_outline_px: np.ndarray,
    semantic_guidance: Optional[Mapping[str, Any]],
    adjacent_contexts: Mapping[str, Mapping[str, Any]],
    image_shape_hw: Tuple[int, int],
    external_exclusion_mask: Optional[np.ndarray] = None,
    side_band_px: int = 48,
    foreground_occlusion_ratio: float = 0.50,
) -> Dict[str, Any]:
    """Prepare inside/outside searches while semantic masks are still uncut."""
    shape = tuple(int(value) for value in image_shape_hw)
    projection = _polygon_mask(shape, target_outline_px)
    building = _mask_from_guidance(
        semantic_guidance, "selected_building_mask", shape
    )
    if not building.any():
        building = _mask_from_guidance(
            semantic_guidance, "target_semantic_mask", shape
        )
    foreground = _mask_from_guidance(
        semantic_guidance, "foreground_mask", shape
    )
    background = _mask_from_guidance(
        semantic_guidance, "background_mask", shape
    )
    sky = _mask_from_guidance(semantic_guidance, "sky_mask", shape)
    hard_occluder = _mask_from_guidance(
        semantic_guidance, "hard_occluder_mask", shape
    )
    generic_non_target = _mask_from_guidance(
        semantic_guidance, "generic_non_target_mask", shape
    )
    external = np.zeros(shape, dtype=bool)
    if external_exclusion_mask is not None:
        candidate = np.asarray(external_exclusion_mask, dtype=bool)
        if candidate.shape == shape:
            external = candidate
    cleaned_building = (
        building
        & (~foreground)
        & (~hard_occluder)
        & (~generic_non_target)
        & (~external)
    )
    context = background | sky
    interface = _boundary(cleaned_building) & _dilate(context, 5)
    background_maps = (
        semantic_guidance.get("background_aware_boundary_maps", {})
        if isinstance(semantic_guidance, Mapping)
        else {}
    )
    if isinstance(background_maps, Mapping) and "wall" in background_maps:
        wall_interface = np.asarray(background_maps["wall"], dtype=bool)
        if wall_interface.shape == shape:
            interface |= wall_interface & (~foreground)

    sides = {}
    extension = np.zeros(shape, dtype=bool)
    for side in ("left", "right"):
        context_row = dict(adjacent_contexts.get(side) or {})
        line = context_row.get("target_line_px")
        if line is None:
            sides[side] = {
                "side": side,
                "decision": "no_safe_edge",
                "reason": "missing_projected_side_line",
                "search_enabled": False,
                "inside_search_mask": np.zeros(shape, dtype=bool),
                "outside_search_mask": np.zeros(shape, dtype=bool),
                "preferred_inside_mask": np.zeros(shape, dtype=bool),
                "preferred_outside_mask": np.zeros(shape, dtype=bool),
            }
            continue
        line = np.asarray(line, dtype=np.float64).reshape(2, 2)
        band = _line_band(shape, line, side_band_px)
        inside = band & projection
        outside = band & (~projection)
        extension_domain = outside.copy()
        occlusion_fraction = _edge_bin_coverage(foreground, line)
        occluded = bool(
            occlusion_fraction >= float(foreground_occlusion_ratio)
        )
        adjacent_visible = bool(context_row.get("adjacent_visible", False))
        if adjacent_visible:
            adjacent_projection = np.asarray(
                context_row.get(
                    "adjacent_projection_mask",
                    np.zeros(shape, dtype=bool),
                ),
                dtype=bool,
            )
            if adjacent_projection.shape != shape:
                adjacent_projection = np.zeros(shape, dtype=bool)
            adjacent_visible_mask = np.asarray(
                context_row.get(
                    "adjacent_visible_mask",
                    np.zeros(shape, dtype=bool),
                ),
                dtype=bool,
            )
            if adjacent_visible_mask.shape != shape:
                adjacent_visible_mask = np.zeros(shape, dtype=bool)
            # Do not open the complete projection of a wall that is visible in
            # only a small longitudinal span.  The small dilation retains the
            # actual wall-wall interface while excluding z-buffer-hidden areas.
            extension_domain &= adjacent_projection
            outside &= adjacent_projection & _dilate(
                adjacent_visible_mask, 6
            )
            outside_rgb_allowed = bool(outside.any())
            outside_mode = "zbuffer_visible_adjacent_wall_projection"
        else:
            outside_rgb_allowed = False
            outside_mode = "background_semantic_interface_only"
        preferred_inside = interface & inside
        preferred_outside = interface & outside
        if occluded:
            outside[:] = False
            extension_domain[:] = False
            preferred_inside[:] = False
            preferred_outside[:] = False
            decision = "keep_current_occlusion"
        else:
            # Search only where the neighbour is actually z-buffer visible,
            # but retain the semantic target pixels in the narrow projected
            # neighbour strip as candidates.  They are promoted later only
            # inside the corridor bounded by a validated outside edge.
            candidate_extension = cleaned_building & extension_domain
            extension |= candidate_extension
            decision = "search_inside_then_outside"
        if occluded:
            candidate_extension = np.zeros(shape, dtype=bool)
        sides[side] = {
            "side": side,
            "target_line_px": line,
            "view_type": context_row.get("view_type", "unknown"),
            "adjacent_global_index": context_row.get("neighbor_global_index"),
            "adjacent_visible": adjacent_visible,
            "adjacent_front_facing": bool(
                context_row.get("adjacent_front_facing", False)
            ),
            "foreground_occlusion_fraction": float(occlusion_fraction),
            "major_foreground_occlusion": occluded,
            "decision": decision,
            "search_enabled": not occluded,
            "outside_rgb_allowed": outside_rgb_allowed,
            "outside_mode": outside_mode,
            "inside_search_mask": inside,
            "outside_search_mask": outside,
            "preferred_inside_mask": preferred_inside,
            "preferred_outside_mask": preferred_outside,
            "candidate_extension_mask": candidate_extension,
            "extension_pixels": int(candidate_extension.sum()),
            "adjacent_visibility": {
                "front_facing_distance_m": context_row.get(
                    "front_facing_distance_m"
                ),
                "visible_pixel_count": int(
                    context_row.get("visible_pixel_count", 0)
                ),
                "visible_fraction": float(
                    context_row.get("visible_fraction", 0.0)
                ),
                "visible_edge_span_ratio": float(
                    context_row.get("visible_edge_span_ratio", 0.0)
                ),
            },
        }
    return {
        "enabled": True,
        "sides": sides,
        "content_extension_mask": extension,
        "content_extension_status": "candidate_until_outside_edge_is_validated",
        "semantic_interface_mask": interface,
        "foreground_mask": foreground,
        "projection_mask": projection,
    }


def warp_side_evidence_to_rectified(
    evidence: Mapping[str, Any],
    source_to_rectified_h: np.ndarray,
    output_shape_hw: Tuple[int, int],
) -> Dict[str, Any]:
    height, width = [int(value) for value in output_shape_hw]
    source_extension = np.asarray(
        evidence.get("content_extension_mask", []), dtype=np.uint8
    )
    if source_extension.ndim == 2:
        rectified_extension = cv2.warpPerspective(
            source_extension,
            np.asarray(source_to_rectified_h, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
    else:
        rectified_extension = np.zeros((height, width), dtype=bool)
    output = {
        "enabled": bool(evidence.get("enabled", False)),
        "reason": evidence.get("reason"),
        "content_extension_status": evidence.get(
            "content_extension_status"
        ),
        "content_extension_mask": rectified_extension,
        "sides": {},
    }
    for side, row_value in dict(evidence.get("sides") or {}).items():
        row = dict(row_value)
        transformed = {}
        for key in (
            "inside_search_mask",
            "outside_search_mask",
            "preferred_inside_mask",
            "preferred_outside_mask",
            "candidate_extension_mask",
        ):
            mask = np.asarray(
                row.pop(key, np.zeros((height, width), dtype=bool)),
                dtype=np.uint8,
            )
            transformed[key] = cv2.warpPerspective(
                mask,
                np.asarray(source_to_rectified_h, dtype=np.float64),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
        target_line = row.get("target_line_px")
        if target_line is not None:
            row["source_target_line_px"] = np.asarray(
                target_line, dtype=np.float64
            ).tolist()
            row["target_line_px"] = apply_homography(
                np.asarray(target_line, dtype=np.float64),
                source_to_rectified_h,
            )
        row.update(transformed)
        row["rectified_candidate_extension_pixels"] = int(
            transformed["candidate_extension_mask"].sum()
        )
        output["sides"][str(side)] = row
    return output


def side_evidence_metadata(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove raster masks before serializing the side decision."""
    output = {
        "enabled": bool(evidence.get("enabled", False)),
        "reason": evidence.get("reason"),
        "content_extension_status": evidence.get(
            "content_extension_status"
        ),
        "candidate_extension_pixels": int(
            np.asarray(
                evidence.get("content_extension_mask", []), dtype=bool
            ).sum()
        ),
        "sides": {},
    }
    for side, row_value in dict(evidence.get("sides") or {}).items():
        row = {}
        for key, value in dict(row_value).items():
            if key.endswith("_mask"):
                continue
            if isinstance(value, np.ndarray):
                row[key] = value.astype(float).tolist()
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                row[key] = value.item()
            else:
                row[key] = value
        output["sides"][str(side)] = row
    return output
