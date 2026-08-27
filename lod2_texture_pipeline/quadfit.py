# -*- coding: utf-8 -*-
"""Mask cleaning, quadrilateral fitting, Hough analysis, and affine wall fitting."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import *

def _rgba_image_to_bgr_on_background(img_pil: Image.Image, bg=(246, 246, 244)) -> np.ndarray:
    rgba = np.asarray(img_pil.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    bg_rgb = np.empty(rgba.shape[:2] + (3,), dtype=np.float32)
    bg_rgb[:, :, 0] = float(bg[0])
    bg_rgb[:, :, 1] = float(bg[1])
    bg_rgb[:, :, 2] = float(bg[2])
    rgb = rgba[:, :, :3] * alpha + bg_rgb * (1.0 - alpha)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def _binary_mask_stats(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return {
        "area": int(len(xs)),
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "xmin": int(xs.min()),
        "xmax": int(xs.max()),
        "ymin": int(ys.min()),
        "ymax": int(ys.max()),
    }

def np_bool_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)

def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            out[labels == lab] = 1
    return out.astype(bool)

def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    m = np_bool_to_u8(mask)
    h, w = m.shape
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, holes)
    return filled > 0

def clean_selected_mask(mask: np.ndarray) -> np.ndarray:
    m = np_bool_to_u8(mask)

    if QUAD_MORPH_CLOSE_PX > 0:
        k = 2 * QUAD_MORPH_CLOSE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    if QUAD_MORPH_OPEN_PX > 0:
        k = 2 * QUAD_MORPH_OPEN_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    m = remove_small_components(m > 0, QUAD_MIN_COMPONENT_AREA_PX)

    if QUAD_FILL_HOLES:
        m = fill_mask_holes(m)

    return m

def order_box_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    pts = pts[idx]

    # rotate so the top-left-ish point comes first
    sums = pts[:, 0] + pts[:, 1]
    start = int(np.argmin(sums))
    pts = np.roll(pts, -start, axis=0)
    return pts

def _fit_quad_from_points(pts: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pts = np.asarray(pts, dtype=np.float32)

    if pts.ndim == 3:
        pts = pts[:, 0, :]

    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
        return None, None

    contour = pts.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(contour)

    quad = None
    peri = cv2.arcLength(hull, True)

    for frac in np.linspace(0.005, 0.08, 60):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            quad = approx[:, 0, :].astype(np.float32)
            break

    if quad is None:
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype(np.float32)

    quad = order_box_points_clockwise(quad)
    return quad, hull

def fit_quadrilateral_from_mask(
    mask: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]], List[np.ndarray]]:
    """
    Returns:
        final_quad, final_hull, contours, chunk_quads

    Logic:
      1) find all valid disconnected chunks
      2) fit one quad per chunk
      3) fit the final/main quad from all chunk-quad corners together
    """
    m = np_bool_to_u8(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, []

    contours = [c for c in contours if cv2.contourArea(c) >= QUAD_MIN_CONTOUR_AREA_PX]
    if not contours:
        return None, None, None, []

    chunk_quads: List[np.ndarray] = []
    for contour in contours:
        chunk_quad, _ = _fit_quad_from_points(contour)
        if chunk_quad is not None:
            chunk_quads.append(chunk_quad)

    if not chunk_quads:
        return None, None, contours, []

    if len(chunk_quads) == 1:
        final_quad, final_hull = _fit_quad_from_points(contours[0])
        return final_quad, final_hull, contours, chunk_quads

    all_chunk_quad_pts = np.vstack(chunk_quads).astype(np.float32)
    final_quad, final_hull = _fit_quad_from_points(all_chunk_quad_pts)

    return final_quad, final_hull, contours, chunk_quads

def build_edge_map_for_hough(rgb_img: np.ndarray, alpha_mask_bool: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    if HOUGH_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    gray[~alpha_mask_bool] = 0

    edges = cv2.Canny(gray, HOUGH_CANNY_LOW, HOUGH_CANNY_HIGH)

    if HOUGH_CANNY_DILATE_PX > 0:
        k = 2 * HOUGH_CANNY_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def build_alpha_boundary_edge_map_for_hough(alpha_mask_bool: np.ndarray) -> np.ndarray:
    """
    Build Hough input from the segmentation silhouette, not interior image
    texture. This is the correct signal for aligning a segmented facade outline
    to the projected facade polygon.
    """
    alpha = (np.asarray(alpha_mask_bool).astype(np.uint8) * 255)
    if alpha.ndim != 2 or int((alpha > 0).sum()) == 0:
        return np.zeros_like(alpha, dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)

    if HOUGH_CANNY_DILATE_PX > 0:
        k = 2 * HOUGH_CANNY_DILATE_PX + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        edge = cv2.dilate(edge, dilate_kernel, iterations=1)

    return edge

def angle_deg_of_segment(p0: np.ndarray, p1: np.ndarray) -> float:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    ang = math.degrees(math.atan2(dy, dx))
    ang = ang % 180.0
    return ang

def angle_diff_deg_180(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def point_line_distance(px: float, py: float, p0: np.ndarray, p1: np.ndarray) -> float:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    num = abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0)
    den = math.hypot(y1 - y0, x1 - x0)
    return num / max(den, 1e-9)

def build_line_search_band(
    height: int,
    width: int,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    wall_mask_bool: np.ndarray,
    band_px: int
) -> np.ndarray:
    band = np.zeros((height, width), dtype=np.uint8)
    p0 = tuple(np.round(line_p0).astype(np.int32))
    p1 = tuple(np.round(line_p1).astype(np.int32))
    thickness = max(3, int(2 * band_px))
    cv2.line(band, p0, p1, 255, thickness=thickness)

    wall_u8 = (wall_mask_bool.astype(np.uint8) * 255)
    k = max(3, int(2 * band_px + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    wall_expand = cv2.dilate(wall_u8, kernel, iterations=1) > 0

    band = (band > 0) & wall_expand
    return band.astype(np.uint8)

def detect_hough_segments(edge_map_u8: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
    work = edge_map_u8.copy()
    if roi_mask is not None:
        work = work.copy()
        work[roi_mask == 0] = 0

    lines = cv2.HoughLinesP(
        work,
        rho=1,
        theta=np.pi / 180.0,
        threshold=50,
        minLineLength=HOUGH_MIN_LENGTH_PX,
        maxLineGap=HOUGH_MAX_GAP_PX
    )

    if lines is None:
        return []

    out = []
    lines = np.asarray(lines).reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        out.append(np.array([[x1, y1], [x2, y2]], dtype=np.float64))
    return out

def line_overlap_with_edge_map(seg_line: np.ndarray, edge_map_u8: np.ndarray, thickness: int = 3) -> Tuple[int, int]:
    h, w = edge_map_u8.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)

    p0 = tuple(np.round(seg_line[0]).astype(np.int32))
    p1 = tuple(np.round(seg_line[1]).astype(np.int32))
    cv2.line(line_mask, p0, p1, 255, thickness=thickness)

    line_pixels = int((line_mask > 0).sum())
    overlap = int(((line_mask > 0) & (edge_map_u8 > 0)).sum())
    return overlap, line_pixels

def fit_dominant_line_from_segments(
    kept_segments: List[np.ndarray],
    target_p0: np.ndarray,
    target_p1: np.ndarray,
    edge_map_u8: np.ndarray,
    *,
    extend_px: float = 20.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    info = {
        "num_candidates": int(len(kept_segments)),
        "best_length_px": None,
        "best_angle_diff_deg": None,
        "best_distance_px": None,
        "best_overlap_ratio": None,
    }

    if len(kept_segments) == 0:
        return None, info

    pts = []
    for seg in kept_segments:
        pts.append(seg[0])
        pts.append(seg[1])
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    v = np.array([float(vx), float(vy)], dtype=np.float64)
    v /= max(np.linalg.norm(v), 1e-9)
    p = np.array([float(x0), float(y0)], dtype=np.float64)

    target_dir = np.asarray(target_p1, dtype=np.float64) - np.asarray(target_p0, dtype=np.float64)
    target_dir /= max(np.linalg.norm(target_dir), 1e-9)
    if np.dot(v, target_dir) < 0:
        v = -v

    scalars = []
    for seg in kept_segments:
        for q in seg:
            scalars.append(np.dot(np.asarray(q, dtype=np.float64) - p, v))
    scalars = np.asarray(scalars, dtype=np.float64)

    tmin = float(np.min(scalars))
    tmax = float(np.max(scalars))

    extension = max(0.0, float(extend_px))
    p_start = p + (tmin - extension) * v
    p_end   = p + (tmax + extension) * v

    selected_line = np.vstack([p_start, p_end]).astype(np.float64)

    fitted_length = float(np.linalg.norm(selected_line[1] - selected_line[0]))
    fitted_ang = angle_deg_of_segment(selected_line[0], selected_line[1])
    target_ang = angle_deg_of_segment(target_p0, target_p1)
    fitted_ang_diff = angle_diff_deg_180(fitted_ang, target_ang)

    fitted_mid = 0.5 * (selected_line[0] + selected_line[1])
    fitted_dist = point_line_distance(fitted_mid[0], fitted_mid[1], target_p0, target_p1)

    overlap_px, line_px = line_overlap_with_edge_map(selected_line, edge_map_u8, thickness=3)
    fitted_overlap = float(overlap_px / max(line_px, 1))

    info["best_length_px"] = fitted_length
    info["best_angle_diff_deg"] = float(fitted_ang_diff)
    info["best_distance_px"] = float(fitted_dist)
    info["best_overlap_ratio"] = float(fitted_overlap)

    return selected_line, info


def _sampled_line_support_ratio(
    line_xy: np.ndarray,
    mask_u8: np.ndarray,
    *,
    sample_step_px: float = 2.0,
    dilation_px: int = 2,
) -> float:
    """Return longitudinal support, without rewarding a thick line raster."""
    line = np.asarray(line_xy, dtype=np.float64).reshape(2, 2)
    mask = np.asarray(mask_u8) > 0
    if mask.ndim != 2 or not mask.any():
        return 0.0
    radius = max(0, int(dilation_px))
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    delta = line[1] - line[0]
    length = float(np.linalg.norm(delta))
    if length < 1.0e-9:
        return 0.0
    count = max(2, int(math.ceil(length / max(float(sample_step_px), 0.5))) + 1)
    t = np.linspace(0.0, 1.0, count)
    points = line[0][None, :] + t[:, None] * delta[None, :]
    xs = np.rint(points[:, 0]).astype(np.int64)
    ys = np.rint(points[:, 1]).astype(np.int64)
    valid = (
        (xs >= 0)
        & (xs < mask.shape[1])
        & (ys >= 0)
        & (ys < mask.shape[0])
    )
    if not valid.any():
        return 0.0
    supported = np.zeros(count, dtype=bool)
    supported[valid] = mask[ys[valid], xs[valid]]
    return float(supported.sum() / count)


def _line_band_occupancy_ratio(
    line_xy: np.ndarray,
    search_band_u8: np.ndarray,
    *,
    sample_step_px: float = 3.0,
) -> float:
    return _sampled_line_support_ratio(
        line_xy,
        search_band_u8,
        sample_step_px=sample_step_px,
        dilation_px=0,
    )


def _merged_interval_coverage(
    intervals: List[Tuple[float, float]],
    lower: float,
    upper: float,
    *,
    merge_gap_px: float,
) -> Tuple[float, float]:
    """Return union and longest-run coverage inside a finite target segment."""
    clipped = []
    for start, end in intervals:
        lo = max(float(lower), min(float(start), float(end)))
        hi = min(float(upper), max(float(start), float(end)))
        if hi > lo:
            clipped.append((lo, hi))
    if not clipped:
        return 0.0, 0.0
    clipped.sort()
    merged = []
    cur_lo, cur_hi = clipped[0]
    for lo, hi in clipped[1:]:
        if lo <= cur_hi + max(0.0, float(merge_gap_px)):
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    union = float(sum(hi - lo for lo, hi in merged))
    longest = float(max(hi - lo for lo, hi in merged))
    target_length = max(float(upper - lower), 1.0e-9)
    return union / target_length, longest / target_length


def _cluster_segments_by_target_offset(
    segments: List[np.ndarray],
    target_p0: np.ndarray,
    target_p1: np.ndarray,
    tolerance_px: float,
) -> List[List[np.ndarray]]:
    if not segments:
        return []
    target = np.asarray(target_p1, dtype=np.float64) - np.asarray(
        target_p0, dtype=np.float64
    )
    target /= max(float(np.linalg.norm(target)), 1.0e-9)
    normal = np.array([-target[1], target[0]], dtype=np.float64)
    rows = []
    for segment in segments:
        midpoint = np.mean(np.asarray(segment, dtype=np.float64), axis=0)
        offset = float(np.dot(midpoint - target_p0, normal))
        rows.append((offset, segment))
    rows.sort(key=lambda item: item[0])
    clusters: List[List[np.ndarray]] = []
    cluster_offsets: List[List[float]] = []
    for offset, segment in rows:
        if (
            not clusters
            or abs(offset - float(np.median(cluster_offsets[-1])))
            > max(1.0, float(tolerance_px))
        ):
            clusters.append([segment])
            cluster_offsets.append([offset])
        else:
            clusters[-1].append(segment)
            cluster_offsets[-1].append(offset)
    return clusters

def select_best_hough_line_for_target(
    lines: List[np.ndarray],
    target_p0: np.ndarray,
    target_p1: np.ndarray,
    search_band_u8: np.ndarray,
    edge_map_u8: np.ndarray,
    min_length_px: float,
    angle_thresh_deg: float,
    *,
    minimum_target_coverage_ratio: Optional[float] = None,
    maximum_length_ratio: Optional[float] = None,
    maximum_distance_px: Optional[float] = None,
    minimum_band_occupancy_ratio: Optional[float] = None,
    minimum_edge_support_ratio: Optional[float] = None,
    offset_cluster_px: Optional[float] = None,
    preferred_edge_map_u8: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    target_ang = angle_deg_of_segment(target_p0, target_p1)
    H, W = search_band_u8.shape[:2]

    kept_segments = []
    rejection_counts = {
        "too_short_absolute": 0,
        "angle_mismatch": 0,
        "outside_search_band": 0,
        "too_far_from_target": 0,
    }

    for seg in lines:
        p0 = seg[0]
        p1 = seg[1]

        length = float(np.linalg.norm(p1 - p0))
        if length < min_length_px:
            rejection_counts["too_short_absolute"] += 1
            continue

        ang = angle_deg_of_segment(p0, p1)
        ang_diff = angle_diff_deg_180(ang, target_ang)
        if ang_diff > angle_thresh_deg:
            rejection_counts["angle_mismatch"] += 1
            continue

        band_occupancy = _line_band_occupancy_ratio(seg, search_band_u8)
        required_band_occupancy = (
            0.0
            if minimum_band_occupancy_ratio is None
            else float(minimum_band_occupancy_ratio)
        )
        if band_occupancy <= 0.0 or band_occupancy < required_band_occupancy:
            rejection_counts["outside_search_band"] += 1
            continue

        if maximum_distance_px is not None:
            distances = [
                point_line_distance(
                    float(point[0]), float(point[1]), target_p0, target_p1
                )
                for point in (p0, p1, 0.5 * (p0 + p1))
            ]
            if float(np.percentile(distances, 90)) > float(maximum_distance_px):
                rejection_counts["too_far_from_target"] += 1
                continue

        kept_segments.append(seg)

    strict_validation = any(
        value is not None
        for value in (
            minimum_target_coverage_ratio,
            maximum_length_ratio,
            maximum_distance_px,
            minimum_band_occupancy_ratio,
            minimum_edge_support_ratio,
            offset_cluster_px,
            preferred_edge_map_u8,
        )
    )
    if not strict_validation:
        line, info = fit_dominant_line_from_segments(
            kept_segments=kept_segments,
            target_p0=target_p0,
            target_p1=target_p1,
            edge_map_u8=edge_map_u8,
        )
        info["rejection_counts"] = rejection_counts
        return line, info

    target_p0 = np.asarray(target_p0, dtype=np.float64)
    target_p1 = np.asarray(target_p1, dtype=np.float64)
    target_vector = target_p1 - target_p0
    target_length = float(np.linalg.norm(target_vector))
    target_direction = target_vector / max(target_length, 1.0e-9)
    clusters = _cluster_segments_by_target_offset(
        kept_segments,
        target_p0,
        target_p1,
        12.0 if offset_cluster_px is None else float(offset_cluster_px),
    )
    cluster_records = []
    for cluster_index, cluster in enumerate(clusters):
        selected_line, fit_info = fit_dominant_line_from_segments(
            kept_segments=cluster,
            target_p0=target_p0,
            target_p1=target_p1,
            edge_map_u8=edge_map_u8,
            extend_px=0.0,
        )
        if selected_line is None:
            continue
        intervals = []
        for segment in cluster:
            along = (
                np.asarray(segment, dtype=np.float64) - target_p0[None, :]
            ) @ target_direction
            intervals.append((float(np.min(along)), float(np.max(along))))
        union_coverage, continuous_coverage = _merged_interval_coverage(
            intervals,
            0.0,
            target_length,
            merge_gap_px=max(float(HOUGH_MAX_GAP_PX) * 2.0, 8.0),
        )
        fitted_length = float(np.linalg.norm(selected_line[1] - selected_line[0]))
        length_ratio = fitted_length / max(target_length, 1.0e-9)
        fitted_angle = angle_deg_of_segment(selected_line[0], selected_line[1])
        angle_difference = angle_diff_deg_180(fitted_angle, target_ang)
        distances = np.asarray(
            [
                point_line_distance(
                    float(point[0]), float(point[1]), target_p0, target_p1
                )
                for point in (
                    selected_line[0],
                    selected_line[1],
                    0.5 * (selected_line[0] + selected_line[1]),
                )
            ],
            dtype=np.float64,
        )
        p90_distance = float(np.percentile(distances, 90))
        band_occupancy = _line_band_occupancy_ratio(
            selected_line, search_band_u8
        )
        edge_support = _sampled_line_support_ratio(
            selected_line, edge_map_u8
        )
        preferred_support = (
            0.0
            if preferred_edge_map_u8 is None
            else _sampled_line_support_ratio(
                selected_line, preferred_edge_map_u8
            )
        )
        reasons = []
        minimum_coverage = float(minimum_target_coverage_ratio or 0.0)
        if union_coverage < minimum_coverage:
            reasons.append("insufficient_target_length_coverage")
        if continuous_coverage < minimum_coverage:
            reasons.append("insufficient_continuous_target_support")
        if length_ratio < minimum_coverage:
            reasons.append("fitted_line_too_short_relative_to_target")
        if maximum_length_ratio is not None and length_ratio > float(maximum_length_ratio):
            reasons.append("fitted_line_too_long_relative_to_target")
        if angle_difference > float(angle_thresh_deg):
            reasons.append("fitted_angle_mismatch")
        if maximum_distance_px is not None and p90_distance > float(maximum_distance_px):
            reasons.append("fitted_line_too_far_from_target")
        if (
            minimum_band_occupancy_ratio is not None
            and band_occupancy < float(minimum_band_occupancy_ratio)
        ):
            reasons.append("insufficient_search_band_occupancy")
        if (
            minimum_edge_support_ratio is not None
            and edge_support < float(minimum_edge_support_ratio)
        ):
            reasons.append("insufficient_longitudinal_edge_support")
        distance_score = 1.0 - min(
            p90_distance / max(float(maximum_distance_px or 1.0), 1.0), 1.0
        )
        angle_score = 1.0 - min(
            angle_difference / max(float(angle_thresh_deg), 1.0), 1.0
        )
        score = float(
            3.0 * min(union_coverage, 1.0)
            + 2.0 * min(continuous_coverage, 1.0)
            + 1.5 * edge_support
            + 1.5 * preferred_support
            + distance_score
            + angle_score
        )
        cluster_records.append({
            "cluster_index": int(cluster_index),
            "segment_count": int(len(cluster)),
            "selected_line": selected_line,
            "fit_info": fit_info,
            "target_union_coverage_ratio": float(union_coverage),
            "target_continuous_coverage_ratio": float(continuous_coverage),
            "length_ratio_to_target": float(length_ratio),
            "angle_diff_deg": float(angle_difference),
            "p90_distance_px": p90_distance,
            "band_occupancy_ratio": float(band_occupancy),
            "edge_support_ratio": float(edge_support),
            "preferred_edge_support_ratio": float(preferred_support),
            "score": score,
            "rejection_reasons": reasons,
        })

    accepted = [row for row in cluster_records if not row["rejection_reasons"]]
    best = max(accepted, key=lambda row: row["score"]) if accepted else None
    serializable_clusters = [
        {
            key: value
            for key, value in row.items()
            if key not in {"selected_line", "fit_info"}
        }
        for row in cluster_records
    ]
    info = {
        "num_candidates": int(len(kept_segments)),
        "num_offset_clusters": int(len(cluster_records)),
        "rejection_counts": rejection_counts,
        "cluster_diagnostics": serializable_clusters,
        "validation_enabled": True,
        "best_length_px": None,
        "best_angle_diff_deg": None,
        "best_distance_px": None,
        "best_overlap_ratio": None,
        "target_length_px": float(target_length),
    }
    if best is None:
        info["rejection_reason"] = (
            "no_cluster_passed_relative_length_distance_and_support_gates"
            if cluster_records
            else "no_candidate_segment_passed_basic_gates"
        )
        return None, info

    info.update({
        "selected_cluster_index": int(best["cluster_index"]),
        "selected_segment_count": int(best["segment_count"]),
        "best_length_px": float(
            np.linalg.norm(best["selected_line"][1] - best["selected_line"][0])
        ),
        "best_angle_diff_deg": float(best["angle_diff_deg"]),
        "best_distance_px": float(best["p90_distance_px"]),
        "best_overlap_ratio": float(best["edge_support_ratio"]),
        "target_union_coverage_ratio": float(best["target_union_coverage_ratio"]),
        "target_continuous_coverage_ratio": float(
            best["target_continuous_coverage_ratio"]
        ),
        "length_ratio_to_target": float(best["length_ratio_to_target"]),
        "band_occupancy_ratio": float(best["band_occupancy_ratio"]),
        "preferred_edge_support_ratio": float(
            best["preferred_edge_support_ratio"]
        ),
        "selection_score": float(best["score"]),
        "rejection_reason": None,
    })
    return np.asarray(best["selected_line"], dtype=np.float64), info

def save_hough_all_lines_overlay(
    img_pil: Image.Image,
    wall_quad_xy: np.ndarray,
    all_lines: List[np.ndarray],
    selected_left: Optional[np.ndarray],
    selected_right: Optional[np.ndarray],
    selected_top: Optional[np.ndarray],
    out_path: str,
    selected_edges: Optional[List[Dict[str, Any]]] = None,
) -> None:
    out_bgr = _rgba_image_to_bgr_on_background(img_pil)

    if all_lines is not None:
        for seg in all_lines:
            p0 = tuple(np.round(seg[0]).astype(np.int32))
            p1 = tuple(np.round(seg[1]).astype(np.int32))
            cv2.line(out_bgr, p0, p1, (0, 220, 220), 1)

    if wall_quad_xy is not None:
        q_i = np.round(np.asarray(wall_quad_xy, dtype=np.float64)).astype(np.int32)
        n = len(q_i)
        for i in range(n):
            p1 = tuple(q_i[i])
            p2 = tuple(q_i[(i + 1) % n])
            cv2.line(out_bgr, p1, p2, (0, 0, 255), 2)

    def draw_selected(seg: Optional[np.ndarray], color_bgr: Tuple[int, int, int], label: str):
        if seg is None:
            return
        p0 = tuple(np.round(seg[0]).astype(np.int32))
        p1 = tuple(np.round(seg[1]).astype(np.int32))
        cv2.line(out_bgr, p0, p1, color_bgr, 4)
        cv2.circle(out_bgr, p0, 4, color_bgr, -1)
        cv2.circle(out_bgr, p1, 4, color_bgr, -1)
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        cv2.putText(out_bgr, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if selected_edges:
        palette = [
            (0, 255, 0),
            (255, 0, 255),
            (0, 165, 255),
            (255, 180, 0),
            (40, 220, 255),
            (180, 80, 255),
        ]
        for k, edge in enumerate(selected_edges):
            seg = edge.get("selected_line")
            if seg is None:
                continue
            draw_selected(
                np.asarray(seg, dtype=np.float64),
                palette[k % len(palette)],
                f"E{int(edge.get('edge_index', k))}"
            )
    else:
        draw_selected(selected_left,  (0, 255, 0),   "SEL_LEFT")
        draw_selected(selected_right, (255, 0, 255), "SEL_RIGHT")
        draw_selected(selected_top,   (0, 165, 255), "SEL_TOP")

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_rgb).save(out_path)

def x_at_y_on_line(line_xy: np.ndarray, y: float) -> float:
    """
    Infinite line x(y) from 2 points.
    """
    p0 = np.asarray(line_xy[0], dtype=np.float64)
    p1 = np.asarray(line_xy[1], dtype=np.float64)

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    if abs(y1 - y0) < 1e-9:
        return 0.5 * (x0 + x1)

    t = (y - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)

def y_at_x_on_line(line_xy: np.ndarray, x: float) -> float:
    """
    Infinite line y(x) from 2 points.
    """
    p0 = np.asarray(line_xy[0], dtype=np.float64)
    p1 = np.asarray(line_xy[1], dtype=np.float64)

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    if abs(x1 - x0) < 1e-9:
        return 0.5 * (y0 + y1)

    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def inverse_piecewise_horizontal_map(
    xd: float,
    y: float,
    src_left_line: np.ndarray,
    src_right_line: np.ndarray,
    dst_left_line: np.ndarray,
    dst_right_line: np.ndarray,
    width: int
) -> float:
    """
    Inverse horizontal mapping:
      destination x -> source x
    so that src_left/right lines move to dst_left/right lines.
    """
    xs_l = x_at_y_on_line(src_left_line, y)
    xs_r = x_at_y_on_line(src_right_line, y)
    xd_l = x_at_y_on_line(dst_left_line, y)
    xd_r = x_at_y_on_line(dst_right_line, y)

    # enforce ordering
    if xs_l > xs_r:
        xs_l, xs_r = xs_r, xs_l
    if xd_l > xd_r:
        xd_l, xd_r = xd_r, xd_l

    eps = 1e-6
    Wm1 = float(width - 1)

    # left side
    if xd <= xd_l:
        denom = max(xd_l, eps)
        return xd * (xs_l / denom)

    # middle strip
    if xd <= xd_r:
        denom = max(xd_r - xd_l, eps)
        t = (xd - xd_l) / denom
        return xs_l + t * (xs_r - xs_l)

    # right side
    denom = max(Wm1 - xd_r, eps)
    t = (xd - xd_r) / denom
    return xs_r + t * (Wm1 - xs_r)

def forward_piecewise_horizontal_map(
    xs: float,
    y: float,
    src_left_line: np.ndarray,
    src_right_line: np.ndarray,
    dst_left_line: np.ndarray,
    dst_right_line: np.ndarray,
    width: int
) -> float:
    """
    Forward horizontal mapping:
      source x -> destination x
    Used to map the selected top line into the horizontally-warped frame.
    """
    xs_l = x_at_y_on_line(src_left_line, y)
    xs_r = x_at_y_on_line(src_right_line, y)
    xd_l = x_at_y_on_line(dst_left_line, y)
    xd_r = x_at_y_on_line(dst_right_line, y)

    if xs_l > xs_r:
        xs_l, xs_r = xs_r, xs_l
    if xd_l > xd_r:
        xd_l, xd_r = xd_r, xd_l

    eps = 1e-6
    Wm1 = float(width - 1)

    # left side
    if xs <= xs_l:
        denom = max(xs_l, eps)
        return xs * (xd_l / denom)

    # middle strip
    if xs <= xs_r:
        denom = max(xs_r - xs_l, eps)
        t = (xs - xs_l) / denom
        return xd_l + t * (xd_r - xd_l)

    # right side
    denom = max(Wm1 - xs_r, eps)
    t = (xs - xs_r) / denom
    return xd_r + t * (Wm1 - xd_r)

def inverse_piecewise_vertical_map(
    yd: float,
    x: float,
    src_top_line: np.ndarray,
    dst_top_line: np.ndarray,
    height: int
) -> float:
    """
    Inverse vertical mapping:
      destination y -> source y
    so that src_top line moves to dst_top line.
    Bottom stays anchored.
    """
    ys_t = y_at_x_on_line(src_top_line, x)
    yd_t = y_at_x_on_line(dst_top_line, x)

    eps = 1e-6
    Hm1 = float(height - 1)

    # above top line
    if yd <= yd_t:
        denom = max(yd_t, eps)
        return yd * (ys_t / denom)

    # below top line
    denom = max(Hm1 - yd_t, eps)
    t = (yd - yd_t) / denom
    return ys_t + t * (Hm1 - ys_t)

def warp_line_horizontally(
    line_xy: np.ndarray,
    src_left_line: np.ndarray,
    src_right_line: np.ndarray,
    dst_left_line: np.ndarray,
    dst_right_line: np.ndarray,
    width: int
) -> np.ndarray:
    """
    Apply only the horizontal piecewise mapping to a line's endpoints.
    """
    p0 = np.asarray(line_xy[0], dtype=np.float64)
    p1 = np.asarray(line_xy[1], dtype=np.float64)

    q0x = forward_piecewise_horizontal_map(
        p0[0], p0[1],
        src_left_line, src_right_line,
        dst_left_line, dst_right_line,
        width
    )
    q1x = forward_piecewise_horizontal_map(
        p1[0], p1[1],
        src_left_line, src_right_line,
        dst_left_line, dst_right_line,
        width
    )

    q0 = np.array([q0x, p0[1]], dtype=np.float64)
    q1 = np.array([q1x, p1[1]], dtype=np.float64)
    return np.vstack([q0, q1])

def apply_hough_guided_ortho_warp(
    ortho_rgba: np.ndarray,
    sel_left_line: Optional[np.ndarray],
    sel_right_line: Optional[np.ndarray],
    sel_top_line: Optional[np.ndarray],
    proj_left_line: np.ndarray,
    proj_right_line: np.ndarray,
    proj_top_line: Optional[np.ndarray],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Two-stage inverse remap in ortho space:
      1) horizontal warp to align detected side lines to projected lines
      2) vertical warp to align top selected line to projected top line

    A missing left or right detection becomes an identity anchor at that
    projected side. This lets one reliable side correct the texture while the
    opposite side stays fixed.
    """
    H, W = ortho_rgba.shape[:2]
    eps = 1e-6

    if sel_left_line is None and sel_right_line is None:
        return ortho_rgba.copy()
    if sel_left_line is None:
        sel_left_line = np.asarray(proj_left_line, dtype=np.float64)
    if sel_right_line is None:
        sel_right_line = np.asarray(proj_right_line, dtype=np.float64)

    def _x_at_y_vec(line_xy: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        p0 = np.asarray(line_xy[0], dtype=np.float64)
        p1 = np.asarray(line_xy[1], dtype=np.float64)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        if abs(y1 - y0) < 1e-9:
            return np.full_like(y_vals, 0.5 * (x0 + x1), dtype=np.float64)
        t = (y_vals - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    def _y_at_x_vec(line_xy: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        p0 = np.asarray(line_xy[0], dtype=np.float64)
        p1 = np.asarray(line_xy[1], dtype=np.float64)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        if abs(x1 - x0) < 1e-9:
            return np.full_like(x_vals, 0.5 * (y0 + y1), dtype=np.float64)
        t = (x_vals - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # ---------- stage 1: horizontal remap ----------
    y_vals = np.arange(H, dtype=np.float64)
    x_vals = np.arange(W, dtype=np.float64)
    xd = x_vals[None, :]

    xs_l = _x_at_y_vec(sel_left_line, y_vals)[:, None]
    xs_r = _x_at_y_vec(sel_right_line, y_vals)[:, None]
    xd_l = _x_at_y_vec(proj_left_line, y_vals)[:, None]
    xd_r = _x_at_y_vec(proj_right_line, y_vals)[:, None]

    src_swap = xs_l > xs_r
    dst_swap = xd_l > xd_r
    xs_l_o = np.where(src_swap, xs_r, xs_l)
    xs_r_o = np.where(src_swap, xs_l, xs_r)
    xd_l_o = np.where(dst_swap, xd_r, xd_l)
    xd_r_o = np.where(dst_swap, xd_l, xd_r)

    left_src = xd * (xs_l_o / np.maximum(xd_l_o, eps))
    middle_t = (xd - xd_l_o) / np.maximum(xd_r_o - xd_l_o, eps)
    middle_src = xs_l_o + middle_t * (xs_r_o - xs_l_o)
    right_t = (xd - xd_r_o) / np.maximum(float(W - 1) - xd_r_o, eps)
    right_src = xs_r_o + right_t * (float(W - 1) - xs_r_o)
    map_x_h = np.where(
        xd <= xd_l_o,
        left_src,
        np.where(xd <= xd_r_o, middle_src, right_src),
    )
    map_x_h = np.clip(map_x_h, 0.0, float(W - 1)).astype(np.float32)
    map_y_h = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)

    ortho_rgba_h = cv2.remap(
        ortho_rgba,
        map_x_h,
        map_y_h,
        interpolation=int(interpolation),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if sel_top_line is None or proj_top_line is None:
        return ortho_rgba_h

    # map selected top line into the horizontally-warped frame
    sel_top_after_h = warp_line_horizontally(
        line_xy=sel_top_line,
        src_left_line=sel_left_line,
        src_right_line=sel_right_line,
        dst_left_line=proj_left_line,
        dst_right_line=proj_right_line,
        width=W
    )

    # ---------- stage 2: vertical remap ----------
    yd = y_vals[:, None]
    ys_t = _y_at_x_vec(sel_top_after_h, x_vals)[None, :]
    yd_t = _y_at_x_vec(proj_top_line, x_vals)[None, :]

    above_src = yd * (ys_t / np.maximum(yd_t, eps))
    below_t = (yd - yd_t) / np.maximum(float(H - 1) - yd_t, eps)
    below_src = ys_t + below_t * (float(H - 1) - ys_t)
    map_y_v = np.where(yd <= yd_t, above_src, below_src)
    map_y_v = np.clip(map_y_v, 0.0, float(H - 1)).astype(np.float32)
    map_x_v = np.repeat(np.arange(W, dtype=np.float32)[None, :], H, axis=0)

    ortho_rgba_hv = cv2.remap(
        ortho_rgba_h,
        map_x_v,
        map_y_v,
        interpolation=int(interpolation),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return ortho_rgba_hv

def apply_hough_guided_polygon_affine_warp(
    ortho_rgba: np.ndarray,
    selected_edges: List[Dict[str, Any]]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Estimate one affine correction from many matched facade-outline edges.

    Each selected edge supplies an observed source line from Hough and the
    target projected facade edge. Points sampled on the source line are matched
    to the corresponding target edge positions, then one global affine warp is
    applied to the RGBA texture.
    """
    info = {
        "applied": False,
        "method": "affine_from_selected_polygon_edge_lines",
        "num_edge_matches": 0,
        "num_point_pairs": 0,
        "reason": None,
        "matrix_2x3": None,
    }

    if ortho_rgba is None or ortho_rgba.ndim != 3 or ortho_rgba.shape[2] != 4:
        info["reason"] = "invalid_rgba"
        return ortho_rgba, None, info

    H, W = ortho_rgba.shape[:2]
    src_pts = []
    dst_pts = []
    used_edges = 0

    for edge in selected_edges or []:
        selected_line = edge.get("selected_line")
        target_p0 = edge.get("target_p0")
        target_p1 = edge.get("target_p1")
        if selected_line is None or target_p0 is None or target_p1 is None:
            continue

        src_line = np.asarray(selected_line, dtype=np.float64)
        target = np.vstack([
            np.asarray(target_p0, dtype=np.float64),
            np.asarray(target_p1, dtype=np.float64),
        ])
        if src_line.shape != (2, 2) or target.shape != (2, 2):
            continue
        if not (np.isfinite(src_line).all() and np.isfinite(target).all()):
            continue

        target_dir = target[1] - target[0]
        target_len = float(np.linalg.norm(target_dir))
        src_dir = src_line[1] - src_line[0]
        src_len = float(np.linalg.norm(src_dir))
        if target_len < 1e-6 or src_len < 1e-6:
            continue

        target_u = target_dir / target_len
        src_u = src_dir / src_len
        if float(np.dot(src_u, target_u)) < 0.0:
            src_u = -src_u

        target_center = 0.5 * (target[0] + target[1])
        src_center = 0.5 * (src_line[0] + src_line[1])
        for target_pt in (target[0], target_center, target[1]):
            along = float(np.dot(target_pt - target_center, target_u))
            src_pt = src_center + along * src_u
            src_pts.append(src_pt)
            dst_pts.append(target_pt)
        used_edges += 1

    info["num_edge_matches"] = int(used_edges)
    info["num_point_pairs"] = int(len(src_pts))
    if used_edges < 2:
        info["reason"] = "not_enough_edge_matches"
        return ortho_rgba, None, info
    if len(src_pts) < 3:
        info["reason"] = "not_enough_point_pairs"
        return ortho_rgba, None, info

    src_arr = np.asarray(src_pts, dtype=np.float32)
    dst_arr = np.asarray(dst_pts, dtype=np.float32)
    if np.linalg.matrix_rank(dst_arr - dst_arr.mean(axis=0, keepdims=True)) < 2:
        info["reason"] = "degenerate_target_points"
        return ortho_rgba, None, info
    M, inliers = cv2.estimateAffine2D(
        src_arr,
        dst_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=2000,
        confidence=0.98,
        refineIters=10,
    )
    if M is None:
        M, inliers = cv2.estimateAffinePartial2D(
            src_arr,
            dst_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=8.0,
            maxIters=2000,
            confidence=0.98,
            refineIters=10,
        )
    if M is None:
        info["reason"] = "affine_estimation_failed"
        return ortho_rgba, None, info

    M = np.asarray(M, dtype=np.float64)
    det = float(np.linalg.det(M[:, :2]))
    max_trans = float(max(W, H) * 0.75)
    if not np.isfinite(M).all() or abs(det) < 0.20 or abs(det) > 5.0:
        info["reason"] = "affine_sanity_failed"
        info["determinant"] = det
        return ortho_rgba, None, info
    if float(np.linalg.norm(M[:, 2])) > max_trans:
        info["reason"] = "affine_translation_too_large"
        info["translation_norm_px"] = float(np.linalg.norm(M[:, 2]))
        return ortho_rgba, None, info

    warped = cv2.warpAffine(
        ortho_rgba,
        M.astype(np.float32),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    info.update({
        "applied": True,
        "reason": "applied",
        "determinant": det,
        "translation_norm_px": float(np.linalg.norm(M[:, 2])),
        "matrix_2x3": [[float(v) for v in row] for row in M.tolist()],
        "inliers": int(np.asarray(inliers).sum()) if inliers is not None else None,
    })
    return warped, M, info

def save_hough_warp_overlay(
    img_pil: Image.Image,
    wall_quad_xy: np.ndarray,
    out_path: str
) -> None:
    """
    Save transformed result with projected wall quad overlay.
    """
    out_bgr = _rgba_image_to_bgr_on_background(img_pil)

    q_i = np.round(np.asarray(wall_quad_xy, dtype=np.float64)).astype(np.int32)
    n = len(q_i)
    for i in range(n):
        p1 = tuple(q_i[i])
        p2 = tuple(q_i[(i + 1) % n])
        cv2.line(out_bgr, p1, p2, (0, 0, 255), 2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(
            out_bgr,
            f"W{i}",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(out_rgb).save(out_path)

def quad_area_abs(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def apply_affine2x3(xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return xy @ M[:, :2].T + M[:, 2]

def min_signed_dist_points_to_quad(pts: np.ndarray, quad: np.ndarray) -> float:
    contour = np.asarray(quad, dtype=np.float32).reshape((-1, 1, 2))
    dmin = 1e18
    for p in np.asarray(pts, dtype=np.float64):
        d = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
        dmin = min(dmin, float(d))
    return float(dmin)

def fit_seg_quad_inside_wall_quad(
    seg_quad: np.ndarray,
    wall_quad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit segmentation quad inside wall quad using:
      - uniform scale
      - translation

    IMPORTANT:
      This does NOT try to make the two quads coincide.
      It finds the largest admissible scale such that the segmentation quad
      remains inside the wall quad.

    Output:
      seg_quad_fitted, M_2x3, fit_info
    """
    seg_quad = np.asarray(seg_quad, dtype=np.float64)
    wall_quad = np.asarray(wall_quad, dtype=np.float64)

    seg_center = seg_quad.mean(axis=0)
    wall_center = wall_quad.mean(axis=0)

    # Work with a centered quad so scale happens around seg center
    seg_centered = seg_quad - seg_center

    wxmin, wymin = wall_quad.min(axis=0)
    wxmax, wymax = wall_quad.max(axis=0)
    wall_w = max(float(wxmax - wxmin), 1.0)
    wall_h = max(float(wymax - wymin), 1.0)
    wall_diag = max(float(np.hypot(wall_w, wall_h)), 1.0)

    seg_area = max(quad_area_abs(seg_quad), 1e-6)
    wall_area = max(quad_area_abs(wall_quad), 1e-6)

    # area-based starting scale
    s_guess = math.sqrt(wall_area / seg_area)

    center_dx_vals = np.linspace(
        -PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * wall_w,
        +PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * wall_w,
        PERSPECTIVE_FIT_CENTER_SHIFT_STEPS
    )
    center_dy_vals = np.linspace(
        -PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * wall_h,
        +PERSPECTIVE_FIT_CENTER_SHIFT_FRAC * wall_h,
        PERSPECTIVE_FIT_CENTER_SHIFT_STEPS
    )

    best_scale = -1.0
    best_center_dist = 1e18
    best_min_dist = -1e18
    best_center_xy = wall_center.copy()
    best_quad = seg_quad.copy()

    def build_quad(center_xy: np.ndarray, scale: float) -> np.ndarray:
        return seg_centered * scale + center_xy

    def is_inside(q: np.ndarray) -> Tuple[bool, float]:
        dmin = min_signed_dist_points_to_quad(q, wall_quad)
        return dmin >= PERSPECTIVE_FIT_INSET_PX, dmin

    for dx in center_dx_vals:
        for dy in center_dy_vals:
            center_xy = wall_center + np.array([dx, dy], dtype=np.float64)

            # First grow an upper bound until we leave the wall quad
            s_lo = 0.0
            s_hi = max(0.05, s_guess)

            q_hi = build_quad(center_xy, s_hi)
            ok_hi, d_hi = is_inside(q_hi)

            grow_iter = 0
            while ok_hi and s_hi < PERSPECTIVE_FIT_MAX_SCALE and grow_iter < 20:
                s_lo = s_hi
                s_hi *= PERSPECTIVE_FIT_SCALE_GROWTH
                q_hi = build_quad(center_xy, s_hi)
                ok_hi, d_hi = is_inside(q_hi)
                grow_iter += 1

            # Binary search the largest admissible scale
            lo = s_lo
            hi = s_hi

            for _ in range(PERSPECTIVE_FIT_BINARY_STEPS):
                mid = 0.5 * (lo + hi)
                q_mid = build_quad(center_xy, mid)
                ok_mid, d_mid = is_inside(q_mid)
                if ok_mid:
                    lo = mid
                else:
                    hi = mid

            s_best_here = lo
            q_best_here = build_quad(center_xy, s_best_here)
            _, d_best_here = is_inside(q_best_here)

            q_center_here = q_best_here.mean(axis=0)
            center_dist_here = float(np.linalg.norm(q_center_here - wall_center) / wall_diag)

            # MAIN objective: maximize scale
            # tie-break 1: smaller center distance
            # tie-break 2: min_dist closest to inset from above
            if (
                (s_best_here > best_scale + 1e-9) or
                (abs(s_best_here - best_scale) <= 1e-9 and center_dist_here < best_center_dist - 1e-9) or
                (abs(s_best_here - best_scale) <= 1e-9 and abs(center_dist_here - best_center_dist) <= 1e-9 and d_best_here < best_min_dist)
            ):
                best_scale = s_best_here
                best_center_dist = center_dist_here
                best_min_dist = d_best_here
                best_center_xy = center_xy
                best_quad = q_best_here

    tx = float(best_center_xy[0] - best_scale * seg_center[0])
    ty = float(best_center_xy[1] - best_scale * seg_center[1])

    M = np.array([
        [best_scale, 0.0, tx],
        [0.0, best_scale, ty]
    ], dtype=np.float64)

    fit_info = {
        "scale": float(best_scale),
        "tx": float(tx),
        "ty": float(ty),
        "final_center_x": float(best_center_xy[0]),
        "final_center_y": float(best_center_xy[1]),
        "min_signed_dist_px": float(best_min_dist),
        "center_dist": float(best_center_dist),
        "touching_like": bool(best_min_dist <= (PERSPECTIVE_FIT_INSET_PX + 0.5)),
    }

    return best_quad, M, fit_info

def warp_rgba_by_affine2x3(
    rgba_img: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    Apply only scale + translation to the full RGBA image in the ORIGINAL perspective frame.
    """
    if rgba_img is None or rgba_img.ndim != 3 or rgba_img.shape[2] != 4:
        raise ValueError("rgba_img must be HxWx4 RGBA uint8")

    H_img, W_img = rgba_img.shape[:2]

    src_bgra = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGRA)
    warped_bgra = cv2.warpAffine(
        src_bgra,
        M.astype(np.float32),
        (W_img, H_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    warped_rgba = cv2.cvtColor(warped_bgra, cv2.COLOR_BGRA2RGBA)
    return warped_rgba
