# -*- coding: utf-8 -*-
"""Camera geometry, native-source selection, SAM3 loading, and rectification."""

import math
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    HAVE_MATPLOTLIB = True
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    MplPolygon = None
    HAVE_MATPLOTLIB = False

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
# Import Pillow before SAM3/torchvision on Windows to avoid PIL DLL loading conflicts.
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .config import *
from .streetview import (
    fetch_sv_image_by_id,
    grid_heading_to_true_deg,
    solve_fov_deg,
    true_bearing_deg,
    wrap_delta_deg,
)
from .utils import safe_unit, save_with_overlay
from .wireframe_fit import (
    apply_homography,
    create_wireframe_fit_overlay,
    fit_wireframe_to_image,
    make_production_fit_config,
    wireframe_fit_metadata,
)


def _camera_xyz_for_record(rec, base_z, camera_elevation_resolver=None):
    px, py = rec["utm"]
    fallback_z = float(base_z) + float(FIXED_HEIGHT_M)
    if camera_elevation_resolver is None:
        elevation_info = {
            "used_dgm": False,
            "reason": "dgm_resolver_not_provided",
            "camera_z_m": fallback_z,
            "ground_z_m": None,
            "fallback_camera_z_m": fallback_z,
            "difference_from_fallback_m": None,
            "camera_height_m": float(FIXED_HEIGHT_M),
            "x": float(px),
            "y": float(py),
        }
        return np.array([float(px), float(py), fallback_z], dtype=np.float64), elevation_info

    decision = camera_elevation_resolver.resolve(
        float(px),
        float(py),
        source_label=f"pano {rec.get('pano_id', 'unknown')}",
    )
    return (
        np.array(
            [float(px), float(py), float(decision.camera_z_m)],
            dtype=np.float64,
        ),
        decision.as_dict(),
    )


def build_pose_from_heading_pitch(cam_xyz, heading_deg, pitch_deg, img_size=SV_SIZE, fov_deg=90.0):
    W, H = [int(v) for v in img_size.lower().split("x")]
    fx = fy = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx],
                  [0,  fy, cy],
                  [0,  0,  1]], float)
    C = np.array(cam_xyz, float)
    az = math.radians(heading_deg)
    el = math.radians(pitch_deg)
    f = np.array([math.sin(az)*math.cos(el),
                  math.cos(az)*math.cos(el),
                  math.sin(el)], float)
    f = safe_unit(f)
    world_up = np.array([0.0, 0.0, 1.0])
    r = safe_unit(np.cross(f, world_up))
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    u = safe_unit(np.cross(r, f))
    R_wc = np.vstack([r, u, f])
    return K, R_wc, C

def project_points_world_to_image(pts_xyz, K, R_wc, C, clip_behind=True):
    X = (pts_xyz - C).T
    Xc = R_wc @ X
    Zc = Xc[2, :]
    mask = Zc > 1e-6 if clip_behind else np.ones_like(Zc, dtype=bool)
    Xc = Xc[:, mask]
    if Xc.shape[1] == 0:
        return np.zeros((0, 2)), mask
    u = K[0,0] * (Xc[0, :] / Xc[2, :]) + K[0,2]
    v = K[1,1] * (-Xc[1, :] / Xc[2, :]) + K[1,2]
    return np.vstack([u, v]).T, mask

def _iter_mesh_triangles_world(meshes_named):
    for _name, mesh in meshes_named or []:
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            continue
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            continue
        if len(vertices) == 0:
            continue
        valid_faces = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        for face in faces[valid_faces]:
            tri = vertices[face]
            if np.isfinite(tri).all():
                yield tri

def _rasterize_camera_triangle_depth(depth_map, tri_camera, K, near_m):
    z = tri_camera[:, 2]
    if not np.isfinite(z).all() or np.any(z <= 0.0):
        return

    uv = np.empty((3, 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (tri_camera[:, 0] / z) + K[0, 2]
    uv[:, 1] = K[1, 1] * (-tri_camera[:, 1] / z) + K[1, 2]
    if not np.isfinite(uv).all():
        return

    height, width = depth_map.shape
    x0 = max(0, int(math.floor(float(np.min(uv[:, 0])))))
    x1 = min(width - 1, int(math.ceil(float(np.max(uv[:, 0])))))
    y0 = max(0, int(math.floor(float(np.min(uv[:, 1])))))
    y1 = min(height - 1, int(math.ceil(float(np.max(uv[:, 1])))))
    if x1 < x0 or y1 < y0:
        return

    local = uv - np.array([x0, y0], dtype=np.float64)
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local).astype(np.int32), 1, lineType=cv2.LINE_8)
    if mask.max() == 0:
        return

    normal = np.cross(tri_camera[1] - tri_camera[0], tri_camera[2] - tri_camera[0])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        return
    plane_d = -float(np.dot(normal, tri_camera[0]))

    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    rays = np.empty((yy.shape[0], yy.shape[1], 3), dtype=np.float64)
    rays[:, :, 0] = (xx.astype(np.float64) + 0.5 - K[0, 2]) / K[0, 0]
    rays[:, :, 1] = -(yy.astype(np.float64) + 0.5 - K[1, 2]) / K[1, 1]
    rays[:, :, 2] = 1.0
    denom = rays[:, :, 0] * normal[0] + rays[:, :, 1] * normal[1] + rays[:, :, 2] * normal[2]
    valid = (mask > 0) & (np.abs(denom) > 1e-12)
    if not valid.any():
        return

    depth = -plane_d / denom
    valid &= np.isfinite(depth) & (depth >= float(near_m))
    if not valid.any():
        return

    current = depth_map[y0:y1 + 1, x0:x1 + 1]
    update = valid & (depth < current)
    if update.any():
        current[update] = depth[update].astype(np.float32)

def render_model_depth_map(meshes_named, K, R_wc, C, image_size, near_m=None):
    """Render visible model depth in camera-forward meters on the given canvas."""
    width, height = [int(v) for v in image_size]
    near = float(MODEL_DEPTH_NEAR_M if near_m is None else near_m)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    for tri_world in _iter_mesh_triangles_world(meshes_named):
        tri_camera = (R_wc @ (tri_world - C).T).T
        if np.all(tri_camera[:, 2] < near):
            continue
        clipped = _clip_polygon_3d_halfspace(tri_camera, lambda p: p[2] - near)
        if len(clipped) < 3:
            continue
        clipped = np.asarray(clipped, dtype=np.float64)
        for idx in range(1, len(clipped) - 1):
            _rasterize_camera_triangle_depth(
                depth,
                np.vstack([clipped[0], clipped[idx], clipped[idx + 1]]),
                K,
                near,
            )

    depth[~np.isfinite(depth)] = np.nan
    return depth


def evaluate_target_wall_model_visibility(
    meshes_named,
    target_mesh_names,
    K,
    R_wc,
    C,
    image_size,
):
    """Measure target-wall pixels that are not hidden by the rest of the model."""
    unavailable = {
        "target_model_visibility_available": False,
        "target_model_visibility_reason": "model_visibility_not_available",
        "target_self_visibility_fraction": None,
        "target_depth_pixel_count": 0,
        "target_visible_pixel_count": 0,
        "target_occluded_pixel_count": 0,
        "target_visibility_render_size_px": None,
        "target_visibility_visible_mask": None,
        "target_visibility_occluded_mask": None,
    }
    if not bool(globals().get("ENABLE_FACADE_SOURCE_MODEL_VISIBILITY", True)):
        return {**unavailable, "target_model_visibility_reason": "disabled"}

    all_meshes = list(meshes_named or [])
    wanted_names = {str(name) for name in (target_mesh_names or []) if name is not None}
    if not all_meshes or not wanted_names:
        return {**unavailable, "target_model_visibility_reason": "missing_model_or_target"}
    target_meshes = [
        (name, mesh)
        for name, mesh in all_meshes
        if str(name) in wanted_names
    ]
    if not target_meshes:
        return {**unavailable, "target_model_visibility_reason": "target_mesh_not_found"}

    source_width, source_height = [int(v) for v in image_size]
    max_dimension = max(1, int(globals().get(
        "FACADE_SOURCE_VISIBILITY_RENDER_MAX_DIM_PX",
        320,
    )))
    render_scale = min(
        1.0,
        float(max_dimension) / max(float(source_width), float(source_height), 1.0),
    )
    render_width = max(1, int(round(source_width * render_scale)))
    render_height = max(1, int(round(source_height * render_scale)))
    scale_x = float(render_width) / max(float(source_width), 1.0)
    scale_y = float(render_height) / max(float(source_height), 1.0)
    render_K = np.asarray(K, dtype=np.float64).copy()
    render_K[0, :] *= scale_x
    render_K[1, :] *= scale_y

    try:
        full_depth = render_model_depth_map(
            all_meshes,
            render_K,
            R_wc,
            C,
            (render_width, render_height),
        )
        target_depth = render_model_depth_map(
            target_meshes,
            render_K,
            R_wc,
            C,
            (render_width, render_height),
        )
    except Exception as exc:
        return {
            **unavailable,
            "target_model_visibility_reason": f"depth_render_failed: {exc}",
        }

    target_mask = np.isfinite(target_depth) & (target_depth > 0.0)
    if not target_mask.any():
        return {
            **unavailable,
            "target_model_visibility_reason": "target_not_rasterized_in_frame",
            "target_visibility_render_size_px": [render_width, render_height],
        }

    erosion_px = max(0, int(globals().get(
        "FACADE_SOURCE_VISIBILITY_MASK_ERODE_PX",
        1,
    )))
    evaluation_mask = target_mask
    if erosion_px > 0:
        kernel_size = 2 * erosion_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(
            target_mask.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0
        if eroded.any():
            evaluation_mask = eroded

    tolerance_m = max(0.0, float(globals().get(
        "FACADE_SOURCE_VISIBILITY_DEPTH_TOLERANCE_M",
        0.05,
    )))
    full_valid = np.isfinite(full_depth) & (full_depth > 0.0)
    visible_mask = (
        evaluation_mask
        & full_valid
        & (full_depth + tolerance_m >= target_depth)
    )
    occluded_mask = evaluation_mask & ~visible_mask
    target_pixels = int(evaluation_mask.sum())
    visible_pixels = int(visible_mask.sum())
    occluded_pixels = int(occluded_mask.sum())
    visible_fraction = float(visible_pixels / max(target_pixels, 1))

    return {
        "target_model_visibility_available": True,
        "target_model_visibility_reason": "measured_with_full_model_z_buffer",
        "target_self_visibility_fraction": visible_fraction,
        "target_depth_pixel_count": target_pixels,
        "target_visible_pixel_count": visible_pixels,
        "target_occluded_pixel_count": occluded_pixels,
        "target_visibility_render_size_px": [render_width, render_height],
        "target_visibility_visible_mask": visible_mask,
        "target_visibility_occluded_mask": occluded_mask,
    }

def warp_depth_map_to_canvas(depth_map, homography, image_size):
    """Warp a metric depth map into another image-space canvas."""
    width, height = [int(v) for v in image_size]
    H = np.asarray(homography, dtype=np.float64)
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth_fill = np.where(valid, depth, 0.0).astype(np.float32)
    valid_u8 = valid.astype(np.uint8) * 255
    warped_depth = cv2.warpPerspective(
        depth_fill,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    warped_valid = cv2.warpPerspective(
        valid_u8,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    warped_depth[~warped_valid] = np.nan
    return warped_depth

def depth_map_to_uint16_mm(depth_map, max_mm=None):
    max_value = int(np.clip(MODEL_DEPTH_MAX_MM_PNG if max_mm is None else max_mm, 1, 65535))
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, dtype=np.uint16)
    if valid.any():
        mm = np.rint(depth[valid] * 1000.0)
        out[valid] = np.clip(mm, 1, max_value).astype(np.uint16)
    return out

def depth_map_to_visual_png(depth_map):
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    visual = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return visual
    lo = float(np.nanpercentile(depth[valid], 2.0))
    hi = float(np.nanpercentile(depth[valid], 98.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(depth.shape, dtype=np.uint8)
    norm[valid] = np.clip((depth[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    visual[valid] = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)[valid]
    return visual

def _clip_polygon_3d_halfspace(poly, dist_fn, eps=1e-9):
    poly = [np.asarray(p, dtype=np.float64) for p in poly]
    if not poly:
        return []
    out = []
    prev = poly[-1]
    prev_d = float(dist_fn(prev))
    prev_in = prev_d >= -eps
    for cur in poly:
        cur_d = float(dist_fn(cur))
        cur_in = cur_d >= -eps
        if cur_in != prev_in:
            denom = prev_d - cur_d
            if abs(denom) > eps:
                t = prev_d / denom
                out.append(prev + t * (cur - prev))
        if cur_in:
            out.append(cur)
        prev = cur
        prev_d = cur_d
        prev_in = cur_in
    return out

def _clip_polygon_2d_halfspace(poly, inside_fn, intersect_fn):
    poly = [np.asarray(p, dtype=np.float64) for p in poly]
    if not poly:
        return []
    out = []
    prev = poly[-1]
    prev_in = bool(inside_fn(prev))
    for cur in poly:
        cur_in = bool(inside_fn(cur))
        if cur_in != prev_in:
            out.append(intersect_fn(prev, cur))
        if cur_in:
            out.append(cur)
        prev = cur
        prev_in = cur_in
    return out

def _clip_polygon_2d_to_rect(poly, W, H):
    x_min, y_min = 0.0, 0.0
    x_max, y_max = float(W - 1), float(H - 1)
    out = [np.asarray(p, dtype=np.float64) for p in poly]
    if not out:
        return np.zeros((0, 2), dtype=np.float64)

    def ix_at_x(a, b, x):
        den = b[0] - a[0]
        t = 0.0 if abs(den) < 1e-9 else (x - a[0]) / den
        return a + t * (b - a)

    def ix_at_y(a, b, y):
        den = b[1] - a[1]
        t = 0.0 if abs(den) < 1e-9 else (y - a[1]) / den
        return a + t * (b - a)

    out = _clip_polygon_2d_halfspace(out, lambda p: p[0] >= x_min, lambda a, b: ix_at_x(a, b, x_min))
    out = _clip_polygon_2d_halfspace(out, lambda p: p[0] <= x_max, lambda a, b: ix_at_x(a, b, x_max))
    out = _clip_polygon_2d_halfspace(out, lambda p: p[1] >= y_min, lambda a, b: ix_at_y(a, b, y_min))
    out = _clip_polygon_2d_halfspace(out, lambda p: p[1] <= y_max, lambda a, b: ix_at_y(a, b, y_max))
    if len(out) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)

def project_polygon_world_to_image_clipped(
    pts_xyz,
    K,
    R_wc,
    C,
    image_size,
    near_m=None,
    clip_to_image=True,
):
    """
    Project a world-space polygon after clipping it to the camera near plane.

    This prevents close Street View debug overlays from wrapping across the image
    when some facade vertices are behind or extremely close to the camera.
    """
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return np.zeros((0, 2), dtype=np.float64)

    near = float(FACADE_PROJECTION_NEAR_PLANE_M if near_m is None else near_m)
    Xc = (np.asarray(R_wc, dtype=np.float64) @ (pts - np.asarray(C, dtype=np.float64)).T).T
    clipped = _clip_polygon_3d_halfspace(Xc, lambda p: p[2] - near)
    if len(clipped) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    clipped = np.asarray(clipped, dtype=np.float64)
    z = np.maximum(clipped[:, 2], near)
    uv = np.empty((clipped.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * (clipped[:, 0] / z) + K[0, 2]
    uv[:, 1] = K[1, 1] * (-clipped[:, 1] / z) + K[1, 2]

    if clip_to_image:
        W, H = image_size
        uv = _clip_polygon_2d_to_rect(uv, int(W), int(H))
    return uv


def _segments_intersect_2d(a0, a1, b0, b1, eps=1e-8):
    def cross(p, q, r):
        return float(np.cross(q - p, r - p))

    def on_segment(p, q, r):
        return bool(
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = cross(a0, a1, b0)
    o2 = cross(a0, a1, b1)
    o3 = cross(b0, b1, a0)
    o4 = cross(b0, b1, a1)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a0, b0, a1):
        return True
    if abs(o2) <= eps and on_segment(a0, b1, a1):
        return True
    if abs(o3) <= eps and on_segment(b0, a0, b1):
        return True
    if abs(o4) <= eps and on_segment(b0, a1, b1):
        return True
    return False


def _closed_polyline_self_intersects(points):
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count < 4:
        return False
    for edge0 in range(count):
        edge0_next = (edge0 + 1) % count
        for edge1 in range(edge0 + 1, count):
            edge1_next = (edge1 + 1) % count
            if edge0 == edge1 or edge0_next == edge1 or edge1_next == edge0:
                continue
            if edge0 == 0 and edge1_next == 0:
                continue
            if _segments_intersect_2d(
                points[edge0],
                points[edge0_next],
                points[edge1],
                points[edge1_next],
            ):
                return True
    return False


def project_outline_world_edges_near_clipped(
    pts_xyz,
    K,
    R_wc,
    C,
    near_m=None,
):
    """Project only real outline edges, clipping each one at the near plane.

    No synthetic edge is added between near-plane intersections. This avoids
    the self-crossing loop produced when a closed polygon straddles the camera.
    """
    points_world = np.asarray(pts_xyz, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[0] < 3 or points_world.shape[1] != 3:
        raise ValueError("A world outline needs at least three XYZ points.")

    near = float(FACADE_PROJECTION_NEAR_PLANE_M if near_m is None else near_m)
    R_wc = np.asarray(R_wc, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    camera_points = (R_wc @ (points_world - C).T).T
    depths = camera_points[:, 2]

    projected_points = []
    projected_world_points = []
    segment_indices = []
    clipped_edge_count = 0
    skipped_edge_count = 0

    for index0 in range(len(points_world)):
        index1 = (index0 + 1) % len(points_world)
        camera0 = camera_points[index0].copy()
        camera1 = camera_points[index1].copy()
        world0 = points_world[index0].copy()
        world1 = points_world[index1].copy()
        depth0, depth1 = float(camera0[2]), float(camera1[2])

        if depth0 < near and depth1 < near:
            skipped_edge_count += 1
            continue
        if depth0 < near:
            denominator = depth1 - depth0
            if abs(denominator) < 1e-12:
                skipped_edge_count += 1
                continue
            amount = float(np.clip((near - depth0) / denominator, 0.0, 1.0))
            camera0 = camera0 + amount * (camera1 - camera0)
            world0 = world0 + amount * (world1 - world0)
            clipped_edge_count += 1
        if depth1 < near:
            denominator = depth0 - depth1
            if abs(denominator) < 1e-12:
                skipped_edge_count += 1
                continue
            amount = float(np.clip((near - depth1) / denominator, 0.0, 1.0))
            camera1 = camera1 + amount * (camera0 - camera1)
            world1 = world1 + amount * (world0 - world1)
            clipped_edge_count += 1

        uv0 = np.array([
            K[0, 0] * (camera0[0] / camera0[2]) + K[0, 2],
            K[1, 1] * (-camera0[1] / camera0[2]) + K[1, 2],
        ], dtype=np.float64)
        uv1 = np.array([
            K[0, 0] * (camera1[0] / camera1[2]) + K[0, 2],
            K[1, 1] * (-camera1[1] / camera1[2]) + K[1, 2],
        ], dtype=np.float64)
        if not np.isfinite(np.vstack([uv0, uv1])).all():
            skipped_edge_count += 1
            continue

        start_index = len(projected_points)
        projected_points.extend([uv0, uv1])
        projected_world_points.extend([world0, world1])
        segment_indices.append((start_index, start_index + 1))

    all_vertices_in_front = bool(np.isfinite(depths).all() and np.all(depths >= near))
    original_projection = np.zeros((0, 2), dtype=np.float64)
    self_intersects = False
    if all_vertices_in_front:
        original_projection, _ = project_points_world_to_image(
            points_world,
            K,
            R_wc,
            C,
            clip_behind=False,
        )
        self_intersects = bool(
            not np.isfinite(original_projection).all()
            or _closed_polyline_self_intersects(original_projection)
        )

    info = {
        "near_plane_m": near,
        "minimum_vertex_depth_m": float(np.nanmin(depths)) if len(depths) else None,
        "front_vertex_count": int(np.count_nonzero(depths >= near)),
        "vertex_count": int(len(points_world)),
        "visible_real_edge_count": int(len(segment_indices)),
        "near_clipped_edge_count": int(clipped_edge_count),
        "skipped_edge_count": int(skipped_edge_count),
        "all_vertices_in_front": all_vertices_in_front,
        "unclipped_projection_self_intersects": self_intersects,
        "full_outline_topology_valid": bool(all_vertices_in_front and not self_intersects),
    }
    return (
        np.asarray(projected_points, dtype=np.float64).reshape(-1, 2),
        segment_indices,
        np.asarray(projected_world_points, dtype=np.float64).reshape(-1, 3),
        info,
    )

def _normalized_line_through(p0, p1):
    v = np.array([p1[0]-p0[0], p1[1]-p0[1]], float)
    n = np.array([-v[1], v[0]], float)
    n = safe_unit(n)
    a, b = n[0], n[1]
    c = -(a*p0[0] + b*p0[1])
    return a, b, c

def _offset_line(a, b, c, offset): return a, b, c - offset

def _x_at_y(a, b, c, y, fallback_x):
    eps = 1e-9
    if abs(a) < eps: return fallback_x
    return (-(b*y + c)) / a

def build_lr_band_polygon_outward(uv_quad, img_w, img_h, buffer_px):
    if uv_quad.shape[0] < 4:
        return None
    b1, b2, t2, t1 = uv_quad[0], uv_quad[1], uv_quad[2], uv_quad[3]
    center = uv_quad.mean(axis=0)
    def inward_line(p0, p1):
        a, b, c = _normalized_line_through(p0, p1)
        mid = 0.5*(np.array(p0)+np.array(p1))
        if np.dot(np.array([a, b]), center - mid) < 0:
            a, b, c = -a, -b, -c
        return a, b, c
    aL_in, bL_in, cL_in = inward_line(b1, t1)
    aR_in, bR_in, cR_in = inward_line(b2, t2)
    aL, bL, cL = _offset_line(-aL_in, -bL_in, -cL_in, +buffer_px)
    aR, bR, cR = _offset_line(-aR_in, -bR_in, -cR_in, +buffer_px)
    y_top, y_bot = 0.0, float(img_h)
    fallback_L = float(b1[0]); fallback_R = float(b2[0])
    xL_top = _x_at_y(aL, bL, cL, y_top, fallback_L)
    xR_top = _x_at_y(aR, bR, cR, y_top, fallback_R)
    xL_bot = _x_at_y(aL, bL, cL, y_bot, fallback_L)
    xR_bot = _x_at_y(aR, bR, cR, y_bot, fallback_R)
    def clamp_x(x): return float(np.clip(x, -10*img_w, 10*img_w))
    return [
        (clamp_x(xL_top), y_top),
        (clamp_x(xR_top), y_top),
        (clamp_x(xR_bot), y_bot),
        (clamp_x(xL_bot), y_bot),
    ]

def build_lr_band_rgba(img_pil: Image.Image, uv_quad: np.ndarray, buffer_px: int):
    W, H = img_pil.width, img_pil.height
    band_poly = build_lr_band_polygon_outward(uv_quad, W, H, buffer_px)
    if band_poly is None:
        return None, None, None
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(band_poly, fill=255)
    rgba = img_pil.convert("RGBA")
    r, g, b, _ = rgba.split()
    rgba_band = Image.merge("RGBA", (r, g, b, mask))
    bbox = mask.getbbox()
    return rgba_band, band_poly, bbox

def load_sam3(prompt_facade: str = SAM3_PROMPT_FACADE, prompt_roof: str = SAM3_PROMPT_ROOF):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build SAM3 model (downloads checkpoints via HF; requires hf auth login)
    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model, device=str(device))

    return device, processor, prompt_facade, prompt_roof

def homography_from_4pts(src4x2, dst4x2):
    H, _ = cv2.findHomography(src4x2.astype(np.float32), dst4x2.astype(np.float32), 0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    return (H / H[2,2]).astype(np.float64)


def S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip: bool):
    """Map facade-plane meters to rectified texture pixels."""
    scale = float(ppm)
    if flip:
        return np.array(
            [
                [scale, 0.0, -scale * float(xmin)],
                [0.0, -scale, scale * float(ymax)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [scale, 0.0, -scale * float(xmin)],
            [0.0, scale, -scale * float(ymin)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def choose_orientation_from_poly(poly_m, xmin, ymin, xmax, ymax, ppm):
    """Choose the facade-plane vertical direction that keeps its roof above its base."""
    poly_m = np.asarray(poly_m, dtype=np.float64)
    if poly_m.shape != (4, 2) or not np.isfinite(poly_m).all():
        raise ValueError("poly_m must be a finite [base-left, base-right, roof-right, roof-left] quadrilateral")

    unflipped = apply_homography(
        poly_m,
        S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip=False),
    )
    flipped = apply_homography(
        poly_m,
        S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip=True),
    )

    def is_upright(poly_px):
        roof_y = 0.5 * (poly_px[2, 1] + poly_px[3, 1])
        base_y = 0.5 * (poly_px[0, 1] + poly_px[1, 1])
        return bool(roof_y < base_y)

    unflipped_upright = is_upright(unflipped)
    flipped_upright = is_upright(flipped)
    if unflipped_upright != flipped_upright:
        return flipped_upright
    return not unflipped_upright


def wall_metric_target_from_corners(b1, b2, t2, t1):
    b1 = np.asarray(b1, float); b2 = np.asarray(b2, float)
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)

    u_dir = safe_unit(t2 - t1)  # along roof
    v_seed = 0.5 * ((t1 - b1) + (t2 - b2))
    v_dir = v_seed - np.dot(v_seed, u_dir) * u_dir
    v_dir = safe_unit(v_dir)
    if np.dot(v_dir, (t1 + t2)/2 - (b1 + b2)/2) < 0:
        v_dir = -v_dir

    O = t1
    def to_uv(p):
        d = p - O
        return np.array([np.dot(d, u_dir), np.dot(d, v_dir)], float)

    t1_m = to_uv(t1); t2_m = to_uv(t2)
    b1_m = to_uv(b1); b2_m = to_uv(b2)

    v_top = 0.5*(t1_m[1] + t2_m[1])
    t1_m[1] -= v_top; t2_m[1] -= v_top
    b1_m[1] -= v_top; b2_m[1] -= v_top

    dst_m = np.vstack([b1_m, b2_m, t2_m, t1_m])  # [b1,b2,t2,t1]
    width_m  = float(t2_m[0] - t1_m[0])
    h_left   = float(t1_m[1] - b1_m[1])
    h_right  = float(t2_m[1] - b2_m[1])

    meta = {
        "origin_xyz": [float(v) for v in O.tolist()],
        "u_dir": [float(v) for v in u_dir.tolist()],
        "v_dir": [float(v) for v in v_dir.tolist()],
        "width_m": width_m,
        "height_left_m": h_left,
        "height_right_m": h_right
    }
    return dst_m, meta

def uv_inside_image(uv, W, H, B):
    return np.all((uv[:,0] >= B) & (uv[:,0] <= W - B) &
                  (uv[:,1] >= B) & (uv[:,1] <= H - B))

def yaw_pitch_of_points(cam, pts_xyz):
    cam = np.asarray(cam, float)
    yaws, pits = [], []
    for p in pts_xyz:
        dx = p[0]-cam[0]; dy = p[1]-cam[1]
        dz = p[2]-cam[2]
        yaw = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0  # 0 deg is +Y
        rho = max(np.hypot(dx, dy), 1e-9)
        pit = np.degrees(np.arctan2(dz, rho))
        yaws.append(yaw); pits.append(pit)
    return np.array(yaws), np.array(pits)

def circular_span(angles_deg):
    a = np.sort(np.mod(angles_deg, 360.0))
    a_ext = np.concatenate([a, a + 360.0])
    gaps = a_ext[1:] - a_ext[:-1]
    k = np.argmax(gaps)
    start = a_ext[k+1]
    end   = a_ext[k] + 360.0
    span  = end - start
    center = (start + end) * 0.5
    start_mod = start % 360.0
    end_mod   = (start + span) % 360.0
    center_mod= center % 360.0
    return center_mod, span, start_mod, end_mod

def _poly_area_abs_2d(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _facade_wireframe_fit_config():
    return make_production_fit_config(
        allow_rotation=bool(globals().get("FACADE_WIREFRAME_FIT_ALLOW_ROTATION", False)),
        minimum_score_improvement=float(globals().get(
            "FACADE_WIREFRAME_FIT_MIN_SCORE_IMPROVEMENT",
            0.025,
        )),
    )


def _refine_effective_camera_parameters(src, outline_xyz, fitted_outline_px):
    """Estimate effective heading/pitch/FOV that reproduce a fitted raw outline.

    The camera center stays fixed. These parameters describe the best effective
    pinhole mapping for this image; direct fitted pixels remain authoritative.
    """
    if not bool(globals().get("FACADE_WIREFRAME_FIT_REFINE_CAMERA_PARAMETERS", True)):
        return {"attempted": False, "accepted": False, "reason": "disabled"}

    world_points = np.asarray(outline_xyz, dtype=np.float64)
    target = np.asarray(fitted_outline_px, dtype=np.float64)
    if world_points.shape[0] != target.shape[0] or world_points.shape[0] < 3:
        return {"attempted": False, "accepted": False, "reason": "invalid_correspondences"}

    image_width, image_height = src["img"].size
    image_size = f"{int(image_width)}x{int(image_height)}"
    request_heading0 = float(src["heading"])
    heading0 = float(src.get("projection_heading", request_heading0))
    meridian_convergence = float(
        src.get(
            "meridian_convergence_deg",
            wrap_delta_deg(request_heading0, heading0),
        )
    )
    pitch0 = float(src["pitch"])
    fov0 = float(src["fov"])
    camera_xyz = np.asarray(src["C"], dtype=np.float64)

    fov_lower = max(float(FOV_MIN), min(fov0 * 0.70, fov0 - 0.1))
    fov_upper = min(float(FOV_MAX), max(fov0 * 1.30, fov0 + 0.1))

    def project_effective(parameters):
        heading = (heading0 + float(parameters[0])) % 360.0
        pitch = pitch0 + float(parameters[1])
        fov = float(parameters[2])
        K_fit, R_fit, _ = build_pose_from_heading_pitch(
            camera_xyz,
            heading,
            pitch,
            img_size=image_size,
            fov_deg=fov,
        )
        uv, _ = project_points_world_to_image(
            world_points,
            K_fit,
            R_fit,
            camera_xyz,
            clip_behind=False,
        )
        return uv, K_fit, R_fit

    def residual(parameters):
        uv, _K, _R = project_effective(parameters)
        if uv.shape != target.shape or not np.isfinite(uv).all():
            return np.full(target.size, 1.0e4, dtype=np.float64)
        return (uv - target).reshape(-1)

    initial = np.array([0.0, 0.0, fov0], dtype=np.float64)
    try:
        initial_residual = residual(initial)
        initial_rmse = float(np.sqrt(np.mean(initial_residual ** 2)))
        optimized = least_squares(
            residual,
            initial,
            bounds=(
                np.array([-20.0, -20.0, fov_lower], dtype=np.float64),
                np.array([20.0, 20.0, fov_upper], dtype=np.float64),
            ),
            loss="soft_l1",
            f_scale=4.0,
            x_scale=np.array([5.0, 5.0, 10.0], dtype=np.float64),
            max_nfev=160,
        )
        fitted_uv, fitted_K, fitted_R = project_effective(optimized.x)
        final_rmse = float(np.sqrt(np.mean((fitted_uv - target) ** 2)))
    except Exception as exc:
        return {
            "attempted": True,
            "accepted": False,
            "reason": f"optimization_failed: {exc}",
        }

    accepted = bool(
        optimized.success
        and np.isfinite(final_rmse)
        and final_rmse < initial_rmse
    )
    corrected_heading = (heading0 + float(optimized.x[0])) % 360.0
    corrected_request_heading = (
        corrected_heading + meridian_convergence
    ) % 360.0
    corrected_pitch = pitch0 + float(optimized.x[1])
    corrected_fov = float(optimized.x[2])
    return {
        "attempted": True,
        "accepted": accepted,
        "reason": "accepted" if accepted else "no_camera_reprojection_improvement",
        "interpretation": (
            "effective raw-image pinhole parameters; camera center held fixed; "
            "request heading uses true north and projection heading uses CRS grid north"
        ),
        "original": {
            "heading_deg": request_heading0,
            "projection_heading_deg": heading0,
            "meridian_convergence_deg": meridian_convergence,
            "pitch_deg": pitch0,
            "fov_deg": fov0,
        },
        "corrected": {
            "heading_deg": corrected_request_heading,
            "projection_heading_deg": corrected_heading,
            "meridian_convergence_deg": meridian_convergence,
            "pitch_deg": corrected_pitch,
            "fov_deg": corrected_fov,
            "K": fitted_K.astype(float).tolist(),
            "R_wc": fitted_R.astype(float).tolist(),
        },
        "delta": {
            "heading_deg": float(optimized.x[0]),
            "pitch_deg": float(optimized.x[1]),
            "fov_deg": float(corrected_fov - fov0),
        },
        "target_reprojection_rmse_before_px": initial_rmse,
        "target_reprojection_rmse_after_px": final_rmse,
        "optimizer_message": str(optimized.message),
    }


def _fit_raw_source_wireframe(src, outline_xyz, facade_tag, source_index):
    if not bool(globals().get("ENABLE_FACADE_WIREFRAME_FIT", True)):
        return
    if not bool(globals().get("FACADE_WIREFRAME_FIT_RAW_SOURCES", True)):
        return

    projected_uv, real_edge_indices, projected_world_points, projection_info = (
        project_outline_world_edges_near_clipped(
            np.asarray(outline_xyz, dtype=np.float64),
            src["K"],
            src["Rwc"],
            src["C"],
            near_m=FACADE_PROJECTION_NEAR_PLANE_M,
        )
    )
    src["wireframe_projection_info"] = projection_info
    topology_valid = bool(projection_info["full_outline_topology_valid"])
    if len(real_edge_indices) < 2:
        src["wireframe_fit"] = {
            "applied": False,
            "reason": "fewer_than_two_visible_real_edges",
            "projection_info": projection_info,
            "eligible_for_source_correction": False,
        }
        src["wireframe_fit_H"] = np.eye(3, dtype=np.float64)
        src["wireframe_fit_applied_to_source"] = False
        src["effective_camera_fit"] = {
            "attempted": False,
            "accepted": False,
            "reason": "insufficient_visible_real_edges",
        }
        print(
            f"[{facade_tag}] source {source_index:02d} wireframe fit skipped | "
            "fewer than two real wall edges are visible in front of the camera"
        )
        return

    try:
        image_bgr = cv2.cvtColor(np.asarray(src["img"].convert("RGB")), cv2.COLOR_RGB2BGR)
        fit_result = fit_wireframe_to_image(
            image_bgr,
            projected_uv,
            config=_facade_wireframe_fit_config(),
            segment_indices=real_edge_indices,
        )
        applied_to_source = bool(fit_result["applied"] and topology_valid)
        if fit_result["applied"] and not topology_valid:
            fit_result["status_label"] = "visible-edge fit only; source too close for full projection"
        overlay_bgr = create_wireframe_fit_overlay(
            image_bgr,
            fit_result,
            config=_facade_wireframe_fit_config(),
        )
        fit_metadata = wireframe_fit_metadata(fit_result)
        fit_metadata.update({
            "geometry_mode": "independently_near_clipped_real_outline_edges",
            "projection_info": projection_info,
            "eligible_for_source_correction": topology_valid,
            "applied_to_source_correction": applied_to_source,
        })
        src["wireframe_fit"] = fit_metadata
        src["wireframe_fit_H"] = (
            np.asarray(fit_result["homography"], dtype=np.float64)
            if applied_to_source else np.eye(3, dtype=np.float64)
        )
        src["wireframe_fit_applied_to_source"] = applied_to_source
        src["wireframe_fit_overlay"] = Image.fromarray(
            cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        )
        if applied_to_source:
            src["effective_camera_fit"] = _refine_effective_camera_parameters(
                src,
                projected_world_points,
                fit_result["fitted_points"],
            )
        else:
            src["effective_camera_fit"] = {
                "attempted": False,
                "accepted": False,
                "reason": (
                    "source_too_close_for_full_outline_topology"
                    if not topology_valid else "image_space_fit_not_applied"
                ),
            }
        fit_result.pop("canny", None)
        fit_result.pop("line_map", None)
        if fit_result["applied"] and not topology_valid:
            status = "visual fit only (too close; excluded from source correction)"
        else:
            status = "applied" if applied_to_source else "kept original"
        transform = fit_result["transform"]
        print(
            f"[{facade_tag}] source {source_index:02d} wireframe fit {status} | "
            f"scale={transform['scale']:.4f}, tx={transform['tx']:.1f}px, "
            f"ty={transform['ty']:.1f}px, gain={fit_result['score_improvement']:.4f}"
        )
    except Exception as exc:
        src["wireframe_fit"] = {
            "applied": False,
            "reason": f"fit_failed: {exc}",
            "projection_info": projection_info,
            "eligible_for_source_correction": False,
        }
        src["wireframe_fit_H"] = np.eye(3, dtype=np.float64)
        src["wireframe_fit_applied_to_source"] = False
        src["effective_camera_fit"] = {
            "attempted": False,
            "accepted": False,
            "reason": "image_space_fit_failed",
        }
        print(f"[{facade_tag}] source {source_index:02d} wireframe fit failed: {exc}")

def _source_projection_metric(src: Dict[str, Any], outline_xyz, image_size):
    W, H = image_size
    full_uv_raw, _ = project_points_world_to_image(
        np.asarray(outline_xyz, dtype=np.float64),
        src["K"],
        src["Rwc"],
        src["C"],
        clip_behind=False,
    )
    uv_clipped = project_polygon_world_to_image_clipped(
        outline_xyz,
        src["K"],
        src["Rwc"],
        src["C"],
        image_size,
        near_m=FACADE_PROJECTION_NEAR_PLANE_M,
        clip_to_image=True,
    )
    selection_outline = np.asarray(
        src.get("selection_visible_wall_outline_px", []),
        dtype=np.float64,
    )
    uses_near_clipped_projection = bool(
        src.get("selection_uses_near_clipped_projection", False)
        and selection_outline.ndim == 2
        and selection_outline.shape[0] >= 3
        and selection_outline.shape[1] == 2
        and np.isfinite(selection_outline).all()
    )
    uv_raw = (
        selection_outline
        if uses_near_clipped_projection
        else full_uv_raw
    )
    if uses_near_clipped_projection:
        uv_clipped = uv_raw.copy()
    projection_fit_H = np.asarray(
        src.get(
            "selection_projection_H",
            src.get("wireframe_fit_H", np.eye(3)),
        ),
        dtype=np.float64,
    )
    uv = (
        apply_homography(uv_raw, projection_fit_H)
        if uv_raw.shape[0] > 0 else uv_raw
    )
    uv_clipped_fitted = (
        apply_homography(uv_clipped, projection_fit_H)
        if uv_clipped.shape[0] > 0 else uv_clipped
    )
    full_uv = (
        apply_homography(full_uv_raw, projection_fit_H)
        if full_uv_raw.shape[0] > 0 else full_uv_raw
    )
    if uses_near_clipped_projection:
        finite = np.isfinite(uv).all(axis=1)
        front = np.ones((len(uv),), dtype=bool)
    else:
        depth = (
            src["Rwc"]
            @ (np.asarray(outline_xyz, dtype=np.float64) - src["C"]).T
        )[2, :]
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth)
        front = depth > float(FACADE_PROJECTION_NEAR_PLANE_M)
    valid = finite & front
    vertex_count = int(len(uv))
    projection_info = src.get("wireframe_projection_info") or {}
    topology_valid = bool(
        src.get(
            "selection_projection_topology_valid",
            projection_info.get(
                "full_outline_topology_valid",
                bool(np.all(valid) and not _closed_polyline_self_intersects(uv)),
            ),
        )
    )
    if uv.shape[0] == 0 or int(valid.sum()) < 3:
        return {
            "uv": uv,
            "raw_uv": uv_raw,
            "area": 0.0,
            "full_projected_area": 0.0,
            "visible_projected_area": 0.0,
            "coverage_fraction": 0.0,
            "inside_fraction": 0.0,
            "front_fraction": float((finite & front).sum() / max(vertex_count, 1)),
            "full_frame_coverage": False,
            "bbox_area": 0.0,
            "projected_area_fraction": 0.0,
            "min_projected_span_px": 0.0,
            "nondegenerate_projection": False,
            "inside_count": 0,
            "front_count": int((finite & front).sum()),
            "finite_count": int(finite.sum()),
            "vertex_count": vertex_count,
            "sane_span": False,
            "projection_topology_valid": False,
            "uses_near_plane_clipped_projection": (
                uses_near_clipped_projection
            ),
            "score": (0, 0, 0.0, 0.0, int((finite & front).sum()), int(finite.sum()), 0, 0.0),
            "clipped_uv": uv_clipped_fitted,
        }

    inside = (
        valid &
        (uv[:, 0] >= 0.0) &
        (uv[:, 0] <= float(W - 1)) &
        (uv[:, 1] >= 0.0) &
        (uv[:, 1] <= float(H - 1))
    )
    uv_valid = uv[valid]
    span = np.nanmax(uv_valid, axis=0) - np.nanmin(uv_valid, axis=0)
    bbox_area = float(max(span[0], 1.0) * max(span[1], 1.0))
    full_projected_area = float(max(
        _poly_area_abs_2d(full_uv)
        if full_uv.shape[0] >= 3 and np.isfinite(full_uv).all()
        else _poly_area_abs_2d(uv),
        1.0,
    ))
    visible_mask = np.zeros((int(H), int(W)), dtype=np.uint8)
    if topology_valid and uv.shape[0] >= 3 and np.isfinite(uv).all():
        cv2.fillPoly(
            visible_mask,
            [np.round(uv).astype(np.int32).reshape(-1, 1, 2)],
            1,
        )
    elif uv_clipped_fitted.shape[0] >= 3 and np.isfinite(uv_clipped_fitted).all():
        cv2.fillPoly(
            visible_mask,
            [np.round(uv_clipped_fitted).astype(np.int32).reshape(-1, 1, 2)],
            1,
        )
    visible_projected_area = float(visible_mask.sum())
    projected_area_fraction = float(
        visible_projected_area / max(float(W * H), 1.0)
    )
    min_projected_span_px = float(np.nanmin(span))
    coverage_fraction = float(np.clip(
        visible_projected_area / max(full_projected_area, 1.0),
        0.0,
        1.0,
    ))
    inside_count = int(inside.sum())
    inside_fraction = float(inside_count / max(vertex_count, 1))
    front_count = int((finite & front).sum())
    full_frame_coverage = bool(
        topology_valid
        and not uses_near_clipped_projection
        and front_count == vertex_count
        and inside_count == vertex_count
        and vertex_count >= 3
    )
    if full_frame_coverage:
        coverage_fraction = 1.0
    sane_span = (
        float(span[0]) <= float(FACADE_MAX_PROJECTION_SPAN_FACTOR) * float(W) and
        float(span[1]) <= float(FACADE_MAX_PROJECTION_SPAN_FACTOR) * float(H)
    )
    nondegenerate_projection = bool(
        topology_valid
        and sane_span
        and projected_area_fraction >= float(FACADE_SOURCE_MIN_PROJECTED_AREA_FRACTION)
        and min_projected_span_px >= float(FACADE_SOURCE_MIN_PROJECTED_SPAN_PX)
    )
    score = (
        1 if nondegenerate_projection else 0,
        1 if full_frame_coverage else 0,
        float(coverage_fraction),
        float(inside_fraction),
        front_count,
        int(finite.sum()),
        1 if sane_span else 0,
        float(visible_projected_area),
    )
    return {
        "uv": uv,
        "raw_uv": uv_raw,
        "area": float(visible_projected_area),
        "full_projected_area": float(full_projected_area),
        "visible_projected_area": float(visible_projected_area),
        "coverage_fraction": float(coverage_fraction),
        "inside_fraction": float(inside_fraction),
        "front_fraction": float(front_count / max(vertex_count, 1)),
        "full_frame_coverage": full_frame_coverage,
        "bbox_area": float(bbox_area),
        "projected_area_fraction": projected_area_fraction,
        "min_projected_span_px": min_projected_span_px,
        "nondegenerate_projection": nondegenerate_projection,
        "inside_count": inside_count,
        "front_count": front_count,
        "finite_count": int(finite.sum()),
        "vertex_count": vertex_count,
        "sane_span": bool(sane_span),
        "projection_topology_valid": topology_valid,
        "uses_near_plane_clipped_projection": uses_near_clipped_projection,
        "score": score,
        "clipped_uv": uv_clipped_fitted,
    }

def _target_visibility_selection_terms(src, metric):
    available = bool(src.get("target_model_visibility_available", False))
    self_visibility = (
        float(src.get("target_self_visibility_fraction", 0.0))
        if available else 0.0
    )
    frame_coverage = float(metric.get("coverage_fraction", 0.0))
    usable_visibility = float(np.clip(
        self_visibility * frame_coverage,
        0.0,
        1.0,
    ))
    external_available = bool(
        src.get("external_building_occlusion_available", False)
    )
    external_occlusion = float(np.clip(
        src.get("external_building_occlusion_fraction", 0.0)
        if external_available else 0.0,
        0.0,
        1.0,
    ))
    external_visible = float(1.0 - external_occlusion)
    net_visibility = float(np.clip(
        usable_visibility * external_visible,
        0.0,
        1.0,
    ))
    complete_threshold = float(globals().get(
        "FACADE_SOURCE_VISIBILITY_COMPLETE_THRESHOLD",
        0.999,
    ))
    fully_visible = bool(
        available
        and self_visibility >= complete_threshold
        and metric.get("full_frame_coverage", False)
    )
    return {
        "available": available,
        "self_visibility_fraction": self_visibility,
        "usable_visibility_fraction": usable_visibility,
        "external_visibility_available": external_available,
        "external_occlusion_fraction": external_occlusion,
        "external_visible_fraction": external_visible,
        "net_visibility_fraction": net_visibility,
        "fully_visible": fully_visible,
    }


def _facade_source_selection_key(
    src,
    metric,
    outline_xyz,
    legacy_preference=False,
):
    center_dist = float(np.linalg.norm(
        np.asarray(src["camera_xyz"], dtype=np.float64)[:2]
        - np.nanmean(np.asarray(outline_xyz, dtype=np.float64)[:, :2], axis=0)
    ))
    nondegenerate = metric.get("nondegenerate_projection")
    if nondegenerate is None:
        nondegenerate = bool(metric.get("projection_topology_valid", False))
    candidate_usable = bool(src.get("depth_global_candidate_usable", True))
    semantic_visibility = dict(src.get("depth_global_target_visibility") or {})
    visibility = _target_visibility_selection_terms(src, metric)
    external_available = bool(visibility["external_visibility_available"])
    external_fraction = float(visibility["external_occlusion_fraction"])
    # Below the near-total rejection threshold, OSM visibility remains a soft
    # reduction of otherwise usable target-wall visibility. A nearly complete
    # wall with a small obstruction should still beat a clear image containing
    # only a small fraction of the wall.
    return (
        1 if candidate_usable else 0,
        1 if nondegenerate else 0,
        1 if metric.get("projection_topology_valid", False) else 0,
        1 if visibility["available"] else 0,
        float(visibility["net_visibility_fraction"]),
        1 if external_available else 0,
        -external_fraction,
        float(visibility["usable_visibility_fraction"]),
        float(visibility["self_visibility_fraction"]),
        float(semantic_visibility.get("combined_visible_fraction", 1.0)),
        float(semantic_visibility.get("target_support_fraction", 0.0)),
        1 if visibility["fully_visible"] else 0,
        1 if legacy_preference else 0,
        1 if metric.get("full_frame_coverage", False) else 0,
        float(metric.get("coverage_fraction", 0.0)),
        float(metric.get("inside_fraction", 0.0)),
        float(metric.get("front_fraction", 0.0)),
        1 if metric.get("sane_span", False) else 0,
        float(metric.get("visible_projected_area", metric.get("area", 0.0))),
        -center_dist,
    )

def _facade_source_metrics(sources, outline_xyz, image_size):
    return [
        _source_projection_metric(source, outline_xyz, image_size)
        for source in sources
    ]
def _build_selected_facade_source_result(
    sources,
    metrics,
    selection_order,
    selected_index,
    rect_xyz,
    outline_xyz,
    urls,
    source_selection_policy="projected_coverage",
):
    selected_src = sources[int(selected_index)]
    selected_by_legacy_policy = bool(
        source_selection_policy == "legacy_wall_prism"
        and selected_src.get("legacy_wall_prism", False)
    )
    effective_selection_policy = (
        "legacy_wall_prism" if selected_by_legacy_policy else "projected_coverage"
    )
    projected_outline_raw, _ = project_points_world_to_image(
        outline_xyz,
        selected_src["K"],
        selected_src["Rwc"],
        selected_src["C"],
        clip_behind=False,
    )
    visible_outline_override = np.asarray(
        selected_src.get("selection_visible_wall_outline_px", []),
        dtype=np.float64,
    )
    uv_outline_raw = (
        visible_outline_override
        if bool(selected_src.get("selection_uses_near_clipped_projection", False))
        and visible_outline_override.ndim == 2
        and visible_outline_override.shape[0] >= 3
        and visible_outline_override.shape[1] == 2
        and np.isfinite(visible_outline_override).all()
        else projected_outline_raw
    )
    uv_rect_raw, _ = project_points_world_to_image(
        rect_xyz,
        selected_src["K"],
        selected_src["Rwc"],
        selected_src["C"],
        clip_behind=False,
    )
    if (
        uv_outline_raw.shape[0] < 3
        or uv_rect_raw.shape[0] < 4
        or not np.isfinite(uv_outline_raw).all()
        or not np.isfinite(uv_rect_raw).all()
    ):
        return None

    projection_fit_H = np.asarray(
        selected_src.get("wireframe_fit_H", np.eye(3)),
        dtype=np.float64,
    )
    uv_outline = apply_homography(uv_outline_raw, projection_fit_H)
    uv_rect = apply_homography(uv_rect_raw, projection_fit_H)
    selected_fit = dict(selected_src.get("wireframe_fit") or {})
    selected_fit.update({
        "stage": "selected_native_source_before_segmentation",
        "selected_source_index": int(selected_index),
        "selection_target_model_visibility_available": bool(
            selected_src.get("target_model_visibility_available", False)
        ),
        "selection_target_self_visibility_fraction": (
            None
            if selected_src.get("target_self_visibility_fraction") is None
            else float(selected_src["target_self_visibility_fraction"])
        ),
        "selection_target_usable_visibility_fraction": float(
            selected_src.get("target_usable_visibility_fraction", 0.0)
        ),
        "selection_target_net_visibility_fraction": float(
            selected_src.get("target_net_visibility_fraction", 0.0)
        ),
        "selection_target_fully_visible": bool(
            selected_src.get("target_fully_visible", False)
        ),
        "selection_coverage_fraction": float(
            metrics[int(selected_index)].get("coverage_fraction", 0.0)
        ),
        "selection_full_frame_coverage": bool(
            metrics[int(selected_index)].get("full_frame_coverage", False)
        ),
    })

    ranking = []
    for rank, source_index in enumerate(selection_order):
        source = sources[int(source_index)]
        metric = metrics[int(source_index)]
        visibility = _target_visibility_selection_terms(source, metric)
        ranking.append({
            "rank": int(rank + 1),
            "source_index": int(source_index),
            "pano_id": str(source["rec"].get("pano_id", "")),
            "pano_copyright": str(source["rec"].get("copyright", "")),
            "pano_date": source["rec"].get("date"),
            "imagery_provider": str(
                source["rec"].get("imagery_provider", "unknown")
            ),
            "camera_elevation": dict(source.get("camera_elevation") or {}),
            "selected": bool(int(source_index) == int(selected_index)),
            "nondegenerate_projection": bool(metric.get("nondegenerate_projection", False)),
            "target_model_visibility_available": bool(visibility["available"]),
            "target_model_visibility_reason": str(
                source.get("target_model_visibility_reason", "")
            ),
            "target_self_visibility_fraction": (
                float(visibility["self_visibility_fraction"])
                if visibility["available"] else None
            ),
            "target_usable_visibility_fraction": float(
                visibility["usable_visibility_fraction"]
            ),
            "target_net_visibility_fraction": float(
                visibility["net_visibility_fraction"]
            ),
            "external_building_visible_fraction": (
                float(visibility["external_visible_fraction"])
                if visibility["external_visibility_available"] else None
            ),
            "target_fully_visible": bool(visibility["fully_visible"]),
            "target_depth_pixel_count": int(
                source.get("target_depth_pixel_count", 0)
            ),
            "target_visible_pixel_count": int(
                source.get("target_visible_pixel_count", 0)
            ),
            "target_occluded_pixel_count": int(
                source.get("target_occluded_pixel_count", 0)
            ),
            "depth_global_fit_evaluated_before_selection": bool(
                source.get("depth_global_fit_evaluated_before_selection", False)
            ),
            "depth_global_fit_applied": bool(
                source.get("depth_global_fit_applied", False)
            ),
            "depth_global_fit_reason": str(
                source.get("depth_global_fit_reason", "not_evaluated")
            ),
            "depth_global_score_improvement": float(
                source.get("depth_global_score_improvement", 0.0)
            ),
            "depth_global_candidate_usable": bool(
                source.get("depth_global_candidate_usable", True)
            ),
            "depth_global_candidate_rejection_reason": source.get(
                "depth_global_candidate_rejection_reason"
            ),
            "depth_global_target_visibility": dict(
                source.get("depth_global_target_visibility") or {}
            ),
            "depth_global_sam3_skipped": bool(
                source.get("depth_global_sam3_skipped", False)
            ),
            "depth_global_sam3_skip_reason": source.get(
                "depth_global_sam3_skip_reason"
            ),
            "external_building_occlusion_available": bool(
                source.get("external_building_occlusion_available", False)
            ),
            "external_building_occlusion_fraction": (
                float(source.get("external_building_occlusion_fraction", 0.0))
                if source.get("external_building_occlusion_available", False)
                else None
            ),
            "external_building_raw_projection_occlusion_fraction": (
                float(source.get(
                    "external_building_raw_projection_occlusion_fraction",
                    0.0,
                ))
                if source.get("external_building_occlusion_available", False)
                else None
            ),
            "external_building_clear": bool(
                source.get("external_building_clear", False)
            ),
            "external_building_candidate_blockers": list(
                source.get("external_building_candidate_blockers", [])
            ),
            "external_building_candidate_blocker_terrain": dict(
                source.get(
                    "external_building_candidate_blocker_terrain",
                    {},
                )
            ),
            "external_building_blocker_terrain_source": str(
                source.get(
                    "external_building_blocker_terrain_source",
                    "not_available",
                )
            ),
            "full_frame_coverage": bool(metric.get("full_frame_coverage", False)),
            "coverage_fraction": float(metric.get("coverage_fraction", 0.0)),
            "projected_area_fraction": float(metric.get("projected_area_fraction", 0.0)),
            "min_projected_span_px": float(metric.get("min_projected_span_px", 0.0)),
            "inside_outline_vertices": int(metric.get("inside_count", 0)),
            "outline_vertex_count": int(metric.get("vertex_count", 0)),
            "front_outline_vertices": int(metric.get("front_count", 0)),
            "projection_topology_valid": bool(metric.get("projection_topology_valid", False)),
            "uses_near_plane_clipped_projection": bool(
                metric.get("uses_near_plane_clipped_projection", False)
            ),
            "visible_projected_area_px2": float(metric.get("visible_projected_area", 0.0)),
            "full_projected_area_px2": float(metric.get("full_projected_area", 0.0)),
        })

    width, height = selected_src["img"].size
    external_available = bool(
        selected_src.get("external_building_occlusion_available", False)
    )
    external_clear = bool(selected_src.get("external_building_clear", False))
    external_fallback_mask_required = bool(
        external_available and not external_clear
    )
    selected_src["external_building_fallback_mask_required"] = (
        external_fallback_mask_required
    )
    preselection_active = bool(
        selected_src.get("depth_global_fit_evaluated_before_selection", False)
    )
    return {
        "image": selected_src["img"].convert("RGB"),
        "uv_outline": uv_outline,
        "uv_rect": uv_rect,
        "heading": float(selected_src["heading"]),
        "projection_heading": float(
            selected_src.get("projection_heading", selected_src["heading"])
        ),
        "meridian_convergence_deg": float(
            selected_src.get("meridian_convergence_deg", 0.0)
        ),
        "pitch": float(selected_src["pitch"]),
        "fov": float(selected_src["fov"]),
        "K": selected_src["K"],
        "R_wc": selected_src["Rwc"],
        "C": selected_src["C"],
        "camera_xyz": selected_src["camera_xyz"],
        "camera_elevation": dict(selected_src.get("camera_elevation") or {}),
        "rec": selected_src["rec"],
        "urls_fetched": urls,
        "sources": sources,
        "source_mode": (
            "legacy_wall_prism_single_native_source"
            if selected_by_legacy_policy
            else "best_single_native_source_by_target_model_visibility"
        ),
        "selected_source_index": int(selected_index),
        "best_source_index": int(selected_index),
        "source_selection_method": (
            "corrected_depth_global_projection_then_maximum_net_target_visibility"
            if preselection_active and external_available
            else "corrected_depth_global_projection_then_target_model_visibility"
            if preselection_active
            else "usable_projection_then_target_model_visibility_then_legacy_wall_prism"
            if selected_by_legacy_policy
            else "usable_projection_then_target_model_visibility_then_frame_coverage"
        ),
        "source_selection_policy_requested": str(source_selection_policy),
        "source_selection_policy": effective_selection_policy,
        "source_selection_ranking": ranking,
        "processing_image_size": [int(width), int(height)],
        "wireframe_fit": selected_fit,
        "wireframe_fit_overlay": selected_src.get("wireframe_fit_overlay"),
        "uv_outline_before_wireframe_fit": uv_outline_raw,
        "uv_rect_before_wireframe_fit": uv_rect_raw,
        "selected_source_raw_to_processing_image_H": (
            np.eye(3, dtype=float).tolist()
        ),
        "selected_source_raw_to_aligned_image_H": (
            projection_fit_H.astype(float).tolist()
        ),
        "selected_source_corrected_to_aligned_image_H": (
            np.eye(3, dtype=float).tolist()
        ),
        "selected_source_effective_camera_fit": selected_src.get(
            "effective_camera_fit"
        ),
        "selected_candidate_depth_global_fit": selected_src.get(
            "depth_global_fit_result"
        ),
        "selected_candidate_full_model_depth": selected_src.get(
            "depth_global_full_model_depth"
        ),
        "selected_candidate_prefit_semantic_guidance": selected_src.get(
            "depth_global_prefit_semantic_guidance"
        ),
        "selected_candidate_fit_semantic_guidance": selected_src.get(
            "depth_global_fit_semantic_guidance",
            selected_src.get("depth_global_prefit_semantic_guidance"),
        ),
        "selected_candidate_prefit_semantic_metadata": selected_src.get(
            "depth_global_prefit_semantic_metadata"
        ),
        "selected_external_building_removal_mask": (
            selected_src.get("external_building_occlusion_mask")
            if external_fallback_mask_required else None
        ),
        "selected_external_building_target_mask": (
            selected_src.get("external_building_target_mask")
            if external_fallback_mask_required else None
        ),
        "external_building_occlusion": {
            "available": external_available,
            "clear": external_clear,
            "fallback_mask_required": external_fallback_mask_required,
            "occluded_fraction": (
                float(selected_src.get("external_building_occlusion_fraction", 0.0))
                if external_available else None
            ),
            "candidate_blockers": list(
                selected_src.get("external_building_candidate_blockers", [])
            ),
            "candidate_blocker_terrain": dict(
                selected_src.get(
                    "external_building_candidate_blocker_terrain",
                    {},
                )
            ),
            "terrain_source": str(
                selected_src.get(
                    "external_building_blocker_terrain_source",
                    "not_available",
                )
            ),
            "reason": str(
                selected_src.get("external_building_occlusion_reason", "not_evaluated")
            ),
        },
    }

def select_facade_source_from_panos(geom,
                                    pano_candidates,
                                    base_z,
                                    rect_xyz,
                                    outline_xyz,
                                    facade_tag="",
                                    img_size=SV_SIZE,
                                    source_selection_policy="projected_coverage",
                                    meshes_named=None,
                                    target_mesh_names=None,
                                    facade_alignment_mode="wall_only",
                                    candidate_preselection_evaluator=None,
                                    camera_elevation_resolver=None):
    """Fetch candidates, optionally fit/score all of them, then select one."""
    if not geom or not pano_candidates:
        return None

    alignment_mode = str(facade_alignment_mode).strip().lower()
    if alignment_mode not in {"wall_only", "depth_global"}:
        raise ValueError(
            "facade_alignment_mode must be 'wall_only' or 'depth_global'."
        )

    frame = geom["frame"]
    outline_m = np.asarray(geom["outline_m"], dtype=np.float64)
    v_min = float(np.nanmin(outline_m[:, 1]))
    v_max = float(np.nanmax(outline_m[:, 1]))
    v_center = 0.5 * (v_min + v_max)
    fov = float(np.clip(FACADE_GROUP_SOURCE_FOV, FOV_MIN, FOV_MAX))
    sources = []
    urls = []

    for idx, cand in enumerate(pano_candidates):
        rec = cand.get("rec", cand)
        if rec is None:
            continue
        cam, camera_elevation = _camera_xyz_for_record(
            rec,
            base_z,
            camera_elevation_resolver,
        )
        target_u = float(cand.get(
            "u_clamped",
            0.5 * (np.nanmin(outline_m[:, 0]) + np.nanmax(outline_m[:, 0])),
        ))
        use_legacy_framing = bool(
            source_selection_policy == "legacy_wall_prism"
            and cand.get("legacy_wall_prism", False)
            and cand.get("legacy_wall_target_xyz") is not None
        )
        if use_legacy_framing:
            target_xyz = np.asarray(cand["legacy_wall_target_xyz"], dtype=np.float64)
        else:
            target_xyz = frame["to_xyz"](np.array([target_u, v_center], dtype=np.float64))
        dx, dy = target_xyz[0] - cam[0], target_xyz[1] - cam[1]
        dz = target_xyz[2] - cam[2]
        projection_heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
        heading = true_bearing_deg(cam[:2], target_xyz[:2])
        rho = np.hypot(dx, dy)
        pitch = np.degrees(np.arctan2(dz, max(rho, 1e-9)))
        source_fov = fov
        if use_legacy_framing:
            base_seg = np.asarray(cand.get("legacy_wall_base_seg_xy"), dtype=np.float64)
            normal_xy = np.asarray(cand.get("legacy_wall_normal_xy"), dtype=np.float64)
            if base_seg.shape == (2, 2) and normal_xy.shape == (2,):
                source_fov = solve_fov_deg(
                    cam[:2],
                    projection_heading,
                    (base_seg[0], base_seg[1]),
                    normal_xy,
                    buffer_m=SIDE_BUFFER_M,
                    safety_margin_deg=FOV_MARGIN_DEG,
                )

        try:
            img_pil, url, _, _ = fetch_sv_image_by_id(
                rec["pano_id"], heading, pitch, source_fov, API_KEY, size=img_size
            )
        except Exception as exc:
            print(f"[{facade_tag}] skip pano {idx}: fetch failed ({exc})")
            continue

        K, R_wc, C = build_pose_from_heading_pitch(
            cam,
            projection_heading,
            pitch,
            img_size=img_size,
            fov_deg=source_fov,
        )
        target_visibility = evaluate_target_wall_model_visibility(
            meshes_named=meshes_named,
            target_mesh_names=target_mesh_names,
            K=K,
            R_wc=R_wc,
            C=C,
            image_size=img_pil.size,
        )
        urls.append(url)
        sources.append({
            "rec": rec,
            "img": img_pil,
            "url": url,
            "K": K,
            "Rwc": R_wc,
            "C": C,
            "camera_xyz": cam,
            "camera_elevation": camera_elevation,
            "heading": float(heading),
            "projection_heading": float(projection_heading),
            "meridian_convergence_deg": float(
                wrap_delta_deg(heading, projection_heading)
            ),
            "pitch": float(pitch),
            "fov": float(source_fov),
            "u_clamped": float(target_u),
            "candidate_selection_origin": str(cand.get("selection_origin", "unspecified")),
            "candidate_selection_origins": list(cand.get("selection_origins", [])),
            "candidate_forward_m": float(cand.get("forward_m", 0.0)),
            "candidate_frontality": float(cand.get("frontality", 0.0)),
            "candidate_is_fallback": bool(cand.get("is_fallback", False)),
            "legacy_wall_prism": bool(cand.get("legacy_wall_prism", False)),
            "legacy_wall_framing": bool(use_legacy_framing),
            "source_index": int(idx),
            **target_visibility,
        })

    if not sources:
        return None

    sources.sort(key=lambda s: s["u_clamped"])
    for source in sources:
        source["wireframe_fit"] = {
            "applied": False,
            "reason": "not_selected_for_alignment",
            "stage": "candidate_ranking_before_alignment",
        }
        source["wireframe_fit_H"] = np.eye(3, dtype=np.float64)
        source["wireframe_fit_applied_to_source"] = False
        source["wireframe_fit_overlay"] = None
        source["effective_camera_fit"] = {
            "attempted": False,
            "accepted": False,
            "reason": "not_selected_for_alignment",
        }

    if candidate_preselection_evaluator is not None:
        try:
            candidate_preselection_evaluator(sources)
        except Exception as exc:
            print(
                f"[{facade_tag}] candidate preselection evaluation failed; "
                f"using raw source ranking: {exc}"
            )

    evaluated_candidates = [
        source for source in sources
        if bool(source.get("depth_global_fit_evaluated_before_selection", False))
    ]
    if (
        evaluated_candidates
        and not any(bool(source.get("depth_global_candidate_usable", True))
                    for source in evaluated_candidates)
    ):
        print(
            f"[{facade_tag}] every projected source candidate is OSM- or "
            "semantic-occlusion rejected; leaving this facade unresolved."
        )
        return None

    image_size = sources[0]["img"].size
    metrics = _facade_source_metrics(sources, outline_xyz, image_size)

    def source_order_key(idx):
        source = sources[int(idx)]
        metric = metrics[int(idx)]
        valid_legacy = bool(
            source_selection_policy == "legacy_wall_prism"
            and source.get("legacy_wall_prism", False)
            and metric.get("projection_topology_valid", False)
            and metric.get("nondegenerate_projection", False)
        )
        return _facade_source_selection_key(
            source,
            metric,
            outline_xyz,
            legacy_preference=valid_legacy,
        )

    selection_order = sorted(range(len(sources)), key=source_order_key, reverse=True)
    selected_idx = int(selection_order[0])
    for rank, idx in enumerate(selection_order):
        src = sources[int(idx)]
        metric = metrics[int(idx)]
        src["selected_for_processing"] = bool(int(idx) == selected_idx)
        src["source_selection_rank"] = int(rank + 1)
        src["projection_score"] = tuple(metric["score"])
        src["projected_area_px2"] = float(metric["area"])
        src["projected_area_fraction"] = float(metric.get("projected_area_fraction", 0.0))
        src["min_projected_span_px"] = float(metric.get("min_projected_span_px", 0.0))
        src["nondegenerate_projection"] = bool(metric.get("nondegenerate_projection", False))
        src["uses_near_plane_clipped_projection"] = bool(
            metric.get("uses_near_plane_clipped_projection", False)
        )
        src["projected_coverage_fraction"] = float(metric.get("coverage_fraction", 0.0))
        src["full_frame_coverage"] = bool(metric.get("full_frame_coverage", False))
        visibility_terms = _target_visibility_selection_terms(src, metric)
        src["target_usable_visibility_fraction"] = float(
            visibility_terms["usable_visibility_fraction"]
        )
        src["target_net_visibility_fraction"] = float(
            visibility_terms["net_visibility_fraction"]
        )
        src["external_building_visible_fraction"] = (
            float(visibility_terms["external_visible_fraction"])
            if visibility_terms["external_visibility_available"] else None
        )
        src["target_fully_visible"] = bool(visibility_terms["fully_visible"])
        src["inside_outline_vertices"] = int(metric["inside_count"])
        src["front_outline_vertices"] = int(metric["front_count"])

    if not bool(metrics[selected_idx].get("projection_topology_valid", False)):
        print(
            f"[{facade_tag}] no Street View source has a valid in-front facade projection "
            "- skipping malformed source geometry."
        )
        return None
    if not bool(metrics[selected_idx].get("nondegenerate_projection", False)):
        print(
            f"[{facade_tag}] Street View candidates see the facade only edge-on "
            "- skipping an unusable line-like texture source."
        )
        return None

    selected_source = sources[selected_idx]
    if alignment_mode == "wall_only":
        selected_source["wireframe_fit"] = {
            "applied": False,
            "reason": "selected_wall_only_fit_not_applied",
            "stage": "selected_source_alignment",
        }
        selected_source["effective_camera_fit"] = {
            "attempted": False,
            "accepted": False,
            "reason": "selected_wall_only_fit_not_applied",
        }
        _fit_raw_source_wireframe(
            selected_source,
            outline_xyz,
            facade_tag,
            selected_idx,
        )
    else:
        selected_source["wireframe_fit"].update({
            "reason": "skipped_for_depth_global_alignment",
            "stage": "selected_source_alignment",
        })
        selected_source["effective_camera_fit"]["reason"] = (
            "skipped_for_depth_global_alignment"
        )

    chosen_metric = metrics[selected_idx]
    print(
        f"[{facade_tag}] selected native source {selected_idx:02d} | "
        f"target_self_visibility="
        f"{100.0 * float(sources[selected_idx].get('target_self_visibility_fraction') or 0.0):.1f}% | "
        f"target_usable_visibility="
        f"{100.0 * float(sources[selected_idx].get('target_usable_visibility_fraction', 0.0)):.1f}% | "
        f"full_coverage={bool(chosen_metric.get('full_frame_coverage', False))} | "
        f"coverage={100.0 * float(chosen_metric.get('coverage_fraction', 0.0)):.1f}% | "
        f"visible_area={float(chosen_metric.get('visible_projected_area', 0.0)):.0f}px2"
    )
    return _build_selected_facade_source_result(
        sources=sources,
        metrics=metrics,
        selection_order=selection_order,
        selected_index=selected_idx,
        rect_xyz=rect_xyz,
        outline_xyz=outline_xyz,
        urls=urls,
        source_selection_policy=source_selection_policy,
    )


def fetch_single_wall_source(pano_id, cam, wall_quad_xyz,
                             heading, pitch, fov_deg,
                             img_size=SV_SIZE,
                             coverage_pts_xyz=None):
    """Fetch one native Street View image, reframing once when possible."""
    urls_fetched = []
    coverage_pts = np.asarray(
        coverage_pts_xyz if coverage_pts_xyz is not None else wall_quad_xyz,
        dtype=float,
    )

    request_heading = grid_heading_to_true_deg(cam[:2], heading)
    image, url, _, _ = fetch_sv_image_by_id(
        pano_id,
        request_heading,
        pitch,
        fov_deg,
        API_KEY,
        size=img_size,
    )
    urls_fetched.append(url)
    K, R_wc, C = build_pose_from_heading_pitch(
        cam,
        heading,
        pitch,
        img_size=img_size,
        fov_deg=fov_deg,
    )
    uv, _ = project_points_world_to_image(
        coverage_pts,
        K,
        R_wc,
        C,
        clip_behind=False,
    )
    width, height = image.size

    if not uv_inside_image(uv, width, height, COVER_MARGIN_PX):
        yaws, pitches = yaw_pitch_of_points(cam, coverage_pts)
        yaw_center, yaw_span, _yaw_min, _yaw_max = circular_span(yaws)
        pitch_center = 0.5 * (pitches.min() + pitches.max())
        pitch_span = float(pitches.max() - pitches.min())
        required_fov = max(yaw_span, pitch_span) + 2 * ANGLE_MARGIN_DEG

        if required_fov <= FOV_MAX + 1e-6:
            fov_deg = float(np.clip(required_fov, FOV_MIN, FOV_MAX))
            heading = float(yaw_center)
            request_heading = grid_heading_to_true_deg(cam[:2], heading)
            pitch = float(pitch_center)
            image, url, _, _ = fetch_sv_image_by_id(
                pano_id,
                request_heading,
                pitch,
                fov_deg,
                API_KEY,
                size=img_size,
            )
            urls_fetched.append(url)
            K, R_wc, C = build_pose_from_heading_pitch(
                cam,
                heading,
                pitch,
                img_size=img_size,
                fov_deg=fov_deg,
            )
            uv, _ = project_points_world_to_image(
                coverage_pts,
                K,
                R_wc,
                C,
                clip_behind=False,
            )

    return (
        image,
        uv,
        request_heading,
        heading,
        pitch,
        fov_deg,
        K,
        R_wc,
        C,
        urls_fetched,
    )
def save_overlay_matplotlib(img_pil: Image.Image, uv: np.ndarray, out_path: str, title: str = ""):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] < 3 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        return
    if not HAVE_MATPLOTLIB:
        save_with_overlay(img_pil.convert("RGBA"), uv, out_path)
        return
    W, H = img_pil.size
    fig, ax = plt.subplots(figsize=(W/100.0, H/100.0), dpi=100)
    ax.imshow(img_pil)
    poly = MplPolygon(uv[:, :2], closed=True, facecolor=(1, 0, 0, 0.25), edgecolor=(1, 0, 0, 0.95), linewidth=2.0)
    ax.add_patch(poly)
    if title:
        ax.set_title(title)
    ax.axis('off')
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)
