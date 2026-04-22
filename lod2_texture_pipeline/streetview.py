# -*- coding: utf-8 -*-
"""Street View pano search and wall-to-pano geometric selection."""

import numpy as np
import requests
from requests.exceptions import RequestException
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from PIL import Image
from io import BytesIO

from .config import API_KEY, BACK_EPS, EXTRUSION_LEN_XY, FOV_MARGIN_DEG, FOV_MAX, FOV_MIN, GRID_N, GRID_OFFSET_M, SV_SIZE
from .utils import safe_unit
def get_nearest_pano(lat, lon, api_key, radius=30, timeout=10, verbose=False):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "radius": int(radius), "key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except RequestException as e:
        if verbose:
            print(f"[SV] HTTP error @ ({lat:.6f},{lon:.6f}) r={radius}: {e}")
        return None
    if resp.status_code != 200:
        if verbose:
            print(f"[SV] HTTP {resp.status_code} @ ({lat:.6f},{lon:.6f}) r={radius}")
        return None
    try:
        data = resp.json()
    except ValueError:
        if verbose:
            print(f"[SV] Non-JSON response @ ({lat:.6f},{lon:.6f}) r={radius}")
        return None
    if data.get("status") != "OK":
        if verbose:
            print(f"[SV] status={data.get('status')} msg={data.get('error_message','')}")
        return None
    return data

def fetch_sv_image_by_id(pano_id, heading, pitch, fov, api_key, size=SV_SIZE, timeout=20):
    url = ("https://maps.googleapis.com/maps/api/streetview"
           f"?pano={pano_id}&size={size}&heading={heading:.4f}&pitch={pitch:.4f}&fov={fov:.4f}&key={api_key}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    return Image.open(BytesIO(raw)).convert("RGB"), url, raw, resp.headers.get("Content-Type", "")

def build_search_grid_and_collect_panos(base_line_geoms, transformer, back_tx, api_key, offset=20, n=10, verbose=False):
    footprint = unary_union(base_line_geoms)
    zone      = footprint.buffer(offset)

    minx, miny, maxx, maxy = zone.bounds
    side   = max(maxx - minx, maxy - miny)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    sq_minx, sq_miny = cx - side/2, cy - side/2
    dx = side / n

    print(" Searching for images around the building...")

    seen = set()
    pano_records = []
    kept = 0

    for i in range(n + 1):
        for j in range(n + 1):
            gx = sq_minx + i * dx
            gy = sq_miny + j * dx
            lon, lat = transformer.transform(gx, gy)
            radius_m = max(30.0, 1.2 * dx)

            meta = get_nearest_pano(lat, lon, api_key, radius=radius_m, verbose=verbose)
            if not meta:
                continue

            plc = meta["location"]
            key = (round(plc["lat"], 6), round(plc["lng"], 6))
            if key in seen:
                continue

            ux, uy = back_tx.transform(plc["lng"], plc["lat"])
            if not zone.contains(Point(ux, uy)):
                continue

            kept += 1
            seen.add(key)
            pano_records.append({
                "utm": (float(ux), float(uy)),
                "lat": float(plc["lat"]),
                "lng": float(plc["lng"]),
                "pano_id": meta["pano_id"]
            })

    print(f"✅ {kept} images found")
    return pano_records

def bearing_deg(src_xy, dst_xy):
    dx = dst_xy[0] - src_xy[0]
    dy = dst_xy[1] - src_xy[1]
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

def wrap_delta_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def solve_fov_deg(cam_xy, heading_deg, base_seg_xy, n_xy_unit, buffer_m, safety_margin_deg=FOV_MARGIN_DEG):
    b1_xy, b2_xy = base_seg_xy
    az1 = bearing_deg(cam_xy, b1_xy)
    az2 = bearing_deg(cam_xy, b2_xy)
    d1 = abs(wrap_delta_deg(az1, heading_deg))
    d2 = abs(wrap_delta_deg(az2, heading_deg))
    half_core = max(d1, d2)
    center_xy = 0.5 * (b1_xy + b2_xy)
    fwd = abs(np.dot(center_xy - cam_xy, n_xy_unit))
    fwd = max(fwd, 0.5)
    half_buf = np.degrees(np.arctan2(buffer_m, fwd))
    half_total = half_core + half_buf + safety_margin_deg
    return float(np.clip(2.0 * half_total, FOV_MIN, FOV_MAX))

def compute_wall_normals_from_wall_faces(corners, wall_edges, id_to_idx):
    wall_normals, centers_xyz, base_segs = [], [], []
    for i in range(len(wall_edges)):
        (s1, t1) = wall_edges[i]
        (s2, t2) = wall_edges[(i + 1) % len(wall_edges)]
        if any(nid not in id_to_idx for nid in [s1, t1, s2, t2]):
            wall_normals.append(np.array([0,0,0])); centers_xyz.append(None); base_segs.append(None); continue
        p1 = corners[id_to_idx[s1]]; p2 = corners[id_to_idx[t1]]
        p3 = corners[id_to_idx[s2]]; p4 = corners[id_to_idx[t2]]
        def by_z(a,b): return (a,b) if a[2] <= b[2] else (b,a)
        b1, t1p = by_z(p1, p2); b2, t2p = by_z(p3, p4)
        v1 = b2 - b1; v2 = t1p - b1
        normal = np.cross(v1, v2)
        unit_normal = safe_unit(normal)
        center_xy = 0.25*(b1[:2] + b2[:2] + t1p[:2] + t2p[:2])
        center_z  = 0.25*(b1[2] + b2[2] + t1p[2] + t2p[2])
        wall_normals.append(unit_normal)
        centers_xyz.append(np.array([center_xy[0], center_xy[1], center_z], float))
        base_segs.append((b1[:2].copy(), b2[:2].copy()))
    return wall_normals, centers_xyz, base_segs

def select_pano_per_wall_using_prism_base(wall_edges, wall_normals, corners, id_to_idx, pano_records, L_out=EXTRUSION_LEN_XY, back_eps=BACK_EPS):
    selected_xy, selected_recs = [], []
    n = len(wall_edges)
    for i in range(n):
        (s1, t1) = wall_edges[i]
        (s2, t2) = wall_edges[(i + 1) % n]
        if any(nid not in id_to_idx for nid in [s1, t1, s2, t2]):
            selected_xy.append(None); selected_recs.append(None); continue
        p1a = corners[id_to_idx[s1]]; p1b = corners[id_to_idx[t1]]
        p2a = corners[id_to_idx[s2]]; p2b = corners[id_to_idx[t2]]
        def by_z(a,b): return (a,b) if a[2] <= b[2] else (b,a)
        b1, t1p = by_z(p1a, p1b); b2, t2p = by_z(p2a, p2b)
        b1_xy = b1[:2]; b2_xy = b2[:2]
        base_dir = b2_xy - b1_xy
        base_len = np.linalg.norm(base_dir)
        if base_len < 1e-9:
            selected_xy.append(None); selected_recs.append(None); continue
        base_dir /= base_len
        n_xy = wall_normals[i][:2].copy()
        n_xy = safe_unit(n_xy)
        if np.linalg.norm(n_xy) < 1e-9:
            n_xy = np.array([-base_dir[1], base_dir[0]])
        q0 = b1_xy - back_eps * n_xy
        q1 = b2_xy - back_eps * n_xy
        q2 = b2_xy + L_out   * n_xy
        q3 = b1_xy + L_out   * n_xy
        poly = Polygon([tuple(q0), tuple(q1), tuple(q2), tuple(q3)])
        if not poly.is_valid: poly = poly.buffer(0)
        if poly.is_empty:
            selected_xy.append(None); selected_recs.append(None); continue
        poly_prep = prep(poly)
        best_rec, best_key = None, None
        for rec in pano_records:
            px, py = rec["utm"]
            if not poly_prep.covers(Point(px, py)):
                continue
            vec = np.array([px, py]) - 0.5*(b1_xy + b2_xy)
            forward = float(np.dot(vec, n_xy))
            if forward < 0:
                continue
            lateral = abs(float(np.dot(vec, base_dir)))
            dist    = float(np.linalg.norm(vec))
            key = (lateral, forward, dist)
            if (best_key is None) or (key < best_key):
                best_key, best_rec = key, rec
        if best_rec is not None:
            selected_xy.append(best_rec["utm"]); selected_recs.append(best_rec)
        else:
            selected_xy.append(None); selected_recs.append(None)
    return selected_xy, selected_recs
