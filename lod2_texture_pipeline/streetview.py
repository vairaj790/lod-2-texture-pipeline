# -*- coding: utf-8 -*-
"""Street View pano search and wall-to-pano geometric selection."""

import hashlib
import json
import math
import numpy as np
import requests
from requests.exceptions import RequestException
from pyproj import Geod
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from PIL import Image
from io import BytesIO
from pathlib import Path

from .config import (
    API_KEY,
    BACK_EPS,
    ENABLE_STREETVIEW_CACHE,
    EXTRUSION_LEN_XY,
    FACADE_GROUP_CANDIDATE_MAX_LATERAL_OUTSIDE_M,
    FACADE_GROUP_CANDIDATE_MIN_FORWARD_M,
    FACADE_GROUP_CANDIDATE_TARGET_SPACING_M,
    FACADE_GROUP_MAX_CANDIDATE_PANOS,
    FACADE_GROUP_RECOVERY_FORWARD_DISTANCES_M,
    FACADE_GROUP_RECOVERY_LATERAL_PAD_M,
    FACADE_GROUP_RECOVERY_MIN_FRONTALITY,
    FACADE_GROUP_RECOVERY_QUERY_RADIUS_M,
    FOV_MARGIN_DEG,
    FOV_MAX,
    FOV_MIN,
    GRID_N,
    GRID_OFFSET_M,
    STREETVIEW_GOOGLE_IMAGERY_ONLY,
    STREETVIEW_CACHE_DIR,
    STREETVIEW_SEARCH_SOURCE,
    SV_SIZE,
    transformer as projected_to_geographic,
)
from .utils import safe_unit


_GEOD = Geod(ellps="GRS80")


def _cache_root() -> Path:
    return Path(STREETVIEW_CACHE_DIR)


def _cache_key(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_cache(path: Path):
    if not ENABLE_STREETVIEW_CACHE:
        return None
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_cache(path: Path, data) -> None:
    if not ENABLE_STREETVIEW_CACHE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, sort_keys=True)
        tmp.replace(path)
    except Exception:
        return


def _read_bytes_cache(path: Path):
    if not ENABLE_STREETVIEW_CACHE:
        return None
    try:
        if not path.exists():
            return None
        return path.read_bytes()
    except Exception:
        return None


def _write_bytes_cache(path: Path, data: bytes) -> None:
    if not ENABLE_STREETVIEW_CACHE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception:
        return


def panorama_is_google_owned(metadata) -> bool:
    """Return whether Street View metadata attributes a panorama to Google."""
    copyright_text = str((metadata or {}).get("copyright", "")).casefold()
    return "google" in copyright_text


def _panorama_metadata_allowed(metadata, google_only=None) -> bool:
    if not isinstance(metadata, dict) or metadata.get("status") != "OK":
        return False
    require_google = (
        bool(STREETVIEW_GOOGLE_IMAGERY_ONLY)
        if google_only is None
        else bool(google_only)
    )
    return not require_google or panorama_is_google_owned(metadata)


def _pano_record_from_metadata(metadata, back_tx, selection_origin=None):
    location = metadata.get("location") or {}
    lng = float(location["lng"])
    lat = float(location["lat"])
    ux, uy = back_tx.transform(lng, lat)
    record = {
        "utm": (float(ux), float(uy)),
        "lat": lat,
        "lng": lng,
        "pano_id": str(metadata["pano_id"]),
        "copyright": str(metadata.get("copyright", "")),
        "date": metadata.get("date"),
        "imagery_provider": (
            "google" if panorama_is_google_owned(metadata) else "non_google"
        ),
        "search_source": str(STREETVIEW_SEARCH_SOURCE),
    }
    if selection_origin is not None:
        record["selection_origin"] = str(selection_origin)
    return record


def true_bearing_deg(src_xy, dst_xy, transformer=projected_to_geographic):
    """Geodetic bearing between projected points, clockwise from true north."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    lon1, lat1 = transformer.transform(float(src[0]), float(src[1]))
    lon2, lat2 = transformer.transform(float(dst[0]), float(dst[1]))
    azimuth, _back_azimuth, _distance = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(azimuth % 360.0)


def grid_heading_to_true_deg(
    cam_xy,
    grid_heading_deg,
    transformer=projected_to_geographic,
):
    """Convert a projected-grid heading to the true-north heading Google uses."""
    angle = math.radians(float(grid_heading_deg))
    origin = np.asarray(cam_xy, dtype=np.float64)
    probe = origin + 10.0 * np.array([math.sin(angle), math.cos(angle)])
    return true_bearing_deg(origin, probe, transformer=transformer)


def get_nearest_pano(
    lat,
    lon,
    api_key,
    radius=30,
    timeout=10,
    verbose=False,
    source=None,
    google_only=None,
):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    search_source = str(
        STREETVIEW_SEARCH_SOURCE if source is None else source
    ).strip().lower()
    params = {"location": f"{lat},{lon}", "radius": int(radius), "key": api_key}
    if search_source:
        params["source"] = search_source
    cache_payload = {
        "kind": "metadata",
        "location": params["location"],
        "radius": int(radius),
        "source": search_source,
    }
    cache_path = _cache_root() / "metadata" / f"{_cache_key(cache_payload)}.json"
    cached = _read_json_cache(cache_path)
    if cached is not None:
        data = cached.get("data") if isinstance(cached, dict) else None
        if _panorama_metadata_allowed(data, google_only=google_only):
            return data
        if (
            verbose
            and isinstance(data, dict)
            and data.get("status") == "OK"
            and not panorama_is_google_owned(data)
        ):
            print(
                f"[SV] rejected non-Google panorama {data.get('pano_id', '')} "
                f"({data.get('copyright', 'unknown copyright')})"
            )
        return None

    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except RequestException as e:
        if verbose:
            print(
                f"[SV] HTTP error @ ({lat:.6f},{lon:.6f}) r={radius}: "
                f"{type(e).__name__}"
            )
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
    if data.get("status") in {"OK", "ZERO_RESULTS"}:
        _write_json_cache(cache_path, {
            "url": url,
            "params": {k: v for k, v in params.items() if k != "key"},
            "data": data,
        })
    if data.get("status") != "OK":
        if verbose:
            print(f"[SV] status={data.get('status')} msg={data.get('error_message','')}")
        return None
    if not _panorama_metadata_allowed(data, google_only=google_only):
        if verbose:
            print(
                f"[SV] rejected non-Google panorama {data.get('pano_id', '')} "
                f"({data.get('copyright', 'unknown copyright')})"
            )
        return None
    return data

def fetch_sv_image_by_id(pano_id, heading, pitch, fov, api_key, size=SV_SIZE, timeout=20):
    url = ("https://maps.googleapis.com/maps/api/streetview"
           f"?pano={pano_id}&size={size}&heading={heading:.4f}&pitch={pitch:.4f}&fov={fov:.4f}&key={api_key}")
    cache_payload = {
        "kind": "image",
        "pano": str(pano_id),
        "size": str(size),
        "heading": round(float(heading), 4),
        "pitch": round(float(pitch), 4),
        "fov": round(float(fov), 4),
    }
    key = _cache_key(cache_payload)
    image_path = _cache_root() / "images" / f"{key}.jpg"
    meta_path = _cache_root() / "images" / f"{key}.json"
    raw = _read_bytes_cache(image_path)
    if raw is not None:
        meta = _read_json_cache(meta_path) or {}
        content_type = str(meta.get("content_type", "image/jpeg"))
        return Image.open(BytesIO(raw)).convert("RGB"), url, raw, content_type

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    content_type = resp.headers.get("Content-Type", "")
    if raw:
        _write_bytes_cache(image_path, raw)
        _write_json_cache(meta_path, {
            "params": cache_payload,
            "content_type": content_type,
        })
    return Image.open(BytesIO(raw)).convert("RGB"), url, raw, content_type

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

            rec = _pano_record_from_metadata(meta, back_tx)
            ux, uy = rec["utm"]
            if not zone.contains(Point(ux, uy)):
                continue

            kept += 1
            seen.add(key)
            pano_records.append(rec)

    print(f"Found {kept} images")
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

def _pano_alignment_key(lateral_center, forward, dist):
    # Preserve the original production selector's deterministic priority.
    return (float(abs(lateral_center)), float(forward), float(dist))

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
            key = _pano_alignment_key(lateral, forward, dist)
            if (best_key is None) or (key < best_key):
                best_key, best_rec = key, rec
        if best_rec is not None:
            selected_xy.append(best_rec["utm"]); selected_recs.append(best_rec)
        else:
            selected_xy.append(None); selected_recs.append(None)
    return selected_xy, selected_recs

def select_pano_for_facade_group(geom, pano_records, L_out=EXTRUSION_LEN_XY, back_eps=BACK_EPS):
    """
    Select a Street View pano for an arbitrary grouped facade.

    This mirrors the per-wall prism logic, but uses the grouped facade frame:
    - u-axis is the grouped facade horizontal direction
    - normal_xy is the outward facade normal
    - outline_m gives the horizontal span of the whole grouped wall

    Ranking prefers a pano centered in front of the grouped facade, then a
    pano with smaller off-axis angle, then closer perpendicular/total distance.
    """
    if not geom or not pano_records:
        return None, None

    frame = geom["frame"]
    outline = np.asarray(geom["outline_xyz"], dtype=np.float64)
    outline_m = np.asarray(geom["outline_m"], dtype=np.float64)
    n_xy = safe_unit(np.asarray(frame["normal_xy"], dtype=np.float64)[:2])
    u_dir = safe_unit(np.asarray(frame["u_dir"], dtype=np.float64)[:2])
    if np.linalg.norm(n_xy) < 1e-9 or np.linalg.norm(u_dir) < 1e-9:
        return None, None

    center_xy = np.nanmean(outline[:, :2], axis=0)
    u_vals = outline_m[:, 0]
    u_min = float(np.nanmin(u_vals))
    u_max = float(np.nanmax(u_vals))
    u_center = 0.5 * (u_min + u_max)

    origin = np.asarray(frame["origin"], dtype=np.float64)
    u_dir3 = np.asarray(frame["u_dir"], dtype=np.float64)
    base_left_xy = (origin + float(u_min) * u_dir3)[:2]
    base_right_xy = (origin + float(u_max) * u_dir3)[:2]
    q0 = base_left_xy - back_eps * n_xy
    q1 = base_right_xy - back_eps * n_xy
    q2 = base_right_xy + L_out * n_xy
    q3 = base_left_xy + L_out * n_xy
    prism = Polygon([tuple(q0), tuple(q1), tuple(q2), tuple(q3)])
    if not prism.is_valid:
        prism = prism.buffer(0)
    prism_prep = prep(prism) if not prism.is_empty else None

    best_rec = None
    best_key = None
    fallback_rec = None
    fallback_key = None

    for rec in pano_records:
        px, py = rec["utm"]
        pano_xy = np.array([float(px), float(py)], dtype=np.float64)
        vec = pano_xy - center_xy
        forward = float(np.dot(vec, n_xy))
        if forward <= 0:
            continue

        rel3 = np.array([pano_xy[0], pano_xy[1], origin[2]], dtype=np.float64) - origin
        u = float(np.dot(rel3, u_dir3))
        lateral_center = abs(u - u_center)
        lateral_outside = max(u_min - u, 0.0, u - u_max)
        dist = float(np.linalg.norm(vec))
        key = _pano_alignment_key(lateral_center, forward, dist)
        fallback_key_here = (
            float(lateral_outside),
            *key,
        )

        if fallback_key is None or fallback_key_here < fallback_key:
            fallback_key = fallback_key_here
            fallback_rec = rec

        if prism_prep is not None and not prism_prep.covers(Point(px, py)):
            continue
        if best_key is None or key < best_key:
            best_key = key
            best_rec = rec

    chosen = best_rec if best_rec is not None else fallback_rec
    if chosen is None:
        return None, None
    return chosen["utm"], chosen


def _facade_candidate_from_record(geom, rec, selection_origin):
    frame = geom["frame"]
    outline = np.asarray(geom["outline_xyz"], dtype=np.float64)
    outline_m = np.asarray(geom["outline_m"], dtype=np.float64)
    n_xy = safe_unit(np.asarray(frame["normal_xy"], dtype=np.float64)[:2])
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u_dir3 = np.asarray(frame["u_dir"], dtype=np.float64)
    u_vals = outline_m[:, 0]
    u_min = float(np.nanmin(u_vals))
    u_max = float(np.nanmax(u_vals))
    u_center = 0.5 * (u_min + u_max)
    center_xy = np.nanmean(outline[:, :2], axis=0)

    px, py = rec["utm"]
    pano_xy = np.array([float(px), float(py)], dtype=np.float64)
    vec = pano_xy - center_xy
    dist = float(np.linalg.norm(vec))
    forward = float(np.dot(vec, n_xy))
    rel3 = np.array([pano_xy[0], pano_xy[1], origin[2]], dtype=np.float64) - origin
    u = float(np.dot(rel3, u_dir3))
    u_clamped = float(np.clip(u, u_min, u_max))
    lateral_outside = max(u_min - u, 0.0, u - u_max)
    off_axis_deg = math.degrees(math.atan2(abs(u - u_clamped), max(forward, 0.5)))
    frontality = max(0.0, min(1.0, forward / max(dist, 1e-9)))
    return {
        "rec": rec,
        "utm": rec["utm"],
        "u": u,
        "u_clamped": u_clamped,
        "forward_m": forward,
        "dist_m": dist,
        "lateral_center_m": float(abs(u - u_center)),
        "lateral_outside_m": float(lateral_outside),
        "off_axis_deg": float(off_axis_deg),
        "frontality": float(frontality),
        "selection_origin": str(selection_origin),
        "selection_origins": [str(selection_origin)],
        "is_fallback": False,
    }


def select_legacy_panos_for_wall_quads(wall_quads,
                                        wall_normals,
                                        pano_records,
                                        L_out=EXTRUSION_LEN_XY,
                                        back_eps=BACK_EPS):
    """Return the original production prism winner for each wall fragment."""
    selected = []
    seen = set()
    quads = [] if wall_quads is None else wall_quads
    normals = [] if wall_normals is None else wall_normals
    for wall_quad, wall_normal in zip(quads, normals):
        quad = np.asarray(wall_quad, dtype=np.float64)
        if quad.shape != (4, 3) or not np.isfinite(quad).all():
            continue
        b1_xy = quad[0, :2]
        b2_xy = quad[1, :2]
        base_dir = b2_xy - b1_xy
        base_len = float(np.linalg.norm(base_dir))
        if base_len < 1e-9:
            continue
        base_dir /= base_len
        n_xy = safe_unit(np.asarray(wall_normal, dtype=np.float64)[:2])
        if np.linalg.norm(n_xy) < 1e-9:
            n_xy = np.array([-base_dir[1], base_dir[0]], dtype=np.float64)

        q0 = b1_xy - float(back_eps) * n_xy
        q1 = b2_xy - float(back_eps) * n_xy
        q2 = b2_xy + float(L_out) * n_xy
        q3 = b1_xy + float(L_out) * n_xy
        prism = Polygon([tuple(q0), tuple(q1), tuple(q2), tuple(q3)])
        if not prism.is_valid:
            prism = prism.buffer(0)
        if prism.is_empty:
            continue
        prism_prep = prep(prism)

        wall_center = 0.5 * (b1_xy + b2_xy)
        best_rec = None
        best_key = None
        for rec in pano_records:
            px, py = rec["utm"]
            if not prism_prep.covers(Point(px, py)):
                continue
            vec = np.array([float(px), float(py)], dtype=np.float64) - wall_center
            forward = float(np.dot(vec, n_xy))
            if forward < 0.0:
                continue
            lateral = abs(float(np.dot(vec, base_dir)))
            dist = float(np.linalg.norm(vec))
            key = _pano_alignment_key(lateral, forward, dist)
            if best_key is None or key < best_key:
                best_key = key
                best_rec = rec

        if best_rec is None:
            continue
        pano_id = str(best_rec.get("pano_id", ""))
        if pano_id in seen:
            continue
        seen.add(pano_id)
        # Preserve the exact wall geometry used by the original production
        # path so grouped processing can reproduce its camera framing later.
        selected_rec = dict(best_rec)
        selected_rec["_legacy_wall_target_xyz"] = np.nanmean(quad, axis=0).tolist()
        selected_rec["_legacy_wall_base_seg_xy"] = np.vstack([b1_xy, b2_xy]).tolist()
        selected_rec["_legacy_wall_normal_xy"] = np.asarray(n_xy, dtype=np.float64).tolist()
        selected.append(selected_rec)
    return selected


def facade_group_candidates_need_recovery(
    candidates,
    min_forward_m=FACADE_GROUP_CANDIDATE_MIN_FORWARD_M,
    min_frontality=FACADE_GROUP_RECOVERY_MIN_FRONTALITY,
):
    return not any(
        not bool(candidate.get("is_fallback", False))
        and float(candidate.get("forward_m", 0.0)) >= float(min_forward_m)
        and float(candidate.get("frontality", 0.0)) >= float(min_frontality)
        for candidate in candidates
    )


def discover_recovery_panos_for_facade_group(
    geom,
    transformer,
    back_tx,
    api_key,
    existing_records=None,
    forward_distances_m=FACADE_GROUP_RECOVERY_FORWARD_DISTANCES_M,
    lateral_pad_m=FACADE_GROUP_RECOVERY_LATERAL_PAD_M,
    radius_m=FACADE_GROUP_RECOVERY_QUERY_RADIUS_M,
    min_forward_m=FACADE_GROUP_CANDIDATE_MIN_FORWARD_M,
    min_frontality=FACADE_GROUP_RECOVERY_MIN_FRONTALITY,
    verbose=False,
):
    """Search a small outward-facing fan when the normal grid has no usable view."""
    if not geom:
        return []
    frame = geom["frame"]
    outline_m = np.asarray(geom["outline_m"], dtype=np.float64)
    origin = np.asarray(frame["origin"], dtype=np.float64)
    u_dir3 = np.asarray(frame["u_dir"], dtype=np.float64)
    n_xy = safe_unit(np.asarray(frame["normal_xy"], dtype=np.float64)[:2])
    if np.linalg.norm(n_xy) < 1e-9:
        return []

    u_min = float(np.nanmin(outline_m[:, 0]))
    u_max = float(np.nanmax(outline_m[:, 0]))
    u_center = 0.5 * (u_min + u_max)
    sample_u = (u_min - float(lateral_pad_m), u_center, u_max + float(lateral_pad_m))
    existing_ids = {
        str(rec.get("pano_id", ""))
        for rec in (existing_records or [])
        if rec is not None
    }
    found = []

    for forward in forward_distances_m:
        for u in sample_u:
            target = origin + float(u) * u_dir3
            target[:2] += float(forward) * n_xy
            lon, lat = transformer.transform(float(target[0]), float(target[1]))
            meta = get_nearest_pano(
                float(lat),
                float(lon),
                api_key,
                radius=float(radius_m),
                verbose=verbose,
            )
            if not meta:
                continue
            pano_id = str(meta.get("pano_id", ""))
            if not pano_id or pano_id in existing_ids:
                continue
            try:
                rec = _pano_record_from_metadata(
                    meta,
                    back_tx,
                    selection_origin="outward_recovery_search",
                )
            except (KeyError, TypeError, ValueError):
                continue
            candidate = _facade_candidate_from_record(geom, rec, "outward_recovery_search")
            if float(candidate["forward_m"]) < float(min_forward_m):
                continue
            if float(candidate["frontality"]) < float(min_frontality):
                continue
            if float(candidate["lateral_outside_m"]) > float(lateral_pad_m) + float(radius_m):
                continue
            existing_ids.add(pano_id)
            found.append(rec)

    scored = [
        (_facade_candidate_from_record(geom, rec, "outward_recovery_search"), rec)
        for rec in found
    ]
    scored.sort(key=lambda row: (
        -float(row[0]["frontality"]),
        float(row[0]["dist_m"]),
        str(row[1].get("pano_id", "")),
    ))
    return [rec for _candidate, rec in scored]

def select_panos_for_facade_group(geom,
                                  pano_records,
                                  max_panos=FACADE_GROUP_MAX_CANDIDATE_PANOS,
                                  target_spacing_m=FACADE_GROUP_CANDIDATE_TARGET_SPACING_M,
                                  min_forward_m=FACADE_GROUP_CANDIDATE_MIN_FORWARD_M,
                                  max_lateral_outside_m=FACADE_GROUP_CANDIDATE_MAX_LATERAL_OUTSIDE_M,
                                  L_out=EXTRUSION_LEN_XY,
                                  back_eps=BACK_EPS,
                                  wall_quads=None,
                                  wall_normals=None):
    """
    Select Street View candidates distributed along a grouped facade.

    Candidate discovery is independent of the later best-source selection.
    """
    if not geom or not pano_records:
        return []

    frame = geom["frame"]
    outline = np.asarray(geom["outline_xyz"], dtype=np.float64)
    outline_m = np.asarray(geom["outline_m"], dtype=np.float64)
    n_xy = safe_unit(np.asarray(frame["normal_xy"], dtype=np.float64)[:2])
    u_dir = safe_unit(np.asarray(frame["u_dir"], dtype=np.float64)[:2])
    if np.linalg.norm(n_xy) < 1e-9 or np.linalg.norm(u_dir) < 1e-9:
        return []

    center_xy = np.nanmean(outline[:, :2], axis=0)
    u_vals = outline_m[:, 0]
    u_min = float(np.nanmin(u_vals))
    u_max = float(np.nanmax(u_vals))
    span = max(u_max - u_min, 0.0)
    if span <= 1e-6:
        return []

    candidates = []

    for rec in pano_records:
        selection_origin = str(rec.get("selection_origin", "facade_span_sampling"))
        candidate = _facade_candidate_from_record(geom, rec, selection_origin)
        if float(candidate["dist_m"]) < 1e-9:
            continue
        if (
            float(candidate["forward_m"]) < float(min_forward_m)
            or float(candidate["forward_m"]) > float(L_out)
        ):
            continue
        if (
            selection_origin != "outward_recovery_search"
            and float(candidate["lateral_outside_m"]) > float(max_lateral_outside_m)
        ):
            continue
        candidates.append(candidate)

    contains_recovery = any(
        candidate.get("selection_origin") == "outward_recovery_search"
        for candidate in candidates
    )
    spacing_count = max(
        1,
        int(math.ceil(span / max(float(target_spacing_m), 1e-6))) + 1,
    )
    target_count = int(min(
        max(1, int(max_panos)),
        len(candidates),
        max(1, int(max_panos)) if contains_recovery else spacing_count,
    ))
    targets = np.linspace(u_min, u_max, target_count) if candidates else []

    selected = []
    used = set()
    for target_u in targets:
        best = None
        best_key = None
        for cand in candidates:
            pano_id = cand["rec"].get("pano_id")
            if pano_id in used:
                continue
            key = (
                abs(cand["u_clamped"] - float(target_u)),
                cand["off_axis_deg"],
                cand["lateral_outside_m"],
                -cand["frontality"],
                cand["forward_m"],
                cand["dist_m"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        if best is not None:
            used.add(best["rec"].get("pano_id"))
            selected.append(best)

    if not selected:
        candidates.sort(key=lambda c: (
            c["lateral_outside_m"],
            c["off_axis_deg"],
            -c["frontality"],
            c["dist_m"],
        ))
        selected = candidates[:max(1, int(max_panos))]

    legacy_records = select_legacy_panos_for_wall_quads(
        wall_quads,
        wall_normals,
        pano_records,
        L_out=L_out,
        back_eps=back_eps,
    )
    legacy_candidates = [
        _facade_candidate_from_record(geom, rec, "legacy_wall_prism")
        for rec in legacy_records
    ]
    for legacy in legacy_candidates:
        legacy_rec = legacy["rec"]
        legacy["legacy_wall_target_xyz"] = legacy_rec.get("_legacy_wall_target_xyz")
        legacy["legacy_wall_base_seg_xy"] = legacy_rec.get("_legacy_wall_base_seg_xy")
        legacy["legacy_wall_normal_xy"] = legacy_rec.get("_legacy_wall_normal_xy")

    selected_by_id = {
        str(candidate["rec"].get("pano_id", "")): index
        for index, candidate in enumerate(selected)
    }
    protected_ids = set()
    for legacy in legacy_candidates:
        pano_id = str(legacy["rec"].get("pano_id", ""))
        if pano_id in selected_by_id:
            current = selected[selected_by_id[pano_id]]
            origins = current.setdefault("selection_origins", [current.get("selection_origin", "")])
            if "legacy_wall_prism" not in origins:
                origins.append("legacy_wall_prism")
            current["legacy_wall_prism"] = True
            current["legacy_wall_target_xyz"] = legacy.get("legacy_wall_target_xyz")
            current["legacy_wall_base_seg_xy"] = legacy.get("legacy_wall_base_seg_xy")
            current["legacy_wall_normal_xy"] = legacy.get("legacy_wall_normal_xy")
            protected_ids.add(pano_id)
            continue

        legacy["legacy_wall_prism"] = True
        if len(selected) < max(1, int(max_panos)):
            selected.append(legacy)
            selected_by_id[pano_id] = len(selected) - 1
            protected_ids.add(pano_id)
            continue

        replaceable = [
            (abs(float(candidate["u_clamped"]) - float(legacy["u_clamped"])), index)
            for index, candidate in enumerate(selected)
            if str(candidate["rec"].get("pano_id", "")) not in protected_ids
        ]
        if replaceable:
            _, replace_index = min(replaceable)
            old_id = str(selected[replace_index]["rec"].get("pano_id", ""))
            selected_by_id.pop(old_id, None)
            selected[replace_index] = legacy
            selected_by_id[pano_id] = replace_index
            protected_ids.add(pano_id)

    if not selected:
        _pick_xy, rec = select_pano_for_facade_group(
            geom,
            pano_records,
            L_out=L_out,
            back_eps=back_eps,
        )
        if rec is None:
            return []
        fallback = _facade_candidate_from_record(geom, rec, "nonfrontal_fallback")
        fallback["is_fallback"] = True
        selected = [fallback]

    selected.sort(key=lambda c: c["u_clamped"])
    return selected
