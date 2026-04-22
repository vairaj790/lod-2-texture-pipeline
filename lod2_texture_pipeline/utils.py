# -*- coding: utf-8 -*-
"""General helpers, naming, overlays, and viewer bundle export."""

import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import NAMING_STYLE, STAGE_PATTERNS
def ensure_outdir(path): os.makedirs(path, exist_ok=True)

def sanitize_for_fname(s: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)

def safe_unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v*0.0

def _mask_key(u: str) -> str:
    return re.sub(r'(?<=key=)[^&]+', '****', u)

def _wallbase_verbose(wall_idx, rec, heading, pitch, fov):
    return (
        f"wall_{wall_idx:02d}"
        f"__pano_{sanitize_for_fname(rec['pano_id'])}"
        f"__hdg_{int(round(heading))}"
        f"__pit_{int(round(pitch))}"
        f"__fov_{int(round(fov))}"
    )

def name_for(stage, *, base, wall=None, rec=None, heading=None, pitch=None, fov=None):
    pat = STAGE_PATTERNS[NAMING_STYLE][stage]
    wallbase = None
    if "{wallbase}" in pat:
        wallbase = _wallbase_verbose(wall, rec, heading, pitch, fov)
    return pat.format(
        base=base,
        wall=wall if wall is not None else 0,
        wallbase=wallbase,
    )

def overlay_polygon_on_pil(img_pil: Image.Image,
                           poly_xy: np.ndarray,
                           fill_rgba=(255, 0, 0, 64),
                           edge_rgba=(255, 0, 0, 255),
                           edge_width=2) -> Image.Image:
    out = img_pil.convert("RGBA")
    W, H = out.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    pts = [tuple(map(float, p)) for p in poly_xy]
    if len(pts) >= 3:
        draw.polygon(pts, fill=fill_rgba)
    if edge_width > 0 and len(pts) >= 2:
        draw.line(pts + [pts[0]], fill=edge_rgba, width=edge_width)
    return Image.alpha_composite(out, layer)

def save_with_overlay(img_pil: Image.Image, poly_xy: np.ndarray, out_path: str):
    img_ov = overlay_polygon_on_pil(img_pil, poly_xy)
    Image.fromarray(np.array(img_ov)).save(out_path)

def save_quad_fit_debug_overlay(
    img_pil: Image.Image,
    wall_quad_xy: np.ndarray,
    out_path: str,
    seg_main_quad_xy: Optional[np.ndarray] = None,
    seg_chunk_quads_xy: Optional[List[np.ndarray]] = None,
    seg_contours: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Draws:
      - cyan   : segmentation contours
      - orange : per-chunk segmentation quads
      - green  : main segmentation quad
      - red    : projected wall quad
    """
    out = np.array(img_pil.convert("RGBA")).copy()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGBA2BGR)

    if seg_contours is not None:
        cv2.drawContours(out_bgr, seg_contours, -1, (255, 255, 0), 2)  # cyan in BGR

    if seg_chunk_quads_xy is not None:
        for qi, q in enumerate(seg_chunk_quads_xy, start=1):
            q_i = np.round(q).astype(np.int32)
            for i in range(4):
                p1 = tuple(q_i[i])
                p2 = tuple(q_i[(i + 1) % 4])
                cv2.line(out_bgr, p1, p2, (0, 165, 255), 2)  # orange
            for i, p in enumerate(q_i):
                cv2.circle(out_bgr, tuple(p), 4, (0, 165, 255), -1)
                cv2.putText(
                    out_bgr,
                    f"C{qi}P{i+1}",
                    (int(p[0]) + 4, int(p[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    if seg_main_quad_xy is not None:
        q_i = np.round(seg_main_quad_xy).astype(np.int32)
        for i in range(4):
            p1 = tuple(q_i[i])
            p2 = tuple(q_i[(i + 1) % 4])
            cv2.line(out_bgr, p1, p2, (0, 255, 0), 3)  # green
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                out_bgr,
                f"S{i+1}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        for i, p in enumerate(q_i):
            cv2.circle(out_bgr, tuple(p), 5, (0, 255, 0), -1)
            cv2.putText(
                out_bgr,
                f"SP{i+1}",
                (int(p[0]) + 6, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    if wall_quad_xy is not None:
        q_i = np.round(wall_quad_xy).astype(np.int32)
        for i in range(4):
            p1 = tuple(q_i[i])
            p2 = tuple(q_i[(i + 1) % 4])
            cv2.line(out_bgr, p1, p2, (0, 0, 255), 3)  # red
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                out_bgr,
                f"W{i+1}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        for i, p in enumerate(q_i):
            cv2.circle(out_bgr, tuple(p), 5, (255, 255, 255), -1)
            cv2.putText(
                out_bgr,
                f"WP{i+1}",
                (int(p[0]) + 6, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    out_rgba = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGBA)
    Image.fromarray(out_rgba).save(out_path)

def save_sam3_instance_debug_overlay(
    base_img_pil: Image.Image,
    facade_stack: np.ndarray,
    roof_mask: np.ndarray,
    selected_idx: int,
    facade_scores,
    out_path: str
):
    """
    Save a debug overlay in the SAM3 inference frame.

    Shows:
      - each facade instance as translucent overlay
      - KEEP / REMOVE label
      - score and basic stats
      - projected wall polygon in red
    """
    base = base_img_pil.convert("RGBA")
    W, H = base.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()

    # optional roof overlay (very light cyan, just for context)
    if roof_mask is not None and roof_mask.any():
        roof_u8 = (roof_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(roof_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = [tuple(map(int, p[0])) for p in cnt]
            draw.polygon(pts, fill=(0, 255, 255, 28), outline=(0, 255, 255, 90))

    # score lookup by instance id
    score_map = {}
    for row in facade_scores:
        i, score, area, inter, outside, center_dist = row
        score_map[int(i)] = {
            "score": float(score),
            "area": int(area),
            "inter": int(inter),
            "outside": int(outside),
            "center_dist": float(center_dist),
        }

    # draw each facade instance
    n = int(facade_stack.shape[0]) if facade_stack is not None else 0
    for i in range(n):
        mask = facade_stack[i] & (~roof_mask)
        if not mask.any():
            continue

        is_keep = (i == selected_idx)

        if is_keep:
            fill_rgba = (0, 255, 0, 120)
            edge_rgba = (0, 255, 0, 255)
            status = "KEEP"
        else:
            fill_rgba = (255, 180, 0, 95)
            edge_rgba = (255, 180, 0, 255)
            status = "REMOVE"

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_x = []
        all_y = []

        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = [tuple(map(int, p[0])) for p in cnt]
            draw.polygon(pts, fill=fill_rgba, outline=edge_rgba)
            for x, y in pts:
                all_x.append(x)
                all_y.append(y)

        if not all_x or not all_y:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            all_x = xs.tolist()
            all_y = ys.tolist()

        x0, x1 = int(min(all_x)), int(max(all_x))
        y0, y1 = int(min(all_y)), int(max(all_y))

        info = score_map.get(i, None)
        if info is None:
            label = f"facade_{i} {status}"
        else:
            label = (
                f"facade_{i} {status} | "
                f"score={info['score']:.3f} | "
                f"area={info['area']} | "
                f"inter={info['inter']} | "
                f"outside={info['outside']}"
            )

        # place label near top-left of the instance bbox
        tx = max(4, x0 + 4)
        ty = max(4, y0 + 4)

        # simple text background
        try:
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
        except Exception:
            pass

        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(base, overlay)
    out.save(out_path)

def _json_to_u8(obj) -> np.ndarray:
    b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return np.frombuffer(b, dtype=np.uint8)

def save_viewer_bundle_npz(bundle_path: str,
                           corners_xyz: np.ndarray,
                           id_to_idx: Dict[int, int],
                           edges_by_type: Dict[str, List[Tuple[int,int]]],
                           wall_quads_xyz_b1b2t2t1: np.ndarray,
                           wall_meta: List[Dict[str, Any]],
                           viewer_index: List[Dict[str, Any]]):
    """
    Writes a portable bundle for local viewing (no Open3D on cluster).
    """
    # idx_to_node_id for reverse mapping (optional but handy)
    idx_to_node_id = np.empty((len(id_to_idx),), dtype=np.int64)
    for nid, idx in id_to_idx.items():
        idx_to_node_id[idx] = int(nid)

    def edges_to_idx(ed):
        out = []
        for s, t in ed:
            if s in id_to_idx and t in id_to_idx:
                out.append((id_to_idx[s], id_to_idx[t]))
        return np.array(out, dtype=np.int32) if out else np.zeros((0,2), dtype=np.int32)

    edges_wall_idx = edges_to_idx(edges_by_type.get("wall", []))
    edges_base_idx = edges_to_idx(edges_by_type.get("base", []))
    edges_roof_idx = edges_to_idx(edges_by_type.get("roof", []))

    np.savez_compressed(
        bundle_path,
        corners_xyz=corners_xyz.astype(np.float64),
        idx_to_node_id=idx_to_node_id.astype(np.int64),

        edges_wall_idx=edges_wall_idx,
        edges_base_idx=edges_base_idx,
        edges_roof_idx=edges_roof_idx,

        wall_quads_xyz_b1b2t2t1=wall_quads_xyz_b1b2t2t1.astype(np.float64),

        wall_meta_json_u8=_json_to_u8(wall_meta),
        viewer_index_json_u8=_json_to_u8(viewer_index),
    )
