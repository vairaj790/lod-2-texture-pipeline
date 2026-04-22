# -*- coding: utf-8 -*-
"""Camera geometry, LR-band construction, SAM3 loading, rectification, and coverage tiling."""

import math
from typing import Any, List, Optional

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
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .config import *
from .streetview import fetch_sv_image_by_id, wrap_delta_deg
from .utils import safe_unit, save_with_overlay
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

def to_homog(xy): return np.hstack([xy, np.ones((xy.shape[0],1), dtype=xy.dtype)])

def apply_H(xy, H):
    P = to_homog(xy) @ H.T
    P /= P[:,2:3]
    return P[:,:2]

def S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip: bool):
    s = float(ppm)
    if flip:
        return np.array([[ s,  0., -s*xmin],
                         [ 0., -s,  s*ymax],
                         [ 0.,  0., 1.0]], dtype=np.float64)
    else:
        return np.array([[ s, 0., -s*xmin],
                         [ 0., s, -s*ymin],
                         [ 0., 0.,  1.0]], dtype=np.float64)

def choose_orientation_from_poly(poly_m, xmin, ymin, xmax, ymax, ppm):
    S_no = S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip=False)
    S_fl = S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm, flip=True)
    Pno  = apply_H(poly_m, S_no)
    Pfl  = apply_H(poly_m, S_fl)
    def upright(P):
        y_top = 0.5*(P[2,1] + P[3,1])  # t2,t1
        y_bot = 0.5*(P[0,1] + P[1,1])  # b1,b2
        return y_top < y_bot
    if upright(Pno) and not upright(Pfl): return False
    if upright(Pfl) and not upright(Pno): return True
    return not upright(Pno)

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
        yaw = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0  # 0° is +Y
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

def linspace_centers(start, end, span, tile_fov, overlap):
    step = tile_fov - overlap
    n = max(1, int(math.ceil(span / step)))
    if n == 1:
        return [((start + end) * 0.5) % 360.0]
    centers = []
    for i in range(n):
        a = (start + (i + 0.5) * (span / n)) % 360.0
        centers.append(a)
    return centers

def plane_from_triangle(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    n = safe_unit(n)
    d = -float(np.dot(n, p0))
    return n, d

def crop_around_poly(img_pil: Image.Image, poly_uv: np.ndarray, pad: int = 16):
    W, H = img_pil.size
    if poly_uv is None or poly_uv.size == 0:
        return img_pil, poly_uv, (0,0,W,H), np.eye(3, dtype=np.float64)

    x0 = int(max(0,          math.floor(poly_uv[:,0].min() - pad)))
    y0 = int(max(0,          math.floor(poly_uv[:,1].min() - pad)))
    x1 = int(min(W,          math.ceil (poly_uv[:,0].max() + pad)))
    y1 = int(min(H,          math.ceil (poly_uv[:,1].max() + pad)))
    if x1 <= x0 or y1 <= y0:
        return img_pil, poly_uv, (0,0,W,H), np.eye(3, dtype=np.float64)

    out = img_pil.crop((x0, y0, x1, y1))
    uv2 = poly_uv - np.array([x0, y0], dtype=np.float64)
    S  = np.array([[1.,0.,-x0],
                   [0.,1.,-y0],
                   [0.,0., 1.]], dtype=np.float64)
    return out, uv2, (x0,y0,x1,y1), S

def stitch_tiles_to_mosaic(ref_img_pil, ref_K, ref_Rwc, ref_C, tiles, plane_n, plane_d, uv_ref=None):
    import cv2, numpy as np
    from PIL import Image
    FEATURE_PREF = "auto"
    LOWE_RATIO   = 0.75
    RANSAC_PX    = 4.0
    CANVAS_PAD   = 800

    def feat2d(pref:str):
        pref = pref.lower()
        if pref=="sift"  and hasattr(cv2,"SIFT_create"):  return cv2.SIFT_create()
        if pref=="akaze" and hasattr(cv2,"AKAZE_create"): return cv2.AKAZE_create()
        if pref=="orb":  return cv2.ORB_create(5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        if hasattr(cv2,"SIFT_create"):  return cv2.SIFT_create()
        if hasattr(cv2,"AKAZE_create"): return cv2.AKAZE_create()
        return cv2.ORB_create(5000)

    def norm_for(f): return cv2.NORM_L2 if "sift" in type(f).__name__.lower() else cv2.NORM_HAMMING

    def kpdesc(img_bgr, f):
        k = f.detect(img_bgr, None)
        k, des = f.compute(img_bgr, k)
        if not k or des is None or len(k) < 4:
            return np.zeros((0,2), np.float32), None
        return np.float32([p.pt for p in k]), des

    def match(desA, desB, norm, ratio):
        m = cv2.BFMatcher(norm).knnMatch(desA, desB, k=2)
        return [a for a, b in m if a.distance < ratio*b.distance]

    def estimate_H(A, B, f):
        n = norm_for(f)
        pA, dA = kpdesc(A, f)
        pB, dB = kpdesc(B, f)
        if dA is None or dB is None: return None, 0
        g = match(dA, dB, n, LOWE_RATIO)
        if len(g) < 4: return None, 0
        src = np.float32([pB[t.trainIdx] for t in g])
        dst = np.float32([pA[t.queryIdx] for t in g])
        H, msk = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        return (None if H is None else np.asarray(H, np.float64)), (int(msk.sum()) if msk is not None else 0)

    def chain_H(imgs, f):
        n = len(imgs)
        Hf = [None] * (n - 1)
        for i in range(n - 1):
            H, inl = estimate_H(imgs[i+1], imgs[i], f)  # map i -> i+1
            if H is None or inl < 30: print(f"[warn] {i}->{i+1} inliers:{inl}")
            else:                     print(f"[ok]   {i}->{i+1} inliers:{inl}")
            Hf[i] = H
        return Hf

    def to_ref(Hf, ref:int):
        n = len(Hf) + 1
        H = [None] * n
        H[ref] = np.eye(3, dtype=np.float64)
        for k in range(ref - 1, -1, -1):
            H[k] = None if (Hf[k] is None or H[k+1] is None) else (Hf[k] @ H[k+1])
        for k in range(ref + 1, n):
            ok = True; M = np.eye(3, dtype=np.float64)
            for j in range(ref, k):
                if Hf[j] is None: ok = False; break
                M = Hf[j] @ M
            H[k] = None if (not ok or abs(np.linalg.det(M)) < 1e-12) else np.linalg.inv(M)
        return H

    def warp_corners(hw, H):
        h, w = hw
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        return cv2.perspectiveTransform(pts, H).reshape(-1,2)

    def centered_canvas(imgs, Href, ref:int, pad:int):
        corners = [warp_corners(im.shape[:2], H) if H is not None else None for im, H in zip(imgs, Href)]
        allp = np.vstack([c for c in corners if c is not None])
        mn = np.floor(allp.min(0)); mx = np.ceil(allp.max(0))
        W0 = int((mx[0]-mn[0]) + 2*pad + 0.5);  H0 = int((mx[1]-mn[1]) + 2*pad + 0.5)
        T0 = np.array([[1,0,-mn[0]+pad],[0,1,-mn[1]+pad],[0,0,1]], np.float64)
        ref_c = corners[ref].mean(0)
        ref_c0 = (T0 @ np.array([ref_c[0], ref_c[1], 1.0])).ravel()[:2]
        dxy = np.array([W0/2.0, H0/2.0]) - ref_c0
        min_after = np.array([pad, pad]) + dxy
        max_after = np.array([pad+(mx[0]-mn[0]), pad+(mx[1]-mn[1])]) + dxy
        extra = np.maximum([0,0,0,0],
                           [-min_after[0], -min_after[1], max_after[0]-W0, max_after[1]-H0])
        W, Hh = int(W0+extra[0]+extra[2]+0.5), int(H0+extra[1]+extra[3]+0.5)
        T = np.array([[1,0,extra[0]],[0,1,extra[1]],[0,0,1]], np.float64) @ \
            np.array([[1,0,dxy[0]],[0,1,dxy[1]],[0,0,1]], np.float64) @ T0
        return [(None if H is None else (T@H).astype(np.float64)) for H in Href], (W, Hh), T

    def dist_w(mask):
        m = (mask > 0).astype(np.uint8)
        if m.sum() == 0: return np.zeros_like(m, np.float32)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        if d.max() > 0: d /= d.max()
        return np.clip(d.astype(np.float32), 1e-3, 1.0)

    def expo_match(base, wbase, add, addmask, clamp=(0.6,1.6)):
        ov = (wbase > 0) & (addmask > 0)
        if ov.sum() < 500: return add
        bg = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ag = cv2.cvtColor(add.astype(np.uint8),  cv2.COLOR_BGR2GRAY).astype(np.float32)
        mb, ma = float(bg[ov].mean()), float(ag[ov].mean())
        if ma < 1e-6: return add
        g = float(np.clip(mb/ma, clamp[0], clamp[1]))
        return np.clip(add * g, 0, 255)

    ref_bgr = cv2.cvtColor(np.array(ref_img_pil), cv2.COLOR_RGB2BGR)
    imgs_bgr = [ref_bgr] + [cv2.cvtColor(np.array(t["img"]), cv2.COLOR_RGB2BGR) for t in tiles]
    print(f"[info] Loaded {len(imgs_bgr)} images for mosaic (anchor is first).")
    f = feat2d(FEATURE_PREF)
    print(f"[info] Using feature: {type(f).__name__}")

    Hf = chain_H(imgs_bgr, f)
    Href = to_ref(Hf, ref=0)
    for k, H in enumerate(Href):
        print(f"[info] H[{k}] to anchor: {'OK' if H is not None else 'SKIP'}")

    Hfin, (W, Hh), T = centered_canvas(imgs_bgr, Href, ref=0, pad=CANVAS_PAD)
    print(f"[info] Global canvas size (centered): {W} x {Hh}")

    pano = np.zeros((Hh, W, 3), np.float32)
    wgt  = np.zeros((Hh, W), np.float32)

    for k, (im, Hk) in enumerate(zip(imgs_bgr, Hfin)):
        if Hk is None:
            print(f"[skip] image #{k} (no homography to anchor)"); continue
        h, w = im.shape[:2]
        warped  = cv2.warpPerspective(im, Hk, (W, Hh))
        mask_u8 = cv2.warpPerspective(np.ones((h,w), np.uint8)*255, Hk, (W, Hh))
        w_new   = dist_w(mask_u8)
        warpedF = expo_match(pano, wgt, warped.astype(np.float32), (mask_u8 > 0))
        pano += warpedF * w_new[...,None]
        wgt  += w_new
        print(f"[blend] Added image #{k}")

    nz = wgt > 0
    out = np.zeros_like(pano, np.uint8)
    out[nz] = np.clip(pano[nz] / wgt[nz,None], 0, 255).astype(np.uint8)
    mosaic_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    M = T.astype(np.float64)
    return mosaic_pil, (float(M[0,2]), float(M[1,2])), M

def ensure_wall_coverage(pano_id, cam, wall_quad_xyz,
                         heading, pitch, fov_deg,
                         img_size=SV_SIZE):
    urls_fetched = []
    img_pil, url, _, _ = fetch_sv_image_by_id(pano_id, heading, pitch, fov_deg, API_KEY, size=img_size)
    urls_fetched.append(url)

    K_ref, Rwc_ref, C_ref = build_pose_from_heading_pitch(cam, heading, pitch, img_size=img_size, fov_deg=fov_deg)
    uv_ref, _ = project_points_world_to_image(wall_quad_xyz, K_ref, Rwc_ref, C_ref, clip_behind=False)
    W, H = img_pil.size

    if uv_inside_image(uv_ref, W, H, COVER_MARGIN_PX):
        return img_pil, uv_ref, heading, pitch, fov_deg, K_ref, Rwc_ref, C_ref, False, np.eye(3), urls_fetched

    yaws, pits = yaw_pitch_of_points(cam, wall_quad_xyz)
    yaw_center, yaw_span, yaw_min, yaw_max = circular_span(yaws)
    pit_center = 0.5 * (pits.min() + pits.max())
    pit_span   = float(pits.max() - pits.min())
    need_span = max(yaw_span, pit_span) + 2*ANGLE_MARGIN_DEG

    if need_span <= FOV_MAX + 1e-6:
        fov_new     = float(np.clip(need_span, FOV_MIN, FOV_MAX))
        heading_new = yaw_center
        pitch_new   = pit_center
        img2, url2, _, _ = fetch_sv_image_by_id(pano_id, heading_new, pitch_new, fov_new, API_KEY, size=img_size)
        urls_fetched.append(url2)

        K2, R2, C2 = build_pose_from_heading_pitch(cam, heading_new, pitch_new, img_size=img_size, fov_deg=fov_new)
        uv2, _ = project_points_world_to_image(wall_quad_xyz, K2, R2, C2, clip_behind=False)
        return img2, uv2, heading_new, pitch_new, fov_new, K2, R2, C2, False, np.eye(3), urls_fetched

    yaw_centers = linspace_centers(yaw_min, (yaw_min + yaw_span) % 360.0, yaw_span, TILE_FOV, TILE_OVERLAP_DEG)
    pit_center  = 0.5 * (pits.min() + pits.max())
    ny = max(1, int(math.ceil(pit_span / (TILE_FOV - TILE_OVERLAP_DEG))))
    pit_centers = [pit_center] if ny == 1 else list(np.linspace(pits.min(), pits.max(), ny))

    tiles = []
    for yc in yaw_centers:
        for pc in pit_centers:
            if abs(((yc - heading + 180) % 360) - 180) < 1e-3 and abs(pc - pitch) < 1e-3 and abs(TILE_FOV - fov_deg) < 1e-3:
                continue
            try:
                im, urlt, _, _ = fetch_sv_image_by_id(pano_id, yc, pc, TILE_FOV, API_KEY, size=img_size)
                urls_fetched.append(urlt)
                Ki, Ri, Ci = build_pose_from_heading_pitch(cam, yc, pc, img_size=img_size, fov_deg=TILE_FOV)
                tiles.append({"img": im, "K": Ki, "Rwc": Ri, "C": Ci, "heading": yc, "pitch": pc, "url": urlt})
            except Exception:
                continue

    for t in tiles:
        d = wrap_delta_deg(t["heading"], heading)
        side = "LEFT" if d < 0 else ("RIGHT" if d > 0 else "CENTER")
        print(f"    • tile @ heading {t['heading']:.1f}°, pitch {t['pitch']:.1f}°  -> {side} (Δyaw={d:+.1f}°)")

    b1, b2, t2, t1 = wall_quad_xyz
    plane_n, plane_d = plane_from_triangle(b1, b2, t2)

    mosaic, offset, M = stitch_tiles_to_mosaic(img_pil, K_ref, Rwc_ref, C_ref, tiles, plane_n, plane_d, uv_ref=uv_ref)
    uv_mosaic = apply_H(uv_ref, M)
    CROP_PAD = max(LR_BAND_BUFFER_PX + 6, 12)
    mosaic_cropped, uv_cropped, crop_box, S_crop = crop_around_poly(mosaic, uv_mosaic, pad=CROP_PAD)
    M_cropped = S_crop @ M
    return mosaic_cropped, uv_cropped, heading, pitch, fov_deg, K_ref, Rwc_ref, C_ref, True, M_cropped, urls_fetched

def save_overlay_matplotlib(img_pil: Image.Image, uv: np.ndarray, out_path: str, title: str = ""):
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
