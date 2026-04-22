# -*- coding: utf-8 -*-
"""Batch pipeline orchestration for LoD-2 facade and roof texturing."""

import json
import math
import os
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
import torch
import trimesh
from PIL import Image
from shapely.geometry import Point

from .config import *
from .geojson_io import build_edge_loops_from_gdf, load_3d_geojson
from .inpainting import build_wall_region_mask, lama_fill_rectified_wall
from .mesh import _build_wall_mesh_from_verts, build_closed_roof_polygons, rasterize_polygons_to_mask, triangulate_surface
from .projection import *
from .quadfit import *
from .streetview import *
from .utils import _mask_key, ensure_outdir, name_for, save_quad_fit_debug_overlay, save_sam3_instance_debug_overlay, save_viewer_bundle_npz, save_with_overlay
def process_building(geojson_path: str,
                     out_root: str,
                     geotiff_path: Optional[str] = None,
                     *,
                     device=None,
                     processor=None,
                     sam3_prompt_facade=None,
                     sam3_prompt_roof=None):

    geojson_base = os.path.splitext(os.path.basename(geojson_path))[0]
    per_building_out = os.path.join(out_root, geojson_base)
    ensure_outdir(per_building_out)
    viewer_index = []  # will be saved as viewer_index.json
    all_wall_quads_global = []      # list of (4,3) arrays in global index order
    all_wall_meta_global = []


    gdf, corners, edge_groups, id_to_idx, wall_centers, base_z = load_3d_geojson(geojson_path)

    # Build loops
    wall_loops = build_edge_loops_from_gdf(gdf, 'wall')
    base_loops = build_edge_loops_from_gdf(gdf, 'base')
    roof_loops = build_edge_loops_from_gdf(gdf, 'roof')

    if not wall_loops:
        print("No wall loops found. Exiting.")
        return

    # Base edges for pano search area (use all base lines)
    base_edges_gdf = gdf[gdf['type'] == 'base'].copy()

    # --- pano discovery (one big set around all base lines in this geojson)
    pano_records = build_search_grid_and_collect_panos(
        list(base_edges_gdf.geometry), transformer, back_tx, API_KEY, offset=GRID_OFFSET_M, n=GRID_N
    )
    if len(pano_records) == 0:
        print("No pano candidates found. Exiting.")
        return

    # ---- PREBUILD PLACEHOLDERS ----
    meshes_named = []
    mesh_by_name = {}

    # Roof placeholders per loop
    # Roof placeholder (SPLIT into separate islands)
    roof_meshes = []  # (name, mesh, coords) per roof island
    roof_edges = edge_groups.get("roof", [])
    if roof_edges:
        rc_list, rf_list = triangulate_surface(roof_edges, corners, id_to_idx, split_components=True)
        if rc_list and rf_list:
            for ridx, (rc, rf) in enumerate(zip(rc_list, rf_list)):
                if rc is None or rf is None:
                    continue
                m = trimesh.Trimesh(vertices=rc, faces=rf, process=False)
                m.visual.face_colors = [220, 220, 220, 255]
                rname = f"roof_{ridx:02d}"
                meshes_named.append((rname, m))
                roof_meshes.append((rname, m, rc))


    # Base placeholders per loop
    for blp in base_loops:
        cid, lid = blp['component_id'], blp['loop_id']
        bc, bf = triangulate_surface(blp['edges'], corners, id_to_idx)
        if bc is None or bf is None:
            continue
        name = f"base_c{cid}_l{lid}"
        m = trimesh.Trimesh(vertices=bc, faces=bf, process=False)
        m.visual.face_colors = [240,240,240,255]
        meshes_named.append((name, m))

    # Wall placeholders (white quads) + build wall_quads bundle in the SAME global indexing scheme
    global_wall_index = 0
    for loop in wall_loops:
        cid, lid = loop['component_id'], loop['loop_id']
        ring_edges = loop['edges']

        wn, centers, base_segs = compute_wall_normals_from_wall_faces(corners, ring_edges, id_to_idx)

        for k in range(len(ring_edges)):
            i_global = global_wall_index
            global_wall_index += 1  # EXACTLY ONCE per wall face

            (s1, t1) = ring_edges[k]
            (s2, t2) = ring_edges[(k + 1) % len(ring_edges)]

            # Default quad placeholder for robustness (keeps array length aligned)
            wall_quad = np.full((4, 3), np.nan, dtype=np.float64)

            if not any(nid not in id_to_idx for nid in [s1, t1, s2, t2]):
                p1a = corners[id_to_idx[s1]]; p1b = corners[id_to_idx[t1]]
                p2a = corners[id_to_idx[s2]]; p2b = corners[id_to_idx[t2]]

                def by_z(a, b): return (a, b) if a[2] <= b[2] else (b, a)
                b1, t1p = by_z(p1a, p1b)
                b2, t2p = by_z(p2a, p2b)

                wall_quad = np.vstack([b1, b2, t2p, t1p]).astype(np.float64)  # [b1,b2,t2,t1]

            # ---- record for bundle (ALWAYS one entry per wall index) ----
            all_wall_quads_global.append(wall_quad.copy())
            all_wall_meta_global.append({
                "component_id": int(cid) if cid is not None else -1,
                "loop_id": int(lid) if lid is not None else -1,
                "loop_index": int(k),
                "global_index": int(i_global),
            })

            # ---- build placeholder mesh ONLY if quad is finite ----
            if np.isfinite(wall_quad).all():
                name = f"wall_c{cid}_l{lid}_w{i_global:02d}"
                placeholder = _build_wall_mesh_from_verts(
                    wall_quad,
                    outward_normal_xyz=wn[k],
                    uv_px=None, tex_img=None, out_w=None, out_h=None,
                    flat_rgba=(240, 240, 240, 255)
                )
                meshes_named.append((name, placeholder))
                mesh_by_name[name] = placeholder


    # Panos & model (passed from main; do NOT reload per building)
    if device is None or processor is None or sam3_prompt_facade is None or sam3_prompt_roof is None:
        raise RuntimeError("SAM3 bundle not provided. Pass device/processor/sam3_prompt_facade/sam3_prompt_roof from main().")




    # Reset running global index for consistent naming in outputs
    global_wall_index = 0

    # ======== Per-loop wall processing ========
    for loop in wall_loops:
        cid, lid = loop['component_id'], loop['loop_id']
        ring_edges = loop['edges']

        wall_normals, centers_xyz, base_segs_xy = compute_wall_normals_from_wall_faces(
            corners, ring_edges, id_to_idx
        )
        sel_xy, sel_recs = select_pano_per_wall_using_prism_base(
            ring_edges, wall_normals, corners, id_to_idx, pano_records
        )

        for k, ((s1, t1id), pick_xy, rec, ctr, seg, n_xy) in enumerate(
            zip(ring_edges, sel_xy, sel_recs, centers_xyz, base_segs_xy, [n[:2] for n in wall_normals])
        ):
            i_global = global_wall_index
            global_wall_index += 1

            if pick_xy is None or rec is None or ctr is None or seg is None or n_xy is None:
                print(f"[c{cid} l{lid} w{k}] no pano selected — skipping.")
                continue

            px, py = pick_xy
            cam = np.array([px, py, base_z + FIXED_HEIGHT_M], float)

            dx, dy = ctr[0] - px, ctr[1] - py
            dz     = ctr[2] - cam[2]
            heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            rho     = np.hypot(dx, dy)
            pitch   = np.degrees(np.arctan2(dz, max(rho, 1e-9)))
            fov_deg = solve_fov_deg(np.array([px, py]), heading, seg, n_xy, buffer_m=SIDE_BUFFER_M, safety_margin_deg=FOV_MARGIN_DEG)

            # Wall quad [b1,b2,t2,t1]
            (s1, t1id) = ring_edges[k]
            (s2, t2id) = ring_edges[(k + 1) % len(ring_edges)]
            p1a = corners[id_to_idx[s1]]; p1b = corners[id_to_idx[t1id]]
            p2a = corners[id_to_idx[s2]]; p2b = corners[id_to_idx[t2id]]
            def by_z(a,b): return (a,b) if a[2] <= b[2] else (b,a)
            b1, t1p = by_z(p1a, p1b); b2, t2p = by_z(p2a, p2b)
            wall_quad = np.vstack([b1, b2, t2p, t1p])  # [b1,b2,t2,t1]

            img_rgb, uv, heading, pitch, fov_deg, K, R_wc, C, is_mosaic, T_mosaic, urls_fetched = ensure_wall_coverage(
                rec["pano_id"], cam, wall_quad, heading, pitch, fov_deg, img_size=SV_SIZE
            )

            print(f"[c{cid} l{lid} w{k}] fetched {len(urls_fetched)} Street View image(s):")
            for u in urls_fetched:
                print("         URL:", u)

            wall_tag = f"c{cid}_l{lid}_w{i_global:02d}"

            sv_jpg_name = (
                f"sv__{geojson_base}__{wall_tag}"
                f"__pano_{rec['pano_id']}"
                f"__hdg_{int(round(heading))}"
                f"__pit_{int(round(pitch))}"
                f"__fov_{int(round(fov_deg))}"
                + ("__mosaic" if is_mosaic else "")
                + ".jpg"
            )

            sv_jpg_path = os.path.join(per_building_out, sv_jpg_name)
            if SAVE_SV_RGB_PER_WALL:
                img_rgb.convert("RGB").save(sv_jpg_path, quality=95)
            else:
                sv_jpg_name = None  # viewer_index will reflect that


            raw_overlay_path = os.path.join(
                per_building_out,
                name_for("raw_overlay", base=geojson_base, wall=i_global, rec=rec,
                        heading=heading, pitch=pitch, fov=fov_deg)
            )
            if SAVE_RAW_OVERLAY_PNG:
                save_overlay_matplotlib(
                    img_rgb, uv, raw_overlay_path,
                    title=f"Wall {wall_tag} — heading {heading:.1f}°, pitch {pitch:.1f}°, fov {fov_deg:.1f}°"
                        + (" (mosaic)" if is_mosaic else "")
                )

            # Choose/define the SV jpg filename (make it unambiguous)
            if SAVE_SV_RGB_PER_WALL:
                sv_jpg_name = (
                    f"sv__{geojson_base}__c{cid}_l{lid}_w{i_global:02d}"
                    f"__pano_{rec['pano_id']}__hdg_{int(round(heading))}"
                    f"__pit_{int(round(pitch))}__fov_{int(round(fov_deg))}"
                    + ("__mosaic" if is_mosaic else "")
                    + ".jpg"
                )
            else:
                sv_jpg_name = None  

            viewer_index.append({
                "geojson": geojson_base,
                "wall_tag": f"c{cid}_l{lid}_w{i_global:02d}",
                "component_id": int(cid) if cid is not None else -1,
                "loop_id": int(lid) if lid is not None else -1,
                "loop_index": int(k),
                "global_index": int(i_global),

                "pano_id": rec["pano_id"],
                "pano_lat": float(rec["lat"]),
                "pano_lng": float(rec["lng"]),
                "camera_utm_xyz": [float(cam[0]), float(cam[1]), float(cam[2])],
                "heading_deg": float(heading),
                "pitch_deg": float(pitch),
                "fov_deg": float(fov_deg),
                "is_mosaic": bool(is_mosaic),

                "wall_quad_xyz_b1b2t2t1": [[float(a), float(b), float(c)] for a,b,c in wall_quad.tolist()],
                "sv_rgb_jpg": sv_jpg_name,
            })


            lr_rgba, band_poly, band_bbox = build_lr_band_rgba(img_rgb, uv, LR_BAND_BUFFER_PX)
            if lr_rgba is None:
                print(f"[c{cid} l{lid} w{k}] LR-band failed — skip.")
                continue
            lr_overlay_path = os.path.join(
                per_building_out,
                name_for("lr_band_overlay", base=geojson_base, wall=i_global, rec=rec,
                        heading=heading, pitch=pitch, fov=fov_deg)
            )
            if SAVE_LR_OVERLAY_PNG:
                save_with_overlay(lr_rgba, uv, lr_overlay_path)

            W, H = lr_rgba.size
            r,g,b,a0 = lr_rgba.split()
            alpha_np = np.array(a0, dtype=np.uint8)
            if CROP_TO_ALPHA_BBOX and (alpha_np.min() < 255):
                L,Tp,R2,B2 = (Image.fromarray(alpha_np).getbbox() or (0,0,W,H))
                img_for_model = lr_rgba.crop((L,Tp,R2,B2)).convert("RGB")
                off_x, off_y  = L, Tp
                out_Wm2f, out_Hm2f  = R2-L, B2-Tp
            else:
                img_for_model = lr_rgba.convert("RGB")
                off_x = off_y = 0
                out_Wm2f, out_Hm2f  = W, H

            # img_for_model is PIL RGB (cropped or not)
            # We need a boolean mask in that same (out_Hm2f, out_Wm2f) space.

            def _extract_mask_stack(out_obj, H, W):
                masks = None
                if isinstance(out_obj, dict):
                    masks = out_obj.get("masks", out_obj.get("mask", out_obj.get("pred_masks", None)))
                else:
                    masks = getattr(out_obj, "masks", None)

                if masks is None:
                    return np.zeros((0, H, W), dtype=bool)

                if torch.is_tensor(masks):
                    m = masks.detach().float().cpu().numpy()
                else:
                    m = np.asarray(masks)

                # (N,1,H,W) -> (N,H,W)
                if m.ndim == 4 and m.shape[1] == 1:
                    m = m[:, 0]

                # (H,W) -> (1,H,W)
                if m.ndim == 2:
                    m = m[None, ...]

                if m.ndim != 3:
                    return np.zeros((0, H, W), dtype=bool)

                stack = (m > 0.5)

                # keep only non-empty masks
                keep = [stack[i] for i in range(stack.shape[0]) if stack[i].any()]
                if len(keep) == 0:
                    return np.zeros((0, H, W), dtype=bool)

                return np.stack(keep, axis=0)


            def _stack_union(mask_stack, H, W):
                if mask_stack.shape[0] == 0:
                    return np.zeros((H, W), dtype=bool)
                return mask_stack.any(axis=0)


            def _polygon_to_mask(H, W, poly_xy):
                mask = np.zeros((H, W), dtype=np.uint8)
                poly = np.round(poly_xy).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 255)
                return mask > 0
            
            def _select_best_facade_instance(facade_stack, roof_mask, wall_poly_xy, H, W):
                """
                Select the best facade instance using a soft score.
                No hard threshold is used.
                """
                if facade_stack.shape[0] == 0:
                    return np.zeros((H, W), dtype=bool), -1, []

                wall_mask = _polygon_to_mask(H, W, wall_poly_xy)
                wall_area = max(int(wall_mask.sum()), 1)

                wy, wx = np.where(wall_mask)
                if len(wx) == 0:
                    # fallback: if wall polygon is invalid, just keep the largest facade instance
                    scored = []
                    best_idx = -1
                    best_score = -1e18
                    best_mask = np.zeros((H, W), dtype=bool)
                    for i in range(facade_stack.shape[0]):
                        cand = facade_stack[i] & (~roof_mask)
                        area = int(cand.sum())
                        score = float(area)
                        scored.append((i, score, area, 0, 0, 0.0))
                        if score > best_score:
                            best_score = score
                            best_idx = i
                            best_mask = cand
                    return best_mask, best_idx, scored

                wall_cx = float(wx.mean())
                wall_cy = float(wy.mean())
                diag = max(float(np.hypot(W, H)), 1.0)

                best_idx = -1
                best_score = -1e18
                best_mask = np.zeros((H, W), dtype=bool)
                scored = []

                for i in range(facade_stack.shape[0]):
                    cand = facade_stack[i] & (~roof_mask)
                    area = int(cand.sum())
                    if area == 0:
                        continue

                    inter = int((cand & wall_mask).sum())
                    outside = int((cand & (~wall_mask)).sum())

                    cy, cx = np.where(cand)
                    cand_cx = float(cx.mean())
                    cand_cy = float(cy.mean())
                    center_dist = np.hypot(cand_cx - wall_cx, cand_cy - wall_cy) / diag

                    inside_ratio = inter / max(area, 1)
                    wall_cover   = inter / wall_area
                    outside_ratio = outside / max(area, 1)

                    # soft ranking only, no threshold
                    score = (
                        4.0 * inside_ratio +
                        3.0 * wall_cover -
                        2.0 * outside_ratio -
                        1.0 * center_dist +
                        0.15 * np.log1p(area)
                    )

                    scored.append((i, score, area, inter, outside, center_dist))

                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_mask = cand

                return best_mask, best_idx, scored

            with torch.no_grad():
                state = processor.set_image(img_for_model)  # caches image embeddings ONCE

                # 1) facade / wall as separate instances
                out_facade = processor.set_text_prompt(state=state, prompt=sam3_prompt_facade)
                facade_stack = _extract_mask_stack(out_facade, out_Hm2f, out_Wm2f)

                # 2) roof can remain unioned
                out_roof = processor.set_text_prompt(state=state, prompt=sam3_prompt_roof)
                roof_stack = _extract_mask_stack(out_roof, out_Hm2f, out_Wm2f)
                roof_mask = _stack_union(roof_stack, out_Hm2f, out_Wm2f)

                # Optional: dilate roof a bit before subtraction
                if ROOF_SUBTRACT_DILATE_PX and ROOF_SUBTRACT_DILATE_PX > 0:
                    k = int(ROOF_SUBTRACT_DILATE_PX)
                    kernel = np.ones((2*k+1, 2*k+1), dtype=np.uint8)
                    roof_mask_u8 = (roof_mask.astype(np.uint8) * 255)
                    roof_mask_u8 = cv2.dilate(roof_mask_u8, kernel, iterations=1)
                    roof_mask = roof_mask_u8 > 0

                # wall polygon in the same cropped model frame
                uv_wall_model = np.array([
                    [uv[0,0] - off_x, uv[0,1] - off_y],  # b1
                    [uv[1,0] - off_x, uv[1,1] - off_y],  # b2
                    [uv[2,0] - off_x, uv[2,1] - off_y],  # t2
                    [uv[3,0] - off_x, uv[3,1] - off_y],  # t1
                ], dtype=float)

                # choose and MERGE facade instances statistically
                building_mask, best_idx, facade_scores = _select_best_facade_instance(
                    facade_stack=facade_stack,
                    roof_mask=roof_mask,
                    wall_poly_xy=uv_wall_model,
                    H=out_Hm2f,
                    W=out_Wm2f
                )

                print(f"[c{cid} l{lid} w{k}] SAM3 facade instances: {facade_stack.shape[0]} | selected: {best_idx}")
                for row in facade_scores:
                    i, score, area, inter, outside, center_dist = row
                    print(f"    facade[{i}] score={score:.4f} area={area} inter={inter} outside={outside} center_dist={center_dist:.4f}")

            sam3_instances_overlay_path = Path(per_building_out) / name_for(
                "sam3_instances_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )

            save_sam3_instance_debug_overlay(
                base_img_pil=img_for_model,
                facade_stack=facade_stack,
                roof_mask=roof_mask,
                selected_idx=best_idx,
                facade_scores=facade_scores,
                out_path=str(sam3_instances_overlay_path)
            )

            pred_full_raw = np.zeros((H, W), dtype=bool)
            pred_full_raw[off_y:off_y+out_Hm2f, off_x:off_x+out_Wm2f] = building_mask
            pred_full_raw[alpha_np == 0] = False

            # clean the selected full-frame segmentation BEFORE quad fitting
            pred_full_clean = clean_selected_mask(pred_full_raw)
            alpha_build = (pred_full_clean.astype(np.uint8) * 255)

            rgba_full = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_build)))

            # full post-SAM3 frame in ORIGINAL perspective image coordinates
            rgba_final = rgba_full

            uv_wall_final = np.array([
                [uv[0,0], uv[0,1]],  # b1
                [uv[1,0], uv[1,1]],  # b2
                [uv[2,0], uv[2,1]],  # t2
                [uv[3,0], uv[3,1]],  # t1
            ], dtype=float)

            # raw overlay: still useful to inspect the selected segmentation before perspective quad fit
            sam3_overlay_path = Path(per_building_out) / name_for(
                "sam3_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_with_overlay(rgba_final, uv_wall_final, str(sam3_overlay_path))
            rgba_for_rectify = np.array(rgba_final, dtype=np.uint8)
            M_seg_scale_translate = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ], dtype=np.float64)

            # -----------------------------------------------------------------
            # Orthorectify the current segmentation into wall-plane ortho space
            # -----------------------------------------------------------------
            dst_m, metric_meta = wall_metric_target_from_corners(
                wall_quad[0], wall_quad[1], wall_quad[2], wall_quad[3]
            )
            H_pix_to_wall_m = homography_from_4pts(uv_wall_final.astype(float), dst_m.astype(float))

            a2 = rgba_for_rectify[:, :, 3]
            if a2.max() == 0:
                print(f"[c{cid} l{lid} w{k}] Empty alpha before orthorectification; skip.")
                continue

            ys, xs = np.where(a2 > 0)
            contour_px = np.stack([xs, ys], axis=1).astype(np.float64)
            contour_m = apply_H(contour_px, H_pix_to_wall_m)

            xmin, ymin = contour_m.min(axis=0)
            xmax, ymax = contour_m.max(axis=0)

            xmin = min(xmin, dst_m[:, 0].min()) - MARGIN_METERS
            ymin = min(ymin, dst_m[:, 1].min()) - MARGIN_METERS
            xmax = max(xmax, dst_m[:, 0].max()) + MARGIN_METERS
            ymax = max(ymax, dst_m[:, 1].max()) + MARGIN_METERS

            if FLIP_VERTICAL == "auto":
                flip = choose_orientation_from_poly(dst_m, xmin, ymin, xmax, ymax, PIXELS_PER_METER)
            else:
                flip = bool(FLIP_VERTICAL)

            S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, PIXELS_PER_METER, flip=flip)
            H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
            Wm, Hm = (xmax-xmin), (ymax-ymin)
            out_Wr = max(int(np.ceil(Wm*PIXELS_PER_METER)), 1)
            out_Hr = max(int(np.ceil(Hm*PIXELS_PER_METER)), 1)

            area_r = out_Wr * out_Hr
            if area_r > MAX_ORTHO_PIXELS:
                scale = math.sqrt(MAX_ORTHO_PIXELS / float(area_r))
                ppm2  = PIXELS_PER_METER * scale
                S_m_to_px = S_meter_to_pixel(xmin, ymin, xmax, ymax, ppm2, flip=flip)
                H_pix_to_ortho_px = S_m_to_px @ H_pix_to_wall_m
                out_Wr = max(1, int(np.ceil((xmax - xmin) * ppm2)))
                out_Hr = max(1, int(np.ceil((ymax - ymin) * ppm2)))

            src_rgba_np = np.array(rgba_for_rectify, dtype=np.uint8)
            src_bgra = cv2.cvtColor(src_rgba_np, cv2.COLOR_RGBA2BGRA)
            ortho_bgra = cv2.warpPerspective(
                src_bgra,
                H_pix_to_ortho_px,
                (out_Wr, out_Hr),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            ortho_rgba = cv2.cvtColor(ortho_bgra, cv2.COLOR_BGRA2RGBA)

            wall_poly_px = apply_H(dst_m, S_m_to_px)

            # -----------------------------------------------------------------
            # NEW: similarity fit in ORTHO space
            #      fit quad on ortho segmentation, then fit it into ortho wall quad
            # -----------------------------------------------------------------
            ortho_alpha_mask = ortho_rgba[:, :, 3] > 0

            ortho_seg_quad = None
            ortho_seg_hull = None
            ortho_seg_contours = None
            ortho_seg_chunk_quads = []

            ortho_seg_quad_fitted = None
            ortho_seg_chunk_quads_fitted = []
            ortho_seg_contours_fitted = None

            ortho_fit_info = {
                "stage": "post_rectification",
                "scale": 1.0,
                "tx": 0.0,
                "ty": 0.0,
                "min_signed_dist_px": 0.0,
                "center_dist": 0.0,
                "touching_like": False,
            }

            if ENABLE_ORTHO_QUAD_FIT:
                ortho_seg_quad, ortho_seg_hull, ortho_seg_contours, ortho_seg_chunk_quads = fit_quadrilateral_from_mask(
                    ortho_alpha_mask
                )

                if ortho_seg_quad is not None:
                    ortho_seg_quad_fitted, M_seg_scale_translate, ortho_fit_info = fit_seg_quad_inside_wall_quad(
                        seg_quad=ortho_seg_quad.astype(np.float64),
                        wall_quad=wall_poly_px.astype(np.float64)
                    )

                    # apply the same affine to the ORTHO RGBA
                    ortho_bgra_tmp = cv2.cvtColor(ortho_rgba, cv2.COLOR_RGBA2BGRA)
                    ortho_bgra_tmp = cv2.warpAffine(
                        ortho_bgra_tmp,
                        M_seg_scale_translate.astype(np.float32),
                        (ortho_rgba.shape[1], ortho_rgba.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0)
                    )
                    ortho_rgba = cv2.cvtColor(ortho_bgra_tmp, cv2.COLOR_BGRA2RGBA)

                    # transform quad/contours for debug overlay
                    ortho_seg_chunk_quads_fitted = [
                        apply_affine2x3(q.astype(np.float64), M_seg_scale_translate)
                        for q in ortho_seg_chunk_quads
                    ]

                    if ortho_seg_contours is not None:
                        ortho_seg_contours_fitted = []
                        for cnt in ortho_seg_contours:
                            pts = cnt[:, 0, :].astype(np.float64)
                            pts_fit = apply_affine2x3(pts, M_seg_scale_translate).astype(np.int32).reshape(-1, 1, 2)
                            ortho_seg_contours_fitted.append(pts_fit)

                    print(
                        f"   ⬚ ortho similarity fit applied | "
                        f"scale={ortho_fit_info['scale']:.4f} | "
                        f"min_dist={ortho_fit_info['min_signed_dist_px']:.3f}px | "
                        f"touching_like={ortho_fit_info['touching_like']} | "
                        f"center_dist={ortho_fit_info['center_dist']:.4f}"
                    )
                else:
                    print("   ⬚ ortho similarity fit skipped (no valid ortho segmentation quadrilateral found)")

            # -----------------------------------------------------------------
            # Hough-based straight-line detection on the ALREADY scaled ortho result
            # -----------------------------------------------------------------
            ortho_alpha_mask_after_fit = ortho_rgba[:, :, 3] > 0

            wall_mask_bool = build_wall_region_mask(
                ortho_rgba.shape[0],
                ortho_rgba.shape[1],
                wall_poly_px
            ) > 0

            hough_edge_map_u8 = build_edge_map_for_hough(
                ortho_rgba[:, :, :3],
                ortho_alpha_mask_after_fit
            )

            hough_lines = []
            hough_left_line = None
            hough_right_line = None
            hough_top_line = None
            hough_left_info = {}
            hough_right_info = {}
            hough_top_info = {}
            hough_total_segments = 0

            if ENABLE_ORTHO_HOUGH_DEBUG:
                hough_lines = detect_hough_segments(
                    hough_edge_map_u8,
                    roi_mask=(wall_mask_bool.astype(np.uint8) * 255)
                )
                hough_total_segments = len(hough_lines)

                # wall_poly_px order is [b1, b2, t2, t1]
                left_p0, left_p1 = wall_poly_px[3], wall_poly_px[0]
                right_p0, right_p1 = wall_poly_px[1], wall_poly_px[2]
                top_p0, top_p1 = wall_poly_px[2], wall_poly_px[3]

                left_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    left_p0, left_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )
                right_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    right_p0, right_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )
                top_band_u8 = build_line_search_band(
                    ortho_rgba.shape[0], ortho_rgba.shape[1],
                    top_p0, top_p1, wall_mask_bool, HOUGH_SEARCH_BAND_PX
                )

                hough_left_line, hough_left_info = select_best_hough_line_for_target(
                    hough_lines, left_p0, left_p1, left_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )
                hough_right_line, hough_right_info = select_best_hough_line_for_target(
                    hough_lines, right_p0, right_p1, right_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )
                hough_top_line, hough_top_info = select_best_hough_line_for_target(
                    hough_lines, top_p0, top_p1, top_band_u8, hough_edge_map_u8,
                    min_length_px=HOUGH_MIN_LENGTH_PX,
                    angle_thresh_deg=HOUGH_ANGLE_THRESH_DEG
                )

                print(f"   ⬚ Hough total segments: {hough_total_segments}")
                print(f"   ⬚ Hough left:  {hough_left_info}")
                print(f"   ⬚ Hough right: {hough_right_info}")
                print(f"   ⬚ Hough top:   {hough_top_info}")

                # -----------------------------------------------------------------
                # Hough-guided ortho warp:
                # align selected left/right/top lines to projected wall lines
                # -----------------------------------------------------------------
                if ENABLE_HOUGH_GUIDED_WARP:
                    if (hough_left_line is not None) and (hough_right_line is not None) and (hough_top_line is not None):
                        # projected wall lines from wall_poly_px order [b1, b2, t2, t1]
                        proj_left_line = np.vstack([wall_poly_px[3], wall_poly_px[0]]).astype(np.float64)
                        proj_right_line = np.vstack([wall_poly_px[1], wall_poly_px[2]]).astype(np.float64)
                        proj_top_line = np.vstack([wall_poly_px[3], wall_poly_px[2]]).astype(np.float64)

                        ortho_rgba = apply_hough_guided_ortho_warp(
                            ortho_rgba=ortho_rgba,
                            sel_left_line=hough_left_line.astype(np.float64),
                            sel_right_line=hough_right_line.astype(np.float64),
                            sel_top_line=hough_top_line.astype(np.float64),
                            proj_left_line=proj_left_line,
                            proj_right_line=proj_right_line,
                            proj_top_line=proj_top_line
                        )

                        print("   ⬚ Hough-guided ortho warp applied")
                    else:
                        print("   ⬚ Hough-guided ortho warp skipped (one or more selected lines missing)")

                    hough_warp_overlay_path = None
                    if SAVE_HOUGH_WARP_DEBUG:
                        hough_warp_overlay_path = Path(per_building_out) / name_for(
                            "hough_warp_overlay", base=geojson_base, wall=i_global, rec=rec,
                            heading=heading, pitch=pitch, fov=fov_deg
                        )
                        save_hough_warp_overlay(
                            img_pil=Image.fromarray(ortho_rgba).convert("RGBA"),
                            wall_quad_xy=wall_poly_px,
                            out_path=str(hough_warp_overlay_path)
                        )

                # freeze the pre-warp ortho result for debug overlays
                ortho_rgba_before_hough_warp = ortho_rgba.copy()

                if HOUGH_SAVE_BAND_MASKS:
                    Image.fromarray(left_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_left_band.png"
                    )
                    Image.fromarray(right_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_right_band.png"
                    )
                    Image.fromarray(top_band_u8 * 255).save(
                        Path(per_building_out) / f"{geojson_base}_wall{i_global:02d}_hough_top_band.png"
                    )

            ortho_prefit_overlay_path = Path(per_building_out) / name_for(
                "ortho_prefit_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_quad_fit_debug_overlay(
                img_pil=Image.fromarray(ortho_rgba_before_hough_warp).convert("RGBA"),
                wall_quad_xy=wall_poly_px,
                seg_main_quad_xy=ortho_seg_quad_fitted,
                seg_chunk_quads_xy=ortho_seg_chunk_quads_fitted,
                seg_contours=ortho_seg_contours_fitted,
                out_path=str(ortho_prefit_overlay_path),
            )

            hough_overlay_path = Path(per_building_out) / name_for(
                "hough_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_hough_all_lines_overlay(
                img_pil=Image.fromarray(ortho_rgba_before_hough_warp).convert("RGBA"),
                wall_quad_xy=wall_poly_px,
                all_lines=hough_lines,
                selected_left=hough_left_line,
                selected_right=hough_right_line,
                selected_top=hough_top_line,
                out_path=str(hough_overlay_path),
            )

            out_png_ortho = Path(per_building_out) / name_for(
                "ortho_png", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )

            # 2) then fill the remaining uncovered wall area
            if ENABLE_LAMA_FILL:
                lama_mask_path = out_png_ortho.with_name(out_png_ortho.stem + "_lama_mask.png") if LAMA_SAVE_DEBUG_MASK else None

                ortho_rgba, lama_hole_mask = lama_fill_rectified_wall(
                    ortho_rgba=ortho_rgba,
                    wall_poly_px=wall_poly_px,
                    debug_mask_path=str(lama_mask_path) if lama_mask_path is not None else None
                )

                filled_px = int((lama_hole_mask > 0).sum())
                if filled_px > 0:
                    print(f"   🦙 LaMa filled {filled_px} pixels on wall {i_global:02d}")

            # keep a FULL version for debug overlay (shows outside-the-wall parts too)
            ortho_rgba_full_debug = ortho_rgba.copy()

            # make a clipped version only for the actual baked texture if requested
            if FIT_CLIP_TO_WALL:
                wall_mask_clip = build_wall_region_mask(
                    ortho_rgba.shape[0], ortho_rgba.shape[1], wall_poly_px
                ) > 0
                ortho_rgba_texture = ortho_rgba.copy()
                ortho_rgba_texture[~wall_mask_clip, :3] = 0
                ortho_rgba_texture[~wall_mask_clip, 3] = 0
            else:
                ortho_rgba_texture = ortho_rgba.copy()

            Image.fromarray(ortho_rgba_texture).save(out_png_ortho)

            ortho_overlay_path = Path(per_building_out) / name_for(
                "ortho_overlay", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg
            )
            save_with_overlay(
                Image.fromarray(ortho_rgba_full_debug).convert("RGBA"),
                wall_poly_px,
                str(ortho_overlay_path)
            )

            W_img, H_img = img_rgb.size
            fx = fy = (W_img / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
            cx, cy = W_img / 2.0, H_img / 2.0

            quad_xyz_re = np.vstack([wall_quad[3], wall_quad[2], wall_quad[1], wall_quad[0]])  # t1,t2,b2,b1
            out_json_ortho = Path(per_building_out) / name_for(
                "ortho_meta", base=geojson_base, wall=i_global, rec=rec,
                heading=heading, pitch=pitch, fov=fov_deg)
            consolidated = {
                "type": "rectified_wall_texture",
                "version": "1.5-per-loop",
                "cropped_image": {
                    "mode": "rgba_band_in_memory",
                    "uv_quad_px_order": "b1,b2,t2,t1",
                    "uv_quad_px_raw": [[float(u), float(v)] for (u, v) in uv.tolist()],
                    "band_polygon_px": band_poly,
                    "band_bbox_px": list(band_bbox) if band_bbox else None,
                    "mosaic_used": bool(is_mosaic)
                },
                "intrinsics": {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx), "cy": float(cy),
                    "fov_deg": float(fov_deg),
                    "image_size_px": [int(W_img), int(H_img)]
                },
                "extrinsics": {
                    "camera_utm_xyz": [float(cam[0]), float(cam[1]), float(cam[2])],
                    "heading_deg": float(heading),
                    "pitch_deg": float(pitch),
                    "world_up": [0.0, 0.0, 1.0]
                },
                "wall_identity": {"component_id": int(cid) if cid is not None else -1,
                                  "loop_id": int(lid) if lid is not None else -1,
                                  "loop_index": int(k),
                                  "global_index": int(i_global)},
                "wall_geometry": {
                    "quad_xyz_order": ",".join(OUR_ORDER),
                    "quad_xyz": [[float(a),float(b),float(c)] for a,b,c in quad_xyz_re],
                    "center_xyz": [float(ctr[0]), float(ctr[1]), float(ctr[2])],
                    "metric_target": {
                        "order": "b1,b2,t2,t1",
                        "dst_wall_m": [[float(x), float(y)] for x,y in dst_m.tolist()],
                        **metric_meta
                    }
                },
                "rectification": {
                    "pixels_per_meter": float(PIXELS_PER_METER),
                    "margin_m": float(MARGIN_METERS),
                    "bounds_m": {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)},
                    "flip_vertical": bool(flip),
                    "homography_chain": {
                        "H_pix_to_wall_m": [[float(v) for v in row] for row in H_pix_to_wall_m.tolist()],
                        "S_m_to_px": [[float(v) for v in row] for row in S_m_to_px.tolist()],
                        "H_pix_to_ortho_px": [[float(v) for v in row] for row in H_pix_to_ortho_px.tolist()]
                    }
                },
                "artifacts": {
                    "sam3_alpha_overlay_png": str(Path(sam3_overlay_path).name),
                    "ortho_prefit_overlay_png": str(Path(ortho_prefit_overlay_path).name),
                    "hough_overlay_png": str(Path(hough_overlay_path).name),
                    "hough_warp_overlay_png": str(Path(hough_warp_overlay_path).name) if hough_warp_overlay_path is not None else None,
                    "ortho_png": str(Path(out_png_ortho).name),
                    "ortho_overlay_png": str(Path(ortho_overlay_path).name)
                },
                "source_sv": {
                    "pano_id": rec["pano_id"],
                    "street_view_url": f"https://maps.googleapis.com/maps/api/streetview?pano={rec['pano_id']}&size={SV_SIZE}&heading={heading:.4f}&pitch={pitch:.4f}&fov={fov_deg:.4f}&key=****",
                    "all_request_urls": [ _mask_key(u) for u in urls_fetched ],
                    "pano_lat": float(rec["lat"]),
                    "pano_lng": float(rec["lng"])
                },
                "notes": "Per-loop processing; coverage falls back to yaw/pitch tiling.",
                "post_rectification_quad_fit": {
                    "enabled": bool(ENABLE_ORTHO_QUAD_FIT),
                    "fit_mode": "uniform_scale_plus_translation_in_ortho_space",
                    "segmentation_quad_px": (
                        [[float(x), float(y)] for x, y in ortho_seg_quad.tolist()]
                        if ortho_seg_quad is not None else None
                    ),
                    "segmentation_quad_fitted_px": (
                        [[float(x), float(y)] for x, y in ortho_seg_quad_fitted.tolist()]
                        if ortho_seg_quad_fitted is not None else None
                    ),
                    "chunk_quadrilaterals_px": [
                        [[float(x), float(y)] for x, y in q.tolist()]
                        for q in ortho_seg_chunk_quads
                    ],
                    "chunk_quadrilaterals_fitted_px": [
                        [[float(x), float(y)] for x, y in q.tolist()]
                        for q in ortho_seg_chunk_quads_fitted
                    ],
                    "projected_wall_quad_px": [[float(x), float(y)] for x, y in wall_poly_px.tolist()],
                    "M_seg_scale_translate_2x3": [
                        [float(v) for v in row] for row in M_seg_scale_translate.tolist()
                    ],
                    "fit_info": ortho_fit_info,
                },
                "ortho_hough_line_detection": {
                    "enabled": bool(ENABLE_ORTHO_HOUGH_DEBUG),
                    "method": "canny_plus_houghlinesp_on_scaled_ortho_result",
                    "total_segments_detected": int(hough_total_segments),
                    "config": {
                        "search_band_px": int(HOUGH_SEARCH_BAND_PX),
                        "min_length_px": float(HOUGH_MIN_LENGTH_PX),
                        "max_gap_px": float(HOUGH_MAX_GAP_PX),
                        "angle_thresh_deg": float(HOUGH_ANGLE_THRESH_DEG),
                        "canny_low": int(HOUGH_CANNY_LOW),
                        "canny_high": int(HOUGH_CANNY_HIGH),
                        "canny_dilate_px": int(HOUGH_CANNY_DILATE_PX),
                        "use_clahe": bool(HOUGH_USE_CLAHE),
                    },
                    "left_line_px": (
                        [[float(x), float(y)] for x, y in hough_left_line.tolist()]
                        if hough_left_line is not None else None
                    ),
                    "right_line_px": (
                        [[float(x), float(y)] for x, y in hough_right_line.tolist()]
                        if hough_right_line is not None else None
                    ),
                    "top_line_px": (
                        [[float(x), float(y)] for x, y in hough_top_line.tolist()]
                        if hough_top_line is not None else None
                    ),
                    "left_info": hough_left_info,
                    "right_info": hough_right_info,
                    "top_info": hough_top_info,
                },
            }
            with open(out_json_ortho, "w", encoding="utf-8") as f:
                json.dump(consolidated, f, ensure_ascii=False, indent=2)

            print(f"   ✅ Saved overlays & texture: {Path(out_png_ortho).name}")

            # Update wall mesh with the rectified texture
            name = f"wall_c{cid}_l{lid}_w{i_global:02d}"
            uvs_px = np.vstack([  # [b1,b2,t2,t1]
                wall_poly_px[0],
                wall_poly_px[1],
                wall_poly_px[2],
                wall_poly_px[3],
            ]).astype(np.float64)
            tex_img = Image.open(out_png_ortho).convert("RGBA")
            mesh = mesh_by_name.get(name, None)
            if mesh is not None:
                uv = np.empty_like(uvs_px, dtype=np.float64)
                uv[:, 0] = uvs_px[:, 0] / float(out_Wr)
                uv[:, 1] = 1.0 - (uvs_px[:, 1] / float(out_Hr))
                mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)

    if SAVE_VIEWER_INDEX_JSON:
        # ---- Save viewer index + bundle for LOCAL debugging ----
        index_path = os.path.join(per_building_out, "viewer_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(viewer_index, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved viewer index: {index_path}")

        bundle_path = os.path.join(per_building_out, "viewer_bundle.npz")
        if len(all_wall_quads_global) > 0:
            wall_quads_np = np.stack(all_wall_quads_global, axis=0).astype(np.float64)
        else:
            wall_quads_np = np.zeros((0,4,3), dtype=np.float64)

        save_viewer_bundle_npz(
            bundle_path=bundle_path,
            corners_xyz=corners,
            id_to_idx=id_to_idx,
            edges_by_type=edge_groups,   # this is your edges dict from load_3d_geojson
            wall_quads_xyz_b1b2t2t1=wall_quads_np,
            wall_meta=all_wall_meta_global,
            viewer_index=viewer_index
        )
        print(f"✅ Saved viewer bundle: {bundle_path}")


    # --------------------- Update roof textures (masked by base loops) ---------------------
    if roof_meshes and geotiff_path and Path(geotiff_path).exists():
        try:
            # Closed base loops (in EPSG:25832) for masking
            base_edges_gdf = gdf[gdf['type'] == 'base'].copy()
            roof_loops_all = build_closed_roof_polygons(base_edges_gdf)  # List[Polygon]

            with rasterio.open(geotiff_path) as src:
                width, height = src.width, src.height
                inv = ~src.transform
                rgb = src.read([1, 2, 3])
                rgb = np.moveaxis(rgb, 0, -1).astype(np.uint8)

            # For each roof island, pick the polygon that contains its centroid (fallback: nearest)
            for rname, rmesh, rcoords in roof_meshes:
                # Compute UVs for this roof mesh from GeoTIFF grid
                uv_coords = []
                for x, y in rcoords[:, :2]:
                    col, row = inv * (x, y)
                    u = col / width
                    v = 1.0 - (row / height)
                    uv_coords.append([u, v])
                uv_roof = np.array(uv_coords, dtype=np.float64)

                # Find containing (or nearest) base polygon for masking
                cen = Point(float(rcoords[:, 0].mean()), float(rcoords[:, 1].mean()))
                chosen = None
                best_d = float("inf")
                for poly in roof_loops_all:
                    if poly.contains(cen):
                        chosen = poly
                        break
                    d = poly.distance(cen)
                    if d < best_d:
                        best_d = d
                        chosen = poly

                # Rasterize only the chosen polygon to create alpha for this island
                alpha_mask = rasterize_polygons_to_mask([chosen] if chosen is not None else [],
                                                        width, height, inv)

                # Compose per-island RGBA
                rgba = np.dstack([rgb, alpha_mask]).astype(np.uint8)
                texture_img = Image.fromarray(rgba, mode="RGBA")

                # Assign texture to this roof island
                rmesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv_roof, image=texture_img)

        except Exception as e:
            print(f"⚠️ Roof texture (masked) failed ({e}); keeping white roof.")


    # --------------------- Export Scene as GLB ---------------------
    if meshes_named:
        scene = trimesh.Scene()
        for name, m in meshes_named:
            scene.add_geometry(m, node_name=name)
        glb_path = Path(per_building_out) / name_for("glb", base=os.path.splitext(os.path.basename(geojson_path))[0])
        scene.export(glb_path)
        print(f"\n🟢 Exported textured GLB: {glb_path}")
    else:
        print("⚠️ No geometry to export (no meshes built).")

def main():
    ensure_outdir(OUTPUT_DIR)
    geojson_dir = Path(GEOJSON_DIR)
    geotiff_dir = Path(GEOTIFF_DIR)

    files = sorted(geojson_dir.glob("*.geojson"))
    if not files:
        print(f"No .geojson files found in: {geojson_dir}")
        return

    print(f"Found {len(files)} .geojson files. Starting batch...")

    device, processor, sam3_prompt_facade, sam3_prompt_roof = load_sam3(
    prompt_facade=SAM3_PROMPT_FACADE,
    prompt_roof=SAM3_PROMPT_ROOF
    )
    print(f"✅ SAM3 loaded once on device: {device} | facade_prompt={sam3_prompt_facade!r} | roof_prompt={sam3_prompt_roof!r}")



    for idx, gj in enumerate(files, 1):
        base = gj.stem  # e.g., "building_267160681_3d"
        roof_base = base[:-3] if base.endswith("_3d") else base
        tif_path  = geotiff_dir / f"{roof_base}.tif"
        tif_path_alt = geotiff_dir / f"{roof_base}.tiff"

        geotiff_for_this = None
        if tif_path.exists():
            geotiff_for_this = str(tif_path)
        elif tif_path_alt.exists():
            geotiff_for_this = str(tif_path_alt)
        else:
            print(f"⚠️ No matching GeoTIFF for {base} in {geotiff_dir} (looked for {roof_base}.tif/.tiff). Roof will remain white.")

        print(f"\n[{idx}/{len(files)}] Processing: {gj.name}")
        try:
            process_building(
                str(gj),
                OUTPUT_DIR,
                geotiff_path=geotiff_for_this,
                device=device,
                processor=processor,
                sam3_prompt_facade=sam3_prompt_facade,
                sam3_prompt_roof=sam3_prompt_roof
            )

            # Optional: free cached memory between buildings (useful on GPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Failed on {gj.name}: {e}")
            traceback.print_exc()


    print("\n✅ Batch complete.")
