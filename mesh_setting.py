from openai import OpenAI
import os
import argparse
import numpy as np
import math
import trimesh
import pyrender
from pyrender import RenderFlags
import imageio.v2 as imageio
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import json

from utils.config import MODEL
from utils.prompts import (
    INSTRUCTIONS_POSE_EST,
    INSTRUCTIONS_TARGET_SELECT1,
    INSTRUCTIONS_TARGET_SELECT2,
    INSTRUCTIONS_EVAL_FAITHFULNESS,
)
from utils.io_utils import png_to_data_url, rgba_to_data_url
from utils.scene_utils import load_all_obj_meshes
from utils.geometry import (
    compute_bbox_world,
    compute_bbox_world_from_scene,
    make_rotate,
    build_coordinate_arrows,
    make_view_pose_from_meshes,
    transform_bbox_aabb,
)
from utils.materials import make_blender_like_material
from utils.parsing import (
    PoseParseError,
    TargetSelectParseError,
    parse_pose_response,
    parse_target_select_response,
    parse_best_view_response,
    parse_best_view_response_v2,
)
from utils.drawing import image_numbering
from utils.light_utils import add_area_like_light


client = OpenAI()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_scene_dir", type=str, required=True)
    ap.add_argument("--text_prompt", type=str, required=True)
    ap.add_argument("--lens", type=int, default=70)
    ap.add_argument("--elevation", type=int, default=20)
    ap.add_argument("--light", type=int, default=15)
    ap.add_argument("--num_views", type=int, default=10)
    ap.add_argument("--render_width", type=int, default=1024)
    ap.add_argument("--render_height", type=int, default=1024)
    ap.add_argument("--sensor_width_mm", type=float, default=36.0)
    ap.add_argument("--camera_dist_ratio", type=float, default=2.0)
    return ap.parse_args()


def main():
    args = parse_args()

    target_scene_dir = args.target_scene_dir
    mesh_dir = os.path.join(target_scene_dir, "mesh")

    output_dir = os.path.join(target_scene_dir, "reasoning_process")
    os.makedirs(output_dir, exist_ok=True)

    H, W = args.render_height, args.render_width

    # Load meshes
    meshes, mesh_indices, obb_center_dict = load_all_obj_meshes(mesh_dir)
    mn, mx, center, bbox_dim, size = compute_bbox_world(meshes)

    camera_target = center.copy()
    camera_distance = size * 2.0

    fx = (W * float(args.lens)) / args.sensor_width_mm
    fy = (H * float(args.lens)) / args.sensor_width_mm
    cx = W * 0.5
    cy = H * 0.5

    ############################################## ---- Build pyrender scene ----
    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.64, 0.64, 0.64],
    )

    nodes_by_obj = {}
    node_pose = {}
    mesh_nodes = []

    for m, obj_idx in zip(meshes, mesh_indices):
        mat = make_blender_like_material(m)
        pm = pyrender.Mesh.from_trimesh(m, material=mat, smooth=False)
        #pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        node = scene.add(pm, pose=np.eye(4))
        nodes_by_obj.setdefault(obj_idx, []).append(node)
        node_pose[node] = np.eye(4)
        mesh_nodes.append(node)

    # ---- Lights: pyrender has no true area lights; approximate with 6 directional lights ----
    area = float(max(bbox_dim[0], bbox_dim[1]) * 10.0)
    add_area_like_light(scene, camera_target + np.array([0, 0, 300]), area, args.light, [0, 0, -1], camera_target)
    add_area_like_light(scene, camera_target + np.array([0, 0, -300]), area, args.light, [0, 0, 1], camera_target)
    add_area_like_light(scene, camera_target + np.array([0, -300, 0]), area, args.light, [0, 1, 0], camera_target)
    add_area_like_light(scene, camera_target + np.array([0, 300, 0]), area, args.light, [0, -1, 0], camera_target)
    add_area_like_light(scene, camera_target + np.array([-300, 0, 0]), area, args.light, [1, 0, 0], camera_target)
    add_area_like_light(scene, camera_target + np.array([300, 0, 0]), area, args.light, [-1, 0, 0], camera_target)

    # ---- Camera ----
    cam = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=0.01,
        zfar=max(1.0, size * 10.0),
    )
    cam_node = scene.add(cam, pose=np.eye(4))

    # Offscreen renderer
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    ###################### Task 1) select target object (mesh)
    init_selection_dir = os.path.join(output_dir, "init_selection")
    os.makedirs(init_selection_dir, exist_ok=True)

    num_views = args.num_views
    angle_increment = 360.0 / num_views
    elev = np.deg2rad(float(args.elevation))

    png_list = []
    color_list = []
    T_cw_bl_list = []
    for i in range(num_views):        
        angle = -90.0 + i * angle_increment + 180.0
        T_cw_bl, _, _, _ = make_view_pose_from_meshes(
            center, size,
            elev_deg=-float(args.elevation),
            azim_deg=angle,
            dist_scale=args.camera_dist_ratio,
            min_dist=0.01,
            up=(0.0, 1.0, 0.0),
            orbit_axis="y",   # raw OBJ
        )
        scene.set_pose(cam_node, pose=T_cw_bl)


        render_flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SKIP_CULL_FACES
        color, depth = r.render(scene, flags=render_flags)
        color_list.append(color)

        color = image_numbering(color, i)
        png_list.append(rgba_to_data_url(color))
        T_cw_bl_list.append(T_cw_bl)

    iter_llm_fails = 0
    while iter_llm_fails < 5:
        content = []
        for p in png_list:
            content.append({
                "type": "input_image",
                "image_url": p,
                "detail": "high"
            })

        req = dict(
            model=MODEL,
            instructions=INSTRUCTIONS_TARGET_SELECT1.format(N=num_views),
            input=[{"role": "user", "content": content}],
            reasoning={
                "effort": "medium"
            },
            text={
                "verbosity": "low"
            }
        )

        try:
            resp = client.responses.create(**req)
        except Exception:
            iter_llm_fails += 1
            continue

        try:
            parsed = parse_best_view_response(resp.output_text, N=num_views, strict=True)
            break
        except TargetSelectParseError:
            iter_llm_fails += 1
            continue

    if iter_llm_fails >= 5:
        print("VLM error: in target selection")
        return

    color = color_list[parsed["Image number"] - 1]
    select_view_path = os.path.join(init_selection_dir, "select_view.png")
    imageio.imwrite(select_view_path, color)
    T_cw_bl = T_cw_bl_list[parsed["Image number"] - 1]
    ### view selection: Done

    scene.set_pose(cam_node, pose=T_cw_bl)

    # (1) full-scene depth (for occlusion-aware visible mask)
    depth_full = r.render(
        scene,
        flags=RenderFlags.DEPTH_ONLY | RenderFlags.SKIP_CULL_FACES
    )
    if isinstance(depth_full, (tuple, list)):
        depth_full = depth_full[0]

    # (2) Prepare overlay
    base_rgb = color[..., :3] if color.ndim == 3 else color
    base = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # mild palette (not too aggressive)
    PALETTE = [
        (255, 80, 80), (80, 180, 255), (80, 220, 140), (255, 200, 80),
        (200, 120, 255), (255, 120, 200), (160, 160, 160), (120, 200, 255),
    ]


    alpha_fill = 70  # 0~255
    bbox_w = 1

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    # obj list (same order as load_all_obj_meshes: sorted)
    obj_files = [fn for fn in sorted(os.listdir(mesh_dir)) if fn.lower().endswith(".obj")]

    # depth compare tolerance
    eps = 1e-3

    for obj_i, fn in enumerate(obj_files):
        obj_path = os.path.join(mesh_dir, fn)

        # --- load OBJ as scene/mesh ---
        tm = trimesh.load(obj_path, force="scene", process=False)

        pieces = []
        if isinstance(tm, trimesh.Scene):
            for g in tm.dump():
                if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0:
                    gg = g.copy()
                    pieces.append(gg)
        elif isinstance(tm, trimesh.Trimesh) and len(tm.vertices) > 0:
            gg = tm.copy()
            pieces.append(gg)

        # Scene for rendering this object alone
        scene_obj = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.0, 0.0, 0.0])

        cam_obj = pyrender.IntrinsicsCamera(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            znear=0.01, zfar=max(1.0, float(size) * 10.0)
        )
        scene_obj.add(cam_obj, pose=T_cw_bl)

        for gg in pieces:
            pm = pyrender.Mesh.from_trimesh(gg, smooth=False)
            scene_obj.add(pm, pose=np.eye(4))

        # depth render
        depth_obj = r.render(
            scene_obj,
            flags=RenderFlags.DEPTH_ONLY | RenderFlags.SKIP_CULL_FACES
        )
        if isinstance(depth_obj, (tuple, list)):
            depth_obj = depth_obj[0]

        # visible mask
        mask = (depth_obj > 0) & (np.abs(depth_obj - depth_full) < eps)
        mask_u8 = (mask.astype(np.uint8) * 255)

        if not mask.any():
            continue

        # bbox from mask
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        color = PALETTE[obj_i % len(PALETTE)]

        # overlay fill (semi-transparent)
        color_layer = Image.new("RGBA", (W, H), (color[0], color[1], color[2], alpha_fill))
        overlay.paste(color_layer, (0, 0), Image.fromarray(mask_u8, mode="L"))

        # bbox
        draw.rectangle([x0, y0, x1, y1], outline=(color[0], color[1], color[2], 255), width=bbox_w)

        # label at bbox center
        cx_box = (x0 + x1) * 0.5
        cy_box = (y0 + y1) * 0.5
        txt = str(obj_i + 1)

        pad = 3

        # Center alignment: place both the bbox label background and text around (cx_box, cy_box) using "mm"
        try:
            tb = draw.textbbox((cx_box, cy_box), txt, font=font, anchor="mm")
            draw.rectangle([tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad], fill=(0, 0, 0, 170))
            draw.text((cx_box, cy_box), txt, fill=(255, 255, 255, 255), font=font, anchor="mm")
        except TypeError:
            tb0 = draw.textbbox((0, 0), txt, font=font)
            tw, th = tb0[2] - tb0[0], tb0[3] - tb0[1]
            x_txt = cx_box - tw * 0.5 - tb0[0]
            y_txt = cy_box - th * 0.5 - tb0[1]
            draw.rectangle([x_txt - pad, y_txt - pad, x_txt + tw + pad, y_txt + th + pad], fill=(0, 0, 0, 170))
            draw.text((x_txt, y_txt), txt, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(base, overlay).convert("RGB")


    overlay_path = os.path.join(init_selection_dir, "mask_overlay.png")
    out.save(overlay_path)

    user_text = f"Text description: '{args.text_prompt}'"
    png_list = [png_to_data_url(select_view_path), png_to_data_url(overlay_path)]

    iter_llm_fails = 0
    while iter_llm_fails < 5:
        content = [{"type": "input_text", "text": user_text}]
        for p in png_list:
            content.append({
                "type": "input_image",
                "image_url": p,
                "detail": "high"
            })

        req = dict(
            model=MODEL,
            instructions=INSTRUCTIONS_TARGET_SELECT2,
            input=[{"role": "user", "content": content}],
            reasoning={
                "effort": "low"
            },
            text={
                "verbosity": "low"
            }
        )

        try:
            resp = client.responses.create(**req)
        except Exception:
            iter_llm_fails += 1
            continue

        try:
            parsed = parse_target_select_response(resp.output_text, strict=True)
            break
        except TargetSelectParseError:
            iter_llm_fails += 1
            continue

    if iter_llm_fails >= 5:
        print("VLM error: in target selection")
        return

    target_idx = parsed["Target label"] - 1
    target_mesh = trimesh.load(os.path.join(mesh_dir, obj_files[target_idx]))
    target_pose_tm = np.eye(4)
    target_v0 = np.asarray(target_mesh.vertices)  # (N,3)
    bmin_tm, bmax_tm = target_mesh.bounds
    
    bbox_extents = bmax_tm - bmin_tm
    bbox_min = float(np.min(bbox_extents))
    bbox_max = float(np.max(bbox_extents))

    if bbox_min > 0.0:
        arrow_len = bbox_min if 2.0 * bbox_min <= bbox_max else bbox_min / 2.0
    else:
        arrow_len = bbox_max if bbox_max > 0.0 else 1.0
    
    related_labels = parsed["Related labels"]
    # target + related object indices (0-based)
    focus_obj_indices = {target_idx}
    for lbl in related_labels:
        idx = lbl - 1
        if 0 <= idx < len(obj_files):
            focus_obj_indices.add(idx)

    # select mesh pieces belonging to target/related objects
    focus_mask = [obj_idx in focus_obj_indices for obj_idx in mesh_indices]
    focus_meshes = [m for m, keep in zip(meshes, focus_mask) if keep]
    focus_nodes = [n for n, keep in zip(mesh_nodes, focus_mask) if keep]
    ###################### Task 1) End

    ###################### Task 2) pose estimation
    user_text = f"Text description: '{args.text_prompt}'"

    total_steps = 5
    pbar = tqdm(total=total_steps, desc="LLM iters", unit="iter")

    target_nodes = nodes_by_obj.get(target_idx, [])
    base_target_pose = node_pose[target_nodes[0]] if target_nodes else np.eye(4)

    arrows = build_coordinate_arrows((bmin_tm, bmax_tm), arrow_len=arrow_len)
    arrow_nodes = []
    for a in arrows:
        pm = pyrender.Mesh.from_trimesh(a, smooth=False)
        node = scene.add(pm, pose=base_target_pose)
        arrow_nodes.append(node)
        node_pose[node] = base_target_pose.copy()

    target_bbox_len = target_mesh.bounds[1, :] - target_mesh.bounds[0, :]
    bbox_center = (bmin_tm + bmax_tm) * 0.5

    pose_log = {}
    previous_id = None
    iter_llm = 0
    iter_llm_fails = 0
    render_flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SKIP_CULL_FACES
    while iter_llm < 5 and iter_llm_fails < 5:
        ###################### Task 2-1) Best view selection
        mn, mx, center, bbox_dim, size = compute_bbox_world_from_scene(focus_meshes, focus_nodes, node_pose)
        camera_target = center.copy()
        camera_distance = size * args.camera_dist_ratio
        cam.zfar = max(1.0, size * 10.0)
        
        png_list = []
        color_list = []
        T_cw_bl_list = []
        for i in range(num_views):
            #angle = np.deg2rad(-90.0 + i * angle_increment)
            #T_cw_bl = get_view_pose(center, camera_target, camera_distance, elev, angle)
            angle = -90.0 + i * angle_increment + 180.0
            T_cw_bl, _, _, _ = make_view_pose_from_meshes(
                center, size,
                elev_deg=-float(args.elevation),
                azim_deg=angle,
                dist_scale=args.camera_dist_ratio,
                min_dist=0.01,
                up=(0.0, 1.0, 0.0),
                orbit_axis="y",   # raw OBJ
            )
            scene.set_pose(cam_node, pose=T_cw_bl)
            
            color, depth = r.render(scene, flags=render_flags)
            bright_gain = 1.0
            color = np.clip(color.astype(np.float32) * bright_gain, 0, 255).astype(np.uint8)
            color_list.append(color)

            color = image_numbering(color, i)
            png_list.append(rgba_to_data_url(color))
            T_cw_bl_list.append(T_cw_bl)

        while iter_llm_fails < 5:
            content = [{"type": "input_text", "text": user_text}]
            for p in png_list:
                content.append({
                    "type": "input_image",
                    "image_url": p,
                    "detail": "high"
                })

            req = dict(
                model=MODEL,
                instructions=INSTRUCTIONS_EVAL_FAITHFULNESS.format(N=num_views),
                input=[{"role": "user", "content": content}],
                reasoning={
                    "effort": "medium"
                },
                text={
                    "verbosity": "low"
                }
            )
            if previous_id is not None:
                req["previous_response_id"] = previous_id

            try:
                resp = client.responses.create(**req)
            except Exception:
                print("[VLM API ERROR]", repr(e))
                iter_llm_fails += 1
                continue

            try:
                parsed = parse_best_view_response_v2(resp.output_text, N=num_views, strict=True)
                previous_id = resp.id
                break
            except TargetSelectParseError:
                iter_llm_fails += 1
                continue

        if iter_llm_fails >= 5:
            print("VLM error: in faithfulness check")
            return

        print(f"-------------------Iter {iter_llm+1}-------------------")
        print("Faithfulness:", parsed["Faithfulness"])
        print("Reasoning:", parsed["Reasoning"])
        print("\n")

        color = color_list[parsed["Image number"] - 1]
        select_view_path = os.path.join(output_dir, f"select_view_before_iter_{iter_llm+1:05d}.png")
        imageio.imwrite(select_view_path, color)
        T_cw_bl = T_cw_bl_list[parsed["Image number"] - 1]

        if parsed["Faithfulness"] == "Yes":
            break
        ###################### Task 2-1) End

        png_list = [png_to_data_url(select_view_path)]

        content = [{"type": "input_text", "text": user_text}]
        for p in png_list:
            content.append({
                "type": "input_image",
                "image_url": p,
                "detail": "high"
            })

        req = dict(
            model=MODEL,
            instructions=INSTRUCTIONS_POSE_EST,
            input=[{"role": "user", "content": content}],
            reasoning={
                "effort": "high"
            },
            text={
                "verbosity": "low"
            }
        )
        
        if previous_id is not None:
            req["previous_response_id"] = previous_id

        try:
            resp = client.responses.create(**req)
        except Exception:
            iter_llm_fails += 1
            pbar.set_postfix_str(f"ok={iter_llm}/5 fail={iter_llm_fails}/5 (api)")
            continue

        try:
            parsed = parse_pose_response(resp.output_text, strict=True)
            previous_id = resp.id
        except PoseParseError:
            iter_llm_fails += 1
            pbar.set_postfix_str(f"ok={iter_llm}/5 fail={iter_llm_fails}/5 (api)")
            continue

        print("Translation:", parsed["Translation"])
        print("Dominant rotation axis:", parsed["Dominant rotation axis"])
        print("Angle:", parsed["Angle"])
        print("Reasoning:", parsed["Reasoning"])
        print("\n\n")

        ################### Make transformation
        t_norm = np.asarray(parsed["Translation"])
        translation = t_norm * arrow_len
        
        if parsed["Dominant rotation axis"].lower() == "x":
            rot = make_rotate(math.radians(parsed["Angle"]), 0, 0)
        elif parsed["Dominant rotation axis"].lower() == "y":
            rot = make_rotate(0, math.radians(parsed["Angle"]), 0)
        elif parsed["Dominant rotation axis"].lower() == "z":
            rot = make_rotate(0, 0, math.radians(parsed["Angle"]))
        else:
            raise Exception("invalid axis")
        
        bmin_w, bmax_w = transform_bbox_aabb(bmin_tm, bmax_tm, target_pose_tm)
        bbox_center_world = (bmin_w + bmax_w) * 0.5
        
        # R(X_b - t_center) + t_center + t, X_b = R_b @ X_init + t_b ==> RR_b @ X_init + R(t_b - t_center) + t_center + t
        R_f = rot @ target_pose_tm[:3, :3]
        t_f = rot @ (target_pose_tm[:3, 3] - bbox_center_world) + bbox_center_world + translation
        target_pose_tm[:3, :3] = R_f
        target_pose_tm[:3, 3] = t_f
        pose_log[str(iter_llm + 1)] = target_pose_tm.tolist()
        
        transform_matrix_for_axis = np.eye(4)
        transform_matrix_for_axis[:3, 3] = translation
        ###################

        ####### for debuging
        debugging_mesh = target_mesh.copy()
        debugging_mesh.apply_transform(target_pose_tm)
        debug_path = os.path.join(output_dir, f"debug_after_iter_{iter_llm+1:05d}.obj")
        debugging_mesh.export(debug_path)
        #######

        for node in nodes_by_obj.get(target_idx, []):
            new_pose = target_pose_tm
            node_pose[node] = new_pose
            scene.set_pose(node, pose=new_pose)
            
        for node in arrow_nodes:
            scene.remove_node(node)
            node_pose.pop(node, None)
        arrow_nodes.clear()
        bmin_w, bmax_w = transform_bbox_aabb(bmin_tm, bmax_tm, target_pose_tm)

        arrows = build_coordinate_arrows((bmin_w, bmax_w), arrow_len=arrow_len)
        for a in arrows:
            pm = pyrender.Mesh.from_trimesh(a, smooth=False)
            node = scene.add(pm, pose=np.eye(4))
            arrow_nodes.append(node)
            node_pose[node] = np.eye(4)

        scene.set_pose(cam_node, pose=T_cw_bl)
        render_flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SKIP_CULL_FACES
        color, depth = r.render(scene, flags=render_flags)
        render_path = os.path.join(output_dir, f"select_view_after_iter_{iter_llm+1:05d}.png")
        imageio.imwrite(render_path, color)
        
        iter_llm += 1

        pbar.update(1)
        pbar.set_postfix_str(f"ok={iter_llm}/5 fail={iter_llm_fails}/5")

    pbar.close()
    ###################### Task 2) End
    r.delete()
    
    pose_log_path = os.path.join(output_dir, "target_pose_tm_by_iter.json")
    with open(pose_log_path, "w") as f:
        json.dump(pose_log, f, indent=2)


if __name__ == "__main__":
    main()
