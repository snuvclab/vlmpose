import math
import numpy as np
import trimesh
from typing import List, Tuple


def compute_bbox_world(meshes):
    all_v = np.concatenate([m.vertices for m in meshes], axis=0)
    mn = all_v.min(axis=0)
    mx = all_v.max(axis=0)
    center = (mn + mx) * 0.5
    bbox_dim = (mx - mn)
    size = float(np.linalg.norm(bbox_dim))
    return mn, mx, center, bbox_dim, size


def compute_bbox_world_from_scene(meshes, mesh_nodes, node_pose):
    if len(meshes) == 0:
        raise RuntimeError("No meshes for bbox recompute.")
    all_v = []
    for m, node in zip(meshes, mesh_nodes):
        T = node_pose.get(node, np.eye(4))
        R = T[:3, :3]
        t = T[:3, 3]
        v = (R @ m.vertices.T).T + t[None, :]
        all_v.append(v)
    all_v = np.concatenate(all_v, axis=0)
    mn = all_v.min(axis=0)
    mx = all_v.max(axis=0)
    center = (mn + mx) * 0.5
    bbox_dim = (mx - mn)
    size = float(np.linalg.norm(bbox_dim))
    return mn, mx, center, bbox_dim, size


def look_at_opengl(eye, target, up=(0.0, 0.0, 1.0)):
    """
    Returns T_cw (camera->world) pose in OpenGL camera convention used by pyrender:
      +X right, +Y up, +Z points away from the scene (camera looks along -Z).
    """
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    z = eye - target
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-12:
        raise ValueError("eye and target are too close for look_at.")
    z = z / z_norm  # camera +Z axis (away from scene)

    x = np.cross(up, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        # up parallel to z; pick another up
        up2 = np.array([0.0, 0.0, 1.0])
        x = np.cross(up2, z)
        x = x / (np.linalg.norm(x) + 1e-12)
    else:
        x = x / x_norm  # camera +X axis

    y = np.cross(z, x)  # camera +Y axis

    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def make_transformation(center, translation, rot_axis, rot_angle):
    # rot_axis: 'x'/'y'/'z' or 3-vector (any direction)
    if isinstance(rot_axis, str):
        if rot_axis == "x":
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif rot_axis == "y":
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif rot_axis == "z":
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise Exception("Unknown axis string")
    else:
        axis = np.asarray(rot_axis, dtype=np.float64)

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise Exception("Rotation axis vector is near zero")
    axis = axis / axis_norm

    theta = math.radians(rot_angle)
    ax, ay, az = axis[0], axis[1], axis[2]

    # Rodrigues' rotation formula
    K = np.array(
        [
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0],
        ],
        dtype=np.float64,
    )

    R = np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation + center - R @ center
    return T


def aabb_extents_after_pose(verts0: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    verts0: (N,3) in trimesh object coords
    T: (4,4) pose in trimesh/world coords
    return: extents (3,) of AABB in trimesh/world axes
    """
    R = T[:3, :3]
    t = T[:3, 3]
    v = (R @ verts0.T).T + t[None, :]
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    return (mx - mn)

def set_mesh_color(mesh: trimesh.Trimesh, rgb: Tuple[int, int, int]) -> None:
    rgba = np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))


def darken_color(rgb: Tuple[int, int, int], ratio: float = 0.25) -> Tuple[int, int, int]:
    color = np.asarray(rgb, dtype=np.float64)
    dark = np.clip(color * (1.0 - ratio), 0, 255).astype(np.uint8)
    return int(dark[0]), int(dark[1]), int(dark[2])


def build_arrow(
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    body_color: Tuple[int, int, int],
) -> trimesh.Trimesh:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    length = float(length)
    head_len = max(length * 0.28, 1e-5)
    body_len = max(length - head_len, 1e-5)
    radius = max(length * 0.03, 1e-5)
    head_radius = radius * 1.9

    start = np.asarray(start, dtype=np.float64)
    body_end = start + direction * body_len

    body = trimesh.creation.cylinder(
        radius=radius, segment=np.vstack([start, body_end]), sections=32
    )
    set_mesh_color(body, body_color)

    head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=32)
    align = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    if align is None:
        align = np.eye(4)
    head.apply_transform(align)
    head.apply_translation(body_end)
    set_mesh_color(head, darken_color(body_color, ratio=0.35))

    return trimesh.util.concatenate([body, head])


def build_coordinate_arrows(bounds, arrow_len=None) -> List[trimesh.Trimesh]:
    bmin, bmax = bounds
    center = (bmin + bmax) * 0.5
    extents = bmax - bmin

    if arrow_len is None:
        arrow_len = np.min(extents) / 2 if np.max(extents) < 2 * np.min(extents) else np.min(extents)
        if arrow_len <= 0:
            arrow_len = float(np.max(extents))
        if arrow_len <= 0:
            arrow_len = 1.0

    arrows: List[trimesh.Trimesh] = []

    # Front face start, +Z direction (far -> near).
    blue_start = np.array([center[0], center[1], bmax[2]], dtype=np.float64)
    arrows.append(build_arrow(blue_start, np.array([0.0, 0.0, 1.0]), arrow_len, (0, 0, 255)))

    # Right face start, +X direction (left -> right).
    red_start = np.array([bmax[0], center[1], center[2]], dtype=np.float64)
    arrows.append(
        build_arrow(red_start, np.array([1.0, 0.0, 0.0]), arrow_len, (255, 0, 0))
    )

    # Top face start, +Y direction (bottom -> up).
    green_start = np.array([center[0], bmax[1], center[2]], dtype=np.float64)
    arrows.append(build_arrow(green_start, np.array([0.0, 1.0, 0.0]), arrow_len, (0, 255, 0)))

    return arrows

def transform_bbox_aabb(bmin, bmax, T):
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmax[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmax[0], bmax[1], bmax[2]],
    ], dtype=np.float64)
    corners_h = np.concatenate([corners, np.ones((8,1))], axis=1)
    world = (T @ corners_h.T).T[:, :3]
    bmin_w = world.min(axis=0)
    bmax_w = world.max(axis=0)
    return bmin_w, bmax_w
    
def rodrigues(axis, theta):
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def adjust_camera_pitch_image_axis(T_cw_gl, target_xyz, delta_deg):
    # camera basis
    right = T_cw_gl[:3,0]
    up = T_cw_gl[:3,1]

    cam_pos = T_cw_gl[:3,3]
    v = cam_pos - target_xyz  # vector from target to camera

    # rotate v around camera right axis
    R = rodrigues(right, np.radians(delta_deg))
    v_new = R @ v
    cam_pos_new = target_xyz + v_new

    # also rotate up vector consistently
    up_new = R @ up

    # rebuild pose
    T_new = look_at_opengl(cam_pos_new, target_xyz, up=up_new)
    return T_new
'''
def make_view_pose_from_meshes(center, size, elev_deg=20.0, azim_deg=0.0, dist_scale=2.0, min_dist=0.25, up=(0.0,0.0,1.0)):
    """
    pcd_list: list of (N,3) or (N,>=3) arrays in base/world frame
    elev_deg: image-frame y-axis elevation (pitch around camera right axis)
    azim_deg: azimuth around world Z axis (xy-plane angle)
    dist_scale: camera distance = bbox_size * dist_scale
    min_dist: lower bound of camera distance
    return: T_cw_gl, center, size, dist
    """
    dist = max(size * dist_scale, min_dist)

    # azimuth in world XY plane (Z up)
    az = np.radians(float(azim_deg))
    eye = np.array([
        center[0] + dist * np.cos(az),
        center[1] + dist * np.sin(az),
        center[2] + 0.0,
    ])

    # base look-at (OpenGL camera convention)
    T = look_at_opengl(eye, center, up=up)

    # image-y elevation (rotate around camera right axis)
    if abs(elev_deg) > 1e-6:
        T = adjust_camera_pitch_image_axis(T, center, delta_deg=float(elev_deg))

    return T, center, size, dist
'''

def make_view_pose_from_meshes(
    center, size,
    elev_deg=20.0,
    azim_deg=0.0,
    dist_scale=2.0,
    min_dist=0.25,
    up=(0.0,0.0,1.0),
    orbit_axis="z",
):
    dist = max(size * dist_scale, min_dist)
    az = np.radians(float(azim_deg))

    if orbit_axis == "z":
        # GL: z-up
        eye = np.array([
            center[0] + dist * np.cos(az),
            center[1] + dist * np.sin(az),
            center[2] + 0.0,
        ])
    elif orbit_axis == "y":
        # raw OBJ: y-up
        eye = np.array([
            center[0] + dist * np.cos(az),
            center[1] + 0.0,
            center[2] + dist * np.sin(az),
        ])
    else:
        raise ValueError("orbit_axis must be 'z' or 'y'")

    T = look_at_opengl(eye, center, up=up)
    if abs(elev_deg) > 1e-6:
        T = adjust_camera_pitch_image_axis(T, center, delta_deg=float(elev_deg))
    return T, center, size, dist