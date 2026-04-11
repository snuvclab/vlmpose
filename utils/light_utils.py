import numpy as np
import pyrender

from utils.geometry import look_at_opengl


def add_area_like_light(scene, center, size, intensity, normal, camera_target):
    offsets = [np.array([size, size, 0.0])]

    # Rotate the offsets based on the normal axis
    if np.allclose(normal, [1, 0, 0]) or np.allclose(normal, [-1, 0, 0]):
        offsets = [o[[0, 2, 1]] for o in offsets]

    # For lights on positive axes, flip the offset sign to place them symmetrically
    n = tuple(normal)
    if n in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        offsets = [o * -1.0 for o in offsets]

    for off in offsets:
        pos = center + off
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity / 4.0)
        pose = look_at_opengl(pos, camera_target, up=(0.0, 0.0, 1.0))
        scene.add(light, pose=pose)
