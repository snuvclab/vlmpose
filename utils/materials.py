import numpy as np
import pyrender


def make_blender_like_material(tri):
    base_color = [1.0, 1.0, 1.0, 1.0]
    base_tex = None

    if tri.visual.kind == "texture" and hasattr(tri.visual, "material"):
        mat = tri.visual.material

        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image

            # Force PIL images to RGB
            try:
                img = img.convert("RGB")
            except Exception:
                pass

            img = np.array(img)

            # Expand grayscale images to 3 channels
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Drop the alpha channel for RGBA images
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Specify the source channel string explicitly
            try:
                base_tex = pyrender.Texture(source=img, source_channels="RGB")
            except Exception:
                base_tex = None

        if hasattr(mat, "diffuse"):
            d = np.array(mat.diffuse[:3], dtype=np.float32)
            if d.max() > 1.0:
                d = d / 255.0
            base_color = [float(d[0]), float(d[1]), float(d[2]), 1.0]

    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color,
        baseColorTexture=base_tex,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
