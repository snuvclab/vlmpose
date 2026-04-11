import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def preprocess_open6dor_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)
    width, height = image.size
    edge_width = int(width * 0.1)
    edge_height = int(height * 0.1)
    white_image = Image.new("RGB", (width, height), "white")
    white_image.paste(image.crop((edge_width, edge_height, width - edge_width, height - edge_height * 2)),
                      (edge_width, edge_height))
    return white_image


def image_numbering(color, idx):
    # --- draw top-left label ---
    img_pil = Image.fromarray(color)  # color is (H,W,3 or 4)
    img_pil = img_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_pil)

    label = str(idx + 1)

    box_size = 38     # black square side length in pixels (reduce to 16-20 if needed)
    font_size = 30    # font size (adjust to match box_size)
    pad = 0           # start exactly at (0, 0) with no padding

    # Load the font, or fall back to the default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # black square at top-left (no padding)
    draw.rectangle([pad, pad, pad + box_size - 1, pad + box_size - 1], fill=(0, 0, 0, 255))

    # centered white text inside square
    cx = pad + box_size / 2.0
    cy = pad + box_size / 2.0
    try:
        # If Pillow is recent enough, use anchor support for true center alignment
        draw.text((cx, cy), label, fill=(255, 255, 255, 255), font=font, anchor="mm")
    except TypeError:
        # Fallback for older Pillow versions
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = (tb[2] - tb[0]), (tb[3] - tb[1])
        x = pad + (box_size - tw) / 2.0 - tb[0]
        y = pad + (box_size - th) / 2.0 - tb[1]
        draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)

    # back to numpy for saving
    color_out = np.array(img_pil.convert("RGB"))
    return color_out

def draw_axes_overlay(color, origin=(20, 80), axis_len=80, z_tilt=(-0.2, 0.2), z_len=80, line_w=3):
    """
    color: (H,W,3) or (H,W,4) numpy array
    origin: axis start point (x, y) in image coordinates (top-left origin)
    axis_len: length of the x/y axes
    z_tilt: (dx, dy) ratio for the z-axis (draw forward as a diagonal)
    """
    if color.ndim != 3 or color.shape[2] not in (3, 4):
        raise ValueError("color must be HxWx3 or HxWx4")

    rgb = color[..., :3].copy()
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # font (use the default if unavailable)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    ox, oy = origin

    # axis endpoints
    x_end = (ox + axis_len, oy)
    y_end = (ox, oy - axis_len)

    zx, zy = float(z_tilt[0]), float(z_tilt[1])
    znorm = math.hypot(zx, zy)
    if znorm < 1e-6:
        zx, zy = 0.0, 1.0
        znorm = 1.0
    zx /= znorm
    zy /= znorm
    z_end = (ox + int(z_len * zx), oy + int(z_len * zy))

    # draw arrows
    draw.line([ox, oy, x_end[0], x_end[1]], fill=(255, 0, 0), width=line_w)
    draw.line([ox, oy, y_end[0], y_end[1]], fill=(0, 255, 0), width=line_w)
    draw.line([ox, oy, z_end[0], z_end[1]], fill=(0, 0, 255), width=line_w)

    # arrowheads (small triangles)
    def arrow_head(p0, p1, size=8, color=(0, 0, 0)):
        # Small triangle pointing from p0 to p1
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy) + 1e-9
        ux, uy = dx / norm, dy / norm
        # Perpendicular vector
        vx, vy = -uy, ux
        tip = (p1[0], p1[1])
        left = (p1[0] - ux * size + vx * (size * 0.6), p1[1] - uy * size + vy * (size * 0.6))
        right = (p1[0] - ux * size - vx * (size * 0.6), p1[1] - uy * size - vy * (size * 0.6))
        draw.polygon([tip, left, right], fill=color)

    # arrowheads in same axis color
    arrow_head((ox, oy), x_end, size=8, color=(255, 0, 0))
    arrow_head((ox, oy), y_end, size=8, color=(0, 255, 0))
    arrow_head((ox, oy), z_end, size=8, color=(0, 0, 255))

    # labels (black text) near arrow tips
    draw.text((x_end[0] + 6, x_end[1] - 6), "x", fill=(0, 0, 0), font=font)
    draw.text((y_end[0] - 10, y_end[1] - 18), "y", fill=(0, 0, 0), font=font)
    draw.text((z_end[0] + 6, z_end[1] + 2), "z", fill=(0, 0, 0), font=font)

    out_rgb = np.array(img)
    if color.shape[2] == 4:
        return np.concatenate([out_rgb, color[..., 3:4]], axis=2)
    return out_rgb
