import base64
from pathlib import Path
import io

import imageio.v2 as imageio
import numpy as np


def png_to_data_url(path: str) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def rgba_to_data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    imageio.imwrite(buf, rgba.astype(np.uint8), format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
