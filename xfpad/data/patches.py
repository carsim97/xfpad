"""Minutiae-guided 224x224 patch extraction using NIST MINDTCT.

Public entry point: extract_patch(image_path, dst_path, ...). Wraps a single
mindtct invocation and crops the input image around the median minutiae
coordinate.

The mindtct binary is bundled at the repo root (or can be passed explicitly).
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Default search locations for the mindtct binary.
_BINARY_NAMES = {
    "Windows": "mindtct.exe",
    "Linux": "mindtct",
    "Darwin": "mindtct",
}


def find_mindtct(explicit: Optional[str] = None) -> Path:
    """Locate the mindtct binary.

    Resolution order:
        1. explicit argument
        2. environment variable XFPAD_MINDTCT
        3. binary named after the host OS in the current working directory
        4. shutil.which('mindtct') (system PATH)
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"mindtct binary not found at: {p}")
        return p

    env = os.environ.get("XFPAD_MINDTCT")
    if env and Path(env).exists():
        return Path(env)

    name = _BINARY_NAMES.get(platform.system(), "mindtct")
    local = Path.cwd() / name
    if local.exists():
        return local

    onpath = shutil.which("mindtct")
    if onpath:
        return Path(onpath)

    raise FileNotFoundError(
        "mindtct binary not found. Place it in the repo root, set the "
        "XFPAD_MINDTCT environment variable, or pass --binary explicitly."
    )


def extract_patch(image_path: str | Path,
                  dst_path: str | Path,
                  patch_size: int = 224,
                  image_size: int = 500,
                  mindtct_binary: Optional[str | Path] = None,
                  workdir: Optional[str | Path] = None) -> Path:
    """Extract a single minutiae-centred patch and save it to dst_path.

    Mirrors the logic of the original extract_patches.get_patch:
        - convert to grayscale
        - run mindtct -m1 -b on the temporary PNG
        - read the resulting .xyt; if empty fall back to image centre
        - crop a patch of size `patch_size` around the median minutiae
          coordinate, clamped to the image boundaries

    Parameters
    ----------
    image_path     : input fingerprint image (any PIL-readable format).
    dst_path       : output PNG/BMP path; parent dirs are created.
    patch_size     : side length of the square patch.
    image_size     : assumed full-image side length used for boundary
                     clamping (must match the original code).
    mindtct_binary : optional explicit path to the mindtct executable.
    workdir        : directory where mindtct intermediate files are
                     written (default: tempfile-style next to dst_path).

    Returns
    -------
    The destination path.
    """
    src = Path(image_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    binary = find_mindtct(str(mindtct_binary) if mindtct_binary else None)

    work = Path(workdir) if workdir else dst.parent
    work.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    tmp_png = work / f"{stem}_tmp.png"
    tmp_out = work / f"{stem}_tmp"  # mindtct will append .xyt, .brw etc.

    img = Image.open(src).convert("L")
    img.save(tmp_png)

    # Make sure mindtct is executable (no-op on Windows).
    try:
        os.chmod(binary, 0o755)
    except PermissionError:
        pass

    cmd = [str(binary), "-m1", "-b", str(tmp_png), str(tmp_out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    xyt = tmp_out.with_suffix(".xyt")
    if xyt.exists():
        with xyt.open("r") as f:
            coords = [list(map(int, line.split()[:2])) for line in f if line.strip()]
    else:
        coords = []

    half = patch_size // 2
    if not coords:
        center = [image_size // 2, image_size // 2]
    else:
        arr = np.array(coords)
        center = [int(np.median(arr[:, 0])), int(np.median(arr[:, 1]))]
        center[0] = max(half, min(center[0], image_size - half))
        center[1] = max(half, min(center[1], image_size - half))

    # Cleanup mindtct outputs and the temporary PNG.
    for path in glob.glob(str(tmp_out) + ".*"):
        try:
            os.remove(path)
        except OSError:
            pass
    if tmp_png.exists():
        os.remove(tmp_png)

    patch = img.crop((
        center[0] - half, center[1] - half,
        center[0] + half, center[1] + half,
    ))
    patch.save(dst)
    return dst
