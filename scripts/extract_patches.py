"""Pre-processing utility — extract 224x224 minutiae-guided patches.

Reads a list of source fingerprint images from a split file, runs NIST
MINDTCT to detect minutiae, and saves a 224x224 patch centred on the
median minutiae coordinate (with image-centre fallback when no minutiae
are found, matching the protocol in the paper).

The output directory layout mirrors the input layout, rooted at
--output-root and stripping the longest leading directory whose
components are common across all input paths.

Usage examples
--------------
    python scripts/extract_patches.py \\
        --split-file data/splits/greenbit_train.txt \\
        --output-root images/greenbit_train

    # Optional explicit binary location (Linux):
    python scripts/extract_patches.py \\
        --split-file data/splits/dermalog_test.txt \\
        --output-root images/dermalog_test \\
        --binary ./mindtct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xfpad.data.patches import extract_patch  # noqa: E402
from xfpad.utils import ensure_dir, get_logger, read_split  # noqa: E402

LOG = get_logger("extract_patches")


def _common_root(paths: List[str]) -> str:
    """Longest leading directory shared by all paths (empty if none)."""
    if not paths:
        return ""
    parts = [Path(p).parts for p in paths]
    common: list[str] = []
    for components in zip(*parts):
        if len(set(components)) == 1:
            common.append(components[0])
        else:
            break
    return str(Path(*common)) if common else ""


def _resolve_dst(src: str, common: str, output_root: Path) -> Path:
    rel = Path(src)
    try:
        rel = Path(src).relative_to(common) if common else Path(src)
    except ValueError:
        rel = Path(src).name  # fallback if src is not under common
    return output_root / rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract minutiae-guided 224x224 patches.")
    parser.add_argument("--split-file", required=True,
                        help="Text file with one source path per line.")
    parser.add_argument("--output-root", required=True,
                        help="Destination root directory.")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=500,
                        help="Assumed full-image side (used for boundary clamping).")
    parser.add_argument("--binary", default=None,
                        help="Path to the mindtct binary (default: auto-detect).")
    parser.add_argument("--workdir", default=None,
                        help="Where mindtct intermediate files are written "
                             "(default: alongside each output image).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N entries (debugging).")
    args = parser.parse_args()

    paths = read_split(args.split_file)
    if args.limit:
        paths = paths[:args.limit]

    common = _common_root(paths)
    out_root = ensure_dir(args.output_root)
    LOG.info("Extracting %d patches  root=%s  -> %s", len(paths), common, out_root)

    workdir = Path(args.workdir) if args.workdir else None
    failures = 0
    for src in tqdm(paths, desc="patches"):
        try:
            dst = _resolve_dst(src, common, out_root)
            dst = dst.with_suffix(".png")
            extract_patch(
                image_path=src,
                dst_path=dst,
                patch_size=args.patch_size,
                image_size=args.image_size,
                mindtct_binary=args.binary,
                workdir=workdir,
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("Failed for %s: %s", src, e)
            failures += 1

    LOG.info("Done. %d processed, %d failed.", len(paths) - failures, failures)


if __name__ == "__main__":
    main()
