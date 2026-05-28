from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .ellseg_adapter import EllSegAdapter, EllSegPaths


REQUIRED_MODULES = ["numpy", "cv2", "torch", "torchvision", "skimage", "scipy", "tqdm"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the Python environment for online EllSeg.")
    parser.add_argument("--ellseg-root", type=Path, default=EllSegPaths.root)
    parser.add_argument("--weights", type=Path, default=EllSegPaths.weights)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true", help="Load the model and run one synthetic frame.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    missing = []
    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if found else 'missing'}")
        if not found:
            missing.append(module)

    paths = EllSegPaths(root=args.ellseg_root, weights=args.weights)
    try:
        paths.validate()
        print(f"ellseg root: {paths.root}")
        print(f"weights: {paths.weights}")
    except Exception as exc:
        print(f"path check: failed: {exc}")
        return

    if missing:
        print("environment check: failed")
        print("missing modules: " + ", ".join(missing))
        return

    if args.smoke:
        import numpy as np

        adapter = EllSegAdapter(paths, device=args.device)
        smoke = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
        result = adapter.predict_gray(smoke)
        print(f"smoke status: {result['status']}")
        print(f"pupil: {result['pupil']}")

    print("environment check: ok")


if __name__ == "__main__":
    main()
