from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .types import EllipseTuple, ellipse_to_tuple, invalid_ellipse


class EllSegImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class EllSegPaths:
    root: Path = Path(r"E:\Users\guocong1.wen\PycharmProjects\eye_test\ellseg_denseelnet")
    weights: Path = Path(r"E:\Users\guocong1.wen\PycharmProjects\eye_test\ellseg_denseelnet\weights\all.git_ok")

    def validate(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"EllSeg root does not exist: {self.root}")
        if not (self.root / "tools" / "ellseg_runtime.py").exists():
            raise FileNotFoundError(f"Missing EllSeg runtime: {self.root / 'tools' / 'ellseg_runtime.py'}")
        if not self.weights.exists():
            raise FileNotFoundError(f"Missing EllSeg weights: {self.weights}")


class EllSegAdapter:
    """Small runtime adapter around the existing EllSeg/DenseElNet repository."""

    def __init__(self, paths: EllSegPaths, device: str = "auto", ellipse_source: str = "network", dark_threshold: int = 20):
        paths.validate()
        self.paths = paths
        self.device_name = device
        self.ellipse_source = ellipse_source
        self.dark_threshold = dark_threshold
        self.runtime = self._import_runtime(paths.root)
        self.device = self.runtime.resolve_device(device)
        self.model = self.runtime.load_model(paths.weights, self.device)

    @staticmethod
    def _import_runtime(root: Path):
        root = root.resolve()
        tools = root / "tools"
        for path in (str(root), str(tools)):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            return importlib.import_module("ellseg_runtime")
        except ModuleNotFoundError as exc:
            raise EllSegImportError(
                "Failed to import EllSeg runtime dependencies. "
                "Install numpy, opencv-python, torch, torchvision, scipy, scikit-image, and tqdm in the selected environment."
            ) from exc

    def predict_gray(self, gray) -> Dict[str, object]:
        result = self.runtime.predict_frame(
            self.model,
            gray,
            self.device,
            ellipse_source=self.ellipse_source,
            dark_threshold=self.dark_threshold,
        )
        return {
            "status": str(result["status"]),
            "seg_map": result["seg_map"],
            "pupil": ellipse_to_tuple(result.get("pupil", invalid_ellipse())),
            "iris": ellipse_to_tuple(result.get("iris", invalid_ellipse())),
        }

    def overlay(self, gray, seg_map, pupil: EllipseTuple, iris: EllipseTuple):
        return self.runtime.overlay_prediction(gray, seg_map, pupil, iris)
