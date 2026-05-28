from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .types import Vector3


@dataclass(frozen=True)
class GazeMapping:
    order: int
    coeff_x: List[float]
    coeff_y: List[float]
    feature_names: List[str]
    display_width: int
    display_height: int
    rmse_px: float
    source: str
    phi_wrap: bool = False
    feature_means: List[float] | None = None
    feature_scales: List[float] | None = None
    feature_clip: float = 4.0

    def predict_values(self, values: Dict[str, float]) -> Tuple[float, float]:
        values = dict(values)
        if self.phi_wrap and "phi_rad" in values and values["phi_rad"] < 0:
            values["phi_rad"] += 2.0 * math.pi
        vector = [float(values[name]) for name in self.feature_names]
        features = polynomial_features(self.normalize_values(vector), self.order)
        x = float(np.dot(np.asarray(self.coeff_x, dtype=float), features))
        y = float(np.dot(np.asarray(self.coeff_y, dtype=float), features))
        return x, y

    def normalize_values(self, values: Sequence[float]) -> List[float]:
        if (
            self.feature_means is None
            or self.feature_scales is None
            or len(self.feature_means) != len(values)
            or len(self.feature_scales) != len(values)
        ):
            return [float(value) for value in values]
        clip = float(self.feature_clip)
        normalized = []
        for value, mean, scale in zip(values, self.feature_means, self.feature_scales):
            scale = float(scale) if abs(float(scale)) > 1e-9 else 1.0
            z = (float(value) - float(mean)) / scale
            if math.isfinite(clip) and clip > 0.0:
                z = max(-clip, min(clip, z))
            normalized.append(z)
        return normalized

    def predict_angles(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
        return self.predict_values({"yaw_deg": yaw_deg, "pitch_deg": pitch_deg})

    def predict_normal(self, normal: Vector3) -> Tuple[float, float, float, float]:
        yaw, pitch = direction_angles_from_normal(normal)
        theta, phi = spherical_angles_from_normal(normal)
        x, y = self.predict_values(
            {
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "theta_rad": theta,
                "phi_rad": phi,
            }
        )
        return x, y, yaw, pitch


@dataclass(frozen=True)
class FusedGazePrediction:
    valid: bool
    x: float
    y: float
    reason: str
    left_x: float = math.nan
    left_y: float = math.nan
    right_x: float = math.nan
    right_y: float = math.nan
    left_weight: float = 0.0
    right_weight: float = 0.0
    disagreement_px: float = math.nan


@dataclass(frozen=True)
class BinocularGazeMapping:
    left: GazeMapping
    right: GazeMapping
    feature_names: List[str]
    display_width: int
    display_height: int
    rmse_px: float
    source: str
    max_disagreement_px: float = 250.0
    reject_weight_ratio: float = 1.25

    def predict_values(
        self,
        left_values: Dict[str, float] | None,
        right_values: Dict[str, float] | None,
        left_weight: float,
        right_weight: float,
        max_disagreement_px: float | None = None,
    ) -> FusedGazePrediction:
        left_point = None
        right_point = None
        if left_values is not None and left_weight > 0.0:
            left_point = self.left.predict_values(left_values)
        if right_values is not None and right_weight > 0.0:
            right_point = self.right.predict_values(right_values)
        return fuse_gaze_points(
            left_point,
            right_point,
            left_weight,
            right_weight,
            self.max_disagreement_px if max_disagreement_px is None else max_disagreement_px,
            self.reject_weight_ratio,
        )


def direction_angles_from_normal(normal: Vector3) -> Tuple[float, float]:
    """Convert a camera-facing virtual pupil normal to forward yaw/pitch degrees."""

    nx, ny, nz = [float(v) for v in normal]
    dx, dy, dz = -nx, -ny, -nz
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Invalid normal vector")
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    yaw = math.degrees(math.atan2(dx, dz))
    pitch = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))
    return yaw, pitch


def spherical_angles_from_normal(normal: Vector3) -> Tuple[float, float]:
    """Return the paper's spherical coordinates theta, phi for a pupil normal."""

    nx, ny, nz = [float(v) for v in normal]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Invalid normal vector")
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    theta = math.acos(max(-1.0, min(1.0, nz)))
    phi = math.atan2(ny, nx)
    return theta, phi


def angle_features(yaw_deg: float, pitch_deg: float, order: int) -> np.ndarray:
    return polynomial_features([yaw_deg, pitch_deg], order)


def polynomial_features(values: Sequence[float], order: int) -> np.ndarray:
    values = [float(value) for value in values]
    if order == 1:
        return np.array(values + [1.0], dtype=float)
    if order == 2:
        features: List[float] = []
        features.extend(values)
        features.extend([value * value for value in values])
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                features.append(values[i] * values[j])
        features.append(1.0)
        return np.array(features, dtype=float)
    raise ValueError("order must be 1 or 2")


def fit_mapping(
    samples: Sequence[Dict[str, float]],
    order: int,
    source: str,
    feature_names: Sequence[str] = ("yaw_deg", "pitch_deg"),
    phi_wrap: bool = False,
) -> GazeMapping:
    feature_names = list(feature_names)
    needed = feature_count(order, len(feature_names))
    if len(samples) < needed:
        raise ValueError(f"Need at least {needed} target samples for order {order}, got {len(samples)}")

    value_matrix = np.asarray([[sample[name] for name in feature_names] for sample in samples], dtype=float)
    feature_means = np.mean(value_matrix, axis=0)
    feature_scales = np.std(value_matrix, axis=0)
    feature_scales = np.where(feature_scales < 1e-9, 1.0, feature_scales)
    normalized_values = (value_matrix - feature_means) / feature_scales
    features = np.vstack([polynomial_features(row, order) for row in normalized_values])
    target_x = np.asarray([sample["target_x"] for sample in samples], dtype=float)
    target_y = np.asarray([sample["target_y"] for sample in samples], dtype=float)

    coeff_x, *_ = np.linalg.lstsq(features, target_x, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(features, target_y, rcond=None)
    pred_x = features @ coeff_x
    pred_y = features @ coeff_y
    rmse = float(np.sqrt(np.mean((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)))

    display_width = int(round(max(sample["display_width"] for sample in samples)))
    display_height = int(round(max(sample["display_height"] for sample in samples)))
    return GazeMapping(
        order=order,
        coeff_x=[float(v) for v in coeff_x],
        coeff_y=[float(v) for v in coeff_y],
        feature_names=feature_names,
        display_width=display_width,
        display_height=display_height,
        rmse_px=rmse,
        source=source,
        phi_wrap=phi_wrap,
        feature_means=[float(v) for v in feature_means],
        feature_scales=[float(v) for v in feature_scales],
    )


def feature_count(order: int, value_count: int = 2) -> int:
    if order == 1:
        return value_count + 1
    if order == 2:
        return value_count + value_count + value_count * (value_count - 1) // 2 + 1
    raise ValueError("order must be 1 or 2")


def save_mapping(path: Path, mapping: GazeMapping, samples: Sequence[Dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "single",
        "order": mapping.order,
        "coeff_x": mapping.coeff_x,
        "coeff_y": mapping.coeff_y,
        "feature_names": mapping.feature_names,
        "display_width": mapping.display_width,
        "display_height": mapping.display_height,
        "rmse_px": mapping.rmse_px,
        "source": mapping.source,
        "phi_wrap": mapping.phi_wrap,
        "feature_means": mapping.feature_means,
        "feature_scales": mapping.feature_scales,
        "feature_clip": mapping.feature_clip,
        "samples": list(samples),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_binocular_mapping(
    path: Path,
    mapping: BinocularGazeMapping,
    samples: Sequence[Dict[str, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "binocular",
        "fusion": "quality_weighted_point",
        "feature_names": mapping.feature_names,
        "display_width": mapping.display_width,
        "display_height": mapping.display_height,
        "rmse_px": mapping.rmse_px,
        "source": mapping.source,
        "max_disagreement_px": mapping.max_disagreement_px,
        "reject_weight_ratio": mapping.reject_weight_ratio,
        "left": _mapping_payload(mapping.left),
        "right": _mapping_payload(mapping.right),
        "samples": list(samples),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mapping(path: Path) -> GazeMapping:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "binocular":
        raise ValueError("This is a binocular mapping. Use load_mapping_model() instead.")
    return _mapping_from_payload(payload)


def load_mapping_model(path: Path) -> GazeMapping | BinocularGazeMapping:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "binocular":
        left = _mapping_from_payload(payload["left"])
        right = _mapping_from_payload(payload["right"])
        return BinocularGazeMapping(
            left=left,
            right=right,
            feature_names=list(payload.get("feature_names", left.feature_names)),
            display_width=int(payload["display_width"]),
            display_height=int(payload["display_height"]),
            rmse_px=float(payload.get("rmse_px", math.nan)),
            source=str(payload.get("source", "")),
            max_disagreement_px=float(payload.get("max_disagreement_px", 250.0)),
            reject_weight_ratio=float(payload.get("reject_weight_ratio", 1.25)),
        )
    return _mapping_from_payload(payload)


def _mapping_payload(mapping: GazeMapping) -> Dict[str, object]:
    return {
        "type": "single",
        "order": mapping.order,
        "coeff_x": mapping.coeff_x,
        "coeff_y": mapping.coeff_y,
        "feature_names": mapping.feature_names,
        "display_width": mapping.display_width,
        "display_height": mapping.display_height,
        "rmse_px": mapping.rmse_px,
        "source": mapping.source,
        "phi_wrap": mapping.phi_wrap,
        "feature_means": mapping.feature_means,
        "feature_scales": mapping.feature_scales,
        "feature_clip": mapping.feature_clip,
    }


def _mapping_from_payload(payload: Dict[str, object]) -> GazeMapping:
    return GazeMapping(
        order=int(payload["order"]),
        coeff_x=[float(v) for v in payload["coeff_x"]],
        coeff_y=[float(v) for v in payload["coeff_y"]],
        feature_names=list(payload.get("feature_names", ["yaw_deg", "pitch_deg"])),
        display_width=int(payload["display_width"]),
        display_height=int(payload["display_height"]),
        rmse_px=float(payload.get("rmse_px", math.nan)),
        source=str(payload.get("source", "")),
        phi_wrap=bool(payload.get("phi_wrap", False)),
        feature_means=[float(v) for v in payload["feature_means"]] if payload.get("feature_means") is not None else None,
        feature_scales=[float(v) for v in payload["feature_scales"]] if payload.get("feature_scales") is not None else None,
        feature_clip=float(payload.get("feature_clip", 4.0)),
    )


def fuse_gaze_points(
    left_point: Tuple[float, float] | None,
    right_point: Tuple[float, float] | None,
    left_weight: float,
    right_weight: float,
    max_disagreement_px: float = 250.0,
    reject_weight_ratio: float = 1.25,
) -> FusedGazePrediction:
    left_weight = _finite_nonnegative(left_weight)
    right_weight = _finite_nonnegative(right_weight)
    if left_point is None:
        left_weight = 0.0
    if right_point is None:
        right_weight = 0.0

    if left_weight <= 0.0 and right_weight <= 0.0:
        return FusedGazePrediction(False, math.nan, math.nan, "no_valid_eye")
    if left_weight > 0.0 and right_weight <= 0.0:
        return FusedGazePrediction(
            True,
            float(left_point[0]),
            float(left_point[1]),
            "left_only",
            left_x=float(left_point[0]),
            left_y=float(left_point[1]),
            left_weight=left_weight,
        )
    if right_weight > 0.0 and left_weight <= 0.0:
        return FusedGazePrediction(
            True,
            float(right_point[0]),
            float(right_point[1]),
            "right_only",
            right_x=float(right_point[0]),
            right_y=float(right_point[1]),
            right_weight=right_weight,
        )

    lx, ly = float(left_point[0]), float(left_point[1])
    rx, ry = float(right_point[0]), float(right_point[1])
    disagreement = math.hypot(lx - rx, ly - ry)
    if math.isfinite(disagreement) and disagreement > max_disagreement_px:
        ratio = max(reject_weight_ratio, 1.0)
        if left_weight >= right_weight * ratio:
            return FusedGazePrediction(
                True,
                lx,
                ly,
                "right_rejected_disagreement",
                lx,
                ly,
                rx,
                ry,
                left_weight,
                right_weight,
                disagreement,
            )
        if right_weight >= left_weight * ratio:
            return FusedGazePrediction(
                True,
                rx,
                ry,
                "left_rejected_disagreement",
                lx,
                ly,
                rx,
                ry,
                left_weight,
                right_weight,
                disagreement,
            )
        return FusedGazePrediction(
            False,
            math.nan,
            math.nan,
            "binocular_disagreement",
            lx,
            ly,
            rx,
            ry,
            left_weight,
            right_weight,
            disagreement,
        )

    total = left_weight + right_weight
    x = (left_weight * lx + right_weight * rx) / total
    y = (left_weight * ly + right_weight * ry) / total
    return FusedGazePrediction(
        True,
        x,
        y,
        "quality_weighted",
        lx,
        ly,
        rx,
        ry,
        left_weight,
        right_weight,
        disagreement,
    )


def _finite_nonnegative(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


def clip_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    px = int(round(max(0.0, min(float(width - 1), x))))
    py = int(round(max(0.0, min(float(height - 1), y))))
    return px, py


def smooth_point(
    previous: Tuple[float, float] | None,
    current: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    if previous is None:
        return current
    alpha = max(0.0, min(1.0, float(alpha)))
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )
