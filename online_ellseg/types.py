from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


EllipseTuple = Tuple[float, float, float, float, float]
Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Roi:
    """Rectangular ROI in full-frame pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_string(cls, value: str) -> "Roi":
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI must be formatted as x,y,w,h")
        return cls(*parts)

    def clamp(self, frame_shape: Tuple[int, int]) -> "Roi":
        height, width = frame_shape[:2]
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return Roi(x, y, w, h)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def crop(self, frame):
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass
class PupilQuality:
    valid: bool
    area: float
    axis_ratio: float
    center_jump: float
    reasons: List[str]

    def reason_text(self) -> str:
        return "ok" if self.valid else "|".join(self.reasons)


@dataclass
class VirtualAxisEstimate:
    valid: bool
    reason: str
    center_ray: Vector3
    normal: Vector3
    alpha: float
    candidate0: Vector3
    candidate1: Vector3
    selected_index: int

    @classmethod
    def invalid(cls, reason: str) -> "VirtualAxisEstimate":
        nan3 = (math.nan, math.nan, math.nan)
        return cls(False, reason, nan3, nan3, math.nan, nan3, nan3, -1)


@dataclass
class EllipseObservation:
    frame_index: int
    timestamp: float
    roi: Roi
    status: str
    pupil_roi: EllipseTuple
    iris_roi: EllipseTuple
    pupil_frame: EllipseTuple
    iris_frame: EllipseTuple
    quality: PupilQuality
    filtered_pupil_frame: Optional[EllipseTuple] = None
    virtual_axis: Optional[VirtualAxisEstimate] = None
    real_axis: Optional[Any] = None
    seg_map: Optional[Any] = None

    @property
    def valid(self) -> bool:
        return self.quality.valid

    def csv_row(self) -> Dict[str, object]:
        pupil = self.pupil_frame
        iris = self.iris_frame
        filtered = self.filtered_pupil_frame or (math.nan, math.nan, math.nan, math.nan, math.nan)
        axis = self.virtual_axis or VirtualAxisEstimate.invalid("disabled")
        real_axis = self.real_axis
        if real_axis is None:
            real_axis = _invalid_real_axis("disabled")
        return {
            "frame": self.frame_index,
            "timestamp": f"{self.timestamp:.6f}",
            "status": self.status,
            "valid": int(self.valid),
            "quality_reason": self.quality.reason_text(),
            "roi_x": self.roi.x,
            "roi_y": self.roi.y,
            "roi_w": self.roi.w,
            "roi_h": self.roi.h,
            "pupil_cx": pupil[0],
            "pupil_cy": pupil[1],
            "pupil_axis_a": pupil[2],
            "pupil_axis_b": pupil[3],
            "pupil_theta": pupil[4],
            "iris_cx": iris[0],
            "iris_cy": iris[1],
            "iris_axis_a": iris[2],
            "iris_axis_b": iris[3],
            "iris_theta": iris[4],
            "filtered_pupil_cx": filtered[0],
            "filtered_pupil_cy": filtered[1],
            "filtered_pupil_axis_a": filtered[2],
            "filtered_pupil_axis_b": filtered[3],
            "filtered_pupil_theta": filtered[4],
            "pupil_area": self.quality.area,
            "pupil_axis_ratio": self.quality.axis_ratio,
            "pupil_center_jump": self.quality.center_jump,
            "virtual_axis_valid": int(axis.valid),
            "virtual_axis_reason": axis.reason,
            "virtual_axis_selected": axis.selected_index,
            "virtual_axis_alpha_rad": axis.alpha,
            "virtual_axis_nx": axis.normal[0],
            "virtual_axis_ny": axis.normal[1],
            "virtual_axis_nz": axis.normal[2],
            "virtual_axis_ray_x": axis.center_ray[0],
            "virtual_axis_ray_y": axis.center_ray[1],
            "virtual_axis_ray_z": axis.center_ray[2],
            "virtual_axis_c0_x": axis.candidate0[0],
            "virtual_axis_c0_y": axis.candidate0[1],
            "virtual_axis_c0_z": axis.candidate0[2],
            "virtual_axis_c1_x": axis.candidate1[0],
            "virtual_axis_c1_y": axis.candidate1[1],
            "virtual_axis_c1_z": axis.candidate1[2],
            "real_axis_valid": int(real_axis.valid),
            "real_axis_reason": real_axis.reason,
            "real_axis_selected": real_axis.selected_index,
            "real_axis_alpha_rad": real_axis.alpha,
            "real_axis_gamma_rad": real_axis.gamma,
            "real_axis_nx": real_axis.normal[0],
            "real_axis_ny": real_axis.normal[1],
            "real_axis_nz": real_axis.normal[2],
            "real_axis_c0_x": real_axis.candidate0[0],
            "real_axis_c0_y": real_axis.candidate0[1],
            "real_axis_c0_z": real_axis.candidate0[2],
            "real_axis_c1_x": real_axis.candidate1[0],
            "real_axis_c1_y": real_axis.candidate1[1],
            "real_axis_c1_z": real_axis.candidate1[2],
            "real_axis_c0_gamma_rad": real_axis.candidate0_gamma,
            "real_axis_c1_gamma_rad": real_axis.candidate1_gamma,
        }


def invalid_ellipse() -> EllipseTuple:
    return (-1.0, -1.0, -1.0, -1.0, -1.00)


def ellipse_to_tuple(value: Iterable[float]) -> EllipseTuple:
    cx, cy, axis_a, axis_b, theta = [float(item) for item in value]
    return cx, cy, axis_a, axis_b, theta


def offset_ellipse(ellipse: EllipseTuple, roi: Roi) -> EllipseTuple:
    cx, cy, axis_a, axis_b, theta = ellipse
    if axis_a < 0 or axis_b < 0:
        return ellipse
    return cx + roi.x, cy + roi.y, axis_a, axis_b, theta


def _invalid_real_axis(reason: str):
    nan3 = (math.nan, math.nan, math.nan)
    return type(
        "InvalidRealAxis",
        (),
        {
            "valid": False,
            "reason": reason,
            "normal": nan3,
            "gamma": math.nan,
            "alpha": math.nan,
            "candidate0": nan3,
            "candidate1": nan3,
            "candidate0_gamma": math.nan,
            "candidate1_gamma": math.nan,
            "selected_index": -1,
        },
    )()
