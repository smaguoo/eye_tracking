from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .types import Vector3


@dataclass(frozen=True)
class ScreenGeometry:
    """Screen plane in the eye-camera coordinate system.

    Camera coordinates follow OpenCV convention: +x right, +y down, +z from
    the camera toward the user. The screen is modeled as a plane parallel to
    the camera image plane at z_mm.
    """

    width_mm: float
    height_mm: float
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    z_mm: float = 0.0

    def validate(self) -> None:
        for value in (self.width_mm, self.height_mm, self.center_x_mm, self.center_y_mm, self.z_mm):
            if not math.isfinite(value):
                raise ValueError("Screen geometry values must be finite.")
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise ValueError("Screen width/height must be positive in --geometry mode.")


@dataclass(frozen=True)
class EyeOriginEstimate:
    left_origin_mm: np.ndarray
    right_origin_mm: np.ndarray
    depth_mm: float
    reason: str


@dataclass(frozen=True)
class GeometryPrediction:
    valid: bool
    x_px: float
    y_px: float
    point_mm: Tuple[float, float, float]
    ray_t_mm: float
    reason: str
    direction_sign: int = 1

    @classmethod
    def invalid(cls, reason: str) -> "GeometryPrediction":
        nan3 = (math.nan, math.nan, math.nan)
        return cls(False, math.nan, math.nan, nan3, math.nan, reason, 0)


def estimate_binocular_eye_origins(
    left_center_ray: Iterable[float],
    right_center_ray: Iterable[float],
    interpupillary_distance_mm: float,
    fallback_depth_mm: float,
) -> EyeOriginEstimate:
    """Estimate eye centers from two camera rays and an assumed IPD.

    The two eye centers are constrained to share one camera-space depth. This
    gives a practical scale for the otherwise depth-free center rays.
    """

    try:
        left_ray = normalize(left_center_ray)
        right_ray = normalize(right_center_ray)
        if abs(left_ray[2]) < 1e-6 or abs(right_ray[2]) < 1e-6:
            raise ValueError("center ray is parallel to the camera plane")
        left_at_z1 = left_ray / left_ray[2]
        right_at_z1 = right_ray / right_ray[2]
        separation_at_z1 = float(np.linalg.norm(right_at_z1 - left_at_z1))
        if separation_at_z1 < 1e-6:
            raise ValueError("eye center rays are too close")
        depth_mm = float(interpupillary_distance_mm) / separation_at_z1
        if not math.isfinite(depth_mm) or not 100.0 <= depth_mm <= 2000.0:
            raise ValueError("estimated eye depth is outside a plausible range")
        reason = "ipd"
    except Exception:
        depth_mm = float(fallback_depth_mm)
        reason = "fallback_depth"

    return EyeOriginEstimate(
        left_origin_mm=origin_at_depth(left_center_ray, depth_mm),
        right_origin_mm=origin_at_depth(right_center_ray, depth_mm),
        depth_mm=depth_mm,
        reason=reason,
    )


def origin_at_depth(center_ray: Iterable[float], depth_mm: float) -> np.ndarray:
    ray = normalize(center_ray)
    if abs(ray[2]) < 1e-6:
        raise ValueError("center ray is parallel to the camera plane")
    return ray * (float(depth_mm) / ray[2])


def binocular_origins_from_screen_center_offset(
    geometry: ScreenGeometry,
    interpupillary_distance_mm: float,
    offset_x_mm: float,
    offset_y_mm: float,
    offset_z_mm: float,
) -> EyeOriginEstimate:
    """Place the binocular eye midpoint at a fixed offset from screen center."""

    geometry.validate()
    for value in (interpupillary_distance_mm, offset_x_mm, offset_y_mm, offset_z_mm):
        if not math.isfinite(float(value)):
            raise ValueError("Fixed eye offset values must be finite")
    if interpupillary_distance_mm <= 0.0:
        raise ValueError("Interpupillary distance must be positive")

    midpoint = np.array(
        [
            geometry.center_x_mm + float(offset_x_mm),
            geometry.center_y_mm + float(offset_y_mm),
            geometry.z_mm + float(offset_z_mm),
        ],
        dtype=float,
    )
    half_ipd = float(interpupillary_distance_mm) / 2.0
    return EyeOriginEstimate(
        left_origin_mm=midpoint + np.array([half_ipd, 0.0, 0.0], dtype=float),
        right_origin_mm=midpoint + np.array([-half_ipd, 0.0, 0.0], dtype=float),
        depth_mm=float(midpoint[2]),
        reason="screen_center_offset",
    )


def binocular_origins_from_midpoint(
    midpoint_x_mm: float,
    midpoint_y_mm: float,
    midpoint_z_mm: float,
    interpupillary_distance_mm: float,
) -> EyeOriginEstimate:
    """Place left/right eye origins from an absolute camera-space midpoint."""

    for value in (midpoint_x_mm, midpoint_y_mm, midpoint_z_mm, interpupillary_distance_mm):
        if not math.isfinite(float(value)):
            raise ValueError("Fixed eye midpoint values must be finite")
    if interpupillary_distance_mm <= 0.0:
        raise ValueError("Interpupillary distance must be positive")

    midpoint = np.array(
        [float(midpoint_x_mm), float(midpoint_y_mm), float(midpoint_z_mm)],
        dtype=float,
    )
    half_ipd = float(interpupillary_distance_mm) / 2.0
    return EyeOriginEstimate(
        left_origin_mm=midpoint + np.array([half_ipd, 0.0, 0.0], dtype=float),
        right_origin_mm=midpoint + np.array([-half_ipd, 0.0, 0.0], dtype=float),
        depth_mm=float(midpoint[2]),
        reason="fixed_midpoint",
    )


def origin_from_screen_center_offset(
    geometry: ScreenGeometry,
    offset_x_mm: float,
    offset_y_mm: float,
    offset_z_mm: float,
) -> np.ndarray:
    geometry.validate()
    return np.array(
        [
            geometry.center_x_mm + float(offset_x_mm),
            geometry.center_y_mm + float(offset_y_mm),
            geometry.z_mm + float(offset_z_mm),
        ],
        dtype=float,
    )


def origin_at_screen_center_distance(
    center_ray: Iterable[float],
    geometry: ScreenGeometry,
    distance_mm: float,
) -> np.ndarray:
    """Place an eye origin on its camera ray at a known screen-center distance."""

    geometry.validate()
    distance_mm = float(distance_mm)
    if not math.isfinite(distance_mm) or distance_mm <= 0.0:
        raise ValueError("Eye-to-screen-center distance must be positive")

    ray = normalize(center_ray)
    screen_center = np.array(
        [geometry.center_x_mm, geometry.center_y_mm, geometry.z_mm],
        dtype=float,
    )
    dot = float(np.dot(ray, screen_center))
    radius_term = float(np.dot(screen_center, screen_center) - distance_mm * distance_mm)
    discriminant = dot * dot - radius_term
    if discriminant < 0.0:
        raise ValueError("Center ray does not intersect the eye-distance sphere")

    root = math.sqrt(max(0.0, discriminant))
    candidates = [dot - root, dot + root]
    positive = [value for value in candidates if math.isfinite(value) and value > 0.0]
    if not positive:
        raise ValueError("Eye-distance sphere is behind the camera")
    return ray * min(positive)


def predict_screen_point(
    axis,
    eye_origin_mm: Iterable[float],
    geometry: ScreenGeometry,
    display_width: int,
    display_height: int,
    kappa_yaw_deg: float = 0.0,
    kappa_pitch_deg: float = 0.0,
    invert_x: bool = False,
    invert_y: bool = False,
) -> GeometryPrediction:
    """Intersect one pupil axis with the configured screen plane."""

    if axis is None or not getattr(axis, "valid", False):
        return GeometryPrediction.invalid("no_axis")
    geometry.validate()

    try:
        origin = np.asarray(tuple(eye_origin_mm), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("bad eye origin")
        normal = normalize(getattr(axis, "normal"))
    except Exception:
        return GeometryPrediction.invalid("bad_axis")

    candidates = ((normal, 1), (-normal, -1))
    for direction, sign in candidates:
        direction = apply_direction_flips(direction, invert_x, invert_y)
        direction = apply_kappa(direction, kappa_yaw_deg, kappa_pitch_deg)
        hit = intersect_z_plane(origin, direction, geometry.z_mm)
        if hit is None:
            continue
        point_mm, t_mm = hit
        x_px, y_px = screen_mm_to_pixels(point_mm, geometry, display_width, display_height)
        return GeometryPrediction(
            valid=True,
            x_px=x_px,
            y_px=y_px,
            point_mm=(float(point_mm[0]), float(point_mm[1]), float(point_mm[2])),
            ray_t_mm=t_mm,
            reason="ok",
            direction_sign=sign,
        )

    return GeometryPrediction.invalid("ray_misses_screen_plane")


def apply_direction_flips(direction: Iterable[float], invert_x: bool, invert_y: bool) -> np.ndarray:
    direction = normalize(direction)
    if invert_x:
        direction[0] *= -1.0
    if invert_y:
        direction[1] *= -1.0
    return normalize(direction)


def apply_kappa(direction: Iterable[float], yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Apply visual-axis correction in screen-facing angular coordinates.

    Positive yaw moves the intersection to the right, and positive pitch moves
    it downward. The direction is expected to point from the eye toward the
    screen, so forward is camera-space -z.
    """

    direction = normalize(direction)
    yaw_delta = math.radians(float(yaw_deg))
    pitch_delta = math.radians(float(pitch_deg))
    if not math.isfinite(yaw_delta) or not math.isfinite(pitch_delta):
        raise ValueError("kappa angles must be finite")
    if abs(yaw_delta) < 1e-12 and abs(pitch_delta) < 1e-12:
        return direction

    yaw = math.atan2(float(direction[0]), -float(direction[2])) + yaw_delta
    pitch = math.atan2(float(direction[1]), math.hypot(float(direction[0]), float(direction[2]))) + pitch_delta
    cos_pitch = math.cos(pitch)
    corrected = np.array(
        [
            math.sin(yaw) * cos_pitch,
            math.sin(pitch),
            -math.cos(yaw) * cos_pitch,
        ],
        dtype=float,
    )
    return normalize(corrected)


def intersect_z_plane(origin: np.ndarray, direction: np.ndarray, z_mm: float):
    dz = float(direction[2])
    if abs(dz) < 1e-9:
        return None
    t_mm = (float(z_mm) - float(origin[2])) / dz
    if not math.isfinite(t_mm) or t_mm <= 0.0:
        return None
    point_mm = origin + t_mm * direction
    if not np.all(np.isfinite(point_mm)):
        return None
    return point_mm, float(t_mm)


def screen_mm_to_pixels(
    point_mm: np.ndarray,
    geometry: ScreenGeometry,
    display_width: int,
    display_height: int,
) -> Tuple[float, float]:
    left_mm = geometry.center_x_mm - geometry.width_mm / 2.0
    top_mm = geometry.center_y_mm - geometry.height_mm / 2.0
    x_px = (float(point_mm[0]) - left_mm) * float(display_width) / geometry.width_mm
    y_px = (float(point_mm[1]) - top_mm) * float(display_height) / geometry.height_mm
    return x_px, y_px


def normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(tuple(vector), dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Cannot normalize zero or non-finite vector")
    return arr / norm


def to_vector3(vector: Iterable[float]) -> Vector3:
    x, y, z = [float(item) for item in vector]
    return x, y, z
