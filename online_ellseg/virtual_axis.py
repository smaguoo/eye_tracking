from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .types import EllipseTuple, Vector3, VirtualAxisEstimate


def parse_camera_matrix(value: str) -> np.ndarray:
    """Parse either fx,fy,cx,cy or a full 3x3 camera matrix."""

    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 4:
        fx, fy, cx, cy = parts
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    if len(parts) == 9:
        return np.array(parts, dtype=float).reshape(3, 3)
    raise ValueError("--intrinsics must be fx,fy,cx,cy or 9 comma-separated matrix values")


def load_calibration(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=float)
    dist_coeffs = payload.get("dist_coeffs", payload.get("distortion_coefficients"))
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    return camera_matrix, dist_coeffs


def approximate_camera_matrix(width: int, height: int, focal_px: Optional[float] = None) -> np.ndarray:
    """Build a usable demo intrinsic matrix when real calibration is unavailable."""

    focal = float(focal_px) if focal_px and focal_px > 0 else float(max(width, height))
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def estimate_virtual_axis(
    ellipse: EllipseTuple,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    previous_normal: Optional[Vector3] = None,
    first_candidate: int = 0,
) -> VirtualAxisEstimate:
    """Estimate the virtual pupil axis candidates from an image ellipse.

    The result is the classic two-solution inverse projection of an imaged
    circle. The selected normal uses temporal continuity when a previous
    normal is available; otherwise first_candidate is used.
    """

    try:
        k = np.asarray(camera_matrix, dtype=float)
        if k.shape != (3, 3) or abs(float(np.linalg.det(k))) < 1e-12:
            return VirtualAxisEstimate.invalid("bad_intrinsics")
        if dist_coeffs is not None:
            ellipse = undistort_ellipse(ellipse, k, np.asarray(dist_coeffs, dtype=float))
        conic = ellipse_to_conic(ellipse)

        q = k.T @ conic @ k
        candidates = circular_section_normals(q)
        ray = normalize(np.linalg.inv(k) @ np.array([ellipse[0], ellipse[1], 1.0], dtype=float))
    except Exception:
        return VirtualAxisEstimate.invalid("geometry_failed")

    oriented = []
    for candidate in candidates:
        normal = normalize(candidate)
        if np.dot(normal, ray) > 0:
            normal = -normal
        oriented.append(normal)

    if len(oriented) != 2:
        return VirtualAxisEstimate.invalid("no_candidates")

    selected_index = select_candidate(oriented, previous_normal, first_candidate)
    normal = oriented[selected_index]
    alpha = math.acos(clamp(-float(np.dot(normal, ray)), -1.0, 1.0))

    return VirtualAxisEstimate(
        valid=True,
        reason="ok",
        center_ray=to_vector3(ray),
        normal=to_vector3(normal),
        alpha=alpha,
        candidate0=to_vector3(oriented[0]),
        candidate1=to_vector3(oriented[1]),
        selected_index=selected_index,
    )


def undistort_ellipse(ellipse: EllipseTuple, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> EllipseTuple:
    import cv2

    cx, cy, axis_a, axis_b, theta = ellipse
    if axis_a <= 0 or axis_b <= 0:
        raise ValueError("Ellipse axes must be positive")

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 80, endpoint=False):
        x_local = axis_a * math.cos(angle)
        y_local = axis_b * math.sin(angle)
        x = cx + cos_t * x_local - sin_t * y_local
        y = cy + sin_t * x_local + cos_t * y_local
        points.append([x, y])

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    (u_cx, u_cy), (diam_a, diam_b), angle_deg = cv2.fitEllipse(undistorted.astype(np.float32))
    return float(u_cx), float(u_cy), float(diam_a) / 2.0, float(diam_b) / 2.0, math.radians(float(angle_deg))


def ellipse_to_conic(ellipse: EllipseTuple) -> np.ndarray:
    cx, cy, axis_a, axis_b, theta = ellipse
    if not all(math.isfinite(value) for value in ellipse):
        raise ValueError("Ellipse contains non-finite values")
    if axis_a <= 0 or axis_b <= 0:
        raise ValueError("Ellipse axes must be positive")

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    inv_axes = np.diag([1.0 / (axis_a * axis_a), 1.0 / (axis_b * axis_b)])
    quad = rotation @ inv_axes @ rotation.T
    center = np.array([cx, cy], dtype=float)

    conic = np.empty((3, 3), dtype=float)
    conic[:2, :2] = quad
    conic[:2, 2] = -quad @ center
    conic[2, :2] = conic[:2, 2]
    conic[2, 2] = float(center.T @ quad @ center - 1.0)
    return conic


def circular_section_normals(cone: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cone = 0.5 * (np.asarray(cone, dtype=float) + np.asarray(cone, dtype=float).T)
    values, vectors = np.linalg.eigh(cone)

    if np.count_nonzero(values > 0) == 1 and np.count_nonzero(values < 0) == 2:
        values = -values
        cone = -cone
        values, vectors = np.linalg.eigh(cone)

    positive = [idx for idx, value in enumerate(values) if value > 0]
    negative = [idx for idx, value in enumerate(values) if value < 0]
    if len(positive) != 2 or len(negative) != 1:
        raise ValueError(f"Cone must have two positive eigenvalues and one negative eigenvalue, got {values}")

    positive.sort(key=lambda idx: values[idx], reverse=True)
    idx1, idx2 = positive
    idx3 = negative[0]
    lambda1, lambda2, lambda3 = float(values[idx1]), float(values[idx2]), float(values[idx3])
    basis = np.column_stack([vectors[:, idx1], vectors[:, idx2], vectors[:, idx3]])

    denom = lambda1 - lambda3
    if denom <= 0:
        raise ValueError("Invalid eigenvalue ordering")

    x_component = math.sqrt(max(0.0, (lambda1 - lambda2) / denom))
    z_component = math.sqrt(max(0.0, (lambda2 - lambda3) / denom))

    normal0 = basis @ np.array([x_component, 0.0, z_component], dtype=float)
    normal1 = basis @ np.array([-x_component, 0.0, z_component], dtype=float)
    return normalize(normal0), normalize(normal1)


def select_candidate(
    candidates: Sequence[np.ndarray],
    previous_normal: Optional[Vector3],
    first_candidate: int,
) -> int:
    if previous_normal is None:
        return 1 if first_candidate == 1 else 0

    previous = normalize(np.array(previous_normal, dtype=float))
    scores = [float(np.dot(candidate, previous)) for candidate in candidates]
    return int(np.argmax(scores))


def normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Cannot normalize zero or non-finite vector")
    return arr / norm


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_vector3(vector: Iterable[float]) -> Vector3:
    x, y, z = [float(item) for item in vector]
    return x, y, z
