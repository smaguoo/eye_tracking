from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from .types import Vector3, VirtualAxisEstimate


@dataclass(frozen=True)
class CornealRefractionModel:
    """Two-sphere corneal optical model used to correct virtual pupil axes.

    The defaults are typical schematic-eye values in millimeters. The model
    follows the paper's ideal two-sphere cornea approximation: the outer and
    inner corneal surfaces share the same center, and the virtual pupil axis is
    related to the real pupil axis through the equivalent optical system.
    """

    outer_radius_mm: float = 7.7
    inner_radius_mm: float = 6.8
    refractive_index_air: float = 1.0
    refractive_index_cornea: float = 1.376
    refractive_index_aqueous: float = 1.336
    pupil_to_inner_cornea_distance_mm: float = 3.05
    lookup_samples: int = 4096

    @property
    def cornea_thickness_mm(self) -> float:
        return abs(self.outer_radius_mm - self.inner_radius_mm)

    @property
    def cornea_center_to_pupil_center_mm(self) -> float:
        return self.inner_radius_mm - self.pupil_to_inner_cornea_distance_mm


@dataclass(frozen=True)
class CornealOptics:
    f_inner: float
    f_inner_prime: float
    f_outer: float
    f_outer_prime: float
    equivalent_f: float
    equivalent_f_prime: float
    h_from_cornea_center: float
    h_prime_from_cornea_center: float
    hc: float
    ch_prime: float
    hh_prime: float


@dataclass(frozen=True)
class RealPupilAxisEstimate:
    valid: bool
    reason: str
    normal: Vector3
    gamma: float
    alpha: float
    virtual_normal: Vector3
    candidate0: Vector3
    candidate1: Vector3
    candidate0_gamma: float
    candidate1_gamma: float
    selected_index: int

    @classmethod
    def invalid(cls, reason: str) -> "RealPupilAxisEstimate":
        nan3 = (math.nan, math.nan, math.nan)
        return cls(False, reason, nan3, math.nan, math.nan, nan3, nan3, nan3, math.nan, math.nan, -1)


def compute_corneal_optics(model: CornealRefractionModel = CornealRefractionModel()) -> CornealOptics:
    """Compute the equivalent optical-system parameters from the paper.

    The paper's signed distances are used directly in the refraction matrix:
    HC is positive in the H-to-C direction, CH' satisfies f + f' + CH' = 0,
    and HH' = HC + CH' = HC - f - f'.
    """

    r1 = float(model.inner_radius_mm)
    r2 = float(model.outer_radius_mm)
    n_air = float(model.refractive_index_air)
    n_cornea = float(model.refractive_index_cornea)
    n_aqueous = float(model.refractive_index_aqueous)
    d_c = float(model.cornea_thickness_mm)

    if min(r1, r2, n_air, n_cornea, n_aqueous, d_c) <= 0:
        raise ValueError("Corneal model parameters must be positive")
    if model.pupil_to_inner_cornea_distance_mm <= 0:
        raise ValueError("Pupil-to-inner-cornea distance must be positive")
    if model.cornea_center_to_pupil_center_mm <= 0:
        raise ValueError("Pupil center must be behind the inner cornea")

    f1 = -n_cornea / (n_aqueous - n_cornea) * r1
    f1_prime = n_aqueous / (n_aqueous - n_cornea) * r1
    f2 = -n_air / (n_cornea - n_air) * r2
    f2_prime = n_cornea / (n_cornea - n_air) * r2

    delta = d_c - f1_prime + f2
    if abs(delta) < 1e-12:
        raise ValueError("Degenerate equivalent corneal optical system")

    equivalent_f = f1 * f2 / delta
    equivalent_f_prime = -f1_prime * f2_prime / delta

    vh = -equivalent_f_prime * d_c / f2
    v_prime_h_prime = -equivalent_f_prime * d_c / f1_prime
    h_from_cornea_center = r1 + vh
    hc = h_from_cornea_center
    ch_prime = -(equivalent_f + equivalent_f_prime)
    h_prime_from_cornea_center = -ch_prime
    hh_prime = hc + ch_prime

    return CornealOptics(
        f_inner=f1,
        f_inner_prime=f1_prime,
        f_outer=f2,
        f_outer_prime=f2_prime,
        equivalent_f=equivalent_f,
        equivalent_f_prime=equivalent_f_prime,
        h_from_cornea_center=h_from_cornea_center,
        h_prime_from_cornea_center=h_prime_from_cornea_center,
        hc=hc,
        ch_prime=ch_prime,
        hh_prime=hh_prime,
    )


def gamma_from_alpha(alpha: float, model: CornealRefractionModel = CornealRefractionModel()) -> float:
    """Approximate the paper's gamma ~= f(alpha) relation.

    The paper first derives gamma as a function of the virtual pupil angle phi'
    and then uses alpha as an approximation to phi'. This function implements
    that lookup numerically from the two-sphere corneal model.
    """

    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")

    sign = -1.0 if alpha < 0 else 1.0
    target = abs(float(alpha))
    max_angle = math.radians(89.0)
    target = max(0.0, min(max_angle, target))

    optics = compute_corneal_optics(model)
    samples = max(128, int(model.lookup_samples))
    phis = np.linspace(0.0, max_angle, samples)
    virtual_angles = np.asarray([virtual_pupil_angle(phi, model, optics) for phi in phis], dtype=float)
    gammas = phis - virtual_angles

    gamma = float(np.interp(target, virtual_angles, gammas))
    return sign * gamma


def virtual_pupil_angle(
    phi: float,
    model: CornealRefractionModel = CornealRefractionModel(),
    optics: Optional[CornealOptics] = None,
) -> float:
    """Compute phi' from the paper's virtual pupil normal expression."""

    optics = compute_corneal_optics(model) if optics is None else optics
    cp = float(model.cornea_center_to_pupil_center_mm)
    f_prime = float(optics.equivalent_f_prime)
    if abs(f_prime) < 1e-12:
        raise ValueError("Equivalent image-side focal length is zero")

    y = math.sin(phi)
    z = math.cos(phi) + (optics.hh_prime * math.cos(phi) + cp) / f_prime
    return math.atan2(y, z)


def correct_virtual_axis(
    virtual_normal: Iterable[float],
    center_ray: Iterable[float],
    model: CornealRefractionModel = CornealRefractionModel(),
) -> Tuple[np.ndarray, float, float]:
    """Correct one virtual pupil normal to one real pupil normal."""

    ray = normalize(center_ray)
    virtual = normalize(virtual_normal)
    if float(np.dot(virtual, ray)) > 0.0:
        virtual = -virtual

    alpha = math.acos(clamp(-float(np.dot(virtual, ray)), -1.0, 1.0))
    gamma = gamma_from_alpha(alpha, model)
    real = rotate_virtual_to_real(virtual, ray, gamma)
    if float(np.dot(real, ray)) > 0.0:
        real = -real
    return normalize(real), alpha, gamma


def estimate_real_axis(
    virtual_axis: VirtualAxisEstimate,
    model: CornealRefractionModel = CornealRefractionModel(),
) -> RealPupilAxisEstimate:
    """Correct a selected virtual-axis estimate and both of its candidates."""

    if not virtual_axis.valid:
        return RealPupilAxisEstimate.invalid(virtual_axis.reason)

    try:
        candidate0, alpha0, gamma0 = correct_virtual_axis(virtual_axis.candidate0, virtual_axis.center_ray, model)
        candidate1, alpha1, gamma1 = correct_virtual_axis(virtual_axis.candidate1, virtual_axis.center_ray, model)
        selected = 1 if virtual_axis.selected_index == 1 else 0
        normal = candidate1 if selected == 1 else candidate0
        alpha = alpha1 if selected == 1 else alpha0
        gamma = gamma1 if selected == 1 else gamma0
    except Exception:
        return RealPupilAxisEstimate.invalid("real_axis_failed")

    return RealPupilAxisEstimate(
        valid=True,
        reason="ok",
        normal=to_vector3(normal),
        gamma=gamma,
        alpha=alpha,
        virtual_normal=virtual_axis.normal,
        candidate0=to_vector3(candidate0),
        candidate1=to_vector3(candidate1),
        candidate0_gamma=gamma0,
        candidate1_gamma=gamma1,
        selected_index=selected,
    )


def rotate_virtual_to_real(virtual_normal: np.ndarray, center_ray: np.ndarray, gamma: float) -> np.ndarray:
    """Apply n = R_12 Ry(-gamma) R_21 n' from the paper."""

    z_axis = normalize(center_ray)
    y_axis = np.cross(z_axis, virtual_normal)
    if float(np.linalg.norm(y_axis)) < 1e-12:
        return normalize(virtual_normal)
    y_axis = normalize(y_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    basis = np.column_stack([x_axis, y_axis, z_axis])

    local_virtual = basis.T @ virtual_normal
    local_real = rotation_y(-gamma) @ local_virtual
    return normalize(basis @ local_real)


def rotation_y(angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=float,
    )


def normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(tuple(vector), dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("Cannot normalize zero or non-finite vector")
    return arr / norm


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_vector3(vector: Iterable[float]) -> Vector3:
    x, y, z = [float(item) for item in vector]
    return x, y, z
