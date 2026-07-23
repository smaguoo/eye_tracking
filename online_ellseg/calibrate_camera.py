from __future__ import annotations

import argparse
import glob
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class CalibrationSample:
    image_path: Optional[Path]
    corners: np.ndarray


@dataclass
class CalibrationResult:
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    per_view_errors: list[float]
    mean_error: float


def parse_size(value: str, name: str) -> tuple[int, int]:
    cleaned = value.lower().replace("*", "x").replace(",", "x")
    parts = [part.strip() for part in cleaned.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must look like 7x10")
    try:
        cols, rows = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from exc
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError(f"{name} values must be >= 2")
    return cols, rows


def parse_inner_corners(value: str) -> tuple[int, int]:
    return parse_size(value, "--inner-corners")


def parse_board_squares(value: str) -> tuple[int, int]:
    squares_cols, squares_rows = parse_size(value, "--board-squares")
    return squares_cols - 1, squares_rows - 1


def backend_flag(backend: str) -> int:
    backend = backend.lower()
    if backend in ("", "auto", "any"):
        return cv2.CAP_ANY
    if backend == "dshow":
        return cv2.CAP_DSHOW
    if backend == "msmf":
        return cv2.CAP_MSMF
    if backend == "vfw":
        return cv2.CAP_VFW
    raise ValueError(f"Unsupported camera backend: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a webcam with an OpenCV checkerboard. Defaults are 9x6 "
            "inner corners and 25 mm square size."
        )
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 for the built-in front camera.")
    parser.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "vfw", "auto"])
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument(
        "--inner-corners",
        type=parse_inner_corners,
        default=(9, 6),
        metavar="COLSxROWS",
        help=(
            "Checkerboard inner corner count as cols x rows. Default: 9x6. "
            "If your printed board has 10x7 squares, use --board-squares 10x7 instead."
        ),
    )
    parser.add_argument(
        "--board-squares",
        type=parse_board_squares,
        default=None,
        metavar="COLSxROWS",
        help="Optional square count; inner corners are computed as (cols - 1) x (rows - 1).",
    )
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument(
        "--images",
        default="",
        help="Optional image directory or glob. If omitted, samples are captured from the camera.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/camera_calibration.json"))
    parser.add_argument("--output-npz", type=Path, default=Path("outputs/camera_calibration.npz"))
    parser.add_argument("--save-frames", type=Path, default=Path("outputs/calibration_frames"))
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically accept detected boards while the user moves/tilts the checkerboard.",
    )
    parser.add_argument("--auto-delay", type=float, default=0.8)
    parser.add_argument("--min-corner-motion", type=float, default=12.0)
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--no-window", action="store_true", help="Do not show the live preview window.")
    parser.add_argument("--show-detections", action="store_true", help="Show detection preview for --images mode.")
    return parser


def checkerboard_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_size_mm)
    return points


def detect_checkerboard(gray: np.ndarray, pattern_size: tuple[int, int]) -> Optional[np.ndarray]:
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
        if found:
            return corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def collect_image_paths(images: str) -> list[Path]:
    path = Path(images)
    if path.is_dir():
        return sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in IMAGE_EXTENSIONS)
    if path.is_file():
        return [path]
    return sorted(Path(match) for match in glob.glob(images) if Path(match).suffix.lower() in IMAGE_EXTENSIONS)


def load_samples_from_images(
    images: str,
    pattern_size: tuple[int, int],
    show_detections: bool,
) -> tuple[list[CalibrationSample], tuple[int, int]]:
    image_paths = collect_image_paths(images)
    if not image_paths:
        raise RuntimeError(f"No calibration images found: {images}")

    samples: list[CalibrationSample] = []
    image_size: Optional[tuple[int, int]] = None
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        height, width = frame.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            print(f"Skipping {image_path}: image size differs from the first image.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect_checkerboard(gray, pattern_size)
        if corners is None:
            print(f"No checkerboard: {image_path}")
            continue

        samples.append(CalibrationSample(image_path=image_path, corners=corners))
        print(f"Accepted {len(samples):02d}: {image_path}")

        if show_detections:
            preview = frame.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners, True)
            cv2.imshow("checkerboard detections", preview)
            key = cv2.waitKey(250) & 0xFF
            if key in (27, ord("q")):
                break

    if show_detections:
        cv2.destroyAllWindows()
    if image_size is None:
        raise RuntimeError("No readable calibration images were found.")
    return samples, image_size


def corner_motion(corners: np.ndarray, last_corners: Optional[np.ndarray]) -> float:
    if last_corners is None or corners.shape != last_corners.shape:
        return math.inf
    delta = corners.reshape(-1, 2) - last_corners.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def draw_status(
    frame: np.ndarray,
    found: bool,
    sample_count: int,
    min_samples: int,
    max_samples: int,
    auto: bool,
) -> np.ndarray:
    overlay = frame.copy()
    status = "FOUND" if found else "SEARCHING"
    color = (30, 220, 30) if found else (30, 30, 230)
    lines = [
        f"checkerboard: {status}",
        f"samples: {sample_count}/{max_samples} (minimum {min_samples})",
        "SPACE/S save  C calibrate  Q/ESC quit",
    ]
    if auto:
        lines.append("auto mode: move and tilt the board between captures")

    y = 28
    for idx, line in enumerate(lines):
        line_color = color if idx == 0 else (245, 245, 245)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, line_color, 1, cv2.LINE_AA)
        y += 28
    return overlay


def accept_sample(
    frame: np.ndarray,
    corners: np.ndarray,
    samples: list[CalibrationSample],
    save_dir: Optional[Path],
) -> None:
    image_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"calib_{len(samples) + 1:03d}.png"
        cv2.imwrite(str(image_path), frame)
    samples.append(CalibrationSample(image_path=image_path, corners=corners.copy()))
    print(f"Accepted sample {len(samples):02d}" + (f": {image_path}" if image_path else ""))


def capture_samples_from_camera(args, pattern_size: tuple[int, int]) -> tuple[list[CalibrationSample], tuple[int, int]]:
    capture = cv2.VideoCapture(args.camera, backend_flag(args.backend))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.fps)

    samples: list[CalibrationSample] = []
    image_size: Optional[tuple[int, int]] = None
    last_auto_time = 0.0
    last_accepted_corners: Optional[np.ndarray] = None
    save_dir = None if args.no_save_frames else args.save_frames

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            if args.flip_horizontal:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            image_size = (width, height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_checkerboard(gray, pattern_size)
            found = corners is not None

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern_size, corners, True)
                now = time.perf_counter()
                motion = corner_motion(corners, last_accepted_corners)
                if (
                    args.auto
                    and len(samples) < args.max_samples
                    and now - last_auto_time >= args.auto_delay
                    and motion >= args.min_corner_motion
                ):
                    accept_sample(frame, corners, samples, save_dir)
                    last_auto_time = now
                    last_accepted_corners = corners.copy()

            if not args.no_window:
                cv2.imshow(
                    "camera calibration",
                    draw_status(preview, found, len(samples), args.min_samples, args.max_samples, args.auto),
                )
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key in (ord(" "), ord("s")) and found and len(samples) < args.max_samples:
                accept_sample(frame, corners, samples, save_dir)
                last_accepted_corners = corners.copy()
            elif key == ord("c") and len(samples) >= args.min_samples:
                break
            elif key in (27, ord("q")):
                break

            if len(samples) >= args.max_samples:
                break
    finally:
        capture.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No camera frames were captured.")
    return samples, image_size

###
def calibrate(
    samples: list[CalibrationSample],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> CalibrationResult:
    object_template = checkerboard_object_points(pattern_size, square_size_mm)
    object_points = [object_template.copy() for _ in samples]
    image_points = [sample.corners.astype(np.float32) for sample in samples]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    per_view_errors: list[float] = []
    all_squared_errors: list[np.ndarray] = []
    for obj_points, img_points, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        diff = img_points.reshape(-1, 2) - projected.reshape(-1, 2)
        squared = np.sum(diff * diff, axis=1)
        per_view_errors.append(float(np.sqrt(np.mean(squared))))
        all_squared_errors.append(squared)

    mean_error = float(np.sqrt(np.mean(np.concatenate(all_squared_errors))))
    return CalibrationResult(
        rms=float(rms),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=tuple(rvecs),
        tvecs=tuple(tvecs),
        per_view_errors=per_view_errors,
        mean_error=mean_error,
    )


def distortion_names(count: int) -> list[str]:
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y"]
    if count <= len(names):
        return names[:count]
    return names + [f"d{i}" for i in range(len(names), count)]


def save_result(
    result: CalibrationResult,
    samples: list[CalibrationSample],
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    square_size_mm: float,
    output_json: Path,
    output_npz: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    dist = result.dist_coeffs.ravel()
    fx = float(result.camera_matrix[0, 0])
    fy = float(result.camera_matrix[1, 1])
    cx = float(result.camera_matrix[0, 2])
    cy = float(result.camera_matrix[1, 2])
    sample_paths = [str(sample.image_path) if sample.image_path is not None else "" for sample in samples]

    payload = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "checkerboard": {
            "inner_corners": {"cols": pattern_size[0], "rows": pattern_size[1]},
            "square_size_mm": square_size_mm,
        },
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_model": "opencv_standard",
        "distortion_order": distortion_names(len(dist)),
        "distortion_coefficients": dist.tolist(),
        "rms_reprojection_error_px": result.rms,
        "mean_reprojection_error_px": result.mean_error,
        "per_view_reprojection_error_px": result.per_view_errors,
        "sample_count": len(samples),
        "sample_images": sample_paths,
    }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez(
        output_npz,
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        rvecs=np.asarray(result.rvecs),
        tvecs=np.asarray(result.tvecs),
        image_size=np.asarray(image_size),
        inner_corners=np.asarray(pattern_size),
        square_size_mm=np.asarray([square_size_mm]),
        rms=np.asarray([result.rms]),
        mean_error=np.asarray([result.mean_error]),
        per_view_errors=np.asarray(result.per_view_errors),
    )


def print_result(result: CalibrationResult, output_json: Path, output_npz: Path) -> None:
    dist = result.dist_coeffs.ravel()
    fx = result.camera_matrix[0, 0]
    fy = result.camera_matrix[1, 1]
    cx = result.camera_matrix[0, 2]
    cy = result.camera_matrix[1, 2]

    print("\nCalibration finished")
    print(f"fx, fy, cx, cy = {fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}")
    print("camera_matrix =")
    print(np.array2string(result.camera_matrix, precision=6, suppress_small=False))
    print("distortion coefficients:")
    for name, value in zip(distortion_names(len(dist)), dist):
        print(f"  {name}: {float(value): .10f}")
    print(f"OpenCV dist array: {np.array2string(dist, precision=10, separator=', ')}")
    print(f"RMS reprojection error: {result.rms:.6f} px")
    print(f"Mean reprojection error: {result.mean_error:.6f} px")
    print(f"Saved JSON: {output_json}")
    print(f"Saved NPZ:  {output_npz}")


def main() -> None:
    args = build_parser().parse_args()
    pattern_size = args.board_squares if args.board_squares is not None else args.inner_corners
    if args.square_size_mm <= 0:
        raise ValueError("--square-size-mm must be positive")

    if args.images:
        samples, image_size = load_samples_from_images(args.images, pattern_size, args.show_detections)
    else:
        samples, image_size = capture_samples_from_camera(args, pattern_size)

    if len(samples) < args.min_samples:
        raise RuntimeError(f"Need at least {args.min_samples} valid checkerboard views, got {len(samples)}")

    result = calibrate(samples, image_size, pattern_size, args.square_size_mm)
    save_result(
        result=result,
        samples=samples,
        image_size=image_size,
        pattern_size=pattern_size,
        square_size_mm=args.square_size_mm,
        output_json=args.output_json,
        output_npz=args.output_npz,
    )
    print_result(result, args.output_json, args.output_npz)


if __name__ == "__main__":
    main()
