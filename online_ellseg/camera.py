from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 0
    height: int = 0
    fps: int = 0
    backend: str = "dshow"


class LatestFrameCamera:
    """Capture thread that keeps only the newest frame to avoid latency buildup."""

    def __init__(self, config: CameraConfig):
        import cv2

        self.cv2 = cv2
        self.config = config
        backend = self._backend_flag(cv2, config.backend)
        self.capture = cv2.VideoCapture(config.index, backend) if backend is not None else cv2.VideoCapture(config.index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {config.index}")
        if config.width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        if config.height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        if config.fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, config.fps)

        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LatestFrameCamera":
        self._thread = threading.Thread(target=self._loop, name="LatestFrameCamera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            timestamp = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._timestamp = timestamp

    def read_latest(self) -> Tuple[bool, Optional[object], float]:
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            return True, self._frame.copy(), self._timestamp

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.capture.release()

    @staticmethod
    def _backend_flag(cv2, backend: str):
        backend = backend.lower()
        if backend in ("", "auto", "any"):
            return None
        if backend == "dshow":
            return cv2.CAP_DSHOW
        if backend == "msmf":
            return cv2.CAP_MSMF
        if backend == "vfw":
            return cv2.CAP_VFW
        raise ValueError(f"Unsupported camera backend: {backend}")
