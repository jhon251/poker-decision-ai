"""V2N — Frame Acquisition / RuntimeMetrics（采集循环 + 观测指标）。

- FrameAcquirer：capture() → frame_id++ → timestamp → digest →
  FrameStream（默认 2 FPS）；完全同画面不进完整 Pipeline（精确 digest，
  无近似去重）；窗口/DPI/分辨率变化 → 显式事件 + recalibration 标志；
- RuntimeMetrics：全部指标来自真实运行计数，禁止编造。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .capture import (CAPTURE_BACKEND_UNAVAILABLE, CAPTURE_FAILED,
                      CAPTURE_OK, LAYOUT_UNKNOWN, ScreenCaptureBackend,
                      detect_backend)
from .runtime import FrameStream, VisualEventBus, image_digest


@dataclass
class RuntimeMetrics:
    frames_captured: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    frames_deduplicated: int = 0
    capture_failures: int = 0
    layout_recalibrations: int = 0
    roi_changed_count: int = 0
    recognition_count: int = 0
    state_conflicts: int = 0
    stale_frames: int = 0
    latencies_ms: list = field(default_factory=list)
    backend_status: str = CAPTURE_BACKEND_UNAVAILABLE

    def record_latency(self, ms: float) -> None:
        self.latencies_ms.append(round(ms, 3))

    def percentiles(self) -> dict:
        xs = sorted(self.latencies_ms)
        n = len(xs)
        if not xs:
            return {"p50": None, "p95": None, "p99": None, "avg": None}
        return {"p50": round(xs[n // 2], 1),
                "p95": round(xs[int(n * 0.95)] if n > 1 else xs[-1], 1),
                "p99": round(xs[int(n * 0.99)] if n > 1 else xs[-1], 1),
                "avg": round(sum(xs) / n, 1)}

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k != "latencies_ms"}
        d.update(self.percentiles())
        return d


class FrameAcquirer:
    """采集循环：backend（或 replay 迭代器）→ 帧 → FrameStream。"""

    def __init__(self, fps: float = 2.0, backend: ScreenCaptureBackend | None
                 = None):
        self.fps = fps
        self.backend = backend
        self.stream = FrameStream(fps=fps)
        self.metrics = RuntimeMetrics()
        self.bus = VisualEventBus()
        self.frame_id = 0
        self._last_window = None
        self._last_dpi = None
        self._last_resolution = None
        self.needs_recalibration = False
        if backend is not None:
            self.metrics.backend_status = (CAPTURE_OK
                                           if backend.available()
                                           else
                                           CAPTURE_BACKEND_UNAVAILABLE)

    def acquire_from_capture(self) -> dict:
        """真实截图路径；不可用/失败 → 明确状态（不伪装新帧）。"""
        if self.backend is None or not self.backend.available():
            self.metrics.capture_failures += 1
            self.metrics.backend_status = CAPTURE_BACKEND_UNAVAILABLE
            return {"frame": None, "status": CAPTURE_BACKEND_UNAVAILABLE}
        res = self.backend.capture()
        self.metrics.frames_captured += 1
        if res["status"] != CAPTURE_OK:
            self.metrics.capture_failures += 1
            return {"frame": None, "status": res["status"],
                    "reason": res.get("reason")}
        self._check_geometry(res)
        return self._admit(res)

    def acquire_replay(self, image, *, w, h, captured_at=None) -> dict:
        """Replay 路径（测试/回放源；与真实路径共用 FrameStream 语义）。"""
        res = {"status": CAPTURE_OK, "image_bytes": None, "width": w,
               "height": h, "window_rect": (0, 0, w, h), "dpi_scale": 1.0,
               "captured_at": captured_at if captured_at is not None
               else float(self.frame_id)}
        self.metrics.frames_captured += 1
        self._check_geometry(res)
        res["_image"] = image
        return self._admit(res)

    def _check_geometry(self, res: dict) -> None:
        """窗口/DPI/分辨率变化 → 事件 + recalibration。"""
        win = res.get("window_rect")
        dpi = res.get("dpi_scale")
        resolution = (res.get("width"), res.get("height"))
        if self._last_window is not None and win != self._last_window:
            self.bus.emit("ROI_CHANGED", self.frame_id + 1,
                          {"reason": "WINDOW_MOVED"})
            self.needs_recalibration = True
        if self._last_dpi is not None and dpi != self._last_dpi:
            self.bus.emit("ROI_CHANGED", self.frame_id + 1,
                          {"reason": "DPI_CHANGED"})
            self.needs_recalibration = True
        if (self._last_resolution is not None
                and resolution != self._last_resolution):
            self.bus.emit("ROI_CHANGED", self.frame_id + 1,
                          {"reason": "RESOLUTION_CHANGED"})
            self.needs_recalibration = True
        if self.needs_recalibration:
            self.metrics.layout_recalibrations += 1
        self._last_window, self._last_dpi = win, dpi
        self._last_resolution = resolution

    def _admit(self, res: dict) -> dict:
        image = res.get("_image")
        digest = image_digest(image) if image is not None else ""
        acc = self.stream.accept(self.frame_id + 1, res["captured_at"],
                                 image if image is not None else _DummyImg(
                                     digest.encode()))
        if not acc["accepted"]:
            if acc["reason"] == "duplicate_frame":
                self.metrics.frames_deduplicated += 1
            else:
                self.metrics.frames_dropped += 1
            return {"frame": None, "status": "FRAME_SKIPPED",
                    "reason": acc["reason"]}
        self.frame_id += 1
        if acc.get("stale"):
            self.metrics.stale_frames += 1
            self.bus.emit("FRAME_STALE", self.frame_id,
                          {"gap_s": acc.get("gap_s")})
        return {"frame": {"frame_id": self.frame_id,
                          "captured_at": res["captured_at"],
                          "width": res.get("width"),
                          "height": res.get("height"),
                          "window_rect": res.get("window_rect"),
                          "dpi_scale": res.get("dpi_scale"),
                          "digest": digest}, "status": CAPTURE_OK}


class _DummyImg:
    def __init__(self, data):
        self.data = data


def detected_backend_status() -> dict:
    return detect_backend()
