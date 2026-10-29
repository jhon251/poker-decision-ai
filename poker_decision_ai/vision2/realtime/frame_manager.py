"""Frame Manager：轮询频率、去重（dedup）、staleness 与 Frame 载体。

- 默认 2 FPS；支持 1 / 2 / 5 / 10 FPS（FPS 只控制检测频率）。
- 去重：字节级相等比例（cheap 层）→ 相同画面不触发识别。
- 桌面长时间未变化 → FRAME_STALE；截图失败 → CAPTURE_FAILED。
- 禁止每帧触发远程多模态（由识别流水线/fallback gate 保证，见 fallback.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .capture import (CapturedFrame, CaptureResult, CaptureStatus,
                      ScreenCaptureProvider)
from .coords import GeometryContext
from .image import ImageData, byte_equality_ratio

ALLOWED_FPS = (1, 2, 5, 10)
DEFAULT_FPS = 2.0


class FrameStatus(str, Enum):
    FRESH = "FRESH"
    DUPLICATE = "DUPLICATE"
    WAIT_RATE_LIMIT = "WAIT_RATE_LIMIT"
    FRAME_STALE = "FRAME_STALE"
    CAPTURE_FAILED = "CAPTURE_FAILED"


@dataclass(frozen=True)
class Frame:
    frame_id: int
    captured_at: float
    image: ImageData
    geometry: GeometryContext
    source: str = "fixture"
    synthetic_test_only: bool = True
    deduplicated: bool = False

    @classmethod
    def from_capture(cls, captured: CapturedFrame, frame_id: int,
                     deduplicated: bool = False) -> "Frame":
        return cls(frame_id=frame_id, captured_at=captured.captured_at,
                   image=captured.image, geometry=captured.geometry,
                   source=captured.source,
                   synthetic_test_only=captured.synthetic_test_only,
                   deduplicated=deduplicated)


@dataclass
class FrameResult:
    status: FrameStatus
    frame: Frame | None = None
    error: str | None = None


@dataclass
class FrameManagerStats:
    polls: int = 0
    fresh: int = 0
    duplicates: int = 0
    rate_limited: int = 0
    stale: int = 0
    capture_failures: int = 0

    def to_dict(self) -> dict:
        return {"polls": self.polls, "fresh": self.fresh,
                "duplicates": self.duplicates, "rate_limited": self.rate_limited,
                "stale": self.stale, "capture_failures": self.capture_failures}


class FrameManager:
    def __init__(self, provider: ScreenCaptureProvider, *,
                 fps: float = DEFAULT_FPS,
                 dedup_threshold: float = 0.0,
                 stale_after_s: float = 3.0,
                 clock=None):
        import time as _time
        if fps not in ALLOWED_FPS:
            raise ValueError(f"FPS 必须属于 {ALLOWED_FPS}，实际 {fps}")
        self.provider = provider
        self.fps = float(fps)
        self.dedup_threshold = dedup_threshold
        self.stale_after_s = stale_after_s
        self.clock = clock or _time.monotonic
        self.interval = 1.0 / self.fps
        self.last_poll_at: float | None = None
        self.last_success_at: float | None = None
        self.last_change_at: float | None = None
        self.last_frame: Frame | None = None
        self.frame_id = 0
        self.stats = FrameManagerStats()

    def next_poll_at(self) -> float | None:
        if self.last_poll_at is None:
            return None
        return self.last_poll_at + self.interval

    def reset(self) -> None:
        self.last_poll_at = self.last_success_at = self.last_change_at = None
        self.last_frame = None
        self.frame_id = 0
        self.stats = FrameManagerStats()

    def poll(self, now: float | None = None) -> FrameResult:
        """按 FPS 轮询一帧（去重与 staleness 判定在此完成）。"""
        now = self.clock() if now is None else now
        self.stats.polls += 1
        if self.last_poll_at is not None and now - self.last_poll_at < self.interval:
            self.stats.rate_limited += 1
            return FrameResult(status=FrameStatus.WAIT_RATE_LIMIT)
        self.last_poll_at = now

        result: CaptureResult = self.provider.capture(now=now)
        if not result.ok:
            self.stats.capture_failures += 1
            return FrameResult(status=FrameStatus.CAPTURE_FAILED,
                               error=result.error or "capture failed")
        captured = result.frame
        self.last_success_at = now

        dedup = False
        if self.last_frame is not None:
            same_geo = (self.last_frame.image.same_geometry(captured.image)
                        and self.last_frame.geometry.to_dict()
                        == captured.geometry.to_dict())
            if same_geo:
                ratio = byte_equality_ratio(self.last_frame.image.rgba_bytes(),
                                            captured.image.rgba_bytes())
                # 帧级去重只认"字节完全相同"；任何细微变化都属于 ROI 层
                # cheap 检测的职责（pot 只变一个数字也必须被识别）
                if ratio >= (1.0 - self.dedup_threshold):
                    dedup = True

        if dedup:
            self.frame_id += 1
            frame = Frame.from_capture(captured, self.frame_id, deduplicated=True)
            self.last_frame = frame
            if self.last_change_at is not None \
                    and now - self.last_change_at > self.stale_after_s:
                self.stats.stale += 1
                return FrameResult(status=FrameStatus.FRAME_STALE,
                                   error=f"desktop unchanged for "
                                         f"{now - self.last_change_at:.2f}s")
            self.stats.duplicates += 1
            return FrameResult(status=FrameStatus.DUPLICATE, frame=frame)

        self.frame_id += 1
        frame = Frame.from_capture(captured, self.frame_id)
        self.last_frame = frame
        self.last_change_at = now
        self.stats.fresh += 1
        return FrameResult(status=FrameStatus.FRESH, frame=frame)
