"""只读 Screen Capture 接口。

红线（硬约束）：
  - 截图 = 只读观察。本包与实现均不含鼠标/键盘/点击/投注注入；
  - 真实系统截图后端位于 sibling 包 vision2_platform（可选依赖，可用性门控）；
  - 本包提供接口 + 确定性 Fixture 后端（synthetic_test_only=True）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum

from .coords import GeometryContext
from .image import ImageData


class CaptureStatus(str, Enum):
    CAPTURE_OK = "CAPTURE_OK"
    CAPTURE_FAILED = "CAPTURE_FAILED"


@dataclass
class CapturedFrame:
    image: ImageData
    captured_at: float
    geometry: GeometryContext
    source: str = "fixture"
    synthetic_test_only: bool = True
    extra: dict | None = None

    def to_dict(self) -> dict:
        return {
            "captured_at": self.captured_at,
            "width": self.image.width,
            "height": self.image.height,
            "channels": self.image.channels,
            "source": self.source,
            "synthetic_test_only": self.synthetic_test_only,
            "geometry": self.geometry.to_dict(),
            "extra": self.extra,
        }


@dataclass
class CaptureResult:
    status: CaptureStatus
    frame: CapturedFrame | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is CaptureStatus.CAPTURE_OK and self.frame is not None


class ScreenCaptureProvider(ABC):
    """可插拔只读截图接口。实现必须是纯观察（无任何控制注入）。"""
    name = "screen_capture"

    @abstractmethod
    def capture(self, now: float | None = None) -> CaptureResult:
        """截取一帧。失败 → CAPTURE_FAILED（带原因），绝不返回伪造帧。"""


class FixtureCaptureProvider(ScreenCaptureProvider):
    """确定性帧序列回放（synthetic_test_only=True；不代表真实视觉能力）。

    - cycle=False：序列耗尽 → CAPTURE_FAILED("fixture_exhausted")；
    - fail_after：第 N 次成功后注入失败（测试 CAPTURE_FAILED 路径）。
    """
    name = "fixture_capture"

    def __init__(self, frames=None, *, cycle: bool = False,
                 fail_after: int | None = None,
                 fail_error: str = "fixture_injected_failure"):
        self.frames = list(frames or [])
        self.cycle = cycle
        self.fail_after = fail_after
        self.fail_error = fail_error
        self.index = 0
        self.captures = 0

    def add_frame(self, frame: CapturedFrame) -> None:
        self.frames.append(frame)

    def capture(self, now: float | None = None) -> CaptureResult:
        self.captures += 1
        if self.fail_after is not None and self.captures > self.fail_after:
            return CaptureResult(status=CaptureStatus.CAPTURE_FAILED,
                                 error=self.fail_error)
        if self.index >= len(self.frames):
            if self.cycle and self.frames:
                base = self.frames[self.index % len(self.frames)]
                frame = replace(base, captured_at=(now if now is not None
                                                   else base.captured_at))
                self.index += 1
                return CaptureResult(status=CaptureStatus.CAPTURE_OK, frame=frame)
            return CaptureResult(status=CaptureStatus.CAPTURE_FAILED,
                                 error="fixture_exhausted")
        frame = self.frames[self.index]
        self.index += 1
        if now is not None:
            frame = replace(frame, captured_at=now)
        return CaptureResult(status=CaptureStatus.CAPTURE_OK, frame=frame)
