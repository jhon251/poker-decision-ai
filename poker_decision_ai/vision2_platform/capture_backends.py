"""真实系统截图后端（只读）。

依赖可选（Pillow ImageGrab 或 mss；两者皆未安装时 available()=False）。
截图 API 只产生像素字节，不涉及任何鼠标/键盘/点击/窗口消息注入。
窗口级定位不在 V1 截图后端范围：当前实现为全屏捕获（geometry.window_rect
=None），anchor 校准层在归一化空间工作，不依赖窗口句柄。
"""
from __future__ import annotations

import time as _time

from poker_decision_ai.vision2.realtime.capture import (CaptureResult,
                                                        CaptureStatus,
                                                        CapturedFrame,
                                                        ScreenCaptureProvider)
from poker_decision_ai.vision2.realtime.coords import GeometryContext
from poker_decision_ai.vision2.realtime.image import ImageData


def _pil_grab():
    """Pillow ImageGrab 适配：() -> (rgba_bytes, w, h)；不可用 → None。"""
    try:
        from PIL import ImageGrab  # noqa: PLC0415 - 可选依赖
    except Exception:
        return None

    def grab():
        img = ImageGrab.grab(all_screens=True)
        rgba = img.convert("RGBA").tobytes()
        return rgba, img.size[0], img.size[1]
    return grab


def _mss_grab():
    """mss 适配：() -> (rgba_bytes, w, h)；BGRA → RGBA 确定性转换。"""
    try:
        import mss  # noqa: PLC0415 - 可选依赖
    except Exception:
        return None

    def grab():
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
            bgra = bytearray(shot.bgra)
            w, h = shot.size.width, shot.size.height
            for i in range(0, len(bgra), 4):
                bgra[i], bgra[i + 2] = bgra[i + 2], bgra[i]
            return bytes(bgra), w, h
    return grab


def _available_grab():
    return _pil_grab() or _mss_grab()


class SystemScreenCaptureProvider(ScreenCaptureProvider):
    """全屏只读截图（可选后端，注入 grab_fn 可测）。"""
    name = "system_screen"

    def __init__(self, *, dpi_scale: float = 1.0, grab_fn=None,
                 clock=_time.monotonic):
        self.dpi_scale = dpi_scale
        self._grab = grab_fn
        self.clock = clock

    def available(self) -> bool:
        return (self._grab is not None) or (_available_grab() is not None)

    def unavailable_reason(self) -> str:
        if self._grab is not None:
            return ""
        if _pil_grab() is None and _mss_grab() is None:
            return ("no read-only capture backend available "
                    "(install Pillow or mss)")
        return ""

    def capture(self, now: float | None = None) -> CaptureResult:
        now = self.clock() if now is None else now
        grab = self._grab or _available_grab()
        if grab is None:
            return CaptureResult(status=CaptureStatus.CAPTURE_FAILED,
                                 error=self.unavailable_reason())
        try:
            rgba, w, h = grab()
            if len(rgba) != w * h * 4:
                return CaptureResult(
                    status=CaptureStatus.CAPTURE_FAILED,
                    error=f"capture returned {len(rgba)} bytes, expected "
                          f"{w * h * 4}")
            geometry = GeometryContext(
                screen_size=(float(w), float(h)), window_rect=None,
                client_offset=(0.0, 0.0),
                client_size_physical=(float(w), float(h)),
                dpi_scale=float(self.dpi_scale))
            frame = CapturedFrame(
                image=ImageData.from_rgba_bytes(w, h, rgba),
                captured_at=now, geometry=geometry, source=self.name,
                synthetic_test_only=False)
            return CaptureResult(status=CaptureStatus.CAPTURE_OK, frame=frame)
        except Exception as e:  # 截图失败 → 明确 CAPTURE_FAILED
            return CaptureResult(status=CaptureStatus.CAPTURE_FAILED,
                                 error=f"capture error: {e}")
