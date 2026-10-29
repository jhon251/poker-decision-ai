"""V2N — Real Screen Capture Backend（platform isolation，只读截图）。

原则：
  - 不假设依赖已安装：available() 以 importlib 实测为准；
  - backend 不可用 → 明确 CAPTURE_BACKEND_UNAVAILABLE，绝不伪装成功；
  - 只允许只读截图 API（Pillow ImageGrab / mss）；禁止任何窗口控制；
  - capture 失败 → 明确状态进入 provenance，绝不复用旧图伪装新帧。
"""
from __future__ import annotations

import importlib.util
import time

#: capture 状态（冻结枚举）
CAPTURE_OK = "CAPTURE_OK"
CAPTURE_BACKEND_UNAVAILABLE = "CAPTURE_BACKEND_UNAVAILABLE"
CAPTURE_FAILED = "CAPTURE_FAILED"
WINDOW_NOT_FOUND = "WINDOW_NOT_FOUND"
WINDOW_MINIMIZED = "WINDOW_MINIMIZED"
BLACK_FRAME = "BLACK_FRAME"
STALE_FRAME = "STALE_FRAME"
DPI_UNKNOWN = "DPI_UNKNOWN"
LAYOUT_UNKNOWN = "LAYOUT_UNKNOWN"


class ScreenCaptureBackend:
    """统一截图后端接口（只读）。"""
    name = "abstract"

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return "abstract backend"

    def capture(self) -> dict:
        """→ {status, image_bytes|None, width, height, window_rect,
        dpi_scale, captured_at}。失败 = 明确状态，不伪装。"""
        return {"status": CAPTURE_BACKEND_UNAVAILABLE,
                "reason": self.unavailable_reason(), "image_bytes": None,
                "width": None, "height": None, "window_rect": None,
                "dpi_scale": None, "captured_at": time.time()}


class PillowImageGrabBackend(ScreenCaptureBackend):
    name = "pillow_imagegrab"

    def __init__(self):
        self._pil = importlib.util.find_spec("PIL") is not None

    def available(self) -> bool:
        return self._pil

    def unavailable_reason(self) -> str:
        return "Pillow 未安装（importlib 实测 MISSING）"

    def capture(self) -> dict:
        if not self.available():
            return {"status": CAPTURE_BACKEND_UNAVAILABLE,
                    "reason": self.unavailable_reason(), "image_bytes": None,
                    "width": None, "height": None, "window_rect": None,
                    "dpi_scale": None, "captured_at": time.time()}
        try:
            from PIL import ImageGrab                      # noqa: PLC0415
            img = ImageGrab.grab()                          # 只读全屏截图
            raw = img.convert("RGBA").tobytes()
            return {"status": CAPTURE_OK, "image_bytes": raw,
                    "width": img.width, "height": img.height,
                    "window_rect": None, "dpi_scale": None,
                    "captured_at": time.time()}
        except Exception as e:                              # noqa: BLE001
            return {"status": CAPTURE_FAILED, "reason": str(e),
                    "image_bytes": None, "width": None, "height": None,
                    "window_rect": None, "dpi_scale": None,
                    "captured_at": time.time()}


class MSSBackend(ScreenCaptureBackend):
    name = "mss"

    def __init__(self):
        self._mss = importlib.util.find_spec("mss") is not None

    def available(self) -> bool:
        return self._mss

    def unavailable_reason(self) -> str:
        return "mss 未安装（importlib 实测 MISSING）"

    def capture(self) -> dict:
        if not self.available():
            return {"status": CAPTURE_BACKEND_UNAVAILABLE,
                    "reason": self.unavailable_reason(), "image_bytes": None,
                    "width": None, "height": None, "window_rect": None,
                    "dpi_scale": None, "captured_at": time.time()}
        try:
            import mss                                    # noqa: PLC0415
            with mss.mss() as sct:
                mon = sct.monitors[0]                       # 只读主屏
                shot = sct.grab(mon)
                raw = shot.rgb + (b"\xff" * shot.width * shot.height)
                return {"status": CAPTURE_OK, "image_bytes": raw,
                        "width": shot.width, "height": shot.height,
                        "window_rect": None, "dpi_scale": None,
                        "captured_at": time.time()}
        except Exception as e:                              # noqa: BLE001
            return {"status": CAPTURE_FAILED, "reason": str(e),
                    "image_bytes": None, "width": None, "height": None,
                    "window_rect": None, "dpi_scale": None,
                    "captured_at": time.time()}


class CtypesBitBltBackend(ScreenCaptureBackend):
    """纯标准库只读截屏（ctypes + user32/gdi32 BitBlt → RGBA）。

    只做屏幕读取：GetDC/BitBlt/GetDIBits——零输入注入、零窗口控制、
    零消息发送。Pillow/mss 不可用（pip 阻塞）时的唯一真实捕获通道。
    """

    name = "ctypes_bitblt"

    def __init__(self):
        self._user32 = None
        self._gdi32 = None
        try:
            import ctypes
            from ctypes import wintypes
            self._user32 = ctypes.windll.user32
            self._gdi32 = ctypes.windll.gdi32
        except Exception:  # noqa: BLE001
            self._user32 = self._gdi32 = None

    def available(self) -> bool:
        return self._user32 is not None and self._gdi32 is not None

    def unavailable_reason(self) -> str:
        return "ctypes user32/gdi32 不可用"

    def capture(self) -> dict:
        """全屏捕获（只读）。"""
        return self._bitblt(hwnd=None)

    def capture_window(self, hwnd: int) -> dict:
        """指定窗口区域捕获（GetWindowDC 只读）。"""
        return self._bitblt(hwnd=int(hwnd))

    def _bitblt(self, hwnd: int | None) -> dict:
        if not self.available():
            return {"status": CAPTURE_BACKEND_UNAVAILABLE,
                    "reason": self.unavailable_reason(), "image_bytes": None,
                    "width": None, "height": None, "window_rect": None,
                    "dpi_scale": None, "captured_at": time.time()}
        try:
            import ctypes
            from ctypes import wintypes
            u32, g32 = self._user32, self._gdi32
            if hwnd is None:
                w = int(u32.GetSystemMetrics(0))
                h = int(u32.GetSystemMetrics(1))
                hdc = u32.GetDC(0)                       # 只读屏幕 DC
                win = None
            else:
                r = wintypes.RECT()
                if not u32.GetWindowRect(hwnd, ctypes.byref(r)):
                    return {"status": CAPTURE_FAILED,
                            "reason": "getwindowrect_failed",
                            "image_bytes": None, "width": None,
                            "height": None, "window_rect": None,
                            "dpi_scale": None, "captured_at": time.time()}
                w = int(r.right - r.left)
                h = int(r.bottom - r.top)
                hdc = u32.GetWindowDC(hwnd)              # 只读窗口 DC
                win = (r.left, r.top, w, h)
            if w <= 0 or h <= 0:
                u32.ReleaseDC(hwnd or 0, hdc)
                return {"status": CAPTURE_FAILED, "reason": "zero_size",
                        "image_bytes": None, "width": w, "height": h,
                        "window_rect": win, "dpi_scale": None,
                        "captured_at": time.time()}
            memdc = g32.CreateCompatibleDC(hdc)
            bmp = g32.CreateCompatibleBitmap(hdc, w, h)
            g32.SelectObject(memdc, bmp)
            ok = g32.BitBlt(memdc, 0, 0, w, h, hdc, 0, 0, 0x00CC0020)
            u32.ReleaseDC(hwnd or 0, hdc)
            if not ok:
                g32.DeleteDC(memdc)
                g32.DeleteObject(bmp)
                return {"status": CAPTURE_FAILED, "reason": "bitblt_failed",
                        "image_bytes": None, "width": w, "height": h,
                        "window_rect": win, "dpi_scale": None,
                        "captured_at": time.time()}
            # BITMAPINFOHEADER + GetDIBits → 32bpp BGRX
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [("biSize", wintypes.DWORD),
                            ("biWidth", wintypes.LONG),
                            ("biHeight", wintypes.LONG),
                            ("biPlanes", wintypes.WORD),
                            ("biBitCount", wintypes.WORD),
                            ("biCompression", wintypes.DWORD),
                            ("biSizeImage", wintypes.DWORD),
                            ("biXPelsPerMeter", wintypes.LONG),
                            ("biYPelsPerMeter", wintypes.LONG),
                            ("biClrUsed", wintypes.DWORD),
                            ("biClrImportant", wintypes.DWORD)]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h                      # top-down
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0
            buf = ctypes.create_string_buffer(w * h * 4)
            got = g32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
            g32.DeleteDC(memdc)
            g32.DeleteObject(bmp)
            if not got:
                return {"status": CAPTURE_FAILED, "reason": "getdibits_failed",
                        "image_bytes": None, "width": w, "height": h,
                        "window_rect": win, "dpi_scale": None,
                        "captured_at": time.time()}
            raw = bytearray(w * h * 4)
            # BGRX → RGBA（numpy 向量化，避免纯 Python O(w·h) 循环）
            try:
                import numpy as np
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                rgba = np.empty((h, w, 4), dtype=np.uint8)
                rgba[..., 0] = arr[..., 2]
                rgba[..., 1] = arr[..., 1]
                rgba[..., 2] = arr[..., 0]
                rgba[..., 3] = 255
                raw = bytearray(rgba.tobytes())
            except Exception:  # noqa: BLE001  兜底：保留 BGRX（调用方按需处理）
                raw = bytearray(buf)
            return {"status": CAPTURE_OK, "image_bytes": bytes(raw),
                    "width": w, "height": h, "window_rect": win,
                    "dpi_scale": None, "captured_at": time.time()}
        except Exception as e:  # noqa: BLE001
            return {"status": CAPTURE_FAILED, "reason": str(e),
                    "image_bytes": None, "width": None, "height": None,
                    "window_rect": None, "dpi_scale": None,
                    "captured_at": time.time()}


def detect_backend() -> dict:
    """实测选择可用后端（mss/Pillow/纯标准库 BitBlt）。"""
    for b in (PillowImageGrabBackend(), MSSBackend(), CtypesBitBltBackend()):
        if b.available():
            return {"backend": b.name, "status": CAPTURE_OK}
    return {"backend": None, "status": CAPTURE_BACKEND_UNAVAILABLE,
            "reason": ("Pillow ImageGrab / mss / 纯标准库 BitBlt 均不可用")}
