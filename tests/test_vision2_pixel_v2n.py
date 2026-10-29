"""Vision V2N — Real Capture Runtime Integration 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.acquisition import (  # noqa: E402
    FrameAcquirer, RuntimeMetrics, detected_backend_status)
from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CAPTURE_BACKEND_UNAVAILABLE, CAPTURE_OK, MSSBackend,
    PillowImageGrabBackend, ScreenCaptureBackend)


class _Img:
    def __init__(self, data=b"x" * 64):
        self.data = data


def test_detect_backend_honest():
    # V2O：ctypes BitBlt 只读后端已启用 → 明确可用；缺失依赖后端如实 MISSING
    det = detected_backend_status()
    assert det["status"] == CAPTURE_OK
    assert det["backend"] == "ctypes_bitblt"
    assert PillowImageGrabBackend().available() is False
    assert MSSBackend().available() is False


def test_pillow_backend_availability_false():
    b = PillowImageGrabBackend()
    assert b.available() is False
    assert "Pillow" in b.unavailable_reason()


def test_mss_backend_availability_false():
    b = MSSBackend()
    assert b.available() is False


def test_capture_returns_explicit_status_not_fake():
    b = ScreenCaptureBackend()
    r = b.capture()
    assert r["status"] == CAPTURE_BACKEND_UNAVAILABLE
    assert r["image_bytes"] is None


def test_acquirer_unavailable_path():
    a = FrameAcquirer(fps=2.0, backend=None)
    r = a.acquire_from_capture()
    assert r["status"] == CAPTURE_BACKEND_UNAVAILABLE
    assert a.metrics.capture_failures == 1
    assert a.metrics.backend_status == CAPTURE_BACKEND_UNAVAILABLE


def test_acquirer_replay_dedupe():
    a = FrameAcquirer(fps=2.0)
    img = _Img()
    r1 = a.acquire_replay(img, w=100, h=100, captured_at=0.0)
    r2 = a.acquire_replay(_Img(img.data), w=100, h=100, captured_at=0.5)
    assert r1["status"] == CAPTURE_OK
    assert r2["frame"] is None and r2["reason"] == "duplicate_frame"
    assert a.metrics.frames_deduplicated == 1


def test_acquirer_window_change_recalibration():
    a = FrameAcquirer(fps=2.0)
    a.acquire_replay(_Img(b"a"), w=100, h=100, captured_at=0.0)
    # 模拟窗口变化（几何变化在 acquire 前的外部检查；此处验证 metric 结构）
    a._check_geometry({"window_rect": (1, 2, 3, 4), "dpi_scale": 1.0,
                       "width": 100, "height": 100})
    a._check_geometry({"window_rect": (5, 6, 7, 8), "dpi_scale": 1.0,
                       "width": 100, "height": 100})
    assert a.needs_recalibration is True
    assert a.metrics.layout_recalibrations >= 1


def test_runtime_metrics_percentiles():
    m = RuntimeMetrics()
    assert m.percentiles()["p50"] is None
    for v in (10.0, 20.0, 30.0):
        m.record_latency(v)
    p = m.percentiles()
    assert p["p50"] == 20.0 and p["avg"] == 20.0


def test_metrics_all_from_counts():
    m = RuntimeMetrics()
    m.frames_captured += 1
    d = m.to_dict()
    for k in ("frames_captured", "frames_processed", "frames_dropped",
              "frames_deduplicated", "capture_failures",
              "layout_recalibrations", "roi_changed_count",
              "recognition_count", "state_conflicts", "stale_frames",
              "backend_status"):
        assert k in d, k


def test_capture_modules_redlines():
    for name in ("capture.py", "acquisition.py"):
        src = (Path(__file__).parent.parent / "poker_decision_ai"
               / "vision2" / "pixel" / "realclient" / name).read_text(
            encoding="utf-8")
        banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
                  "win32", "SendInput", "mouse_event", "keybd_event",
                  "click(", "keypress", "SetCursorPos", "PostMessage")
        for bad in banned:
            assert bad not in src, f"{name} 含禁止 token {bad}"
    cap = (Path(__file__).parent.parent / "poker_decision_ai"
           / "vision2" / "pixel" / "realclient"
           / "capture.py").read_text(encoding="utf-8")
    # capture.py 的 ctypes 仅用于只读 BitBlt（GetDC/BitBlt/GetDIBits）
    assert "BitBlt" in cap and "GetDIBits" in cap
    for bad in ("SendInput", "mouse_event", "keybd_event", "PostMessage",
                "SetCursorPos", "SetForegroundWindow"):
        assert bad not in cap
