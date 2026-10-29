"""Vision V2O — Real Capture Environment Enablement 测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.capture import (  # noqa: E402
    CAPTURE_OK, CtypesBitBltBackend, detect_backend)


def test_detect_backend_prefers_available():
    det = detect_backend()
    assert det["status"] == CAPTURE_OK
    assert det["backend"] in ("pillow_imagegrab", "mss", "ctypes_bitblt")


def test_bitblt_backend_available():
    b = CtypesBitBltBackend()
    assert b.available() is True


def test_bitblt_capture_shape():
    b = CtypesBitBltBackend()
    r = b.capture()
    assert r["status"] == CAPTURE_OK
    w, h = r["width"], r["height"]
    assert w > 0 and h > 0
    assert len(r["image_bytes"]) == w * h * 4      # RGBA 完整
    assert r["captured_at"] > 0


def test_bitblt_capture_consistent_size():
    b = CtypesBitBltBackend()
    r1 = b.capture()
    r2 = b.capture()
    assert (r1["width"], r1["height"]) == (r2["width"], r2["height"])


def test_smoke_test_artifact_exists():
    p = Path("data/vision2/live_capture_smoke_test.json")
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert len(d["frames"]) >= 3
    for f in d["frames"]:
        assert f["status"] == "CAPTURE_OK"
        assert f["width"] > 0 and f["bytes"] > 0
        assert f["capture_latency_ms"] >= 0


def test_live_runtime_report_exists():
    p = Path("data/vision2/live_runtime_report.json")
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["backend"] == "ctypes_bitblt"
    # 当前屏幕 1966×1106 ≠ profile 1848×1388 → LAYOUT_UNKNOWN（诚实）
    assert d["layout_status"] == "LAYOUT_UNKNOWN"
    assert d["counters"]["layout_recalibrations"] >= 1
    # 布局未知 → 不得用旧坐标识别 → 字段为空（0 猜测）
    assert d["fields"]["state_version"] == 0
    assert d["fields"]["fields"] == {}


def test_v2o_script_redlines():
    src = (Path(__file__).parent.parent / "scripts"
           / "phase_v2o_live.py").read_text(encoding="utf-8")
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "SendInput", "mouse_event", "keybd_event", "SetCursorPos",
              "PostMessage", "expected_value", "recommend", "exploit",
              "gto", "bluff")
    for bad in banned:
        assert bad not in src, f"含禁止 token {bad}"
    # ctypes 仅用于只读进程内存（GetProcessMemoryInfo）——零输入注入
    assert "GetProcessMemoryInfo" in src
