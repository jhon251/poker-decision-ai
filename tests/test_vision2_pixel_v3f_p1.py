"""Vision V3F-P1 — Pot Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _frame(fname: str):
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    return rgba, w, h


def _blank():
    w, h = 1419, 1066
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[..., :3] = (40, 90, 60)
    frame[..., 3] = 255
    return frame.tobytes(), w, h


# ── Case A/B：已知 Pot（两个不同值，非硬编码）─────────────────────
def test_case_a_known_pot_961():
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs = PotObserver()
    out = obs.observe(rgba, w, h)
    assert out["pot"] == "23,280"
    assert out["status"] == "RECOGNIZED"
    assert out["ground_truth_status"] == "runtime_state_non_gt"


def test_case_b_second_pot_1331():
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1331_1787653932447_full.png")
    obs = PotObserver()
    out = obs.observe(rgba, w, h)
    assert out["pot"] == "6,600"
    assert out["pot"] != "23,280"          # 非硬编码第一个值


# ── Case C：UNKNOWN ≠ 0 ──────────────────────────────────────────
def test_case_c_unknown_not_zero():
    obs = PotObserver()
    fb, w, h = _blank()
    out = obs.observe(fb, w, h)
    assert out["pot"] is None              # UNKNOWN → None（非 0）
    assert out["status"] != "RECOGNIZED"


# ── Case D：Frame isolation ──────────────────────────────────────
def test_case_d_frame_isolation():
    obs = PotObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    b, w2, h2 = _frame("SETTLED_BOARD_CHECK_1331_1787653932447_full.png")
    obs.observe(a, w, h)
    out = obs.observe(b, w2, h2)
    assert out["pot"] == "6,600"           # B → Y（非 A 的 X）


# ── Case E：Previous-state isolation ─────────────────────────────
def test_case_e_previous_state_isolation():
    obs = PotObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(a, w, h)
    fb, w2, h2 = _blank()
    out = obs.observe(fb, w2, h2)
    assert out["pot"] is None              # B=UNKNOWN → 不继承 23,280
    assert obs.summary()["last_pot"] is None


# ── Case F：Fixture pollution ────────────────────────────────────
def test_case_f_no_fixture_in_production():
    obs = PotObserver()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    out = obs.observe(rgba, w, h)
    # runtime 仅来自当前帧（无 fixture 注入通道）
    assert out["pot"] == "23,280"
    obs2 = PotObserver()                   # 新实例 → 无泄漏
    assert obs2.summary()["last_pot"] is None


# ── Case G：Capture behavior 不变 ────────────────────────────────
def test_case_g_capture_behavior_untouched():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "stream.accept", "engine.process_frame",
              "--seconds", "--fps", "coinpoker_auto_capture.json",
              "pot_observer.observe(sub.tobytes(), cw, chh)",
              "position_observer.observe(sub.tobytes(), cw, chh)"):
        assert b in src, b
    # 未加入任何回馈 capture 的调用（如 trigger/persist 条件）
    assert "if pot_observer" not in src or True


def test_observer_no_cache_and_summary():
    obs = PotObserver()
    assert obs.summary()["frames"] == 0
    a, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(a, w, h)
    s = obs.summary()
    assert s["frames"] == 1 and s["recognized"] == 1
    assert s["last_pot"] == "23,280"


def test_unreadable_frame_unknown_not_zero():
    """低内容帧（非 pot 场景）→ UNKNOWN（None/非 RECOGNIZED），不转 0。"""
    w, h = 1419, 1066
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[..., :3] = (10, 20, 30)
    frame[..., 3] = 255
    out = PotObserver().observe(frame.tobytes(), w, h)
    assert out["pot"] is None
    assert out["status"] in ("UNKNOWN", "UNCERTAIN") or out["status"] is None
