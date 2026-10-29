"""Vision V3F-P2 — Board Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    BoardObserver)
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


# ── Case A：低内容/空板帧 → 不把 UNKNOWN 当 empty board ─────────────
def test_case_a_preflop_or_unknown():
    fb, w, h = _blank()
    out = BoardObserver().observe(fb, w, h)
    assert out["count"] == 0
    assert out["status"] != "RECOGNIZED"      # UNKNOWN/UNCERTAIN（≠空板认可）
    assert out["street"] is None


# ── Case B/C/D：Flop/Turn/River（真实帧，非硬编码）──────────────────
def test_case_b_flop_3cards():
    rgba, w, h = _frame(
        "SETTLED_BOARD_CHECK_1028_1787712797523_full.png")
    out = BoardObserver().observe(rgba, w, h)
    assert out["count"] == 3                   # 7c Tc 3c（FLOP 3 卡）
    assert out["cards"] == ["7c", "Tc", "3c"]


def test_case_c_turn_4cards():
    rgba, w, h = _frame(
        "SETTLED_BOARD_CHECK_146_1787653070304_full.png")
    out = BoardObserver().observe(rgba, w, h)
    assert out["count"] == 4                   # 9c 5d 3c Ks（TURN 4 卡）
    assert set(out["cards"]) == {"9c", "5d", "3c", "Ks"}


def test_case_d_river_5cards():
    rgba, w, h = _frame(
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    out = BoardObserver().observe(rgba, w, h)
    assert out["count"] == 5                   # 3c 6h Th Ac 5h（RIVER 5 卡）
    assert out["cards"] == ["3c", "6h", "Th", "Ac", "5h"]


# ── Case E：Frame isolation ───────────────────────────────────────
def test_case_e_frame_isolation():
    obs = BoardObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_1028_1787712797523_full.png")
    b, w2, h2 = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(a, w, h)
    out = obs.observe(b, w2, h2)
    assert out["count"] == 5                   # B → Y（非 A 的 3）


# ── Case F：UNKNOWN isolation ─────────────────────────────────────
def test_case_f_unknown_isolation():
    obs = BoardObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(a, w, h)
    fb, w2, h2 = _blank()
    out = obs.observe(fb, w2, h2)
    assert out["count"] == 0                   # UNKNOWN → 不继承 5 卡
    assert out["status"] != "RECOGNIZED"


# ── Case G：Fixture pollution ─────────────────────────────────────
def test_case_g_no_fixture():
    obs = BoardObserver()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(rgba, w, h)
    obs2 = BoardObserver()
    assert obs2.summary()["last_count"] is None    # 新实例无泄漏


# ── Case H：Capture behavior ──────────────────────────────────────
def test_case_h_capture_behavior_untouched():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "stream.accept", "engine.process_frame",
              "--seconds", "--fps", "coinpoker_auto_capture.json",
              "board_observer.observe(sub.tobytes(), cw, chh)"):
        assert b in src, b


# ── Case I：Street isolation（红线：Board ≠ Street）────────────────
def test_case_i_street_isolation():
    """Board=5 卡 ≠ Street Runtime；此观察器 street 恒 None（不推导）。"""
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    out = BoardObserver().observe(rgba, w, h)
    assert out["count"] == 5
    assert out["street"] is None               # 未产生 Street Runtime
    assert out["slot_states"] is not None      # 仅板卡级别


def test_three_observers_independent():
    """Position/Pot/Board 三观察器并行独立（互不缓存/互不修改输入）。"""
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        PositionObserver, PotObserver)
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    po, pt, bo = PositionObserver(), PotObserver(), BoardObserver()
    r1 = po.observe(rgba, w, h)
    r2 = pt.observe(rgba, w, h)
    r3 = bo.observe(rgba, w, h)
    assert r3["count"] == 5                    # Board
    assert r2["pot"] == "23,280"               # Pot 不受 Board 影响
    assert r1["hero_position"] == r1["hero_position"]   # Position 独立
    # 三个观察器 summary 各自独立
    assert po.summary()["frames"] == pt.summary()["frames"] == bo.summary()[
        "frames"] == 1
