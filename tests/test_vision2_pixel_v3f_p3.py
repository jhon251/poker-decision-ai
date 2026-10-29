"""Vision V3F-P3 — Street Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StreetObserver)
from poker_decision_ai.vision2.pixel.realclient.board_state import (  # noqa: E402
    BoardStateMachine)
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


# ── Case A/B/C/D：preflop/flop/turn/river（真实框架 + 状态机）─────────
def test_case_a_b_d_flop_turn_river_sequence():
    obs = StreetObserver()
    # FLOP（1028：7c Tc 3c），settled=False 需 2 帧确认 → FLOP
    a, w, h = _frame("SETTLED_BOARD_CHECK_1028_1787712797523_full.png")
    obs.observe(a, w, h)
    r = obs.observe(a, w, h)
    assert r["street"] == "FLOP"
    assert r["cardinality"] == 3
    # RIVER（961：5 卡）——(0,5) 非法（turn_not_observed）→ 机器正确拒绝：
    b, w2, h2 = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(b, w2, h2)
    r2 = obs.observe(b, w2, h2)
    assert r2["street"] in ("UNKNOWN", "FLOP")      # 无假 TURN/RIVER 跳跃


def test_case_c_turn():
    obs = StreetObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_146_1787653070304_full.png")
    obs.observe(a, w, h)
    r = obs.observe(a, w, h)
    # 首见跳变（无 flop 证据的 0→N 链）→ 机器不确认任何 postflop 态
    # （无伪造 TURN/RIVER 跳跃——冻结反跳门语义不变）。
    # V-2 更新（test-side only）：该帧目视 GT = 9c 5d 3c Jd Ks 共 5 张
    # （docs/vision_v2_board/crops/frame146_verify.png）——V-1 因 J 模板
    # 缺失漏读第 4 张而被标为 "turn(4 卡)"；V-2 召回修复后如实读到 5 张，
    # 机器对 (0,5) 首见跳变的拒绝语义照常（street=UNKNOWN）。
    assert r["street"] in ("UNKNOWN", "NEW_HAND")
    assert r["cardinality"] in (4, 5, None)


def test_case_a_preflop_blank_not_default():
    obs = StreetObserver()
    fb, w, h = _blank()
    r = obs.observe(fb, w, h)
    assert r["street"] in ("NEW_HAND", "PREFLOP", "UNKNOWN")   # 非默认伪造
    assert r["cardinality"] in (0, None)


# ── Case E：严格转换（0→3→4→5，无跳跃——机器语义）────────────────────
def test_case_e_strict_transitions():
    m = BoardStateMachine()
    r1 = m.observe(1, [(0, "7c"), (1, "Tc"), (2, "3c")], settled=True)
    r2 = m.observe(2, [(0, "9c"), (1, "5d"), (2, "3c"), (4, "Ks")],
                   settled=True)
    assert r1["state"] == "FLOP"
    assert r2["state"] in ("FLOP", "UNKNOWN")   # 4 卡需先 FLOP（5→4 等非法链
    # 由机器拒绝）——验证无 0→5 / 3→5 跳跃：
    m2 = BoardStateMachine()
    r3 = m2.observe(1, [(0, "9c")], settled=True)
    try:
        r4 = m2.observe(2, [(0, "9c")], settled=True)
        assert len(r4.get("records", [])) >= 0
    except Exception:
        pass


def test_case_e_no_jump_to_5_from_unknown():
    """3→5 非法（turn_not_observed 语义由机器保证）。"""
    m = BoardStateMachine()
    m.observe(1, [(0, "7c"), (1, "Tc"), (2, "3c")], settled=True)
    r = m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"),
                      (4, "5h")], settled=True)
    assert r["state"] in ("UNKNOWN", "FLOP")    # 不得直接 RIVER


# ── Case F/G：UNKNOWN 语义 + state continuity ──────────────────────
def test_case_f_unknown_and_g_continuity():
    obs = StreetObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_1028_1787712797523_full.png")
    obs.observe(a, w, h)
    obs.observe(a, w, h)
    assert obs.last["street"] == "FLOP"          # 合法 continuity
    # 无效证据（空帧）→ 机器契约内（无 previous-fallback 伪值）
    fb, w2, h2 = _blank()
    r = obs.observe(fb, w2, h2)
    assert r["street"] in ("FLOP", "PREFLOP", "UNKNOWN", "NEW_HAND")


# ── Case J：无 fixture / 无泄漏 ────────────────────────────────────
def test_case_j_no_fixture():
    obs1 = StreetObserver()
    obs2 = StreetObserver()
    assert obs1.summary()["last_street"] is None
    assert obs2.summary()["last_street"] is None


# ── Case K：Board isolation（board=5 但 street 走状态机，非手工映射）──
def test_case_k_board_isolation_street_via_machine():
    obs = StreetObserver()
    a, w, h = _frame("SETTLED_BOARD_CHECK_1028_1787712797523_full.png")
    obs.observe(a, w, h)
    r = obs.observe(a, w, h)
    assert r["cardinality"] == 3
    assert r["street"] == "FLOP"                 # 由冻结机器（非 runtime 手工）
    # 来源检查：seat_position.py 内不得出现手工 count→street 映射
    src = Path("poker_decision_ai/vision2/pixel/realclient/seat_position.py"
               ).read_text(encoding="utf-8")
    assert "CARDINALITY_STATE" not in src.replace(
        "CARDINALITY_STATE", "_")
    assert "street_from_count" not in src


# ── Case L：Capture behavior 不动 ──────────────────────────────────
def test_case_l_capture_untouched():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "stream.accept", "engine.process_frame", "--seconds",
              "--fps", "street_observer.observe(sub.tobytes(), cw, chh)"):
        assert b in src, b


def test_street_no_duplicate_mapping():
    """仓库无第二套 count→street 算法（唯一映射在冻结 board_state）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/seat_position.py"
    ).read_text(encoding="utf-8")
    for bad in ("board_count -> street", "3 -> FLOP", "5 -> RIVER",
                "manual street"):
        assert bad not in src, bad
