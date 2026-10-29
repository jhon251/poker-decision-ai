"""Vision V3F-P9 — Runtime Snapshot Aggregation 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    BoardObserver, HeroCardsObserver, PositionObserver, PotObserver,
    StackObserver, StatusObserver, StreetObserver)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _observers():
    return {
        "position_runtime": PositionObserver(window="1419"),
        "pot_runtime": PotObserver(),
        "board_runtime": BoardObserver(),
        "street_runtime": StreetObserver(),
        "hero_cards_runtime": HeroCardsObserver(),
        "status_runtime": StatusObserver(),
        "stack_runtime": StackObserver(),
    }


def _frame(fname: str):
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    return rgba, w, h


# ── A：字段完整性 ─────────────────────────────────────────────
def test_field_completeness():
    obs = _observers()
    snap = build_snapshot(obs)
    for k in ("position", "pot", "board", "street", "hero_cards", "status",
              "stack", "bet", "action", "ground_truth_status"):
        assert k in snap, k
    assert snap["ground_truth_status"] == "runtime_state_non_gt"


# ── B：当前帧隔离（快照不污染）─────────────────────────────────
def test_current_frame_isolation():
    obs = _observers()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    snap1 = build_snapshot(obs)
    snap2 = build_snapshot(obs)
    assert snap1["stack"] == snap2["stack"]          # 同帧一致（无状态漂移）
    assert snap1["position"] == snap2["position"]


# ── C：UNKNOWN 传播 ────────────────────────────────────────────
def test_unknown_propagation_stack():
    obs = _observers()
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    snap = build_snapshot(obs)
    # 718,902 源（37 BL）经 Runtime = UNKNOWN → Snapshot 保留 UNKNOWN
    s4 = snap["stack"]["last_seats"].get("S4", {})
    assert s4.get("stack") is None or s4.get("status") == "UNKNOWN"
    # 不转 0/默认
    assert s4.get("stack") != 0


def test_bet_action_explicit_unknown():
    obs = _observers()
    snap = build_snapshot(obs)
    assert snap["bet"]["status"] == "UNKNOWN"
    assert snap["bet"]["reason"] == "evidence_limited"
    assert snap["action"]["status"] == "UNKNOWN"
    assert snap["action"]["reason"] == "not_ready"


def test_status_not_observed_preserved():
    obs = _observers()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    snap = build_snapshot(obs)
    st = snap["status"]
    assert st["last_status"] in ("NOT_OBSERVED", "RECOGNIZED", "UNKNOWN")


# ── D：零转换 ─────────────────────────────────────────────────
def test_zero_substitution():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/runtime_snapshot.py"
    ).read_text(encoding="utf-8")
    for bad in ("or 0", "previous", "default", "estimated", "inferred",
                "cache", "fixture"):
        assert bad not in src, bad


# ── E：718,902 isolation ──────────────────────────────────────
def test_718902_not_auto_called():
    """快照不调用 Foundation 校准（无 recognize_in_window/recognize_tight）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/runtime_snapshot.py"
    ).read_text(encoding="utf-8")
    assert "recognize_in_window" not in src
    assert "recognize_tight" not in src
    assert "718" not in src


# ── F：Observer isolation ─────────────────────────────────────
def test_observer_not_modified():
    obs = _observers()
    before = [id(o) for o in obs.values()]
    build_snapshot(obs)
    after = [id(o) for o in obs.values()]
    assert before == after


# ── G/H：Decision / Bet / Action isolation ────────────────────
def test_decision_isolation():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/runtime_snapshot.py"
    ).read_text(encoding="utf-8")
    # 仅检查代码体（函数体——排除 docstring 的"决策层"描述）
    body = src[src.index("def build_snapshot"):]
    for bad in ("recommend", "strategy", "policy", "expected_value",
                "decision", "betting"):
        assert bad not in body.lower(), bad


def test_capture_untouched():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "engine.process_frame", "--seconds", "--fps"):
        assert b in src, b
    assert "runtime_snapshot" in src
