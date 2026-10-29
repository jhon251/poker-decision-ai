"""Vision V3F-P11-A2 — State/Replay/UNKNOWN Integration 测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.state_bridge import (  # noqa: E402
    snapshot_to_replay_state, validate_replay_state)
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


# ── Snapshot → State → Replay round-trip ────────────────────────
def test_snapshot_to_state_to_json_roundtrip():
    obs = _observers()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    snap = build_snapshot(obs)
    state = snapshot_to_replay_state(snap)
    s = json.dumps(state, default=str)
    state2 = json.loads(s)
    assert set(state2) >= {"position", "pot", "board", "street",
                           "hero_cards", "status", "stack", "bet", "action"}


# ── UNKNOWN preservation ───────────────────────────────────────
def test_bet_action_unknown_preserved():
    state = snapshot_to_replay_state(build_snapshot(_observers()))
    assert state["bet"]["status"] == "UNKNOWN"
    assert state["bet"]["reason"] == "evidence_limited"
    assert state["action"]["status"] == "UNKNOWN"
    assert state["action"]["reason"] == "not_ready"
    assert state["bet"]["status"] != 0               # 不转 0
    assert state["action"]["status"] != "CHECK"      # 不默认


def test_stack_unknown_preserved_718902():
    obs = _observers()
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    state = snapshot_to_replay_state(build_snapshot(obs))
    # 718,902 源（37 BL）Runtime=UNKNOWN → State 保留 UNKNOWN（不自动升级）
    s4 = state["stack"]["last_seats"].get("S4", {})
    assert s4.get("stack") is None or s4.get("status") == "UNKNOWN"
    assert s4.get("stack") != 718902                  # 不自动调用 Foundation 值


def test_validate_notes():
    state = snapshot_to_replay_state(build_snapshot(_observers()))
    notes = validate_replay_state(state)
    assert any("bet" in n for n in notes)
    assert any("action" in n for n in notes)


def test_no_conversion_inference():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/state_bridge.py"
    ).read_text(encoding="utf-8")
    for bad in ("or 0", "previous", "default", "inferred", "estimated",
                "recommend", "strategy", "equity", "decision"):
        assert bad not in src.lower(), bad


# ── 不完整状态被正确保存（不偷偷补值）─────────────────────────
def test_incomplete_state_preserved_not_filled():
    state = snapshot_to_replay_state(build_snapshot(_observers()))
    # bet/action 保持 UNKNOWN（不补值）
    assert "bet" in state and state["bet"]["status"] == "UNKNOWN"
    assert "action" in state and state["action"]["status"] == "UNKNOWN"
    # 序列化不丢失 UNKNOWN 标记
    s = json.loads(json.dumps(state, default=str))
    assert s["bet"]["status"] == "UNKNOWN"
    assert s["action"]["status"] == "UNKNOWN"
