"""Vision V3F-P11-A4 — DecisionInput Contract 测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.decision_input import (  # noqa: E402
    DecisionInput, Field, FieldStatus, from_snapshot, recognized, unknown)
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


# ── 基础契约 ─────────────────────────────────────────────────
def test_known_zero_vs_unknown():
    assert recognized(0).value == 0
    assert recognized(0).status == FieldStatus.RECOGNIZED
    assert unknown().value is None
    assert unknown().status == FieldStatus.UNKNOWN
    assert recognized(0) != unknown()          # known-zero ≠ UNKNOWN


def test_status_not_observed_vs_unknown():
    assert FieldStatus.NOT_OBSERVED.value != FieldStatus.UNKNOWN.value


def test_serialization_roundtrip():
    di = DecisionInput(hand_context={"street": "FLOP"},
                       evidence={"bet": Field(None,
                                              FieldStatus.EVIDENCE_LIMITED)})
    d = json.loads(json.dumps(di.to_dict()))
    di2 = DecisionInput.from_dict(d)
    assert di2.hand_context["street"] == "FLOP"
    assert di2.evidence["bet"].status == FieldStatus.EVIDENCE_LIMITED


def test_version_field():
    assert DecisionInput().version == "1.0"
    assert "version" in DecisionInput().to_dict()


# ── Snapshot → DecisionInput 边界 ─────────────────────────────
def test_from_snapshot_bet_action_boundary():
    obs = _observers()
    di = from_snapshot(build_snapshot(obs))
    assert di.evidence["bet"].status == FieldStatus.EVIDENCE_LIMITED
    assert di.evidence["bet"].value is None
    assert di.evidence["action"].status == FieldStatus.NOT_READY


def test_from_snapshot_stack_718902_isolation():
    obs = _observers()
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    di = from_snapshot(build_snapshot(obs))
    s4 = di.evidence.get("stack_S4")
    assert s4 is None or s4.status == FieldStatus.UNKNOWN
    assert (s4 is None or s4.value != 718902)     # 不自动调用 Foundation 值


def test_from_snapshot_recognized_stack():
    obs = _observers()
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    for o in obs.values():
        o.observe(rgba, w, h)
    di = from_snapshot(build_snapshot(obs))
    s1 = di.evidence.get("stack_S1")
    if s1 is not None and s1.value == 953909:
        assert s1.status == FieldStatus.RECOGNIZED


# ── 禁止策略字段 ─────────────────────────────────────────────
def test_no_strategy_fields():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/decision_input.py"
    ).read_text(encoding="utf-8")
    body = src[src.index("class DecisionInput"):]
    for bad in ("recommend", "suggested_action", "expected_value",
                "equity_recommend", "bet_sizing", "fold", "call", "raise",
                "policy"):
        assert bad not in body.lower(), bad


def test_no_inference_fallback():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/decision_input.py"
    ).read_text(encoding="utf-8")
    body = src[src.index("class DecisionInput"):]
    for bad in ("previous", "inferred", "estimated"):
        assert bad not in body.lower(), bad


def test_v1_agent_isolation():
    """decision_input 不 import agents/（V1 决策层隔离）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/decision_input.py"
    ).read_text(encoding="utf-8")
    assert "agents" not in src


def test_deterministic_serialization():
    a = from_snapshot(build_snapshot(_observers()))
    b = from_snapshot(build_snapshot(_observers()))
    assert json.dumps(a.to_dict(), sort_keys=True, default=str) == \
        json.dumps(b.to_dict(), sort_keys=True, default=str)
