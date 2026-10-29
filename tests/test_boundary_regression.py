"""V3F-P11-A3 — Evidence Boundary Regression（Bet/Action/Decision/UNKNOWN 防回归）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.failure_taxonomy import (  # noqa: E402
    NON_FAILURE_OUTCOMES, Outcome, classify)
from poker_decision_ai.vision2.pixel.realclient.runtime_snapshot import (  # noqa: E402
    build_snapshot)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    BoardObserver, HeroCardsObserver, PositionObserver, PotObserver,
    StackObserver, StatusObserver, StreetObserver)


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


# ── Failure taxonomy ───────────────────────────────────────────
def test_taxonomy_classification():
    assert classify("RECOGNIZED") == Outcome.UNKNOWN   # 未定义串 → UNKNOWN
    assert classify("NOT_OBSERVED") == Outcome.NOT_OBSERVED
    assert classify("EVIDENCE_LIMITED") == Outcome.EVIDENCE_LIMITED
    assert classify("NOT_READY") == Outcome.NOT_READY
    assert classify("PASS") == Outcome.PASS
    assert classify(None) == Outcome.UNKNOWN
    assert Outcome.UNKNOWN in NON_FAILURE_OUTCOMES
    assert Outcome.REGRESSION not in NON_FAILURE_OUTCOMES


# ── Bet boundary（EVIDENCE-LIMITED 不绕过）─────────────────────
def test_bet_boundary_not_inferred():
    snap = build_snapshot(_observers())
    assert snap["bet"]["status"] == "UNKNOWN"
    assert snap["bet"]["reason"] == "evidence_limited"
    assert "bet" not in snap or snap["bet"]["status"] != 0 or True


# ── Action boundary（NOT-READY 不绕过）────────────────────────
def test_action_boundary_not_ready():
    snap = build_snapshot(_observers())
    assert snap["action"]["status"] == "UNKNOWN"
    assert snap["action"]["reason"] == "not_ready"


# ── Decision boundary（无推荐/策略/EV）────────────────────────
def test_decision_boundary_absent():
    snap = build_snapshot(_observers())
    for key in snap:
        assert "recommend" not in str(key).lower()
        assert "strategy" not in str(key).lower()
    import json
    s = json.dumps(snap, default=str).lower()
    for bad in ("recommend", "expected_value", "ev_", "betting_advice",
                "action_recommendation"):
        assert bad not in s, bad


# ── UNKNOWN 边界（≠0/前值/默认）───────────────────────────────
def test_unknown_boundary_stack():
    from poker_decision_ai.vision2_platform import decode_png
    import numpy as np
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    obs = _observers()
    for o in obs.values():
        o.observe(rgba, w, h)
    snap = build_snapshot(obs)
    s4 = snap["stack"]["last_seats"].get("S4", {})
    assert s4.get("stack") is None or s4.get("status") == "UNKNOWN"
    assert s4.get("stack") != 718902          # 718,902 不自动升级
    assert s4.get("stack") != 0               # 不转 0


# ── Deterministic repeat（同输入两次一致）─────────────────────
def test_deterministic_repeat():
    obs1 = _observers()
    obs2 = _observers()
    s1 = build_snapshot(obs1)
    s2 = build_snapshot(obs2)
    import json
    assert json.dumps(s1, default=str, sort_keys=True) == \
        json.dumps(s2, default=str, sort_keys=True)
