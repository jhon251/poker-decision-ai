"""Vision V3F-P12-OFFLINE — Decision Foundation Components 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.decision_foundation import (  # noqa: E402
    ActionType, ComponentStatus, Result, equity, legal_actions, pot_odds,
    range_size, spr)


# ── Legal Action ───────────────────────────────────────────────
def test_legal_actions_check_zero():
    r = legal_actions(0)
    assert r.status == ComponentStatus.OK
    assert ActionType.CHECK.value in r.value
    assert ActionType.FOLD.value in r.value


def test_legal_actions_call():
    r = legal_actions(10)
    assert ActionType.CALL.value in r.value
    assert ActionType.CHECK.value not in r.value


def test_legal_actions_unknown_not_inferred():
    r = legal_actions(None)
    assert r.status == ComponentStatus.UNKNOWN        # 不推断（Bet 封存）
    assert r.value is None


def test_legal_actions_invalid():
    assert legal_actions(-5).status == ComponentStatus.INVALID_INPUT


# ── Pot Odds ──────────────────────────────────────────────────
def test_pot_odds_known():
    r = pot_odds(100, 50)
    assert r.status == ComponentStatus.OK
    assert r.value == pytest.approx(50 / 150)


def test_pot_odds_known_zero_call():
    assert pot_odds(100, 0).value == 0.0            # known-zero ≠ UNKNOWN


def test_pot_odds_unknown():
    r = pot_odds(None, 50)
    assert r.status == ComponentStatus.UNKNOWN
    assert r.value is None


def test_pot_odds_invalid():
    assert pot_odds(100, -1).status == ComponentStatus.INVALID_INPUT


# ── SPR ───────────────────────────────────────────────────────
def test_spr_known():
    r = spr(200, 50)
    assert r.status == ComponentStatus.OK
    assert r.value == pytest.approx(4.0)


def test_spr_unknown_and_not_zero():
    r = spr(None, 50)
    assert r.status == ComponentStatus.UNKNOWN
    assert r.value is None
    r2 = spr(None, 50)
    assert r2.value != 0                            # UNKNOWN ≠ 0


def test_spr_invalid_pot_zero():
    assert spr(100, 0).status == ComponentStatus.INVALID_INPUT


# ── Equity Adapter ───────────────────────────────────────────
def test_equity_adapter_deterministic():
    r1 = equity(["As", "Ks"], ["9h", "7s", "3c"], ["2c", "2d"])
    r2 = equity(["As", "Ks"], ["9h", "7s", "3c"], ["2c", "2d"])
    assert r1.status == ComponentStatus.OK
    assert r1.value == r2.value
    assert 0.0 <= r1.value["hero"] <= 1.0


def test_equity_missing_input():
    r = equity(None, ["9h"], ["2c", "2d"])
    assert r.status == ComponentStatus.UNKNOWN


# ── Range Adapter ────────────────────────────────────────────
def test_range_size():
    assert range_size([1, 2, 3]).value == 3
    assert range_size(None).status == ComponentStatus.UNKNOWN


# ── 语义 / 边界 ──────────────────────────────────────────────
def test_status_semantics():
    assert ComponentStatus.UNKNOWN.value != ComponentStatus.INVALID_INPUT.value
    assert ComponentStatus.OK.value != ComponentStatus.UNKNOWN.value


def test_no_strategy_no_v1():
    src = Path(
        "poker_decision_ai/decision_foundation.py"
    ).read_text(encoding="utf-8")
    body = src[src.index("def legal_actions"):]
    for bad in ("recommend", "best", "optimal", "expected_value", "agents",
                "exploit", "bluff"):
        assert bad not in body.lower(), bad


def test_deterministic_repeat():
    assert pot_odds(100, 50).value == pot_odds(100, 50).value
    assert spr(200, 50).value == spr(200, 50).value
