"""Regression tests: identical inputs must not produce wildly different outputs.

Deterministic components (equity with fixed seed, ICM exact, judge fusion)
are compared against golden baselines with tight tolerances.
"""
from __future__ import annotations

from poker_decision_ai.core.schemas import new_id
from poker_decision_ai.domain.poker import EquityCalculator, icm, parse_range


def test_equity_golden_values():
    calc = EquityCalculator(iterations=20000, seed=42)
    # AA vs KK preflop MC with fixed seed must be stable across runs
    r1 = calc.hand_vs_hand(["As", "Ad"], [], ["Kh", "Kd"])
    r2 = calc.hand_vs_hand(["As", "Ad"], [], ["Kh", "Kd"])
    assert r1.hero_equity == r2.hero_equity
    assert 0.78 < r1.hero_equity < 0.86
    # river exact equity is deterministic to the last digit
    r3 = calc.hand_vs_hand(["As", "Ad"], ["Kc", "Kh", "2d", "2s"], ["Kd", "Ks"])
    assert r3.method == "exact"
    assert r3.hero_equity == 0.0


def test_icm_golden_values():
    ev = icm({"A": 90.0, "B": 10.0}, [100.0, 50.0]).ev
    assert abs(ev["A"] - 95.0) < 1e-9
    assert abs(ev["B"] - 55.0) < 1e-9
    ev3 = icm({"A": 100.0, "B": 100.0, "C": 100.0}, [100.0, 60.0, 40.0]).ev
    for v in ev3.values():
        assert abs(v - 200.0 / 3.0) < 1e-9


def test_range_golden_counts():
    assert len(parse_range("22+,AKs,AKo")) == 15
    assert len(parse_range("A5s-A2s")) == 4
    assert len(parse_range("22+")) == 13


async def test_judge_deterministic_on_same_input(runtime, cash_hand_text):
    """Same fixed hand -> same judge output (no hidden randomness)."""
    decisions = []
    for _ in range(2):
        ctx = runtime.new_context(new_id("rg"), "rg-session")
        report = await runtime.runner.run({
            "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
        }, ctx)
        decisions.append(report.base_decision)
    d1, d2 = decisions
    assert d1 is not None and d2 is not None
    assert d1["action"] == d2["action"]
    if d1.get("recommended_size") is not None:
        assert abs(d1["recommended_size"] - d2["recommended_size"]) < 1e-6
