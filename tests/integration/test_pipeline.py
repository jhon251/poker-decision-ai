"""Integration tests: full pipeline (perception -> analysis -> judge -> deviation).

Tests the V1 chain end-to-end in fully offline mode (mock providers,
in-memory repos).  These are also the smoke tests for the Manager's dynamic
orchestration: cash hands must skip tournament agents, tournament hands must
run them, and degraded inputs must never crash the pipeline.
"""
from __future__ import annotations

from poker_decision_ai.core.schemas import new_id


async def test_cash_hand_full_pipeline(runtime, cash_hand_text):
    ctx = runtime.new_context(new_id("it"), "it-session")
    report = await runtime.runner.run({
        "kind": "cash_hand",
        "hand_history_text": cash_hand_text,
        "hero_id": "Hero",
    }, ctx)
    # pipeline must complete (mock solver degrades into warnings, not failures)
    assert report.status in ("ok", "degraded")
    # core agents must all have produced envelopes
    for name in ("state", "board", "statistics", "population", "opponent",
                 "confidence", "range", "equity", "solver", "exploit", "bluff",
                 "judge", "deviation", "evaluation", "learning"):
        assert name in report.envelopes, f"agent '{name}' missing"
    # tournament agents must be skipped on a cash hand
    for name in ("icm", "tournament_stage", "satellite", "icm_exploit"):
        assert name in report.skipped or name not in report.envelopes, f"{name} should be skipped on cash"
    # final decision must be structurally valid
    final = report.final_decision
    assert final and final.get("action") in ("fold", "check", "call", "bet", "raise", "allin")
    assert report.base_decision is not None
    # state normalization sanity
    state = report.envelopes["state"]["output"]
    assert state["game_type"] == "cash"
    assert state["board"] == ["Qh", "7d", "2c"]
    assert state["hero_id"] == "Hero"
    assert state["to_call"] == 1.50
    assert state["pot"] == 3.35
    assert state["cash"]["stakes"] == "NL10"
    hero_p = next(p for p in state["players"] if p["is_hero"])
    assert hero_p["position"] == "BTN"   # 按钮 seat 3 → Hero 必须得到 BTN（轮转修复）


async def test_tournament_hand_full_pipeline(runtime, tournament_hand_text):
    ctx = runtime.new_context(new_id("it"), "it-session")
    report = await runtime.runner.run({
        "kind": "tournament_hand",
        "hand_history_text": tournament_hand_text,
        "hero_id": "Hero",
    }, ctx)
    assert report.status in ("ok", "degraded")
    # tournament agents must run (or degrade gracefully) on an MTT hand
    assert "icm" in report.envelopes
    assert "tournament_stage" in report.envelopes
    state = report.envelopes["state"]["output"]
    assert state["game_type"] == "mtt"
    assert state["tournament"]["players_remaining"] == 45
    # satellite agent must skip (this is an MTT, not a satellite)
    assert "satellite" in report.skipped or report.envelopes.get("satellite") is None
    final = report.final_decision
    assert final and final.get("action") in ("fold", "check", "call", "bet", "raise", "allin")


async def test_decision_persisted(runtime, cash_hand_text):
    ctx = runtime.new_context(new_id("it"), "it-session")
    await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx)
    rec = runtime.repos.decisions.get(ctx.request_id)
    assert rec is not None
    assert "base_decision" in rec and "final_decision" in rec


async def test_manager_agent_via_registry(runtime, cash_hand_text):
    """The Manager is itself an agent and must behave like one."""
    manager = runtime.registry.get("manager")
    ctx = runtime.new_context(new_id("it"), "it-session")
    env = await manager.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx)
    assert env.status in ("ok", "degraded")
    assert env.output["final_decision"] is not None
    assert env.output["base_decision"] is not None


async def test_vision_observation_passthrough(runtime):
    ctx = runtime.new_context(new_id("it"), "it-session")
    report = await runtime.runner.run({
        "kind": "cash_hand",
        "vision_observation": {
            "source": "manual", "hero_seat": 3, "button_seat": 2, "street": "flop",
            "hole_cards": ["As", "Kd"], "board": ["Qh", "7d", "2c"],
            "seats": [
                {"seat": 1, "name": "Alice", "stack": 10.0, "in_hand": False},
                {"seat": 2, "name": "Bob", "stack": 12.5, "in_hand": True},
                {"seat": 3, "name": "Hero", "stack": 9.75, "in_hand": True},
                {"seat": 4, "name": "Carol", "stack": 8.2, "in_hand": False},
                {"seat": 5, "name": "Dave", "stack": 11.3, "in_hand": False},
                {"seat": 6, "name": "Eve", "stack": 10.0, "in_hand": False},
            ],
            "pot": 4.05, "current_bet": 1.50, "to_call": 1.50,
            "blinds": {"sb": 0.05, "bb": 0.10}, "ante": 0.0,
            "available_actions": ["fold", "call", "raise"],
            "action_text": "Bob bets $1.50", "confidence": 0.85, "warnings": [],
            "extra": {"stakes": "NL10"},
        },
        "hero_id": "Hero",
    }, ctx)
    assert report.status in ("ok", "degraded")
    state = report.envelopes["state"]["output"]
    assert state["board"] == ["Qh", "7d", "2c"]
    assert state["to_call"] == 1.50
