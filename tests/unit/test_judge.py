"""Unit tests for the Judge Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.judge.agent import JudgeAgent, JudgeOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return JudgeAgent()


def _state(to_call=1.0, pot=5.0, bb=0.1):
    return {"to_call": to_call, "pot": pot, "bb": bb,
            "players": [{"player_id": "hero", "is_hero": True, "stack": 10.0}]}


async def test_schema_contract(agent):
    schema = json_schema_for(JudgeOutput)
    assert schema["type"] == "object"
    for field in ("action", "confidence", "reasoning", "key_evidence",
                  "uncertainty", "alternative_action", "weights_used",
                  "final_inputs_summary"):
        assert field in schema["properties"]
    out = JudgeOutput()
    validate_instance(JudgeOutput, out)


async def test_applicable_always(agent, ctx):
    ok, _ = await agent.applicable({}, ctx)
    assert ok


async def test_strong_equity_calls_or_raises(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "equity": {"hero_equity": 0.8, "pot_odds": {"required_equity": 0.3}},
    }, ctx)
    out = env.output
    assert out["action"] in ("call", "raise")
    assert out["confidence"] > 0.5


async def test_weak_equity_folds(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "equity": {"hero_equity": 0.05, "pot_odds": {"required_equity": 0.4}},
    }, ctx)
    out = env.output
    assert out["action"] == "fold"


async def test_exploit_overrides_solver_fold(agent, ctx):
    env = await agent.run({
        "state": _state(),
        "equity": {"hero_equity": 0.6, "pot_odds": {"required_equity": 0.3}},
        "solver": {"gto_action": "fold", "is_approximate": True, "confidence": 0.35},
        "exploit": {"best_exploit": {"suggested_action": "raise", "confidence": 0.9}},
    }, ctx)
    out = env.output
    assert out["action"] in ("raise", "call")
    assert out["action"] != "fold"


async def test_all_none_conservative(agent, ctx):
    inp = {k: None for k in ("state", "board", "range", "equity", "solver",
                             "exploit", "opponent", "population", "human_style",
                             "confidence", "bluff", "image", "emotion",
                             "strategy_evolution", "risk", "icm", "tournament_stage",
                             "satellite", "icm_exploit", "blind_pressure", "debate")}
    env = await agent.run(inp, ctx)
    out = env.output
    assert out["action"] in ("fold", "check", "call", "bet", "raise", "allin")
    assert out["confidence"] <= 0.5
    assert out["uncertainty"]


async def test_deterministic(agent, ctx):
    inp = {
        "state": _state(),
        "equity": {"hero_equity": 0.6, "pot_odds": {"required_equity": 0.3}},
        "solver": {"gto_action": "fold", "is_approximate": True},
        "exploit": {"best_exploit": {"suggested_action": "raise", "confidence": 0.9}},
    }
    o1 = (await agent.run(inp, ctx)).output
    o2 = (await agent.run(inp, ctx)).output
    assert o1 == o2


async def test_size_synthesized_for_bet(agent, ctx):
    env = await agent.run({
        "state": _state(to_call=0.0),
        "equity": {"hero_equity": 0.75, "pot_odds": {"required_equity": 0.0}},
    }, ctx)
    out = env.output
    assert out["action"] in ("bet", "check", "raise")
    if out["action"] == "bet":
        assert out["recommended_size"] is not None
