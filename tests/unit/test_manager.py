"""Unit tests for the Manager Agent (总调度)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.manager.agent import ManagerAgent, ManagerOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return ManagerAgent()


@pytest.fixture
def ctx_with_registry(ctx):
    from poker_decision_ai.core.orchestration.registry import register_all_agents

    ctx.registry = register_all_agents()
    return ctx


async def test_schema_contract(agent):
    schema = json_schema_for(ManagerOutput)
    assert schema["type"] == "object"
    for f in ("final_decision", "base_decision", "deviated_decision",
              "deviation_applied", "agents_run", "status"):
        assert f in schema["properties"]
    validate_instance(ManagerOutput, ManagerOutput(status="ok", kind="cash_hand"))


async def test_not_applicable_without_kind(agent, ctx_with_registry):
    ok, reason = await agent.applicable({}, ctx_with_registry)
    assert not ok
    assert "kind" in reason


async def test_manager_runs_full_pipeline(agent, ctx_with_registry, cash_hand_text):
    env = await agent.run({
        "kind": "cash_hand",
        "hand_history_text": cash_hand_text,
        "hero_id": "Hero",
    }, ctx_with_registry)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["final_decision"] is not None
    assert out["base_decision"] is not None
    assert out["final_decision"]["action"] in ("fold", "check", "call", "bet", "raise", "allin")
    # manager runs the whole plan
    for name in ("state", "equity", "judge", "deviation", "evaluation"):
        assert name in out["agents_run"]
    # vision is in the cash plan but skipped (no image); tournament agents are
    # simply not part of the cash plan at all
    assert "vision" in out["agents_skipped"]
    assert "icm" not in out["agents_run"]
    # base vs deviated are permanently separated
    assert "deviated_decision" in out


async def test_unknown_kind_returns_error(agent, ctx_with_registry):
    env = await agent.run({"kind": "no_such_plan"}, ctx_with_registry)
    assert env.output["status"] == "error"
    assert env.output["final_decision"] is None
