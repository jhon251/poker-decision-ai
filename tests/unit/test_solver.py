"""Unit tests for the Solver Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.solver.agent import (
    SolverAgent,
    SolverOutput,
    _SafeMockSolverProvider,
)
from poker_decision_ai.core.errors import ProviderNotConfiguredError
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return SolverAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(SolverOutput)
    assert schema["type"] == "object"
    assert "actions" in schema["properties"]
    assert "gto_action" in schema["properties"]
    validate_instance(SolverOutput, SolverOutput())


async def test_applicable(agent, ctx):
    ok, _reason = await agent.applicable({}, ctx)
    assert not ok
    ok2, _reason2 = await agent.applicable({"state": {"players": []}}, ctx)
    assert ok2


async def test_mock_baseline(agent, ctx, sample_cash_state):
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["is_approximate"] is True
    assert out["confidence"] == pytest.approx(0.35)
    assert out["actions"]
    assert sum(a["frequency"] for a in out["actions"]) == pytest.approx(1.0)
    assert out["gto_action"] in ("call", "raise", "fold", "check", "bet")


async def test_query_fields_complete(agent, ctx, sample_cash_state):
    class CapturingSolver(_SafeMockSolverProvider):
        def __init__(self):
            super().__init__(seed=42)
            self.query = None

        async def solve(self, query):
            self.query = query
            return await super().solve(query)

    cap = CapturingSolver()
    ctx.providers.solver = cap
    await agent.run({"state": sample_cash_state, "equity": {"hero_equity": 0.6}}, ctx)

    q = cap.query
    assert q is not None
    for k in ("board", "hole_cards", "pot", "effective_stack", "street",
              "to_call", "hero_equity", "bet_sizings"):
        assert k in q
    assert q["hole_cards"] == ["As", "Kd"]
    assert q["bet_sizings"] == [50, 75]
    assert q["hero_equity"] == 0.6


async def test_degrade_on_provider_error(agent, ctx, sample_cash_state):
    class BoomSolver(_SafeMockSolverProvider):
        async def solve(self, query):
            raise ProviderNotConfiguredError("postflop solver not configured")

    ctx.providers.solver = BoomSolver()
    env = await agent.run({"state": sample_cash_state}, ctx)
    assert env.output["is_approximate"] is True
    assert any(w.get("code") == "SOLVER_DEGRADED" for w in env.output.get("warnings", []))


async def test_determinism(agent, ctx, sample_cash_state):
    a = await agent.run({"state": sample_cash_state}, ctx)
    b = await agent.run({"state": sample_cash_state}, ctx)
    assert a.output == b.output
