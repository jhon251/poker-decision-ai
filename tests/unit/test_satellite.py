"""Unit tests for the Satellite Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.satellite.agent import SatelliteAgent, SatelliteOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return SatelliteAgent()


def _satellite_state(sample_tournament_state) -> dict:
    s = dict(sample_tournament_state)
    s["game_type"] = "satellite"
    s["tournament"] = dict(s["tournament"])
    s["tournament"]["satellite_seats"] = 1
    return s


async def test_schema_contract(agent):
    schema = json_schema_for(SatelliteOutput)
    assert schema["type"] == "object"
    for f in ("applicable", "seat_ev", "hero_seat_ev", "seats_remaining",
              "players_remaining", "satellite_mode", "notes"):
        assert f in schema["properties"]
    out = SatelliteOutput(applicable=True, hero_seat_ev=0.5, satellite_mode="seats")
    validate_instance(SatelliteOutput, out)


async def test_not_applicable_cash(agent, ctx, sample_cash_state):
    ok, _reason = await agent.applicable({"state": sample_cash_state}, ctx)
    assert not ok


async def test_satellite_seat_ev(agent, ctx, sample_tournament_state):
    state = _satellite_state(sample_tournament_state)
    ok, _reason = await agent.applicable({"state": state}, ctx)
    assert ok
    env = await agent.run({"state": state}, ctx)
    out = env.output
    assert out["applicable"] is True
    # seat probabilities sum to 1 over active players
    assert sum(out["seat_ev"].values()) == pytest.approx(1.0, abs=0.01)
    assert 0.0 < out["hero_seat_ev"] < 1.0
    assert out["satellite_mode"] == "seats"
    assert out["seats_remaining"] == 1
    assert any("Seat EV" in n for n in out["notes"])
