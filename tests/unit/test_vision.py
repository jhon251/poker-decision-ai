"""Unit tests for the Vision Agent."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.vision.agent import VisionAgent, VisionOutput
from poker_decision_ai.core.errors import ProviderError
from poker_decision_ai.core.providers import MockVisionProvider
from poker_decision_ai.core.schemas import json_schema_for, validate_instance


@pytest.fixture
def agent():
    return VisionAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(VisionOutput)
    assert schema["type"] == "object"
    assert "observation" in schema["properties"]
    assert "provider" in schema["properties"]
    assert "confidence" in schema["properties"]
    out = VisionOutput(observation={}, provider="mock", confidence=1.0)
    validate_instance(VisionOutput, out)


async def test_not_applicable_without_input(agent, ctx):
    ok, reason = await agent.applicable({}, ctx)
    assert not ok
    assert "no image" in reason


async def test_applicable_with_observation(agent, ctx):
    ok, _ = await agent.applicable({"request": {"vision_observation": {"source": "x"}}}, ctx)
    assert ok


async def test_applicable_with_image(agent, ctx):
    ok, _ = await agent.applicable({"request": {}, "image": "data:image/png;base64,AAAA"}, ctx)
    assert ok


async def test_direct_pass_through(agent, ctx):
    env = await agent.run({"request": {"vision_observation": {
        "source": "mock", "street": "flop", "hole_cards": ["As", "Kd"],
        "board": ["Qh", "7d", "2c"], "confidence": 0.9,
    }}, "image": None}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["observation"]["street"] == "flop"
    assert out["observation"]["hole_cards"] == ["As", "Kd"]
    assert out["provider"] == "mock"
    assert out["confidence"] == 0.9


async def test_image_path_mock_provider(agent, ctx):
    ctx.providers.vision = MockVisionProvider(fixture={
        "source": "mock", "street": "flop", "hole_cards": ["As", "Kd"],
        "board": ["Qh", "7d", "2c"], "pot": 5.95, "confidence": 0.8,
    })
    env = await agent.run({"request": {}, "image": "data:image/png;base64,AAAA"}, ctx)
    out = env.output
    assert out["observation"] is not None
    assert out["observation"]["street"] == "flop"
    assert out["provider"] == "mock"


async def test_no_vision_provider_degraded(agent, ctx):
    ctx.providers.vision = None
    env = await agent.run({"request": {}, "image": "data:image/png;base64,AAAA"}, ctx)
    out = env.output
    assert out["observation"] is None
    assert env.status == "degraded"
    assert any(w["code"] == "NO_VISION_PROVIDER" for w in out["warnings"])


async def test_provider_error_degraded(agent, ctx):
    class FailingProvider:
        name = "mock"

        async def observe(self, image_bytes=None, *, hint=None):
            raise ProviderError("boom")

    ctx.providers.vision = FailingProvider()
    env = await agent.run({"request": {}, "image": "data:image/png;base64,AAAA"}, ctx)
    out = env.output
    assert out["observation"] is None
    assert env.status == "degraded"
    assert any(w["code"] == "PROVIDER_ERROR" for w in out["warnings"])
