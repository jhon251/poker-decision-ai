"""Phase 3 (V2) Vision Provider 测试：逐字段置信度 + 低置信降级（不假装成功）。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from poker_decision_ai.agents.vision.agent import VisionAgent
from poker_decision_ai.core.providers.vision import (
    VISION_OBSERVATION_SCHEMA,
    VisionObservation,
    MockVisionProvider,
)


_IMG = "data:image/png;base64,iVBORw0KGgo="   # 1x1 png 占位（Mock 忽略内容）


def _agent_ctx_with_provider(ctx, provider):
    ctx.providers.vision = provider
    return ctx


class CountingProvider(MockVisionProvider):
    """记录 observe 调用次数；fixture 带低置信字段。"""

    def __init__(self, fixture: dict):
        super().__init__(fixture=fixture)
        self.calls = 0

    async def observe(self, image_bytes=None, *, hint=None):
        self.calls += 1
        return await super().observe(image_bytes, hint=hint)


def _good_fixture() -> dict:
    return {
        "source": "mock", "street": "flop",
        "hole_cards": ["As", "Kd"], "board": ["Qh", "7d", "2c"],
        "seats": [{"seat": 2, "name": "Bob", "stack": 31.25, "in_hand": True},
                  {"seat": 3, "name": "Hero", "stack": 24.38, "in_hand": True}],
        "pot": 3.35, "current_bet": 1.5, "to_call": 1.5,
        "blinds": {"sb": 0.1, "bb": 0.25}, "confidence": 0.9,
    }


async def test_schema_includes_field_confidences():
    assert "field_confidences" in VISION_OBSERVATION_SCHEMA["properties"]
    assert VISION_OBSERVATION_SCHEMA["properties"]["field_confidences"]["type"] == "object"


async def test_mock_fills_default_field_confidences(ctx):
    provider = MockVisionProvider(fixture=_good_fixture())
    obs = await provider.observe(None)
    assert obs.field_confidence("hole_cards") == 0.9
    assert obs.field_confidence("pot") == 0.9


async def test_high_confidence_no_warnings(ctx):
    agent = VisionAgent()
    env = await agent.run({"image": _IMG}, _agent_ctx_with_provider(ctx, MockVisionProvider(fixture=_good_fixture())))
    assert env.status in ("ok", "degraded")
    out = env.output
    assert out["low_confidence_fields"] == []
    assert out["manual_input_required"] is False
    assert out["observation"]["hole_cards"] == ["As", "Kd"]


async def test_low_confidence_triggers_retry_then_manual_flag(ctx):
    fixture = _good_fixture()
    fixture["field_confidences"] = {"hole_cards": 0.2, "pot": 0.9, "board": 0.9,
                                    "to_call": 0.9, "seats": 0.9}
    provider = CountingProvider(fixture)
    env = await VisionAgent().run({"image": _IMG}, _agent_ctx_with_provider(ctx, provider))
    out = env.output
    # 重试一次 + 初试 = 2 次调用；重试后仍低置信 → 建议人工输入
    assert provider.calls == 2
    assert out["low_confidence_fields"] == ["hole_cards"]
    assert out["manual_input_required"] is True
    codes = [w["code"] for w in env.warnings]
    assert "LOW_FIELD_CONFIDENCE_RETRY" in codes
    assert "LOW_FIELD_CONFIDENCE" in codes


async def test_retry_recovers_when_second_attempt_good(ctx):
    fixture = _good_fixture()
    fixture["field_confidences"] = {"hole_cards": 0.2, "pot": 0.9, "board": 0.9,
                                    "to_call": 0.9, "seats": 0.9}

    class RecoveringProvider(CountingProvider):
        async def observe(self, image_bytes=None, *, hint=None):
            self.calls += 1
            obs = await super().observe(image_bytes, hint=hint)
            if self.calls == 1:
                obs.field_confidences["hole_cards"] = 0.2
            else:
                obs.field_confidences["hole_cards"] = 0.95
            return obs

    env = await VisionAgent().run({"image": _IMG},
                                  _agent_ctx_with_provider(ctx, RecoveringProvider(fixture)))
    assert env.output["low_confidence_fields"] == []
    assert env.output["manual_input_required"] is False


async def test_passthrough_observation_keeps_field_confidences(ctx):
    vo = _good_fixture()
    vo["field_confidences"] = {"hole_cards": 0.99, "pot": 0.97}
    env = await VisionAgent().run({"request": {"vision_observation": vo}, "image": None}, ctx)
    out = env.output
    assert out["field_confidences"]["hole_cards"] == 0.99
    assert out["low_confidence_fields"] == []
