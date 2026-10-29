"""Phase 1 (V2) Provider Architecture 测试：诚实能力状态。"""
from __future__ import annotations

import pytest

from poker_decision_ai.core.providers import (
    AnthropicLLMProvider,
    MockLLMProvider,
    MockSolverProvider,
    MockVisionProvider,
    MultimodalLLMVisionProvider,
    OpenAICompatibleLLMProvider,
    PostflopSolverCLIProvider,
    ProviderStatus,
    build_providers,
)


def test_mock_providers_report_mock():
    assert MockLLMProvider().status() == ProviderStatus.MOCK
    assert MockVisionProvider().status() == ProviderStatus.MOCK
    assert MockSolverProvider().status() == ProviderStatus.MOCK
    info = MockLLMProvider().status_info()
    assert info["status"] == "mock" and info["is_mock"] is True
    assert "mock" in info["reason"]


def test_real_llm_unconfigured_is_unavailable():
    p = OpenAICompatibleLLMProvider(api_key="", base_url="https://api.moonshot.cn/v1")
    assert p.status() == ProviderStatus.UNAVAILABLE
    assert p.is_configured() is False
    assert "API key" in p.status_reason()

    c = AnthropicLLMProvider(api_key="")
    assert c.status() == ProviderStatus.UNAVAILABLE
    assert "ANTHROPIC" in c.status_reason()


def test_real_llm_configured_is_real():
    p = OpenAICompatibleLLMProvider(api_key="sk-test-123")
    assert p.status() == ProviderStatus.REAL
    assert p.status_reason() == ""


def test_solver_unavailable_honest():
    p = PostflopSolverCLIProvider("")
    assert p.status() == ProviderStatus.UNAVAILABLE
    assert "path not set" in p.status_reason()
    info = p.status_info()
    assert info["configured"] is False


def test_multimodal_vision_status():
    v = MultimodalLLMVisionProvider(OpenAICompatibleLLMProvider(api_key=""))
    assert v.status() == ProviderStatus.UNAVAILABLE
    assert "API key" in v.status_reason()
    v2 = MultimodalLLMVisionProvider(OpenAICompatibleLLMProvider(api_key="sk-1"))
    assert v2.status() == ProviderStatus.REAL


def test_registry_status_summary(cfg):
    providers = build_providers(cfg)
    summary = providers.status_summary()
    assert summary["llm"]["status"] == "mock"
    assert summary["vision"]["status"] == "mock"
    assert summary["solver"]["status"] == "mock"
    assert "MOCK" in providers.capabilities_text()


def test_pipeline_report_includes_provider_status(runtime, cash_hand_text):
    import asyncio

    from poker_decision_ai.core.schemas import new_id

    ctx = runtime.new_context(new_id("p1"), "p1")
    report = asyncio.run(runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx))
    assert report.provider_status is not None
    assert report.provider_status["llm"]["status"] == "mock"
    assert report.provider_status["solver"]["status"] == "mock"
    # 决策记录也带 provider_status
    rec = runtime.repos.decisions.get(ctx.request_id)
    assert rec["provider_status"]["solver"]["status"] == "mock"
