"""Phase 2 (V2) LLM Provider 测试：具名 Provider + Fast/Deep 双模型。"""
from __future__ import annotations

import pytest

from poker_decision_ai.core.providers import (
    AnthropicLLMProvider,
    ClaudeProvider,
    DeepSeekProvider,
    GeminiProvider,
    KimiProvider,
    MockLLMProvider,
    OpenAIProvider,
    ProviderStatus,
)


def test_kimi_provider_defaults():
    p = KimiProvider(api_key="sk-k")
    assert p.name == "kimi"
    assert p.base_url == "https://api.moonshot.cn/v1"
    assert p.model_for("fast") == "moonshot-v1-8k"
    assert p.model_for("deep") == "moonshot-v1-32k"
    assert p.status() == ProviderStatus.REAL


def test_openai_provider_defaults():
    p = OpenAIProvider(api_key="sk-o")
    assert p.base_url == "https://api.openai.com/v1"
    assert p.model_for("deep") == "gpt-4o"
    assert p.supports_images() is True


def test_gemini_provider_defaults():
    p = GeminiProvider(api_key="sk-g")
    assert "generativelanguage.googleapis.com" in p.base_url
    assert p.model_for("fast") == "gemini-2.0-flash"


def test_deepseek_provider_defaults():
    p = DeepSeekProvider(api_key="sk-d")
    assert p.name == "deepseek"
    assert p.base_url == "https://api.deepseek.com/v1"
    assert p.model_for("fast") == "deepseek-chat"
    assert p.model_for("deep") == "deepseek-reasoner"
    assert p.status() == ProviderStatus.REAL
    # 空 base_url 时使用 DeepSeek 默认端点（不回落 OpenAI 官方）
    assert DeepSeekProvider(api_key="sk-d", base_url="").base_url == "https://api.deepseek.com/v1"


def test_claude_provider_is_anthropic_channel():
    p = ClaudeProvider(api_key="sk-c")
    assert isinstance(p, AnthropicLLMProvider)
    assert p.name == "claude"
    assert p.status() == ProviderStatus.REAL


def test_fast_deep_fallback_when_deep_unset():
    p = OpenAIProvider(api_key="sk-o", deep_model="")
    assert p.model_for("deep") == p.model_for("fast") == "gpt-4o-mini"
    p2 = OpenAIProvider(api_key="sk-o", deep_model="o1")
    assert p2.model_for("deep") == "o1"


def test_mock_provider_tier_neutral():
    import asyncio

    p = MockLLMProvider()
    r = asyncio.run(p.complete([{"role": "user", "content": "hi"}], tier="deep"))
    assert r.model == "mock"


def test_unconfigured_named_providers_unavailable():
    for p in (KimiProvider(api_key=""), OpenAIProvider(api_key=""),
              GeminiProvider(api_key=""), ClaudeProvider(api_key="")):
        assert p.status() == ProviderStatus.UNAVAILABLE
        assert "key" in p.status_reason().lower() or "KEY" in p.status_reason()


def test_build_providers_from_config_names(tmp_path, monkeypatch):
    import yaml

    from poker_decision_ai.core.config import AppConfig, NestedDict
    from poker_decision_ai.core.providers import build_providers

    def cfg_with(llm_provider):
        data = {
            "app": {"logs_dir": str(tmp_path / "logs"), "data_dir": str(tmp_path / "data")},
            "llm": {"enabled": True, "provider": llm_provider, "api_key": "sk-x",
                    "model": "m-fast", "deep_model": "m-deep"},
            "vision": {"provider": "mock"}, "solver": {"provider": "mock"},
        }
        return AppConfig(root=NestedDict(data), stakes=NestedDict({}),
                         ranges=NestedDict({}), pipeline=NestedDict({}))

    providers = build_providers(cfg_with("kimi"))
    assert providers.get_llm().name == "kimi"
    assert providers.get_llm().model_for("deep") == "m-deep"
    assert providers.get_llm().status() == ProviderStatus.REAL
    assert build_providers(cfg_with("gemini")).get_llm().name == "gemini"
    assert build_providers(cfg_with("claude")).get_llm().name == "claude"


def test_context_llm_json_tier(ctx):
    """ctx.llm_json 支持 tier 参数（mock 下 deep 也走 mock，不抛）。"""
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class Simple:
        note: str

    r = asyncio.run(ctx.llm_json(system="s", user="u", schema_cls=Simple, tier="deep"))
    assert r is not None and "note" in r[0]
