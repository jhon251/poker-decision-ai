"""AgentContext: everything an agent may touch, injected - never global.

Agents access config, providers, repositories, logging and metrics through
this context only.  This keeps agents testable in isolation and lets the
Manager swap implementations freely (memory repo in tests, JSON repo live).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..logging import EventLogger, MetricsCollector
from ..providers import LLMProvider, ProviderRegistry, complete_json
from ..repositories.interfaces import RepositorySet


@dataclass
class AgentContext:
    request_id: str
    session_id: str
    config: AppConfig
    providers: ProviderRegistry
    repos: RepositorySet
    logger: EventLogger
    metrics: MetricsCollector
    allow_llm: bool = True
    registry: Any = None          # AgentRegistry (set by Runtime; used by Manager)
    game_context: Any = None      # GameContext (V1.1 模式上下文；Runner 填充)
    caches: Any = None            # CacheManager (V2：模型/计算缓存)
    extra: dict = field(default_factory=dict)

    def llm(self) -> LLMProvider | None:
        """Return the LLM provider when LLM use is enabled and configured."""
        if not self.allow_llm or not self.config.llm_enabled():
            return None
        p = self.providers.get_llm()
        if p is None:
            return None
        try:
            if not p.is_configured() and p.name != "mock":
                return None
        except Exception:
            return None
        return p

    async def llm_json(self, *, system: str, user: str, schema_cls: type,
                       temperature: float | None = None,
                       tier: str | None = None) -> tuple[dict, Any] | None:
        """Convenience: LLM completion validated against a dataclass schema.

        Returns None when no LLM is available; raises ProviderError when the
        provider exists but fails - callers then fall back to deterministic
        logic and add a warning.
        tier: "fast" | "deep"（V2 双模型路径）；None 时由上下文决定：
        Deep 路径（ctx.extra["deep"]）自动使用 deep 模型。
        """
        if tier is None:
            deep_ctx = self.extra.get("deep") if isinstance(self.extra, dict) else None
            tier = "deep" if (deep_ctx or {}).get("llm_tier") else "fast"
        provider = self.llm()
        if provider is None:
            return None
        result = await complete_json(provider, system=system, user=user, schema_cls=schema_cls,
                                     temperature=temperature, tier=tier)
        if result:
            data, usage = result
            self.metrics.record_llm(schema_cls.__name__, usage.prompt_tokens, usage.completion_tokens, usage.cost_usd)
            self.logger.log_llm_usage(schema_cls.__name__, provider.model_for(tier),
                                      usage.prompt_tokens, usage.completion_tokens, usage.cost_usd,
                                      request_id=self.request_id, session_id=self.session_id)
            return data, usage
        return None

    def child(self, **overrides: Any) -> "AgentContext":
        """Derive a context with overrides (used by Manager for sub-requests)."""
        base = {
            "request_id": self.request_id, "session_id": self.session_id, "config": self.config,
            "providers": self.providers, "repos": self.repos, "logger": self.logger,
            "metrics": self.metrics, "allow_llm": self.allow_llm, "registry": self.registry,
            "game_context": self.game_context, "caches": self.caches,
            "extra": dict(self.extra),
        }
        base.update(overrides)
        return AgentContext(**base)
