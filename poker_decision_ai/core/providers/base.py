"""Provider base classes and registry factory.

Design rule (spec §18 - 不要伪造能力): every external capability has
    Provider Interface  +  Mock Provider  +  Real Provider Adapter.
If a real provider is not configured, selecting it raises
ProviderNotConfiguredError and the pipeline degrades to mock - it never
silently pretends the real thing ran.

V2（§三十二）：每个 Provider 都必须诚实报告能力状态 provider_status：
    MOCK          - 当前是 Mock 实现（不是真实能力）
    REAL          - 真实能力已配置可用
    UNAVAILABLE   - 真实适配器存在但不可用（缺 Key/路径/模型），带 reason
状态会写入每次 PipelineReport 与决策记录，任何地方都不会把 Mock 当真实。
"""
from __future__ import annotations

import asyncio
import enum
from abc import ABC, abstractmethod
from typing import Any

from ..config import AppConfig
from ..errors import ConfigError


class ProviderStatus(str, enum.Enum):
    """诚实能力状态（V2）。"""

    MOCK = "mock"                # Mock 实现（非真实能力）
    REAL = "real"                # 真实能力已配置并可用
    UNAVAILABLE = "unavailable"  # 真实适配器存在但当前不可用（未配置/配置错误）


class Provider(ABC):
    """Base provider. Every provider has a stable name used in config."""

    name: str = "base"
    #: Mock 实现必须显式声明 is_mock=True；真实适配器保持 False
    is_mock: bool = False

    def is_configured(self) -> bool:
        return True

    # ---- V2 诚实能力状态 ----
    def status(self) -> ProviderStatus:
        if self.is_mock:
            return ProviderStatus.MOCK
        return ProviderStatus.REAL if self.is_configured() else ProviderStatus.UNAVAILABLE

    def status_reason(self) -> str:
        """UNAVAILABLE 时的具体原因（真实适配器覆盖此方法）。"""
        if self.is_mock:
            return "mock provider（非真实能力）"
        if self.is_configured():
            return ""
        return "real provider not configured"

    def status_info(self) -> dict:
        return {
            "provider": self.name,
            "status": self.status().value,
            "configured": self.is_configured(),
            "is_mock": self.is_mock,
            "reason": self.status_reason(),
            "model": str(getattr(self, "model", "") or ""),
        }

    def info(self) -> dict:
        return self.status_info()

    async def close(self) -> None:  # noqa: B027 - optional hook
        return None


class ProviderRegistry:
    """Holds the active llm / vision / solver providers for one runtime."""

    def __init__(self, llm: Provider | None = None, vision: Provider | None = None,
                 solver: Provider | None = None) -> None:
        self.llm = llm
        self.vision = vision
        self.solver = solver

    def get_llm(self) -> Provider | None:
        return self.llm

    def get_vision(self) -> Provider | None:
        return self.vision

    def get_solver(self) -> Provider | None:
        return self.solver

    # ---- V2 provider_status：系统级诚实能力声明 ----
    def status_summary(self) -> dict:
        """{llm: {...}|None, vision: {...}|None, solver: {...}|None}"""
        out: dict[str, Any] = {}
        for key, p in (("llm", self.llm), ("vision", self.vision), ("solver", self.solver)):
            out[key] = p.status_info() if p is not None else None
        return out

    def capabilities_text(self) -> str:
        """一句话能力摘要（用于 CLI/报告）。"""
        parts = []
        for key in ("llm", "vision", "solver"):
            info = self.status_summary()[key]
            if info is None:
                parts.append(f"{key}: NONE")
            else:
                parts.append(f"{key}: {info['status'].upper()}")
        return " | ".join(parts)

    async def close_all(self) -> None:
        for p in (self.llm, self.vision, self.solver):
            if p is not None:
                try:
                    await p.close()
                except Exception:  # pragma: no cover
                    pass


async def _aclose(obj: Any) -> None:
    if hasattr(obj, "aclose"):
        await obj.aclose()
    elif hasattr(obj, "close"):
        await asyncio.get_running_loop().run_in_executor(None, obj.close)


def build_providers(config: AppConfig) -> ProviderRegistry:
    """Build provider instances from configuration. Never raises for 'mock'.

    Real providers are constructed lazily and only raise ProviderNotConfiguredError
    when they are actually *used* without proper credentials, so a fresh clone
    runs out of the box in fully-mock mode.

    V2 LLM 具名 Provider：mock | openai | kimi | anthropic | claude | gemini
    （kimi/openai/gemini 走 OpenAI 兼容通道；claude 走 Anthropic Messages API）。
    Fast/Deep 双模型：llm.model（fast）+ llm.deep_model（deep，缺省回退 fast）。
    """
    from .llm import (
        AnthropicLLMProvider, ClaudeProvider, DeepSeekProvider, GeminiProvider,
        KimiProvider, MockLLMProvider, OpenAIProvider, OpenAICompatibleLLMProvider,
    )
    from .solver import MockSolverProvider, PostflopSolverCLIProvider
    from .vision import MockVisionProvider, MultimodalLLMVisionProvider

    llm_cfg = config.llm_config()
    vision_cfg = config.vision_config()
    solver_cfg = config.solver_config()

    # 模型名留空 → 各具名 Provider 使用自己的默认模型
    # （config 里的通用默认不再覆盖 deepseek-chat / moonshot-v1-8k 等）
    common = dict(
        api_key_env=llm_cfg.get("api_key_env", "LLM_API_KEY"),
        api_key=llm_cfg.get("api_key"),
        base_url=llm_cfg.get("base_url") or "",
        timeout_s=float(llm_cfg.get("timeout_s", 60)),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        price_in_per_1m=float(llm_cfg.get("price_in_per_1m", 0.0)),
        price_out_per_1m=float(llm_cfg.get("price_out_per_1m", 0.0)),
    )
    if llm_cfg.get("model"):
        common["model"] = llm_cfg.get("model")
    if llm_cfg.get("deep_model"):
        common["deep_model"] = llm_cfg.get("deep_model")

    llm_name = (llm_cfg.get("provider") or "mock").lower()
    if llm_name == "mock":
        llm: Provider = MockLLMProvider()
    elif llm_name == "kimi":
        llm = KimiProvider(**common)
    elif llm_name == "openai":
        llm = OpenAIProvider(**common)
    elif llm_name == "deepseek":
        llm = DeepSeekProvider(**common)
    elif llm_name == "gemini":
        llm = GeminiProvider(**common)
    elif llm_name == "claude":
        llm = ClaudeProvider(
            api_key_env=llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            api_key=llm_cfg.get("api_key"),
            model=llm_cfg.get("model") or "claude-3-5-haiku-latest",
            deep_model=llm_cfg.get("deep_model") or "claude-3-5-sonnet-latest",
            timeout_s=float(llm_cfg.get("timeout_s", 60)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        )
    elif llm_name == "anthropic":
        llm = AnthropicLLMProvider(
            api_key_env=llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
            api_key=llm_cfg.get("api_key"),
            model=llm_cfg.get("model") or "claude-3-5-haiku-latest",
            deep_model=llm_cfg.get("deep_model") or "",
            timeout_s=float(llm_cfg.get("timeout_s", 60)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        )
    else:
        raise ConfigError(f"unknown llm.provider '{llm_name}' (expected: mock|openai|kimi|deepseek|claude|anthropic|gemini)")

    vision_name = (vision_cfg.get("provider") or "mock").lower()
    if vision_name == "mock":
        vision: Provider = MockVisionProvider(fixture_path=vision_cfg.get("fixture") or None)
    elif vision_name == "multimodal":
        vision = MultimodalLLMVisionProvider(
            llm_provider=llm,
            model=vision_cfg.get("model") or llm_cfg.get("model"),
        )
    else:
        raise ConfigError(f"unknown vision.provider '{vision_name}' (expected: mock|multimodal)")

    solver_name = (solver_cfg.get("provider") or "mock").lower()
    if solver_name == "mock":
        solver: Provider = MockSolverProvider(seed=int(solver_cfg.get("seed", 42)))
    elif solver_name == "postflop":
        solver = PostflopSolverCLIProvider(executable=solver_cfg.get("path") or "")
    elif solver_name == "realengine":
        from .real_engine import RealEngineSolverProvider
        solver = RealEngineSolverProvider()
    else:
        raise ConfigError(f"unknown solver.provider '{solver_name}' (expected: mock|postflop|realengine)")

    return ProviderRegistry(llm=llm, vision=vision, solver=solver)
