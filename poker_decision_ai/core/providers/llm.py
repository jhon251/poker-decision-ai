"""LLM Provider Interface + Mock + real adapters.

Interfaces:
  - LLMProvider.complete(messages, json_schema=None, ...) -> LLMResponse

Implementations:
  - MockLLMProvider                 : offline, deterministic, schema-aware
  - OpenAICompatibleLLMProvider     : OpenAI / Kimi / DeepSeek / Gemini(OpenAI-compat) / local vLLM & Ollama
  - AnthropicLLMProvider            : Claude Messages API

Rule: LLM explains and reasons; it never *computes* equity/odds/ICM/stats.
Agents call the helper `complete_json` which guarantees a schema-valid object
or raises - the caller then falls back to its deterministic path.
"""
from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import ProviderError, ProviderNotConfiguredError
from ..schemas.dataclass_schema import from_dict, json_schema_for, to_dict
from .base import Provider


def extract_json(text: str) -> dict:
    """Extract the first balanced {...} JSON object from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        raise ProviderError("LLM output contains no JSON object")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as e:
                    raise ProviderError(f"LLM returned malformed JSON: {e}") from e
    raise ProviderError("LLM JSON object is not closed")


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    raw: Any = None

    def to_json(self, schema_cls: type | None = None) -> dict:
        data = extract_json(self.text)
        if schema_cls is not None:
            try:
                from_dict(schema_cls, data)
            except Exception as e:
                raise ProviderError(f"LLM JSON does not match required schema: {e}") from e
        return data


class LLMProvider(Provider):
    """Async LLM completion interface.

    V2 Fast/Deep 双模型：complete(..., tier="fast"|"deep") 选择模型。
    """

    name = "llm"
    model: str = ""
    deep_model: str = ""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tier: str = "fast",
    ) -> LLMResponse:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}]"""

    def model_for(self, tier: str = "fast") -> str:
        """Fast/Deep 模型选择：deep 未单独配置时回退 fast 模型。"""
        if tier == "deep" and getattr(self, "deep_model", ""):
            return self.deep_model
        return self.model

    def models_info(self) -> dict:
        return {"fast": self.model_for("fast"), "deep": self.model_for("deep")}

    def supports_images(self) -> bool:
        return False


class MockLLMProvider(LLMProvider):
    """Deterministic offline provider.

    If a json_schema is given it fabricates a schema-valid instance so the
    whole pipeline (including LLM-dependent agents) runs offline with clearly
    marked mock content ("[mock]" prefixes).  If a `scripted` mapping is
    provided, a message containing the key returns the mapped text.
    """

    name = "mock"
    model = "mock"
    is_mock = True

    def __init__(self, scripted: dict[str, str] | None = None) -> None:
        self.scripted = scripted or {}

    @staticmethod
    def _fabricate(schema: dict, key: str = "") -> Any:
        t = schema.get("type")
        if "enum" in schema:
            return schema["enum"][0]
        if t == "string":
            return f"[mock] {key}"
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return False
        if t == "array":
            items = schema.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                return []
            return []
        if t == "object" or "properties" in schema:
            props = schema.get("properties", {})
            out = {}
            for k, sub in props.items():
                out[k] = MockLLMProvider._fabricate(sub, key=f"{key}.{k}" if key else k)
            return out
        return None

    async def complete(self, messages, *, json_schema=None, temperature=None,
                       max_tokens=None, tier="fast") -> LLMResponse:
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        for marker, text in self.scripted.items():
            if marker and marker in joined:
                return LLMResponse(text=text, usage=LLMUsage(), model=self.model_for(tier))
        if json_schema:
            payload = self._fabricate(json_schema)
            text = json.dumps(payload, ensure_ascii=False)
        else:
            text = "[mock] 离线推理结果（未配置真实 LLM Provider）"
        return LLMResponse(text=text, usage=LLMUsage(), model=self.model_for(tier))

    def supports_images(self) -> bool:
        return False


class OpenAICompatibleLLMProvider(LLMProvider):
    """Chat-Completions client. Works with any OpenAI-compatible endpoint:

    - OpenAI  : base_url https://api.openai.com/v1
    - Kimi    : base_url https://api.moonshot.cn/v1
    - DeepSeek: base_url https://api.deepseek.com/v1
    - Gemini  : base_url https://generativelanguage.googleapis.com/v1beta/openai
    - local   : base_url http://localhost:8000/v1  (vLLM) / http://localhost:11434/v1 (Ollama)
    """

    name = "openai"

    def __init__(self, *, api_key: str = "", api_key_env: str = "LLM_API_KEY",
                 base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini",
                 deep_model: str = "", timeout_s: float = 60.0, temperature: float = 0.0,
                 max_tokens: int = 2048,
                 price_in_per_1m: float = 0.0, price_out_per_1m: float = 0.0) -> None:
        self.api_key_env = api_key_env
        self._key = api_key or os.environ.get(api_key_env, "")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.deep_model = deep_model or model      # V2：deep 未配置回退 fast
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.price_in = price_in_per_1m
        self.price_out = price_out_per_1m

    def supports_images(self) -> bool:
        # OpenAI 兼容通道支持 image_url 内容；视觉能力取决于所选模型
        # （非视觉模型会在调用时返回错误并被 ProviderError 捕获降级）。
        return True

    def is_configured(self) -> bool:
        return bool(self._key)

    def status_reason(self) -> str:
        if self.is_configured():
            return ""
        return f"LLM API key missing (set {self.api_key_env or 'LLM_API_KEY'}), base_url={self.base_url}"

    def _require_key(self) -> None:
        if not self._key:
            raise ProviderNotConfiguredError(
                "LLM API key missing: set LLM_API_KEY (or llm.api_key) in .env / config.yaml. "
                "Use llm.provider=mock to run offline."
            )

    async def complete(self, messages, *, json_schema=None, temperature=None,
                       max_tokens=None, tier="fast") -> LLMResponse:
        self._require_key()
        payload: dict[str, Any] = {
            "model": self.model_for(tier),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_schema:
            payload["response_format"] = {"type": "json_object"}
            if messages:
                last = dict(messages[-1])
                last["content"] = str(last.get("content", "")) + (
                    "\n\nIMPORTANT: respond with a single JSON object matching this schema:\n"
                    + json.dumps(json_schema, ensure_ascii=False)
                )
                payload["messages"] = messages[:-1] + [last]
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"LLM HTTP {e.response.status_code}: {e.response.text[:400]}") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"LLM network error: {e}") from e
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"unexpected LLM response shape: {str(data)[:300]}") from e
        usage_data = data.get("usage") or {}
        pin = int(usage_data.get("prompt_tokens") or 0)
        pout = int(usage_data.get("completion_tokens") or 0)
        cost = pin / 1_000_000 * self.price_in + pout / 1_000_000 * self.price_out
        return LLMResponse(text=text, usage=LLMUsage(pin, pout, pin + pout, cost),
                           model=self.model_for(tier), raw=data)


class AnthropicLLMProvider(LLMProvider):
    """Claude Messages API adapter."""

    name = "anthropic"

    def __init__(self, *, api_key: str = "", api_key_env: str = "ANTHROPIC_API_KEY",
                 model: str = "claude-3-5-haiku-latest", deep_model: str = "",
                 timeout_s: float = 60.0, max_tokens: int = 2048) -> None:
        self.api_key_env = api_key_env
        self._key = api_key or os.environ.get(api_key_env, "")
        self.model = model
        self.deep_model = deep_model or model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def is_configured(self) -> bool:
        return bool(self._key)

    def status_reason(self) -> str:
        if self.is_configured():
            return ""
        return "ANTHROPIC_API_KEY missing"

    async def complete(self, messages, *, json_schema=None, temperature=None,
                       max_tokens=None, tier="fast") -> LLMResponse:
        if not self._key:
            raise ProviderNotConfiguredError(
                "ANTHROPIC_API_KEY missing. Set it in .env or switch llm.provider to mock/openai."
            )
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        if json_schema and chat:
            chat[-1]["content"] += "\n\nRespond ONLY with JSON matching:\n" + json.dumps(json_schema, ensure_ascii=False)
        payload = {
            "model": self.model_for(tier),
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "messages": chat,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"},
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"Anthropic HTTP {e.response.status_code}: {e.response.text[:400]}") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Anthropic network error: {e}") from e
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        pin, pout = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        return LLMResponse(text=text, usage=LLMUsage(pin, pout, pin + pout, 0.0),
                           model=self.model_for(tier), raw=data)


# ---------------------------------------------------------------- V2 具名 Provider
class KimiProvider(OpenAICompatibleLLMProvider):
    """Kimi（Moonshot）Provider：OpenAI 兼容通道的具名适配器。

    默认端点 https://api.moonshot.cn/v1；配置 llm.provider=kimi 即用。
    """

    name = "kimi"

    def __init__(self, *, api_key: str = "", api_key_env: str = "LLM_API_KEY",
                 model: str = "moonshot-v1-8k", deep_model: str = "moonshot-v1-32k",
                 base_url: str = "", **kw) -> None:
        if not base_url:
            base_url = "https://api.moonshot.cn/v1"      # 空 base_url → Kimi 默认端点
        super().__init__(api_key=api_key, api_key_env=api_key_env, base_url=base_url,
                         model=model, deep_model=deep_model, **kw)


class OpenAIProvider(OpenAICompatibleLLMProvider):
    """OpenAI 官方端点具名适配器（llm.provider=openai）。"""

    name = "openai"

    def __init__(self, *, api_key: str = "", api_key_env: str = "LLM_API_KEY",
                 model: str = "gpt-4o-mini", deep_model: str = "gpt-4o",
                 base_url: str = "", **kw) -> None:
        if not base_url:
            base_url = "https://api.openai.com/v1"       # 空 base_url → OpenAI 官方端点
        super().__init__(api_key=api_key, api_key_env=api_key_env, base_url=base_url,
                         model=model, deep_model=deep_model, **kw)


class GeminiProvider(OpenAICompatibleLLMProvider):
    """Gemini 的 OpenAI 兼容端点适配器（llm.provider=gemini）。"""

    name = "gemini"

    def __init__(self, *, api_key: str = "", api_key_env: str = "LLM_API_KEY",
                 model: str = "gemini-2.0-flash", deep_model: str = "gemini-2.0-pro",
                 base_url: str = "", **kw) -> None:
        if not base_url:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"  # 默认 Gemini OpenAI-compat 端点
        super().__init__(api_key=api_key, api_key_env=api_key_env, base_url=base_url,
                         model=model, deep_model=deep_model, **kw)


class ClaudeProvider(AnthropicLLMProvider):
    """Claude 具名适配器（llm.provider=claude；Anthropic Messages API）。"""

    name = "claude"


class DeepSeekProvider(OpenAICompatibleLLMProvider):
    """DeepSeek 具名适配器（llm.provider=deepseek；OpenAI 兼容通道）。

    默认端点 https://api.deepseek.com/v1；模型：deepseek-chat（快）/
    deepseek-reasoner（深）。注意：reasoner 若忽略 json_object 格式请求，
    JSON 提取失败会被 ProviderError 捕获并自动回退确定性模板（诚实降级）。
    """

    name = "deepseek"

    def __init__(self, *, api_key: str = "", api_key_env: str = "LLM_API_KEY",
                 model: str = "deepseek-chat", deep_model: str = "deepseek-reasoner",
                 base_url: str = "", **kw) -> None:
        if not base_url:
            base_url = "https://api.deepseek.com/v1"     # DeepSeek 官方 OpenAI 兼容端点
        super().__init__(api_key=api_key, api_key_env=api_key_env, base_url=base_url,
                         model=model, deep_model=deep_model, **kw)


async def complete_json(
    provider: LLMProvider | None,
    *,
    system: str,
    user: str,
    schema_cls: type,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tier: str = "fast",
) -> tuple[dict, LLMUsage] | None:
    """Helper: run LLM and validate output against a dataclass schema.

    Returns (validated_dict, usage) or None when no provider is available.
    Raises ProviderError when the provider exists but fails or returns
    invalid JSON - callers decide whether to degrade.
    tier: "fast" | "deep"（V2 双模型路径）。
    """
    if provider is None:
        return None
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = await provider.complete(messages, json_schema=json_schema_for(schema_cls),
                                   temperature=temperature, max_tokens=max_tokens, tier=tier)
    return to_dict(from_dict(schema_cls, extract_json(resp.text))), resp.usage
