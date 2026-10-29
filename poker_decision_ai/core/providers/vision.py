"""Vision Provider Interface + Mock + multimodal adapter.

The Vision Agent never hard-codes a specific OCR/vision tool.  Providers:
  - MockVisionProvider      : reads a YAML fixture (or inline data) - offline
  - MultimodalLLMVisionProvider : sends the image to a multimodal LLM with a
    strict JSON schema and validates the answer (requires an image-capable
    LLM provider, e.g. openai + gpt-4o, or a local vision model)
"""
from __future__ import annotations

import base64
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ProviderError, ProviderNotConfiguredError
from .base import Provider
from .llm import LLMProvider, LLMUsage, extract_json

#: JSON schema every vision provider must return (the Vision Agent contract)
VISION_OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "VisionObservation",
    "properties": {
        "source": {"type": "string"},
        "hero_seat": {"type": ["integer", "null"]},
        "button_seat": {"type": ["integer", "null"]},
        "street": {"type": "string", "enum": ["preflop", "flop", "turn", "river"]},
        "hole_cards": {"type": "array", "items": {"type": "string"}},
        "board": {"type": "array", "items": {"type": "string"}},
        "seats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seat": {"type": "integer"},
                    "name": {"type": "string"},
                    "stack": {"type": "number"},
                    "in_hand": {"type": "boolean"},
                },
                "required": ["seat", "name", "stack", "in_hand"],
            },
        },
        "pot": {"type": "number"},
        "current_bet": {"type": "number"},
        "to_call": {"type": "number"},
        "blinds": {"type": "object"},
        "ante": {"type": "number"},
        "available_actions": {"type": "array", "items": {"type": "string"}},
        "action_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "field_confidences": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "extra": {"type": "object"},
    },
    "required": ["source", "street", "confidence"],
    "additionalProperties": False,
}


@dataclass
class VisionObservation:
    source: str = "vision"
    hero_seat: int | None = None
    button_seat: int | None = None
    street: str = "preflop"
    hole_cards: list = field(default_factory=list)
    board: list = field(default_factory=list)
    seats: list = field(default_factory=list)
    pot: float = 0.0
    current_bet: float = 0.0
    to_call: float = 0.0
    blinds: dict = field(default_factory=dict)
    ante: float = 0.0
    available_actions: list = field(default_factory=list)
    action_text: str = ""
    confidence: float = 0.0
    # V2：逐字段置信度（field -> 0..1）；缺省用整体 confidence
    field_confidences: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def field_confidence(self, field: str, default: float = 0.0) -> float:
        v = self.field_confidences.get(field)
        if isinstance(v, (int, float)):
            return float(v)
        return float(default)


class VisionProvider(Provider):
    """Pluggable table-vision interface (screenshot / frame -> observation)."""

    name = "vision"

    @abstractmethod
    async def observe(self, image_bytes: bytes | None, *, hint: dict | None = None) -> VisionObservation:
        """image_bytes=None allows fixture-only / text-only providers."""


def load_image_bytes(path_or_dataurl: str) -> bytes:
    """Load image bytes from a file path or a `data:image/...;base64,` URL."""
    if path_or_dataurl.startswith("data:"):
        _, _, b64 = path_or_dataurl.partition(",")
        return base64.b64decode(b64)
    p = Path(path_or_dataurl)
    if not p.exists():
        raise ProviderError(f"image file not found: {p}")
    return p.read_bytes()


def load_fixture(fixture_path: str | None, inline: dict | None) -> dict | None:
    """Load a vision fixture (YAML file or inline dict)."""
    if inline:
        return inline
    if fixture_path:
        p = Path(fixture_path)
        if not p.exists():
            raise ProviderError(f"vision fixture not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return None


class MockVisionProvider(VisionProvider):
    """Offline provider: returns a fixture observation (file or inline)."""

    name = "mock"
    is_mock = True

    def __init__(self, fixture: dict | None = None, fixture_path: str | None = None) -> None:
        self.fixture = fixture
        self.fixture_path = fixture_path

    async def observe(self, image_bytes=None, *, hint=None) -> VisionObservation:
        data = load_fixture(self.fixture_path, self.fixture)
        if not data:
            raise ProviderError("MockVisionProvider has no fixture. Configure vision.fixture "
                                "or pass vision_observation directly to the pipeline.")
        obs = {k: v for k, v in data.items() if k in VisionObservation.__dataclass_fields__}
        obs.setdefault("source", "mock")
        obs.setdefault("street", "preflop")
        obs.setdefault("confidence", 0.8)
        # V2：fixture 可提供逐字段置信度；缺省用整体 confidence
        fconf = obs.get("field_confidences") or {}
        overall = float(obs["confidence"])
        for key in ("hole_cards", "board", "pot", "to_call", "current_bet",
                    "button_seat", "hero_seat", "seats", "blinds", "available_actions"):
            fconf.setdefault(key, overall)
        obs["field_confidences"] = fconf
        return VisionObservation(**obs)


class MultimodalLLMVisionProvider(VisionProvider):
    """Real adapter: image -> multimodal LLM -> validated observation.

    Requires an LLM provider that supports image content (OpenAI gpt-4o class,
    Kimi vision models, or a local multimodal model via OpenAI-compat).
    """

    name = "multimodal"

    def __init__(self, llm_provider: LLMProvider, model: str = "") -> None:
        self.llm = llm_provider
        self.model = model or getattr(llm_provider, "model", "")

    def is_configured(self) -> bool:
        return self.llm is not None and self.llm.is_configured() and self.llm.supports_images()

    def status_reason(self) -> str:
        if self.is_configured():
            return ""
        if self.llm is None:
            return "no LLM provider attached for multimodal vision"
        if not self.llm.is_configured():
            return "attached LLM provider is not configured (no API key)"
        return "attached LLM provider does not support images（需 vision 模型）"

    async def observe(self, image_bytes=None, *, hint=None) -> VisionObservation:
        if image_bytes is None:
            raise ProviderNotConfiguredError("multimodal vision requires an image")
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Multimodal vision requires an image-capable LLM (llm.provider=openai with a vision model). "
                "Fallback: use vision.provider=mock with a fixture."
            )
        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png"
        if image_bytes[:4] == b"\xff\xd8\xff\xe0" or image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        prompt = (
            "Read this poker table screenshot and return a JSON object matching the schema. "
            "Cards use short notation (As, Kh, Td, 9c, 2s). Be precise; if you cannot read a value, "
            "omit it and add a warning. "
            "You MUST also fill field_confidences: for each recognized field (hole_cards, board, "
            "pot, to_call, current_bet, button_seat, hero_seat, seats, blinds, available_actions) "
            "give your confidence 0..1 that the value is correct - a low value is better than a "
            f"wrong value. Table hint: {hint or 'none'}"
        )
        from ..schemas.dataclass_schema import from_dict as _fd

        try:
            # OpenAI-compatible multimodal: content list with image_url
            if hasattr(self.llm, "base_url"):
                import json as _json

                payload_msg = [
                    {"type": "text", "text": prompt + "\nSchema:\n" + _json.dumps(VISION_OBSERVATION_SCHEMA)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]
                from .llm import LLMResponse, LLMUsage, OpenAICompatibleLLMProvider

                async with __import__("httpx").AsyncClient(timeout=self.llm.timeout_s) as client:
                    self.llm._require_key()
                    resp = await client.post(
                        f"{self.llm.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.llm._key}"},
                        json={"model": self.model, "messages": [{"role": "user", "content": payload_msg}],
                              "response_format": {"type": "json_object"}},
                    )
                    resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage_data = data.get("usage") or {}
                usage = LLMUsage(int(usage_data.get("prompt_tokens") or 0), int(usage_data.get("completion_tokens") or 0))
                raw = _fd(VisionObservation, extract_json(text))
                return raw
            raise ProviderNotConfiguredError(
                "The configured LLM provider does not support multimodal input for vision. "
                "Use llm.provider=openai with a vision-capable model."
            )
        except ProviderError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise ProviderError(f"multimodal vision failed: {e}") from e

    def supports_images(self) -> bool:
        return True
