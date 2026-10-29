"""Vision Agent (感知层): screenshot / dataURL -> VisionObservation.

Input : {request: dict|None, image: str|None}
Output: VisionOutput {observation: dict|None, provider, confidence, evidence, warnings}

Priority / logic (contract §4 vision):
  1. request.vision_observation (dict) present -> wrap & return directly
  2. else image (file path or dataURL) -> ctx.providers.get_vision().observe(bytes)
  3. neither -> applicable() returns (False, ...)

Degradation: missing vision provider / ProviderNotConfiguredError / ProviderError
are all caught -> observation=None + warning (never raise non-Framework errors).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.errors import ProviderError, ProviderNotConfiguredError
from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.providers.vision import VisionObservation, load_image_bytes


def _normalize_observation(data: dict) -> dict:
    """Whitelist a raw dict down to VisionObservation's declared fields.

    只保留 VisionObservation 定义的字段，缺失字段用 dataclass 默认值补齐，
    从而保证输出 observation 的形状稳定（契约 §4 vision 的字段集合）。
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(VisionObservation):
        name = f.name
        if name in data:
            out[name] = data[name]
        elif f.default is not dataclasses.MISSING:
            out[name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[name] = f.default_factory()
        else:
            out[name] = None
    return out


def _obs_to_dict(obs: VisionObservation | dict | None) -> dict | None:
    """VisionObservation (dataclass) or raw dict -> plain dict (contract field set)."""
    if obs is None:
        return None
    if isinstance(obs, VisionObservation):
        return dataclasses.asdict(obs)
    if isinstance(obs, dict):
        return _normalize_observation(obs)
    return None


@dataclass
class VisionOutput:
    observation: dict | None = None
    provider: str = ""
    confidence: float = 0.0
    field_confidences: dict = field(default_factory=dict)   # V2 逐字段置信度
    low_confidence_fields: list = field(default_factory=list)  # 低于阈值的字段
    requires_recapture: bool = False        # 建议重新截图识别
    manual_input_required: bool = False     # 重试仍失败：降级到人工输入
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# V2 关键字段：置信度低会触发重新识别/人工输入降级（不允许假装识别成功）
_CRITICAL_FIELDS = ("hole_cards", "board", "pot", "to_call", "seats")


class VisionAgent(BaseAgent):
    name: ClassVar[str] = "vision"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "视觉感知 Agent"
    layer: ClassVar[str] = "perception"
    depends: ClassVar[list[str]] = []
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = VisionOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        request = raw_input.get("request") or {}
        image = raw_input.get("image")
        if request.get("vision_observation"):
            return True, ""
        if image:
            return True, ""
        return False, "no image or vision_observation"

    async def handle(self, inp: dict, ctx: AgentContext) -> VisionOutput:
        request = inp.get("request") or {}
        image = inp.get("image")
        warnings: list[dict] = []

        # ---- 1) 直通：request 里已有 vision_observation -> 直接包装返回 ----
        vo = request.get("vision_observation")
        if isinstance(vo, dict) and vo:
            obs = _normalize_observation(vo)
            provider = str(vo.get("source") or "vision")
            # 调用方已给出观测：默认完全信任；若显式给出 confidence 则尊重之
            if "confidence" in vo and isinstance(vo.get("confidence"), (int, float)):
                confidence = float(vo["confidence"])
            else:
                confidence = 1.0
            fconf = vo.get("field_confidences") if isinstance(vo.get("field_confidences"), dict) else {}
            return VisionOutput(
                observation=obs,
                provider=provider,
                confidence=round(max(0.0, min(1.0, confidence)), 4),
                field_confidences={k: float(v) for k, v in fconf.items()},
                low_confidence_fields=[],
                evidence=[{"kind": "vision", "source": "request.vision_observation",
                           "text": f"直接使用 request.vision_observation (source={provider})",
                           "weight": 1.0}],
                warnings=warnings,
            )

        # ---- 防御：两者皆无（applicable 已拦截，但 handle 直接调用时也要安全）----
        if not image:
            warnings.append({"code": "NO_INPUT", "message": "no image or vision_observation"})
            return VisionOutput(observation=None, provider="", confidence=0.0,
                                evidence=[], warnings=warnings)

        # ---- 2) image 路径（V2：低置信 → 重试 → 人工输入降级） ----
        provider = ctx.providers.get_vision()
        if provider is None:
            warnings.append({"code": "NO_VISION_PROVIDER",
                             "message": "no vision provider configured; observation unavailable"})
            return VisionOutput(observation=None, provider="", confidence=0.0,
                                evidence=[], warnings=warnings)

        vision_cfg = ctx.config.vision_config()
        min_field_conf = float(vision_cfg.get("min_field_confidence", 0.6) or 0.6)
        retry_attempts = int(vision_cfg.get("retry_attempts", 1) or 1)

        try:
            image_bytes = load_image_bytes(str(image))
        except ProviderError as e:
            warnings.append({"code": e.code, "message": str(e)})
            return VisionOutput(observation=None, provider=getattr(provider, "name", ""),
                                confidence=0.0, evidence=[], warnings=warnings)

        obs: VisionObservation | None = None
        for attempt in range(retry_attempts + 1):
            try:
                obs = await provider.observe(image_bytes)
            except (ProviderNotConfiguredError, ProviderError) as e:
                warnings.append({"code": getattr(e, "code", "PROVIDER_ERROR"),
                                 "message": str(e)})
                break
            fconf = obs.field_confidences or {}
            low = [f for f in _CRITICAL_FIELDS
                   if f in fconf and float(fconf[f]) < min_field_conf]
            if not low:
                break
            if attempt < retry_attempts:
                warnings.append({"code": "LOW_FIELD_CONFIDENCE_RETRY",
                                 "message": f"低置信字段 {low}，第 {attempt + 1} 次重新识别"})

        if obs is None:
            return VisionOutput(observation=None,
                                provider=getattr(provider, "name", ""),
                                confidence=0.0, evidence=[], warnings=warnings)

        fconf = obs.field_confidences or {}
        low = [f for f in _CRITICAL_FIELDS if f in fconf and float(fconf[f]) < min_field_conf]
        requires_recapture = bool(low) and retry_attempts == 0
        manual_input_required = bool(low) and retry_attempts > 0
        if low:
            warnings.append({
                "code": "LOW_FIELD_CONFIDENCE",
                "message": (f"关键字段置信度低于 {min_field_conf}: {low}；"
                            + ("请求重新截图识别" if requires_recapture
                               else "重试后仍低置信，建议降级到人工输入")),
            })

        obs_dict = _obs_to_dict(obs)
        confidence = float(getattr(obs, "confidence", 0.0) or 0.0)
        return VisionOutput(
            observation=obs_dict,
            provider=getattr(provider, "name", "vision"),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            field_confidences={k: float(v) for k, v in fconf.items()},
            low_confidence_fields=low,
            requires_recapture=requires_recapture,
            manual_input_required=manual_input_required,
            evidence=[{"kind": "vision", "source": getattr(provider, "name", "vision"),
                       "text": f"视觉识别完成 (provider={getattr(provider, 'name', 'vision')})",
                       "weight": 1.0}],
            warnings=warnings,
        )
