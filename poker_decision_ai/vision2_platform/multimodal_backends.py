"""多模态 LLM 兜底后端（复用核心 MultimodalLLMVisionProvider）。

- 仅当存在 vision-capable LLM（provider.is_configured() 且
  supports_images()）时 available()=True；否则诚实降级 UNAVAILABLE；
- 输出映射为 RawVisionState 字段字典，随后仍必须经过 V0 validator
  （realtime 流水线保证），后端本身不能绕过；
- 身份规则：不保存真实昵称 —— seats 一律转为匿名 player_N。
"""
from __future__ import annotations

import asyncio
from typing import Any

from poker_decision_ai.vision2.realtime.fallback import (MultimodalBackend,
                                                         MultimodalResult,
                                                         MultimodalStatus)
from poker_decision_ai.vision2.realtime.image import ImageData

from .png_encode import encode_rgba_png


def _map_observation(obs: Any) -> dict:
    """VisionObservation → RawVisionState 字段字典（只取可映射事实）。"""
    fields: dict = {}
    if getattr(obs, "hero_seat", None) is not None:
        fields["hero_seat"] = int(obs.hero_seat)
    if getattr(obs, "button_seat", None) is not None:
        fields["button_seat"] = int(obs.button_seat)
    if getattr(obs, "street", None) in ("preflop", "flop", "turn", "river"):
        fields["street"] = obs.street
    hole = list(getattr(obs, "hole_cards", None) or [])
    if hole:
        fields["hero_hole_cards"] = hole
    board = list(getattr(obs, "board", None) or [])
    if board:
        n = len(board)
        fields["board"] = {
            "flop": board[:3] if n >= 3 else None,
            "turn": board[3:4] if n >= 4 else None,
            "river": board[4:] if n >= 5 else None,
        }
    pot = getattr(obs, "pot", None)
    if isinstance(pot, (int, float)) and pot == pot:
        fields["pot"] = float(pot)
    to_call = getattr(obs, "to_call", None)
    if isinstance(to_call, (int, float)) and to_call == to_call:
        fields["hero_amount_to_call"] = float(to_call)
    current_bet = getattr(obs, "current_bet", None)
    if isinstance(current_bet, (int, float)) and current_bet == current_bet:
        fields["hero_current_bet"] = float(current_bet)
    blinds = getattr(obs, "blinds", None)
    if isinstance(blinds, dict) and blinds:
        fields["blinds"] = dict(blinds)
    seats = getattr(obs, "seats", None) or []
    raw_seats = []
    from poker_decision_ai.vision2.schemas import RawSeat
    for s in seats:
        seat = s.get("seat") if isinstance(s, dict) else None
        if seat is None:
            continue
        raw_seats.append(RawSeat(
            seat=int(seat),
            anonymous_player_id=f"player_{int(seat)}",  # 禁止真实昵称
            stack=(float(s["stack"]) if isinstance(s.get("stack"),
                                                   (int, float)) else None),
            in_hand=(s.get("in_hand") if isinstance(s.get("in_hand"), bool)
                     else None),
        ))
    if raw_seats:
        fields["seats"] = raw_seats
    return fields


class LLMMultimodalBackend(MultimodalBackend):
    """真实多模态兜底（依赖 vision-capable LLM 配置）。"""
    name = "llm_multimodal"
    synthetic_test_only = False

    def __init__(self, provider=None, model: str = ""):
        self.provider = provider
        self.model = model

    def available(self) -> bool:
        if self.provider is None:
            return False
        is_conf = getattr(self.provider, "is_configured", None)
        supports = getattr(self.provider, "supports_images", None)
        return callable(is_conf) and is_conf() and callable(supports) \
            and supports()

    def unavailable_reason(self) -> str:
        if self.provider is None:
            return "no LLM vision provider attached"
        if not callable(getattr(self.provider, "is_configured", None)) \
                or not getattr(self.provider, "is_configured")():
            return "attached LLM provider is not configured"
        return "attached LLM provider does not support images (需 vision 模型)"

    def observe(self, frame_image: ImageData, roi_crops: dict | None = None,
                context: dict | None = None) -> MultimodalResult:
        if not self.available():
            return MultimodalResult(status=MultimodalStatus.UNAVAILABLE,
                                    error=self.unavailable_reason(),
                                    synthetic_test_only=False)
        target = ((roi_crops or {}).get("board") or frame_image)
        png = encode_rgba_png(target.width, target.height,
                              target.rgba_bytes())
        try:
            obs = asyncio.run(self.provider.observe(png, hint=context))
        except Exception as e:
            return MultimodalResult(status=MultimodalStatus.FAILED,
                                    error=f"multimodal provider error: {e}",
                                    synthetic_test_only=False)
        fields = _map_observation(obs)
        if not fields:
            return MultimodalResult(status=MultimodalStatus.UNCERTAIN,
                                    error="multimodal returned no mappable "
                                          "fields",
                                    synthetic_test_only=False)
        return MultimodalResult(status=MultimodalStatus.OK,
                                raw_fields=fields, synthetic_test_only=False)


def build_llm_multimodal_backend(config_path: str | None = None
                                 ) -> LLMMultimodalBackend | None:
    """按当前框架配置构造（LLM 未配置/非 vision 模型 → None）。"""
    try:
        from poker_decision_ai.core.config import load_config
        from poker_decision_ai.core.providers.base import build_providers
        cfg = load_config(config_path) if config_path else load_config()
        registry = build_providers(cfg)
        vision = registry.get_vision()
        if vision is None:
            return LLMMultimodalBackend(provider=None)
        return LLMMultimodalBackend(provider=vision,
                                    model=getattr(vision, "model", ""))
    except Exception:
        return None
