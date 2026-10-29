"""Level 4 Multimodal Fallback：低频、事件触发、预算受控。

只允许以下情况触发（全部满足其一才调用，绝不每帧调用远程模型）：
  NEW_HAND / STATE_CONFLICT / LAYOUT_UNKNOWN / OCR_FAILURE_STREAK /
  CARD_UNCERTAIN / KEY_FIELD_CONFLICT / TEMPORAL_UNEXPLAINED

硬约束：
  - cooldown（min_interval_s）与每手预算（max_calls_per_hand）双重闸门；
  - 后端不可用 → 明确 UNAVAILABLE（不伪造结果）；
  - 后端输出仍必须经过 V0 schema/validator（见 pipeline），不得绕过。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .image import ImageData


class FallbackTrigger(str, Enum):
    NEW_HAND = "NEW_HAND"
    STATE_CONFLICT = "STATE_CONFLICT"
    LAYOUT_UNKNOWN = "LAYOUT_UNKNOWN"
    OCR_FAILURE_STREAK = "OCR_FAILURE_STREAK"
    CARD_UNCERTAIN = "CARD_UNCERTAIN"
    KEY_FIELD_CONFLICT = "KEY_FIELD_CONFLICT"
    TEMPORAL_UNEXPLAINED = "TEMPORAL_UNEXPLAINED"


@dataclass
class FallbackPolicyConfig:
    enabled: bool = True
    min_interval_s: float = 3.0       # 两次调用最小间隔（cooldown）
    max_calls_per_hand: int = 2       # 每手牌调用预算


@dataclass(frozen=True)
class FallbackDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class FallbackCall:
    trigger: FallbackTrigger
    at: float
    frame_id: int
    outcome: str = "pending"
    latency_s: float | None = None

    def to_dict(self) -> dict:
        return {"trigger": self.trigger.value, "at": self.at,
                "frame_id": self.frame_id, "outcome": self.outcome,
                "latency_s": self.latency_s}


class FallbackGate:
    """触发条件闸门：保证「绝不每帧调用」。"""

    def __init__(self, config: FallbackPolicyConfig | None = None, clock=None):
        import time as _time
        self.config = config or FallbackPolicyConfig()
        self.clock = clock or _time.monotonic
        self.last_call_at: float | None = None
        self.calls_this_hand = 0
        self.calls: list = []

    def request(self, trigger: FallbackTrigger, *, frame_id: int,
                now: float | None = None) -> FallbackDecision:
        now = self.clock() if now is None else now
        if not self.config.enabled:
            return FallbackDecision(False, "fallback_disabled")
        if self.last_call_at is not None \
                and now - self.last_call_at < self.config.min_interval_s:
            return FallbackDecision(False, "cooldown")
        if self.calls_this_hand >= self.config.max_calls_per_hand:
            return FallbackDecision(False, "hand_budget_exhausted")
        return FallbackDecision(True, "allowed")

    def record(self, call: FallbackCall) -> None:
        self.calls.append(call)
        self.last_call_at = call.at
        self.calls_this_hand += 1

    def on_new_hand(self) -> None:
        self.calls_this_hand = 0

    def stats(self) -> dict:
        return {"calls_total": len(self.calls),
                "calls_this_hand": self.calls_this_hand,
                "last_call_at": self.last_call_at,
                "config": {"enabled": self.config.enabled,
                           "min_interval_s": self.config.min_interval_s,
                           "max_calls_per_hand":
                               self.config.max_calls_per_hand}}


class MultimodalStatus(str, Enum):
    OK = "OK"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class MultimodalResult:
    status: MultimodalStatus = MultimodalStatus.UNAVAILABLE
    raw_fields: dict | None = None   # RawVisionState 字段字典（仍需过 validator）
    error: str | None = None
    synthetic_test_only: bool = True

    def to_dict(self) -> dict:
        return {"status": self.status.value, "raw_fields": self.raw_fields,
                "error": self.error,
                "synthetic_test_only": self.synthetic_test_only}


class MultimodalBackend(ABC):
    name = "multimodal"
    synthetic_test_only = True

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return "no multimodal backend configured"

    @abstractmethod
    def observe(self, frame_image: ImageData,
                roi_crops: dict, context: dict | None = None) -> MultimodalResult:
        """请求多模态重读；结果仅描述所见，仍须经 schema/validator。"""


class UnavailableMultimodalBackend(MultimodalBackend):
    """无多模态后端时的诚实占位。"""
    name = "unavailable_multimodal"

    def observe(self, frame_image=None, roi_crops=None,
                context=None) -> MultimodalResult:
        return MultimodalResult(status=MultimodalStatus.UNAVAILABLE,
                                error=self.unavailable_reason())


class SyntheticMultimodalBackend(MultimodalBackend):
    """场景地面真值多模态（synthetic_test_only=True）。"""
    name = "synthetic_multimodal"

    def __init__(self, get_scene: Callable[[], Any], *,
                 call_log: list | None = None,
                 override_fields: dict | None = None):
        self.get_scene = get_scene
        self.call_log = call_log
        self.override_fields = override_fields

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def observe(self, frame_image=None, roi_crops=None,
                context=None) -> MultimodalResult:
        if self.call_log is not None:
            self.call_log.append("multimodal_called")
        scene = self.get_scene()
        fields = scene.raw_fields()
        if self.override_fields:
            fields = {**fields, **self.override_fields}
        return MultimodalResult(status=MultimodalStatus.OK,
                                raw_fields=fields)
