"""V3F-P30-10-D — Capture → Event Stream Consumer（事件源消费者）。

最小 adapter：FrameStream/VisualEventBus 的视觉事实 → ActionEventSource
（仅 VISUAL_FACT——绝不自动升级为 poker 行为）。

来源边界（冻结）：
  HAND_HISTORY → 行为事件（合法重放）；
  EVENT_STREAM → 行为事件（StateEngine 流）；
  VISUAL_FACT  → 视觉事实（ROI_CHANGED/BET_CHANGED/STACK_CHANGED/...）；
  NONE/UNKNOWN → 单帧/无源（不猜）。

行为来源检测：若调用方提供 hand_history/events（合法）→ RECOGNIZED；
否则仅视觉事实（source=VISUAL_FACT，status=VISUAL_FACT_ONLY）→
action_history 语义 UNKNOWN。

红线：visual diff → 绝不产生 CALL/CHECK/BET/RAISE/FOLD/ALL_IN；
每个 event 带 provenance；previous-frame 不用于猜测。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .action_event import ActionEvent, visual_fact_event, unknown_source
from .action_event_source import (ActionEventSource, from_hand_history,
                                  from_event_stream, no_source)


@dataclass
class EventStreamConsumer:
    """消费视觉事实事件流 → ActionEventSource（visual fact 层）。"""

    events: list = field(default_factory=list)        # ActionEvent（visual fact only）
    source: ActionEventSource = field(default_factory=no_source)

    def ingest_visual_fact(self, kind: str, frame_id: int,
                           detail: dict | None = None) -> ActionEvent:
        """V2M VisualEventBus 视觉事实 → ActionEvent（VISUAL_FACT——
        绝不归因行为）。"""
        ev = visual_fact_event(kind, frame_id, detail)
        if kind not in ("ROI_CHANGED", "BET_CHANGED", "STACK_CHANGED",
                        "ACTION_PANEL_APPEARED",
                        "ACTION_PANEL_DISAPPEARED", "STATE_CONFLICT",
                        "CARD_APPEARED", "CARD_CHANGED", "BUTTON_MOVED"):
            ev.provenance["outside_whitelist"] = True
            ev.status = "UNKNOWN"
        self.events.append(ev)
        return ev

    def with_hand_history(self, parsed_actions: list | None
                          ) -> ActionEventSource:
        """提供合法 hand_history → 行为事件源（RECOGNIZED）。"""
        self.source = from_hand_history(parsed_actions)
        return self.source

    def with_event_stream(self, events: list | None) -> ActionEventSource:
        """提供 StateEngine 事件流 → 行为事件源（RECOGNIZED/UNKNOWN）。"""
        self.source = from_event_stream(events)
        return self.source

    def current_source(self) -> ActionEventSource:
        """无合法行为源 → NONE/UNKNOWN（仅视觉事实——不猜）。"""
        if self.source.status != "RECOGNIZED":
            self.source = no_source()
        return self.source

    def summary(self) -> dict:
        return {"event_count": len(self.events),
                "visual_fact_only": all(
                    e.status == "VISUAL_FACT_ONLY" for e in self.events),
                "behavior_events": sum(
                    1 for e in self.events if e.is_behavior()),
                "source": self.current_source().to_dict(),
                "events": [e.to_dict() for e in self.events]}


def fact_to_action_history(consumer: EventStreamConsumer) -> str:
    """视觉事实层 → action_history 语义（始终 UNKNOWN——不猜行为）。

    行为历史只能由 HAND_HISTORY/EVENT_STREAM 提供；视觉事实流单独存在
    于 summary()（VISUAL_FACT_ENVOS——供范围/审计消费）。
    """
    return "UNKNOWN"
