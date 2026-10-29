"""V2M — Runtime Temporal Vision State Engine（实时视觉事实状态机）。

原则：
  - 连续帧 → 字段级状态机 → 事件总线 → 状态快照；全链路可追溯；
  - 时序复用只在"ROI 完全 unchanged"时允许（source=temporal_reuse +
    原 evidence frame）；ROI changed / recognition uncertain → 不复用；
  - Board 转换只接受 0→3→4→5 视觉牌数递增；异常跳变 → STATE_CONFLICT；
  - 事件只允许视觉事实白名单（禁止行为归因事件）；
  - 无 future frames；丢帧/过期 → FRAME_STALE / STALE，不得伪装正常；
  - 部分字段缺失 → PARTIAL_VALID_STATE（不崩溃）；冲突字段单独阻断。
"""
from __future__ import annotations

import hashlib
import time as _time
from dataclasses import dataclass, field

#: 字段级状态（冻结枚举）
FIELD_STATUSES = ("STABLE", "CHANGED", "APPEARED", "DISAPPEARED",
                  "UNCERTAIN", "STALE")
#: 视觉事件白名单（冻结；行为归因事件不存在于此枚举）
EVENT_KINDS = ("FRAME_CHANGED", "ROI_CHANGED", "ROI_APPEARED",
               "ROI_DISAPPEARED", "CARD_APPEARED", "CARD_CHANGED",
               "STACK_CHANGED", "BET_CHANGED", "BUTTON_MOVED",
               "BUTTON_APPEARED", "ACTION_PANEL_APPEARED",
               "ACTION_PANEL_DISAPPEARED", "STATE_CONFLICT",
               "FRAME_STALE", "VISION_STALE", "CAPTURE_FAILURE")


def image_digest(image) -> str:
    """确定性图像摘要（采样哈希；相同帧去重用）。"""
    data = getattr(image, "data", b"") or b""
    step = max(1, len(data) // 4096)
    return hashlib.sha1(data[::step]).hexdigest()


def roi_digest(crop) -> str:
    if crop is None:
        return ""
    return image_digest(crop)


@dataclass
class FieldState:
    name: str
    value: float | str | None = None
    confidence: float | None = None
    source: str | None = None                    # observed | temporal_reuse
    evidence_frame_ids: list = field(default_factory=list)
    temporal_status: str = "UNCERTAIN"
    last_update_frame: int | None = None
    state_version: int = 0
    roi_digest: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value,
                "confidence": self.confidence, "source": self.source,
                "evidence_frame_ids": list(self.evidence_frame_ids),
                "temporal_status": self.temporal_status,
                "last_update_frame": self.last_update_frame,
                "state_version": self.state_version,
                "roi_digest": self.roi_digest}


@dataclass
class VisualEvent:
    kind: str
    frame_id: int
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "frame_id": self.frame_id,
                "detail": dict(self.detail)}


class VisualEventBus:
    """视觉事件总线（白名单 + 行为归因拒绝）。"""

    def __init__(self):
        self.events: list = []

    def emit(self, kind: str, frame_id: int, detail=None) -> None:
        if kind not in EVENT_KINDS:
            raise ValueError(f"事件不在视觉白名单: {kind}")
        self.events.append(VisualEvent(kind, frame_id, detail or {}))

    def drain(self, since: int | None = None) -> list:
        if since is None:
            return list(self.events)
        return [e for e in self.events if e.frame_id > since]

    def to_dict(self) -> list:
        return [e.to_dict() for e in self.events]


class FrameStream:
    """连续帧流：FPS 节流、同帧去重、顺序强制（无 future）、stale 检测。"""

    def __init__(self, fps: float = 2.0, stale_gap_s: float = 5.0):
        self.fps = fps
        self.stale_gap_s = stale_gap_s
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self.last_frame_id: int | None = None
        self.last_digest: str | None = None
        self.last_ts: float | None = None

    def accept(self, frame_id: int, captured_at: float, image,
               *, force: bool = False) -> dict:
        """接收帧 → 处理/丢弃/过期判定（不读 future）。"""
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            return {"accepted": False, "reason": "out_of_order"}
        if self.last_ts is not None and captured_at < self.last_ts - 1e-9:
            return {"accepted": False, "reason": "future_rejected"}
        digest = image_digest(image)
        if not force and digest == self.last_digest:
            return {"accepted": False, "reason": "duplicate_frame"}
        if (self.last_ts is not None
                and captured_at - self.last_ts > self.stale_gap_s):
            return {"accepted": True, "stale": True, "reason": "frame_gap",
                    "gap_s": round(captured_at - self.last_ts, 3)}
        self.last_frame_id = frame_id
        self.last_digest = digest
        self.last_ts = captured_at
        return {"accepted": True, "stale": False}


@dataclass
class TemporalVisionState:
    """逐字段时间状态 + 严格复用规则 + 冲突阻断。"""
    fields: dict = field(default_factory=dict)
    state_version: int = 0
    stale_gap_s: float = 30.0

    def observe(self, name: str, value, confidence, frame_id,
                roi_digest: str | None, source: str = "observed") -> dict:
        prev = self.fields.get(name)
        self.state_version += 1
        if prev is None or prev.temporal_status in ("UNCERTAIN", "STALE",
                                                    "DISAPPEARED"):
            status = "APPEARED"
        elif prev.value == value and prev.source in ("observed",
                                                     "temporal_reuse"):
            status = "STABLE"
        else:
            status = "CHANGED"
        fs = FieldState(name=name, value=value, confidence=confidence,
                        source=source,
                        evidence_frame_ids=[frame_id],
                        temporal_status=status, last_update_frame=frame_id,
                        state_version=self.state_version,
                        roi_digest=roi_digest)
        self.fields[name] = fs
        return fs.to_dict()

    def reuse(self, name: str, frame_id: int, roi_digest: str | None) -> dict:
        """temporal reuse：仅当 ROI digest 与上次一致（视觉区域 unchanged）。"""
        prev = self.fields.get(name)
        if prev is None or prev.source != "observed":
            return {"name": name, "reused": False, "reason": "no_observed"}
        if roi_digest is not None and prev.roi_digest != roi_digest:
            return {"name": name, "reused": False, "reason": "roi_changed"}
        self.state_version += 1
        prev.temporal_status = "STABLE"
        prev.source = "temporal_reuse"
        prev.last_update_frame = frame_id
        return {"name": name, "reused": True, "value": prev.value,
                "source": prev.source,
                "evidence_frame_ids": list(prev.evidence_frame_ids),
                "temporal_status": prev.temporal_status}

    def uncertain(self, name: str, frame_id: int) -> dict:
        """recognition 失败 → UNCERTAIN（不复用旧值）。"""
        prev = self.fields.get(name)
        self.state_version += 1
        fs = FieldState(name=name, value=None, confidence=None,
                        source=None, evidence_frame_ids=[frame_id],
                        temporal_status="UNCERTAIN",
                        last_update_frame=frame_id,
                        state_version=self.state_version,
                        roi_digest=(prev.roi_digest if prev else None))
        self.fields[name] = fs
        return fs.to_dict()

    def conflict(self, name: str, frame_id: int, detail=None) -> dict:
        """STATE_CONFLICT：字段阻断（value=None + 冲突状态）。"""
        self.state_version += 1
        fs = FieldState(name=name, value=None, confidence=None,
                        source="conflict_blocked",
                        evidence_frame_ids=[frame_id],
                        temporal_status="UNCERTAIN",
                        last_update_frame=frame_id,
                        state_version=self.state_version)
        self.fields[name] = fs
        return {"field": name, "blocked": True, "detail": detail or {}}

    def stale_check(self, frame_id: int) -> list:
        """字段级 STALE（超过 stale_gap 帧未更新）。"""
        out = []
        for name, fs in self.fields.items():
            if (fs.last_update_frame is not None
                    and frame_id - fs.last_update_frame > self.stale_gap_s):
                fs.temporal_status = "STALE"
                out.append(name)
        return out

    def to_dict(self) -> dict:
        return {"state_version": self.state_version,
                "fields": {n: f.to_dict() for n, f in self.fields.items()}}


def board_transition(prev_count: int | None, new_count: int) -> dict:
    """Board 视觉牌数转换（0→3→4→5）；异常跳变 → STATE_CONFLICT。"""
    if new_count not in (0, 3, 4, 5):
        return {"ok": False, "conflict": True, "reason": "invalid_count"}
    if prev_count is None:
        return {"ok": True, "conflict": False}
    if new_count == prev_count:
        return {"ok": True, "conflict": False}
    if prev_count == 0 and new_count == 3:
        return {"ok": True, "conflict": False}
    if prev_count == 3 and new_count == 4:
        return {"ok": True, "conflict": False}
    if prev_count == 4 and new_count == 5:
        return {"ok": True, "conflict": False}
    return {"ok": False, "conflict": True,
            "reason": f"board_jump_{prev_count}->{new_count}"}


@dataclass
class RealtimePokerVisionState:
    """实时状态快照（部分有效合法；字段带 provenance + temporal 状态）。"""
    timestamp: float = 0.0
    frame_id: int | None = None
    state_version: int = 0
    vision_status: str = "PARTIAL_VALID_STATE"
    table: dict = field(default_factory=dict)
    street: dict = field(default_factory=dict)
    board: dict = field(default_factory=dict)
    hero_cards: dict = field(default_factory=dict)
    pot: dict = field(default_factory=dict)
    players: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "frame_id": self.frame_id,
                "state_version": self.state_version,
                "vision_status": self.vision_status,
                "table": self.table, "street": self.street,
                "board": self.board, "hero_cards": self.hero_cards,
                "pot": self.pot, "players": self.players}
