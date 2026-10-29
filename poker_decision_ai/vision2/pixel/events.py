"""Pixel Vision Core —— 事实性像素事件与 RawPixelObservation 契约。

null 纪律（继承 V0，硬约束）：
  - 所有观测字段默认 None；未知 = null，禁用 0/[]/""/false 填充；
  - 只有"已计算"的数值才有值（首帧无比较基准时 pixel_change_ratio 为
    null，不是 0）。

事件白名单（事实性）：只描述"像素/区域发生了什么"，禁止行动者与行动
语义（"某人下注"类）、禁止意图推断。SHAPE_CHANGED 与 TEMPLATE_MATCHED
在几何层/模板层启用前恒不产出（枚举预留，不伪造）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PixelEventKind(str, Enum):
    PIXEL_CHANGED = "PIXEL_CHANGED"
    ROI_APPEARED = "ROI_APPEARED"
    ROI_DISAPPEARED = "ROI_DISAPPEARED"
    COLOR_CHANGED = "COLOR_CHANGED"
    SHAPE_CHANGED = "SHAPE_CHANGED"          # 几何层启用前恒不产出
    TEMPLATE_MATCHED = "TEMPLATE_MATCHED"    # 模板层启用前恒不产出
    TEXT_REGION_CHANGED = "TEXT_REGION_CHANGED"
    CARD_REGION_CHANGED = "CARD_REGION_CHANGED"
    STACK_REGION_CHANGED = "STACK_REGION_CHANGED"
    POT_REGION_CHANGED = "POT_REGION_CHANGED"


#: 事件白名单（冻结）；任何其它 kind 一律拒绝
ALL_PIXEL_EVENT_KINDS = tuple(k for k in PixelEventKind)


@dataclass(frozen=True)
class PixelEvent:
    """单条像素事实事件（必须携带证据帧；无语义推断）。"""
    kind: PixelEventKind
    roi: str | None = None
    frame_id: int | None = None
    timestamp: float | None = None
    bbox: dict | None = None
    changed_bbox: dict | None = None
    pixel_change_ratio: float | None = None
    evidence_frame_ids: tuple | None = None
    confidence: float | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "roi": self.roi,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "bbox": self.bbox,
            "changed_bbox": self.changed_bbox,
            "pixel_change_ratio": self.pixel_change_ratio,
            "evidence_frame_ids": None if self.evidence_frame_ids is None
            else list(self.evidence_frame_ids),
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class RawPixelObservation:
    """单 ROI 的像素观测（所有字段默认 null；未知不填值）。"""
    frame_id: int | None = None
    captured_at: float | None = None
    roi: str | None = None
    coordinate_system: str | None = None     # 恒为 "client"（物理px）
    bbox: dict | None = None                 # {x,y,width,height}
    pixel_change_ratio: float | None = None
    changed_bbox: dict | None = None
    changed_pixel_count: int | None = None
    template_id: str | None = None           # 模板层启用前恒 null
    template_score: float | None = None
    color_features: dict | None = None
    geometry_features: dict | None = None    # 几何层启用前恒 null
    temporal_features: dict | None = None
    segmentation: dict | None = None         # 分割层启用前恒 null
    source: str | None = None
    confidence: float | None = None          # 未校准 → null（诚实）
    evidence_frame_ids: tuple | None = None
    synthetic_test_only: bool | None = None
    uncertainty: list | None = None          # [{field, status, confidence, reason?}]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "captured_at": self.captured_at,
            "roi": self.roi,
            "coordinate_system": self.coordinate_system,
            "bbox": self.bbox,
            "pixel_change_ratio": self.pixel_change_ratio,
            "changed_bbox": self.changed_bbox,
            "changed_pixel_count": self.changed_pixel_count,
            "template_id": self.template_id,
            "template_score": self.template_score,
            "color_features": self.color_features,
            "geometry_features": self.geometry_features,
            "temporal_features": self.temporal_features,
            "segmentation": self.segmentation,
            "source": self.source,
            "confidence": self.confidence,
            "evidence_frame_ids": None if self.evidence_frame_ids is None
            else list(self.evidence_frame_ids),
            "synthetic_test_only": self.synthetic_test_only,
            "uncertainty": self.uncertainty,
        }


@dataclass
class PixelFrameResult:
    """一帧的像素层输出：每 ROI 一条 observation + 事实性事件序列。"""
    frame_id: int | None = None
    captured_at: float | None = None
    synthetic_test_only: bool | None = None
    observations: dict = field(default_factory=dict)   # roi_name -> RawPixelObservation
    events: tuple = ()

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "captured_at": self.captured_at,
            "synthetic_test_only": self.synthetic_test_only,
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "events": [e.to_dict() for e in self.events],
        }

    def events_of(self, kind: PixelEventKind) -> tuple:
        return tuple(e for e in self.events if e.kind is kind)
