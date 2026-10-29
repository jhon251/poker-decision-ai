"""逐字段 provenance（每个观测值都可追溯：来自哪个 ROI / 哪个识别层 /
哪个 frame / 什么置信度）。

provenance 只记录事实来源，不做任何推断与修正。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .coords import Rect


@dataclass(frozen=True)
class FieldProvenance:
    field: str                     # 字段路径，如 "pot" / "hero.hole_cards"
    value: Any = None
    source_layer: str = "none"     # ocr | card_recognition | multimodal |
                                   # bootstrap | reuse | derived | unavailable
    roi: str | None = None         # 语义 ROI 名（哪块区域）
    bbox: Rect | None = None       # client 物理px 坐标
    frame_id: int | None = None
    confidence: float | None = None
    status: str = "recognized"     # recognized | uncertain | missing | reused

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "source_layer": self.source_layer,
            "roi": self.roi,
            "bbox": None if self.bbox is None else self.bbox.to_dict(),
            "frame_id": self.frame_id,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class ProvenanceMap:
    entries: dict = None

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = {}

    def add(self, entry: FieldProvenance) -> None:
        self.entries[entry.field] = entry

    def get(self, field: str) -> FieldProvenance | None:
        return self.entries.get(field)

    def to_dict(self) -> dict:
        return {f: e.to_dict() for f, e in self.entries.items()}
