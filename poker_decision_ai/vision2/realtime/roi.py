"""ROI 层：只回答「在哪里看」，绝不判断牌面/数字含义。

每个 ROI 必含契约字段：semantic_name / coordinate_system / x / y / width /
height / confidence / source / calibration_version。

解析规则（确定性，禁止猜测）：
  - ROI = anchor(归一化) + offset(归一化偏移)；anchor-relative 主方案；
  - 结果必须完整落在客户区内，否则 ROI_MISSING（不裁剪越界区域）；
  - anchor 缺失 → ROI_MISSING；零面积 → ROI_MISSING。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .anchors import CalibrationProfile
from .coords import (CoordinateSystem, CoordinateTransformer, Rect)
from .image import ImageData


class ROIName(str, Enum):
    HERO_CARDS = "hero_cards"
    BOARD = "board"
    POT = "pot"
    ACTION_AREA = "action_area"
    BET_AREA = "bet_area"
    STACKS = "stacks"
    PLAYERS = "players"
    TABLE_META = "table_meta"


@dataclass(frozen=True)
class ROISpec:
    semantic_name: ROIName
    anchor: str                     # 相对 anchor_id（anchor-relative）
    offset: Rect                    # normalized 偏移（相对 anchor rect）
    required: bool = True
    read_layer: str = "none"        # none | number | text | seats |
                                    # seat_stacks | table_meta | cards


@dataclass(frozen=True)
class ROI:
    """已解析的 ROI 实例（client 物理px 坐标）。只负责位置，不负责含义。"""
    semantic_name: ROIName
    coordinate_system: CoordinateSystem
    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: str                     # bootstrap | calibrated
    calibration_version: int

    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    def to_dict(self) -> dict:
        return {
            "semantic_name": self.semantic_name.value,
            "coordinate_system": self.coordinate_system.value,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "confidence": self.confidence,
            "source": self.source,
            "calibration_version": self.calibration_version,
        }


class ROIManager:
    def __init__(self, profile: CalibrationProfile,
                 specs: dict, transformer: CoordinateTransformer | None = None):
        self.profile = profile
        self.specs = specs
        self.transformer = transformer or CoordinateTransformer(profile.geometry)

    def resolve(self) -> tuple:
        """返回 (resolved: dict[ROIName, ROI], missing: dict[ROIName, str])。"""
        resolved: dict = {}
        missing: dict = {}
        for name, spec in sorted(self.specs.items(), key=lambda kv: kv[0].value):
            anchor = self.profile.anchor(spec.anchor)
            if anchor is None:
                missing[name] = f"anchor_missing:{spec.anchor}"
                continue
            # offset 为 anchor 尺寸相对量：ROI = anchor 原点 + offset × anchor 尺寸
            rn = Rect(anchor.rect_norm.x + spec.offset.x * anchor.rect_norm.width,
                      anchor.rect_norm.y + spec.offset.y * anchor.rect_norm.height,
                      spec.offset.width * anchor.rect_norm.width,
                      spec.offset.height * anchor.rect_norm.height)
            unit = Rect(0.0, 0.0, 1.0, 1.0)
            if not unit.contains(rn.topleft) or not unit.contains(rn.bottomright):
                missing[name] = "out_of_bounds_normalized"
                continue
            rc = self.transformer.transform(rn, CoordinateSystem.NORMALIZED,
                                            CoordinateSystem.CLIENT)
            min_px = 2.0
            if rc.width < min_px or rc.height < min_px:
                missing[name] = "too_small"
                continue
            resolved[name] = ROI(
                semantic_name=name,
                coordinate_system=CoordinateSystem.CLIENT,
                x=rc.x, y=rc.y, width=rc.width, height=rc.height,
                confidence=anchor.confidence,
                source=anchor.source,
                calibration_version=anchor.calibration_version,
            )
        return resolved, missing

    def extract(self, image: ImageData, roi: ROI) -> ImageData | None:
        """裁剪 ROI 像素；越界 → None（调用方记 ROI_MISSING，不猜测）。"""
        return image.crop(roi.rect())

    def extract_all(self, image: ImageData, rois: dict) -> tuple:
        crops: dict = {}
        missing: dict = {}
        for name, roi in sorted(rois.items(), key=lambda kv: kv[0].value):
            crop = self.extract(image, roi)
            if crop is None:
                missing[name] = "out_of_bounds_pixels"
            else:
                crops[name] = crop
        return crops, missing


def default_roi_specs() -> dict:
    """bootstrap 种子 ROI（anchor="table" 的归一化偏移；仅初始值）。

    校准后应替换为真实 anchor-relative 布局；固定坐标不是长期稳定坐标。
    """
    return {
        ROIName.HERO_CARDS: ROISpec(ROIName.HERO_CARDS, "table",
                                    Rect(0.42, 0.75, 0.16, 0.10),
                                    required=True, read_layer="cards"),
        ROIName.BOARD: ROISpec(ROIName.BOARD, "table",
                               Rect(0.35, 0.30, 0.30, 0.18),
                               required=True, read_layer="cards"),
        ROIName.POT: ROISpec(ROIName.POT, "table",
                             Rect(0.35, 0.18, 0.30, 0.08),
                             required=True, read_layer="number"),
        ROIName.ACTION_AREA: ROISpec(ROIName.ACTION_AREA, "table",
                                     Rect(0.55, 0.55, 0.25, 0.15),
                                     required=False, read_layer="text"),
        ROIName.BET_AREA: ROISpec(ROIName.BET_AREA, "table",
                                  Rect(0.42, 0.88, 0.16, 0.08),
                                  required=False, read_layer="number"),
        ROIName.STACKS: ROISpec(ROIName.STACKS, "table",
                                Rect(0.08, 0.35, 0.20, 0.30),
                                required=False, read_layer="seat_stacks"),
        ROIName.PLAYERS: ROISpec(ROIName.PLAYERS, "table",
                                 Rect(0.08, 0.15, 0.84, 0.25),
                                 required=False, read_layer="seats"),
        ROIName.TABLE_META: ROISpec(ROIName.TABLE_META, "table",
                                    Rect(0.85, 0.05, 0.12, 0.15),
                                    required=False, read_layer="table_meta"),
    }
