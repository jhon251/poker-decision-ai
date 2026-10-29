"""Anchor 与 Calibration Profile（anchor-relative 主方案）。

- 主方案：ROI 相对 anchor 的归一化偏移（anchor 随窗口移动，ROI 跟随）。
- 固定坐标仅作 bootstrap 种子（source="bootstrap"，低置信）。
- 校准后所有 anchor 以 normalized（客户区相对）存储；使用帧几何反算。
- anchor 定位失败/漂移 → 触发 recalibration 或 LAYOUT_UNKNOWN；禁止猜测 ROI。
"""
from __future__ import annotations

from dataclasses import dataclass

from .coords import (CoordinateTransformer, GeometryContext, Rect)


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    semantic_name: str
    rect_norm: Rect                 # 规范存储：normalized（客户区相对）
    source: str = "bootstrap"       # bootstrap | calibrated | manual
    confidence: float = 0.3
    calibration_version: int = 1

    def client_rect(self, transformer: CoordinateTransformer) -> Rect:
        from .coords import CoordinateSystem
        return transformer.transform(self.rect_norm, CoordinateSystem.NORMALIZED,
                                     CoordinateSystem.CLIENT)

    def to_dict(self) -> dict:
        return {"anchor_id": self.anchor_id,
                "semantic_name": self.semantic_name,
                "rect_norm": self.rect_norm.to_dict(),
                "source": self.source,
                "confidence": self.confidence,
                "calibration_version": self.calibration_version}


@dataclass(frozen=True)
class CalibrationProfile:
    profile_id: str
    geometry: GeometryContext       # 校准时刻的几何（漂移检测基线）
    anchors: tuple
    source: str = "bootstrap"       # bootstrap | calibrated
    calibrated_at: str | None = None
    calibration_version: int = 1

    def anchor(self, anchor_id: str) -> Anchor | None:
        for a in self.anchors:
            if a.anchor_id == anchor_id:
                return a
        return None

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "geometry": self.geometry.to_dict(),
            "anchors": [a.to_dict() for a in self.anchors],
            "source": self.source,
            "calibrated_at": self.calibrated_at,
            "calibration_version": self.calibration_version,
        }


def calibrate(profile_id: str, geometry: GeometryContext,
              detections: dict, calibrated_at: str | None = None,
              base_version: int = 1) -> CalibrationProfile:
    """已检测 anchor（client 物理px rect + 匹配分数）→ 校准 profile。

    detections: {anchor_id: (Rect, score)}。所有 rect 转 normalized 存储。
    只转换已检测项；缺失 anchor 由调用方处理（LAYOUT_UNKNOWN，不猜测）。
    """
    tr = CoordinateTransformer(geometry)
    from .coords import CoordinateSystem
    anchors = []
    for anchor_id, (rect, score) in sorted(detections.items()):
        rn = tr.transform(rect, CoordinateSystem.CLIENT,
                          CoordinateSystem.NORMALIZED)
        anchors.append(Anchor(anchor_id=anchor_id, semantic_name=anchor_id,
                              rect_norm=rn, source="calibrated",
                              confidence=float(score),
                              calibration_version=base_version))
    return CalibrationProfile(profile_id=profile_id, geometry=geometry,
                              anchors=tuple(anchors), source="calibrated",
                              calibrated_at=calibrated_at,
                              calibration_version=base_version)


def bootstrap_profile(profile_id: str, geometry: GeometryContext,
                      bootstrap_anchors: dict, calibrated_at: str | None = None
                      ) -> CalibrationProfile:
    """固定坐标 bootstrap 种子（低置信；仅初始值，不是长期稳定坐标）。

    bootstrap_anchors: {anchor_id: (x, y, w, h)} 全部 normalized。
    """
    anchors = tuple(
        Anchor(anchor_id=k, semantic_name=k,
               rect_norm=Rect(v[0], v[1], v[2], v[3]),
               source="bootstrap", confidence=0.3, calibration_version=1)
        for k, v in sorted(bootstrap_anchors.items())
    )
    return CalibrationProfile(profile_id=profile_id, geometry=geometry,
                              anchors=anchors, source="bootstrap",
                              calibrated_at=calibrated_at,
                              calibration_version=1)


def detect_anchor_drift(profile: CalibrationProfile,
                        new_geometry: GeometryContext) -> bool:
    """几何变化（窗口移动/缩放/分辨率/DPI 改变）→ 需要 recalibration。"""
    g0, g1 = profile.geometry, new_geometry
    if g0.window_rect is None or g1.window_rect is None:
        return g0.screen_size != g1.screen_size \
            or g0.client_size_physical != g1.client_size_physical \
            or g0.dpi_scale != g1.dpi_scale
    return (g0.window_rect.to_dict() != g1.window_rect.to_dict()
            or g0.client_offset != g1.client_offset
            or g0.client_size_physical != g1.client_size_physical
            or g0.dpi_scale != g1.dpi_scale)


def anchors_missing(profile: CalibrationProfile, required_ids) -> list:
    return [aid for aid in required_ids if profile.anchor(aid) is None]
