"""PixelVisionEngine —— V2 像素核心引擎（V2A 基础 + V2B 对象层扩展）。

输入：Frame + 已解析 ROI + 已裁剪 crops（与 V1 ROIManager 同类型，直接
对接）；输出：PixelFrameResult（逐 ROI RawPixelObservation + PixelEvent）。

V2B 扩展（本阶段）：几何特征（连通域/边缘密度）、区域分割候选、
模板库匹配（TEMPLATE_MATCHED）、形状变化（SHAPE_CHANGED）——均只描述
像素事实，不识别牌局语义、不推断行动者与意图。

边界：
  - 只算像素/几何/模板事实；不做行动/意图/决策推断；
  - 未知 = null（首帧无基准 → change 字段 null，不是 0）；
  - 事件必须携带 evidence_frame_ids；禁止引用未来帧；
  - synthetic_test_only 从 Frame 继承（synthetic 与 real 严格区分）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..realtime.coords import Rect
from ..realtime.frame_manager import Frame
from ..realtime.image import ImageData, to_gray
from .events import (PixelEvent, PixelEventKind, PixelFrameResult,
                     RawPixelObservation)
from .geometry import connected_components, edge_density, threshold_mask
from .segmentation import classify_components
from .stats import (changed_bbox, changed_pixel_count, changed_pixel_ratio,
                    color_distance, dominant_color, hsv_stats, rgb_stats)
from .templates import TemplateLibrary, match_library
from .temporal import PixelTemporalConfig, PixelTemporalTracker


@dataclass
class PixelEngineConfig:
    change_threshold: int = 12          # 逐通道差阈值（0-255）
    min_change_ratio: float = 0.005     # 低于此比率视为未变化
    color_change_threshold: float = 30.0  # 主色 RGB 欧氏距离阈值
    geometry: bool = True               # V2B：几何特征
    segmentation: bool = True           # V2B：区域分割候选
    template_library: TemplateLibrary | None = None   # V2B：模板库
    template_min_score: float = 0.7
    shape_change_eps_components: int = 1      # 组件数变化 ≥ 该值 → SHAPE_CHANGED
    shape_change_eps_edge: float = 0.05       # 边缘密度变化 ≥ 该值 → SHAPE_CHANGED
    temporal: PixelTemporalConfig = field(default_factory=PixelTemporalConfig)


#: ROI 语义名 → 区域级事实事件（只表示"该区域像素变化"，无语义推断）
_REGION_EVENT = {
    "pot": PixelEventKind.POT_REGION_CHANGED,
    "hero_cards": PixelEventKind.CARD_REGION_CHANGED,
    "board": PixelEventKind.CARD_REGION_CHANGED,
    "stacks": PixelEventKind.STACK_REGION_CHANGED,
    "action_area": PixelEventKind.TEXT_REGION_CHANGED,
    "table_meta": PixelEventKind.TEXT_REGION_CHANGED,
}


class PixelVisionEngine:
    def __init__(self, config: PixelEngineConfig | None = None):
        self.config = config or PixelEngineConfig()
        self.tracker = PixelTemporalTracker(self.config.temporal)
        self._prev_frame_id: int | None = None
        self._prev_crops: dict = {}
        self._last_present_crops: dict = {}
        self._prev_presence: dict = {}
        self._prev_dominant: dict = {}
        self._prev_geo: dict = {}
        self._geo_cache: dict = {}

    def reset(self) -> None:
        self.tracker.reset()
        self._prev_frame_id = None
        self._prev_crops = {}
        self._last_present_crops = {}
        self._prev_presence = {}
        self._prev_dominant = {}
        self._prev_geo = {}
        self._geo_cache = {}

    # ------------------------------------------------------------ 主入口
    def process(self, frame: Frame, rois: dict, crops: dict,
                missing: dict | None = None) -> PixelFrameResult:
        missing = missing or {}
        prev_id = self._prev_frame_id
        result = PixelFrameResult(frame_id=frame.frame_id,
                                  captured_at=frame.captured_at,
                                  synthetic_test_only=frame.synthetic_test_only)
        observations: dict = {}
        events: list = []
        presence: dict = {}
        changed_flags: dict = {}
        bbox_rects: dict = {}

        for name in sorted(rois, key=lambda r: r.value):
            roi = rois[name]
            crop = crops.get(name)
            obs, evs, changed_flag, changed_rect = self._observe(
                name, roi, crop, frame, missing.get(name), prev_id)
            observations[name.value] = obs
            events.extend(evs)
            presence[name.value] = crop is not None
            changed_flags[name.value] = changed_flag
            bbox_rects[name.value] = changed_rect

        temporal_features, temporal_events = self.tracker.update(
            frame.frame_id, presence, changed_flags, bbox_rects)
        for roi_name, feats in temporal_features.items():
            obs = observations.get(roi_name)
            if obs is not None:
                obs.temporal_features = feats
        events.extend(temporal_events)

        # 状态前进（只向未来，禁止回读未来帧）
        self._prev_frame_id = frame.frame_id
        self._prev_crops = {k.value: v for k, v in crops.items()}
        self._prev_presence = dict(presence)
        self._prev_dominant = {
            roi_name: obs.color_features.get("dominant_rgb")
            for roi_name, obs in observations.items()
            if obs.color_features and "dominant_rgb" in obs.color_features}
        self._prev_geo = {
            roi_name: {"component_count": obs.geometry_features["component_count"],
                       "edge_density": obs.geometry_features["edge_density"]}
            for roi_name, obs in observations.items()
            if obs.geometry_features
            and "component_count" in obs.geometry_features}
        for k, v in crops.items():
            self._last_present_crops[k.value] = v

        result.observations = observations
        result.events = tuple(events)
        return result

    # ------------------------------------------------------------ 单 ROI
    def _observe(self, name, roi, crop, frame, missing_reason, prev_id):
        roi_name = name.value
        bbox = roi.rect()
        obs = RawPixelObservation(
            frame_id=frame.frame_id, captured_at=frame.captured_at,
            roi=roi_name, coordinate_system="client", bbox=bbox.to_dict(),
            source="pixel_engine", synthetic_test_only=frame.synthetic_test_only,
            uncertainty=[])
        events: list = []

        if crop is None:
            obs.uncertainty.append({"field": "color_features", "status": "missing",
                                    "confidence": None})
            if missing_reason:
                obs.uncertainty.append({"field": "roi_crop", "status": "missing",
                                        "confidence": None,
                                        "reason": missing_reason})
            return obs, events, None, None

        base_crop = self._prev_crops.get(roi_name) \
            or self._last_present_crops.get(roi_name)

        changed_flag = False
        changed_rect: Rect | None = None
        if base_crop is None:
            # 首帧：无比较基准 → 变化字段 null（不填 0）
            obs.uncertainty.append({"field": "pixel_change_ratio",
                                    "status": "missing", "confidence": None,
                                    "reason": "no_previous_frame"})
            obs.uncertainty.append({"field": "changed_bbox", "status": "missing",
                                    "confidence": None,
                                    "reason": "no_previous_frame"})
        else:
            ratio = changed_pixel_ratio(base_crop, crop,
                                        self.config.change_threshold)
            if ratio is None:
                obs.uncertainty.append({"field": "pixel_change_ratio",
                                        "status": "missing", "confidence": None,
                                        "reason": "geometry_mismatch"})
                obs.uncertainty.append({"field": "changed_bbox", "status": "missing",
                                        "confidence": None,
                                        "reason": "geometry_mismatch"})
            else:
                obs.pixel_change_ratio = round(ratio, 6)
                obs.changed_pixel_count = changed_pixel_count(
                    base_crop, crop, self.config.change_threshold)
                rect = changed_bbox(base_crop, crop, self.config.change_threshold)
                if rect is not None:
                    obs.changed_bbox = rect.to_dict()
                    changed_rect = rect
                changed_flag = ratio >= self.config.min_change_ratio
                if changed_flag:
                    evidence = (prev_id, frame.frame_id) \
                        if prev_id is not None else (frame.frame_id,)
                    events.append(PixelEvent(
                        kind=PixelEventKind.PIXEL_CHANGED, roi=roi_name,
                        frame_id=frame.frame_id, timestamp=frame.captured_at,
                        bbox=bbox.to_dict(), changed_bbox=obs.changed_bbox,
                        pixel_change_ratio=obs.pixel_change_ratio,
                        evidence_frame_ids=evidence,
                        description=f"{roi_name} ROI changed "
                                    f"(ratio {obs.pixel_change_ratio})"))
                    region_kind = _REGION_EVENT.get(roi_name)
                    if region_kind is not None:
                        events.append(PixelEvent(
                            kind=region_kind, roi=roi_name,
                            frame_id=frame.frame_id, timestamp=frame.captured_at,
                            bbox=bbox.to_dict(), changed_bbox=obs.changed_bbox,
                            pixel_change_ratio=obs.pixel_change_ratio,
                            evidence_frame_ids=evidence,
                            description=f"{roi_name} region changed"))

        # 颜色特征（真实像素统计，无语义）
        rgb = rgb_stats(crop)
        hsv = hsv_stats(crop)
        dom = dominant_color(crop)
        obs.color_features = {
            "rgb_mean": rgb["mean_rgb"] if rgb else None,
            "rgb_std": rgb["std_rgb"] if rgb else None,
            "hsv_mean": hsv["mean_hsv"] if hsv else None,
            "dominant_rgb": dom["dominant_rgb"] if dom else None,
            "dominant_ratio": dom["dominant_ratio"] if dom else None,
        }
        if dom is None or rgb is None or hsv is None:
            obs.uncertainty.append({"field": "color_features", "status": "missing",
                                    "confidence": None})

        # 主色变化（仅当上一帧该 ROI 也在场时比较——跨缺席不比较）
        prev_dom = self._prev_dominant.get(roi_name)
        if changed_flag and prev_dom is not None and dom is not None \
                and self._prev_presence.get(roi_name):
            dist = color_distance({"dominant_rgb": prev_dom}, dom)
            if dist > self.config.color_change_threshold:
                evidence = (prev_id, frame.frame_id) \
                    if prev_id is not None else (frame.frame_id,)
                events.append(PixelEvent(
                    kind=PixelEventKind.COLOR_CHANGED, roi=roi_name,
                    frame_id=frame.frame_id, timestamp=frame.captured_at,
                    bbox=bbox.to_dict(), changed_bbox=obs.changed_bbox,
                    evidence_frame_ids=evidence,
                    description=f"{roi_name} dominant color changed "
                                f"(distance {dist:.1f})"))

        # V2B：几何特征 + 分割候选 + 模板匹配（只描述像素事实）
        comps = []
        if self.config.geometry or self.config.segmentation:
            gray = to_gray(crop)
            if changed_flag or self._geo_cache.get(roi_name) is None:
                mask = threshold_mask(gray)
                comps = connected_components(mask, min_area=4)
                ed = edge_density(gray)
                geo = {"edge_density": round(ed, 6),
                       "component_count": len(comps),
                       "components": [c.to_dict() for c in comps[:12]]}
                self._geo_cache[roi_name] = geo
            else:
                geo = self._geo_cache.get(roi_name)
            obs.geometry_features = geo
            prev_geo = self._prev_geo.get(roi_name)
            if changed_flag and prev_geo is not None and geo is not None:
                d_count = abs(geo["component_count"]
                              - prev_geo["component_count"])
                d_edge = abs(geo["edge_density"] - prev_geo["edge_density"])
                if d_count >= self.config.shape_change_eps_components \
                        or d_edge >= self.config.shape_change_eps_edge:
                    evidence = (prev_id, frame.frame_id) \
                        if prev_id is not None else (frame.frame_id,)
                    events.append(PixelEvent(
                        kind=PixelEventKind.SHAPE_CHANGED, roi=roi_name,
                        frame_id=frame.frame_id, timestamp=frame.captured_at,
                        bbox=bbox.to_dict(), changed_bbox=obs.changed_bbox,
                        evidence_frame_ids=evidence,
                        description=f"{roi_name} shape changed "
                                    f"(components {prev_geo['component_count']}"
                                    f" -> {geo['component_count']})"))
            if self.config.segmentation:
                obs.segmentation = classify_components(comps, gray)

        if self.config.template_library is not None:
            matches = match_library(crop, self.config.template_library,
                                    roi_hint=roi_name,
                                    score_threshold=self.config.template_min_score)
            if matches:
                top = matches[0]
                obs.template_id = top.template_id
                obs.template_score = round(top.score, 4)
                evidence = (prev_id, frame.frame_id) \
                    if prev_id is not None else (frame.frame_id,)
                for m in matches[:3]:
                    events.append(PixelEvent(
                        kind=PixelEventKind.TEMPLATE_MATCHED, roi=roi_name,
                        frame_id=frame.frame_id, timestamp=frame.captured_at,
                        bbox=m.bbox, changed_bbox=obs.changed_bbox,
                        evidence_frame_ids=evidence,
                        confidence=round(m.score, 4),
                        description=f"{roi_name} matched template "
                                    f"{m.template_id} (score {m.score:.3f})"))

        obs.evidence_frame_ids = ((prev_id, frame.frame_id)
                                  if prev_id is not None
                                  else (frame.frame_id,))
        return obs, events, changed_flag, changed_rect
