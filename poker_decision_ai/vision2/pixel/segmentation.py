"""区域分割（启发式候选，非语义识别）。

基于大津二值化 + 连通域，把组件按启发式规则分为候选类：
  card_candidates / text_candidates / chip_candidates。

硬约束：候选只是"几何上像"的区域，不意味着"就是牌/文字/筹码"；
confidence 一律 null（未校准）；任何下游语义识别必须走独立证据。
"""
from __future__ import annotations

from ..realtime.image import GrayImage
from .geometry import (Component, connected_components, otsu_threshold,
                       threshold_mask)


def segment_components(gray: GrayImage, *, min_area: int = 4,
                       threshold: int | None = None,
                       invert: bool = False) -> list:
    """二值化 → 连通域 → 组件列表（面积降序）。"""
    mask = threshold_mask(gray, threshold, invert=invert)
    return connected_components(mask, min_area=min_area)


def classify_components(comps: list, gray: GrayImage | None = None
                        ) -> dict:
    """启发式候选分类。返回 {class: [{bbox, area, scores...}]}。

    规则（启发式，仅候选）：
      - card_candidates: 面积前 5 大且宽高比 0.35–1.1、填充率 ≥ 0.4；
      - text_candidates: 面积 ≥ 4、宽度 ≤ 40% 图宽、高度 ≤ 30% 图高、
        细长（宽高比 ≥ 1.2 或 ≤ 0.83）；
      - chip_candidates: 圆似度 ≥ 0.55。
    组件可同时落入多个候选类；confidence 一律 null。
    """
    out = {"card_candidates": [], "text_candidates": [],
           "chip_candidates": [], "component_count": len(comps)}
    img_w = gray.width if gray is not None else None
    img_h = gray.height if gray is not None else None
    for c in comps:
        entry = {"bbox": c.bbox.to_dict(), "area": c.area,
                 "aspect_ratio": round(c.aspect_ratio, 4),
                 "fill_ratio": round(c.fill_ratio, 4),
                 "confidence": None}
        if 0.35 <= c.aspect_ratio <= 1.1 and c.fill_ratio >= 0.4:
            out["card_candidates"].append(entry)
        if img_w is not None and c.bbox.width <= 0.4 * img_w \
                and img_h is not None and c.bbox.height <= 0.3 * img_h \
                and c.area >= 4:
            out["text_candidates"].append(entry)
        if c.circle_score >= 0.55:
            out["chip_candidates"].append(entry)
    return out
