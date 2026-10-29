"""V2C — 真实客户端候选资产提取（确定性；复用 V2B 分割/几何，不新建核心）。

从真实截图提取**未标注候选**：
  - digit_glyph：pot/bet_area/stacks/table_meta 的字符单元（列投影 run）；
  - card_glyph：hero_cards/board 卡槽内的内部字形组件；
  - card_slot：整卡槽裁剪；
  - disc_candidate：圆似度 ≥ 阈值的组件（庄家钮候选）；
  - region_snapshot：hero_cards/board 区域快照。

全部产出 ground_truth_status="unlabeled_candidate"、semantic_label=None——
**绝不把候选标记为已识别**。
"""
from __future__ import annotations

from ..digit import segment_runs
from ..geometry import connected_components, threshold_mask
from ..templates import foreground_is_dark, invert_image
from ...realtime.coords import Rect
from ...realtime.image import ImageData, to_gray
from .assets import RealClientTemplate

_DIGIT_ROIS = ("pot", "bet_area", "stacks", "table_meta")
_CARD_ROIS = ("hero_cards", "board")
_DISC_ROIS = ("players", "table_meta", "action_area")
_MIN_GLYPH_AREA = 6
_MIN_DISC_AREA = 40
_MAX_DISC_AREA = 40000


def _canonicalize(img: ImageData) -> ImageData:
    """极规范化：深字形浅底 → 反色（存储资产恒为亮字形黑底规范形）。

    依赖裁剪自带背景上下文（字形贴边会污染极性判定，故字形裁剪必须
    带 padding）；确定性。
    """
    gray = to_gray(img)
    mask = threshold_mask(gray)
    if foreground_is_dark(mask, gray):
        return invert_image(img)
    return img


def _pad_rect(crop_w: float, crop_h: float, rect: Rect,
              pad_frac: float = 0.3) -> Rect:
    px = max(1.0, rect.width * pad_frac)
    py = max(1.0, rect.height * pad_frac)
    x0 = max(0.0, rect.x - px)
    y0 = max(0.0, rect.y - py)
    x1 = min(crop_w, rect.x + rect.width + px)
    y1 = min(crop_h, rect.y + rect.height + py)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _digit_candidates(frame_id: int, roi_name: str, crop: ImageData,
                      client: str, source_frame: str | None) -> list:
    """数字区 → 字符单元候选（未标注）。"""
    out = []
    if crop is None:
        return out
    runs, mask = segment_runs(crop)
    if not runs:
        return out
    h_mask = len(mask)
    heights = []
    for (s, e) in runs:
        ys = [y for x in range(s, e) for y in range(h_mask) if mask[y][x]]
        heights.append(max(ys) - min(ys) + 1 if ys else 0)
    cell_h = sorted(heights)[len(heights) // 2] if heights else 0
    if cell_h <= 0:
        return out
    all_ys = [y for y in range(h_mask)
              if any(mask[y][x] for x in range(len(mask[0])))]
    line_bottom = max(all_ys) + 1 if all_ys else 0
    for n, ((s, e), hh) in enumerate(zip(runs, heights)):
        if hh <= 0:
            continue
        cell = Rect(float(s), float(max(0, line_bottom - cell_h)),
                    float(e - s), float(cell_h))
        sub = crop.crop(cell)
        if sub is None or sub.width * sub.height < _MIN_GLYPH_AREA:
            continue
        out.append(RealClientTemplate(
            template_id=f"f{frame_id}_{roi_name}_digit_{n:03d}",
            kind="digit_glyph", source_frame=source_frame,
            source_roi=roi_name, client=client,
            crop_bbox=cell.to_dict()))
    return out


def _card_candidates(frame_id: int, roi_name: str, crop: ImageData,
                     client: str, source_frame: str | None) -> list:
    """卡区 → card_slot + 卡内字形候选（未标注）。"""
    from ..cards import card_candidate_slots
    out = []
    if crop is None:
        return out
    slots = card_candidate_slots(crop)
    for n, rect in enumerate(slots):
        sub = crop.crop(rect)
        if sub is None:
            continue
        out.append(RealClientTemplate(
            template_id=f"f{frame_id}_{roi_name}_slot_{n:03d}",
            kind="card_slot", source_frame=source_frame,
            source_roi=roi_name, client=client,
            crop_bbox=rect.to_dict()))
        gray = to_gray(sub)
        mask = threshold_mask(gray)
        comps = connected_components(mask, min_area=4)
        interior = [c for c in comps
                    if c.bbox.x > 0 and c.bbox.y > 0
                    and c.bbox.x + c.bbox.width < sub.width
                    and c.bbox.y + c.bbox.height < sub.height]
        for m, comp in enumerate(interior[:6]):
            padded = _pad_rect(float(sub.width), float(sub.height),
                               comp.bbox)
            g = sub.crop(padded)
            if g is None:
                continue
            out.append(RealClientTemplate(
                template_id=f"f{frame_id}_{roi_name}_glyph_{n:03d}_{m}",
                kind="card_glyph", source_frame=source_frame,
                source_roi=roi_name, client=client,
                crop_bbox={"x": rect.x + padded.x,
                           "y": rect.y + padded.y,
                           "width": padded.width,
                           "height": padded.height}))
    return out


def _disc_candidates(frame_id: int, roi_name: str, crop: ImageData,
                     client: str, source_frame: str | None) -> list:
    """圆似组件候选（庄家钮等）。"""
    out = []
    if crop is None:
        return out
    gray = to_gray(crop)
    mask = threshold_mask(gray)
    for comp in connected_components(mask, min_area=4):
        if not (_MIN_DISC_AREA <= comp.area <= _MAX_DISC_AREA):
            continue
        if comp.circle_score < 0.8:          # 圆似度从严（减少噪声候选）
            continue
        out.append(RealClientTemplate(
            template_id=f"f{frame_id}_{roi_name}_disc_{comp.id:03d}",
            kind="disc_candidate", source_frame=source_frame,
            source_roi=roi_name, client=client,
            crop_bbox=comp.bbox.to_dict(),
            confidence=None))
    return out


def extract_hero_card_glyphs(frame_id: int, crop: ImageData, client: str,
                             source_frame: str | None) -> list:
    """hero 双卡 → rank/suit 字形候选（白卡专用提取，域适配）。

    实测：该客户端 hero 两白卡相邻且被白色光晕包围，通用卡槽启发式会把
    双卡+光晕连成一个组件而失败；此处按"亮 >220 = 卡身、暗 <100 = 卡上
    字形"两级掩码确定性提取（root at client 白卡视觉事实）。
    """
    import numpy as _np
    from ..geometry import connected_components as _cc
    from ...realtime.image import to_gray as _tg

    out = []
    if crop is None:
        return out
    gray = _np.asarray(_tg(crop).data, dtype="uint8")
    # 卡身（白卡）→ 亮度 >220 掩码；卡上字形 → 亮度 <100 暗掩码
    card_m = (_np.zeros(gray.shape, dtype="uint8"))
    card_m[gray > 220] = 1
    comps = _cc(card_m, min_area=500)
    cards = [c for c in comps
             if 0.4 <= c.aspect_ratio <= 1.2 and c.fill_ratio >= 0.55]
    cards.sort(key=lambda c: c.bbox.x)
    for n, card in enumerate(cards[:2]):
        slot = crop.crop(card.bbox)
        if slot is None:
            continue
        g = _np.asarray(_tg(slot).data, dtype="uint8")
        glyph_m = _np.zeros(g.shape, dtype="uint8")
        glyph_m[g < 100] = 1
        for m, comp in enumerate(_cc(glyph_m, min_area=20)[:8]):
            if comp.bbox.x <= 0 or comp.bbox.y <= 0:
                continue
            gsub = slot.crop(comp.bbox)
            if gsub is None or gsub.width < 4 or gsub.height < 4:
                continue
            out.append(RealClientTemplate(
                template_id=f"herochar_f{frame_id}_card{n}_glyph_{m}",
                kind="card_glyph", source_frame=source_frame,
                source_roi="hero_cards", client=client,
                crop_bbox={"x": card.bbox.x + comp.bbox.x,
                           "y": card.bbox.y + comp.bbox.y,
                           "width": comp.bbox.width,
                           "height": comp.bbox.height}))
    return out


def extract_candidates(frame_id: int, crops: dict, client: str,
                       source_frame: str | None = None) -> list:
    """一帧真实截图 → 未标注候选资产列表（确定性顺序）。"""
    out = []
    for roi_name in _DIGIT_ROIS:
        out.extend(_digit_candidates(frame_id, roi_name, crops.get(roi_name),
                                     client, source_frame))
    for roi_name in _CARD_ROIS:
        out.extend(_card_candidates(frame_id, roi_name, crops.get(roi_name),
                                    client, source_frame))
    for roi_name in _DISC_ROIS:
        out.extend(_disc_candidates(frame_id, roi_name, crops.get(roi_name),
                                    client, source_frame))
    for roi_name in ("hero_cards", "board"):
        crop = crops.get(roi_name)
        if crop is not None:
            out.append(RealClientTemplate(
                template_id=f"f{frame_id}_{roi_name}_region",
                kind="region_snapshot", source_frame=source_frame,
                source_roi=roi_name, client=client,
                crop_bbox={"x": 0.0, "y": 0.0,
                           "width": crop.width, "height": crop.height}))
    return out
