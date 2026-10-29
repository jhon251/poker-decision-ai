"""确定性卡牌模板识别（rank 13 + suit 4 + 牌背）。

- 卡槽识别：rank 左上 40%×50%、suit 左下 40%×50% 子区，最近邻归一化到
  模板尺寸后 NCC 匹配；置信度 = min(rank_score, suit_score)；
- hero_cards：2 等分槽；board：连通域候选槽（0 候选 → UNCERTAIN，
  不宣称空板）；
- 实现 V1 CardRecognitionBackend，可直接接入 V1 流水线；
- 模板库 builtin_synthetic 字体：真实客户端字体命中与否实测如实说明，
  未标注验证前不宣称识别成功；任何低于阈值的匹配 → (None, None)。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..realtime.coords import Rect
from ..realtime.image import ImageData, ncc_template_match, to_gray
from ..realtime.recognition import (CardRecognitionBackend,
                                    CardRecognitionResult,
                                    CardRecognitionStatus, DetectedCard)
from .fonts import (builtin_card_back, builtin_rank_templates,
                    builtin_suit_templates)
from .geometry import connected_components, otsu_threshold, threshold_mask
from .templates import foreground_is_dark, invert_image


@dataclass
class CardTemplateLibrary:
    ranks: dict = None
    suits: dict = None
    back: ImageData | None = None
    min_score: float = 0.7
    source: str = "builtin_synthetic"

    def __post_init__(self) -> None:
        if self.ranks is None:
            self.ranks = builtin_rank_templates()
        if self.suits is None:
            self.suits = builtin_suit_templates()
        if self.back is None:
            self.back = builtin_card_back()

    def to_dict(self) -> dict:
        return {"ranks": sorted(self.ranks), "suits": sorted(self.suits),
                "min_score": self.min_score, "source": self.source,
                "rank_size": (next(iter(self.ranks.values())).width,
                              next(iter(self.ranks.values())).height),
                "suit_size": (next(iter(self.suits.values())).width,
                              next(iter(self.suits.values())).height)}


def builtin_card_library(min_score: float = 0.7) -> CardTemplateLibrary:
    return CardTemplateLibrary(min_score=min_score)


def _best_glyph(region: ImageData | None, templates: dict) -> tuple:
    """区域 vs 模板族（双极性竞争，确定性）。

    区域背景极性无法可靠先验（边框可能被桌面/相邻牌污染）→ 两种极性
    都跑一遍，取最高分。每极性：连通域 → 优先不贴边的内部组件 → 单元
    按模板宽高比补齐 → 同尺寸 NCC。
    """
    if region is None or not templates:
        return None, None
    tw = th = None
    for tpl in templates.values():
        tw, th = tpl.width, tpl.height
        break
    if tw is None:
        return None, None
    gray = to_gray(region)
    t = otsu_threshold(gray)
    best_char, best_score = None, -1.0
    for invert in (False, True):
        char, score = _try_polarity(region, gray, t, invert, templates, tw, th)
        if char is not None and score is not None and score > best_score:
            best_char, best_score = char, score
    return (best_char, best_score) if best_char is not None \
        else (None, None)


def _try_polarity(region, gray, t, invert, templates, tw, th) -> tuple:
    mask = threshold_mask(gray, threshold=t, invert=invert)
    comps = connected_components(mask, min_area=3)
    if not comps:
        return None, None
    interior = [c for c in comps
                if c.bbox.x > 0 and c.bbox.y > 0
                and c.bbox.x + c.bbox.width < region.width
                and c.bbox.y + c.bbox.height < region.height]
    largest = max(interior if interior else comps,
                  key=lambda c: c.area)
    cell_w = largest.bbox.height * (tw / th)
    pad = max(0.0, (cell_w - largest.bbox.width) / 2.0)
    cell = Rect(max(0.0, largest.bbox.x - pad), largest.bbox.y,
                min(float(region.width), cell_w), largest.bbox.height)
    cell = Rect(cell.x, cell.y,
                min(cell.width, region.width - cell.x), cell.height)
    sub = region.crop(cell)
    if sub is None or sub.width < tw or sub.height < th:
        return None, None
    # 前景为深色 → 反色成亮字形（模板恒为亮字形黑底）
    if foreground_is_dark(mask, gray):
        sub = invert_image(sub)
    out = bytearray(tw * th * 4)
    for y in range(th):
        sy = min(sub.height - 1, int(y * sub.height / th))
        for x in range(tw):
            sx = min(sub.width - 1, int(x * sub.width / tw))
            i = (sy * sub.width + sx) * 4
            o = (y * tw + x) * 4
            out[o:o + 4] = sub.data[i:i + 4]
    norm = ImageData.from_rgba_bytes(tw, th, bytes(out))
    best_char, best_score = None, -1.0
    for ch, tpl in templates.items():
        found = ncc_template_match(norm, tpl, score_threshold=-1.0)
        if found is not None and found[2] > best_score:
            best_char, best_score = ch, found[2]
    return (best_char, best_score) if best_char is not None \
        else (None, None)


def recognize_card_slot(crop: ImageData, lib: CardTemplateLibrary
                        ) -> tuple:
    """单卡槽 → (rank, suit, confidence, detail_dict)。

    子区：rank 左上 40%×50%、suit 左下 40%×50%（标准牌面布局）。
    任一子区不可用/低于阈值 → (None, None, None, detail)。
    """
    if crop is None or crop.width < 8 or crop.height < 10:
        return None, None, None, {}
    w, h = crop.width, crop.height
    rank_region = crop.crop(Rect(0.0, 0.0, w * 0.6, h * 0.5))
    suit_region = crop.crop(Rect(0.0, h * 0.5, w * 0.6, h * 0.5))
    rank, rank_score = _best_glyph(rank_region, lib.ranks)
    suit, suit_score = _best_glyph(suit_region, lib.suits)
    detail = {"rank_score": None if rank_score is None else round(rank_score, 4),
              "suit_score": None if suit_score is None else round(suit_score, 4),
              "templates_source": lib.source}
    if rank is None or suit is None \
            or min(rank_score, suit_score) < lib.min_score:
        return None, None, None, detail
    return rank, suit, round(min(rank_score, suit_score), 4), detail


def card_candidate_slots(crop: ImageData, *, min_area: int = 40,
                         max_slots: int = 5) -> list:
    """board/多卡区域 → 卡牌候选槽（连通域启发式；左→右）。"""
    if crop is None:
        return []
    gray = to_gray(crop)
    mask = threshold_mask(gray)                     # 自动极性
    comps = connected_components(mask, min_area=min_area)
    slots = [c.bbox for c in comps
             if 0.35 <= c.aspect_ratio <= 1.1 and c.fill_ratio >= 0.35]
    slots.sort(key=lambda r: r.x)
    return slots[:max_slots]


class CardTemplateRecognitionBackend(CardRecognitionBackend):
    """V1 CardRecognitionBackend 实现（确定性模板，无 LLM/分类器）。"""
    name = "card_template_recognition"
    synthetic_test_only = False

    def __init__(self, lib: CardTemplateLibrary | None = None):
        self.lib = lib or builtin_card_library()

    def available(self) -> bool:
        return self.lib is not None

    def unavailable_reason(self) -> str:
        return "" if self.lib is not None else "no card template library"

    def recognize(self, crop=None, roi_name="") -> CardRecognitionResult:
        if crop is None:
            return CardRecognitionResult(
                status=CardRecognitionStatus.FAILED, error="crop_missing",
                synthetic_test_only=False)
        if roi_name == "hero_cards":
            return self._recognize_hero(crop)
        if roi_name == "board":
            return self._recognize_board(crop)
        return CardRecognitionResult(
            status=CardRecognitionStatus.FAILED,
            error=f"unsupported roi {roi_name}", synthetic_test_only=False)

    def _recognize_hero(self, crop: ImageData) -> CardRecognitionResult:
        w, h = crop.width, crop.height
        if w < 16 or h < 10:
            return CardRecognitionResult(
                status=CardRecognitionStatus.UNCERTAIN,
                error="hero_cards crop too small", synthetic_test_only=False)
        cards = []
        for i in range(2):
            slot = crop.crop(Rect(i * w / 2.0, 0.0, w / 2.0, float(h)))
            rank, suit, conf, _ = recognize_card_slot(slot, self.lib)
            cards.append(DetectedCard(
                rank=rank, suit=suit,
                bbox=Rect(i / 2.0, 0.0, 1.0 / 2.0, 1.0),
                confidence=conf, source=self.name))
        ok = any(c.rank is not None and c.suit is not None for c in cards)
        return CardRecognitionResult(
            cards=tuple(cards),
            status=(CardRecognitionStatus.OK if ok
                    else CardRecognitionStatus.UNCERTAIN),
            error=None if ok else "no card slot above threshold",
            synthetic_test_only=False)

    def _recognize_board(self, crop: ImageData) -> CardRecognitionResult:
        slots = card_candidate_slots(crop)
        if not slots:
            # 0 候选 ≠ 空板：不能可靠确认，不宣称空 board（诚实 UNCERTAIN）
            return CardRecognitionResult(
                status=CardRecognitionStatus.UNCERTAIN,
                error="no card-like components segmented",
                synthetic_test_only=False)
        w, h = crop.width, crop.height
        cards = []
        for rect in slots:
            slot = crop.crop(rect)
            rank, suit, conf, _ = recognize_card_slot(slot, self.lib)
            cards.append(DetectedCard(
                rank=rank, suit=suit,
                bbox=Rect(rect.x / w, rect.y / h,
                          rect.width / w, rect.height / h),
                confidence=conf, source=self.name))
        ok = any(c.rank is not None and c.suit is not None for c in cards)
        return CardRecognitionResult(
            cards=tuple(cards),
            status=(CardRecognitionStatus.OK if ok
                    else CardRecognitionStatus.UNCERTAIN),
            error=None if ok else "no card slot above threshold",
            synthetic_test_only=False)


def render_test_card(rank: str, suit: str, *, scale: int = 3,
                     width: int = 40, height: int = 56) -> ImageData:
    """测试/基准用：builtin 字体渲染单张牌（白底深色符号；非真实客户端）。"""
    from .fonts import _render, GLYPHS_5X7, RANK_GLYPHS, SUIT_GLYPHS
    rank_rows = GLYPHS_5X7.get(rank) or RANK_GLYPHS.get(rank)
    suit_rows = SUIT_GLYPHS[suit]
    if rank_rows is None:
        raise ValueError(f"未知 rank {rank}")
    buf = bytearray(bytes((208, 208, 208, 255)) * (width * height))
    rank_img = _render(rank_rows, scale, fg=20, bg=208)
    suit_img = _render(suit_rows, scale, fg=20, bg=208)

    def blit(img, ox, oy):
        for y in range(img.height):
            row_start = ((oy + y) * width + ox) * 4
            buf[row_start:row_start + img.width * 4] = \
                img.data[y * img.width * 4:(y + 1) * img.width * 4]

    blit(rank_img, 2, 2)
    blit(suit_img, 2, height - 2 - suit_img.height)
    return ImageData.from_rgba_bytes(width, height, bytes(buf))
