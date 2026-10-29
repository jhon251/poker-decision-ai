"""V2C — 真实客户端识别后端（复用 V2B 识别机制，仅换资产域）。

门控（硬约束）：
  - 资产库中没有"已标注"的真实客户端资产 → available()=False →
    UNAVAILABLE → 字段 null + uncertainty（绝不猜、绝不回退 synthetic）；
  - 已标注资产（visually_labeled/verified/test）才参与识别；
  - 识别机制复用 V2B 的字符单元重建/极性归一化/NCC（不新建核心）。
"""
from __future__ import annotations

from ...realtime.coords import Rect
from ...realtime.image import ImageData, to_gray
from ...realtime.recognition import (CardRecognitionBackend,
                                     CardRecognitionResult,
                                     CardRecognitionStatus, OCRBackend,
                                     OCRResult, OCRStatus)
from ..digit import recognize_digits
from ..geometry import threshold_mask
from ..templates import foreground_is_dark, invert_image
from .assets import DIGITS, RANKS, SUITS, RealClientTemplateLibrary


def _normalize_set(templates: dict, tw: int, th: int) -> dict:
    """模板族统一尺寸（最近邻；确定性）。

    资产在提取阶段已极规范化（亮字形黑底）；此处仅尺寸归一。
    """
    out = {}
    for label, img in templates.items():
        buf = bytearray(tw * th * 4)
        for y in range(th):
            sy = min(img.height - 1, int(y * img.height / th))
            for x in range(tw):
                sx = min(img.width - 1, int(x * img.width / tw))
                i = (sy * img.width + sx) * 4
                o = (y * tw + x) * 4
                buf[o:o + 4] = img.data[i:i + 4]
        out[label] = ImageData.from_rgba_bytes(tw, th, bytes(buf))
    return out


def _median_dims(images) -> tuple:
    ws = sorted(img.width for img in images)
    hs = sorted(img.height for img in images)
    if not ws:
        return 1, 1
    return ws[len(ws) // 2], hs[len(hs) // 2]


def _binarize(img: ImageData):
    """灰度 → 二值（亮度均值阈值；颜色无关：黑字/红字/金字统一字形轮廓）。"""
    import numpy as _np
    arr = _np.asarray(to_gray(img).data, dtype="float")
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return arr > 0
    return arr > (lo + hi) / 2.0


def _binary_ncc(a, b) -> float:
    import numpy as _np
    A = _binarize(a).astype(float)
    B = _binarize(b).astype(float)
    A0 = A - A.mean()
    B0 = B - B.mean()
    d1 = _np.sqrt((A0 * A0).sum()) * _np.sqrt((B0 * B0).sum())
    n1 = float((A0 * B0).sum() / d1) if d1 > 0 else 0.0
    B1 = (1.0 - B) - (1.0 - B).mean()          # 反极性
    d2 = _np.sqrt((A0 * A0).sum()) * _np.sqrt((B1 * B1).sum())
    n2 = float((A0 * B1).sum() / d2) if d2 > 0 else 0.0
    return n1 if n1 >= n2 else n2


def _slide_family_best(slot_g, templates, stride: int = 4) -> tuple:
    """滑动 NCC 全匹配（组件无关；双极性 = |NCC|；确定性）。

    逐字符最佳窗口分 → (最佳字符, 分, 位置, 跨字符判别余量)。
    """
    import numpy as _np
    char_scores = {}
    for ch, tpl in templates.items():
        g = _np.asarray(to_gray(tpl).data, dtype="float")
        th, tw = g.shape
        if th >= slot_g.shape[0] or tw >= slot_g.shape[1]:
            continue
        # 自适应步长：rank 级小字形 stride1（奇数对齐不 miss）；大花色 stride2
        us = 2 if th >= 55 else 1
        wins = _np.lib.stride_tricks.sliding_window_view(slot_g, (th, tw))
        sub = wins[::us, ::us]
        t0 = g - g.mean()
        tn = _np.sqrt((t0 * t0).sum())
        if tn < 1e-9:
            continue
        wm = sub.mean(axis=(2, 3), keepdims=True)
        wc = sub - wm
        wn = _np.sqrt((wc * wc).sum(axis=(2, 3)))
        num = (wc * t0).sum(axis=(2, 3))
        scores = num / _np.maximum(wn * tn, 1e-9)
        abs_scores = _np.abs(scores)
        idx = _np.unravel_index(_np.argmax(abs_scores), abs_scores.shape)
        char_scores[ch] = (float(abs_scores[idx]),
                           (int(idx[1] * us), int(idx[0] * us)))
    if not char_scores:
        return (None, -1.0, None, -1.0)
    ordered = sorted(char_scores.items(), key=lambda kv: -kv[1][0])
    best_ch, (best_sc, best_pos) = ordered[0]
    second = ordered[1][1][0] if len(ordered) > 1 else 0.0
    return best_ch, best_sc, best_pos, best_sc - second


def _recognize_hero_v2(slot: ImageData, card_lib,
                       min_margin: float = 0.08) -> tuple:
    """hero 卡槽滑动匹配（rank/suit 全槽搜索 + 双极性 + 判别余量）。"""
    import numpy as _np
    slot_g = _np.asarray(to_gray(slot).data, dtype="float")
    r, rs, rp, rm = _slide_family_best(slot_g, card_lib.ranks)
    s, ss, sp, sm = _slide_family_best(slot_g, card_lib.suits)
    if r is None or s is None or rs is None or ss is None:
        return None, None, None
    if min(rs, ss) < card_lib.min_score:
        return None, None, None
    if min(rm, sm) < min_margin:               # 判别力不足 → 拒绝
        return None, None, None
    return r, s, round(min(rs, ss), 4)


def _match_family(component: ImageData, templates: dict,
                  binary: bool = True) -> tuple:
    """单组件 vs 模板族 → (best_char, best_score, norm)。

    卡牌默认二值化 NNC（颜色/极性不变：黑字/红字统一字形轮廓）；
    金色数字走 binary=False（灰度 NCC，金/深极性明确时更优）。
    """
    if not templates:
        return None, None, None
    tw = th = None
    for tpl in templates.values():
        tw, th = tpl.width, tpl.height
        break
    img = component
    out = bytearray(tw * th * 4)
    for y in range(th):
        sy = min(img.height - 1, int(y * img.height / th))
        for x in range(tw):
            sx = min(img.width - 1, int(x * img.width / tw))
            i = (sy * img.width + sx) * 4
            o = (y * tw + x) * 4
            out[o:o + 4] = img.data[i:i + 4]
    norm = ImageData.from_rgba_bytes(tw, th, bytes(out))
    best_char, best_score = None, -1.0
    for ch, tpl in templates.items():
        if binary:
            score = _binary_ncc(norm, tpl)
        else:
            from ...realtime.image import ncc_template_match
            f1 = ncc_template_match(norm, tpl, score_threshold=-1.0)
            f2 = ncc_template_match(norm, invert_image(tpl), score_threshold=-1.0)
            s1 = f1[2] if f1 is not None else -1.0
            s2 = f2[2] if f2 is not None else -1.0
            score = s1 if s1 >= s2 else s2
        if score > best_score:
            best_char, best_score = ch, score
    return best_char, best_score, norm


def _slot_components(slot: ImageData) -> list:
    """卡槽内字形候选组件（真实客户端域适配：允许贴单边）。

    排除：贴 ≥2 边的组件（卡身/边界）、面积 > 60% 槽面的组件；
    其余按面积降序返回。
    """
    from ..geometry import connected_components
    gray = to_gray(slot)
    mask = threshold_mask(gray)
    comps = connected_components(mask, min_area=6)
    out = []
    for c in comps:
        touches = 0
        if c.bbox.x <= 0:
            touches += 1
        if c.bbox.y <= 0:
            touches += 1
        if c.bbox.x + c.bbox.width >= slot.width:
            touches += 1
        if c.bbox.y + c.bbox.height >= slot.height:
            touches += 1
        if touches >= 2:
            continue
        if c.area > 0.6 * slot.width * slot.height:
            continue
        out.append(c)
    return sorted(out, key=lambda c: -c.area)[:8]


def _pad_component(slot: ImageData, comp, pad_frac: float = 0.3):
    """组件裁剪（带 30% 上下文 padding，与提取时模板一致）。"""
    w, h = slot.width, slot.height
    px = max(1.0, comp.bbox.width * pad_frac)
    py = max(1.0, comp.bbox.height * pad_frac)
    x0 = max(0, int(comp.bbox.x - px))
    y0 = max(0, int(comp.bbox.y - py))
    x1 = min(w, int(comp.bbox.x + comp.bbox.width + px))
    y1 = min(h, int(comp.bbox.y + comp.bbox.height + py))
    return slot.crop(Rect(float(x0), float(y0), float(x1 - x0),
                          float(y1 - y0)))


def _recognize_slot_real(slot: ImageData, card_lib) -> tuple:
    """整槽组件匹配（真实客户端域适配：字形占比大时固定子区会裁剪）。

    每组件 ×（原始/带垫）×（原极性/反色）四组合取最优——紧凑字形极性
    不可靠先验，双极性竞争 + 垫边竞争为确定性最强结果。
    """
    best_rank = (None, None, None)
    best_suit = (None, None, None)
    for comp in _slot_components(slot)[:8]:
        candidates = [slot.crop(comp.bbox), _pad_component(slot, comp)]
        for g in candidates:
            if g is None or g.width < 4 or g.height < 4:
                continue
            for g2 in (g, invert_image(g)):
                r, rs, _ = _match_family(g2, card_lib.ranks)
                s, ss, _ = _match_family(g2, card_lib.suits)
                if r is not None and (best_rank[1] is None
                                      or rs > best_rank[1]):
                    best_rank = (r, rs, comp)
                if s is not None and (best_suit[1] is None
                                      or ss > best_suit[1]):
                    best_suit = (s, ss, comp)
    rank, rank_score = best_rank[0], best_rank[1]
    suit, suit_score = best_suit[0], best_suit[1]
    if rank is None or suit is None or rank_score is None or suit_score is None \
            or min(rank_score, suit_score) < card_lib.min_score:
        return None, None, None
    return rank, suit, round(min(rank_score, suit_score), 4)


class RealClientCardLibrary:
    """从 RealClientTemplateLibrary 提取已标注卡牌模板（rank/suit 分族）。

    source_roi 非空时仅取该 ROI 来源资产（hero/board 域隔离）。
    """
    source = "real_client"

    def __init__(self, library: RealClientTemplateLibrary,
                 min_score: float = 0.7, source_roi: str | None = None):
        self.library = library
        self.min_score = min_score
        self.source_roi = source_roi
        self.ranks = library.labeled_templates("card_glyph", RANKS,
                                               source_roi=source_roi)
        self.suits = library.labeled_templates("card_glyph", SUITS,
                                               source_roi=source_roi)
        if self.ranks:
            tw, th = _median_dims(self.ranks.values())
            self.ranks = _normalize_set(self.ranks, tw, th)
        if self.suits:
            tw, th = _median_dims(self.suits.values())
            self.suits = _normalize_set(self.suits, tw, th)

    def available(self) -> bool:
        return bool(self.ranks) and bool(self.suits)


class RealClientCardRecognitionBackend(CardRecognitionBackend):
    """真实客户端卡牌识别（V1 CardRecognitionBackend 实现）。

    无已标注 rank/suit 资产 → UNAVAILABLE（诚实降级）；角色级标注
    （role_identified）不开启值识别。
    """
    name = "real_client_card_recognition"
    synthetic_test_only = False

    def __init__(self, library: RealClientTemplateLibrary | None = None,
                 min_score: float = 0.7):
        self.library = library or RealClientTemplateLibrary()
        self.min_score = min_score
        # 域隔离：board / hero_cards 各自使用自己 ROI 来源的模板
        self.board_lib = RealClientCardLibrary(self.library, min_score,
                                               source_roi="board")
        self.hero_lib = RealClientCardLibrary(self.library, min_score,
                                              source_roi="hero_cards")

    def available(self) -> bool:
        return self.board_lib.available() or self.hero_lib.available()

    def role_status(self) -> dict:
        """角色级进度（字符级标注 pending 时明确如实报告）。"""
        from collections import Counter
        roles = Counter(a.role for a in self.library.assets.values()
                        if a.role)
        return {"known_roles": dict(roles),
                "character_labels_pending": not self.available(),
                "available": self.available(),
                "board_available": self.board_lib.available(),
                "hero_available": self.hero_lib.available()}

    def unavailable_reason(self) -> str:
        if not self.board_lib.available() and not self.hero_lib.available():
            return ("no labeled real-client rank/suit assets "
                    "(board or hero_cards)")
        return ""

    def recognize(self, crop=None, roi_name="") -> CardRecognitionResult:
        if crop is None:
            return CardRecognitionResult(
                status=CardRecognitionStatus.FAILED, error="crop_missing",
                synthetic_test_only=False)
        if roi_name == "hero_cards":
            if not self.hero_lib.available():
                return CardRecognitionResult(
                    status=CardRecognitionStatus.UNAVAILABLE,
                    error="no labeled hero_cards rank/suit assets",
                    synthetic_test_only=False)
            return self._recognize_hero(crop, self.hero_lib)
        if roi_name == "board":
            if not self.board_lib.available():
                return CardRecognitionResult(
                    status=CardRecognitionStatus.UNAVAILABLE,
                    error="no labeled board rank/suit assets",
                    synthetic_test_only=False)
            return self._recognize_board(crop, self.board_lib)
        return CardRecognitionResult(
            status=CardRecognitionStatus.FAILED,
            error=f"unsupported roi {roi_name}", synthetic_test_only=False)

    def _recognize_hero(self, crop: ImageData,
                        lib: RealClientCardLibrary) -> CardRecognitionResult:
        from ...realtime.recognition import DetectedCard
        from ...realtime.coords import Rect
        w, h = crop.width, crop.height
        if w < 16 or h < 10:
            return CardRecognitionResult(
                status=CardRecognitionStatus.UNCERTAIN,
                error="hero_cards crop too small", synthetic_test_only=False)
        cards = []
        for i in range(2):
            slot = crop.crop(Rect(i * w / 2.0, 0.0, w / 2.0, float(h)))
            rank, suit, conf = _recognize_hero_v2(slot, lib)
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

    def _recognize_board(self, crop: ImageData,
                         lib: RealClientCardLibrary) -> CardRecognitionResult:
        from ...realtime.recognition import DetectedCard
        from ..cards import card_candidate_slots
        slots = card_candidate_slots(crop)
        if not slots:
            return CardRecognitionResult(
                status=CardRecognitionStatus.UNCERTAIN,
                error="no card-like components segmented",
                synthetic_test_only=False)
        w, h = crop.width, crop.height
        cards = []
        for rect in slots:
            slot = crop.crop(rect)
            rank, suit, conf = _recognize_slot_real(slot, lib)
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


class RealClientDigitOCRBackend(OCRBackend):
    """真实客户端数字识别（V1 OCRBackend 实现，kind="number"）。

    无已标注数字资产 → UNAVAILABLE（诚实降级）；角色级标注不开启值识别。
    """
    name = "real_client_digit_ocr"
    synthetic_test_only = False

    def __init__(self, library: RealClientTemplateLibrary | None = None,
                 min_score: float = 0.7):
        self.library = library or RealClientTemplateLibrary()
        self.min_score = min_score

    def available(self) -> bool:
        return bool(self.library.labeled_templates("digit_glyph", DIGITS))

    def role_status(self) -> dict:
        from collections import Counter
        roles = Counter(a.role for a in self.library.assets.values()
                        if a.role)
        return {"known_roles": dict(roles),
                "character_labels_pending": not self.available(),
                "available": self.available()}

    def unavailable_reason(self) -> str:
        return "no labeled real-client digit assets"

    def recognize(self, crop=None, roi_name="", kind="number") -> OCRResult:
        if kind != "number":
            return OCRResult(status=OCRStatus.FAILED,
                             error=f"unsupported kind {kind}",
                             synthetic_test_only=False)
        if crop is None:
            return OCRResult(status=OCRStatus.FAILED, error="crop_missing",
                             synthetic_test_only=False)
        templates = self.library.labeled_templates("digit_glyph",
                                                    DIGITS + "MKB",
                                                    source_roi="pot")
        if not templates:
            templates = self.library.labeled_templates("digit_glyph",
                                                       DIGITS + "MKB")
        if not templates:
            return OCRResult(status=OCRStatus.UNAVAILABLE,
                             error=self.unavailable_reason(),
                             synthetic_test_only=False)
        if roi_name in ("pot", "bet_area", "stacks"):
            # 同款金色字体（pot/hero-stack/bet 均为金色文本）→ 共用金色路径
            return self._recognize_gold(crop, templates)
        tw, th = _median_dims(templates.values())
        templates = _normalize_set(templates, tw, th)
        rec = recognize_digits(crop, templates, self.min_score)
        status = {"OK": OCRStatus.OK, "UNCERTAIN": OCRStatus.UNCERTAIN,
                  "UNAVAILABLE": OCRStatus.UNAVAILABLE}[rec.status.value]
        return OCRResult(value=rec.value, raw_text=rec.text,
                         confidence=rec.confidence, status=status,
                         error=rec.error, synthetic_test_only=False,
                         extra={"glyphs": [g.to_dict() for g in rec.glyphs],
                                "templates_source": "real_client_labeled"})

    def _recognize_gold(self, crop: ImageData, templates: dict) -> OCRResult:
        """pot 金色文本 → 金色掩码切字符 → 逐字符模板匹配 → 文本串。

        数值化语义（千分位/金额尺度）不做猜测：value=None，
        raw_text=识别出的字符序列（未知字符 → UNCERTAIN）。
        """
        from .pot_seg import segments_gold_chars
        tw, th = _median_dims(templates.values())
        templates = _normalize_set(templates, tw, th)
        rects, _ = segments_gold_chars(crop)
        if not rects:
            return OCRResult(status=OCRStatus.UNCERTAIN,
                             error="no gold glyph cells",
                             synthetic_test_only=False)
        chars, scores = [], []
        for r in rects:
            # 与提取一致的 30% 上下文 padding（保证极性/几何一致）
            px = max(1.0, r.width * 0.3)
            py = max(1.0, r.height * 0.3)
            padded = Rect(max(0.0, r.x - px), max(0.0, r.y - py),
                          min(crop.width, r.x + r.width + px) -
                          max(0.0, r.x - px),
                          min(crop.height, r.y + r.height + py) -
                          max(0.0, r.y - py))
            sub = crop.crop(padded)
            if sub is None or sub.width < 4 or sub.height < 4:
                return OCRResult(status=OCRStatus.UNCERTAIN,
                                 error="gold char cell too small",
                                 synthetic_test_only=False)
            ch, score, _ = _match_family(sub, templates, binary=False)
            if ch is None or score is None or score < self.min_score:
                return OCRResult(
                    status=OCRStatus.UNCERTAIN,
                    error=f"gold char below threshold ({score})",
                    synthetic_test_only=False)
            chars.append(ch)
            scores.append(round(score, 4))
        text = "".join(chars)
        value, unit = _parse_gold_value(text)
        return OCRResult(value=value, raw_text=text,
                         confidence=min(scores) if scores else None,
                         status=OCRStatus.OK, error=None,
                         synthetic_test_only=False,
                         extra={"glyphs": chars, "scores": scores,
                                "unit": unit,
                                "templates_source": "real_client_labeled"})


_UNIT_MULT = {"M": 1e6, "K": 1e3, "B": 1e9}


def _parse_gold_value(text: str) -> tuple:
    """gold 文本 → (value, unit)。确定性；无法解析 → (None, None)。

    - 末尾 M/K/B → 数值 × 单位倍率；
    - 纯数字含千分位逗号 → int（客户端 6 位筹码计数格式）；
    - 其余（含小数但无单位/异常）→ None（不猜）。
    """
    if not text:
        return None, None
    unit = None
    body = text
    if text[-1] in _UNIT_MULT:
        unit = text[-1]
        body = text[:-1]
    try:
        if unit is not None:
            return float(body.replace(",", "")) * _UNIT_MULT[unit], unit
        if "," in body:
            return int(body.replace(",", "")), None
        if "." in body:
            return None, None
        return int(body), None
    except (ValueError, OverflowError):
        return None, unit
