"""数字字形识别（确定性模板路线；不依赖 OCR）。

- glyph segmentation：大津二值化 → 连通域 → 过滤 → 左→右排序；
- 识别：每字形最近邻归一化到模板尺寸 → NCC vs 数字模板 → 最佳字符；
- 置信度 = 各字形 score 的 min（最弱字形决定整体置信）；
- 实现 V1 OCRBackend 接口（kind="number"），可直接接入 V1 流水线；
- 模板库为 builtin_synthetic 字体（source 恒标注）；真实客户端字体是否
  命中由实测如实说明，未标注验证前不宣称识别成功。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..realtime.coords import Rect
from ..realtime.image import ImageData, ncc_template_match, to_gray
from ..realtime.recognition import OCRBackend, OCRResult, OCRStatus
from .fonts import builtin_digit_templates
from .geometry import connected_components, threshold_mask


class DigitRecognitionStatus(str, Enum):
    OK = "OK"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class GlyphMatch:
    char: str | None
    score: float | None
    bbox: Rect | None

    def to_dict(self) -> dict:
        return {"char": self.char, "score": None if self.score is None
                else round(self.score, 4),
                "bbox": None if self.bbox is None else self.bbox.to_dict()}


@dataclass
class DigitRecognition:
    text: str | None = None
    value: float | None = None
    glyphs: tuple = ()
    confidence: float | None = None
    status: DigitRecognitionStatus = DigitRecognitionStatus.UNAVAILABLE
    error: str | None = None
    templates_source: str = "builtin_synthetic"

    def to_dict(self) -> dict:
        return {
            "text": self.text, "value": self.value,
            "glyphs": [g.to_dict() for g in self.glyphs],
            "confidence": None if self.confidence is None
            else round(self.confidence, 4),
            "status": self.status.value, "error": self.error,
            "templates_source": self.templates_source,
        }


def segment_glyphs(crop: ImageData, *, min_area: int = 3,
                   max_glyphs: int = 16,
                   threshold: int | None = None) -> tuple:
    """裁剪 → 字形包围盒（左→右）。无字形 → ()。"""
    if crop is None:
        return ()
    gray = to_gray(crop)
    mask = threshold_mask(gray, threshold)          # 自动极性
    comps = connected_components(mask, min_area=min_area)
    glyphs = [c.bbox for c in comps
              if c.bbox.width <= crop.width
              and c.bbox.height <= crop.height]
    glyphs.sort(key=lambda r: (r.x, r.y))
    return tuple(glyphs[:max_glyphs])


def normalize_glyph(crop: ImageData, rect: Rect, tw: int, th: int,
                    invert: bool | None = None) -> ImageData | None:
    """字形单元最近邻采样到模板尺寸 + 极性翻转（确定性）。

    invert=None → 旧规则（均值 ≥128 反色）；显式 True/False 按调用方
    mask 前景/背景亮度比较传入。
    """
    from .templates import invert_image
    sub = crop.crop(rect)
    if sub is None:
        return None
    out = bytearray(tw * th * 4)
    for y in range(th):
        sy = min(sub.height - 1, int(y * sub.height / th))
        for x in range(tw):
            sx = min(sub.width - 1, int(x * sub.width / tw))
            i = (sy * sub.width + sx) * 4
            o = (y * tw + x) * 4
            out[o:o + 4] = sub.data[i:i + 4]
    img = ImageData.from_rgba_bytes(tw, th, bytes(out))
    if invert is None:
        total = sum(out[i] for i in range(0, tw * th * 4, 4))
        if total / (tw * th) >= 128.0:
            img = invert_image(img)
    elif invert:
        img = invert_image(img)
    return img


def segment_runs(crop: ImageData, threshold: int | None = None) -> tuple:
    """亮列 run 列表（左闭右开区间）与掩码。确定性。"""
    gray = to_gray(crop)
    mask = threshold_mask(gray, threshold)
    h = len(mask)
    w = len(mask[0]) if h else 0
    runs = []
    if h == 0 or w == 0:
        return (), mask
    col_bright = [any(mask[y][x] for y in range(h)) for x in range(w)]
    x = 0
    while x < w:
        if not col_bright[x]:
            x += 1
            continue
        x0 = x
        while x < w and col_bright[x]:
            x += 1
        runs.append((x0, x))
    return tuple(runs), mask


def segment_cells(crop: ImageData, *, min_area: int = 3,
                  max_cells: int = 16,
                  threshold: int | None = None) -> tuple:
    """列投影 + 间隙中点 → 字符单元（比组件包围盒更接近真实字格）。

    对 '.'/',' 等子字格字形，组件包围盒丢失暗部结构；间隙中点法利用
    相邻字形间距恢复单元左右边界（确定性）。
    """
    if crop is None:
        return ()
    runs, mask = segment_runs(crop, threshold)
    h = len(mask)
    if not runs:
        return ()
    bounds = [runs[0][0]]
    for (_, e1), (s2, _) in zip(runs, runs[1:]):
        bounds.append((e1 + s2) // 2)
    bounds.append(runs[-1][1])
    cells = []
    for i in range(len(bounds) - 1):
        x0, x1 = bounds[i], bounds[i + 1]
        if x1 <= x0:
            continue
        ys = [y for x in range(x0, x1) for y in range(h) if mask[y][x]]
        if not ys:
            continue
        y0, y1 = min(ys), max(ys) + 1
        cells.append(Rect(float(x0), float(y0), float(x1 - x0),
                          float(y1 - y0)))
    return tuple(cells[:max_cells])


def _median(values: list) -> float:
    s = sorted(values)
    return s[len(s) // 2] if s else 0.0


def recognize_digits(crop: ImageData, templates: dict | None = None,
                     min_score: float = 0.7) -> DigitRecognition:
    """数字串识别（字符单元归一化 + 同尺寸 NCC，确定性）。

    - 单元尺寸 = 行内字形包围盒的中位数（'.'/',' 等小字形居中于单元）；
    - 单元最近邻归一化到模板尺寸 → 同尺寸 NCC → 最佳字符；
    - 模板缺失 → UNAVAILABLE；无字形/低分 → UNCERTAIN。
    """
    if not templates:
        return DigitRecognition(status=DigitRecognitionStatus.UNAVAILABLE,
                                error="no digit templates")
    if crop is None:
        return DigitRecognition(status=DigitRecognitionStatus.UNCERTAIN,
                                error="crop_missing")
    tw = th = None
    for tpl in templates.values():
        tw, th = tpl.width, tpl.height
        break
    if tw is None:
        return DigitRecognition(status=DigitRecognitionStatus.UNAVAILABLE,
                                error="empty template set")
    runs, mask = segment_runs(crop)
    if not runs:
        return DigitRecognition(status=DigitRecognitionStatus.UNCERTAIN,
                                error="no glyphs segmented")
    from .templates import foreground_is_dark
    invert_glyph = foreground_is_dark(mask, to_gray(crop))
    h_mask = len(mask)
    w_mask = len(mask[0]) if h_mask else 0
    run_heights = []
    for (s, e) in runs:
        ys = [y for x in range(s, e) for y in range(h_mask) if mask[y][x]]
        run_heights.append(max(ys) - min(ys) + 1 if ys else 0)
    cell_h = _median(run_heights)
    full_runs = [(r, hh) for r, hh in zip(runs, run_heights)
                 if hh >= 0.6 * cell_h]
    full_w = _median([e - s for (s, e), _ in full_runs]) if full_runs else 1.0
    all_ys = [y for y in range(h_mask)
              if any(mask[y][x] for x in range(w_mask))]
    line_bottom = (max(all_ys) + 1) if all_ys else 0
    if cell_h <= 0 or line_bottom <= 0:
        return DigitRecognition(status=DigitRecognitionStatus.UNCERTAIN,
                                error="degenerate glyph cells")
    chars: list = []
    matches: list = []
    y0_base = float(line_bottom - cell_h)
    for i, ((s, e), hh) in enumerate(zip(runs, run_heights)):
        # 三种单元假设竞争（确定性）：run 自身 / 2 列子字格 / 3 列子字格
        # （'1' 等窄字形为 3 列，'.' 等标点为 2 列；不猜列数，取最高分）
        pitch2 = (e - s) / 2.0
        pitch3 = (e - s) / 3.0
        candidate_cells = [
            Rect(float(s), y0_base, float(e - s), float(cell_h)),
            Rect(float(s - pitch2), y0_base, float(pitch2 * 5.0),
                 float(cell_h)),
            Rect(float(s - pitch3), y0_base, float(pitch3 * 5.0),
                 float(cell_h)),
        ]
        best_char, best_score = None, -1.0
        for cand in candidate_cells:
            x0c = max(0.0, cand.x)
            x1c = min(float(crop.width), cand.x + cand.width)
            y0c = max(0.0, cand.y)
            y1c = min(float(crop.height), cand.y + cand.height)
            cell = Rect(x0c, y0c, max(1.0, x1c - x0c),
                        max(1.0, y1c - y0c))
            norm = normalize_glyph(crop, cell, tw, th, invert=invert_glyph)
            if norm is None:
                continue
            for ch, tpl in templates.items():
                found = ncc_template_match(norm, tpl, score_threshold=-1.0)
                if found is not None and found[2] > best_score:
                    best_char, best_score = ch, found[2]
        if best_char is not None and best_score >= min_score:
            chars.append(best_char)
        else:
            chars.append(None)
        matches.append(GlyphMatch(
            char=best_char,
            score=None if best_char is None else best_score,
            bbox=Rect(float(s), y0_base, float(e - s), float(cell_h))))
    scores = [m.score for m in matches if m.score is not None]
    confidence = min(scores) if scores else None
    if any(c is None for c in chars) or not chars:
        return DigitRecognition(
            glyphs=tuple(matches), confidence=confidence,
            status=DigitRecognitionStatus.UNCERTAIN,
            error="glyph below threshold" if chars else "no glyphs recognized")
    text = "".join(chars)
    value = None
    try:
        value = float(text)
    except ValueError:
        pass
    return DigitRecognition(text=text, value=value, glyphs=tuple(matches),
                            confidence=confidence,
                            status=DigitRecognitionStatus.OK)


class DigitTemplateOCRBackend(OCRBackend):
    """V1 OCRBackend 实现（kind="number"，确定性字形模板；非 tesseract）。"""
    name = "digit_template_ocr"
    synthetic_test_only = False

    def __init__(self, templates: dict | None = None, min_score: float = 0.7):
        self.templates = templates if templates is not None \
            else builtin_digit_templates()
        self.min_score = min_score

    def available(self) -> bool:
        return bool(self.templates)

    def unavailable_reason(self) -> str:
        return "" if self.templates else "no digit templates"

    def recognize(self, crop=None, roi_name="", kind="number") -> OCRResult:
        if kind != "number":
            return OCRResult(status=OCRStatus.FAILED,
                             error=f"unsupported kind {kind}",
                             synthetic_test_only=False)
        if crop is None:
            return OCRResult(status=OCRStatus.FAILED, error="crop_missing",
                             synthetic_test_only=False)
        rec = recognize_digits(crop, self.templates, self.min_score)
        status = {
            DigitRecognitionStatus.OK: OCRStatus.OK,
            DigitRecognitionStatus.UNCERTAIN: OCRStatus.UNCERTAIN,
            DigitRecognitionStatus.UNAVAILABLE: OCRStatus.UNAVAILABLE,
        }[rec.status]
        return OCRResult(value=rec.value, raw_text=rec.text,
                         confidence=rec.confidence, status=status,
                         error=rec.error, synthetic_test_only=False,
                         extra={"glyphs": [g.to_dict() for g in rec.glyphs],
                                "templates_source": rec.templates_source})
