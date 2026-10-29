"""真实 OCR 后端（tesseract，可选依赖；未安装 → 诚实降级 UNAVAILABLE）。

- 仅数字/文字区域（pot/stack/bet/blind/table 元数据/行动文字）；
- 卡牌识别绝不走 OCR（那是识别层 Level 3 的职责）；
- 依赖缺失 → available()=False；运行时 tesseract 错误 → FAILED；
- 输出只描述所见数字/文字，不做任何牌局推断。
"""
from __future__ import annotations

import importlib.util

from poker_decision_ai.vision2.realtime.recognition import (OCRBackend,
                                                            OCRResult,
                                                            OCRStatus)

_TEXT_WHITELIST = ("abcdefghijklmnopqrstuvwxyz"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .")


def _tesseract_available() -> bool:
    try:
        if importlib.util.find_spec("pytesseract") is None:
            return False
        if importlib.util.find_spec("PIL") is None:
            return False
        return True
    except Exception:
        return False


def _parse_number(text: str) -> float | None:
    """数字文本 → float；多种分隔符兼容（确定性；失败 → None）。"""
    candidates = [text.strip(),
                  text.strip().replace(",", "."),
                  text.strip().replace(",", "").replace(" ", "")]
    for cand in candidates:
        try:
            v = float(cand)
            if v == v:
                return v
        except ValueError:
            continue
    return None


class TesseractNumberOCRBackend(OCRBackend):
    """tesseract 数字/文字 OCR（synthetic_test_only=False）。"""
    name = "tesseract_number_ocr"
    synthetic_test_only = False

    def __init__(self, *, psm: int = 7, lang: str = "eng"):
        self.psm = psm
        self.lang = lang

    def available(self) -> bool:
        return _tesseract_available()

    def unavailable_reason(self) -> str:
        if importlib.util.find_spec("pytesseract") is None:
            return "pytesseract not installed"
        if importlib.util.find_spec("PIL") is None:
            return "Pillow not installed (required by tesseract adapter)"
        return ""

    def recognize(self, crop=None, roi_name="", kind="number") -> OCRResult:
        if not self.available():
            return OCRResult(status=OCRStatus.UNAVAILABLE,
                             error=self.unavailable_reason(),
                             synthetic_test_only=False)
        if crop is None:
            return OCRResult(status=OCRStatus.FAILED, error="crop_missing",
                             synthetic_test_only=False)
        if kind not in ("number", "text"):
            return OCRResult(status=OCRStatus.FAILED,
                             error=f"unsupported kind {kind}",
                             synthetic_test_only=False)
        try:
            from PIL import Image  # noqa: PLC0415 - 可选依赖
            import pytesseract  # noqa: PLC0415 - 可选依赖
            img = Image.frombytes("RGBA", (crop.width, crop.height),
                                  crop.rgba_bytes()).convert("L")
            whitelist = "0123456789.," if kind == "number" else _TEXT_WHITELIST
            cfg = f"--psm {self.psm} -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(img, lang=self.lang,
                                               config=cfg).strip()
        except Exception as e:
            return OCRResult(status=OCRStatus.FAILED,
                             error=f"tesseract error: {e}",
                             synthetic_test_only=False)
        if not text:
            return OCRResult(status=OCRStatus.UNCERTAIN,
                             error="empty OCR output",
                             synthetic_test_only=False)
        if kind == "number":
            value = _parse_number(text)
            if value is None:
                return OCRResult(status=OCRStatus.UNCERTAIN,
                                 raw_text=text,
                                 error=f"unparseable number: {text!r}",
                                 synthetic_test_only=False)
            return OCRResult(value=value, raw_text=text, confidence=None,
                             status=OCRStatus.OK, synthetic_test_only=False)
        return OCRResult(value=text, raw_text=text, confidence=None,
                         status=OCRStatus.OK, synthetic_test_only=False)
