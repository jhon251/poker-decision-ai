"""分层识别层：Level 2 OCR（数字/文字）与 Level 3 卡牌识别。

- OCR 仅用于数字/文字区域（pot / stack / bet / blind / table metadata /
  action 文字）；卡牌识别绝不走 OCR。
- 卡牌识别输出 rank/suit + bbox + confidence + source。
- 所有后端输出只描述所见，不推断牌面含义之外的任何结论。
- 后端不可用 → 明确 UNAVAILABLE（值 None + uncertainty），绝不伪造。
- 真实后端（tesseract 等）位于 sibling 包 vision2_platform（本包保持
  零 OCR/CV 第三方依赖，兼容 V0 红线扫描）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .coords import Rect
from .image import ImageData


# ---------------------------------------------------------------- OCR（Level 2）
class OCRStatus(str, Enum):
    OK = "OK"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class OCRResult:
    value: Any = None               # float | str | dict | None（按 kind）
    raw_text: str | None = None
    confidence: float | None = None
    status: OCRStatus = OCRStatus.UNAVAILABLE
    error: str | None = None
    synthetic_test_only: bool = True
    extra: dict | None = None       # 附加视觉事实（如 text 的 actor_seat）

    def to_dict(self) -> dict:
        return {"value": self.value, "raw_text": self.raw_text,
                "confidence": self.confidence, "status": self.status.value,
                "error": self.error, "extra": self.extra,
                "synthetic_test_only": self.synthetic_test_only}


class OCRBackend(ABC):
    name = "ocr"
    synthetic_test_only = True

    @abstractmethod
    def recognize(self, crop: ImageData | None, roi_name: str,
                  kind: str = "number") -> OCRResult:
        """识别 ROI 裁剪区域。kind: number | text | seat_stacks | table_meta。

        crop 为 None（ROI_MISSING）时实现必须返回 FAILED/UNCERTAIN，不猜测。
        """

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""


class UnavailableOCRBackend(OCRBackend):
    """无 OCR 后端时的诚实占位：一切返回 UNAVAILABLE（不伪造）。"""
    name = "unavailable_ocr"

    def recognize(self, crop=None, roi_name="", kind="number") -> OCRResult:
        return OCRResult(status=OCRStatus.UNAVAILABLE,
                         error="no OCR backend configured")


class SyntheticSceneOCRBackend(OCRBackend):
    """场景地面真值 OCR（synthetic_test_only=True；绝不从像素猜）。

    get_scene: 返回当前 SyntheticScene 的回调；本后端只读取其 ground truth。
    fail_next / uncertain_rois：测试故障注入与不确定性传播用。
    """
    name = "synthetic_scene_ocr"

    def __init__(self, get_scene: Callable[[], Any], *, call_log: list | None = None,
                 fail_next: int = 0, uncertain_rois: set | None = None):
        self.get_scene = get_scene
        self.call_log = call_log
        self.fail_next = fail_next
        self.uncertain_rois = uncertain_rois or set()

    def recognize(self, crop=None, roi_name="", kind="number") -> OCRResult:
        if self.call_log is not None:
            self.call_log.append((roi_name, kind))
        if self.fail_next > 0:
            self.fail_next -= 1
            return OCRResult(status=OCRStatus.FAILED, error="synthetic_injected_failure")
        if roi_name in self.uncertain_rois:
            return OCRResult(status=OCRStatus.UNCERTAIN,
                             error="synthetic_injected_uncertainty")
        scene = self.get_scene()
        truth = scene.truth.get(roi_name)
        if truth is None:
            return OCRResult(status=OCRStatus.FAILED, error=f"no truth for {roi_name}")
        if kind == "number":
            return OCRResult(value=truth.number, raw_text=str(truth.number),
                             confidence=0.99, status=OCRStatus.OK)
        if kind == "text":
            return OCRResult(value=truth.text, raw_text=truth.text,
                             confidence=0.99, status=OCRStatus.OK,
                             extra={"actor_seat": truth.actor_seat})
        if kind == "seat_stacks":
            return OCRResult(value=truth.seat_stacks, confidence=0.99,
                             status=OCRStatus.OK)
        if kind == "seats":
            seats = {s: {"occupied": t.occupied, "stack": t.stack,
                         "current_bet": t.current_bet, "folded": t.folded,
                         "all_in": t.all_in, "in_hand": t.in_hand,
                         "visible_action": t.visible_action}
                     for s, t in (truth.seats or {}).items()}
            return OCRResult(value=seats, confidence=0.99,
                             status=OCRStatus.OK)
        if kind == "table_meta":
            return OCRResult(value={
                "blinds": truth.blinds,
                "to_call": truth.to_call,
                "hero_seat": scene.hero_seat,
                "button_seat": scene.button_seat,
                "table_size": scene.table_size,
                "poker_variant": scene.poker_variant,
                "game_type": scene.game_type,
            }, confidence=0.99, status=OCRStatus.OK)
        return OCRResult(status=OCRStatus.FAILED, error=f"unknown kind {kind}")


# ---------------------------------------------------------- 卡牌识别（Level 3）
class CardRecognitionStatus(str, Enum):
    OK = "OK"
    UNCERTAIN = "UNCERTAIN"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class DetectedCard:
    rank: str | None
    suit: str | None
    bbox: Rect | None = None        # ROI 内归一化坐标
    confidence: float | None = None
    source: str = "card_recognition"

    def to_dict(self) -> dict:
        return {"rank": self.rank, "suit": self.suit,
                "bbox": None if self.bbox is None else self.bbox.to_dict(),
                "confidence": self.confidence, "source": self.source}


@dataclass
class CardRecognitionResult:
    cards: tuple | None = None      # tuple[DetectedCard, ...]；None=未知
    status: CardRecognitionStatus = CardRecognitionStatus.UNAVAILABLE
    error: str | None = None
    synthetic_test_only: bool = True

    def to_dict(self) -> dict:
        return {"cards": None if self.cards is None
                else [c.to_dict() for c in self.cards],
                "status": self.status.value, "error": self.error,
                "synthetic_test_only": self.synthetic_test_only}


class CardRecognitionBackend(ABC):
    name = "card_recognition"
    synthetic_test_only = True

    @abstractmethod
    def recognize(self, crop: ImageData | None, roi_name: str
                  ) -> CardRecognitionResult:
        """识别 ROI 内卡牌。crop None → 不猜测；未知牌以 None 元素表达。"""

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""


class UnavailableCardBackend(CardRecognitionBackend):
    name = "unavailable_cards"

    def recognize(self, crop=None, roi_name="") -> CardRecognitionResult:
        return CardRecognitionResult(status=CardRecognitionStatus.UNAVAILABLE,
                                     error="no card recognition backend configured")


class SyntheticSceneCardBackend(CardRecognitionBackend):
    """场景地面真值卡牌识别（synthetic_test_only=True）。"""
    name = "synthetic_scene_cards"

    def __init__(self, get_scene: Callable[[], Any], *,
                 call_log: list | None = None,
                 uncertain_rois: set | None = None):
        self.get_scene = get_scene
        self.call_log = call_log
        self.uncertain_rois = uncertain_rois or set()

    def recognize(self, crop=None, roi_name="") -> CardRecognitionResult:
        if self.call_log is not None:
            self.call_log.append(roi_name)
        scene = self.get_scene()
        truth = scene.truth.get(roi_name)
        if truth is None:
            return CardRecognitionResult(status=CardRecognitionStatus.FAILED,
                                         error=f"no truth for {roi_name}")
        cards = truth.cards
        if roi_name in self.uncertain_rois:
            return CardRecognitionResult(
                cards=tuple(DetectedCard(rank=None, suit=None, confidence=0.1)
                            for _ in (cards or [""])),
                status=CardRecognitionStatus.UNCERTAIN,
                error="synthetic_injected_uncertainty")
        if cards is None:
            return CardRecognitionResult(cards=None,
                                         status=CardRecognitionStatus.UNCERTAIN,
                                         error="cards_unknown")
        n = max(len(cards), 1)
        detected = tuple(
            DetectedCard(rank=(c[0] if c else None), suit=(c[1] if c else None),
                         bbox=Rect(i / n, 0.0, 1.0 / n, 1.0),
                         confidence=0.99, source=self.name)
            for i, c in enumerate(cards)
        )
        return CardRecognitionResult(cards=detected,
                                     status=CardRecognitionStatus.OK)
