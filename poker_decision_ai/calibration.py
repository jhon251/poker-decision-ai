"""置信度校准（V2 Phase 10 / §二十·二十一）。

确定性指标：
  - Brier Score = mean((p - y)^2)
  - Log Loss    = -mean(y*ln(p) + (1-y)*ln(1-p))（clip 到 [eps,1-eps]）
  - Calibration Curve：按预测概率分桶（默认 10 桶）统计 n/mean_pred/mean_actual
  - Reliability = max |mean_pred - mean_actual|（各桶）

Bluff 校准（§二十一）：预测 bluff_probability，实际标签来自 Showdown——
对手摊牌手牌 < 一对（高牌）记 bluff=True，一对及以上记 False；无摊牌记 None。
标签是确定性近似（注明），绝不因为一手牌判断错误就修改模型（需足够样本）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .domain.poker.cards import parse_card
from .domain.poker.evaluator import evaluate7, score_category

EPS = 1e-9


def brier_score(predictions: list[float], actuals: list[float]) -> float | None:
    if not predictions:
        return None
    return sum((p - y) ** 2 for p, y in zip(predictions, actuals)) / len(predictions)


def log_loss(predictions: list[float], actuals: list[float]) -> float | None:
    if not predictions:
        return None
    total = 0.0
    for p, y in zip(predictions, actuals):
        p = min(1.0 - EPS, max(EPS, float(p)))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(predictions)


def calibration_curve(predictions: list[float], actuals: list[float],
                      buckets: int = 10) -> list[dict]:
    """按预测概率分桶统计（校准曲线 + 可靠性原料）。"""
    if not predictions:
        return []
    pairs = sorted(zip(predictions, actuals))
    n = len(pairs)
    out = []
    for b in range(buckets):
        lo = b / buckets
        hi = (b + 1) / buckets
        members = [(p, y) for p, y in pairs if lo <= p < hi] if b < buckets - 1 else \
                  [(p, y) for p, y in pairs if lo <= p <= hi]
        if not members:
            continue
        out.append({
            "bucket": b, "range": [round(lo, 2), round(hi, 2)],
            "n": len(members),
            "mean_pred": round(sum(p for p, _ in members) / len(members), 4),
            "mean_actual": round(sum(y for _, y in members) / len(members), 4),
        })
    return out


def reliability(predictions: list[float], actuals: list[float],
                buckets: int = 10) -> float | None:
    curve = calibration_curve(predictions, actuals, buckets)
    if not curve:
        return None
    return max(abs(b["mean_pred"] - b["mean_actual"]) for b in curve)


@dataclass
class CalibrationSummary:
    kind: str = ""                      # "decision_confidence" | "bluff"
    n: int = 0
    brier_score: float | None = None
    log_loss: float | None = None
    reliability: float | None = None
    curve: list = field(default_factory=list)
    labeled: int = 0                    # 有实际标签的样本数
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "n": self.n,
            "brier_score": None if self.brier_score is None else round(self.brier_score, 4),
            "log_loss": None if self.log_loss is None else round(self.log_loss, 4),
            "reliability": None if self.reliability is None else round(self.reliability, 4),
            "curve": self.curve, "labeled": self.labeled, "note": self.note,
        }


class CalibrationCollector:
    """收集 (预测概率, 实际标签) 并输出校准摘要。样本不足时明确标注。"""

    def __init__(self, kind: str, min_samples: int = 30) -> None:
        self.kind = kind
        self.min_samples = min_samples
        self.predictions: list[float] = []
        self.actuals: list[float] = []

    def add(self, prediction: float, actual: bool | None) -> None:
        if actual is None:
            return
        self.predictions.append(float(prediction))
        self.actuals.append(1.0 if actual else 0.0)

    def summary(self) -> CalibrationSummary:
        note = ""
        if len(self.actuals) < self.min_samples:
            note = (f"样本不足（{len(self.actuals)} < {self.min_samples}）："
                    f"校准指标仅供参考，不据此修改模型")
        return CalibrationSummary(
            kind=self.kind, n=len(self.actuals),
            brier_score=brier_score(self.predictions, self.actuals),
            log_loss=log_loss(self.predictions, self.actuals),
            reliability=reliability(self.predictions, self.actuals),
            curve=calibration_curve(self.predictions, self.actuals),
            labeled=len(self.actuals), note=note,
        )


# ---------------------------------------------------------------- Bluff 标签
def label_bluff(show_cards: list[str], board: list[str]) -> bool | None:
    """确定性近似标签：摊牌手牌 < 一对（高牌）→ 诈唬 True；否则价值 False。

    近似假设（注明）：摊牌最弱类别=高牌视为诈唬；这是标签代理而非完美定义。
    解析失败/无牌 → None（不参与校准）。
    """
    if not show_cards or len(show_cards) < 2:
        return None
    try:
        cards = [parse_card(str(c)) for c in show_cards] + [parse_card(str(c)) for c in board]
    except ValueError:
        return None
    if len(cards) < 5:
        return None
    score = evaluate7(cards)
    return score_category(score) == 0      # 0 = high_card
