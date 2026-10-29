"""V3F-P11-A3 — Failure Taxonomy（语义分类——区分感知失败与回归）。"""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """测试/评估结果分类——UNKNOWN 类语义 ≠ 普通 FAIL。"""

    PASS = "PASS"                        # 语义正确识别
    UNKNOWN = "UNKNOWN"                  # 识别失败（证据不足——非错误）
    NOT_OBSERVED = "NOT_OBSERVED"        # 无目标（合法语义）
    EVIDENCE_LIMITED = "EVIDENCE_LIMITED"  # 证据边界（Bet——正式封存）
    NOT_READY = "NOT_READY"              # 能力未建（Action/Decision）
    INVALID_INPUT = "INVALID_INPUT"      # 输入非法
    REGRESSION = "REGRESSION"            # 既有能力回退（真失败）


#: 语义非失败类（与 PASS/REGRESSION 区分——不当作普通 FAIL）
NON_FAILURE_OUTCOMES = frozenset({
    Outcome.UNKNOWN, Outcome.NOT_OBSERVED, Outcome.EVIDENCE_LIMITED,
    Outcome.NOT_READY,
})


def classify(status: str | None) -> Outcome:
    """字符串状态 → Outcome（未识别 → UNKNOWN——不回退到默认）。"""
    if status is None:
        return Outcome.UNKNOWN
    s = status.strip().upper()
    try:
        return Outcome(s)
    except ValueError:
        return Outcome.UNKNOWN
