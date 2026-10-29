"""V3F-P12-OFFLINE — Decision Foundation Components（纯离线薄组件）。

- 纯数据/规则/数学层（deterministic · side-effect free · UNKNOWN-first）；
- 缺失输入显式传播状态（不补值/不推断/不 fallback）；
- 零策略/EV/推荐/动作选择；零 V1 agents import；零 Bet/Action 推断；
- 未来 Decision Engine 的输入基础（DecisionInput 1.0 之下——STOP 在此）。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class ComponentStatus(str, enum.Enum):
    OK = "OK"                      # 计算完成
    UNKNOWN = "UNKNOWN"            # 输入缺失——不可计算
    INVALID_INPUT = "INVALID_INPUT"  # 输入非法（如 to_call<0）
    NOT_APPLICABLE = "NOT_APPLICABLE"  # 场景不适用（如 heads-up legal set）


@dataclass
class Result:
    value: Any
    status: ComponentStatus
    reason: str | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "status": self.status.value,
                "reason": self.reason}


def _num(v: Any) -> float | None:
    """严格数值提取：None/非数值 → None（UNKNOWN——不转 0）。"""
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ── A. Legal Action / Action Set ─────────────────────────────────────
class ActionType(str, enum.Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


def legal_actions(to_call: Any, can_check: bool = True,
                  min_raise_to: Any = None) -> Result:
    """纯规则：给定 to_call（当前需跟注额）→ 合法动作集。

    - to_call UNKNOWN/None → Result(UNKNOWN)（**不推断**——EVIDENCE_LIMITED
      Bet 状态下不得补值）；
    - to_call == 0（明确已知零）→ check 合法、bet/raise 合法（fold 合法）；
    - min_raise_to 缺失 → raise 合法集按 to_call 给出（不加注规则不猜）。
    """
    t = _num(to_call)
    if t is None:
        return Result(None, ComponentStatus.UNKNOWN,
                      "to_call_missing(EVIDENCE_LIMITED)")
    if t < 0:
        return Result(None, ComponentStatus.INVALID_INPUT,
                      "to_call_negative")
    acts = [ActionType.FOLD.value]
    if t == 0:
        if can_check:
            acts.append(ActionType.CHECK.value)
        acts.extend([ActionType.BET.value, ActionType.RAISE.value])
    else:
        acts.append(ActionType.CALL.value)
        acts.extend([ActionType.BET.value, ActionType.RAISE.value])
    return Result(acts, ComponentStatus.OK)


# ── B. Pot Odds ──────────────────────────────────────────────────────
def pot_odds(pot: Any, to_call: Any) -> Result:
    """pot odds = to_call / (pot + to_call)。缺失/非法 → 显式状态。"""
    p, t = _num(pot), _num(to_call)
    if p is None or t is None:
        return Result(None, ComponentStatus.UNKNOWN,
                      "pot_or_to_call_missing")
    if p < 0 or t < 0:
        return Result(None, ComponentStatus.INVALID_INPUT,
                      "negative_input")
    if t == 0:
        return Result(0.0, ComponentStatus.OK)      # 明确已知零跟注
    return Result(round(t / (p + t), 6), ComponentStatus.OK)


# ── C. SPR ──────────────────────────────────────────────────────────
def spr(stack: Any, pot: Any) -> Result:
    """SPR = stack / pot（有效筹码与底池比）。UNKNOWN 传播。"""
    s, p = _num(stack), _num(pot)
    if s is None or p is None:
        return Result(None, ComponentStatus.UNKNOWN,
                      "stack_or_pot_missing")
    if s < 0 or p <= 0:
        return Result(None, ComponentStatus.INVALID_INPUT,
                      "non_positive_pot_or_negative_stack")
    return Result(round(s / p, 4), ComponentStatus.OK)


# ── D. Equity Adapter ───────────────────────────────────────────────
def equity(hero: Any, board: Any, villain: Any) -> Result:
    """薄 adapter → 域 EquityCalculator.hand_vs_hand（不重实现算法）。"""
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    if not hero or not villain:
        return Result(None, ComponentStatus.UNKNOWN,
                      "hero_or_villain_missing")
    try:
        r = EquityCalculator().hand_vs_hand(hero, board or [], villain)
        return Result({"hero": round(r.hero_equity, 6),
                       "villain": round(r.villain_equity, 6),
                       "tie": round(r.tie, 6), "samples": r.samples,
                       "method": r.method}, ComponentStatus.OK)
    except Exception as e:
        return Result(None, ComponentStatus.INVALID_INPUT, str(e)[:80])


# ── E. Range 基础 Adapter ──────────────────────────────────────────
def range_size(range_obj: Any) -> Result:
    """最小 range 接口：组合数（复用域 Range 的表示——不推断 range）。"""
    if range_obj is None:
        return Result(None, ComponentStatus.UNKNOWN, "range_missing")
    try:
        n = len(range_obj)
        return Result(n, ComponentStatus.OK)
    except TypeError:
        return Result(None, ComponentStatus.INVALID_INPUT,
                      "range_not_measurable")
