"""V3F-P30-4 — Decision Action Contract（合法动作空间冻结）。

Contract（纯函数/无状态/只读）：
- LEGAL_ACTIONS: fold/check/call/bet/raise/allin；
- derive_legal_actions(decision_input) → 合法动作集（仅依赖已知
  DecisionInput：to_call/hero_stack/pot——绝不扑克规则猜测视觉字段）；
- validate_action(action, legal_actions) → (bool, reason)；
- 不允许 action ∉ legal_actions 通过（Judge 安全门）。

红线：不修改视觉字段语义；UNKNOWN 输入 → 不推导（None/拒绝）；
无默认值填充；无 previous-fallback；无 poker-rule 反推视觉。
"""
from __future__ import annotations

#: 合法动作全集（已冻结）
LEGAL_ACTIONS = ("fold", "check", "call", "bet", "raise", "allin")

#: 动作合法性来源（V2 _available_actions 同构——derive 用已知输入）
#: to_call ≤ 0 → check/bet（hero_stack ≤ pot 时 + allin——全下规则已知）
#: to_call > hero_stack → fold/call（只能跟注全下/弃牌——已派生）
#: 否则 → fold/call/raise
#: （注：此规则是决策层 action-space 规则，非视觉字段推导——
#: 唯一输入 = 已识别数值。）

#: 数值安全门（UNKNOWN → 拒绝派生）
_DECISION_UNKNOWN_SENTINEL = None


def derive_legal_actions(to_call, hero_stack, pot) -> list[str] | None:
    """从已知 DecisionInput 派生合法动作集。

    任一输入为 None/非正数栈 → None（UNKNOWN——不推导）。
    """
    if to_call is None or hero_stack is None or pot is None:
        return None
    try:
        tc = float(to_call)
        hs = float(hero_stack)
        pt = float(pot)
    except (TypeError, ValueError):
        return None
    if hs <= 0 or pt < 0 or tc < 0:
        return None                      # 非法输入 → UNKNOWN（不推导）
    if tc <= 0:
        acts = ["check", "bet"]
        if hs <= pt:
            acts.append("allin")
        return acts
    if hs <= tc:
        return ["fold", "call"]
    return ["fold", "call", "raise"]


def validate_action(action: str, legal_actions: list[str] | None) -> tuple:
    """动作合法性门（Judge 安全门 D4-5）。

    - legal_actions=None（UNKNOWN）→ (False, "legal_actions_unknown")；
    - action ∉ LEGAL_ACTIONS → (False, "unknown_action")；
    - action ∉ legal_actions → (False, "illegal_action");
    - 否则 (True, "").
    """
    if action not in LEGAL_ACTIONS:
        return False, f"unknown_action:{action}"
    if legal_actions is None:
        return False, "legal_actions_unknown"
    if action not in legal_actions:
        return False, f"illegal_action:{action}"
    return True, ""


def is_negative(amount) -> bool:
    """负金额检查（Judge 安全门：非法金额）。"""
    if amount is None:
        return False                     # None=UNKNOWN（非负非法）
    try:
        return float(amount) < 0
    except (TypeError, ValueError):
        return False
