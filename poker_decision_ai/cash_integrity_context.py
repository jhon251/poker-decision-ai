"""V3G-2 — CASH integrity → decision-context **证据旁路**（只读适配层）。

职责（**仅**做证据旁路，不做任何决策）
-------------------------------------
把 V3G-1 的门禁结果（`cash_state_integrity.evaluate_cash_state_integrity`）
包装成 JSON-safe 的决策上下文片段，供生产链的 envelope / report 承载。

硬约束
------
1. **旁路**：返回值**不参与**任何 EV / 动作 / 尺寸 / 候选 / 评分计算；
   调用方只应把它放进 `evidence` / report 字段，**不得**回灌进 scoring。
2. **不改 state**：不修改输入；只读调用门禁（门禁本身亦为只读）。
3. **模式隔离（fail closed）**：仅 `game_type == "cash"` 时生成；
   其它模式（mtt / pko / ko / mystery / 未知 / 缺失）一律返回
   `{"present": False, "reason": ...}`，**绝不**沿用 CASH 记录，
   也**不**放宽字段类型去共享。经由
   `strategy_mode_switch.resolve_mode()`（既有唯一口径）判定。
4. **无默认值**：不补 table_size / dealer / hero / street；
   UNKNOWN 原样传播（门禁语义零改动）。
5. **确定性**：同输入 → 同输出；无随机、无 IO、无缓存。
"""
from __future__ import annotations

from .cash_state_integrity import evaluate_cash_state_integrity

CONTEXT_VERSION = "v3g2-cash-integrity-context-v1"

#: 生产 envelope 中承载旁路的 evidence kind（供下游按 kind 检索）
EVIDENCE_KIND = "state_integrity"


def cash_integrity_context(state: dict | None) -> dict:
    """CASH 状态完整性 → 决策上下文旁路片段（只读；确定性）。

    返回（CASH）:
      {"present": True, "context_version", "mode", "state_integrity",
       "completeness", "level", "missing", "conflicts", "unknown",
       "evidence_grade", "kind"}

    返回（非 CASH / 无法判定）:
      {"present": False, "context_version", "reason": str}
    """
    from .strategy_mode_switch import resolve_mode

    if not isinstance(state, dict):
        return {"present": False, "context_version": CONTEXT_VERSION,
                "reason": "state_not_a_dict"}

    mode, mode_reason = resolve_mode(state)
    if mode != "cash":
        # fail closed：CASH 专属字段不得被其它模式继承
        return {"present": False, "context_version": CONTEXT_VERSION,
                "reason": f"not_cash_mode:{mode}"
                          + (f":{mode_reason}" if mode_reason else "")}

    rep = evaluate_cash_state_integrity(state)
    return {
        "present": True,
        "kind": EVIDENCE_KIND,
        "context_version": CONTEXT_VERSION,
        "mode": mode,
        "state_integrity": rep.to_dict(),
        "completeness": rep.completeness,
        "level": rep.level,
        "missing": list(rep.missing),
        "conflicts": list(rep.conflicts),
        "unknown": list(rep.unknown),
        "evidence_grade": rep.evidence_grade,
    }


__all__ = ["CONTEXT_VERSION", "EVIDENCE_KIND", "cash_integrity_context"]
