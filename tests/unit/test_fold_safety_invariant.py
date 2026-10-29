# -*- coding: utf-8 -*-
"""P30-9-ID 回归：**弃牌安全不变量** ✗✗ —— 无法确立"面对下注"时不得给弃牌建议 ✓。

用户实测：界面**能过牌**（免费）却给出 **弃牌** ✗ —— 任何模型下都是**被支配**的动作 ✓。
根因：未观察到按钮集（`legal` 为空 ✗）+ `to_call` 为 None ✗ ⇒ `align_action` 无法对齐 ⇒
fold 原样输出 ✗。本测试锁定修复后的行为 ✓。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.action_semantics import align_final_decision  # noqa: E402


def _fd(action: str) -> dict:
    return {"action": action, "reasoning": "内部求解建议", "recommended_size": None}


def test_fold_becomes_check_when_check_is_legal():
    """**能过牌 ⇒ 弃牌必须被改写成过牌** ✓（免费选项优先 ✓）。"""
    out = align_final_decision(final_decision=_fd("fold"),
                                obs={"available_actions": ["check", "bet"]})
    assert out["action"] == "check", f"应改写为 check，实得 {out.get('action')}"
    assert out.get("recommended_size") is None


def test_fold_allowed_when_facing_a_bet():
    """确实面对下注（合法集只有 fold/call/raise ✓）⇒ 弃牌**允许**保留 ✓。"""
    out = align_final_decision(final_decision=_fd("fold"),
                                obs={"available_actions": ["fold", "call", "raise"]})
    assert out["action"] == "fold", f"面对下注时弃牌应保留 ✓，实得 {out.get('action')}"


def test_fold_allowed_when_to_call_known_positive():
    """`to_call > 0` 明确 ⇒ 面对下注成立 ⇒ 允许弃牌 ✓。"""
    out = align_final_decision(final_decision=_fd("fold"),
                                obs={"to_call": 4.0}, to_call_unknown=False)
    assert out["action"] == "fold"


def test_fold_withdrawn_when_facing_unknown_and_no_check_evidence():
    """既不知能否过牌、也无法确立面对下注 ⇒ **撤回弃牌** ✓（诚实 UNKNOWN ✓，不猜 ✗）。"""
    out = align_final_decision(final_decision=_fd("fold"), obs={"to_call": None})
    assert out["action"] is None, f"应撤回动作，实得 {out.get('action')}"
    assert out.get("block_reason") in ("fold_blocked_facing_unknown",
                                        "facing_unproven_action_withdrawn")
    assert "安全不变量" in str(out.get("reasoning")), "必须在文案里写明原因 ✓"


def test_aggressive_actions_untouched():
    """**进攻性**动作不受该不变量影响 ✓（不误伤 ✓）。"""
    for a in ("check", "raise", "bet"):
        out = align_final_decision(final_decision=_fd(a), obs={"to_call": None})
        assert out["action"] == a, f"{a} 不应被改写，实得 {out.get('action')}"


# ── P30-9-IE：把不变量**推广到 `call`** ✗（用户实测第二例 ✓）──────────────────
def test_call_becomes_check_when_only_check_and_bet_are_legal():
    """用户实测第二例 ✗：界面是 **盖牌/过牌/下注**（**没有跟注** ✓）却建议 **CALL** ✗✗。
    ⇒ `call` 与 `fold` 一样以『面对下注』为前提 ✓，未确立时不得输出 ✓。
    """
    out = align_final_decision(final_decision=_fd("call"),
                                obs={"available_actions": ["check", "bet"]})
    assert out["action"] == "check", f"应改写为 check，实得 {out.get('action')}"


def test_call_allowed_when_call_is_legal():
    """合法集里有 `call`（且无 `check` ✓）⇒ 面对下注成立 ⇒ 跟注保留 ✓。"""
    out = align_final_decision(final_decision=_fd("call"),
                                obs={"available_actions": ["fold", "call", "raise"]})
    assert out["action"] == "call"


def test_call_becomes_check_when_facing_unknown():
    """无合法集、无 `to_call` ⇒ **不得**保留 `call` ✓（改写为 check 或撤回 ✓）。"""
    out = align_final_decision(final_decision=_fd("call"), obs={"to_call": None})
    assert out["action"] in (None, "check"), f"不得保留 call，实得 {out.get('action')}"


def test_call_allowed_when_to_call_known_positive():
    """`to_call > 0` 明确 ⇒ 面对下注成立 ⇒ 跟注允许 ✓。"""
    out = align_final_decision(final_decision=_fd("call"),
                                obs={"to_call": 9.0}, to_call_unknown=False)
    assert out["action"] == "call"


# ── P30-9-IG：**跟注额算术合理性门** ✓（实测：把底池读成了跟注额 ✗）────────────
def test_to_call_equal_to_pot_is_implausible_and_blocks_fold():
    """**实测用户帧**：`to_call == pot`（11,600 == 11,600 ✗）——算术上不可能 ✓
    （底池本身已含那笔下注 ✓）⇒ 不得据此证明面对下注 ✓ ⇒ 弃牌必须被撤回 ✓。
    """
    out = align_final_decision(final_decision=_fd("fold"),
                                obs={"to_call": 11600.0, "pot": 11600.0})
    assert out["action"] is None, f"应撤回，实得 {out.get('action')}"
    assert out.get("block_reason") == "facing_unproven_action_withdrawn"
    assert out.get("to_call_implausible"), "应记录不可信的 to_call（可审计 ✓）"


def test_to_call_less_than_pot_still_proves_facing():
    """合理的跟注额（< 底池 ✓）仍能证明面对下注 ⇒ 弃牌允许保留 ✓（不误伤 ✓）。"""
    out = align_final_decision(final_decision=_fd("fold"),
                                obs={"to_call": 2000.0, "pot": 11600.0})
    assert out["action"] == "fold"


def test_facing_evidence_rejects_to_call_ge_pot():
    """`derived_facing` 同样拒绝不可信的跟注额 ✓（返回 UNKNOWN 且带说明 ✓）。"""
    from poker_decision_ai.facing_evidence import derived_facing
    ev = derived_facing({"to_call": 5000.0, "pot": 5000.0})
    assert ev.get("facing") == "UNKNOWN"
    assert ev.get("source") == "to_call_implausible_ge_pot"
