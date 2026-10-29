# -*- coding: utf-8 -*-
"""P30-9-AY 测试：降级/偏离必须显式可见；同输入必须可复现。"""
from __future__ import annotations

import asyncio
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import CashGameRunner                             # noqa: E402


def test_annotate_marks_degraded_and_deviation():
    fd = {"action": "call", "reasoning": "底池 10bb；最终动作 call。"}
    rep = {"degraded": ["solver", "statistics"], "skipped": ["bluff"],
           "deviation_applied": True,
           "base_decision": {"action": "fold"}}
    out = CashGameRunner._annotate_decision_context(fd, rep)
    assert out["decision_context"]["degraded"] == ["solver", "statistics"]
    assert out["decision_context"]["deviation_applied"] is True
    assert out["decision_context"]["base_action"] == "fold"
    assert "降级" in out["reasoning"]
    assert "EV 层建议 FOLD" in out["reasoning"]
    assert "最终 CALL" in out["reasoning"]


def test_annotate_clean_case_only_reports_equity_used():
    """无降级/无偏离时：不追加任何说明文字；但**如实**记录实际使用的胜率（P30-9-BH）。"""
    fd = {"action": "call", "reasoning": "x"}
    out = CashGameRunner._annotate_decision_context(
        fd, {"degraded": [], "skipped": [], "deviation_applied": False})
    assert "降级" not in out.get("reasoning", "")
    ctx = out.get("decision_context") or {}
    # decision_context 总会带 degraded/skipped/deviation_applied 三个键；
    # 干净局面下**不得**出现降级说明（reasoning 不变），equity_used 可选。
    assert ctx.get("degraded") in (None, [])
    assert ctx.get("skipped") in (None, [])
    assert ctx.get("deviation_applied") in (None, False)
    assert "降级" not in out.get("reasoning", "")
    assert "偏离" not in out.get("reasoning", "")
    if "equity_used" not in ctx:
        assert out["reasoning"] == "x"


def test_same_input_is_reproducible_end_to_end():
    """同输入 3 次 ⇒ 动作/置信完全一致（审计 F3 的回归门）。"""
    from poker_decision_ai import cli as cli_mod
    seats = [{"seat": i, "name": f"seat{i}", "stack": 100.0,
               "in_hand": i <= 2 or i == 5} for i in range(1, 7)]
    obs = {"source": "v3f_snapshot", "street": "flop", "hero_seat": 5,
           "button_seat": 3, "hole_cards": ["Qs", "8d"],
           "board": ["Kd", "7c", "7h"], "seats": seats, "pot": 10.0,
           "blinds": {"sb": 0.05, "bb": 0.15}, "ante": 0.0, "to_call": 5.0,
           "confidence": 0.96,
           "extra": {"v3f_hero_position": "BB", "v3f_bets": {"S1": 5.0}}}

    async def run3():
        runtime = cli_mod.build_runtime()
        outs = []
        for _ in range(3):
            req = {"kind": "cash_hand", "hero_id": "Hero",
                   "vision_observation": copy.deepcopy(obs)}
            ctx = runtime.new_context(cli_mod.new_id("rep"), "cli",
                                       allow_llm=False)
            rep = await runtime.runner.run(req, ctx)
            fd = rep.to_dict().get("final_decision") or {}
            outs.append((fd.get("action"), fd.get("confidence")))
        return outs

    outs = asyncio.run(run3())
    assert len({str(o) for o in outs}) == 1, f"同输入不可复现: {outs}"
