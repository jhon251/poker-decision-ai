# -*- coding: utf-8 -*-
"""P30-9-HH 测试：**翻后 to_call UNKNOWN 必须明确告知**（不得静默当 first_in ✗）。

取证：41 个真实翻后请求里 **25 个（61%）** `to_call` 为 UNKNOWN（完整牌面 19/22 ✗）
⇒ 引擎无法区分"面对下注 / 首行动"，却照常给建议且**没有任何说明** ✗。
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _obs(to_call):
    seats = [{"seat": i, "name": f"s{i}", "stack": 100.0, "in_hand": i in (1, 5)}
             for i in range(1, 7)]
    o = {"source": "tc_test", "street": "flop", "hero_seat": 6, "button_seat": 5,
          "hole_cards": ["Kh", "Kc"], "board": ["Qh", "7d", "2c"], "seats": seats,
          "pot": 40.0, "blinds": {"sb": 0.5, "bb": 1.0}, "ante": 0.0,
          "confidence": 0.98,
          "action_history": [{"street": "preflop", "seat": 5, "action": "raise",
                               "amount": 2.5},
                              {"street": "preflop", "seat": 6, "action": "raise",
                               "amount": 10.0},
                              {"street": "preflop", "seat": 5, "action": "call",
                               "amount": 10.0}],
          "extra": {"v3f_hero_position": "SB", "v3f_bets": {}}}
    if to_call is not None:
        o["to_call"] = to_call
    return o


def _decide(to_call) -> dict:
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner
    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    rep = asyncio.run(rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                  "vision_observation": _obs(to_call),
                                  "session_id": "tc_test"}))
    return rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)


def test_unknown_to_call_is_disclosed():
    dd = _decide(None)
    note = dd.get("to_call_note")
    assert note is not None, "翻后 to_call UNKNOWN 必须给出明确告知"
    assert note["code"] == "TO_CALL_UNKNOWN"
    assert "面对下注" in note["text"]
    assert "面对下注" in str((dd.get("final_decision") or {}).get("reasoning"))


def test_explicit_zero_has_no_note():
    """显式 to_call=0（确实首行动）⇒ 不该有告知 ✓。"""
    dd = _decide(0.0)
    assert dd.get("to_call_note") is None
