# -*- coding: utf-8 -*-
"""P30-9-GV 测试：**多人池（≥3）必须明确告知"没有多人 GTO 参考"**。

取证：此前唯一提示藏在 `ev_core.assumptions[2]`（"A4 多人池不建模范围交互"），
用户可见的建议里**没有**说明 ✗ ⇒ 修复后 `multiway_note` 与建议文本都必须写明 ✓。
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _obs(n_active: int) -> dict:
    seats_in = {1, 5} if n_active == 2 else ({1, 3, 5} if n_active == 3 else {1, 2, 3, 5})
    seats = [{"seat": i, "name": f"s{i}", "stack": 100.0,
               "in_hand": i in seats_in} for i in range(1, 7)]
    return {"source": "mw_test", "street": "flop", "hero_seat": 6, "button_seat": 5,
             "hole_cards": ["Kh", "Kc"], "board": ["Qh", "7d", "2c"], "seats": seats,
             "pot": 40.0, "blinds": {"sb": 0.5, "bb": 1.0}, "ante": 0.0, "to_call": 0.0,
             "confidence": 0.98,
             "action_history": [
                 {"street": "preflop", "seat": 5, "action": "raise", "amount": 2.5},
                 {"street": "preflop", "seat": 6, "action": "raise", "amount": 10.0},
                 {"street": "preflop", "seat": 5, "action": "call", "amount": 10.0},
                 {"street": "preflop", "seat": 1, "action": "call", "amount": 10.0}],
             "extra": {"v3f_hero_position": "SB", "v3f_bets": {}}}


async def _decide(n_active: int) -> dict:
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner
    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                            "vision_observation": _obs(n_active), "session_id": "mw_test"})
    return rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)


def _run(n: int) -> dict:
    return asyncio.run(_decide(n))


def test_two_player_has_no_multiway_note():
    dd = _run(2)
    assert dd.get("multiway_note") is None
    assert "没有多人 GTO 参考" not in str((dd.get("final_decision") or {}).get("reasoning"))


def test_three_player_has_explicit_multiway_note():
    dd = _run(3)
    note = dd.get("multiway_note")
    assert note is not None, "3 人池必须给出明确说明"
    assert note["code"] == "MULTIWAY_NO_REFERENCE" and note["n_active"] == 3
    assert "没有多人 GTO 参考" in note["text"]
    assert "没有多人 GTO 参考" in str((dd.get("final_decision") or {}).get("reasoning"))
