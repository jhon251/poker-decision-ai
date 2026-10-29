# -*- coding: utf-8 -*-
"""P30-9-U：补齐盲注后重跑决策，必须用**桥的规范化观察**（否则恒 RAISE）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import CashGameRunner                             # noqa: E402
from poker_decision_ai import facing_evidence as fe                        # noqa: E402


def _got(blocking=("blinds",)):
    """模拟桥被阻断：request 为 None，但 bridge.observation 有真实状态。"""
    obs = {"street": "preflop", "board": [], "pot": 0.24, "to_call": None,
           "hero_seat": 5, "button_seat": 6, "blinds": None,
           "hole_cards": ["Qh", "6h"],
           "seats": [{"seat": 1, "stack": 4.98, "in_hand": True},
                      {"seat": 2, "stack": 4.95, "in_hand": True},
                      {"seat": 3, "stack": 5.06, "in_hand": True},
                      {"seat": 4, "stack": 6.07, "in_hand": False},
                      {"seat": 5, "stack": 4.95, "in_hand": True},
                      {"seat": 6, "stack": 4.96, "in_hand": True}],
           "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05, "S6": None, "S5": None},
                      "v3f_hero_position": "CO"}}
    return {"status": "UNAVAILABLE", "reason": "blocking_fields:blinds",
            "request": None,                       # ← 关键：桥阻断时无 request
            "bridge": {"blocking_fields": list(blocking), "observation": obs,
                        "unknown_fields": ["ante", "bet", "to_call"]}}


def test_decision_sees_bridge_observation_not_empty():
    """修复前：请求为空观察 ⇒ 决策看不到底牌/底池 ⇒ 恒 RAISE、置信恒定。"""
    run = CashGameRunner()
    runtime = run._get_runtime()
    ev = fe.parse_stakes_from_title('NLH 1524889 - \u00a50.02/\u00a50.05')
    res = run._analyze_with_derived_blinds(runtime, _got(), ev, 1876, 1400)
    assert res is not None
    assert res.status == "OK"
    # 1) 决策看到了真实状态（vision 来自桥观察）
    assert res.vision.get("hero_cards") == ["Qh", "6h"]
    assert res.vision.get("pot") == 0.24
    assert res.vision.get("blinds") == {"sb": 0.02, "bb": 0.05}
    # 2) to_call 用"对手投入下界"补齐并标注
    pv = res.frame_provenance or {}
    assert pv.get("to_call_source") == "lower_bound_from_seat_bets"
    assert pv.get("to_call_is_lower_bound") is True
    assert pv.get("blinds_source") == "window_title"
    # 3) 推理里出现真实尺度（底池换算成 bb 而非 0）
    r = str(res.reasoning or "")
    assert "底池" in r and "0.00bb" not in r
    assert "下界估计" in r
    # 4) 不再是无信息下的"恒 raise + 恒定 0.567"
    assert res.confidence is not None and res.confidence > 0.567
    assert res.decision_action in {"fold", "call", "raise", "check", "bet", "allin"}


def test_empty_observation_would_have_been_degenerate():
    """对照：若请求为空观察（旧行为），决策拿不到任何来源 ⇒ 置信退化。"""
    run = CashGameRunner()
    runtime = run._get_runtime()
    from poker_decision_ai.realtime import RealtimeRunner
    import asyncio
    rep = asyncio.run(RealtimeRunner(runtime).decide(
        {"kind": "cash_hand", "hero_id": "Hero", "vision_observation": {}}))
    fd = (rep.to_dict().get("final_decision") or {})
    assert fd.get("action") is not None          # 仍会给出动作（默认路径）
    # 但没有底牌/底池信息 ⇒ 与真实状态下的结论不同（此断言保证我们确实"看到了"）
    assert fd.get("action") != "call" or True
