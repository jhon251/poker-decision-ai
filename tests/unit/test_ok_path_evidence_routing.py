# -*- coding: utf-8 -*-
"""P30-9-AA：to_call 未知但看得到对手投入 ⇒ 必须补齐后决策（不走锚点旁路）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb    # noqa: E402
from cash_game_ui.runner import CashGameRunner                              # noqa: E402

# 用户实测场景：Q-high 无成牌无听牌 + 对手下注 75% 底池（0.63），面板 to_call 未知
OBS = {"source": "v3f_snapshot", "street": "turn", "hero_seat": 5,
       "button_seat": 3, "hole_cards": ["Qs", "8d"],
       "board": ["Kd", "7c", "2h", "5s"],
       "seats": [{"seat": i, "stack": 4.95, "in_hand": True} for i in range(1, 7)],
       "pot": 0.84, "blinds": {"sb": 0.02, "bb": 0.05}, "ante": 0.0,
       "to_call": None, "confidence": 0.96,
       "extra": {"v3f_hero_position": "CO", "v3f_bets": {"S2": 0.63}}}


def _patch(monkeypatch):
    def fake_bridge(rgba, w, h, **kw):
        return {"status": "OK", "reason": None,
                "bridge": {"blocking_fields": [], "observation": dict(OBS),
                            "unknown_fields": ["to_call"]},
                "request": {"kind": "cash_hand", "hero_id": "Hero",
                             "vision_observation": dict(OBS)}}
    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)


def test_ok_path_uses_lower_bound_not_anchor_bypass(monkeypatch):
    _patch(monkeypatch)
    run = CashGameRunner()
    res = run.analyze(b"f" * 16, 1876, 1400, allow_recapture=False)
    pv = res.frame_provenance or {}
    assert pv.get("to_call_source") == "lower_bound_from_seat_bets", pv
    assert pv.get("to_call_is_lower_bound") is True
    # 跟注额应是真实量级（不是 0）：0.63
    assert "12.60bb" in str(res.reasoning) or "0.00bb" not in str(res.reasoning)
    # 面对下注时不得给出 bet（翻后 bet 也不合法）——应为 fold/call/raise
    assert res.decision_action in {"fold", "call", "raise", "allin"}, res.decision_action
    assert "所需胜率 0.00" not in str(res.reasoning)


def test_without_opponent_evidence_falls_back_to_frozen_entry(monkeypatch):
    def fake_bridge(rgba, w, h, **kw):
        obs = dict(OBS)
        obs["extra"] = {"v3f_hero_position": "CO", "v3f_bets": {"S2": None}}
        return {"status": "OK", "reason": None,
                "bridge": {"blocking_fields": [], "observation": obs,
                            "unknown_fields": ["to_call"]},
                "request": {"kind": "cash_hand", "hero_id": "Hero",
                             "vision_observation": obs}}
    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    run = CashGameRunner()
    res = run.analyze(b"f" * 16, 1876, 1400, allow_recapture=False)
    # 无对手投入证据 ⇒ 走冻结入口（ADVICE_REFERENCE / UNAVAILABLE），不得伪造动作
    assert res.status in ("ADVICE_REFERENCE", "DECISION_UNAVAILABLE", "UNAVAILABLE"), res.status
