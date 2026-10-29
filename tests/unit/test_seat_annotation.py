# -*- coding: utf-8 -*-
"""P30-9-R：座位下注证据注记（S1/S2/S6 → 你 / SB / BB / BTN / 未识别）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


def _obs(hero_seat, button, bets, pos=None):
    o = {"hero_seat": hero_seat, "button_seat": button, "street": "preflop",
         "seats": [{"seat": i} for i in range(1, 7)],
         "extra": {"v3f_bets": bets}}
    if pos:
        o["extra"]["v3f_hero_position"] = pos
    return o


# ── 用户 22:23 实测帧：hero 座未识别、只有 S5 有值 ───────────────────
def test_annotation_marks_unreadable_and_amounts():
    a = fe.annotate_seat_bets(_obs(None, None, {"S1": None, "S2": None,
                                                 "S6": None, "S5": 0.02}))
    assert "S5: 0.02" in a["text"]
    assert "S1: 未识别/无投入" in a["text"]
    assert a["legend"]["None"].startswith("未识别或无投入")
    assert [it["seat"] for it in a["items"]] == [1, 2, 5, 6]


def test_annotation_marks_hero_and_roles():
    a = fe.annotate_seat_bets(_obs(5, 6, {"S1": 2.0, "S2": 5.0, "S5": 20.0},
                                    pos="CO"))
    txt = a["text"]
    assert "S5(你·CO): 20" in txt          # hero 座 + 位置
    assert "S1(SB): 2" in txt              # 庄家 6 ⇒ SB=1、BB=2、BTN=6
    assert "S2(BB): 5" in txt
    a2 = fe.annotate_seat_bets(_obs(5, 6, {"S6": 3.0}))
    assert "S6(BTN): 3" in a2["text"]


def test_annotation_without_blinds_evidence_is_explicit():
    a = fe.annotate_seat_bets({"extra": {"v3f_bets": {}}})
    assert a["text"] == "（无座位投入证据）"
    assert a["items"] == []


def test_runner_exposes_seat_bets_text():
    run = CashGameRunner()
    txt = run._seat_bets_text(_obs(5, 6, {"S1": 2.0, "S5": 20.0}, pos="CO"))
    assert "你" in txt and "SB" in txt
    assert run._seat_bets_text({}) == "（无座位投入证据）"
    assert run._seat_bets_text(None) == "（无座位投入证据）"
