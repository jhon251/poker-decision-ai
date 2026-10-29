# -*- coding: utf-8 -*-
"""P30-9-L：盲注预填建议（用户 18:01 帧：pot=0.12 + S5=0.05 ⇒ 0.02/0.05）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


def _obs(pot, bets, street="preflop", button=6, hero_seat=5):
    return {"street": street, "pot": pot, "button_seat": button,
            "hero_seat": hero_seat, "blinds": None,
            "seats": [{"seat": i} for i in range(1, 7)],
            "extra": {"v3f_bets": bets}}


# ── ① 强证据（盲注座投入齐全） ───────────────────────────────────────
def test_suggestion_strong_when_blind_seats_readable():
    ev = fe.suggest_blinds(_obs(0.07, {"S1": 0.02, "S2": 0.05, "S5": None}))
    assert ev["strong"] is True
    assert (ev["sb"], ev["bb"]) == (0.02, 0.05)
    assert ev["source"] == "derived_from_posted_blinds"


# ── ② 用户 18:01 实测帧：pot=0.12、仅 S5=0.05 ────────────────────────
def test_suggestion_from_partial_level_and_pot_arithmetic():
    ev = fe.suggest_blinds(_obs(0.12, {"S1": None, "S2": None, "S6": None,
                                        "S5": 0.05}))
    assert ev["strong"] is False
    assert ev["bb"] == 0.05
    assert ev["sb"] == 0.02                     # 0.12 − 0.05 − 0.05
    assert ev["source"] == "partial_posted_level"
    assert ev["detail"]["readable_sum"] == 0.05


def test_suggestion_bb_only_when_sb_not_recoverable():
    # pot 与投入无法复原出白名单内 SB ⇒ 只建议 BB（下游容忍 sb 未知）
    ev = fe.suggest_blinds(_obs(0.15, {"S5": 0.05}))
    assert ev["bb"] == 0.05 and ev["sb"] is None
    assert ev["source"] == "bb_only_from_posted_level"


def test_no_suggestion_for_mixed_levels_or_postflop():
    assert fe.suggest_blinds(_obs(0.20, {"S5": 0.05, "S2": 0.10}))["bb"] is None
    assert fe.suggest_blinds(_obs(0.12, {"S5": 0.05}, street="flop"))["bb"] is None
    assert fe.suggest_blinds(_obs(None, {"S5": 0.05}))["bb"] is None
    # 非白名单水平（如 0.07）⇒ 不采用
    assert fe.suggest_blinds(
        _obs(0.14, {"S5": 0.07}))["source"] == "level_not_whitelisted"


def test_suggestion_is_never_auto_adopted():
    """建议只用于预填：runner 的自动证据链不含 suggest_blinds。"""
    import inspect
    src = inspect.getsource(CashGameRunner.analyze)
    assert "suggest_blinds" not in src
    c = fe.evidence_contract()
    assert c["blinds_priority"][0] == "observed"
    assert "no_evidence" == c["blinds_priority"][-1]


# ── BB-only 兜底输入 ────────────────────────────────────────────────
def test_bb_only_override_accepted():
    ev = CashGameRunner._blinds_override_ev({"bb": 0.05})
    assert ev["bb"] == 0.05 and ev["sb"] is None
    assert ev["detail"]["sb_recognized"] is False
    both = CashGameRunner._blinds_override_ev({"sb": 0.02, "bb": 0.05})
    assert both["sb"] == 0.02 and both["detail"]["sb_recognized"] is True


def test_override_still_rejects_invalid():
    for bad in (None, {}, {"bb": 0}, {"bb": "x"}, {"sb": 0.05, "bb": 0.02},
                {"sb": -1, "bb": 0.05}, {"sb": 0, "bb": 0.05}):
        assert CashGameRunner._blinds_override_ev(bad) == {}, bad
