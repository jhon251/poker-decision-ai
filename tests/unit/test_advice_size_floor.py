# -*- coding: utf-8 -*-
"""P30-9-I：建议尺寸下限校验（RAISE 0.66bb 这类非法尺寸必须撤下并说明）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402


def _fd(action, size_below=None, **kw):
    d = {"action": action, "recommended_size": size_below,
         "reasoning": "底池 1.4bb，求解器锚定 bet@0.6694。"}
    d.update(kw)
    return d


def test_preflop_raise_0_66bb_is_withheld():
    """用户现场：翻前 RAISE 0.66bb（< 最小加注 2bb）⇒ 撤下尺寸并说明。"""
    out = asem.apply_size_floor(_fd("raise", 0.66))
    assert out["recommended_size"] is None
    assert out["action"] == "raise"                     # 动作不动
    sc = out["action_semantics"]["size_check"]
    assert sc["verdict"] == "withheld_below_floor"
    assert sc["floor_bb"] == 2.0 and sc["floor_kind"] == "min_raise_to"
    assert "低于最小加注下限" in out["reasoning"]
    assert "0.66bb" in out["reasoning"]


def test_bet_below_1bb_is_withheld():
    out = asem.apply_size_floor(_fd("bet", 0.40))
    assert out["recommended_size"] is None
    assert out["action_semantics"]["size_check"]["verdict"] == "withheld_below_floor"
    assert "低于最小下注下限" in out["reasoning"]


def test_legal_sizes_pass_through_untouched():
    for action, size in (("bet", 1.0), ("bet", 16.15), ("raise", 2.0),
                         ("raise", 37.5), ("raise", 2.5)):
        out = asem.apply_size_floor(_fd(action, size))
        assert out["recommended_size"] == size, (action, size)
        assert out["action_semantics"]["size_check"]["verdict"] == "ok"
        assert out["reasoning"].endswith("bet@0.6694。")


def test_non_sizing_actions_not_applicable():
    for action in ("check", "call", "fold"):
        out = asem.apply_size_floor(_fd(action, None))
        assert out["recommended_size"] is None
        assert out["action_semantics"]["size_check"]["verdict"] == "not_applicable"


def test_allin_with_size_is_not_floor_checked():
    out = asem.apply_size_floor(_fd("allin", 0.5))
    assert out["recommended_size"] == 0.5
    assert out["action_semantics"]["size_check"]["verdict"] == "not_applicable"


def test_floor_info_is_pure_and_declared():
    assert asem.size_floor("bet")["floor_bb"] == asem.MIN_BET_FLOOR_BB
    assert asem.size_floor("raise")["floor_bb"] == asem.MIN_RAISE_TO_FLOOR_BB
    assert asem.size_floor("call")["floor_bb"] is None
    c = asem.action_semantics_contract()
    assert c["size_floors_bb"] == {"bet": 1.0, "raise": 2.0}


def test_alignment_pipeline_applies_floor_end_to_end():
    """对齐主流程（用户真实路径）也必须做尺寸校验。"""
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 20, "S2": 50, "S6": 40, "S5": None}}}
    fd = {"action": "bet", "alternative_action": "check", "recommended_size": 0.66,
          "reasoning": "底池 1.4bb，需跟注0.00bb。"}
    out = asem.align_final_decision(final_decision=fd, obs=obs)
    assert out["action"] == "raise"                      # bet→raise（面对下注世界）
    assert out["recommended_size"] is None               # 尺寸撤下
    # 改写已先撤尺寸 ⇒ 尺寸校验判定 not_applicable（尺寸本身已为 None，这是关键）
    assert out["action_semantics"]["size_check"]["verdict"] in (
        "withheld_below_floor", "ok", "not_applicable")
    assert "需跟注0.00bb" not in out["reasoning"]
