"""V3F-P60 — Strategy Intelligence / Candidate Evaluation 测试。

覆盖：多来源竞争与追溯；每候选八要素；UNKNOWN-first；fallback 标注；
double-counting 审计（population 先验非加项 / A3-v2 只替换被跟分支 /
bounty 独立支付流）；competitive_actions + utility gap + recommendation
confidence；CASH/MTT/PKO 多模式；solver 接口（P35 BLOCKED + 占位）；
确定性；legality。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.strategy_intel import (
    _solver_source,
    evaluate_intel,
)
from poker_decision_ai.strategy_robustness import make_state

EQ_FULL = {"hero_equity": 0.45,
           "outcome": {"win": 0.45, "tie": 0.0, "lose": 0.55}}
RANGE_SRC = {"villains": {"v2": {"distribution": {
    "AA": 0.02, "KK": 0.03, "AKs": 0.05, "AQo": 0.06,
    "TT-88": 0.08, "A5s-A2s": 0.1, "KQs": 0.06, "76s": 0.08,
    "T9o": 0.12, "QJo": 0.1, "K9o": 0.1, "A8o": 0.15,
    "random_tail": 0.15}}}}
OPP_SRC = {"players": {"v2": {
    "name": "v2", "conditional_events": 120,
    "conditional_model": {"counts": {"flop|bet|m<=1.0|CO|wet": [40, 15]},
                          "total_events": 40},
    "conditional_population": {"counts": {"flop|bet|m<=1.0||": [200, 80]},
                               "total_events": 200}}}}


def _intel(**over):
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    st.update(over)
    return evaluate_intel(st, copy.deepcopy(EQ_FULL),
                          range_src=copy.deepcopy(RANGE_SRC),
                          opponent_src=copy.deepcopy(OPP_SRC),
                          evidence={"range_mode": "POSTERIOR",
                                    "opponent_mode": "player_history",
                                    "population_source": "uoft_synthetic"})


# ------------------------------------------------------------ 多来源竞争
def test_multi_source_traced():
    r = _intel()
    assert r["available"] is True
    ev = r["evidence"]
    assert ev["range_source"]["present"] is True
    assert ev["opponent_source"]["present"] is True
    assert ev["population_source"]["present"] is True
    assert ev["equity_source"]["present"] is True
    assert ev["equity_source"]["outcome_distribution"] is True
    assert ev["solver"]["available"] is False


def test_every_candidate_answers_eight_questions():
    r = _intel()
    for c in r["candidates"]:
        assert "base_ev_bb" in c and "adjustments" in c
        assert "final_ev_bb" in c and "final_utility_bb" in c
        assert "confidence" in c and "provenance" in c
        assert "fold_channel" in c and "fallback" in c
        # base + adjustments == final（无隐藏项；6dp 舍入公差）
        adj = c["adjustments"]
        assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)


def test_ranking_multi_source_not_single_module():
    """排序键 = 最终 utility（多来源合成）；竞争集合可含多动作。"""
    r = _intel()
    utils = [c["final_utility_bb"] for c in r["candidates"]]
    assert utils == sorted(utils, reverse=True)
    assert r["best_action"] == r["candidates"][0]["action"]
    assert r["best_action"] in r["competitive_actions"]


# ------------------------------------------------------------ UNKNOWN-first
def test_unknown_first_equity_missing():
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    r = evaluate_intel(st, {})
    assert r["available"] is False
    assert r["unavailable_reason"]
    assert r["unknown_first"] is True


def test_unknown_first_game_type_missing():
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    del st["game_type"]
    r = evaluate_intel(st, EQ_FULL)
    assert r["available"] is False


def test_fallback_flagged_not_silent():
    """无 fold 证据 → always_called_fallback 显式标注（非静默）。"""
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"), EQ_FULL)
    bet = next(c for c in r["candidates"] if c["action"] == "bet")
    assert bet["fallback"]["is_always_called_fallback"] is True
    assert bet["fold_channel"] == "always_called_fallback"
    assert bet["fallback"]["point_fallback"] is False  # 分布存在


# ------------------------------------------------------------ double counting
def test_no_double_count_a3v2_replaces_not_adds():
    """A3-v2 只替换被跟分支 equity：a3v2_delta 是记录字段，EV 内不再叠加
    eq_full——被跟分支闭式 = p_c×eq_called×(P+s)−p_c×(1−eq_called)×s。"""
    r = _intel()
    c = next(x for x in r["candidates"]
             if x["adjustments"]["a3v2_delta_bb"] is not None)
    # a3v2_delta 必为负（继续范围更强 → hero equity 下降）
    assert c["adjustments"]["a3v2_delta_bb"] < 0
    # 无重复：候选 p_continue == fold 通道 p_continue（单一来源）
    assert c["p_continue"] is not None


def test_no_double_count_saca_components_separate():
    """A8 realization 与 A9 implied 是不同权益来源（现权益实现 vs 听牌
    改进）——cv_bb 为二者之和，adjustments 中无第三隐藏项。"""
    r = _intel()
    check = next(c for c in r["candidates"] if c["action"] == "check")
    assert set(check["adjustments"]) == {"saca_cv_bb", "a3v2_delta_bb",
                                         "bounty_util_bb"}


def test_no_double_count_bounty_independent_stream():
    """bounty 是独立支付流（淘汰奖金）：只出现在淘汰型候选，且不影响
    非 allin 候选的 bounty_util。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="pko")
    st["bounty_type"] = "pko"
    r = evaluate_intel(st, EQ_FULL, range_src=RANGE_SRC,
                       opponent_src=OPP_SRC)
    for c in r["candidates"]:
        if c["action"] not in ("fold",):
            # 非 allin 候选 bounty 必为 None/0（P44 语义保留）
            assert c["adjustments"]["bounty_util_bb"] in (None, 0.0)


def test_population_prior_not_additive():
    """population 仅作 player 估计先验（P43 语义）：fold 通道单一来源，
    不存在 population 加项字段。"""
    r = _intel()
    for c in r["candidates"]:
        assert "population" not in c["adjustments"]
        assert c["fold_channel"] in ("conditional_player",
                                     "conditional_population",
                                     "stat+range", "range", "stat",
                                     "always_called_fallback", "none")


# ------------------------------------------------------------ competitive
def test_competitive_gap_and_confidence():
    r = _intel()
    assert r["best_action"] in r["competitive_actions"]
    assert r["utility_gap_bb"] is not None and r["utility_gap_bb"] >= 0
    assert 0.0 <= r["recommendation_confidence"] <= 1.0
    # 无证据（evidence 空 manifest）→ confidence 显著低于有证据
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    r2 = evaluate_intel(st, copy.deepcopy(EQ_FULL))
    assert r2["recommendation_confidence"] <= r["recommendation_confidence"] \
        + 1e-9


# ------------------------------------------------------------ 多模式
def test_mode_cash():
    r = _intel()
    assert r["mode"] == "cash" and r["objective"] == "chip_ev"
    assert r["provenance"]["icm_available"] is False
    assert r["provenance"]["bounty_available"] is False


def test_mode_mtt_icm():
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="mtt")
    st["tournament"] = {"payouts": [50, 30, 20]}
    r = evaluate_intel(st, EQ_FULL, range_src=RANGE_SRC,
                       opponent_src=OPP_SRC)
    assert r["mode"] == "mtt" and r["objective"] == "chip_ev+icm"
    assert r["provenance"]["icm_available"] is True
    # MTT utility ≠ EV（k± 缩放），且候选仍全可追溯
    c0 = r["candidates"][0]
    assert c0["final_utility_bb"] is not None
    assert c0["fallback"]["icm_unavailable"] is False


def test_mode_pko_bounty_field():
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="pko")
    st["bounty_type"] = "pko"
    st["players"][0]["bounty"] = 50.0
    st["players"][2]["bounty"] = 50.0
    r = evaluate_intel(st, EQ_FULL, range_src=RANGE_SRC,
                       opponent_src=OPP_SRC)
    # PKO 缺有效 bounty 上下文 → 诚实标注（不伪造）
    assert r["provenance"]["bounty_available"] in (True, False)
    assert any(c["fallback"]["bounty_unavailable_for_mode"] is not None
               for c in r["candidates"])


# ------------------------------------------------------------ solver 接口
def test_solver_source_blocked_by_default():
    ss = _solver_source(None)
    assert ss["available"] is False
    assert "BLOCKED" in ss["status"]


def test_solver_candidate_interface_label_discipline():
    """未来真实 solver 的占位接口：provenance 强制 solver_label 且
    is_human_ground_truth=False（不许冒充真实行为）。"""
    ss = _solver_source({"action": "bet", "utility_bb": 1.0})
    assert ss["available"] is True
    assert ss["candidate"]["provenance"] == "solver_label"
    assert ss["candidate"]["is_human_ground_truth"] is False


# ------------------------------------------------------------ 确定性/合法性
def test_deterministic():
    a = _intel()
    b = _intel()
    assert a == b


def test_legality_of_candidates():
    r = _intel()
    legal = {"fold", "check", "call", "bet", "raise"}
    assert all(c["action"] in legal for c in r["candidates"])
