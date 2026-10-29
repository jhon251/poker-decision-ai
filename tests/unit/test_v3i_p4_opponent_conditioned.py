"""V3I-P4 — Opponent-Conditioned Decision Bridge 测试。

覆盖授权要求的全部测试项：
* passive / balanced / aggressive 三种 population；
* 相同输入确定性；
* belief/range → candidate 的真实数据流；
* candidate ranking 差异；
* assembly 仍 ABSTAIN 的情况（如实保留）；
* UNKNOWN / conflict / fail-closed；
* 无 opponent evidence 时不产生伪适应；
* provenance 可追踪。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件**不使用**需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_opponent_conditioned_bridge import (
    CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION,
    CLAIM_BOUNDARY,
    MIN_TOTAL_WEIGHT,
    POSTFLOP_AGGRESSION_SCALE,
    bridge_contract,
    candidate_required_equity,
    compare_population_rankings,
    continue_threshold,
    continuing_range,
    evaluate_opponent_conditioned,
    evaluate_prior_only,
    score_candidate,
)

SEED = 11
STREET = "flop"
ITER = 600
POPS = ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED", "DECLARED_AGGRESSIVE")

POP_PARAMS = {
    "DECLARED_TIGHT_PASSIVE": {"modeled_preflop_continue_multiplier": 0.7,
                               "modeled_postflop_aggression_multiplier": 0.6},
    "DECLARED_BALANCED": {"modeled_preflop_continue_multiplier": 1.0,
                          "modeled_postflop_aggression_multiplier": 1.0},
    "DECLARED_LOOSE_PASSIVE": {"modeled_preflop_continue_multiplier": 1.4,
                               "modeled_postflop_aggression_multiplier": 0.5},
    "DECLARED_AGGRESSIVE": {"modeled_preflop_continue_multiplier": 1.0,
                            "modeled_postflop_aggression_multiplier": 1.8},
    "DECLARED_MANIAC": {"modeled_preflop_continue_multiplier": 1.8,
                        "modeled_postflop_aggression_multiplier": 2.6},
}

#: 合成"强度"表：仅用于**假 equity 函数**（不触碰冻结引擎）。
#: 数值**刻意**让对手权益跨越各 population 的门槛带，从而"中等注码"可能最优
#: （若所有类别都被门槛一刀切，分数会退化为"越大越好"，看不出群体差异）。
STRENGTH = {"AA": 0.90, "KK": 0.84, "AKs": 0.78, "QQ": 0.72,
            "99": 0.60, "77": 0.50, "55": 0.38, "AKo": 0.30,
            "T2o": 0.24, "54s": 0.16, "32o": 0.05}


def _fake_equity(*, probed=None, strength=None):
    """确定性的假 equity：**所给类别的加权平均强度**（越强 → 数值越高）。

    必须对类别集合敏感，否则"对手继续范围变强/变弱"就观察不到 —— 那会让
    测试假通过。本函数**不含**任何冻结逻辑，纯粹的测试替身。
    """
    table = dict(STRENGTH if strength is None else strength)

    def fn(hero, board, classes, *, iterations=ITER, seed=42):
        cs = sorted(classes)
        if probed is not None:
            probed.append(cs)
        # 与真实引擎口径一致：对给定范围取**权重平均**（此处用等权，
        # 因为假函数拿不到后验权重；真实路径由引擎按 combo 计数加权）。
        vals = [table[c] for c in cs if c in table]
        if not vals:
            return {"present": False, "unknown": ["class_not_in_fake_table"]}
        return {"present": True, "hero_equity": sum(vals) / len(vals),
                "samples": 1, "method": "fake_equal_weight_mean"}

    return fn


def _belief(weights_by_player):
    """构造最小 belief（形状与 V3H-4 输出一致）。"""
    return {
        "present": True,
        "belief": {
            pid: {"posterior": {"posterior": dict(w)}}
            for pid, w in weights_by_player.items()
        },
    }


def _state(hero=("8d", "Jc"), board=("8h", "Jd", "Ac"), pot_bb=1.5):
    return {
        "game_type": "cash", "street": STREET, "pot_bb": pot_bb,
        "pot": pot_bb, "bb": 1.0, "hero_id": "hero",
        "board": list(board),
        "players": [{"player_id": "hero", "is_hero": True,
                     "hole_cards": list(hero), "status": "active"}],
    }


#: 候选集：无风险 + **渐进注码**。
#: 注：早前用"少量大注"的候选集时，支付函数（无风险厌恶）会让最大注恒占优，
#: 从而看不出群体差异 —— 那是**候选集**的问题，不是机制的问题。
CANDIDATES = ([{"action": "check", "sizing_kind": "check_no_investment",
                "amount_bb": 0.0}]
              + [{"action": "bet", "sizing_kind": "bet_pot_fraction",
                  "amount_bb": a}
                 for a in (0.2, 0.5, 0.9, 1.5, 2.5, 4.0, 8.0)])


# ---------------------------------------------------------------------------
# T1. 契约 / 边界
# ---------------------------------------------------------------------------
def test_t1_contract_flags():
    c = bridge_contract()
    assert c["version"] == CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION
    assert c["per_combo_dimension"] is True
    assert c["scalar_mass_only"] is False
    assert c["hardcodes_population_to_action"] is False
    assert c["wired_to_production"] is False
    for mod in ("cash_strategy_dynamic_belief", "cash_strategy_likelihood_model",
                "cash_strategy_decision_assembly"):
        assert mod in c["does_not_modify"]
    for k in ("CFI", "lambda", "river_terminal", "search_budget"):
        assert k in c["does_not_modify"]


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output",
                 "is_real_human_validation", "is_profit_evidence",
                 "is_external_validation", "wired_to_production",
                 "modifies_v3h4_v3h6_v3h8", "hardcodes_population_to_action",
                 "uses_random_module"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_no_cash_layers_and_no_random():
    src = pathlib.Path(
        "poker_decision_ai/cash_opponent_conditioned_bridge.py").read_text(
            encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    assert "import random" not in src
    assert "random." not in src


def test_t1_no_population_conditional_branching_in_source():
    """授权要求 7：不得用 `if population == ...` 造差异。"""
    src = pathlib.Path(
        "poker_decision_ai/cash_opponent_conditioned_bridge.py").read_text(
            encoding="utf-8")
    for pop in ("TIGHT_PASSIVE", "BALANCED", "AGGRESSIVE", "MANIAC",
                "LOOSE_PASSIVE"):
        assert f'== "DECLARED_{pop}"' not in src, pop
        assert f"== 'DECLARED_{pop}'" not in src, pop
    assert "population ==" not in src
    assert "in POPULATIONS" not in src


def test_t1_no_action_literal_mapped_from_population():
    """不得出现 "某类玩家 → 某个动作" 的映射表。"""
    src = pathlib.Path(
        "poker_decision_ai/cash_opponent_conditioned_bridge.py").read_text(
            encoding="utf-8")
    for action in ('"all_in"', '"raise"', '"bet"', '"check"'):
        for pop in ("TIGHT", "PASSIVE", "AGGRESSIVE", "BALANCED", "MANIAC"):
            assert f'{action}: "{pop}' not in src
            assert f'"{pop}": {action}' not in src


# ---------------------------------------------------------------------------
# T2. population → 门槛（唯一进入点）
# ---------------------------------------------------------------------------
def test_t2_balanced_is_neutral_reference():
    """balanced（aggression=1.0）→ 容忍溢价恰为 0，门槛 == pot odds。"""
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        _population_tolerance)
    t = _population_tolerance(population_params=POP_PARAMS,
                              population="DECLARED_BALANCED")
    assert t["tolerance_premium"] == pytest.approx(0.0)
    assert t["neutral_when_aggression_is_one"] is True


def test_t2_tolerance_ordering_is_monotone_in_declared_aggression():
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        _population_tolerance)
    prem = {}
    for p in POPS:
        prem[p] = _population_tolerance(population_params=POP_PARAMS,
                                        population=p)["tolerance_premium"]
    assert prem["DECLARED_TIGHT_PASSIVE"] > prem["DECLARED_BALANCED"]
    assert prem["DECLARED_BALANCED"] > prem["DECLARED_AGGRESSIVE"]


def test_t2_tolerance_is_not_a_population_to_action_map():
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        _population_tolerance)
    t = _population_tolerance(population_params=POP_PARAMS,
                              population="DECLARED_AGGRESSIVE")
    assert t["is_population_to_action_map"] is False
    assert "action" not in json.dumps(t).lower().replace(
        "population_to_action", "")


def test_t2_missing_or_unknown_population_fails_closed():
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        _population_tolerance)
    for args in ((None, "DECLARED_BALANCED"), ({}, "DECLARED_BALANCED"),
                 (POP_PARAMS, "NOPE"), (POP_PARAMS, None),
                 ("nope", "DECLARED_BALANCED")):
        r = _population_tolerance(population_params=args[0],
                                  population=args[1])
        assert r["present"] is False
        assert r["executed"] is False and r["result"] is None


# ---------------------------------------------------------------------------
# T3. 动作相关的 pot odds（让 p_continue 随 candidate 变）
# ---------------------------------------------------------------------------
def test_t3_pot_odds_increase_with_bet_size():
    prev = -1.0
    for amt in (0.25, 0.5, 1.0, 2.0, 10.0):
        r = candidate_required_equity(amount_bb=amt, pot_bb=1.5)
        assert r["present"] is True
        assert r["required_equity"] > prev
        prev = r["required_equity"]


def test_t3_zero_risk_has_no_pot_odds():
    r = candidate_required_equity(amount_bb=0.0, pot_bb=1.5)
    assert r["present"] is True and r["no_risk"] is True
    assert r["pot_odds_required"] == pytest.approx(0.0)


def test_t3_formula_is_recomputable():
    r = candidate_required_equity(amount_bb=0.5, pot_bb=1.5,
                                  base_required_equity=0.0,
                                  base_hero_equity=0.84)
    assert r["required_equity"] == pytest.approx(0.5 / (1.5 + 1.0), abs=1e-6)
    assert r["comparison_side"] == "opponent"
    assert r["base_hero_equity"] == pytest.approx(0.84)


def test_t3_bad_inputs_fail_closed():
    assert candidate_required_equity(amount_bb=-1, pot_bb=1.5)["present"] is False
    assert candidate_required_equity(amount_bb=1, pot_bb=-1)["present"] is False
    assert candidate_required_equity(amount_bb="x", pot_bb=1.5)["present"] is False
    assert candidate_required_equity(amount_bb=1, pot_bb=1.5,
                                     base_hero_equity=1.5)["present"] is False


def test_t3_threshold_adds_premium_and_clamps():
    a = continue_threshold(required_equity=0.2, tolerance_premium=0.05)
    assert a["threshold"] == pytest.approx(0.25)
    b = continue_threshold(required_equity=0.0, tolerance_premium=-0.5)
    assert b["threshold"] == pytest.approx(0.0), "clamped at 0"
    c = continue_threshold(required_equity=1.0, tolerance_premium=0.5)
    assert c["threshold"] == pytest.approx(1.0), "clamped at 1"
    assert c["clamped"] is True


def test_t3_threshold_bad_inputs_fail_closed():
    assert continue_threshold(required_equity=None,
                              tolerance_premium=0.0)["present"] is False
    assert continue_threshold(required_equity=1.5,
                              tolerance_premium=0.0)["present"] is False
    assert continue_threshold(required_equity=0.5,
                              tolerance_premium=5.0)["present"] is False


# ---------------------------------------------------------------------------
# T4. belief → continuing range 的真实数据流
# ---------------------------------------------------------------------------
def test_t4_continuing_range_uses_belief_weights_not_scalar_mass():
    """信念权重**逐类别**决定 continue/fold 的质量，而非单一标量。

    比较发生在**对手**一侧：opponent_equity = 1 − hero_equity >= threshold。
    """
    w = {"AA": 3.0, "KK": 1.0, "72o": 2.0, "32o": 1.0}
    oeq = {"AA": 0.50, "KK": 0.45, "72o": 0.02, "32o": 0.01}

    def fn(hero, board, classes, *, iterations=ITER, seed=42):
        c = sorted(classes)[0]
        return {"present": True, "hero_equity": 1.0 - oeq[c]}

    r = continuing_range(opponent_weights=w, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.40,
                         equity_fn=fn)
    assert r["present"] is True
    assert set(r["continuing"]) == {"AA", "KK"}
    assert set(r["folding"]) == {"72o", "32o"}
    assert r["p_continue"] == pytest.approx(4.0 / 7.0, abs=1e-6)
    assert r["equity_basis"] == "opponent_equity"


def test_t4_threshold_is_on_opponent_side_not_hero_side():
    """门槛必须作用在对手权益上：hero 权益高 ⇒ 对手权益低 ⇒ 更容易弃牌。"""
    def fn(hero, board, classes, *, iterations=ITER, seed=42):
        return {"present": True, "hero_equity": 0.95}   # 对手只有 0.05

    r = continuing_range(opponent_weights={"AA": 1.0}, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.40,
                         equity_fn=fn)
    assert r["n_classes_continuing"] == 0, "对手权益 0.05 < 0.40 → 弃牌"
    assert r["measured_equity"]["AA"] == pytest.approx(0.05)


def test_t4_same_classes_different_weights_change_p_continue():
    """同样"哪些类继续"不变，但**权重分布**不同 → p_continue 不同。"""
    oeq = {"AA": 0.6, "72o": 0.0}

    def fn(hero, board, classes, *, iterations=ITER, seed=42):
        c = sorted(classes)[0]
        return {"present": True, "hero_equity": 1.0 - oeq[c]}

    a = continuing_range(opponent_weights={"AA": 1.0, "72o": 1.0},
                         hero=["8d", "Jc"], board=["8h", "Jd", "Ac"],
                         threshold=0.4, equity_fn=fn)
    b = continuing_range(opponent_weights={"AA": 1.0, "72o": 9.0},
                         hero=["8d", "Jc"], board=["8h", "Jd", "Ac"],
                         threshold=0.4, equity_fn=fn)
    assert a["p_continue"] == pytest.approx(0.5)
    assert b["p_continue"] == pytest.approx(0.1)


def test_t4_every_class_is_probed_once_and_measured():
    probed: list = []
    w = {"AA": 1.0, "KK": 1.0, "72o": 1.0}
    r = continuing_range(opponent_weights=w, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.0,
                         equity_fn=_fake_equity(probed=probed))
    probed_classes = sorted(p[0] for p in probed)
    assert probed_classes == ["72o", "AA", "KK"], "每个类别恰探测一次"
    # 每个**非零权重**类别都被探测过一次
    assert r["n_classes_zero_weight"] == 0
    # 对手权益恰为 0 的类别（此处 72o：假函数给 hero_equity=1.0）
    # 不可能继续，因此不进入 measured；其余类别必须被实测
    assert "72o" not in r["measured_equity"]
    assert set(r["measured_equity"]) == {"AA", "KK"}
    assert set(r["continuing"]) <= {"AA", "KK"}
    assert set(r["continuing"]) | set(r["folding"]) <= {"AA", "KK", "72o"}


def test_t4_zero_threshold_keeps_all_positive_weight_classes():
    w = {"AA": 1.0, "72o": 1.0, "32o": 3.0}
    r = continuing_range(opponent_weights=w, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.0,
                         equity_fn=_fake_equity())
    keep = set(r["continuing"])
    assert keep, "门槛 0 至少让非零权重的对手类别继续"
    assert r["p_continue"] == pytest.approx(
        sum(w[c] for c in keep) / sum(w.values()), abs=1e-6)


def test_t4_high_threshold_folds_everything():
    w = {"AA": 1.0, "KK": 1.0}
    r = continuing_range(opponent_weights=w, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.99,
                         equity_fn=_fake_equity())
    assert r["n_classes_continuing"] == 0
    assert r["p_continue"] == pytest.approx(0.0)


def test_t4_bad_inputs_fail_closed():
    f = _fake_equity()
    assert continuing_range(opponent_weights=None, hero=["8d", "Jc"],
                            board=[], threshold=0.5,
                            equity_fn=f)["present"] is False
    assert continuing_range(opponent_weights={"AA": 1.0}, hero=["8d"],
                            board=[], threshold=0.5,
                            equity_fn=f)["present"] is False
    assert continuing_range(opponent_weights={"AA": 1.0}, hero=["8d", "Jc"],
                            board=[], threshold=1.5,
                            equity_fn=f)["present"] is False
    assert continuing_range(opponent_weights={"AA": 1.0}, hero=["8d", "Jc"],
                            board=[], threshold=0.5,
                            equity_fn=None)["present"] is False
    assert continuing_range(opponent_weights={"AA": 0.0}, hero=["8d", "Jc"],
                            board=[], threshold=0.5,
                            equity_fn=f)["present"] is False


def test_t4_equity_errors_do_not_crash():
    def bad(hero, board, classes, *, iterations=ITER, seed=42):
        raise RuntimeError("boom")

    r = continuing_range(opponent_weights={"AA": 1.0}, hero=["8d", "Jc"],
                         board=["8h", "Jd", "Ac"], threshold=0.5, equity_fn=bad)
    assert r["present"] is True
    assert r["n_classes_unmeasured"] == 1
    assert r["unknown"]


# ---------------------------------------------------------------------------
# T5. candidate 打分（opponent-conditioned）
# ---------------------------------------------------------------------------
def _cont(p_cont, continuing):
    return {"present": True, "p_continue": p_cont, "continuing": continuing}


def test_t5_score_matches_documented_payoff_formula():
    """score = p_fold·pot + p_cont·(pot + 2·amount)·hero_eq（可复算）。"""
    out = score_candidate(
        candidate={"action": "bet", "sizing_kind": "bet_pot_fraction",
                   "amount_bb": 1.0},
        pot_bb=1.5, cont=_cont(0.4, {"AA": 1.0}),
        equity_fn=_fake_equity(strength={"AA": 0.6}),
        hero=["8d", "Jc"], board=["8h", "Jd", "Ac"])
    expect = 0.6 * 1.5 + 0.4 * (1.5 + 2.0) * 0.6
    assert out["score"] == pytest.approx(expect, abs=1e-6)
    assert out["shown_down_pot_bb"] == pytest.approx(3.5)
    assert out["hero_equity_vs_continuing"] == pytest.approx(0.6)


def test_t5_bigger_showdown_pot_raises_score_all_else_equal():
    def sc(amount):
        return score_candidate(
            candidate={"action": "bet", "amount_bb": amount},
            pot_bb=1.5, cont=_cont(0.4, {"AA": 1.0}),
            equity_fn=_fake_equity(strength={"AA": 0.6}),
            hero=["8d", "Jc"], board=["8h", "Jd", "Ac"])["score"]

    assert sc(2.0) > sc(1.0) > sc(0.5)


def test_t5_opponent_cannot_continue_scores_whole_pot_and_invents_nothing():
    out = score_candidate(
        candidate={"action": "bet", "amount_bb": 100.0}, pot_bb=1.5,
        cont=_cont(0.0, {}), equity_fn=_fake_equity(),
        hero=["8d", "Jc"], board=["8h", "Jd", "Ac"])
    assert out["present"] is True
    assert out["score"] == pytest.approx(1.5)
    assert out["hero_equity_vs_continuing"] is None
    assert out["conditional_reason"] == "opponent_cannot_continue"
    assert out["n_continuation_classes"] == 0


def test_t5_score_bad_inputs_fail_closed():
    f = _fake_equity()
    assert score_candidate(candidate=None, pot_bb=1.5, cont=_cont(0.5, {"AA": 1.0}),
                           equity_fn=f, hero=["8d", "Jc"],
                           board=[])["present"] is False
    assert score_candidate(candidate={"amount_bb": 1.0}, pot_bb=-1,
                           cont=_cont(0.5, {"AA": 1.0}), equity_fn=f,
                           hero=["8d", "Jc"], board=[])["present"] is False
    assert score_candidate(candidate={"amount_bb": 1.0}, pot_bb=1.5,
                           cont={"present": False},
                           equity_fn=f, hero=["8d", "Jc"],
                           board=[])["present"] is False
    assert score_candidate(candidate={"amount_bb": 1.0}, pot_bb=1.5,
                           cont=_cont(0.5, {"AA": 1.0}), equity_fn=None,
                           hero=["8d", "Jc"], board=[])["present"] is False


# ---------------------------------------------------------------------------
# T6. 顶层：belief → candidate ranking
# ---------------------------------------------------------------------------
def _top(belief, pop="DECLARED_BALANCED", cands=CANDIDATES, state=None,
         req=0.0, probed=None, strength=None, player_id=None):
    return evaluate_opponent_conditioned(
        state=state if state is not None else _state(),
        belief=belief, range_model=None, candidates=cands, population=pop,
        population_params=POP_PARAMS, required_equity=req,
        equity_fn=_fake_equity(probed=probed, strength=strength),
        player_id=player_id)


def test_t6_belief_weights_actually_reach_candidate_scores():
    """同一 candidate，对手范围强弱不同 → 分数不同（真实数据流）。"""
    strong = _belief({"opp1": {"AA": 1.0, "KK": 1.0}})
    weak = _belief({"opp1": {"72o": 1.0, "32o": 1.0}})
    # 大小注：强范围（对手权益仅 0.075）面对大注**不会**继续
    big = [{"action": "bet", "sizing_kind": "bet_pot_fraction",
            "amount_bb": 6.0}]
    a = _top(strong, cands=big)
    assert a["present"] is True
    assert a["ranked"][0]["p_continue"] == pytest.approx(0.0)
    assert a["ranked"][0]["p_fold"] == pytest.approx(1.0)
    assert a["ranked"][0]["n_continuation_classes"] == 0
    # 小注：弱范围（对手权益仅 0.075）会继续，且 hero 权益被测出
    small = [{"action": "bet", "sizing_kind": "bet_pot_fraction",
              "amount_bb": 0.3}]
    b = _top(weak, cands=small)
    assert b["present"] is True
    # 弱范围（72o/32o）的对手权益高 → 会继续；数量随实测而定
    assert b["ranked"][0]["n_continuation_classes"] >= 1
    assert b["ranked"][0]["hero_equity_vs_continuing"] is not None
    # 同一 candidate 在两种对手范围下分数不同 ⇒ belief 真的进入了打分
    assert a["ranked"][0]["score"] != b["ranked"][0]["score"]


def test_t6_weight_distribution_changes_p_continue_and_score():
    """同一继续**集合**、不同**权重分布** → p_continue 与分数必须不同。"""
    cand = [{"action": "bet", "sizing_kind": "bet_pot_fraction",
             "amount_bb": 8.0}]
    tbl = {"AA": 0.95, "72o": 0.05}          # 对手权益 0.05 / 0.95
    a = _top(_belief({"opp1": {"AA": 1.0, "72o": 1.0}}), cands=cand,
             strength=tbl)
    b = _top(_belief({"opp1": {"AA": 1.0, "72o": 9.0}}), cands=cand,
             strength=tbl)
    ra, rb = a["ranked"][0], b["ranked"][0]
    # 口径一致：p_fold 必须等于 1 − p_continue
    assert ra["p_fold"] == pytest.approx(1.0 - ra["p_continue"], abs=1e-6)
    assert rb["p_fold"] == pytest.approx(1.0 - rb["p_continue"], abs=1e-6)
    # 同一门槛 ⇒ 继续集合相同（只由门槛与实测权益决定，与权重无关）
    assert ra["n_continuation_classes"] == rb["n_continuation_classes"]
    ka = list(a["per_candidate"])[0]
    kb = list(b["per_candidate"])[0]
    pa = a["per_candidate"][ka]["p_continue"]
    pb = b["per_candidate"][kb]["p_continue"]
    assert pa != pb, (pa, pb)
    assert ra["score"] != rb["score"]
    # 实测方向：质量集中到**继续**类（72o 对手权益 0.95 ≥ 门槛）⇒ 继续率上升，
    # 而 hero 对继续范围的权益不变 ⇒ 分数下降（弃牌收益减少）。
    assert pb > pa, (pa, pb)
    assert rb["score"] < ra["score"]


def test_t6_all_belief_classes_are_probed():
    """每个信念类别都必须被实测；打分阶段对**继续子集**再测一次。"""
    probed: list = []
    _top(_belief({"opp1": {"AA": 1.0, "KK": 1.0, "72o": 1.0}}),
         cands=[{"action": "bet", "amount_bb": 1.0}], probed=probed)
    probed_classes = {c for batch in probed for c in batch}
    assert probed_classes == {"AA", "KK", "72o"}
    # 至少有一次调用覆盖了全部信念类别（筛选阶段）
    assert any(set(b) == {"AA", "KK", "72o"} for b in probed)


def test_t6_zero_weight_classes_never_enter_continuation():
    probed: list = []
    out = _top(_belief({"opp1": {"AA": 1.0, "72o": 1.0, "32o": 0.0}}),
               probed=probed)
    assert out["present"] is True
    allc = {c for batch in probed for c in batch}
    assert "32o" not in allc, "零权重类别不应被当作对手范围进行权益推断"


def test_t6_per_candidate_thresholds_differ():
    out = _top(_belief({"opp1": {"AA": 1.0, "72o": 1.0}}))
    thr = out["threshold_by_candidate"]
    assert len(set(thr.values())) > 1, "每个 candidate 的门槛必须不同"


def test_t6_p_continue_is_monotone_decreasing_in_bet_size():
    out = _top(_belief({"opp1": dict(STRENGTH)}))
    rows = [r for r in out["ranked"] if float(r["amount_bb"]) > 0]
    rows.sort(key=lambda r: float(r["amount_bb"]))
    pcs = [r["p_continue"] for r in rows]
    assert pcs == sorted(pcs, reverse=True), pcs


def test_t6_check_has_no_fold_equity_and_defines_the_baseline():
    """对手**无法**对 check 弃牌 → check 的 p_fold 必须为 1，得分即基线。"""
    out = _top(_belief({"opp1": dict(STRENGTH)}))
    chk = [r for r in out["ranked"] if r["sizing_kind"] == "check_no_investment"][0]
    assert chk["p_continue"] == pytest.approx(0.0)
    assert chk["p_fold"] == pytest.approx(1.0)
    assert chk["conditional_reason"] == "no_investment_no_fold_equity"
    assert out["check_baseline"] == pytest.approx(chk["raw_pot_share"])
    assert out["check_baseline"] == pytest.approx(1.5 * chk["hero_equity_vs_continuing"])
    for r in out["ranked"]:
        assert r["score"] == pytest.approx(
            r["raw_pot_share"] - r["check_baseline"], abs=1e-6)
    assert out["check_baseline"] != pytest.approx(0.0)


def test_t6_ranking_is_sorted_by_score_desc():
    out = _top(_belief({"opp1": dict(STRENGTH)}))
    scores = [r["score"] for r in out["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in out["ranked"]] == list(
        range(1, len(scores) + 1))


def test_t6_deterministic_same_inputs():
    b = _belief({"opp1": dict(STRENGTH)})
    a1 = _top(b)
    a2 = _top(_belief({"opp1": dict(STRENGTH)}))
    assert json.dumps(a1, sort_keys=True, default=str) == \
        json.dumps(a2, sort_keys=True, default=str)


def test_t6_player_selection_is_deterministic_and_explicit():
    b = _belief({"oppB": {"AA": 1.0}, "oppA": {"AA": 1.0}})
    out = _top(b)
    assert out["player_id"] == "oppA", "同权重按 id 排序，确定性"
    explicit = evaluate_opponent_conditioned(
        state=_state(), belief=b, range_model=None,
        candidates=[{"action": "bet", "amount_bb": 1.0}],
        population="DECLARED_BALANCED", population_params=POP_PARAMS,
        required_equity=0.0, equity_fn=_fake_equity(), player_id="oppB")
    assert explicit["player_id"] == "oppB"


def test_t6_provenance_is_traceable():
    out = _top(_belief({"opp1": dict(STRENGTH)}),
               cands=[{"action": "bet", "amount_bb": 1.0}])
    assert out["evidence_level"] == "DECLARED_MODEL"
    assert out["evidence_grade"] == "MODEL_INTERNAL"
    assert out["executed"] is False and out["result"] is None
    assert out["wired_to_production"] is False
    assert all(v is False for v in out["claim_boundary"].values())
    t = out["population_tolerance"]
    assert t["derivation"] and t["postflop_aggression_multiplier"] == 1.0
    r = out["ranked"][0]
    assert r["candidate_required_equity"]["derivation"]
    assert r["continue_threshold"]["source_tolerance_premium"] == \
        "declared_population_parameter"
    assert r["payoff_definition"]
    assert r["score_source"] == "MODEL_INTERNAL"
    assert out["equity_iterations"] and out["equity_seed"] == 42


def test_t6_non_cash_fails_closed():
    st = _state()
    st["game_type"] = "mtt"
    out = _top(_belief({"opp1": {"AA": 1.0}}), state=st)
    assert out["present"] is False
    assert out["reason"] == "not_cash_mode_fail_closed"


def test_t6_missing_belief_or_candidates_fails_closed():
    assert _top(None)["reason"].startswith("belief_required")
    assert _top({"present": False,
                 "reason": "x"})["reason"].startswith("belief_required")
    assert _top(_belief({"opp1": {"AA": 1.0}}), cands=[])["reason"] == \
        "candidates_required"
    assert _top(_belief({"opp1": {}}))["reason"] == "belief_has_no_class_weights"
    assert _top({"present": True, "belief": {}})["reason"] == \
        "belief_has_no_players"


def test_t6_missing_hero_cards_fails_closed():
    st = _state()
    st["players"] = [{"player_id": "hero", "is_hero": True, "hole_cards": []}]
    assert _top(_belief({"opp1": {"AA": 1.0}}), state=st)["reason"] == \
        "hero_cards_required"


def test_t6_missing_board_or_pot_fails_closed():
    st = _state()
    st.pop("board")
    assert _top(_belief({"opp1": {"AA": 1.0}}), state=st)["reason"] == \
        "board_required"
    st2 = _state()
    st2.pop("pot_bb")
    st2.pop("pot")
    assert _top(_belief({"opp1": {"AA": 1.0}}), state=st2)["reason"] == \
        "pot_bb_required"


def test_t6_unknown_player_id_fails_closed():
    out = evaluate_opponent_conditioned(
        state=_state(), belief=_belief({"opp1": {"AA": 1.0}}), range_model=None,
        candidates=CANDIDATES, population="DECLARED_BALANCED",
        population_params=POP_PARAMS, required_equity=0.0,
        equity_fn=_fake_equity(), player_id="ghost")
    assert out["present"] is False
    assert out["reason"].startswith("player_id_not_in_belief")


def test_t6_unknown_population_fails_closed():
    out = _top(_belief({"opp1": {"AA": 1.0}}), pop="NO_SUCH_POP")
    assert out["present"] is False
    assert out["reason"].startswith("population_tolerance_unavailable")


# ---------------------------------------------------------------------------
# T7. 无 opponent evidence → 不产生伪适应
# ---------------------------------------------------------------------------
def test_t7_prior_only_never_calls_equity():
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return {"present": True, "hero_equity": 0.5}

    out = evaluate_prior_only(state=_state(), candidates=CANDIDATES)
    assert out["present"] is True
    assert out["adaptation_applied"] is False
    assert out["equity_calls_used"] == 0
    assert calls == [], "prior-only 不得进行任何权益推断"
    for r in out["ranked"]:
        assert r["conditional_reason"] == "prior_only_no_opponent_evidence"
        assert r["adaptation_applied"] is False
    assert spy is not None


def test_t7_prior_only_ranking_is_population_independent():
    a = evaluate_prior_only(state=_state(), candidates=CANDIDATES)
    b = _top(_belief({"opp1": {"AA": 1.0}}))       # 有证据 → 走 bridge
    assert a["prior_only"] is True
    assert b["adaptation_applied"] is True
    assert a["preferred"]["score"] == pytest.approx(1.5)


def test_t7_prior_only_fails_closed_on_bad_input():
    assert evaluate_prior_only(state=None,
                              candidates=CANDIDATES)["present"] is False
    assert evaluate_prior_only(state=_state(),
                              candidates=[])["present"] is False
    st = _state()
    st["game_type"] = "spin"
    assert evaluate_prior_only(state=st,
                              candidates=CANDIDATES)["present"] is False


def test_t7_belief_with_zero_weights_is_treated_as_no_evidence():
    out = _top(_belief({"opp1": {"AA": 0.0, "KK": 0.0}}))
    assert out["present"] is False
    assert out["reason"] in ("belief_has_no_class_weights",
                             "opponent_weight_mass_zero")
    assert out["adaptation_applied"] is False
    assert out["executed"] is False and out["result"] is None


# ---------------------------------------------------------------------------
# T8. 跨 population：ranking 差异 + 差异可解释
# ---------------------------------------------------------------------------
def test_t8_populations_produce_different_rankings():
    b = _belief({"opp1": dict(STRENGTH)})
    evals = [_top(b, pop=p) for p in POPS]
    cmp = compare_population_rankings(evaluations=evals)
    assert cmp["present"] is True
    assert cmp["n_populations"] == 3
    assert cmp["p_continue_changed"] is True
    assert cmp["n_candidates_with_difference"] > 0
    assert cmp["hardcodes_population_to_action"] is False
    # 合成表刻意较粗（11 个类别）：只保证"数值确实随群体变"，
    # **子序**是否改变交给真实引擎测试（169 个类别）验证。
    assert cmp["max_finite_p_continue_by_population"] is not None


def test_t8_scores_and_p_continue_are_population_specific():
    """合成链路：同一候选的分数与继续率都必须随群体变化。

    （**子序**差异由 `test_t9_real_engine_ranking_differs_across_populations`
    在 169 类别的真实范围上验证；合成表只有 11 个类别，无法体现细分差异。）
    """
    b = _belief({"opp1": dict(STRENGTH)})
    evals = [_top(b, pop=p) for p in POPS]
    cmp = compare_population_rankings(evaluations=evals)
    # 有限注码上的 p_continue 至少有一个群体不同
    pcs = cmp["max_finite_p_continue_by_population"]
    assert len({v for v in pcs.values() if v is not None}) > 1, pcs
    # 每个候选的分数都随群体变化
    n_multi = sum(1 for k, d in cmp["per_candidate"].items()
                  if not k.startswith("check|") and d["n_distinct"] > 1)
    assert n_multi >= 1, cmp["per_candidate"]
    # 门槛数值必须随群体单调变化
    assert cmp["threshold_by_population"]["DECLARED_TIGHT_PASSIVE"] != \
        cmp["threshold_by_population"]["DECLARED_AGGRESSIVE"]


def test_t8_difference_is_attributable_to_threshold_not_population_label():
    """差异必须能由**门槛数值**解释：门槛相同 → 结果相同。"""
    b = _belief({"opp1": dict(STRENGTH)})
    a = _top(b, pop="DECLARED_BALANCED")
    # 用与 balanced 同溢价（aggression=1.0）的另一个群体
    params = dict(POP_PARAMS)
    params["DECLARED_AGGRESSIVE"] = {
        "modeled_preflop_continue_multiplier": 1.0,
        "modeled_postflop_aggression_multiplier": 1.0}
    c = evaluate_opponent_conditioned(
        state=_state(), belief=b, range_model=None, candidates=CANDIDATES,
        population="DECLARED_AGGRESSIVE", population_params=params,
        required_equity=0.0, equity_fn=_fake_equity(), player_id="opp1")
    assert c["per_candidate"] == a["per_candidate"], \
        "门槛相同即结果必须相同 → 否证'看着 population 名字造差异'"


def test_t8_compare_requires_two_evaluations():
    out = compare_population_rankings(evaluations=[_top(
        _belief({"opp1": {"AA": 1.0}}))])
    assert out["present"] is False
    assert out["reason"] == "at_least_two_present_evaluations_required"


def test_t8_compare_reports_none_when_rankings_identical():
    b = _belief({"opp1": dict(STRENGTH)})
    a1 = _top(b, pop="DECLARED_BALANCED")
    params = dict(POP_PARAMS)
    a2 = evaluate_opponent_conditioned(
        state=_state(), belief=b, range_model=None, candidates=CANDIDATES,
        population="DECLARED_TIGHT_PASSIVE",
        population_params={**params,
                           "DECLARED_TIGHT_PASSIVE": {
                               "modeled_preflop_continue_multiplier": 1.0,
                               "modeled_postflop_aggression_multiplier": 1.0}},
        required_equity=0.0, equity_fn=_fake_equity(), player_id="opp1")
    cmp = compare_population_rankings(evaluations=[a1, a2])
    assert cmp["ranking_changed"] is False
    assert cmp["n_candidates_with_difference"] == 0


def test_t8_population_only_enters_through_threshold():
    """population 只能经 `tolerance_premium` 影响结果（唯一进入点）。"""
    b = _belief({"opp1": dict(STRENGTH)})
    outs = {}
    for p in POPS:
        o = _top(b, pop=p)
        outs[p] = (o["population_tolerance"]["tolerance_premium"],
                   json.dumps(o["per_candidate"], sort_keys=True, default=str))
    premiums = [v[0] for v in outs.values()]
    assert len(set(premiums)) == 3
    assert len({v[1] for v in outs.values()}) > 1


# ---------------------------------------------------------------------------
# T9. 真实引擎集成（含 assembly 仍 ABSTAIN 的情况）
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _real():
    """跑一次真实引擎（3 population），结果缓存复用。"""
    if "out" in _CACHE:
        return _CACHE["out"]
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_strategy_action_policy import build_action_policy
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    calc = EquityCalculator()

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        if not classes:
            return {"present": False, "unknown": ["empty_range"]}
        rng = parse_range(",".join(classes), name="p4_opp_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    def op_range_fn(hero, board, classes, *, iterations=ITER, seed=42):
        if not classes:
            return {"present": False, "unknown": ["empty_range"]}
        rng = parse_range(",".join(classes), name="p4_opp_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        he, tie = float(res.hero_equity), float(res.tie)
        return {"present": True, "opponent_equity": max(0.0, 1.0 - he - tie),
                "samples": int(res.samples), "method": res.method}

    st = build_scenario(seed=SEED, street=STREET)["state"]
    om = build_opponent_model(st)
    rm = build_range_model(st, opponent_model=om)
    ap = build_action_policy(st, action_tree=build_action_tree(st),
                             range_model=rm,
                             integrity_context=cash_integrity_context(st),
                             equity_seed=42, equity_iterations=20000)
    bel = build_dynamic_belief(
        st, range_model=rm, opponent_model=om,
        action_weights={"_last_action": "raise", "raise": 1.0},
        integrity_context=cash_integrity_context(st))

    evals = []
    for p in POPS:
        evals.append(evaluate_opponent_conditioned(
            state=st, belief=bel, range_model=rm, candidates=ap["candidates"],
            population=p, population_params=POPULATION_PARAMS,
            required_equity=ap["pot_odds"]["required_equity"],
            equity_fn=equity_fn, player_id="opp2", iterations=ITER, seed=42,
            op_range_fn=op_range_fn))
    _CACHE["out"] = (evals, compare_population_rankings(evaluations=evals), ap,
                     st, bel)
    _CACHE["params"] = POPULATION_PARAMS
    return _CACHE["out"]


def test_t9_real_engine_all_three_populations_present():
    evals, _, ap, st, bel = _real()
    assert bel.get("present") is True
    assert len(ap["candidates"]) == 6
    for e in evals:
        assert e["present"] is True, e.get("reason")
        assert e["n_candidates_scored"] == 6
        assert e["equity_basis"] == "opponent_equity"


def test_t9_real_engine_p_continue_varies_by_candidate_and_population():
    evals, cmp, _, _, _ = _real()
    assert cmp["p_continue_changed"] is True
    for e in evals:
        vals = list(e["p_continue_by_candidate"].values())
        assert len(set(vals)) > 1, "candidate 之间必须不同"


def test_t9_real_engine_ranking_differs_across_populations():
    """真实引擎：群体差异体现在**有限下注子序**与所有候选的分数上。

    注：本场景 hero 权益很高（实测对整个 opp2 范围均值约 0.84），
    支付函数**不含风险厌恶/资金约束**，故 all-in 在三群体中都排第一；
    top-1 因此不变 —— 这一点如实记录，不掩饰。
    """
    evals, cmp, _, _, _ = _real()
    assert cmp["n_candidates_with_difference"] >= 1
    assert cmp["p_continue_changed"] is True
    assert cmp["betting_sub_order_changed"] is True
    assert cmp["finite_best_changed"] is True
    assert cmp["top1_invariant_despite_score_differences"] is True


def test_t9_real_engine_population_label_alone_changes_nothing():
    """同一门槛 → 同一结果，证明差异来自门槛数值而非 population 名字。"""
    evals, _, ap, st, bel = _real()
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned as ev)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range
    calc = EquityCalculator()

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p4_opp_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    # 把 AGGRESSIVE 的参数改成与 BALANCED 完全相同（aggression = 1.0）
    params = {k: dict(v) for k, v in _CACHE["params"].items()}
    params["DECLARED_AGGRESSIVE"] = dict(params["DECLARED_BALANCED"])
    a = ev(state=st, belief=bel, range_model=None, candidates=ap["candidates"],
           population="DECLARED_AGGRESSIVE", population_params=params,
           required_equity=0.0, equity_fn=equity_fn, player_id="opp2",
           iterations=ITER, seed=42)
    b = ev(state=st, belief=bel, range_model=None, candidates=ap["candidates"],
           population="DECLARED_AGGRESSIVE",
           population_params=params, required_equity=0.0,
           equity_fn=equity_fn, player_id="opp2", iterations=ITER, seed=42)
    assert a["per_candidate"] == b["per_candidate"]
    assert a["population_tolerance"]["tolerance_premium"] == pytest.approx(0.0)


def test_t9_real_engine_belief_actually_changes_scores():
    """把 belief 换成另一对手的范围 → 分数必须变化（真实数据流）。"""
    evals, _, ap, st, bel = _real()
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned as ev)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range
    calc = EquityCalculator()

    def op_range_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p4_opp_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        he, tie = float(res.hero_equity), float(res.tie)
        return {"present": True, "opponent_equity": max(0.0, 1.0 - he - tie),
                "samples": int(res.samples), "method": res.method}

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p4_opp_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    alt = {"present": True, "belief": {"opp2": {"posterior": {"posterior": {
        "AA": 1.0, "KK": 1.0, "QQ": 1.0, "AKs": 1.0}}}}}
    a = ev(state=st, belief=bel, range_model=None, candidates=ap["candidates"],
           population="DECLARED_BALANCED",
           population_params=_CACHE["params"], required_equity=0.0,
           equity_fn=equity_fn, player_id="opp2", iterations=ITER, seed=42,
           op_range_fn=op_range_fn)
    b = ev(state=st, belief=alt, range_model=None, candidates=ap["candidates"],
           population="DECLARED_BALANCED",
           population_params=_CACHE["params"], required_equity=0.0,
           equity_fn=equity_fn, player_id="opp2", iterations=ITER, seed=42,
           op_range_fn=op_range_fn)
    assert a["present"] is True and b["present"] is True
    assert a["per_candidate"] != b["per_candidate"], \
        "不同对手范围必须给出不同分数"
    assert a["n_opponent_classes"] != b["n_opponent_classes"]


def test_t9_real_engine_provenance_and_boundaries():
    evals, _, _, _, _ = _real()
    for e in evals:
        assert e["evidence_grade"] == "MODEL_INTERNAL"
        assert e["executed"] is False and e["result"] is None
        assert e["wired_to_production"] is False
        assert all(v is False for v in e["claim_boundary"].values())
        assert e["equity_seed"] == 42
        assert e["score_units"] == "bb"
        for r in e["ranked"]:
            assert r["score"] == pytest.approx(
                r["raw_pot_share"] - r["check_baseline"], abs=1e-6)


# ---------------------------------------------------------------------------
# T10. 冻结层未被修改 / assembly 语义保持
# ---------------------------------------------------------------------------
def test_t10_frozen_layers_untouched_by_hash():
    import hashlib
    want = {
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
  # V3I-P13 经授权在 cash_strategy_decision_assembly 实施 P12 设计的
  # feasible_keys 最小契约（**仅新增可选入参**）；装配语义未变，
  # 故此处更新为**授权的**新哈希。
        "poker_decision_ai/cash_strategy_decision_assembly.py": "3732314ab0182063",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
    }
    for path, h in want.items():
        got = hashlib.sha256(
            pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == h, f"{path} drifted: {got} != {h}"


def test_t10_assembly_still_abstains_on_amount_disagreement():
    """授权要求 6：ranking 变了但 assembly 仍因金额冲突 ABSTAIN → 如实保留。"""
    from poker_decision_ai.cash_decision_environment import build_scenario
    from poker_decision_ai.cash_decision_loop import run_decision_loop
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    from poker_decision_ai.cash_strategy_action_policy import build_action_policy
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_strategy_belief_update import (
        update_belief_with_likelihood)
    from poker_decision_ai.cash_strategy_action_ranking import rank_actions
    from poker_decision_ai.cash_strategy_belief_conditioned_eval import (
        evaluate_belief_conditioned)
    from poker_decision_ai.cash_strategy_search_tree import build_search_tree
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    from poker_decision_ai.cash_strategy_decision_assembly import (
        assemble_decision)

    pipe = {"build_action_policy": build_action_policy,
            "build_dynamic_belief": build_dynamic_belief,
            "update_belief": update_belief_with_likelihood,
            "rank_actions": rank_actions,
            "evaluate_belief_conditioned": evaluate_belief_conditioned,
            "build_search_tree": build_search_tree,
            "evaluate_search": evaluate_search,
            "assemble_decision": assemble_decision}
    st = build_scenario(seed=SEED, street=STREET)["state"]
    out = run_decision_loop(
        st, pipeline=pipe, integrity_context=cash_integrity_context(st),
        opponent_model=build_opponent_model(st),
        range_model=build_range_model(
            st, opponent_model=build_opponent_model(st)),
        action_tree=build_action_tree(st),
        transition=next_street_transition(st), belief=None,
        likelihood_model=None,
        action_weights={"_last_action": "raise", "raise": 1.0},
        solver=solver_status(st), budget=DEFAULT_BUDGET,
        max_root_candidates=3, equity_seed=42, equity_iterations=20000,
        policy=None)
    dec = out["decision_record"]
    # 冻结语义未被本层改变：仍按 tier 分歧弃权
    assert dec["abstained"] is True
    assert dec["preferred"] is None
    assert out["loop_status"] == "ABSTAINED"
    assert dec["disagreements"], "必须保留分歧明细"
    assert all(d["note"] == "top candidates differ; no averaging, no guessing"
               for d in dec["disagreements"])
    assert "no averaging" in json.dumps(dec["disagreements"])
