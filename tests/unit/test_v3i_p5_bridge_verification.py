"""V3I-P5 — P4 bridge 受控稳定性与因果一致性验证测试。

覆盖授权全部要求（1–13），并特别检查 P4 已披露的 **6 个历史缺陷**是否回归。

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

from poker_decision_ai.cash_bridge_verification import (
    CASH_P5_VERIFICATION_VERSION,
    CHAIN_LINKS,
    CLAIM_BOUNDARY,
    P4_DEFECT_PATTERNS,
    REQUIRED_EVIDENCE_GRADE,
    VALID_BOARD_SIZE,
    VALID_POPULATIONS,
    VALID_STREETS,
    audit_p4_regression_patterns,
    check_opponent_range_validity,
    check_postconditions,
    check_scenario,
    counterfactual_population_swap,
    no_evidence_probe,
    stable_fingerprint,
    verification_contract,
    verify_causal_chain,
)

SEED = 11
STREET = "flop"
ITER = 300
OPP = "opp2"
POPS5 = ("DECLARED_TIGHT_PASSIVE", "DECLARED_LOOSE_PASSIVE",
         "DECLARED_BALANCED", "DECLARED_AGGRESSIVE", "DECLARED_MANIAC")
BRIDGE_SRC = pathlib.Path(
    "poker_decision_ai/cash_opponent_conditioned_bridge.py")


# ---------------------------------------------------------------------------
# T1. 契约与"不新增能力"边界（要求 1、12、13）
# ---------------------------------------------------------------------------
def test_t1_contract_flags():
    c = verification_contract()
    assert c["version"] == CASH_P5_VERIFICATION_VERSION
    assert c["adds_strategy_capability"] is False
    assert "v3i-p4-opponent-conditioned-bridge-v1" in c["reuses"]
    assert tuple(c["valid_populations"]) == VALID_POPULATIONS
    assert tuple(c["valid_streets"]) == VALID_STREETS
    assert c["required_evidence_grade"] == "MODEL_INTERNAL"
    assert set(c["p4_defects_guarded"]) == set(P4_DEFECT_PATTERNS)


def test_t1_claim_boundary_all_false_and_no_ev():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "is_profit_evidence", "is_real_human_validation",
                 "wired_to_production", "adds_strategy_capability",
                 "modifies_p4_bridge", "modifies_v3h4_v3h6_v3h8"):
        assert flag in CLAIM_BOUNDARY


def test_t1_verification_module_imports_only_stdlib():
    """本层不得 import 任何项目内模块（bridge 由调用方注入）。"""
    src = pathlib.Path(
        "poker_decision_ai/cash_bridge_verification.py").read_text(
            encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib", "re"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)


def test_t1_p4_bridge_not_modified_by_p5():
    """要求 1：复用 P4 bridge，不重写 —— 校验其版本常量与公有 API 仍在。"""
    src = BRIDGE_SRC.read_text(encoding="utf-8")
    assert 'CASH_OPPONENT_CONDITIONED_BRIDGE_VERSION = "v3i-p4' in src
    for fn in ("def evaluate_opponent_conditioned", "def evaluate_prior_only",
               "def compare_population_rankings", "def continuing_range",
               "def score_candidate", "def candidate_required_equity"):
        assert fn in src, fn


def test_t1_chain_links_declared():
    assert CHAIN_LINKS == ("population", "opponent_range", "hero_equity",
                           "candidate_score", "ranking")


# ---------------------------------------------------------------------------
# T2. 6 个 P4 历史缺陷的静态回归防线
# ---------------------------------------------------------------------------
def test_t2_all_six_p4_defects_guarded():
    aud = audit_p4_regression_patterns(
        source=BRIDGE_SRC.read_text(encoding="utf-8"))
    assert aud["present"] is True
    assert aud["all_defects_guarded"] is True, aud["per_defect"]
    assert len(aud["per_defect"]) == 6
    for name, d in aud["per_defect"].items():
        assert d["ok"] is True, (name, d)
        assert d["missing_required"] == []
        assert d["forbidden_present"] == []


def test_t2_defect_keys_are_the_six_disclosed_ones():
    assert set(P4_DEFECT_PATTERNS) == {
        "D1_equity_side", "D2_fold_double_count", "D3_check_fake_fold_rate",
        "D4_threshold_anchor", "D5_zero_posterior",
        "D6_candidate_size_artifact"}


def test_t2_audit_detects_an_injected_regression():
    """防线必须真的会报警：把 D2 的写法改回重复计算 → 必须失败。"""
    good = BRIDGE_SRC.read_text(encoding="utf-8")
    bad = good.replace("fold_term = p_fold * pot",
                       "fold_term = p_fold * (pot + amount)")
    assert bad != good
    aud = audit_p4_regression_patterns(source=bad)
    assert aud["all_defects_guarded"] is False
    assert aud["per_defect"]["D2_fold_double_count"]["ok"] is False


def test_t2_audit_detects_injected_ev_relabelling():
    good = BRIDGE_SRC.read_text(encoding="utf-8")
    bad = good + '\nEV_LABEL = {"expected_value": True}\n'
    aud = audit_p4_regression_patterns(source=bad)
    assert aud["external_claim_guard_ok"] is False
    assert "expected_value" in aud["ev_relabelling_terms"]


def test_t2_evidence_grade_is_model_internal():
    src = BRIDGE_SRC.read_text(encoding="utf-8")
    assert f'EVIDENCE_GRADE = "{REQUIRED_EVIDENCE_GRADE}"' in src
    aud = audit_p4_regression_patterns(source=src)
    assert aud["evidence_grade_marker_present"] is True
    assert aud["positive_external_claims"] == []
    assert aud["claim_boundary_all_false_in_source"] is True


def test_t2_audit_fail_closed_on_bad_source():
    for bad in (None, "", "   ", 123):
        r = audit_p4_regression_patterns(source=bad)
        assert r["present"] is False
        assert r["executed"] is False and r["result"] is None


# ---------------------------------------------------------------------------
# T3. 场景前置检查（要求 6）
# ---------------------------------------------------------------------------
def _state(hero=("8d", "Jc"), board=("8h", "Jd", "Ac"), street=STREET,
           pot=1.5, stack=100.0, game="cash"):
    return {"game_type": game, "street": street, "board": list(board),
            "pot_bb": pot, "effective_stack_bb": stack, "hero_id": "hero",
            "action_history": [],
            "players": [{"player_id": "hero", "is_hero": True,
                         "hole_cards": list(hero), "status": "active"},
                        {"player_id": "opp1", "is_hero": False,
                         "hole_cards": [], "status": "active"}]}


def test_t3_valid_states_pass_precheck():
    ok = [
        _state(),
        _state(street="turn", board=("8h", "Jd", "Ac", "5s")),
        _state(street="river", board=("8h", "Jd", "Ac", "5s", "2d")),
        _state(street="preflop", board=()),
    ]
    for st in ok:
        r = check_scenario(state=st)
        assert r["present"] is True
        assert r["valid"] is True, (st["street"], r["violations"])


def test_t3_board_size_must_match_street():
    r = check_scenario(state=_state(street="turn", board=("8h", "Jd", "Ac")))
    assert r["valid"] is False
    assert any("board_size_mismatch" in v for v in r["violations"])


def test_t3_duplicate_and_malformed_cards_rejected():
    r = check_scenario(state=_state(board=("8h", "8h", "Ac")))
    assert any("duplicate_board_card" in v for v in r["violations"])
    r2 = check_scenario(state=_state(board=("8x", "Jd", "Ac")))
    assert any("invalid_board_card" in v for v in r2["violations"])
    r3 = check_scenario(state=_state(hero=("8d", "Jc"), board=("8d", "Jd", "Ac")))
    assert any("hero_board_card_collision" in v for v in r3["violations"])


def test_t3_opponent_hole_cards_must_be_empty():
    st = _state()
    st["players"][1]["hole_cards"] = ["Ah", "Kh"]
    r = check_scenario(state=st)
    assert r["valid"] is False
    assert any("opponent_hole_cards_present" in v for v in r["violations"])


def test_t3_bad_pot_stack_and_mode_rejected():
    assert any("pot_bb_invalid" in v for v in
               check_scenario(state=_state(pot=-1))["violations"])
    assert any("effective_stack_bb_invalid" in v for v in
               check_scenario(state=_state(stack=0))["violations"])
    assert any("not_cash_mode" in v for v in
               check_scenario(state=_state(game="mtt"))["violations"])


def test_t3_hero_cards_required():
    st = _state()
    st["players"][0]["hole_cards"] = []
    assert any("hero_hole_cards_invalid" in v for v in
               check_scenario(state=st)["violations"])


def test_t3_precheck_fail_closed_on_non_dict():
    r = check_scenario(state=None)
    assert r["present"] is False and r["reason"] == "state_not_a_dict"


def test_t3_board_size_table_is_consistent():
    assert VALID_BOARD_SIZE == {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


# ---------------------------------------------------------------------------
# T4. 后置条件（要求 3、6、8、13）
# ---------------------------------------------------------------------------
def _row(amount, pc, raw, score, base, he, ft=None, st_=None):
    return {"action": "bet", "sizing_kind": "bet_pot_fraction",
            "amount_bb": amount, "p_continue": pc, "p_fold": 1.0 - pc,
            "raw_pot_share": raw, "score": score, "check_baseline": base,
            "hero_equity_vs_continuing": he,
            "score_terms": {"fold_term": raw if ft is None else ft,
                            "showdown_term": 0.0 if st_ is None else st_},
            "continue_threshold": {"threshold": 0.1},
            "candidate_required_equity": {"amount_bb": amount}}


def _out(rows, base=0.5, pot=1.5):
    rows = sorted(rows, key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"present": True, "pot_bb": pot, "check_baseline": base,
            "ranked": rows, "preferred": rows[0],
            "n_opponent_classes": 3, "range_anchor": {"n_classes": 3},
            "evidence_grade": "MODEL_INTERNAL",
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "executed": False, "result": None, "wired_to_production": False}


def test_t4_valid_output_passes_postconditions():
    out = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6),
                _row(2.0, 0.2, 1.2, 0.7, 0.5, 0.5),
                _row(0.0, 0.0, 0.5, 0.0, 0.5, 0.5)])
    r = check_postconditions(out=out, state=_state())
    assert r["valid"] is True, r["violations"]


def test_t4_detects_p_fold_inconsistency():
    bad = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6)])
    bad["ranked"][0]["p_fold"] = 0.9        # 应为 0.6
    r = check_postconditions(out=bad, state=_state())
    assert r["valid"] is False
    assert any("defect3:p_fold_mismatch" in v for v in r["violations"])


def test_t4_detects_unsubtracted_baseline():
    bad = _out([_row(1.0, 0.4, 1.0, 1.0, 0.5, 0.6)])   # score 未减 base
    r = check_postconditions(out=bad, state=_state())
    assert r["valid"] is False
    assert any("defect6:score_not_baseline_subtracted" in v
               for v in r["violations"])


def test_t4_detects_fold_term_exceeding_pot():
    bad = _out([_row(1.0, 0.4, 3.0, 2.5, 0.5, 0.6, ft=3.0)])
    r = check_postconditions(out=bad, state=_state())
    assert r["valid"] is False
    assert any("defect2:fold_term_exceeds_pot" in v for v in r["violations"])


def test_t4_detects_showdown_term_over_cap():
    bad = _out([_row(1.0, 0.4, 1.0, 0.9, 0.5, 0.6, st_=99.0)])
    r = check_postconditions(out=bad, state=_state())
    assert r["valid"] is False
    assert any("defect2:showdown_term_exceeds_capped_pot" in v
               for v in r["violations"])


def test_t4_detects_out_of_range_values():
    bad = _out([_row(1.0, 1.5, 1.0, 0.5, 0.5, 0.6)])
    assert any("p_continue_out_of_range" in v for v in
               check_postconditions(out=bad, state=_state())["violations"])
    bad2 = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 1.7)])
    assert any("defect1:hero_equity_out_of_range" in v for v in
               check_postconditions(out=bad2, state=_state())["violations"])
    bad3 = _out([_row(-1.0, 0.4, 1.0, 0.5, 0.5, 0.6)])
    assert any("negative_or_invalid_amount" in v for v in
               check_postconditions(out=bad3, state=_state())["violations"])


def test_t4_detects_ranking_not_sorted_and_rank_gaps():
    bad = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6),
                _row(2.0, 0.2, 1.2, 0.9, 0.5, 0.5)])
    bad["ranked"][0]["score"], bad["ranked"][1]["score"] = 0.1, 0.9
    r = check_postconditions(out=bad, state=_state())
    assert any("ranking_not_sorted_by_score" in v for v in r["violations"])


def test_t4_detects_evidence_grade_tampering():
    bad = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6)])
    bad["evidence_grade"] = "EV"
    r = check_postconditions(out=bad, state=_state())
    assert any("evidence_grade_not_model_internal" in v for v in r["violations"])


def test_t4_detects_claim_boundary_true():
    bad = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6)])
    bad["claim_boundary"] = dict(CLAIM_BOUNDARY, is_gto=True)
    r = check_postconditions(out=bad, state=_state())
    assert any("claim_boundary_not_false" in v for v in r["violations"])


def test_t4_detects_anchor_class_count_mismatch():
    bad = _out([_row(1.0, 0.4, 1.0, 0.5, 0.5, 0.6)])
    bad["range_anchor"] = {"n_classes": 99}
    r = check_postconditions(out=bad, state=_state())
    assert any("defect5:anchor_tests_more_classes_than_posterior" in v
               for v in r["violations"])


def test_t4_fail_closed_output_is_not_a_violation():
    r = check_postconditions(out={"present": False, "reason": "x"},
                             state=_state())
    assert r["valid"] is True and r["applicable"] is False
    assert check_postconditions(out=None,
                                state=_state())["present"] is False


# ---------------------------------------------------------------------------
# T5. 因果链（要求 3）
# ---------------------------------------------------------------------------
def _chain_eval(pop, pc, he, sc, rank, check_score=0.0):
    """合成评估：**两个候选**，且调用方可让 bet/check 分数**交叉**，
    从而"排序随分数变化"真的可观测（同序不算变化）。"""
    rows = [
        {"action": "bet", "sizing_kind": "bet_pot_fraction",
         "amount_bb": 1.0, "p_continue": pc,
         "hero_equity_vs_continuing": he, "score": sc,
         "n_continuation_classes": 2,
         "continue_threshold": {"threshold": 0.1}},
        {"action": "check", "sizing_kind": "check_no_investment",
         "amount_bb": 0.0, "p_continue": 0.0,
         "hero_equity_vs_continuing": 0.5, "score": check_score,
         "n_continuation_classes": 2,
         "continue_threshold": {"threshold": 0.0}},
    ]
    rows.sort(key=lambda r: -float(r["score"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"present": True, "population": pop,
            "population_tolerance": {"tolerance_premium": 0.0},
            "ranked": rows,
            "per_candidate": {f"{r['action']}|{r['sizing_kind']}|"
                              f"{r['amount_bb']}": r for r in rows},
            "n_opponent_classes": 5}


def test_t5_chain_complete_when_all_links_vary():
    # bet/check 分数**交叉** ⇒ 排序真的改变
    ev = [_chain_eval("A", 0.1, 0.6, 0.2, "x", check_score=0.5),
          _chain_eval("B", 0.3, 0.4, 0.5, "x", check_score=0.1)]
    r = verify_causal_chain(evaluations=ev)
    assert r["present"] is True
    assert r["chain_complete"] is True
    assert r["first_broken_link"] is None
    assert all(r["chain_links"].values())


def test_t5_chain_reports_first_broken_link():
    ev = [_chain_eval("A", 0.1, 0.6, 0.2, "x"),
          _chain_eval("B", 0.1, 0.6, 0.2, "x")]
    r = verify_causal_chain(evaluations=ev)
    assert r["chain_complete"] is False
    assert r["first_broken_link"] == "opponent_range"


def test_t5_chain_needs_two_present_evaluations():
    r = verify_causal_chain(evaluations=[_chain_eval("A", 0.1, 0.6, 0.2, "x")])
    assert r["present"] is False
    assert r["reason"] == "at_least_two_present_evaluations_required"


# ---------------------------------------------------------------------------
# T6. 反事实（要求 4、5）
# ---------------------------------------------------------------------------
def test_t6_counterfactual_pass_and_restore_identical():
    base = _chain_eval("BALANCED", 0.2, 0.6, 0.3, "a")
    swap = _chain_eval("BALANCED", 0.05, 0.5, 0.8, "b")
    same = _chain_eval("BALANCED", 0.2, 0.6, 0.3, "a")
    r = counterfactual_population_swap(baseline_out=base, swapped_out=swap,
                                       identical_params_out=same)
    assert r["counterfactual_changes_result"] is True
    assert r["restore_is_field_identical"] is True
    assert r["pass"] is True


def test_t6_counterfactual_detects_non_determinism():
    base = _chain_eval("BALANCED", 0.2, 0.6, 0.3, "a")
    same = _chain_eval("BALANCED", 0.2, 0.6, 0.99, "a")   # 恢复后不一致
    r = counterfactual_population_swap(baseline_out=base,
                                       swapped_out=base,
                                       identical_params_out=same)
    assert r["restore_is_field_identical"] is False
    assert r["pass"] is False


def test_t6_counterfactual_fail_closed_on_missing_output():
    r = counterfactual_population_swap(baseline_out=None, swapped_out={},
                                       identical_params_out={})
    assert r["present"] is False


def test_t6_stable_fingerprint_is_order_insensitive_for_dicts():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert stable_fingerprint(a) == stable_fingerprint(b)
    assert stable_fingerprint(a) != stable_fingerprint({"x": 2, "y": {}})


# ---------------------------------------------------------------------------
# T7. 无证据不伪适应（要求 7）
# ---------------------------------------------------------------------------
def test_t7_identical_prior_only_outputs_pass():
    po = {"present": True, "adaptation_applied": False,
          "equity_calls_used": 0, "ranked": [{"score": 1.5}]}
    r = no_evidence_probe(per_population_outputs={p: po for p in POPS5})
    assert r["all_identical"] is True
    assert r["pseudo_adaptation_detected"] is False
    assert r["pass"] is True


def test_t7_detects_pseudo_adaptation():
    a = {"present": True, "adaptation_applied": False, "ranked": [{"score": 1.5}]}
    b = {"present": True, "adaptation_applied": False, "ranked": [{"score": 1.6}]}
    r = no_evidence_probe(per_population_outputs={"A": a, "B": b})
    assert r["all_identical"] is False
    assert r["pass"] is False


def test_t7_detects_adaptation_flag_without_evidence():
    po = {"present": True, "adaptation_applied": True, "ranked": []}
    r = no_evidence_probe(per_population_outputs={"A": po, "B": po})
    assert r["pseudo_adaptation_detected"] is True
    assert r["pass"] is False


def test_t7_probe_fail_closed_on_empty():
    assert no_evidence_probe(per_population_outputs={})["present"] is False


# ---------------------------------------------------------------------------
# T8. 对手范围有效性（要求 8）
# ---------------------------------------------------------------------------
def _belief(post, pid=OPP):
    return {"present": True, "belief": {pid: {"posterior": {"posterior": dict(post)}}}}


def test_t8_positive_classes_counted_zero_excluded():
    out = {"present": True, "player_id": OPP, "n_opponent_classes": 2,
           "range_anchor": {"n_classes": 2}}
    bel = _belief({"AA": 1.0, "KK": 0.5, "ZZ": 0.0})
    r = check_opponent_range_validity(out=out, belief=bel, player_id=OPP)
    assert r["n_positive_posterior_classes"] == 2
    assert r["n_zero_weight_classes"] == 1
    assert r["valid"] is True


def test_t8_mismatch_between_out_and_positive_posterior_flagged():
    out = {"present": True, "player_id": OPP, "n_opponent_classes": 3,
           "range_anchor": {"n_classes": 3}}
    bel = _belief({"AA": 1.0, "KK": 0.5, "ZZ": 0.0})
    r = check_opponent_range_validity(out=out, belief=bel, player_id=OPP)
    assert r["valid"] is False
    assert any("class_count_mismatch" in v for v in r["violations"])


def test_t8_folded_player_must_not_be_used():
    out = {"present": True, "player_id": OPP, "n_opponent_classes": 1,
           "range_anchor": {"n_classes": 1}}
    bel = _belief({"AA": 1.0})
    r = check_opponent_range_validity(out=out, belief=bel, player_id=OPP,
                                      folded_players=[OPP])
    assert r["valid"] is False
    assert any("folded_player_used_as_opponent" in v for v in r["violations"])


def test_t8_range_validity_fail_closed():
    assert check_opponent_range_validity(
        out={"present": False}, belief=_belief({"AA": 1.0}))["present"] is False
    assert check_opponent_range_validity(
        out={"present": True}, belief=None)["present"] is False


# ---------------------------------------------------------------------------
# T9. 真实引擎集成：矩阵 + 因果链 + 反事实 + 无证据（要求 2、3、4、6、7）
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _real():
    if "out" in _CACHE:
        return _CACHE["out"]
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned, evaluate_prior_only,
        compare_population_rankings)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    calc = EquityCalculator()

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p5_test_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    scn = dict(seed=SEED, street="flop", pot_bb=2.0, stack_bb=50.0)
    st = build_scenario(**scn)["state"]
    om = build_opponent_model(st)
    rm = build_range_model(st, opponent_model=om)
    bel = build_dynamic_belief(
        st, range_model=rm, opponent_model=om,
        action_weights={"_last_action": "raise", "raise": 1.0},
        integrity_context=cash_integrity_context(st))
    cands = [{"action": "check", "sizing_kind": "check_no_investment",
              "amount_bb": 0.0},
             {"action": "bet", "sizing_kind": "bet_pot_fraction",
              "amount_bb": 0.8},
             {"action": "bet", "sizing_kind": "bet_pot_fraction",
              "amount_bb": 2.0},
             {"action": "bet", "sizing_kind": "bet_allin", "amount_bb": 49.0}]

    def run(pop, params=None, pid=OPP, cs=cands, state=st):
        return evaluate_opponent_conditioned(
            state=state, belief=bel, range_model=rm, candidates=cs,
            population=pop, population_params=params or POPULATION_PARAMS,
            required_equity=0.0, equity_fn=equity_fn, player_id=pid,
            iterations=ITER, seed=42)

    evals = [run(p) for p in POPS5]
    _CACHE["out"] = {
        "evals": evals, "st": st, "bel": bel, "rm": rm, "cands": cands,
        "run": run, "params": POPULATION_PARAMS,
        "prior_only": evaluate_prior_only(state=st, candidates=cands),
        "cmp": compare_population_rankings(evaluations=evals),
    }
    return _CACHE["out"]


def test_t9_all_five_populations_produce_valid_output():
    d = _real()
    assert len(d["evals"]) == 5
    for e in d["evals"]:
        assert e["present"] is True, e.get("reason")
        assert e["evidence_grade"] == "MODEL_INTERNAL"


def test_t9_precheck_passes_on_real_scenario():
    d = _real()
    r = check_scenario(state=d["st"])
    assert r["valid"] is True, r["violations"]


def test_t9_postconditions_hold_for_all_five_populations():
    d = _real()
    for e in d["evals"]:
        r = check_postconditions(out=e, state=d["st"])
        assert r["valid"] is True, (e["population"], r["violations"])


def test_t9_causal_chain_complete_on_real_engine():
    d = _real()
    r = verify_causal_chain(evaluations=d["evals"])
    assert r["present"] is True
    assert r["chain_complete"] is True, r["chain_links"]
    assert r["first_broken_link"] is None


def test_t9_opponent_range_validity_all_populations():
    d = _real()
    for e in d["evals"]:
        r = check_opponent_range_validity(out=e, belief=d["bel"], player_id=OPP)
        assert r["valid"] is True, (e["population"], r["violations"])
        assert r["n_positive_posterior_classes"] == 169


def test_t9_ranking_or_suborder_varies_across_populations():
    d = _real()
    cmp = d["cmp"]
    assert cmp["present"] is True
    assert cmp["p_continue_changed"] is True
    assert cmp["n_candidates_with_difference"] >= 1
    assert (cmp["ranking_changed"] or cmp["betting_sub_order_changed"]) is True


def test_t9_counterfactual_on_real_engine():
    d = _real()
    run = d["run"]
    base = run("DECLARED_BALANCED")
    same = run("DECLARED_BALANCED")
    params = dict(d["params"])
    params["DECLARED_BALANCED"] = dict(d["params"]["DECLARED_MANIAC"])
    swap = run("DECLARED_BALANCED", params)
    r = counterfactual_population_swap(baseline_out=base, swapped_out=swap,
                                       identical_params_out=same)
    assert r["counterfactual_changes_result"] is True
    assert r["restore_is_field_identical"] is True
    assert r["tolerance_baseline"] != r["tolerance_swapped"]
    assert r["pass"] is True


def test_t9_label_alone_changes_nothing_on_real_engine():
    """要求 5：同一标签 + 不同参数 → 结果随参数；不同标签 + 同参数 → 结果相同。"""
    d = _real()
    run = d["run"]
    params = dict(d["params"])
    params["DECLARED_AGGRESSIVE"] = dict(d["params"]["DECLARED_BALANCED"])
    bal = run("DECLARED_BALANCED")
    agg_as_bal = run("DECLARED_AGGRESSIVE", params)
    assert stable_fingerprint(bal["per_candidate"]) == \
        stable_fingerprint(agg_as_bal["per_candidate"])
    assert bal["population_tolerance"]["tolerance_premium"] == \
        agg_as_bal["population_tolerance"]["tolerance_premium"] == 0.0


def test_t9_no_evidence_produces_no_population_difference():
    d = _real()
    po = d["prior_only"]
    assert po["adaptation_applied"] is False
    assert po["equity_calls_used"] == 0
    r = no_evidence_probe(per_population_outputs={p: po for p in POPS5})
    assert r["all_identical"] is True
    assert r["pass"] is True


def test_t9_deterministic_repeat_on_real_engine():
    d = _real()
    a = d["run"]("DECLARED_MANIAC")
    b = d["run"]("DECLARED_MANIAC")
    assert stable_fingerprint(a) == stable_fingerprint(b)


def test_t9_real_outputs_never_claim_external_validity():
    d = _real()
    for e in d["evals"]:
        assert e["evidence_grade"] == "MODEL_INTERNAL"
        assert e["executed"] is False and e["result"] is None
        assert e["wired_to_production"] is False
        assert all(v is False for v in e["claim_boundary"].values())


# ---------------------------------------------------------------------------
# T10. 不同 street / 短码下的合法性（要求 6）
# ---------------------------------------------------------------------------
def test_t10_precheck_holds_across_streets_and_stacks():
    from poker_decision_ai.cash_decision_environment import build_scenario
    grid = [
        dict(seed=SEED, street="flop", pot_bb=1.5, stack_bb=100.0),
        dict(seed=SEED, street="turn", pot_bb=8.0, stack_bb=100.0),
        dict(seed=SEED, street="river", pot_bb=4.0, stack_bb=20.0),
        dict(seed=SEED, street="flop", pot_bb=2.0, stack_bb=6.0),
        dict(seed=7, street="flop", pot_bb=3.0, stack_bb=25.0),
    ]
    for scn in grid:
        st = build_scenario(**scn)["state"]
        r = check_scenario(state=st)
        assert r["valid"] is True, (scn, r["violations"])
        assert r["board_size"] == VALID_BOARD_SIZE[st["street"]]
