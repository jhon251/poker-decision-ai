# -*- coding: utf-8 -*-
"""V3H-12 测试 — CASH 预注册分析基础（设计面）＋ 结果生成闸门。

覆盖：CASH-only fail closed / 非 dict / 缺 closure / min_rows 非法 /
**设计 ≠ 结果**（三入口 `result` 恒 None、`results_available` 恒 False、
`analysis_executed` 恒 False）/ 真实执行数据被**主动拒绝消费** /
闸门前置条件逐项判定且 `would_run` **恒 False** /
数据集 schema 精确字段集与逐字段校验（缺字段、类型错、重复 id、版本不符、
closed-enum 越界、executed=true）/ 缺失策略禁止插补与静默丢弃 /
endpoint 可测量性声明（E3 需真实执行数据）/ 假设**无方向无优劣** /
独立 blocker M3/B1/B3/B4/B5/solver 登记且非本层测量 /
`analysis_ready` 直接转述 V3H-11 且恒 False / solver 各量恒 None /
provenance 与 evidence_level=DESIGN_DECLARATION / 零统计量零公式
（AST：无 Mult/Div/Pow/FloorDiv/Mod）/ 定输入复现 / state 不变异 /
AST·import 隔离（零 V3G、零任何层）/ 生产 import 图零新增 /
frozen·sealed drift / 真实全栈端到端。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_preregistration import (  # noqa: E402
    ANALYSIS_UNIT, BLOCKERS, CASH_STRATEGY_PREREGISTRATION_VERSION,
    CLAIM_BOUNDARY, DATASET_SCHEMA_VERSION, DEFAULT_MIN_ROWS, ENDPOINTS,
    EVIDENCE_LEVELS, HYPOTHESES, MISSINGNESS_POLICY, REQUIRED_FIELDS,
    RESULT_PRECONDITIONS, build_preregistration, dataset_schema,
    run_analysis_preconditions, validate_candidate_records)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    assemble_decision)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)
from poker_decision_ai.cash_strategy_evidence_closure import (  # noqa: E402
    audit_evidence_closure)
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)
from poker_decision_ai.cash_strategy_multistreet_eval import (  # noqa: E402
    conditional_absolute_equity)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402
from poker_decision_ai.cash_strategy_shadow_evaluation import (  # noqa: E402
    CASH_STRATEGY_SHADOW_EVALUATION_VERSION, evaluate_shadow_decision,
    make_shadow_observation)

NEW_MODULE = "cash_strategy_preregistration"
MODULE_PATH = PKG / f"{NEW_MODULE}.py"

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

V3G_LAYERS = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface")
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60, "max_runout_children": 2}
ALL_FACTS = {k: True for k in ENVIRONMENT_FACT_KEYS}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 3, street: str = "flop", board=None, **over) -> dict:
    pos = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
           4: ("SB", "BB", "MP", "BTN"),
           5: ("SB", "BB", "MP", "CO", "BTN"),
           6: ("SB", "BB", "UTG", "MP", "CO", "BTN")}[n_players]
    hero_seat = n_players
    players, vi = [], 1
    for seat in range(1, n_players + 1):
        if seat == hero_seat:
            players.append({"player_id": "hero", "seat": seat,
                            "position": pos[seat - 1], "is_hero": True,
                            "status": "active", "hole_cards": ["As", "Kd"],
                            "stack": 100.0})
        else:
            players.append({"player_id": f"v{vi}", "seat": seat,
                            "position": pos[seat - 1], "status": "active",
                            "stack": 100.0, "hole_cards": []})
            vi += 1
    st = {
        "game_type": "cash", "street": street,
        "board": ["Qs", "7d", "2c"] if board is None else board,
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": n_players,
        "synthetic_test_only": True, "players": players, "action_history": [],
    }
    st.update(over)
    return st


def valid_row(i: int = 0, **over) -> dict:
    row = {
        "record_id": f"dp-{i:04d}",
        "shadow_version": CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
        "state_hash": f"hash-{i:04d}",
        "street": "flop",
        "divergence": "aligned",
        "recommended_action": "check",
        "recommended_amount_bb": 0.0,
        "observed_action": "check",
        "observed_amount_bb": 0.0,
        "available": True,
        "level": "MODEL_INTERNAL",
        "evidence_grade": "MODEL_INTERNAL",
        "snapshot_present": True,
        "snapshot_method": "synthetic_mc",
        "snapshot_samples": 20000,
        "unknown_tokens": [],
        "executed": False,
    }
    row.update(over)
    return row


def good_closure(**over) -> dict:
    """合成一个 V3H-11 输出（`closure_possible` 默认 False，如实）。"""
    c = {
        "present": True,
        "version": "v3h11-cash-strategy-evidence-closure-v1",
        "n_records": 5,
        "closure_possible": False,
        "counts": {"unknown_reasons": {}},
        "closure_test": {"records_short_by": 1195,
                         "established_blockers": ["B1", "B3", "B4", "B5",
                                                  "solver"]},
    }
    c.update(over)
    return c


def real_stack(state, observations):
    ap = build_action_policy(state, action_tree=build_action_tree(state))
    bl = build_dynamic_belief(state, range_model=build_range_model(state))
    rk = rank_actions(state, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(state, action_policy=ap, ranking=rk)
    tr = build_search_tree(state, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    dec = assemble_decision(state, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(state))
    recs = [evaluate_shadow_decision(
        state, decision=dec, action_policy=ap, observed=obs,
        equity_snapshot=conditional_absolute_equity) for obs in observations]
    clo = audit_evidence_closure(state, shadow_records=recs,
                                 environment=ALL_FACTS)
    return dec, recs, clo


# ── 1. CASH-only fail closed（三入口一致）───────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    for r in (build_preregistration(st, closure=good_closure()),
              validate_candidate_records(st, [valid_row()],
                                        expected_shadow_version="x"),
              run_analysis_preconditions(st, closure=good_closure())):
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["result"] is None
        assert r["results_available"] is False
        assert r["analysis_executed"] is False
        assert r["executed"] is False
        assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert build_preregistration(bad, closure=good_closure())["present"] is False
    assert run_analysis_preconditions(bad)["present"] is False
    assert validate_candidate_records(bad, [])["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True])
def test_missing_closure_fail_closed(bad):
    r = build_preregistration(cash_state(), closure=bad)
    assert r["present"] is False
    assert r["reason"] == "closure_required"
    assert r["result"] is None
    assert r["analysis_ready"] is False


def test_closure_present_false_fail_closed():
    r = build_preregistration(cash_state(),
                              closure={"present": False, "reason": "x"})
    assert r["present"] is False
    assert r["reason"] == "closure_required"


@pytest.mark.parametrize("bad", [0, -1, "1200", 1.5, True, None])
def test_invalid_min_rows_fail_closed(bad):
    assert build_preregistration(cash_state(), closure=good_closure(),
                                min_rows=bad)["reason"] == "min_rows_invalid"
    assert run_analysis_preconditions(cash_state(),
                                      min_rows=bad)["reason"] == "min_rows_invalid"


@pytest.mark.parametrize("bad", [None, "rows", 42, {"a": 1}])
def test_records_required_for_validation(bad):
    r = validate_candidate_records(cash_state(), bad,
                                   expected_shadow_version="x")
    assert r["present"] is False
    assert r["reason"] == "records_required"


@pytest.mark.parametrize("bad", [None, "", "   ", 7])
def test_expected_shadow_version_required(bad):
    r = validate_candidate_records(cash_state(), [valid_row()],
                                   expected_shadow_version=bad)
    assert r["present"] is False
    assert r["reason"] == "expected_shadow_version_required"


# ── 2. 设计 ≠ 结果（核心保证）──────────────────────────────────────
def test_design_only_never_returns_result():
    r = build_preregistration(cash_state(), closure=good_closure(),
                              solver=solver_status(cash_state()),
                              environment=ALL_FACTS)
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["analysis_executed"] is False
    assert r["executed"] is False
    assert r["design_only"] is True
    assert r["analysis_ready"] is False
    assert r["result_generation_refused"] is True
    assert "no result may be produced" in r["result_refusal_reason"]


def test_flat_key_scan_has_no_result_bearing_field():
    r = build_preregistration(cash_state(), closure=good_closure(),
                              environment=ALL_FACTS)
    keys: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r)
    for banned in ("effect_size", "p_value", "confidence_interval", "ci_low",
                   "ci_high", "win_rate", "roi", "pnl", "profit_estimate",
                   "alignment_rate", "mean_outcome", "t_statistic"):
        assert banned not in keys, banned


def test_all_numeric_result_slots_are_none_or_absent():
    r = build_preregistration(cash_state(), closure=good_closure(),
                              environment=ALL_FACTS)
    assert r["solver"]["equilibrium"] is None
    assert r["solver"]["gto"] is None
    assert r["solver"]["exploitability"] is None
    assert r["multiple_comparisons"]["correction_method"] is None
    assert r["uncertainty"]["measured"] is False


@pytest.mark.parametrize("data", [{"pnl": 1.0}, [1, 2, 3], "csv-ish", 42, True])
def test_real_execution_data_is_refused_not_consumed(data):
    r = run_analysis_preconditions(cash_state(), closure=good_closure(),
                                   solver=solver_status(cash_state()),
                                   environment=ALL_FACTS,
                                   real_execution_data=data)
    assert r["result_generation_refused"] is True
    assert "must not be consumed by a design-only layer" in \
        r["result_refusal_reason"]
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["would_run"] is False


def test_no_real_execution_data_gives_other_refusal_reason():
    r = run_analysis_preconditions(cash_state(), closure=good_closure())
    assert r["result_generation_refused"] is False
    assert "no real execution data" in r["result_refusal_reason"]


def test_preregistration_provenance_declares_no_data_consumed():
    r = build_preregistration(cash_state(), closure=good_closure())
    p = r["provenance"]
    assert p["real_execution_data_consumed"] is False
    assert p["solver_invoked"] is False
    assert p["future_cards_generated"] is False
    assert p["runout_sampling_performed"] is False
    assert p["frozen_multistreet_or_sample_runouts_reused"] is False
    assert p["evidence_level"] == "DESIGN_DECLARATION"


# ── 3. 结果生成闸门 ────────────────────────────────────────────────
def test_would_run_always_false_even_with_everything_optimistic():
    """把所有可提供项都设为"看似通过"，`would_run` 仍必须为 False。"""
    fake_ok_solver = {"solver_status": "AVAILABLE", "available": True,
                      "equilibrium": True, "discovery_state": "AVAILABLE"}
    optimistic_closure = good_closure(closure_possible=True,
                                      closure_test={"records_short_by": 0,
                                                    "established_blockers": []})
    r = run_analysis_preconditions(cash_state(), closure=optimistic_closure,
                                   solver=fake_ok_solver,
                                   environment=ALL_FACTS, min_rows=1)
    assert r["would_run"] is False
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["analysis_ready"] is False
    assert r["analysis_executed"] is False
    # 真实执行数据与数据集面四项永远不满足（本层不建立该入口）
    for never in ("real_execution_data_supplied_and_verified",
                  "dataset_has_zero_excluded_rows",
                  "dataset_rows_at_or_above_declared_minimum",
                  "no_unknown_tokens",
                  "no_forbidden_claim_flags"):
        assert never in r["unmet"], never


def test_preconditions_are_enumerated_and_partitioned():
    r = run_analysis_preconditions(cash_state(), closure=good_closure(),
                                   solver=solver_status(cash_state()),
                                   environment=ALL_FACTS)
    assert r["would_run_if"] == list(RESULT_PRECONDITIONS)
    assert len(RESULT_PRECONDITIONS) == 9
    assert sorted(r["met"] + r["unmet"]) == sorted(RESULT_PRECONDITIONS)
    assert r["n_met"] == len(r["met"])
    assert r["n_unmet"] == len(r["unmet"])
    assert r["n_met"] + r["n_unmet"] == 9
    assert r["unknowns" if False else "uncertainty"]["unknowns"] == r["unmet"]


def test_blockers_unresolved_keeps_readiness_false():
    r = run_analysis_preconditions(cash_state(), closure=good_closure())
    assert "independent_blockers_all_resolved" in r["unmet"]
    assert "closure_possible_is_true" in r["unmet"]
    assert "solver_not_blocked_and_equilibrium_available" in r["unmet"]


def test_solver_blocked_precondition_never_met_from_real_solver():
    s = solver_status(cash_state())
    r = run_analysis_preconditions(cash_state(), closure=good_closure(),
                                   solver=s)
    assert "solver_not_blocked_and_equilibrium_available" in r["unmet"]
    assert s["solver_status"] == "BLOCKED"


def test_environment_all_true_marks_one_precondition_met():
    r = run_analysis_preconditions(cash_state(), closure=good_closure(),
                                   environment=ALL_FACTS)
    assert "environment_facts_all_true" in r["met"]
    r2 = run_analysis_preconditions(cash_state(), closure=good_closure(),
                                    environment={"a": True, "b": False})
    assert "environment_facts_all_true" in r2["unmet"]


# ── 4. 数据集 schema ───────────────────────────────────────────────
def test_dataset_schema_shape_and_field_count():
    s = dataset_schema()
    assert s["schema_version"] == DATASET_SCHEMA_VERSION
    assert s["required_fields"] == list(REQUIRED_FIELDS)
    assert s["n_required_fields"] == len(REQUIRED_FIELDS) == 17
    assert set(s["field_kinds"]) == set(REQUIRED_FIELDS)
    assert s["statistics_declared"] == []
    assert s["result"] is None
    assert s["executed"] is False
    assert s["evidence_level"] == "DESIGN_DECLARATION"


def test_dataset_schema_declares_no_statistic_anywhere():
    s = dataset_schema()
    flat = json.dumps(s, ensure_ascii=False)
    for banned in ("p_value", "effect_size", "confidence_interval", "power",
                   "t_test", "bootstrap", "regression", "estimator",
                   "standard_error"):
        assert banned not in flat, banned


def test_missingness_policy_forbids_imputation_and_silent_drop():
    assert MISSINGNESS_POLICY["imputation"] == "forbidden"
    assert MISSINGNESS_POLICY["silent_drop"] == "forbidden"
    for k in ("unavailable_records", "unknown_tokens_present", "abstention",
              "disagreement", "model_internal_only"):
        assert k in MISSINGNESS_POLICY
    assert "excluded_reason" in MISSINGNESS_POLICY["unavailable_records"]
    assert MISSINGNESS_POLICY["imputation"] != "allowed"


def test_analysis_unit_does_not_claim_independence():
    assert ANALYSIS_UNIT["independence_claim"] is False
    assert "B1" in ANALYSIS_UNIT["independence_note"]
    assert ANALYSIS_UNIT["one_row_per"] == "decision point"


def test_valid_rows_pass_structural_validation():
    rows = [valid_row(i) for i in range(5)]
    r = validate_candidate_records(cash_state(), rows,
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["present"] is True
    assert r["n_records"] == 5
    assert r["counts"] == {"valid": 5, "invalid": 0}
    assert r["all_valid"] is True
    assert all(x["valid"] for x in r["rows"])
    assert r["result"] is None
    assert r["analysis_executed"] is False
    assert r["evidence_level"] == "DESIGN_DECLARATION"


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_each_required_field_is_enforced(field):
    row = valid_row()
    del row[field]
    r = validate_candidate_records(cash_state(), [row],
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["counts"]["invalid"] == 1
    assert r["field_missing_counts"].get(field) == 1
    assert f"missing_field:{field}" in r["rows"][0]["issues"]


@pytest.mark.parametrize("field,bad", [
    ("record_id", ""), ("record_id", "   "), ("record_id", None),
    ("shadow_version", "v0"), ("state_hash", None), ("street", None),
    ("divergence", "ALIGNED"), ("divergence", "nonsense"),
    ("divergence", None), ("divergence", 7),
    ("available", "yes"), ("available", 1),
    ("snapshot_present", None), ("snapshot_present", "true"),
    ("snapshot_samples", "20000"), ("snapshot_samples", 1.5),
    ("unknown_tokens", "none"), ("unknown_tokens", [1, 2]),
    ("unknown_tokens", None),
    ("recommended_amount_bb", "3"), ("observed_amount_bb", "1"),
    ("recommended_amount_bb", True),
    ("recommended_action", 7), ("observed_action", 7),
    ("executed", True), ("executed", None),
])
def test_bad_field_values_are_rejected(field, bad):
    r = validate_candidate_records(cash_state(), [valid_row(**{field: bad})],
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["counts"]["invalid"] == 1, (field, bad)
    assert r["all_valid"] is False


def test_duplicate_record_id_detected_not_collapsed():
    rows = [valid_row(), valid_row(0 + 0)]
    rows[1]["record_id"] = rows[0]["record_id"]
    r = validate_candidate_records(cash_state(), rows,
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["counts"]["invalid"] == 1
    assert "duplicate_record_id" in r["rows"][1]["issues"]
    assert r["n_records"] == 2                       # 不合并、不丢弃


def test_version_mismatch_flagged_per_row():
    rows = [valid_row(0), valid_row(1, shadow_version="v3h10-old")]
    r = validate_candidate_records(cash_state(), rows,
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["counts"]["invalid"] == 1
    assert "shadow_version_mismatch" in r["rows"][1]["issues"]
    assert r["rows"][1]["record_id"] == "dp-0001"


def test_non_dict_rows_marked_not_dropped():
    r = validate_candidate_records(cash_state(), [valid_row(), None, "x", 3],
                                   expected_shadow_version=(
                                       CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert r["n_records"] == 4
    assert r["counts"]["invalid"] == 3
    assert len(r["rows"]) == 4
    for x in r["rows"][1:]:
        assert x["issues"] == ["row_not_a_dict"]


def test_none_amounts_and_actions_are_allowed():
    r = validate_candidate_records(
        cash_state(),
        [valid_row(recommended_action=None, recommended_amount_bb=None,
                   observed_action=None, observed_amount_bb=None,
                   snapshot_method=None, snapshot_samples=None,
                   divergence="unavailable", available=False,
                   evidence_grade="NONE", level="UNKNOWN")],
        expected_shadow_version=CASH_STRATEGY_SHADOW_EVALUATION_VERSION)
    assert r["counts"]["valid"] == 1
    assert r["all_valid"] is True


def test_empty_record_list_is_not_all_valid_but_not_an_error():
    r = validate_candidate_records(cash_state(), [],
                                   expected_shadow_version="x")
    assert r["present"] is True
    assert r["n_records"] == 0
    assert r["all_valid"] is False
    assert r["counts"] == {"valid": 0, "invalid": 0}


def test_validation_does_not_mutate_input_rows():
    rows = [valid_row(0), valid_row(1, divergence="nonsense")]
    before = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    validate_candidate_records(cash_state(), rows, expected_shadow_version=(
        CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
    assert json.dumps(rows, sort_keys=True, ensure_ascii=False) == before


# ── 5. 假设 / 结局 / blocker 声明 ─────────────────────────────────
def test_endpoints_declared_upfront_with_measurability():
    ids = [e["id"] for e in ENDPOINTS]
    assert ids == ["E1", "E2", "E3"]
    by_id = {e["id"]: e for e in ENDPOINTS}
    assert by_id["E1"]["measurable_without_execution_data"] is True
    assert by_id["E2"]["measurable_without_execution_data"] is True
    assert by_id["E3"]["measurable_without_execution_data"] is False
    assert by_id["E3"]["kind"] == "outcome_requires_real_execution_data"
    assert by_id["E2"]["kind"] == "descriptive_count"


def test_hypotheses_claim_no_direction_and_no_superiority():
    assert len(HYPOTHESES) == 1
    h = HYPOTHESES[0]
    assert h["id"] == "H1"
    assert h["falsifiable"] is True
    assert h["direction_claimed"] is False
    assert h["superiority_claimed"] is False
    assert h["endpoints"] == ["E2"]


def test_hypothesis_text_contains_no_numeric_estimate():
    txt = json.dumps(HYPOTHESES, ensure_ascii=False)
    for banned in ("better", "worse", "improve", "outperform", "gain", "profit"):
        assert banned not in txt.lower(), banned


def test_blockers_registered_and_not_measured_here():
    assert set(BLOCKERS) == {"M3", "B1", "B3", "B4", "B5", "solver"}
    for k, v in BLOCKERS.items():
        assert v["status"] == "BLOCKED", k
        assert v["measured_by_this_layer"] is False, k
    r = build_preregistration(cash_state(), closure=good_closure())
    assert r["blockers_measured_by_this_layer"] is False
    assert set(r["blockers"]) == set(BLOCKERS)
    assert r["blockers_from_upstream"] == ["B1", "B3", "B4", "B5", "solver"]


def test_analysis_ready_is_transcribed_from_v3h11_not_recomputed():
    r_false = build_preregistration(cash_state(),
                                    closure=good_closure(closure_possible=False))
    assert r_false["analysis_ready"] is False
    assert "transcribed" in r_false["analysis_ready_source"]
    r_true = build_preregistration(cash_state(),
                                   closure=good_closure(closure_possible=True))
    assert r_true["analysis_ready"] is True
    # 即便转述为 True，也仍然**不产生结果**
    assert r_true["result"] is None
    assert r_true["results_available"] is False
    assert r_true["executed"] is False


def test_stopping_rule_never_computes_sample_size():
    r = build_preregistration(cash_state(), closure=good_closure(), min_rows=1200)
    sr = r["stopping_rule"]
    assert sr["min_rows"] == 1200
    assert sr["kind"] == "no_early_stopping_declared"
    assert "NOT measured" in sr["min_rows_source"]
    assert "不" in sr["note"]
    assert DEFAULT_MIN_ROWS == 1200


def test_multiple_comparisons_choice_deferred_not_post_hoc():
    r = build_preregistration(cash_state(), closure=good_closure())
    mc = r["multiple_comparisons"]
    assert mc["policy"] == "declare_all_endpoints_upfront"
    assert mc["n_endpoints"] == len(ENDPOINTS)
    assert mc["correction_method"] is None


def test_eligibility_counted_only():
    r = build_preregistration(
        cash_state(), closure=good_closure(),
        eligibility=[{"present": True, "eligible": False},
                     {"present": True, "eligible": True},
                     {"present": False}])
    ec = r["eligibility_counts"]
    assert ec["n_supplied"] == 2
    assert ec["n_not_eligible"] == 1
    assert "eligible" not in r
    assert "status" not in r


def test_environment_missing_listed_sorted():
    r = build_preregistration(cash_state(), closure=good_closure(),
                              environment={"z": False, "a": None, "ok": True})
    assert r["environment_missing"] == ["a", "z"]
    assert any("environment_missing" in u for u in r["unknown"])


def test_solver_view_never_invents_quantities():
    for s in (None, solver_status(cash_state()),
              {"solver_status": "BLOCKED"}):
        v = build_preregistration(cash_state(), closure=good_closure(),
                                  solver=s)["solver"]
        assert v["equilibrium"] is None
        assert v["gto"] is None
        assert v["exploitability"] is None
        assert v["blocked"] is True
    r = build_preregistration(cash_state(), closure=good_closure())
    assert "solver_status_not_supplied" in r["unknown"]


def test_claim_boundary_all_false_17_flags():
    assert len(CLAIM_BOUNDARY) == 17
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("is_experimental_result", "produces_results_without_execution_data",
              "performs_power_or_sample_size_calculation",
              "computes_effect_size_or_interval"):
        assert CLAIM_BOUNDARY[k] is False
    r = build_preregistration(cash_state(), closure=good_closure())
    assert all(v is False for v in r["claim_boundary"].values())


def test_evidence_levels_include_design_declaration_only():
    assert EVIDENCE_LEVELS == ("DESIGN_DECLARATION", "MODEL_INTERNAL",
                               "UNAVAILABLE")
    r = build_preregistration(cash_state(), closure=good_closure())
    assert r["evidence_level"] == "DESIGN_DECLARATION"
    assert r["evidence_grade"] == "NONE"
    assert r["evidence_levels"] == list(EVIDENCE_LEVELS)


# ── 6. 定输入复现 / 不变异 / 可序列化 ─────────────────────────────
def test_deterministic_same_input_same_output():
    st = cash_state()
    clo = good_closure()
    outs = [build_preregistration(st, closure=clo, environment=ALL_FACTS)
            for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [validate_candidate_records(st, [valid_row(i) for i in range(4)],
                                        expected_shadow_version=(
                                            CASH_STRATEGY_SHADOW_EVALUATION_VERSION))
             for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_state_not_mutated():
    st = cash_state()
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    build_preregistration(st, closure=good_closure(), environment=ALL_FACTS)
    run_analysis_preconditions(st, closure=good_closure(), environment=ALL_FACTS)
    validate_candidate_records(st, [valid_row()],
                               expected_shadow_version="x")
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before


def test_outputs_json_serialisable():
    st = cash_state()
    for r in (build_preregistration(st, closure=good_closure()),
              run_analysis_preconditions(st, closure=good_closure()),
              validate_candidate_records(st, [valid_row()],
                                         expected_shadow_version="x"),
              dataset_schema()):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 7. 零统计量零公式（源码级）───────────────────────────────────
def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned
    assert ops <= {"Sub", "Add", "BitOr"}, ops


def test_source_contains_no_statistical_implementation_tokens():
    src = MODULE_PATH.read_text(encoding="utf-8")
    lower = src.lower()
    for banned in ("scipy", "numpy", "statsmodels", "p_value", "t_test",
                   "chi2", "bootstrap(", "confidence_interval(", "regress(",
                   "effect_size(", "power_analysis"):
        assert banned not in lower, banned


# ── 8. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_PREREGISTRATION_VERSION.startswith("v3h12-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts_name():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


def test_no_execution_or_io_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "subprocess", "socket",
                   "requests", "urllib", "open(", "input(", "screenshot",
                   "sqlite", "csv."):
        assert banned not in src, banned


# ── 9. 生产 import 图零新增 ──────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                mods += [a.name.split(".")[-1] for a in n.names]
            for m in mods:
                if m in V3G_LAYERS or m in V3H_LAYERS:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_module_not_imported_in_production():
    assert NEW_MODULE not in _production_importers()


def test_all_v3h_layers_remain_unwired():
    found = _production_importers()
    for layer in V3H_LAYERS:
        assert layer not in found, layer
    # 集合**包含式**断言（本阶段冻结的全部 V3H 层都必须仍在，且仍零接线）；
    # 后续阶段新增层不会使本测试失效。
    required = {
        "cash_strategy_action_policy",
        "cash_strategy_action_ranking",
        "cash_strategy_belief_conditioned_eval",
        "cash_strategy_belief_eval",
        "cash_strategy_belief_update",
        "cash_strategy_decision_assembly",
        "cash_strategy_dynamic_belief",
        "cash_strategy_evidence_closure",
        "cash_strategy_experiment_eligibility",
        "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval",
        "cash_strategy_policy_evaluator",
        "cash_strategy_preregistration",
        "cash_strategy_search_eval",
        "cash_strategy_search_tree",
        "cash_strategy_shadow_evaluation",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 10. frozen / sealed drift · Solver ───────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_manifest_drift_unchanged():
    man = json.loads((ROOT / "docs" / "V3G_11_FREEZE_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_solver_still_blocked_not_flipped():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["executed"] is False
    assert s["discovery_state"] == "UNAVAILABLE"


# ── 11. 真实全栈端到端 ───────────────────────────────────────────
def test_end_to_end_real_stack_river_only():
    """真实全栈 + river 终局（V3G-7）：闭合判定为假、结果恒 None。"""
    st = cash_state(street="river", board=["Qs", "7d", "2c", "8h", "3s"])
    dec, recs, clo = real_stack(st, [make_shadow_observation("check", 0.0)])
    assert clo["closure_possible"] is False
    r = build_preregistration(st, closure=clo, solver=solver_status(st),
                              environment=ALL_FACTS)
    assert r["present"] is True
    assert r["analysis_ready"] is False
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["analysis_executed"] is False
    assert r["blockers_measured_by_this_layer"] is False
    assert r["solver"]["equilibrium"] is None


def test_end_to_end_real_stack_rows_validate_against_schema():
    """用真实 V3H-10 输出构造 schema 行并校验（字段面与上游一致）。"""
    st = cash_state()
    dec, recs, clo = real_stack(st, [make_shadow_observation("check", 0.0),
                                     make_shadow_observation("fold", None)])
    rows = []
    for i, s in enumerate(recs):
        rec = s.get("record") or {}
        rows.append(valid_row(
            i,
            street=s.get("street") or "flop",
            divergence=s.get("divergence"),
            recommended_action=(rec.get("recommended") or {}).get("action"),
            recommended_amount_bb=(rec.get("recommended") or {}).get("amount_bb"),
            observed_action=(rec.get("observed") or {}).get("action"),
            observed_amount_bb=(rec.get("observed") or {}).get("amount_bb"),
            available=s.get("available") is True,
            level=str(s.get("level")),
            evidence_grade=str(s.get("evidence_grade")),
            snapshot_present=(rec.get("shadow_snapshot") or {}).get("present")
            is True,
            snapshot_method=(rec.get("shadow_snapshot") or {}).get("method"),
            snapshot_samples=(rec.get("shadow_snapshot") or {}).get("samples"),
            unknown_tokens=list(s.get("unknown") or []),
        ))
    v = validate_candidate_records(
        st, rows,
        expected_shadow_version=CASH_STRATEGY_SHADOW_EVALUATION_VERSION)
    assert v["counts"]["valid"] == 2, v["rows"]
    assert v["all_valid"] is True
    assert v["result"] is None
    assert [x["divergence"] for x in rows] == ["aligned", "different"]
    r = build_preregistration(st, closure=clo, solver=solver_status(st),
                              environment=ALL_FACTS)
    assert r["stopping_rule"]["rows_observed"] == 2
    assert r["stopping_rule"]["rows_short_by"] == 1198
    assert r["analysis_ready"] is False
    assert r["result"] is None


def test_end_to_end_design_surface_never_yields_a_number():
    """设计面全字段扫描：不得出现任何"结果数值"。"""
    st = cash_state()
    dec, recs, clo = real_stack(st, [make_shadow_observation("check", 0.0)])
    r = build_preregistration(st, closure=clo, solver=solver_status(st),
                              environment=ALL_FACTS)

    nums: list[tuple[str, float]] = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, float):
            nums.append((path, o))

    walk(r)
    allowed = {".stopping_rule.min_rows", ".schema.n_required_fields",
               ".preconditions.n_met", ".preconditions.n_unmet",
               ".eligibility_counts.n_supplied",
               ".eligibility_counts.n_not_eligible",
               ".multiple_comparisons.n_endpoints"}
    for path, _ in nums:
        assert path in allowed or path.endswith("_rows_short_by") \
            or path.endswith("rows_observed"), path
