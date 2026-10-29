# -*- coding: utf-8 -*-
"""V3H-13 测试 — CASH 真实执行记录接入闸门（执行前接口，**不执行**）。

覆盖：CASH-only fail closed / 非 dict / 缺 preregistration / min_records 非法 /
**不执行**（三入口 `executed`/`result` 恒 False·None）/
骨架模板**零预填**且不伪造环境 / 内容指纹确定性、可复现、
**明确声明非真实性证明** / 篡改检出（digest 不符、缺 digest）/
合成环境与合成牌源**被拒绝** / `evidence_level` 与 `result_provenance` 强制 /
`realised_outcome` 缺失或"模型内造结果"被拒绝 / UNKNOWN 非空 → 拒绝而**不转空** /
去重（同一执行记录不重复计数）/ 系统级前置条件逐项判定 /
**Solver BLOCKED ⇒ `intake_permitted` 恒 False 且 `admitted` 恒 0** /
独立 blocker 未解 → 拒绝 / 环境事实不全 → 拒绝 / 信任锚须带外提供且非本层判定 /
接入汇总只计数、零比率 / 真实全栈端到端 / 定输入复现 / 输入不变异 /
AST·import 隔离（零 V3G、零任何层、无 IO、无 random）/ 生产 import 图零新增 /
frozen·sealed drift。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_execution_intake import (  # noqa: E402
    ACCEPTED_EVIDENCE_LEVELS, BATCH_REFUSAL_REASONS, CLAIM_BOUNDARY,
    CASH_STRATEGY_EXECUTION_INTAKE_VERSION, DEFAULT_MIN_RECORDS,
    EXECUTION_RECORD_FIELDS, INTAKE_PRECONDITIONS, NON_EXECUTION_EVIDENCE_LEVELS,
    RECORD_REFUSAL_REASONS, SYNTHETIC_MARKERS, canonical_payload,
    compute_integrity_digest, execution_record_template,
    intake_execution_records, summarize_intake, validate_execution_record)

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
from poker_decision_ai.cash_strategy_preregistration import (  # noqa: E402
    build_preregistration)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402
from poker_decision_ai.cash_strategy_shadow_evaluation import (  # noqa: E402
    evaluate_shadow_decision, make_shadow_observation)

NEW_MODULE = "cash_strategy_execution_intake"
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


def exec_rec(i: int = 0, **over) -> dict:
    """一条**结构完整**的真实执行记录（**测试合成**；未做任何真实执行）。"""
    rec = {
        "execution_id": f"exec-{i:05d}",
        "shadow_record_id": f"dp-{i:05d}",
        "state_hash": f"state-{i:05d}",
        "execution_env": "live-cash-table-A",
        "executor_kind": "manual_operator",
        "observed_action": "call",
        "observed_amount_bb": 2.0,
        "observed_at_seq": 1000 + i,
        "card_source": "live_deal",
        "evidence_level": "REAL_EXECUTION_DATA",
        "result_provenance": "real_execution",
        "realised_outcome": {"net_bb": -2.0, "currency": "bb"},
        "unknown_tokens": [],
        "integrity_digest": None,
    }
    rec.update(over)
    if rec.get("integrity_digest") is None:
        rec["integrity_digest"] = compute_integrity_digest(rec)["digest"]
    return rec


def good_prereg(**over) -> dict:
    p = {
        "present": True,
        "version": "v3h12-cash-strategy-preregistration-v1",
        "blockers_from_upstream": ["B1", "B3", "B4", "B5", "solver"],
        "blockers": {"M3": {"status": "BLOCKED"},
                     "B1": {"status": "BLOCKED"},
                     "B3": {"status": "BLOCKED"},
                     "B4": {"status": "BLOCKED"},
                     "B5": {"status": "BLOCKED"},
                     "solver": {"status": "BLOCKED"}},
        "analysis_ready": False,
        "result": None,
    }
    p.update(over)
    return p


OPTIMISTIC_SOLVER = {"solver_status": "AVAILABLE", "available": True,
                     "equilibrium": True, "discovery_state": "AVAILABLE"}


def real_closure(state, observations):
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
    return audit_evidence_closure(state, shadow_records=recs,
                                  environment=ALL_FACTS)


# ── 1. CASH-only fail closed ────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    results = [validate_execution_record(st, exec_rec()),
               intake_execution_records(st, records=[exec_rec()],
                                        preregistration=good_prereg()),
               summarize_intake(st, intake={"present": True})]
    for r in results:
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["executed"] is False
        assert r["intake_permitted"] is False
        assert r["evidence_level"] == "UNAVAILABLE"
        assert r["evidence_grade"] == "NONE"
    # 骨架接口为纯模板，非 CASH 判定不适用；仍不得含任何预填值
    t = execution_record_template()
    assert t["prefilled_values"] == []
    assert t["fabricated_environment"] is False
    assert t["executed"] is False
    # 指纹函数**不需要** state（只对内容做确定性哈希），故不参与模式判定
    d = compute_integrity_digest(exec_rec())
    assert d["present"] is True and d["is_authenticity_proof"] is False


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert validate_execution_record(bad, exec_rec())["present"] is False
    assert intake_execution_records(bad, records=[])["present"] is False
    assert summarize_intake(bad, intake={})["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True])
def test_missing_preregistration_fail_closed(bad):
    r = intake_execution_records(cash_state(), records=[exec_rec()],
                                 preregistration=bad)
    assert r["present"] is False
    assert r["reason"] == "preregistration_required"
    assert r["intake_permitted"] is False
    assert r["n_admitted"] == 0


def test_preregistration_present_false_fail_closed():
    r = intake_execution_records(cash_state(), records=[exec_rec()],
                                 preregistration={"present": False})
    assert r["present"] is False
    assert r["reason"] == "preregistration_required"


@pytest.mark.parametrize("bad", [0, -1, "x", 1.5, True, None])
def test_invalid_min_records_fail_closed(bad):
    r = intake_execution_records(cash_state(), records=[exec_rec()],
                                 preregistration=good_prereg(),
                                 min_records=bad)
    assert r["present"] is False
    assert r["reason"] == "min_records_invalid"


@pytest.mark.parametrize("bad", [None, 42, "r", True, {}])
def test_record_required_for_validation(bad):
    assert validate_execution_record(cash_state(), bad)["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42])
def test_intake_required_for_summary(bad):
    assert summarize_intake(cash_state(), intake=bad)["present"] is False


# ── 2. 不执行（核心保证）──────────────────────────────────────────
def test_intake_never_executes_or_produces_results():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["analysis_executed"] is False
    assert r["executed"] is False
    prov = r["provenance"]
    assert prov["execution_performed_by_this_layer"] is False
    assert prov["fabricated_execution_environment"] is False
    assert prov["wrote_any_file"] is False
    assert prov["opened_any_client"] is False
    assert prov["real_money_moved"] is False
    assert r["solver"]["solver_invoked"] is False


def test_template_prefills_nothing_and_fabricates_no_environment():
    t = execution_record_template()
    assert t["prefilled_values"] == []
    assert t["fabricated_environment"] is False
    assert t["executed"] is False
    assert t["n_fields"] == len(EXECUTION_RECORD_FIELDS) == 14
    for f in EXECUTION_RECORD_FIELDS:
        if f == "unknown_tokens":
            assert t["template"][f] == ["UNKNOWN"]
        else:
            assert t["template"][f] is None, f


def test_template_does_not_contain_a_valid_record():
    """骨架本身必须**不可**被接纳（防止把模板当成真实数据）。"""
    t = execution_record_template()["template"]
    v = validate_execution_record(cash_state(), t)
    assert v["valid"] is False
    assert v["n_issues"] > 0


def test_claim_boundary_all_false_17_flags():
    assert len(CLAIM_BOUNDARY) == 17
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("is_authenticity_proof", "is_certification",
              "verifies_source_trust", "fabricates_execution_environment",
              "produces_results_without_execution_data"
              if "produces_results_without_execution_data" in CLAIM_BOUNDARY
              else "is_experimental_result"):
        assert CLAIM_BOUNDARY[k] is False


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = intake_execution_records(cash_state(), records=[exec_rec()],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS, min_records=1)
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
    for banned in ("p_value", "effect_size", "confidence_interval", "win_rate",
                   "roi", "pnl", "profit_estimate", "alignment_rate",
                   "success_rate", "intake_rate"):
        assert banned not in keys, banned


# ── 3. 内容指纹 ────────────────────────────────────────────────────
def test_digest_is_deterministic_and_key_order_independent():
    a = exec_rec(0)
    b = {k: a[k] for k in reversed(list(a))}
    assert compute_integrity_digest(a)["digest"] == \
        compute_integrity_digest(b)["digest"]
    outs = [compute_integrity_digest(a)["digest"] for _ in range(5)]
    assert len(set(outs)) == 1


def test_digest_changes_when_content_changes():
    a = exec_rec(0)
    b = exec_rec(0, observed_amount_bb=2.5)
    assert compute_integrity_digest(a)["digest"] != \
        compute_integrity_digest(b)["digest"]


def test_digest_excludes_its_own_field():
    a = exec_rec(0)
    text, issues = canonical_payload(a)
    assert "integrity_digest" not in text
    assert issues == []


def test_digest_is_sha256_of_canonical_payload():
    a = exec_rec(0)
    text, _ = canonical_payload(a)
    assert compute_integrity_digest(a)["digest"] == \
        hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert compute_integrity_digest(a)["algorithm"] == "sha256"


def test_digest_explicitly_not_an_authenticity_proof():
    d = compute_integrity_digest(exec_rec(0))
    assert d["is_authenticity_proof"] is False
    assert d["is_certification"] is False
    assert d["verifies_source_trust"] is False
    assert d["requires_out_of_band_trust_anchor"] is True
    assert "不" in d["note"]
    assert d["scope"] == "canonical_record_content_excluding_integrity_digest"


def test_digest_refuses_non_dict_and_non_canonicalisable():
    assert compute_integrity_digest(None)["present"] is False
    assert compute_integrity_digest("x")["present"] is False
    r = compute_integrity_digest(exec_rec(0, realised_outcome={"x": object()}))
    assert r["present"] is False
    assert "payload_not_canonicalisable" in r["unknown"]


def test_digest_is_not_random_across_processes():
    """同一内容在**独立进程**中必须得到同一指纹（无随机、无时间戳）。"""
    import subprocess
    code = ("import json,sys;sys.path.insert(0,'.');"
            "from poker_decision_ai.cash_strategy_execution_intake import "
            "compute_integrity_digest;"
            "print(compute_integrity_digest("
            "json.loads(sys.argv[1]))['digest'])")
    payload = json.dumps(exec_rec(0), sort_keys=True, ensure_ascii=False)
    outs = set()
    for _ in range(2):
        p = subprocess.run([r".venv\Scripts\python.exe", "-c", code, payload],
                           capture_output=True, text=True)
        outs.add(p.stdout.strip())
    assert len(outs) == 1
    assert outs.pop() == compute_integrity_digest(exec_rec(0))["digest"]


# ── 4. 记录级校验：篡改与合成检出 ───────────────────────────────────
def test_complete_record_validates_and_matches_digest():
    v = validate_execution_record(cash_state(), exec_rec(0))
    assert v["valid"] is True
    assert v["n_issues"] == 0
    assert v["digest_recomputed_matches"] is True
    assert v["record_fingerprint"] is not None
    assert v["is_authenticity_proof"] is False
    assert v["result"] is None
    assert v["executed"] is False


def test_tampered_amount_is_detected():
    rec = exec_rec(0)
    rec["observed_amount_bb"] = 999.0            # 内容被改，指纹未更新
    v = validate_execution_record(cash_state(), rec)
    assert v["valid"] is False
    assert "integrity_digest_mismatch" in v["issues"]
    assert v["digest_recomputed_matches"] is False


def test_missing_digest_is_detected():
    rec = exec_rec(0)
    rec["integrity_digest"] = None
    v = validate_execution_record(cash_state(), rec)
    assert "integrity_digest_missing" in v["issues"]
    assert v["valid"] is False


def test_blank_digest_is_detected():
    v = validate_execution_record(cash_state(), exec_rec(0, integrity_digest="  "))
    assert "integrity_digest_missing" in v["issues"]


@pytest.mark.parametrize("env", ["synthetic", "test_only", "dry_run",
                                 "paper-trading", "mock_env", "FAKE",
                                 "simulated", "stub", "dummy", "fixture"])
def test_synthetic_execution_env_is_refused(env):
    v = validate_execution_record(cash_state(), exec_rec(0, execution_env=env))
    assert "execution_env_synthetic" in v["issues"]
    assert v["valid"] is False


@pytest.mark.parametrize("cs", ["synthetic_deck", "simulated_shuffle",
                                "mock_cards", "test_only"])
def test_synthetic_card_source_is_refused(cs):
    v = validate_execution_record(cash_state(), exec_rec(0, card_source=cs))
    assert "card_source_synthetic" in v["issues"]


@pytest.mark.parametrize("level", list(NON_EXECUTION_EVIDENCE_LEVELS))
def test_non_execution_evidence_levels_are_refused(level):
    v = validate_execution_record(cash_state(), exec_rec(0,
                                                         evidence_level=level))
    assert "evidence_level_not_real_execution_data" in v["issues"]
    assert v["valid"] is False


def test_model_internal_is_never_accepted_as_external_strength():
    """MODEL_INTERNAL 数据**不得**变成外部强度证据。"""
    v = validate_execution_record(cash_state(), exec_rec(0,
                                                         evidence_level="MODEL_INTERNAL"))
    assert v["valid"] is False
    assert v["evidence_level"] == "UNAVAILABLE"
    assert ACCEPTED_EVIDENCE_LEVELS == ("REAL_EXECUTION_DATA",)
    assert "MODEL_INTERNAL" in NON_EXECUTION_EVIDENCE_LEVELS


@pytest.mark.parametrize("src", ["model_internal", "MODEL_INTERNAL", "derived",
                                 "estimated", "inferred", "simulated"])
def test_non_real_result_provenance_is_refused(src):
    v = validate_execution_record(cash_state(), exec_rec(0,
                                                         result_provenance=src))
    assert "result_provenance_not_real_execution" in v["issues"]


def test_missing_realised_outcome_is_refused_not_inferred():
    v = validate_execution_record(cash_state(), exec_rec(0,
                                                         realised_outcome=None))
    assert "realised_outcome_missing" in v["issues"]
    assert v["valid"] is False


def test_model_generated_outcome_is_refused():
    v = validate_execution_record(
        cash_state(),
        exec_rec(0, realised_outcome={"source": "model_internal",
                                      "net_bb": 5.0}))
    assert "realised_outcome_invents_result" in v["issues"]


@pytest.mark.parametrize("tok", [["decision_abstained"],
                                 ["UNKNOWN"],
                                 ["anything"]])
def test_unknown_tokens_block_admission_and_are_not_emptied(tok):
    v = validate_execution_record(cash_state(), exec_rec(0,
                                                         unknown_tokens=tok))
    assert "unknown_tokens_present" in v["issues"]
    assert v["valid"] is False


def test_unknown_tokens_malformed_detected():
    for bad in ("none", None, [1, 2], {"a": 1}):
        v = validate_execution_record(cash_state(),
                                      exec_rec(0, unknown_tokens=bad))
        assert "unknown_tokens_malformed" in v["issues"], bad


@pytest.mark.parametrize("field", EXECUTION_RECORD_FIELDS)
def test_each_required_field_is_enforced(field):
    rec = exec_rec(0)
    del rec[field]
    v = validate_execution_record(cash_state(), rec)
    assert f"missing_required_field:{field}" in v["issues"]
    assert v["valid"] is False


@pytest.mark.parametrize("field,bad", [
    ("execution_id", ""), ("execution_id", "   "), ("execution_id", None),
    ("shadow_record_id", None), ("state_hash", None),
    ("executor_kind", None), ("observed_action", None),
    ("execution_env", None), ("card_source", None),
    ("observed_amount_bb", "2.0"), ("observed_amount_bb", True),
    ("observed_at_seq", "1000"), ("observed_at_seq", 1.5),
    ("observed_at_seq", None), ("observed_at_seq", True),
])
def test_bad_field_values_are_rejected(field, bad):
    v = validate_execution_record(cash_state(), exec_rec(0, **{field: bad}))
    assert v["valid"] is False, (field, bad)
    assert v["n_issues"] >= 1


def test_non_dict_record_reported_not_crashed():
    r = validate_execution_record(cash_state(), "not-a-record")
    assert r["present"] is False
    assert r["reason"] == "record_required"


def test_refusal_reason_enums_are_closed_and_documented():
    assert len(RECORD_REFUSAL_REASONS) == 21
    for x in ("integrity_digest_mismatch", "execution_env_synthetic",
              "card_source_synthetic", "unknown_tokens_present",
              "evidence_level_not_real_execution_data"):
        assert x in RECORD_REFUSAL_REASONS
    assert len(BATCH_REFUSAL_REASONS) == 8
    for x in ("solver_blocked_and_no_equilibrium_reference",
              "independent_blockers_unresolved", "trust_anchor_not_verified"):
        assert x in BATCH_REFUSAL_REASONS


# ── 5. 接入闸门：Solver BLOCKED ⇒ 恒不接纳 ─────────────────────────
def test_solver_blocked_means_intake_never_permitted_even_if_all_else_ok():
    r = intake_execution_records(cash_state(), records=[exec_rec(i)
                                                       for i in range(3)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[],
                                     blockers={}),
                                 solver=solver_status(cash_state()),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["intake_permitted"] is False
    assert r["admitted"] == 0
    assert r["n_admitted"] == 0
    assert r["admitted_records"] == []
    assert "solver_blocked_and_no_equilibrium_reference" in \
        r["batch_refusal_reasons"]
    assert "solver_available_with_equilibrium" in r["preconditions"]["unmet"]


def test_records_are_candidates_but_not_admitted_while_blocked():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["n_candidates"] == 1
    assert r["admitted"] == 0
    assert len(r["candidate_records"]) == 1
    assert r["candidate_records"][0]["execution_id"] == "exec-00000"


def test_intake_permitted_true_only_with_everything_satisfied():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["preconditions"]["n_unmet"] == 0, r["preconditions"]
    assert r["intake_permitted"] is True
    assert r["admitted"] == 1
    assert r["n_admitted"] == 1
    # 即使 permitted，也**不执行、不产生结果**
    assert r["executed"] is False
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["evidence_level"] == "REAL_EXECUTION_DATA"
    assert r["evidence_grade"] == "NONE"


def test_unresolved_blockers_block_admission():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["intake_permitted"] is False
    assert "independent_blockers_unresolved" in r["batch_refusal_reasons"]
    assert r["blockers_from_preregistration"] == ["B1", "B3", "B4", "B5",
                                                  "solver"]
    assert r["blockers_measured_by_this_layer"] is False


def test_environment_incomplete_blocks_admission():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment={"a": True, "b": False},
                                 trust_anchor_verified=True, min_records=1)
    assert r["intake_permitted"] is False
    assert "environment_facts_incomplete" in r["batch_refusal_reasons"]
    r2 = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                  preregistration=good_prereg(
                                      blockers_from_upstream=[], blockers={}),
                                  solver=OPTIMISTIC_SOLVER,
                                  environment=ALL_FACTS,
                                  trust_anchor_verified=True, min_records=1)
    assert "environment_facts_all_true" in r2["preconditions"]["met"]


def test_trust_anchor_must_be_supplied_out_of_band():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=False, min_records=1)
    assert r["intake_permitted"] is False
    assert "trust_anchor_not_verified" in r["batch_refusal_reasons"]
    ta = r["trust_anchor"]
    assert ta["verified"] is False
    assert ta["verified_by"] is None
    assert ta["verified_by_this_layer"] is False
    assert "带外" in ta["note"]


def test_trust_anchor_true_is_attributed_to_caller_not_this_layer():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["trust_anchor"]["verified"] is True
    assert r["trust_anchor"]["verified_by"] == "caller"
    assert r["trust_anchor"]["verified_by_this_layer"] is False


def test_minimum_rows_enforced():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1200)
    assert "minimum_rows_not_reached" in r["batch_refusal_reasons"]
    assert "records_at_or_above_declared_minimum" in \
        r["preconditions"]["unmet"]
    assert DEFAULT_MIN_RECORDS == 1200


def test_records_not_supplied_reported():
    r = intake_execution_records(cash_state(), records=None,
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True)
    assert r["present"] is True
    assert "records_not_supplied" in r["batch_refusal_reasons"]
    assert r["n_records_supplied"] == 0
    assert r["intake_permitted"] is False


def test_duplicate_records_are_detected_and_not_double_counted():
    a = exec_rec(0)
    b = exec_rec(0)                      # 完全相同的执行记录
    r = intake_execution_records(cash_state(), records=[a, b],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["n_duplicate"] == 1
    assert "zero_duplicate_records" in r["batch_refusal_reasons"]
    assert r["intake_permitted"] is False
    assert r["n_candidates"] == 1
    assert len(r["candidate_records"]) == 1


def test_distinct_records_are_not_treated_as_duplicates():
    r = intake_execution_records(cash_state(), records=[exec_rec(0), exec_rec(1)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["n_duplicate"] == 0
    assert r["intake_permitted"] is True
    assert r["admitted"] == 2


def test_refused_records_are_reported_with_reasons_and_empty_admission():
    bad = exec_rec(0, evidence_level="MODEL_INTERNAL")
    good = exec_rec(1)
    r = intake_execution_records(cash_state(), records=[bad, good],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["n_refused"] == 1
    assert "evidence_level_not_real_execution_data" in \
        r["record_refusal_reasons"]
    assert "zero_refused_records" in r["batch_refusal_reasons"]
    assert r["intake_permitted"] is False
    assert r["admitted"] == 0
    assert any(x["index"] == 1 and x["admitted"] for x in r["row_checks"])


def test_all_rows_are_reported_never_dropped():
    recs = [exec_rec(0), exec_rec(1), None, "x", exec_rec(2, execution_env="mock")]
    r = intake_execution_records(cash_state(), records=recs,
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["n_records_supplied"] == 5
    assert len(r["row_checks"]) == 5
    assert any(x["issues"] == ["record_not_a_dict"] for x in r["row_checks"])


def test_preconditions_partition_is_complete():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    p = r["preconditions"]
    assert p["all"] == list(INTAKE_PRECONDITIONS)
    assert p["would_admit_if"] == list(INTAKE_PRECONDITIONS)
    assert len(INTAKE_PRECONDITIONS) == 8
    assert sorted(p["met"] + p["unmet"]) == sorted(INTAKE_PRECONDITIONS)
    assert p["n_met"] == len(p["met"])
    assert p["n_unmet"] == len(p["unmet"])


def test_solver_view_never_invents_quantities():
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(),
                                 environment=ALL_FACTS, min_records=1)
    s = r["solver"]
    assert s["equilibrium"] is None
    assert s["gto"] is None
    assert s["exploitability"] is None
    assert s["blocked"] is True
    assert s["solver_invoked"] is False
    assert r["uncertainty"]["measured"] is False


def test_caller_facts_do_not_flip_the_frozen_solver():
    """即使调用方伪称 solver 可用，冻结的 solver_status 仍然 BLOCKED。"""
    frozen = solver_status(cash_state())
    r = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                 preregistration=good_prereg(
                                     blockers_from_upstream=[], blockers={}),
                                 solver=OPTIMISTIC_SOLVER,
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["intake_permitted"] is True          # 调用方言称口径
    assert frozen["solver_status"] == "BLOCKED"   # 冻结事实未变
    assert frozen["equilibrium"] is False
    # 但**未执行任何动作、未产生任何结果**
    assert r["executed"] is False
    assert r["result"] is None


def test_preregistration_blockers_read_from_real_v3h12_output():
    st = cash_state()
    clo = real_closure(st, [make_shadow_observation("check", 0.0)])
    pre = build_preregistration(st, closure=clo, solver=solver_status(st),
                                environment=ALL_FACTS)
    r = intake_execution_records(st, records=[exec_rec(0)],
                                 preregistration=pre,
                                 solver=solver_status(st),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["present"] is True
    assert r["intake_permitted"] is False
    assert sorted(r["blockers_from_preregistration"]) == ["B1", "B3", "B4",
                                                          "B5", "solver"]
    assert r["n_admitted"] == 0
    assert r["executed"] is False


# ── 6. 汇总：只计数、零比率 ────────────────────────────────────────
def test_summarize_counts_only_with_zero_rates():
    recs = [exec_rec(0), exec_rec(0), exec_rec(1, evidence_level="MODEL_INTERNAL"),
            None]
    intake = intake_execution_records(cash_state(), records=recs,
                                      preregistration=good_prereg(),
                                      environment=ALL_FACTS,
                                      trust_anchor_verified=True, min_records=1)
    s = summarize_intake(cash_state(), intake=intake)
    assert s["counts"]["n_rows"] == 4
    assert s["counts"]["n_admitted"] == 0
    assert s["counts"]["n_refused"] >= 2
    assert s["counts"]["n_duplicate"] == 1
    assert s["rate"] is None
    assert s["pct"] is None
    assert s["rate_computed"] is False
    assert s["counts_only"] is True
    assert s["rates_forbidden"] is True
    assert s["result"] is None
    assert s["executed"] is False


def test_summarize_refusal_reasons_are_sorted_tokens():
    intake = {"present": True, "row_checks": [
        {"admitted": False, "duplicate": False,
         "issues": ["missing_required_field:a", "execution_env_synthetic"]},
        {"admitted": False, "duplicate": True,
         "issues": ["missing_required_field:b"]}],
        "batch_refusal_reasons": ["z", "a"]}
    s = summarize_intake(cash_state(), intake=intake)
    assert list(s["counts"]["refusal_reasons"]) == \
        ["execution_env_synthetic", "missing_required_field"]
    assert s["counts"]["refusal_reasons"]["missing_required_field"] == 2
    assert s["counts"]["batch_refusal_reasons"] == ["a", "z"]


def test_summarize_handles_malformed_rows_without_crashing():
    s = summarize_intake(cash_state(),
                         intake={"present": True, "row_checks": [None, 7, "x"]})
    assert s["counts"]["n_rows"] == 3
    assert s["counts"]["n_admitted"] == 0


def test_summarize_output_is_json_serialisable():
    intake = intake_execution_records(cash_state(), records=[exec_rec(0)],
                                      preregistration=good_prereg(),
                                      environment=ALL_FACTS, min_records=1)
    s = summarize_intake(cash_state(), intake=intake)
    assert json.loads(json.dumps(s, ensure_ascii=False)) == s


# ── 7. 定输入复现 / 不变异 ─────────────────────────────────────────
def test_deterministic_same_input_same_output():
    st = cash_state()
    recs = [exec_rec(0), exec_rec(1), exec_rec(0)]
    outs = [intake_execution_records(st, records=recs,
                                     preregistration=good_prereg(),
                                     environment=ALL_FACTS,
                                     trust_anchor_verified=True, min_records=1)
            for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [validate_execution_record(st, exec_rec(0)) for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_state_and_records_not_mutated():
    st = cash_state()
    recs = [exec_rec(0), exec_rec(1, execution_env="mock")]
    st_before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    recs_before = json.dumps(recs, sort_keys=True, ensure_ascii=False)
    intake_execution_records(st, records=recs, preregistration=good_prereg(),
                             environment=ALL_FACTS, trust_anchor_verified=True,
                             min_records=1)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == st_before
    assert json.dumps(recs, sort_keys=True, ensure_ascii=False) == recs_before


def test_outputs_json_serialisable():
    st = cash_state()
    for r in (intake_execution_records(st, records=[exec_rec(0)],
                                       preregistration=good_prereg(),
                                       environment=ALL_FACTS, min_records=1),
              validate_execution_record(st, exec_rec(0)),
              compute_integrity_digest(exec_rec(0)),
              execution_record_template()):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 8. 零统计量（源码级）─────────────────────────────────────────
def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned


def test_source_has_no_statistics_or_network_or_io():
    src = MODULE_PATH.read_text(encoding="utf-8")
    lower = src.lower()
    for banned in ("scipy", "numpy", "statsmodels", "p_value", "effect_size",
                   "socket", "requests", "urllib", "http", "subprocess",
                   "open(", "pathlib", "sqlite", "csv."):
        assert banned not in lower, banned


# ── 9. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_EXECUTION_INTAKE_VERSION.startswith("v3h13-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == ["hashlib", "json"], plain


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


def test_no_client_automation_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "screenshot",
                   "input(", "advapi32", "win32"):
        assert banned not in src, banned


# ── 10. 生产 import 图零新增 ──────────────────────────────────────
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
    # 集合包含式断言：本阶段必须存在的层都仍在，且仍零接线
    required = {
        "cash_strategy_action_policy",
        "cash_strategy_action_ranking",
        "cash_strategy_belief_conditioned_eval",
        "cash_strategy_belief_eval",
        "cash_strategy_belief_update",
        "cash_strategy_decision_assembly",
        "cash_strategy_dynamic_belief",
        "cash_strategy_evidence_closure",
        "cash_strategy_execution_intake",
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


# ── 11. frozen / sealed drift · Solver ───────────────────────────
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


# ── 12. 真实全栈端到端 ───────────────────────────────────────────
def test_end_to_end_real_stack_intake_refuses_while_frozen_facts_hold():
    st = cash_state()
    clo = real_closure(st, [make_shadow_observation("check", 0.0),
                            make_shadow_observation("fold", None)])
    pre = build_preregistration(st, closure=clo, solver=solver_status(st),
                                environment=ALL_FACTS)
    recs = [exec_rec(0), exec_rec(1)]
    r = intake_execution_records(st, records=recs, preregistration=pre,
                                 solver=solver_status(st),
                                 environment=ALL_FACTS,
                                 trust_anchor_verified=True, min_records=1)
    assert r["intake_permitted"] is False
    assert r["n_admitted"] == 0
    assert r["n_candidates"] == 2
    assert r["executed"] is False
    assert r["result"] is None
    assert r["provenance"]["evidence_level"] == "UNAVAILABLE"
    assert r["solver"]["equilibrium"] is None
    s = summarize_intake(st, intake=r)
    assert s["counts"]["n_admitted"] == 0
    assert s["rate"] is None


def test_end_to_end_digest_roundtrip_on_real_shadow_identifiers():
    """真实 V3H-10 输出 → 用其标识构造执行记录 → 指纹自洽可校验。"""
    st = cash_state()
    ap = build_action_policy(st, action_tree=build_action_tree(st))
    bl = build_dynamic_belief(st, range_model=build_range_model(st))
    rk = rank_actions(st, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(st, action_policy=ap, ranking=rk)
    tr = build_search_tree(st, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(st, search_tree=tr)
    dec = assemble_decision(st, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(st))
    sh = evaluate_shadow_decision(
        st, decision=dec, action_policy=ap,
        observed=make_shadow_observation("check", 0.0),
        equity_snapshot=conditional_absolute_equity)
    rec = exec_rec(0, observed_action=(sh["record"]["observed"] or {}).get(
        "action"))
    v = validate_execution_record(st, rec)
    assert v["valid"] is True, v["issues"]
    assert v["digest_recomputed_matches"] is True
    assert len(v["integrity_digest"]) == 64
    assert v["record_fingerprint"] is not None
