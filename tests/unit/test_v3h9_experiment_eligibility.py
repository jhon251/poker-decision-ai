# -*- coding: utf-8 -*-
"""V3H-9 测试 — CASH Decision-to-Experiment Eligibility Audit。

覆盖：CASH-only fail closed / 缺决策 / 缺闸门（fail closed，不自行放行）/
真实全栈对账→资格审计端到端 / 闸门**唯一口径**来自 V3G-10（源码解析校验键集）/
上游键名不对称如实建模 / 闸门形状非法被拒 / 弃权·分歧·unknown 三类 fail-closed /
无评分·无 EV·无执行 / UNKNOWN 原样传播 / 定输入复现 / Solver BLOCKED 不翻转 /
AST·import 隔离（零 V3G import）/ 生产 import 图零新增 / frozen·sealed drift。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    CLAIM_BOUNDARY, ENVIRONMENT_FACT_KEYS, GATE_MISSING_KEYS,
    UPSTREAM_KEY_ASYMMETRY, audit_experiment_eligibility)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_profit_interface import evaluate_entry_gate  # noqa: E402
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
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402

NEW_MODULE = "cash_strategy_experiment_eligibility"
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


def GATE(environment=None) -> dict:
    """上游唯一口径（测试侧调用）。"""
    return evaluate_entry_gate(environment=environment or {})


def GATE_PASS() -> dict:
    return evaluate_entry_gate(environment=ALL_FACTS)


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


def good_decision(**over) -> dict:
    """合成一个"已对账成功"的 V3H-8 决策摘要（隔离测试闸门逻辑）。"""
    d = {"present": True, "level": "MODEL_INTERNAL", "abstained": False,
         "disagreements": [], "unknown": [], "street": "flop",
         "evidence_grade": "MODEL_INTERNAL",
         "preferred": {"action": "check", "candidate_key": ["check", 0.0],
                       "selected_by_tier": "search",
                       "score_source": "MODEL_INTERNAL", "leaf_coverage": 1.0}}
    d.update(over)
    return d


def real_decision(state, **over):
    ap = build_action_policy(state, action_tree=build_action_tree(state))
    bl = build_dynamic_belief(state, range_model=build_range_model(state))
    rk = rank_actions(state, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(state, action_policy=ap, ranking=rk)
    tr = build_search_tree(state, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    return assemble_decision(state, ranking=rk, belief_conditioned=bc,
                             search_eval=se, solver=solver_status(state), **over)


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    r = audit_experiment_eligibility(cash_state(game_type=mode),
                                     decision=good_decision(), gate=GATE())
    assert r["present"] is False
    assert r["eligible"] is False
    assert r["executed"] is False
    assert r["status"] == "BLOCKED"
    assert r["result_status"] == "EXECUTION_ENVIRONMENT_REQUIRED"
    assert r["evidence_grade"] == "NONE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42])
def test_non_dict_fail_closed(bad):
    assert audit_experiment_eligibility(bad)["present"] is False


def test_missing_decision_fails_closed():
    r = audit_experiment_eligibility(cash_state(), gate=GATE())
    assert r["present"] is False
    assert r["reason"] == "decision_required"
    assert r["eligible"] is False


def test_missing_gate_fails_closed_and_never_self_authorizes():
    """未提供闸门 → 绝不自行放行（不重实现判据）。"""
    r = audit_experiment_eligibility(cash_state(), decision=good_decision())
    assert r["present"] is True
    assert r["eligible"] is False
    assert r["status"] == "BLOCKED"
    assert "entry_gate_unavailable" in r["unknown"]
    assert r["entry_gate"]["supplied_by_caller"] is False
    assert r["entry_gate"]["allowed"] is False
    assert r["entry_gate"]["missing"] == list(GATE_MISSING_KEYS)


def test_decision_with_present_false_is_audited_as_blocked():
    r = audit_experiment_eligibility(cash_state(),
                                     decision={"present": False,
                                               "reason": "no_scorer_supplied"},
                                     gate=GATE_PASS())
    assert r["present"] is True
    assert r["eligible"] is False
    assert r["audit"][0]["blockers"][0]["blocker"] == "decision_absent"


# ── 2. 端到端：真实全栈 → 资格审计 ────────────────────────────────────
def test_real_full_stack_audit_is_blocked_by_environment():
    st = cash_state()
    dec = real_decision(st)
    r = audit_experiment_eligibility(st, decision=dec, gate=GATE(),
                                     solver=solver_status(st))
    assert r["present"] is True
    assert dec["present"] is True
    assert r["audit"][0]["decision_available"] is True
    assert r["audit"][0]["decision_abstained"] is False
    assert r["eligible"] is False
    assert r["status"] == "BLOCKED"
    assert r["result_status"] == "EXECUTION_ENVIRONMENT_REQUIRED"
    assert r["entry_gate"]["allowed"] is False
    assert len(r["entry_gate"]["missing"]) == len(GATE_MISSING_KEYS)


def test_real_full_stack_audit_records_decision_provenance():
    st = cash_state()
    r = audit_experiment_eligibility(st, decision=real_decision(st), gate=GATE())
    e = r["audit"][0]
    assert e["selected_by_tier"] == "search"
    assert e["candidate_key"] == ["check", 0.0]
    assert e["score_source"] == "MODEL_INTERNAL"
    assert e["leaf_coverage"] == 1.0
    assert e["evidence_grade"] == "MODEL_INTERNAL"


def test_audit_never_executes():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=GATE_PASS())
    assert r["executed"] is False
    assert r["audit"][0]["executed"] is False
    assert r["profit_identifiable"] is False
    assert CLAIM_BOUNDARY["executes_experiment"] is False


# ── 3. 闸门复用：唯一口径来自 V3G-10（源码解析强制）──────────────────
def test_input_fact_keys_match_upstream_source():
    import inspect
    import re
    src = inspect.getsource(evaluate_entry_gate)
    upstream = re.findall(r'env\.get\("([a-z_]+)"\)', src)
    assert sorted(upstream) == sorted(ENVIRONMENT_FACT_KEYS), (
        upstream, ENVIRONMENT_FACT_KEYS)


def test_output_missing_keys_match_upstream_source():
    import inspect
    import re
    src = inspect.getsource(evaluate_entry_gate)
    block = src[src.find("need = {"):]
    block = block[:block.find("}")]
    upstream = re.findall(r'"([a-z_]+)":', block)
    assert sorted(GATE_MISSING_KEYS) == sorted(upstream), (upstream, GATE_MISSING_KEYS)


def test_upstream_key_asymmetry_is_modelled_not_hidden():
    assert UPSTREAM_KEY_ASYMMETRY["input_key"] != \
        UPSTREAM_KEY_ASYMMETRY["output_missing_key"]
    assert "NOT modified" in UPSTREAM_KEY_ASYMMETRY["handling"]
    assert "venue_and_compliance_authorized" in ENVIRONMENT_FACT_KEYS
    assert "venue_and_compliance_authorization" in GATE_MISSING_KEYS


def test_supplying_official_input_key_satisfies_the_criterion():
    r = audit_experiment_eligibility(
        cash_state(), decision=good_decision(),
        gate=GATE({"venue_and_compliance_authorized": True}))
    assert "venue_and_compliance_authorization" not in r["entry_gate"]["missing"]


def test_gate_result_is_passed_through_not_recomputed():
    g = GATE()
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=g)
    assert r["entry_gate"]["source"].startswith("V3G-10")
    assert r["entry_gate"]["reimplemented_here"] is False
    assert r["entry_gate"]["missing"] == g["missing"]
    assert r["entry_gate"]["allowed"] == g["allowed"]
    assert r["upstream_key_asymmetry"] == UPSTREAM_KEY_ASYMMETRY


def test_all_facts_satisfied_makes_decision_eligible_not_executed():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=GATE_PASS())
    assert r["eligible"] is True
    assert r["status"] == "ELIGIBLE"
    assert r["result_status"] == "ELIGIBLE_NOT_EXECUTED"
    assert r["executed"] is False
    assert r["n_blockers"] == 0


def test_partial_facts_record_missing_criteria():
    r = audit_experiment_eligibility(
        cash_state(), decision=good_decision(),
        gate=GATE({"executable_environment": True}))
    assert r["eligible"] is False
    assert "executable_environment" not in r["entry_gate"]["missing"]
    assert len(r["entry_gate"]["missing"]) == len(GATE_MISSING_KEYS) - 1


# ── 4. 闸门形状非法 → 拒绝（不静默补全、不猜测）────────────────────────
def test_gate_allowed_non_boolean_is_rejected():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate={"allowed": "yes", "missing": []})
    assert r["eligible"] is False
    assert any("allowed_not_boolean" in u for u in r["unknown"])


def test_gate_missing_not_a_list_is_rejected():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate={"allowed": False, "missing": "all"})
    assert r["eligible"] is False
    assert any("missing_not_a_list" in u for u in r["unknown"])


def test_inconsistent_gate_is_treated_as_not_allowed():
    """allowed=True 但 missing 非空 → 冲突 → 按未满足处理（不信任而不放行）。"""
    r = audit_experiment_eligibility(
        cash_state(), decision=good_decision(),
        gate={"allowed": True, "missing": ["design_sealed"]})
    assert r["eligible"] is False
    assert any("gate_inconsistent" in u for u in r["unknown"])


def test_gate_with_invalid_missing_entry_is_rejected():
    r = audit_experiment_eligibility(
        cash_state(), decision=good_decision(),
        gate={"allowed": False, "missing": [123]})
    assert any("missing_entry_invalid" in u for u in r["unknown"])


# ── 5. 三类 fail-closed（弃权 / 分歧 / unknown）────────────────────────
def test_abstained_decision_blocked_even_when_gate_passes():
    d = good_decision(abstained=True, preferred=None, level="UNKNOWN",
                      abstention_reason="cross_scorer_disagreement")
    r = audit_experiment_eligibility(cash_state(), decision=d, gate=GATE_PASS())
    assert r["eligible"] is False
    blockers = [b["blocker"] for b in r["audit"][0]["blockers"]]
    assert "decision_abstained" in blockers
    assert "decision_level_not_model_internal" in blockers


def test_unresolved_disagreement_blocks():
    d = good_decision(disagreements=[{"tier_a": "search", "tier_b": "ranking"}])
    r = audit_experiment_eligibility(cash_state(), decision=d, gate=GATE_PASS())
    assert r["eligible"] is False
    assert any(b["blocker"] == "unresolved_disagreements"
               for b in r["audit"][0]["blockers"])


def test_decision_unknown_blocks_and_propagates():
    d = good_decision(unknown=["search_not_supplied"])
    r = audit_experiment_eligibility(cash_state(), decision=d, gate=GATE_PASS())
    assert r["eligible"] is False
    assert any(b["blocker"] == "decision_reports_unknown"
               for b in r["audit"][0]["blockers"])
    assert "search_not_supplied" in r["unknown"]


def test_non_model_internal_level_blocks():
    d = good_decision(level="UNKNOWN")
    r = audit_experiment_eligibility(cash_state(), decision=d, gate=GATE_PASS())
    assert r["eligible"] is False
    assert r["level"] == "UNKNOWN"


def test_real_disagreement_case_end_to_end_blocks():
    """真实分歧：三层 top 不一致 → 弃权 → 审计也必须阻断。"""
    st = cash_state()
    syn_rank = {"present": True, "street": "flop",
                "ranked": [{"action": "bet", "amount_bb": 3.3, "score": 0.2,
                            "score_source": "MODEL_INTERNAL"}]}
    syn_search = {"present": True, "street": "flop",
                  "ranked": [{"action": "check", "amount_bb": 0.0,
                              "search_score": 0.4, "leaf_coverage": 1.0,
                              "score_source": "MODEL_INTERNAL"}]}
    dec = assemble_decision(st, ranking=syn_rank, search_eval=syn_search)
    assert dec["abstained"] is True
    r = audit_experiment_eligibility(st, decision=dec, gate=GATE_PASS())
    assert r["eligible"] is False
    assert "decision_abstained" in [b["blocker"]
                                    for b in r["audit"][0]["blockers"]]


# ── 6. UNKNOWN 原样传播 / Solver ──────────────────────────────────────
def test_unknown_propagation_from_all_sources():
    d = good_decision(unknown=["a_unknown"])
    r = audit_experiment_eligibility(cash_state(), decision=d,
                                     gate={"allowed": False, "missing": [5]})
    assert "a_unknown" in r["unknown"]
    assert any("missing_entry_invalid" in u for u in r["unknown"])


def test_solver_without_input_is_unknown_but_blocked():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=GATE())
    assert r["solver"]["solver_status"] == "UNKNOWN"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["read_only_passthrough"] is True
    assert "solver_status_not_supplied" in r["unknown"]


def test_solver_remains_blocked():
    st = cash_state()
    before = solver_status(st).get("solver_status")
    r = audit_experiment_eligibility(st, decision=real_decision(st),
                                     gate=GATE(), solver=solver_status(st))
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"
    assert r["solver"]["solver_status"] == "BLOCKED"
    assert r["solver"]["blocked"] is True


# ── 7. 无评分 / 无 EV / 无 GTO ────────────────────────────────────────
@pytest.mark.parametrize("flag", ["is_gto", "is_equilibrium", "is_solver_output",
                                  "is_profit_evidence",
                                  "is_real_opponent_evidence",
                                  "computes_scoring_or_ev",
                                  "reimplements_entry_gate",
                                  "executes_experiment",
                                  "imports_any_v3g_layer",
                                  "wired_to_production"])
def test_claim_boundary_flags(flag):
    assert CLAIM_BOUNDARY[flag] is False


def test_no_ev_or_score_fields_in_output():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=GATE_PASS())
    blob = json.dumps(r, default=str).lower()
    for banned in ("hero_equity", "search_score", "equity_minus_required",
                   "expected_value", "bb_per_100", "win_rate", "roi",
                   "p_value", "exploitability"):
        assert banned not in blob, banned


def test_no_positive_gto_flags():
    blob = json.dumps(audit_experiment_eligibility(
        cash_state(), decision=good_decision(), gate=GATE_PASS()),
        default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true'):
        assert bad not in blob


def test_model_internal_never_claimed_as_real_evidence():
    r = audit_experiment_eligibility(cash_state(), decision=good_decision(),
                                     gate=GATE())
    assert "不是" in r["note"]
    assert CLAIM_BOUNDARY["is_real_opponent_evidence"] is False
    assert CLAIM_BOUNDARY["is_profit_evidence"] is False


def test_module_contains_no_scoring_code():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("hero_equity", "pot_share", "equity(", "expected_value",
                   "bb_per_100", "win_rate", "p_value"):
        assert banned not in src, banned


# ── 8. 定输入复现 / 2–6 人 / 四街 ─────────────────────────────────────
def test_deterministic_reproduction():
    st = cash_state()
    dec = real_decision(st)
    a = json.dumps(audit_experiment_eligibility(st, decision=dec,
                                                gate=GATE_PASS()),
                   sort_keys=True, default=str)
    b = json.dumps(audit_experiment_eligibility(st, decision=dec,
                                                gate=GATE_PASS()),
                   sort_keys=True, default=str)
    assert a == b


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_player_counts(n):
    st = cash_state(n_players=n)
    r = audit_experiment_eligibility(st, decision=real_decision(st), gate=GATE())
    assert r["present"] is True
    assert r["eligible"] is False


@pytest.mark.parametrize("street,board,to_call", [
    ("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
    ("turn", ["Qs", "7d", "2c", "Th"], 0.0)])
def test_streets(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call)
    r = audit_experiment_eligibility(st, decision=real_decision(st), gate=GATE())
    assert r["present"] is True
    assert r["street"] == street
    assert r["eligible"] is False


def test_river_upstream_terminal_still_audits_fail_closed():
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    r = audit_experiment_eligibility(st, decision=real_decision(st),
                                     gate=GATE_PASS())
    assert r["present"] is True
    assert r["eligible"] is False   # 决策缺席（上游 terminal）→ 阻断，不伪造


# ── 9. AST / import 隔离 ──────────────────────────────────────────────
def test_no_v3g_import_anywhere_in_module():
    """本层**任何位置**都不得 import V3G 层（含函数内延迟 import）。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)


# ── 10. 生产 import 图零新增 ──────────────────────────────────────────
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


def test_v3g11_boundary_semantics_preserved():
    """V3G-11 语义：包内任何非层文件都不得 import V3G 层 —— 本层同样遵守。"""
    assert "cash_profit_interface" not in _production_importers()


# ── 11. frozen / sealed drift ─────────────────────────────────────────
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
