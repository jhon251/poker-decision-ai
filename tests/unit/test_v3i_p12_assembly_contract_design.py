"""V3I-P12 — V3H-8 契约变更设计 + 最小实验验证（测试）。

本阶段**只做设计 + 最小实验**，**不修改任何冻结代码**。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_assembly_contract_design import (
    CASH_P12_CONTRACT_DESIGN_VERSION,
    CLAIM_BOUNDARY,
    PROPOSED_CHANGE,
    default_path_unchanged_check,
    normalize_feasible,
    p12_contract,
    reproduce_root_cause,
    simulate_feasible_domain_assembly,
    stable_fingerprint,
    true_disagreement_probe,
)

P12_MODULE = pathlib.Path("poker_decision_ai/cash_assembly_contract_design.py")
ASSEMBLY = pathlib.Path("poker_decision_ai/cash_strategy_decision_assembly.py")


# ---------------------------------------------------------------------------
# T1. 契约与边界（要求 1、2、4、6、8、9、10、11）
# ---------------------------------------------------------------------------
def test_t1_contract_target_and_flags():
    c = p12_contract()
    assert c["version"] == CASH_P12_CONTRACT_DESIGN_VERSION
    assert c["modifies_frozen_code"] is False
    pc = c["proposed_change"]
    assert pc["file"].endswith("cash_strategy_decision_assembly.py")
    assert pc["function"] == "assemble_decision"
    assert pc["added_parameter"] == "feasible_keys"
    assert pc["default_preserves_current_behaviour"] is True
    assert pc["modifies_scoring_formula"] is False
    assert pc["modifies_tier_precedence"] is False
    assert pc["modifies_override_semantics"] is False
    assert pc["adds_tolerance_or_voting"] is False
    assert pc["relaxes_fail_closed"] is False


def test_t1_preserved_fail_closed_conditions_declared():
    pc = PROPOSED_CHANGE
    joined = " ".join(pc["preserved_fail_closed_conditions"])
    for key in ("无候选", "仍不一致", "leaf_coverage_below_minimum",
                "search_score_unavailable", "override"):
        assert key in joined, key


def test_t1_affected_existing_tests_declared():
    assert PROPOSED_CHANGE["affected_existing_tests"]
    joined = " ".join(PROPOSED_CHANGE["affected_existing_tests"])
    assert "test_v3h8_decision_assembly" in joined


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "modifies_frozen_code_in_this_stage",
                 "relaxes_abstain", "enables_override",
                 "weakens_unknown_or_fail_closed"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib():
    src = P12_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    banned = {"setattr", "delattr", "globals", "eval", "exec", "__import__"}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned), called & banned


def test_t1_frozen_assembly_not_modified_in_this_stage():
    """本测试原断言"设计阶段未修改冻结 V3H-8"。

    **V3I-P13 经授权实施了该契约**，故该断言按设计失效 —— 现改为断言：
    契约已实施（`feasible_keys` 存在），且**冻结语义未被改动**。
    """
    src = ASSEMBLY.read_text(encoding="utf-8")
    # 契约已实施（V3I-P13）
    assert "feasible_keys" in src
    assert "def _normalize_feasible_keys(" in src
    assert "def _key_allowed(" in src
    # 关键冻结结构仍在（语义未变）
    for key in ("TIERS", "on_disagreement", "abstain_unknown",
                "def assemble_decision(", "override_justification",
                "cross_scorer_disagreement"):
        assert key in src, key


def test_t1_other_frozen_layers_untouched():
    frozen = {
        "poker_decision_ai/cash_strategy_search_tree.py": None,
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    }
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        if want is not None:
            assert got == want, (path, got, want)
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


# ---------------------------------------------------------------------------
# T2. 根因可复现（要求 3）
# ---------------------------------------------------------------------------
def test_t2_root_cause_domain_mismatch_reproduced():
    ap_keys = [("check", 0.0), ("bet", 0.495), ("bet", 0.75),
               ("bet", 0.99), ("bet", 1.5), ("bet", 100.0)]
    search_keys = [("check", 0.0), ("bet", 0.495)]
    ranked = [
        {"action": "bet", "amount_bb": 100.0, "score": 53.518008},
        {"action": "bet", "amount_bb": 1.5, "score": 1.684025},
        {"action": "bet", "amount_bb": 0.99, "score": 1.415646},
        {"action": "bet", "amount_bb": 0.75, "score": 1.28935},
        {"action": "bet", "amount_bb": 0.495, "score": 1.155161},
        {"action": "check", "amount_bb": 0.0, "score": 0.894675},
    ]
    rc = reproduce_root_cause(action_policy_keys=ap_keys,
                             search_domain_keys=search_keys,
                             ranking_ranked=ranked)
    assert rc["present"] is True
    assert rc["domain_mismatch"] is True
    assert rc["action_policy_domain_size"] == 6
    assert rc["search_domain_size"] == 2
    assert rc["ranking_top_global"] == ["bet", 100.0]
    assert rc["ranking_top_limited_to_search_domain"] == ["bet", 0.495]
    assert rc["ranking_top_global_in_search_domain"] is False
    assert ["bet", 100.0] in rc["ranking_only_candidates"]
    assert "不对等候选域" in rc["root_cause"]


def test_t2_no_mismatch_when_domains_equal():
    keys = [("check", 0.0), ("bet", 0.495)]
    rc = reproduce_root_cause(action_policy_keys=keys, search_domain_keys=keys,
                             ranking_ranked=[
                                 {"action": "bet", "amount_bb": 0.495,
                                  "score": 1.1},
                                 {"action": "check", "amount_bb": 0.0,
                                  "score": 0.9}])
    assert rc["domain_mismatch"] is False
    assert rc["ranking_top_global"] == rc["ranking_top_limited_to_search_domain"]


def test_t2_root_cause_tolerates_empty_inputs():
    """空输入不抛异常；域为空时如实报告，不发明候选。"""
    rc = reproduce_root_cause(action_policy_keys=None,
                              search_domain_keys=None, ranking_ranked=None)
    assert rc["present"] is True
    assert rc["action_policy_domain_size"] == 0
    assert rc["search_domain_size"] == 0
    assert rc["ranking_top_global"] is None
    assert rc["ranking_top_limited_to_search_domain"] is None


# ---------------------------------------------------------------------------
# T3. 提议契约消除结构性伪分歧（要求 4、5）
# ---------------------------------------------------------------------------
def test_t3_restriction_removes_false_disagreement():
    sim = simulate_feasible_domain_assembly(
        tier_ranked={
            "search": [{"action": "bet", "amount_bb": 0.495, "score": 0.99}],
            "ranking": [
                {"action": "bet", "amount_bb": 100.0, "score": 53.5},
                {"action": "bet", "amount_bb": 0.495, "score": 1.155},
                {"action": "check", "amount_bb": 0.0, "score": 0.895}],
        },
        feasible_keys=[("check", 0.0), ("bet", 0.495)],
        current_out={"abstained": True, "preferred": None,
                     "disagreements": [{"tier_a": "search",
                                        "tier_b": "ranking"}]})
    assert sim["present"] is True
    assert sim["per_tier"]["ranking"]["changed_by_restriction"] is True
    assert sim["agreed_within_feasible_domain"] is True
    assert sim["would_accept_after_change"] is True
    assert sim["would_abstain_after_change"] is False
    assert sim["selected_key"] == ["bet", 0.495]
    assert sim["current_frozen_output_summary"]["abstained"] is True


def test_t3_restriction_is_explained_as_not_packaging_inconsistency():
    sim = simulate_feasible_domain_assembly(
        tier_ranked={"search": [{"action": "check", "amount_bb": 0.0,
                                 "score": 1.0}],
                     "ranking": [{"action": "check", "amount_bb": 0.0,
                                  "score": 2.0}]},
        feasible_keys=[("check", 0.0)])
    txt = sim["why_not_packaging_inconsistency_as_consistency"]
    assert "不修改评分" in txt and "仍" in txt


def test_t3_simulate_fail_closed_on_missing_inputs():
    assert simulate_feasible_domain_assembly(tier_ranked=None,
                                             feasible_keys=None)["present"] is False
    assert simulate_feasible_domain_assembly(tier_ranked={},
                                             feasible_keys=[])["present"] is False
    assert simulate_feasible_domain_assembly(
        tier_ranked={"a": []}, feasible_keys=[])["present"] is False


# ---------------------------------------------------------------------------
# T4. 真分歧仍弃权（要求 4）
# ---------------------------------------------------------------------------
def test_t4_true_disagreement_still_abstains():
    tp = true_disagreement_probe(tier_ranked={
        "search": [{"action": "bet", "amount_bb": 0.5, "score": 1.0},
                   {"action": "check", "amount_bb": 0.0, "score": 0.5}],
        "ranking": [{"action": "check", "amount_bb": 0.0, "score": 2.0},
                    {"action": "bet", "amount_bb": 0.5, "score": 1.0}],
    })
    assert tp["present"] is True
    assert tp["agreed"] is False
    assert tp["still_abstains"] is True
    assert "真分歧被正确保留" in tp["verdict"]
    assert tp["limited_tops"]["search"] == ["bet", 0.5]
    assert tp["limited_tops"]["ranking"] == ["check", 0.0]


def test_t4_tier_unavailable_in_feasible_domain_still_abstains():
    """某 tier 在共同可行域内无候选 → 该 tier 不可用 → 仍弃权（不伪装）。"""
    sim = simulate_feasible_domain_assembly(
        tier_ranked={
            "search": [{"action": "bet", "amount_bb": 0.5, "score": 1.0}],
            "ranking": [{"action": "bet", "amount_bb": 99.0, "score": 9.0}],
        },
        feasible_keys=[("bet", 0.5)])
    assert sim["tiers_unavailable_within_feasible_domain"] == ["ranking"]
    assert sim["would_abstain_after_change"] is True
    assert sim["would_accept_after_change"] is False


def test_t4_true_disagreement_probe_fail_closed():
    assert true_disagreement_probe(tier_ranked=None)["present"] is False
    assert true_disagreement_probe(tier_ranked={"a": []})["present"] is False


# ---------------------------------------------------------------------------
# T5. baseline 可追溯 + determinism（要求：baseline / deterministic）
# ---------------------------------------------------------------------------
def test_t5_default_path_unchanged():
    dv = default_path_unchanged_check(
        frozen_out={"abstained": True, "preferred": None,
                    "disagreements": [{"tier_a": "search"}]},
        simulated_without_feasible=None)
    assert dv["present"] is True
    assert dv["proposed_default_is_none"] is True
    assert dv["behaviour_change_on_default_path"] is False
    assert dv["baseline_traceable"] is True
    assert dv["frozen_abstained"] is True
    assert dv["frozen_n_disagreements"] == 1


def test_t5_default_path_check_fail_closed():
    assert default_path_unchanged_check(frozen_out=None,
                                        simulated_without_feasible=None
                                        )["present"] is False


def test_t5_deterministic_simulation():
    args = dict(
        tier_ranked={
            "search": [{"action": "bet", "amount_bb": 0.495, "score": 0.99}],
            "ranking": [
                {"action": "bet", "amount_bb": 100.0, "score": 53.5},
                {"action": "bet", "amount_bb": 0.495, "score": 1.155}],
        },
        feasible_keys=[("bet", 0.495), ("check", 0.0)])
    a = simulate_feasible_domain_assembly(**args)
    b = simulate_feasible_domain_assembly(**args)
    assert stable_fingerprint(a) == stable_fingerprint(b)


def test_t5_normalize_feasible_dedupes_and_sorts():
    got = normalize_feasible([("bet", 0.5), ("check", 0.0), ("bet", 0.5)])
    assert got == [("bet", 0.5), ("check", 0.0)]
    assert normalize_feasible(None) == []
    assert normalize_feasible([]) == []


def test_t5_no_external_claims_in_outputs():
    rc = reproduce_root_cause(action_policy_keys=[("check", 0.0)],
                             search_domain_keys=[("check", 0.0)],
                             ranking_ranked=[{"action": "check",
                                              "amount_bb": 0.0, "score": 1.0}])
    # `claim_boundary` 的**键名**本身就包含这些词（且值为 False，即"否认该宣称"），
    # 因此不能用子串匹配判断"是否宣称"——必须检查值。
    assert rc["evidence_grade"] == "MODEL_INTERNAL"
    assert all(v is False for v in rc["claim_boundary"].values())
    # 正向宣称检测：只禁止"被设为 True"
    blob = json.dumps(rc, default=str)
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"is_solver_output": true', '"is_ev": true',
                '"is_external_validation": true',
                '"is_profit_evidence": true', '"is_real_opponent_data": true',
                '"relaxes_abstain": true', '"enables_override": true',
                '"modifies_frozen_code_in_this_stage": true'):
        assert bad not in blob, bad
