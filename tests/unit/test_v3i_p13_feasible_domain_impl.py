"""V3I-P13 — 实施并验证 P12 设计的 V3H-8 feasible-domain 最小契约（测试）。

契约已实施于 `cash_strategy_decision_assembly.assemble_decision`
（新增可选 `feasible_keys`）；本文件验证：默认路径不变、限域消除结构性伪分歧、
真分歧/空域/UNKNOWN 仍弃权、可行域来源合法、deterministic、frozen drift。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_feasible_domain_wiring import (
    CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION,
    CLAIM_BOUNDARY,
    feasible_keys_from_search_tree,
    make_feasible_domain_assembler,
    p13_contract,
    stable_fingerprint,
)
from poker_decision_ai.cash_strategy_decision_assembly import (
    ASSEMBLY_POLICY,
    TIERS,
    assemble_decision,
)

WIRING = pathlib.Path("poker_decision_ai/cash_feasible_domain_wiring.py")
ASSEMBLY = pathlib.Path("poker_decision_ai/cash_strategy_decision_assembly.py")
RECORDS = pathlib.Path(".tmp/v3i_p13_records.json")


def _state():
    return {"game_type": "cash", "street": "flop"}


def _ranked(rows):
    return {"present": True, "ranked": rows, "preferred": rows[0] if rows else None}


# ---------------------------------------------------------------------------
# T1. 契约实施与边界（要求 1、4、7）
# ---------------------------------------------------------------------------
def test_t1_feasible_keys_parameter_implemented():
    sig = inspect.signature(assemble_decision)
    assert "feasible_keys" in sig.parameters
    p = sig.parameters["feasible_keys"]
    assert p.default is None, "默认必须为 None（默认路径不变）"


def test_t1_frozen_semantics_untouched():
    """要求 1：不改评分公式、tier 优先级、override、ABSTAIN/UNKNOWN 语义。"""
    assert TIERS == ("search", "belief_conditioned", "ranking")
    assert ASSEMBLY_POLICY.get("on_disagreement") == "abstain_unknown"
    assert ASSEMBLY_POLICY.get("override_default") is False
    src = ASSEMBLY.read_text(encoding="utf-8")
    assert "override_used = False" in src
    assert "abstained = bool(disagreements) and not override_used" in src
    assert "cross_scorer_disagreement" in src


def test_t1_wiring_contract_declares_no_expansion():
    c = p13_contract()
    assert c["version"] == CASH_P13_FEASIBLE_DOMAIN_WIRING_VERSION
    assert c["expands_search_domain"] is False
    assert c["enables_override"] is False
    assert c["modifies_scoring_formula_or_tier_precedence"] is False
    assert c["modifies_solver_v3h4_v3h6_cfi_lambda_river_budget"] is False
    assert "pipeline['assemble_decision']" in c["injection"]


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "enables_override", "relaxes_abstain",
                 "expands_search_domain"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib():
    src = WIRING.read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# T2. 可行域来源合法（要求 2）
# ---------------------------------------------------------------------------
def test_t2_feasible_keys_come_from_search_expansion():
    tree = {"present": True, "root": {"children": [
        {"action": "check", "amount_bb": 0.0},
        {"action": "bet", "amount_bb": 0.495}]}}
    r = feasible_keys_from_search_tree(search_tree=tree)
    assert r["present"] is True
    assert r["feasible_keys"] == [["check", 0.0], ["bet", 0.495]]
    assert r["n_keys"] == 2
    assert r["source"] == "search_tree.root.children"
    assert r["derived_from_actual_search_expansion"] is True
    assert r["expands_search_domain"] is False
    assert r["fabricated"] is False


def test_t2_feasible_keys_fail_closed_on_bad_tree():
    for bad in (None, {}, {"present": False},
                {"present": True}, {"present": True, "root": {}},
                {"present": True, "root": {"children": []}}):
        r = feasible_keys_from_search_tree(search_tree=bad)
        assert r["present"] is False, bad


def test_t2_feasible_keys_do_not_expand_search_domain():
    """可行域必须**等于** search 展开集，不得多出候选。"""
    tree = {"present": True, "root": {"children": [
        {"action": "check", "amount_bb": 0.0}]}}
    r = feasible_keys_from_search_tree(search_tree=tree)
    assert r["n_keys"] == 1
    assert r["feasible_keys"] == [["check", 0.0]]


# ---------------------------------------------------------------------------
# T3. 默认路径不变（要求 6）
# ---------------------------------------------------------------------------
def _scorers():
    search = _ranked([{"action": "bet", "amount_bb": 0.495,
                       "search_score": 0.99, "leaf_coverage": 1.0,
                       "size_bucket": "small"}])
    ranking = _ranked([{"action": "bet", "amount_bb": 100.0, "score": 53.5},
                       {"action": "bet", "amount_bb": 0.495, "score": 1.155},
                       {"action": "check", "amount_bb": 0.0, "score": 0.895}])
    return search, ranking


def test_t3_default_path_still_abstains():
    search, ranking = _scorers()
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            belief_conditioned=None, solver=None, policy=None)
    assert out["present"] is True
    assert out["abstained"] is True
    assert out["preferred"] is None
    assert len(out["disagreements"]) == 1
    assert out["feasible_domain_applied"] is False
    assert out["feasible_domain_size"] == 0


def test_t3_default_and_explicit_none_identical():
    search, ranking = _scorers()
    a = assemble_decision(_state(), ranking=ranking, search_eval=search,
                          solver=None, policy=None)
    b = assemble_decision(_state(), ranking=ranking, search_eval=search,
                          solver=None, policy=None, feasible_keys=None)
    assert stable_fingerprint(a) == stable_fingerprint(b)


# ---------------------------------------------------------------------------
# T4. 限域消除结构性伪分歧（要求 2、3）
# ---------------------------------------------------------------------------
def test_t4_feasible_domain_removes_false_disagreement():
    search, ranking = _scorers()
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None,
                            feasible_keys=[("check", 0.0), ("bet", 0.495)])
    assert out["feasible_domain_applied"] is True
    assert out["feasible_domain_size"] == 2
    assert out["abstained"] is False
    assert out["preferred"] is not None
    assert out["preferred"]["action"] == "bet"
    assert abs(out["preferred"]["amount_bb"] - 0.495) < 1e-9
    assert out["disagreements"] == []


def test_t4_selected_tier_still_governed_by_precedence():
    """限域不改变 tier 优先级：仍由 search（comparable[0]）选出。"""
    search, ranking = _scorers()
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None,
                            feasible_keys=[("check", 0.0), ("bet", 0.495)])
    # selected_by_tier 位于 preferred 行内（冻结输出形状）
    assert out["preferred"]["selected_by_tier"] == "search"


# ---------------------------------------------------------------------------
# T5. 真分歧 / 空域 / tier 不可用仍弃权（要求 3）
# ---------------------------------------------------------------------------
def test_t5_true_disagreement_still_abstains():
    """同一可行域内两层 top 不同 → 必须仍弃权。"""
    search = _ranked([{"action": "bet", "amount_bb": 0.5,
                       "search_score": 1.0, "leaf_coverage": 1.0}])
    ranking = _ranked([{"action": "check", "amount_bb": 0.0, "score": 2.0},
                       {"action": "bet", "amount_bb": 0.5, "score": 1.0}])
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None,
                            feasible_keys=[("check", 0.0), ("bet", 0.5)])
    assert out["abstained"] is True
    assert out["preferred"] is None
    assert len(out["disagreements"]) == 1


def test_t5_tier_unavailable_in_feasible_domain_is_recorded_not_faked():
    """某 tier 在可行域内无候选 → 记为 `*_top_candidate_unavailable`，
    **不伪装**该 tier 拥有 top；此时不与 search 构成分歧（忠实于冻结语义：
    不可比对者不参与对账），决策由可比的 search 层给出。"""
    search = _ranked([{"action": "bet", "amount_bb": 0.5,
                       "search_score": 1.0, "leaf_coverage": 1.0}])
    ranking = _ranked([{"action": "bet", "amount_bb": 99.0, "score": 9.0}])
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None,
                            feasible_keys=[("bet", 0.5)])
    # ranking 被如实记为"不可用"，且未产生假分歧
    assert any("ranking_top_candidate_unavailable" in str(u)
               for u in (out.get("unknown") or []))
    assert out["disagreements"] == []
    # 未被伪装：ranking 的 99.0 没有成为决策
    assert out["preferred"] is not None
    assert abs(out["preferred"]["amount_bb"] - 0.5) < 1e-9
    assert out["preferred"]["selected_by_tier"] == "search"


def test_t5_empty_feasible_keys_does_not_restrict():
    """空可行域 → 视为未提供（不制造"全部不可用"的假弃权）。"""
    search, ranking = _scorers()
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None, feasible_keys=[])
    assert out["feasible_domain_applied"] is False
    assert out["abstained"] is True  # 仍按原规则弃权


def test_t5_feasible_domain_does_not_enable_override():
    search, ranking = _scorers()
    out = assemble_decision(_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None,
                            feasible_keys=[("check", 0.0), ("bet", 0.495)])
    assert out.get("override_used") is False
    assert out.get("override_justification") is None


def test_t5_unknown_and_fail_closed_preserved():
    """非 CASH → fail-closed；无 scorer → 弃权。"""
    out = assemble_decision({"game_type": "mtt"}, ranking=None, search_eval=None)
    assert out["present"] is False
    out2 = assemble_decision(_state(), ranking=None, search_eval=None,
                             belief_conditioned=None, solver=None, policy=None,
                             feasible_keys=[("check", 0.0)])
    assert out2["present"] is False


# ---------------------------------------------------------------------------
# T6. 闭包接入（要求 4）
# ---------------------------------------------------------------------------
def test_t6_assembler_closure_defaults_to_frozen_behaviour():
    search, ranking = _scorers()
    built = make_feasible_domain_assembler(frozen_assemble=assemble_decision,
                                           feasible_keys=None)
    assert built["present"] is True
    assert built["will_apply_feasible_domain"] is False
    out = built["callable"](_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None)
    assert out["abstained"] is True
    assert out["feasible_domain_applied"] is False


def test_t6_assembler_closure_applies_feasible_domain():
    search, ranking = _scorers()
    built = make_feasible_domain_assembler(
        frozen_assemble=assemble_decision,
        feasible_keys=[("check", 0.0), ("bet", 0.495)])
    assert built["will_apply_feasible_domain"] is True
    assert built["enables_override"] is False
    out = built["callable"](_state(), ranking=ranking, search_eval=search,
                            solver=None, policy=None)
    assert out["abstained"] is False
    assert out["feasible_domain_applied"] is True


def test_t6_closure_signature_matches_frozen():
    built = make_feasible_domain_assembler(frozen_assemble=assemble_decision,
                                           feasible_keys=None)
    got = set(inspect.signature(built["callable"]).parameters)
    want = set(inspect.signature(assemble_decision).parameters) - {"feasible_keys"}
    assert got == want, (got, want)


def test_t6_closure_fail_closed_on_non_callable():
    assert make_feasible_domain_assembler(frozen_assemble=None,
                                          feasible_keys=None)["present"] is False


# ---------------------------------------------------------------------------
# T7. frozen drift（要求 4、6）
# ---------------------------------------------------------------------------
def test_t7_protected_layers_untouched():
    frozen = {
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
        "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
        "poker_decision_ai/realtime.py": "3400a20ff857880d",
    }
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == want, (path, got, want)


def test_t7_search_budget_cap_depth_unchanged():
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


def test_t7_no_production_wiring_added():
    def imports_of(p):
        tree = ast.parse(pathlib.Path(p).read_text(encoding="utf-8"))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module:
                    mods.add(n.module.split(".")[-1])
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
            elif isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
        return mods

    found = {}
    for f in ("poker_decision_ai/realtime.py",
              "poker_decision_ai/agents/judge/agent.py"):
        for m in sorted(imports_of(f)):
            if m.startswith("cash_"):
                found.setdefault(m, []).append(pathlib.Path(f).name)
    assert {k: sorted(set(v)) for k, v in found.items()} == {
        "cash_integrity_context": ["agent.py", "realtime.py"],
        "cash_unknown_e2e": ["realtime.py"],
    }, found
    for mod in ("cash_feasible_domain_wiring", "cash_assembly_contract_design"):
        assert mod not in found


# ---------------------------------------------------------------------------
# T8. 实测覆盖率守护（要求 5：禁止把预测当实测）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p13_records.json（由扫描脚本产出）")
def test_t8_measured_accept_rate_improved_from_zero():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    assert len(recs) == 24
    fd = [r for r in recs if r["mode"] == "feasible_domain"]
    base = [r for r in recs if r["mode"] == "baseline"]
    assert len(fd) == 18 and len(base) == 6
    # 限域运行：全部非 river 均 ACCEPT
    fd_accept = [r for r in fd if r.get("abstained") is False
                 and r.get("preferred") not in (None, {})]
    assert len(fd_accept) == 15, len(fd_accept)
    # 限域运行无 ABSTAIN（其余为 river BLOCKED）
    assert all(r.get("abstained") is not True for r in fd)
    # 基线仍弃权（行为可追溯）
    assert sum(1 for r in base if r.get("abstained") is True) == 5


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p13_records.json（由扫描脚本产出）")
def test_t8_measured_disagreements_eliminated_in_feasible_mode():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    fd = [r for r in recs if r["mode"] == "feasible_domain"]
    assert all((r.get("n_disagreements") or 0) == 0 for r in fd)
    # 提供了可行域的非 river 运行必须真的应用了限域
    with_keys = [r for r in fd if r.get("feasible_keys")]
    assert len(with_keys) == 15
    assert all(r.get("feasible_applied") is True for r in with_keys)
    base = [r for r in recs if r["mode"] == "baseline"]
    assert sum(1 for r in base if (r.get("n_disagreements") or 0) > 0) == 5


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p13_records.json（由扫描脚本产出）")
def test_t8_blocked_runs_are_river_only():
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))
    blocked = [r for r in recs if r["loop_status"] == "BLOCKED"]
    assert blocked
    assert all(r["street"] == "river" for r in blocked)


@pytest.mark.skipif(not RECORDS.exists(),
                    reason="需要 .tmp/v3i_p13_records.json（由扫描脚本产出）")
def test_t8_deterministic_classification():
    from poker_decision_ai.cash_decision_coverage_assessment import (
        classify_record)
    recs = json.loads(RECORDS.read_text(encoding="utf-8"))

    def adapt(r):
        return {"scenario": r["scenario"], "street": r["street"],
                "population": r["population"],
                "n_candidates": (r.get("n_candidates")
                                 if isinstance(r.get("n_candidates"), int)
                                 else (2 if r.get("preferred") else 0)),
                "abstained": r.get("abstained"),
                "loop_status": r["loop_status"],
                "assembly_preferred": r.get("preferred"),
                "disagreements": ([{}] * int(r.get("n_disagreements") or 0)),
                "stage_trace": [{"stage": "action_ranking", "present": True}],
                "candidate_means": (r.get("candidate_means") or {}),
                "conditioning_status": ("conditioned" if r.get("routed")
                                        else "passthrough"),
                "routed": r.get("routed"), "unknown": r.get("unknown") or []}

    a = [classify_record(record=adapt(r)) for r in recs]
    b = [classify_record(record=adapt(r)) for r in recs]
    assert stable_fingerprint(a) == stable_fingerprint(b)
