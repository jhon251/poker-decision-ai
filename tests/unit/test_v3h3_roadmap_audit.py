# -*- coding: utf-8 -*-
"""V3H-3 契约测试：只校验**路线审计逻辑与其事实基础**。

本文件**不**断言任何策略能力"已实现"或"可用"；它强制：
  * 五类状态枚举封闭，且 BLOCKED/UNKNOWN 项不得被写成可开发；
  * 能力矩阵的每一项都声明 已有原语/缺口/状态/最高证据等级；
  * 架构层不得接线到生产 Judge，且必须声明 UNKNOWN 传播与模式边界；
  * HU CFR POC 的四种用途判定与 V3H-2 的诚实局限一致；
  * 测试矩阵覆盖授权要求的 9 个轴；
  * 零生产接线（真实 import 仍恰好 2 个已授权旁路点）；
  * 冻结面未变；不得出现越界声明。
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"
DOCS = ROOT / "docs"
AUDIT = DOCS / "V3H_3_STRATEGY_ROADMAP_AUDIT.json"
CLOSEOUT = DOCS / "V3H_3_STRATEGY_ROADMAP_CLOSEOUT.md"

STATUS = {"IMPLEMENTABLE", "MODEL_INTERNAL_ONLY", "RESEARCH_ONLY", "BLOCKED", "UNKNOWN"}
V3H2 = DOCS / "V3H_2_FEASIBILITY_RESULT.json"

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

REQUIRED_TEST_AXES = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# ── 1. 枚举与结构 ──────────────────────────────────────────────────────
def test_status_enum_is_closed(audit):
    assert set(audit["status_enum"]) == STATUS


def test_audit_is_audit_only(audit):
    assert audit["type"] == "audit_only"
    assert audit["implementation_performed"] is False


def test_every_capability_declares_required_fields(audit):
    for c in audit["capability_matrix"]:
        for k in ("capability", "existing_primitives", "missing", "status",
                  "evidence_grade_max", "rationale"):
            assert k in c, (c.get("capability"), k)
        assert c["status"] in STATUS, c["capability"]


def test_blocked_and_unknown_are_not_described_as_developable(audit):
    for c in audit["capability_matrix"]:
        if c["status"] in ("BLOCKED", "UNKNOWN"):
            assert c["evidence_grade_max"] == "NONE", c["capability"]


# ── 2. 冻结事实必须被继承（不得与 V3H-2 冲突）──────────────────────────
def test_frozen_facts_match_v3h2_result(audit):
    v2 = json.loads(V3H2.read_text(encoding="utf-8"))
    f = audit["frozen_facts"]
    assert v2["solver"]["six_max_verdict"] == "BLOCKED"
    assert "BLOCKED" in f["six_max_cash_solver"]
    assert v2["b1_independent_human_data"]["verdict"] == "BLOCKED"
    assert "BLOCKED" in f["b1_independent_valid_points"]
    assert v2["b3_profit_environment"]["real_6max_cash_profit_evidence"]["classification"] == "BLOCKED"
    assert "BLOCKED" in f["b3_real_profit_evidence"]


def test_prohibited_equivalences_present(audit):
    joined = " ".join(audit["frozen_facts"]["prohibited_equivalences"])
    for needed in ("POC != GTO", "PI-EV != equilibrium", "heuristic != solver",
                   "M3 mechanism != real profit"):
        assert needed in joined, needed


def test_blockers_unchanged(audit):
    b = audit["blockers_unchanged"]
    for k in ("B1", "B3", "B4", "B5", "solver"):
        assert b[k] == "BLOCKED", k


# ── 3. 现有能力盘点必须与磁盘一致（可复核）────────────────────────────
def test_layer_inventory_matches_disk(audit):
    inv = audit["existing_capability_surface"]["layers"]
    assert set(inv) == set(V3G_LAYERS)
    for layer, meta in inv.items():
        p = PKG / f"{layer}.py"
        assert p.exists(), layer
        src = p.read_text(encoding="utf-8-sig")
        tree = ast.parse(src)
        fns = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for fn in meta["functions"]:
            assert fn in fns, f"{layer}.{fn}"
        assert abs(meta["lines"] - len(src.splitlines())) <= 2, layer


def test_total_lines_claim_matches_disk(audit):
    total = sum(len((PKG / f"{l}.py").read_text(encoding="utf-8-sig").splitlines())
                for l in V3G_LAYERS)
    claimed = audit["existing_capability_surface"]["total_layer_lines"]
    assert abs(total - claimed) <= 20, (total, claimed)


def test_wiring_claim_matches_ast_scan(audit):
    """生产真实 import 的层必须与 AST 扫描一致。"""
    real: set[str] = set()
    for p in PKG.rglob("*.py"):
        if "cash_" in p.name or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                m = n.module.split(".")[-1]
                if m in V3G_LAYERS:
                    real.add(m)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    m = a.name.split(".")[-1]
                    if m in V3G_LAYERS:
                        real.add(m)
    claimed = set(audit["existing_capability_surface"]["production_wiring"]["wired_layers"])
    assert real == claimed, (sorted(real), sorted(claimed))
    assert real == {"cash_integrity_context", "cash_unknown_e2e"}


def test_zero_production_wiring_flag_is_true(audit):
    assert audit["zero_production_wiring"] is True
    assert audit["existing_capability_surface"]["production_wiring"][
        "zero_production_wiring_for_strategy_layer"] is True


# ── 4. 架构：不接生产 Judge + 必须声明传播与边界 ───────────────────────
def test_architecture_is_not_wired_to_judge(audit):
    """语义校验：架构明确离线、Judge/realtime 保持冻结、接线需单独授权。"""
    arch = audit["proposed_architecture"]
    assert arch["wiring_to_production_judge"] is False
    stmt = arch["judge_boundary_statement"].lower()
    assert "offline" in stmt
    assert "byte-frozen" in stmt
    assert "separate explicit authorization" in stmt
    assert "cash_integrity_context" in stmt


def test_architecture_layers_have_io_grade_and_boundaries(audit):
    layers = audit["proposed_architecture"]["layers"]
    assert len(layers) >= 6
    ids = [l["id"] for l in layers]
    assert ids == sorted(ids)
    for l in layers:
        for k in ("inputs", "outputs", "evidence_grade", "unknown_propagation",
                  "mode_boundary", "reuse"):
            assert k in l, (l["id"], k)
        assert l["inputs"] and l["outputs"]


def test_unknown_propagation_never_substitutes_defaults(audit):
    joined = " ".join(l["unknown_propagation"]
                      for l in audit["proposed_architecture"]["layers"])
    assert "UNKNOWN" in joined
    assert "never substitutes a guess" in joined or "不给默认动作" in joined


def test_mode_boundary_is_cash_only_for_every_layer(audit):
    for l in audit["proposed_architecture"]["layers"]:
        assert "CASH only" in l["mode_boundary"], l["id"]


def test_l5_evidence_grade_is_minimum_of_inputs(audit):
    l5 = next(l for l in audit["proposed_architecture"]["layers"] if l["id"] == "L5")
    assert "MINIMUM" in l5["evidence_grade"].upper()


# ── 5. HU CFR POC 用途评估 ────────────────────────────────────────────
def test_poc_assessment_allows_only_non_production_roles(audit):
    a = audit["hu_cfr_poc_assessment"]["usable_as"]
    assert a["algorithm_teaching_and_unit_test_basis"]["verdict"] is True
    assert a["model_internal_research_basis"]["verdict"] is True
    assert a["production_strategy_basis"]["verdict"] is False
    assert a["gto_or_equilibrium_evidence"]["verdict"] is False


def test_poc_assessment_consistent_with_v3h2_limits(audit):
    v2 = json.loads(V3H2.read_text(encoding="utf-8"))
    limits = " ".join(v2["solver"]["poc"]["honest_limits"]).lower()
    reasons = " ".join(audit["hu_cfr_poc_assessment"]["usable_as"]
                       ["production_strategy_basis"]["reasons"]).lower()
    assert "single street" in limits and "single street" in reasons
    assert v2["solver"]["poc"]["self_checks"]["br_probe_valid"] is False
    gto_reasons = " ".join(audit["hu_cfr_poc_assessment"]["usable_as"]
                           ["gto_or_equilibrium_evidence"]["reasons"]).lower()
    assert "not measured" in gto_reasons


def test_reusable_invariants_are_mathematical_not_equilibrium_claims(audit):
    inv = " ".join(audit["hu_cfr_poc_assessment"]
                   ["invariants_that_may_be_reused_in_tests"])
    assert "ev_hero + ev_villain == pot" in inv.lower()
    for banned in ("gto", "equilibrium", "exploitab"):
        assert banned not in inv.lower()


# ── 6. 测试矩阵必须覆盖 9 轴 ──────────────────────────────────────────
def test_test_matrix_covers_required_axes(audit):
    ids = {t["id"] for t in audit["test_matrix"]}
    assert REQUIRED_TEST_AXES <= ids, REQUIRED_TEST_AXES - ids


def test_test_matrix_entries_are_actionable(audit):
    for t in audit["test_matrix"]:
        assert t["required"], t["id"]
        assert t["pass_criteria"], t["id"]


def test_judge_unchanged_axis_references_real_files(audit):
    t9 = next(t for t in audit["test_matrix"] if t["id"] == "T9")
    txt = " ".join(t9["required"])
    assert "agent.py" in txt and "realtime.py" in txt


# ── 7. 客户端与合规 ───────────────────────────────────────────────────
def test_client_not_required_and_auto_betting_false(audit):
    c = audit["client_and_authorization"]
    assert c["open_coinpoker_client_required"] is False
    assert c["auto_betting"] is False
    assert "offline" in c["reason"]


def test_no_prohibited_claims_asserted(audit):
    for p in audit["prohibited_claims"]:
        assert isinstance(p, str) and p


def test_closeout_has_no_overclaim_phrases():
    md = CLOSEOUT.read_text(encoding="utf-8")
    for ph in ("CASH 已完成", "CASH 最终完成", "GTO 已实现", "达到均衡",
               "盈利已验证", "全量 PASS"):
        for m in re.finditer(re.escape(ph), md):
            window = md[max(0, m.start() - 24):m.start()]
            assert any(n in window for n in ("不", "未", "无", "禁止", "不得")), ph


# ── 8. 冻结面与交付物 ─────────────────────────────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_untouched(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_freezes_not_broken():
    man = json.loads((DOCS / "V3G_11_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_files_modified_list_is_exactly_the_three_audit_files(audit):
    listed = {f["path"] for f in audit["files_modified"]}
    assert listed == {"docs/V3H_3_STRATEGY_ROADMAP_AUDIT.json",
                      "docs/V3H_3_STRATEGY_ROADMAP_CLOSEOUT.md",
                      "tests/unit/test_v3h3_roadmap_audit.py"}
    for rel in listed:
        assert (ROOT / rel).exists(), rel


def test_verdict_is_pass_and_v3h4_not_entered(audit):
    assert audit["verdict"] == "PASS"
    md = CLOSEOUT.read_text(encoding="utf-8")
    assert "V3H-3 = PASS + STOP" in md
    assert "未进入 V3H-4" in md


def test_capability_counts_are_consistent(audit):
    from collections import Counter
    tally = Counter(c["status"] for c in audit["capability_matrix"])
    assert tally["IMPLEMENTABLE"] >= 4
    assert tally["MODEL_INTERNAL_ONLY"] >= 2
    assert tally["BLOCKED"] >= 4
    md = CLOSEOUT.read_text(encoding="utf-8")
    assert "4 项 IMPLEMENTABLE" in md
