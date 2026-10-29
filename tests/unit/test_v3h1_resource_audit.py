# -*- coding: utf-8 -*-
"""V3H-1 契约测试：只验证**审计逻辑**，不伪造任何外部资源可用性。

本文件**不**断言任何外部 Solver / 数据集 / 环境"可用"；恰恰相反，它强制：
  * 分类枚举封闭；
  * 每条候选必须具备 证据/能力/限制/授权/可得性 字段；
  * 被标为 INCOMPATIBLE/BLOCKED 的候选不得同时被标为 AVAILABLE；
  * HU-only 求解器不得被标为支持 6-max；
  * 本地实测数值必须与证据文件一致（可复算）；
  * 报告不得出现排名式表述或越界声明。
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
AUDIT_JSON = DOCS / "V3H_1_RESOURCE_AUDIT.json"
AUDIT_MD = DOCS / "V3H_1_EXTERNAL_VALIDATION_RESOURCE_AUDIT.md"

ENUM = {"AVAILABLE", "CANDIDATE", "UNKNOWN", "BLOCKED", "INCOMPATIBLE"}

#: 本阶段不得修改的冻结文件（前 16 位）
FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

#: V3H-1 只允许新增的文件（审计 + 测试）
ALLOWED_NEW = {
    "docs/V3H_1_EXTERNAL_VALIDATION_RESOURCE_AUDIT.md",
    "docs/V3H_1_RESOURCE_AUDIT.json",
    "tests/unit/test_v3h1_resource_audit.py",
}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def _all_candidates(audit: dict) -> list[dict]:
    out = []
    for section in ("solver", "human_data", "profit_environment"):
        out.extend(audit[section]["candidates"])
    return out


# ── 1. 枚举封闭 + schema ────────────────────────────────────────────────
def test_classification_enum_is_closed(audit):
    assert set(audit["classification_enum"]) == ENUM
    for c in _all_candidates(audit):
        assert c["classification"] in ENUM, c["id"]


@pytest.mark.parametrize("field", ["id", "name", "classification"])
def test_every_candidate_has_identity(audit, field):
    for c in _all_candidates(audit):
        assert c.get(field), c.get("id")


def test_every_candidate_declares_limits_or_evidence(audit):
    """每条候选必须有"限制/不适用"记录或明确证据来源。"""
    for c in _all_candidates(audit):
        has_limit = any(k for k in c if k in (
            "incompatibility_reason", "reason", "secondary_reason", "cost",
            "limits", "not_available_for", "six_max_note", "conclusion_scope",
            "hole_card_evidence", "fields_verifiable"))
        # 证据来源 = 可核实的外部出处，或明确声明"本阶段未评估/未获取"
        has_evidence = bool(c.get("evidence")) or c["classification"] in (
            "BLOCKED", "INCOMPATIBLE") or "acquisition" in c
        assert has_limit or has_evidence, f"{c['id']} 缺限制/证据说明"


def test_no_candidate_is_both_available_and_incompatible(audit):
    for c in _all_candidates(audit):
        assert c["classification"] != "AVAILABLE" or \
            c.get("secondary_classification") not in ("INCOMPATIBLE", "BLOCKED"), c["id"]


def test_blocked_or_incompatible_never_declared_usable(audit):
    """BLOCKED/INCOMPATIBLE 的候选不得同时宣称可用于目标用途。"""
    for c in _all_candidates(audit):
        if c["classification"] in ("BLOCKED", "INCOMPATIBLE"):
            assert "available_for" not in c, c["id"]


# ── 2. 求解器：HU ≠ 6-max；不得伪造 Solver ─────────────────────────────
def test_no_solver_candidate_claims_six_max_support(audit):
    for c in audit["solver"]["candidates"]:
        assert c.get("six_max") is not True, c["id"]


def test_hu_only_solvers_are_marked_incompatible(audit):
    hu_only = [c for c in audit["solver"]["candidates"]
               if c.get("players") in ("HU only", "HU")]
    assert hu_only, "预期存在 HU-only 候选"
    for c in hu_only:
        assert c["classification"] in ("INCOMPATIBLE", "CANDIDATE"), c["id"]


def test_local_solver_discovery_is_unavailable(audit):
    d = audit["solver"]["local_discovery"]
    assert d["available"] is False
    assert all(ch.get("found") is False for ch in d["channels"].values())


def test_solver_terminology_separates_four_concepts(audit):
    t = audit["solver"]["terminology"]
    assert set(t) == {"solver", "pi_ev", "heuristic", "range_equity"}
    assert "NOT an equilibrium solver" in t["pi_ev"]
    assert "never GTO" in t["heuristic"]


def test_postflop_solver_six_max_claim_is_about_preflop_folds(audit):
    """README 的 '6-max' 必须被如实解释为"翻前最多 4 人已弃牌"，非多路求解。"""
    s1 = next(c for c in audit["solver"]["candidates"] if c["id"] == "S1")
    assert "folded" in s1["six_max_note"]
    assert "NOT multiway" in s1["six_max_note"]
    assert s1["six_max"] is False


# ── 3. 人类样本：B1 条件必须如实为未满足 ───────────────────────────────
def test_b1_valid_points_condition_is_not_met(audit):
    b1 = audit["human_data"]["b1_conditions"]
    assert b1["valid_ge_1200"]["met"] is False
    assert b1["independence_provable"]["met"] is False


def test_no_real_hole_cards_in_local_corpus(audit):
    m = audit["human_data"]["decision_point_measurement"]
    assert m["decision_points_with_real_hole_cards"] == 0
    assert m["decision_points_state_reconstructable"] > 0


def test_fresh_pool_measured_but_not_sufficient(audit):
    c = audit["human_data"]["local_corpus_measured"]
    assert c["unique_unused_player_ids"] > 1000
    assert c["rows_containing_used_players"] == c["pluribus_rows"]
    # 数量满足不等于可用：必须同时记录"英雄手牌缺失"
    assert audit["human_data"]["b1_verdict"] == "BLOCKED"


def test_used_player_union_is_exactly_thirteen(audit):
    used = audit["human_data"]["used_players_union"]
    assert len(used) == 13 == audit["human_data"]["used_players_count"]
    assert len(set(used)) == 13


def test_frozen_thresholds_match_frozen_module_constants(audit):
    """审计引用的阈值必须与冻结模块中的真实常量一致。"""
    src = (ROOT / "poker_decision_ai/research_unfreeze/s3_validation.py"
           ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    ft = audit["human_data"]["frozen_thresholds"]
    for k in ("VALID_POINTS_MIN", "COVERAGE_MIN", "S3_HAND_MIN_INDEX"):
        assert ft[k] == consts[k], k


@pytest.mark.parametrize("key", ["VALID_POINTS_MIN", "COVERAGE_MIN"])
def test_thresholds_are_positive_numbers(audit, key):
    v = audit["human_data"]["frozen_thresholds"][key]
    assert isinstance(v, (int, float)) and v > 0


# ── 4. 收益环境：区分"机制验证"与"真实盈利" ────────────────────────────
def test_no_environment_can_prove_real_cash_profit(audit):
    d = audit["profit_environment"]["distinction"]
    assert d["can_prove_real_6max_cash_profit"] == []
    assert set(d["can_validate_m3_infrastructure"]) <= set(
        c["id"] for c in audit["profit_environment"]["candidates"])


def test_available_envs_restrict_conclusion_scope(audit):
    """AVAILABLE 的环境必须写明结论范围，且不得声称真实盈利。"""
    for c in audit["profit_environment"]["candidates"]:
        if c["classification"] == "AVAILABLE":
            assert c.get("available_for") or c.get("conclusion_scope"), c["id"]
            assert "real 6-max CASH profitability" not in c.get("available_for", "")


# ── 5. Blocker 状态 ───────────────────────────────────────────────────
def test_required_blockers_present_and_blocked(audit):
    b = audit["blockers"]
    for k in ("B1", "B3", "B4", "B5", "solver"):
        assert b[k]["status"] == "BLOCKED", k


def test_b1_and_solver_blockers_have_refined_reason(audit):
    b = audit["blockers"]
    assert b["B1"]["changed_reason"] is True
    assert b["solver"]["changed_reason"] is True
    assert "hole cards" in b["B1"]["reason"]
    assert "6-max" in b["solver"]["reason"]


# ── 6. 冻结面与合规声明 ───────────────────────────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_untouched(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_audit_frozen_baseline_matches_reality(audit):
    for rel, prefix in audit["frozen_baseline"].items():
        assert _sha(rel).startswith(prefix), rel


def test_compliance_flags_all_false_or_true_as_declared(audit):
    c = audit["compliance"]
    assert c["read_only_audit"] is True
    for k in ("frozen_code_modified", "v3g_artifacts_modified", "solver_executed",
              "profit_experiment_executed", "fake_external_resource",
              "license_terms_bypassed", "auto_registered_or_purchased",
              "presence_treated_as_availability", "cash_completion_claimed",
              "entered_v3h2"):
        assert c[k] is False, k


def test_v3g_freeze_drift_is_at_most_the_registered_self_reference(audit):
    """V3G 封存清单不得被本阶段破坏；唯一允许的漂移是 V3G-11 已登记的自引用项。"""
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
    assert audit["compliance"]["v3g_artifacts_modified"] is False


def test_no_ranking_requested_or_performed(audit):
    """不得出现排名式表述；但"不做排名"的**免责声明行**本身不算违规。"""
    assert audit["no_ranking"] is True
    md = AUDIT_MD.read_text(encoding="utf-8")
    for lineno, line in enumerate(md.splitlines(), 1):
        if any(neg in line for neg in ("不得", "禁止", "不做", "未做", "不给",
                                       "无排名", "不含", "not ranking")):
            continue                     # 免责声明：明确表示不做排名
        for ph in ("最好", "最差", "推荐首选", "best solver", "worst solver"):
            assert ph not in line, f"{lineno}: 排名式表述 {ph!r}: {line.strip()[:70]}"


def test_verdict_is_pass_and_v3h2_not_entered(audit):
    assert audit["verdict"] == "PASS"
    assert audit["compliance"]["entered_v3h2"] is False


def test_declared_contract_test_count_matches_this_file(audit):
    """JSON 里登记的测试数量必须与本文件实际数量一致（防口径漂移）。"""
    declared = audit["contract_tests"]["count"]
    src = (ROOT / "tests/unit/test_v3h1_resource_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    n = sum(1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_"))
    # 参数化用例数与函数数不同：JSON 登记的是**用例数**，此处只校验函数数下界
    assert declared >= n, (declared, n)
    assert audit["contract_tests"]["result"] == "all passed"


# ── 7. 报告纪律（不得越界宣称） ────────────────────────────────────────
def test_report_does_not_overclaim_cash_completion():
    md = AUDIT_MD.read_text(encoding="utf-8")
    for ph in ("CASH 已完成", "CASH 最终完成", "GTO 已实现", "达到均衡",
               "盈利已验证", "全量 PASS"):
        for m in re.finditer(re.escape(ph), md):
            window = md[max(0, m.start() - 24):m.start()]
            assert any(n in window for n in ("不", "未", "无", "禁止", "不得")), ph


def test_report_states_no_solver_executed_and_no_profit_experiment():
    md = AUDIT_MD.read_text(encoding="utf-8")
    assert "未执行" in md
    assert "收益实验" in md


def test_audit_files_are_the_only_new_artifacts():
    """本阶段只允许新增 3 个文件（其余文件不得声称由 V3H-1 修改）。"""
    assert len(ALLOWED_NEW) == 3
    for rel in ALLOWED_NEW:
        assert (ROOT / rel).exists(), rel


def test_markdown_and_json_numbers_agree(audit):
    """报告与 JSON 的关键实测数字必须一致（防止两处口径漂移）。"""
    md = AUDIT_MD.read_text(encoding="utf-8")
    c = audit["human_data"]["local_corpus_measured"]
    for value in (c["total_rows"], c["unique_player_ids"],
                  c["unique_unused_player_ids"]):
        assert f"{value:,}" in md or str(value) in md, value
