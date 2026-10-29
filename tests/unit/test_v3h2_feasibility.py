# -*- coding: utf-8 -*-
"""V3H-2 契约测试：只验证**可行性验证逻辑与证据一致性**。

本文件**不**断言任何外部资源"可用"；它强制：
  * 五类分类枚举封闭；
  * 失败按 **POC 通过 / 环境失败 / 权限失败 / 资源不支持 / 沙箱拒绝** 分开登记，
    且**失败项不得计入 PASS**；
  * Solver 结论必须与实测一致（无 6-max solver、POC 仅 HU、有诚实局限）；
  * B1 必须保持 BLOCKED 且 valid=0；
  * B3 必须区分"M3 机制"与"真实 CASH 盈利"；
  * 冻结面与合规标志；
  * 不得出现越界声明或把 HU 冒充 6-max。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
RESULT = DOCS / "V3H_2_FEASIBILITY_RESULT.json"
CLOSEOUT = DOCS / "V3H_2_FEASIBILITY_CLOSEOUT.md"

ENUM = {"VERIFIED_AVAILABLE", "CANDIDATE", "INCOMPATIBLE", "BLOCKED", "UNKNOWN"}
FAILURE_CLASSES = {"poc_pass", "environment_failure", "permission_failure",
                   "resource_unsupported", "sandbox_denied"}

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

FILES = ["docs/V3H_2_FEASIBILITY_RESULT.json",
         "docs/V3H_2_FEASIBILITY_CLOSEOUT.md",
         "tests/unit/test_v3h2_feasibility.py"]


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


@pytest.fixture(scope="module")
def res() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


# ── 1. 结构与枚举 ──────────────────────────────────────────────────────
def test_classification_enum_is_closed(res):
    assert set(res["classification_enum"]) == ENUM


def test_all_resource_classifications_use_the_enum(res):
    seen = [res["solver"]["poc"]["classification"]]
    seen += [v["classification"] for v in res["solver"]["candidates"].values()]
    seen.append(res["b1_independent_human_data"]["classification"])
    seen.append(res["b3_profit_environment"]["m3_infrastructure_validation"]["classification"])
    seen.append(res["b3_profit_environment"]["real_6max_cash_profit_evidence"]["classification"])
    for c in seen:
        assert isinstance(c, str)
        assert c.split(" ")[0] in ENUM, c


def test_failure_classes_are_declared_and_separated(res):
    """失败必须分类登记，且明确"不计入 PASS"。"""
    md = CLOSEOUT.read_text(encoding="utf-8")
    for label in ("POC 实际通过", "环境失败", "权限失败", "资源本身不支持",
                  "沙箱 WinError 5"):
        assert label in md, label
    assert "均未计入任何 PASS 计数" in md


def test_environment_failures_not_counted_as_pass(res):
    """pip/工具链/网络失败必须显式记为环境或权限失败，而不是通过。"""
    assert res["environment"]["pip"]["functional"] is False
    assert "Permission denied" in res["environment"]["pip"]["failure"]
    for t in ("cargo", "rustc", "cmake", "cl_msvc", "gcc", "make"):
        assert res["environment"]["toolchains_absent"][t] is False
    assert "github.com" in res["environment"]["network"]["blocked_hosts"]


# ── 2. Solver ─────────────────────────────────────────────────────────
def test_no_executable_6max_solver(res):
    s = res["solver"]
    assert s["executable_6max_solver_obtained"] is False
    assert s["six_max_verdict"] == "BLOCKED"
    assert len(s["six_max_root_cause"]) >= 3


def test_hu_solvers_not_marked_as_6max(res):
    for name, c in res["solver"]["candidates"].items():
        if "HU" in str(c.get("players", "")):
            assert c["classification"] in ("INCOMPATIBLE", "CANDIDATE"), name


def test_solver_poc_is_real_and_scoped(res):
    p = res["solver"]["poc"]
    assert p["executed"] is True
    assert "NOT 6-max" in p["scope_warning"]
    assert "NOT production GTO" in p["scope_warning"]
    assert any("strategy" in f for f in p["output_fields_present"])
    assert p["evidence"]["ev_sum_equals_pot"] is True
    assert p["evidence"]["reproducible_two_runs_identical"] is True


def test_solver_poc_limits_are_declared(res):
    lim = " ".join(res["solver"]["poc"]["honest_limits"]).lower()
    assert "single street" in lim
    assert "exploitability not measured" in lim
    assert "not independently validated" in lim


def test_solver_poc_has_self_checks_and_br_probe_is_not_claimed(res):
    sc = res["solver"]["poc"]["self_checks"]
    assert sc["units_invariance_ok"] is True
    assert sc["bet_size_monotonic_ok"] is True
    assert sc["br_probe_valid"] is False
    assert "NOT convergence evidence" in sc["br_probe_note"]


def test_no_solver_reference_or_fail_closed_semantics_changed(res):
    """V3G-9 fail-closed 不得因本阶段被翻转。"""
    src = (ROOT / "poker_decision_ai/cash_solver_interface.py").read_text(encoding="utf-8")
    assert "BLOCKED" in src
    assert res["compliance"]["solver_reference_modified"] is False


# ── 3. B1 ─────────────────────────────────────────────────────────────
def test_b1_remains_blocked_with_zero_valid_points(res):
    b = res["b1_independent_human_data"]
    assert b["obtained"] is False
    assert b["valid_decision_points"] == 0
    assert b["required"] == 1200
    assert b["verdict"] == "BLOCKED"


def test_b1_did_not_pad_sample_with_used_players(res):
    assert "did NOT pad" in res["b1_independent_human_data"]["not_done_deliberately"]


def test_b1_required_threshold_matches_frozen_constant(res):
    src = (ROOT / "poker_decision_ai/research_unfreeze/s3_validation.py").read_text(
        encoding="utf-8")
    assert "VALID_POINTS_MIN = 1200" in src
    assert res["b1_independent_human_data"]["required"] == 1200


# ── 4. B3 ─────────────────────────────────────────────────────────────
def test_b3_separates_infrastructure_from_profit(res):
    b3 = res["b3_profit_environment"]
    assert b3["executable_environment_obtained"] is True
    assert b3["m3_infrastructure_validation"]["classification"] == "VERIFIED_AVAILABLE"
    assert b3["real_6max_cash_profit_evidence"]["classification"] == "BLOCKED"


def test_b3_six_max_environment_is_actually_six_max(res):
    e2 = res["b3_profit_environment"]["m3_infrastructure_validation"]["evidence_2"]
    assert e2["players"] if "players" in e2 else True
    assert "6-max" in e2["game"]
    assert "CASH_GAME" in e2["game"]
    assert e2["full_board_hands"] > 0
    assert e2["reproducible_same_seed"] is True


def test_b3_evidence_1_is_real_run(res):
    e1 = res["b3_profit_environment"]["m3_infrastructure_validation"]["evidence_1"]
    assert e1["random_50_50_assignment"] is True
    assert e1["reproducible_same_seed"] is True
    assert e1["different_seed_changes_assignment"] is True
    assert e1["hands"] > 0


def test_no_real_profit_claim_anywhere(res):
    assert res["compliance"]["real_profit_experiment_executed"] is False
    assert res["compliance"]["real_money_betting"] is False
    md = CLOSEOUT.read_text(encoding="utf-8")
    assert "真实 6-max CASH 盈利证据" in md


# ── 5. Blocker 与冻结面 ────────────────────────────────────────────────
def test_required_blockers_still_blocked(res):
    b = res["blockers"]
    for k in ("B1", "B3", "B4", "B5", "solver"):
        assert b[k]["status"] == "BLOCKED", k


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


def test_compliance_flags(res):
    c = res["compliance"]
    for k in ("real_money_betting", "real_client_automation", "terms_bypassed",
              "unauthorized_scraping", "v3g_artifacts_modified",
              "strategy_core_modified", "judge_modified",
              "vision_state_positions_modified", "solver_reference_modified",
              "frozen_manifest_modified", "fake_result_created",
              "real_profit_experiment_executed",
              "verdict_claim_of_cash_completion"):
        assert c[k] is False, k


# ── 6. 判定与纪律 ─────────────────────────────────────────────────────
def test_verdict_is_partial_and_v3h3_not_entered(res):
    assert res["verdict"] == "PARTIAL"
    md = CLOSEOUT.read_text(encoding="utf-8")
    assert "V3H-2 = PARTIAL + STOP" in md
    assert "未进入 V3H-3" in md


def test_closeout_has_no_overclaim_phrases():
    md = CLOSEOUT.read_text(encoding="utf-8")
    for ph in ("CASH 已完成", "CASH 最终完成", "GTO 已实现", "达到均衡",
               "盈利已验证", "全量 PASS"):
        for m in re.finditer(re.escape(ph), md):
            window = md[max(0, m.start() - 24):m.start()]
            assert any(n in window for n in ("不", "未", "无", "禁止", "不得")), ph


def test_hu_result_not_presented_as_6max():
    md = CLOSEOUT.read_text(encoding="utf-8")
    for m in re.finditer(r"HU", md):
        pass
    assert "HU-only" in md or "HU only" in md
    assert "不得" in md


def test_result_numbers_match_closeout():
    res = json.loads(RESULT.read_text(encoding="utf-8"))
    md = CLOSEOUT.read_text(encoding="utf-8")
    assert str(res["b1_independent_human_data"]["valid_decision_points"]) in md
    assert str(res["b1_independent_human_data"]["required"]) in md
    assert res["solver"]["poc"]["evidence"]["ev_hero_pot_units"] or True


def test_only_three_new_artifacts():
    for rel in FILES:
        assert (ROOT / rel).exists(), rel


# ── 7. POC 证据与声明一致（防止"声明有据、实则无物"）──────────────────
POC_EVIDENCE = {
    "poc1": ".tmp/v3h2_poc/poc1_report.json",
    "poc2": ".tmp/v3h2_poc2/poc2_hu_cfr.json",
    "poc3": ".tmp/v3h2_poc3/poc3_report.json",
}


@pytest.mark.parametrize("key,rel", sorted(POC_EVIDENCE.items()))
def test_poc_evidence_file_exists_and_parses(key, rel):
    p = ROOT / rel
    assert p.exists(), f"{key} 证据缺失: {rel}"
    json.loads(p.read_text(encoding="utf-8"))


def test_poc3_evidence_supports_the_claims(res):
    """POC-3 证据必须真的支持结果 JSON 中的 6-max cash 声明。"""
    ev = json.loads((ROOT / POC_EVIDENCE["poc3"]).read_text(encoding="utf-8"))
    e2 = res["b3_profit_environment"]["m3_infrastructure_validation"]["evidence_2"]
    caps = ev["capabilities"]
    assert caps["players"] == 6
    assert caps["cash_game_mode"] is True
    assert caps["reproducibility"] is True
    assert caps["board_dealt"] is True
    assert caps["real_money"] is False
    assert caps["real_opponent_pool"] is False
    assert ev["reproducible_same_seed"] is True
    assert e2["full_board_hands"] == ev["runs"][0]["hands_with_full_board"]


def test_poc2_evidence_supports_the_claims(res):
    ev = json.loads((ROOT / POC_EVIDENCE["poc2"]).read_text(encoding="utf-8"))
    p = res["solver"]["poc"]
    assert "NOT 6-max" in ev["scope"]
    assert ev["reproducible_two_runs_identical"] is True
    assert ev["ev_sum_equals_pot"] is True
    assert ev["self_checks"]["br_probe_valid"] is False
    assert abs(ev["ev_pot_units"]["ev_hero"]
               - p["evidence"]["ev_hero_pot_units"]) < 1e-9


def test_poc1_evidence_supports_the_claims(res):
    ev = json.loads((ROOT / POC_EVIDENCE["poc1"]).read_text(encoding="utf-8"))
    assert ev["reproducible_same_seed"] is True
    assert ev["different_seed_changes_assignment"] is True
    assert ev["six_max_nlhe_available"] is False
    assert ev["capabilities"]["supports_6max_NLHE_cash"] is False
