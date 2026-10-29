"""R1 — 策略质量验证基线测试。

覆盖：冻结 manifest（数据身份/切分/种子/限制/指标定义）、冻结哈希（策略数学
零修改）、证据等级纪律（synthetic 不得标 REAL；MTT/PKO NOT_VALIDATABLE）、
泄漏审计（证据/评估不相交 + AI 座位隔离）、三模式质量矩阵、已知天花板、
CASH 小样本重放确定性、快照结构与诚信条款。

原则：测试只验证「基线是否诚实且可复现」，不验证「指标是否好看」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.quality_benchmark import (  # noqa: E402
    BENCHMARK_VERSION,
    CONFIDENCE_LEVELS,
    EVIDENCE_GRADES,
    FROZEN_METRIC_DEFINITIONS,
    frozen_hashes,
    known_ceilings,
    leakage_audit,
    mode_matrix,
    reference_metrics,
    run_cash_benchmark,
    verify_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json"


def _hands(limit: int) -> list[dict]:
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    out = []
    for row in iter_pluribus_rows(limit=limit):
        p = parse_hand_row(row)
        if p:
            out.append(p)
    return out


# ── 冻结 manifest ──────────────────────────────────────────────────────
def test_manifest_frozen_and_complete():
    from poker_decision_ai.quality_benchmark import build_manifest
    m = build_manifest()
    assert m["benchmark_version"] == BENCHMARK_VERSION == "r1-benchmark-v1"
    ds = m["datasets"]
    cash = ds["cash_pluribus_human_seats"]
    assert cash["parquet_files"] == 37 and cash["license"] == "cc-by-4.0"
    assert ds["cash_uoft_normalized"]["hand_count"] == 2740778
    # MTT/PKO 必须显式为数据缺失
    assert ds["mtt_corpus"]["evidence_grade"] == "NOT_VALIDATABLE"
    assert ds["pko_corpus"]["evidence_grade"] == "NOT_VALIDATABLE"
    assert "无真实锦标赛语料" in ds["mtt_corpus"]["reason"]
    # 切分/种子/限制冻结
    assert m["split"]["evidence_frac"] == 0.7
    assert "字典序" in m["split"]["rule"]
    assert m["seeds_and_limits"]["equity_iterations"] == 1500
    assert m["seeds_and_limits"]["decision_hand_limit"] == 400
    assert m["seeds_and_limits"]["uoft_corpus_sampling_seed"] == 42
    # 指标定义引用冻结实现且禁止重定义
    for name, d in m["metric_definitions"].items():
        assert d["must_not_redefine"] is True, name
        assert d["source"].startswith("poker_decision_ai/"), name
    assert set(FROZEN_METRIC_DEFINITIONS) >= {
        "fold_brier", "fold_agreement", "observed_action_agreement",
        "action_change_rate", "regret_proxy", "legality"}


def test_metric_discipline_declared():
    from poker_decision_ai.quality_benchmark import build_manifest
    md = build_manifest()["metric_discipline"]
    assert "不构成强度证明" in md["no_single_metric_strength_claim"]
    assert "不宣称任何盈利能力" in md["profitability"]
    assert "不得作为调参目标" in md["no_tuning"]


def test_no_synthetic_marked_real():
    from poker_decision_ai.quality_benchmark import build_manifest
    ds = build_manifest()["datasets"]
    syn = ds["synthetic_fixtures"]
    assert syn["evidence_grade"] == "SYNTHETIC"
    assert "不得" in syn["usage"]
    # REAL_OBSERVED 仅出现在真实语料上
    for key, d in ds.items():
        if d.get("evidence_grade") == "REAL_OBSERVED":
            assert "cash" in key, key
    # 证据等级词汇表唯一
    assert set(EVIDENCE_GRADES) == {
        "REAL_OBSERVED", "HISTORICAL_REPLAY", "COUNTERFACTUAL", "SYNTHETIC",
        "MECHANISM_ONLY", "NOT_VALIDATABLE"}
    assert set(CONFIDENCE_LEVELS) == {"HIGH", "MEDIUM", "LOW", "NONE"}


# ── 冻结哈希（策略数学零修改） ──────────────────────────────────────────
def test_strategy_math_frozen():
    h = frozen_hashes()
    assert len(h) == 11 and "MISSING" not in h.values()
    assert verify_frozen(h)["match"] is True
    # 篡改检测能力自证
    bad = dict(h)
    bad["poker_decision_ai/domain/poker/strategy_core.py"] = "0" * 64
    res = verify_frozen(bad)
    assert res["match"] is False
    assert res["mismatches"] == [
        "poker_decision_ai/domain/poker/strategy_core.py"]


# ── 泄漏审计 ───────────────────────────────────────────────────────────
def test_leakage_audit_disjoint_and_ai_excluded():
    la = leakage_audit(_hands(300))
    assert la["leakage_free"] is True and la["hand_id_overlap"] == []
    assert la["evidence_hands"] + la["evaluation_hands"] == la["n_hands"]
    assert la["ai_seat"] == "Pluribus"
    assert la["hands_containing_ai_seat"] > 0        # AI 座位确实存在
    assert "solver_label" in la["ai_seat_policy"]
    assert any("REAL_OBSERVED" in g for g in la["ground_truth_types"])


# ── 三模式质量矩阵 ─────────────────────────────────────────────────────
def test_mode_matrix_validatability():
    m = mode_matrix()
    assert m["cash"]["validatable_now"] is True
    assert m["cash"]["confidence"] == "HIGH"
    assert "fold_brier" in m["cash"]["metrics_available"]
    for mode in ("mtt", "pko"):
        cell = m[mode]
        assert cell["validatable_now"] is False
        assert cell["evidence_grade"] == "NOT_VALIDATABLE"
        assert cell["confidence"] == "NONE"
        assert cell["metrics_available"] == []       # 不得给出质量指标
        assert cell["mechanism_only_checks"]         # 只允许机制检查
        assert "不得" in cell["note"]
    # 契约层：MTT/PKO 契约本身允许（缺口是数据不是契约）
    assert m["mtt"]["contract"]["decision_grade_validation"] is True
    assert m["pko"]["contract"]["objective"] == "chip_ev+icm+bounty"


def test_known_ceilings_recorded():
    ids = {c["id"] for c in known_ceilings()}
    assert "A1_single_street_call_ceiling" in ids
    assert "A1_single_street_bet_bias" in ids
    assert "cash_only_corpus" in ids
    call = next(c for c in known_ceilings()
                if c["id"] == "A1_single_street_call_ceiling")
    assert "0.7%" in call["statement"]
    assert "P55" in call["evidence"]


# ── CASH 小样本重放：确定性 + 冻结口径 ──────────────────────────────────
def test_cash_benchmark_deterministic_small():
    hs = _hands(400)
    a = run_cash_benchmark(decision_hand_limit=5, equity_iterations=120,
                           hands=hs)
    b = run_cash_benchmark(decision_hand_limit=5, equity_iterations=120,
                           hands=hs)
    assert a == b                                    # 逐字节确定性
    assert a["leakage_audit"]["leakage_free"] is True
    assert a["legality"]["illegal"] == 0
    assert a["split"]["evidence_hands"] + a["split"]["evaluation_hands"] \
        == len(hs)
    fc = a["fold_calibration"]["arms"]
    assert {"BASELINE", "REAL"} <= set(fc)
    assert 0.0 <= fc["REAL"]["brier"] <= 1.0
    de = a["decision_ev"]
    assert de["n_decision_points"] >= 1
    assert de["regret_proxy_median"] is not None
    # 证据等级标注齐全
    assert a["evidence_grades"]["fold_calibration"].startswith("REAL_OBSERVED")


def test_reference_metrics_readable():
    refs = reference_metrics()
    assert refs["P54_decision_grade"]["present"] is True
    assert refs["P56_continuum_v31"]["present"] is True
    assert refs["P54_decision_grade"]["fold_calibration"]["arms"]["REAL"][
        "brier"] == 0.125886


# ── 快照（结构 + 诚信；不重写） ─────────────────────────────────────────
def test_snapshot_present_and_honest():
    assert SNAPSHOT.exists(), "run build_snapshot() to create R1 baseline"
    s = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert s["snapshot_version"] == BENCHMARK_VERSION
    for key in ("manifest", "mode_matrix", "cash", "mtt_pko", "known_ceilings",
                "p69_baseline_comparison", "layering", "honesty"):
        assert key in s, key
    h = s["honesty"]
    assert "不构成任何盈利性/GTO 证明" in h["no_profitability_claim"]
    assert "强度证明" in h["no_single_metric_strength"]
    assert h["mtt_pko"].startswith("NOT VALIDATABLE")
    assert s["layering"]["real_world_profitability_validated"] is False
    assert s["layering"]["strategy_quality_validated"].startswith("部分")
    # MTT/PKO 在快照中不得出现质量指标
    assert s["mode_matrix"]["mtt"]["metrics_available"] == []
    assert s["mtt_pko"]["evidence_grade"] == "MECHANISM_ONLY"
    assert "不构成任何质量结论" in s["mtt_pko"]["statement"]
    # CASH 必须带真实样本规模与泄漏审计
    assert s["cash"]["split"]["leakage_free"] is True
    assert s["cash"]["labels"]["observed_human_action"] > 10000
    # 与 P69 baseline 对比存在
    assert s["p69_baseline_comparison"]["comparisons"]
