"""V3F-P45 — Strategy Validation Loop 测试。

覆盖：
  1  dataset manifest（版本化、来源诚实标注、覆盖面）
  2  策略回归记录完整性（C2 字段）
  3  机械指标全 PASS（一致性/排序/单调/UNKNOWN/provenance/合法性/确定性）
  4  确定性可复现（整套装件二次运行逐字节一致）
  5  对拍 reference（RULES ≠ GTO；MTT/PKO → unverified；不伪造）
  6  historical agreement（弱经验信号，显式标注）
  7  adversarial 15 类边界（C5）
  8  evaluation report 快照（C6：PASS/FAIL + verified/unverified + limitations）
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.domain.poker.strategy_eval import (
    adversarial_cases,
    constructed_cases,
    hero_decision_points,
    run_evaluation,
    uniform_range_equity,
)

ROOT = Path(__file__).resolve().parents[2]
HANDS_DIR = ROOT / "data" / "hand_history" / "synthetic_cash_100"
MANIFEST = ROOT / "data" / "evaluation" / "strategy_eval_manifest.json"


# ── 1. 固定 evaluation dataset（C1） ──────────────────────────────────
def test_manifest_versioned_and_honest():
    assert MANIFEST.exists(), "run build_manifest() to create the manifest"
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["version"] == "p45-v1"
    ph = m["sources"]["parsed_hands"]
    assert ph["count"] == 100
    # 诚实标注：合成数据不得冒充真实历史
    assert "不能用于策略有效性证明" in ph["provenance"]
    assert m["sources"]["constructed"]["provenance"].startswith(
        "synthetic_boundary")
    assert "未配置 → 不对拍、不伪造" in m["reference_policy"]["external_solver"]
    assert len(m["parsed_hands"]) == 100


def test_dataset_coverage_matrix():
    cov = {
        "cash": any(c["state"].get("game_type") == "cash"
                    for c in constructed_cases() + adversarial_cases()),
        "mtt": any(c["state"].get("game_type") == "mtt"
                   for c in constructed_cases()),
        "pko_ko_mystery": {c["mode_tag"] for c in constructed_cases()}
        >= {"pko", "ko", "mystery"},
        "heads_up": any(len(c["state"]["players"]) == 2
                        for c in constructed_cases()),
        "multiway": any(len(c["state"]["players"]) >= 3
                        for c in constructed_cases() + adversarial_cases()),
        "streets": {p["street"] for f in sorted(HANDS_DIR.glob("*.txt"))[:20]
                    for p in hero_decision_points(
                        parse_hand_history(
                            f.read_text(encoding="utf-8")).to_dict())}
        >= {"preflop", "flop"},
        "icm_present_missing": len({bool(
            (c["state"].get("tournament") or {}).get("payouts"))
            for c in constructed_cases() + adversarial_cases()
            if c["state"].get("game_type") == "mtt"}) == 2,
        "opponent_sample_sizes": any(
            c.get("opponent_src") for c in adversarial_cases()),
    }
    assert all(cov.values()), cov


# ── 2/3/4. 策略回归记录 + 机械指标 + 确定性（C2/C4） ──────────────────
def test_full_suite_pass_and_deterministic():
    rep1 = asyncio.run(run_evaluation(str(HANDS_DIR)))
    rep2 = asyncio.run(run_evaluation(str(HANDS_DIR)))
    assert rep1["overall"] == "PASS"
    for name, m in rep1["metrics"].items():
        assert m["pass"], f"{name}: {m}"
    assert rep1["_records"] == rep2["_records"]      # 逐记录一致
    assert rep1 == rep2                              # 报告整体一致
    rec = rep1["_records"][0]
    # C2 记录字段完整性（输入状态/候选/EV/选择/provenance/fallback）
    for field in ("street", "mode", "pot_bb", "to_call_bb", "equity",
                  "ev_mode", "objective", "candidates", "best_action",
                  "fallback_flags", "legality_violations",
                  "provenance_violations"):
        assert field in rec, field


def test_parsed_decision_points_replay():
    f = sorted(HANDS_DIR.glob("*.txt"))[0]
    from poker_decision_ai.domain.poker.hand_history import parse_hand_history
    parsed = parse_hand_history(f.read_text(encoding="utf-8")).to_dict()
    pts = hero_decision_points(parsed)
    assert pts, "hero decision points must exist"
    p0 = pts[0]
    st = p0["state"]
    assert st["players"][0]["is_hero"] and st["players"][0]["hole_cards"]
    assert st["pot"] > 0 and st["bb"] > 0 and st["to_call"] >= 0
    assert p0["actual_action"] in ("fold", "check", "call", "bet", "raise")
    # 快照在动作之前：fold 点的 to_call 与动作一致
    eq = uniform_range_equity(st["players"][0]["hole_cards"], p0["board"])
    assert 0.0 <= eq["hero_equity"] <= 1.0
    assert eq["equity_source"] == "uniform_range_mc"


# ── 5. 对拍 reference（C3） ───────────────────────────────────────────
def test_rules_reference_and_unverified():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    ra = rep["reference_agreement"]["rules_engine"]
    assert ra["verified_cases"] > 0
    assert 0.0 <= ra["agreement"] <= 1.0
    assert "RULES ≠ GTO" in ra["note"]               # 不把一致写成正确
    unv = [r for r in rep["_records"]
           if r.get("reference_status") == "unverified_no_reference"]
    assert unv, "MTT/PKO must be unverified (rules v1 cash-only)"
    mtt_unv = [r for r in unv
               if (r.get("reference") or {}).get("reason", "")
               .startswith("rules engine v1 is cash-only")]
    assert mtt_unv                                    # MTT/PKO → 域外 unverified
    assert all((r.get("reference") or {}).get("reason")
               for r in unv)                          # 全部带诚实原因
    vu = rep["verified_unverified"]
    assert vu["verified_rules"] + vu["unverified"] == rep["case_count"]


def test_historical_agreement_labeled_weak():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    h = rep["reference_agreement"]["historical"]
    assert h["cases"] > 0 and h["agreement"] is not None
    assert "弱经验信号" in h["note"] and "非正确性证明" in h["note"]
    assert "不构成策略正确性证明" in \
        rep["reference_agreement"]["realized_outcome"]["note"]


# ── 7. adversarial / boundary（C5） ───────────────────────────────────
def _rec(rep, cid):
    return next(r for r in rep["_records"] if r["case_id"] == cid)


def test_adversarial_extreme_equity():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    r0 = _rec(rep, "adv01_equity_zero")
    assert r0["available"] and r0["best_action"] == "fold"
    assert all(c["ev_bb"] <= 1e-9 for c in r0["candidates"]
               if c["action"] == "call")
    r1 = _rec(rep, "adv02_equity_one")
    assert r1["available"] and r1["best_action"] in ("call", "raise", "bet")
    assert _rec(rep, "adv01_equity_zero")["fallback_flags"]["deterministic"]


def test_adversarial_extreme_fold_probability():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    pf0 = _rec(rep, "adv03_pfold_zero")
    c0 = next(c for c in pf0["candidates"] if c["action"] == "bet")
    assert c0["p_fold"] is not None and c0["p_fold"] < 0.05   # 近 0（P42 融合）
    pf1 = _rec(rep, "adv04_pfold_one")
    c1 = next(c for c in pf1["candidates"] if c["action"] == "bet")
    assert c1["p_fold"] is not None and c1["p_fold"] > 0.95   # 近 1（P42 融合）


def test_adversarial_allin_and_extreme_pot():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    ai = _rec(rep, "adv06_allin_call")
    call = next(c for c in ai["candidates"] if c["action"] == "call")
    assert abs(call["amount_bb"] - 100.0) < 1e-6      # allin 封顶
    pe = _rec(rep, "adv05_pot_extremes")
    assert pe["available"] and not pe["legality_violations"]


def test_adversarial_missing_inputs_never_fabricate():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    for cid in ("adv08_icm_missing", "adv09_bounty_missing"):
        r = _rec(rep, cid)
        assert r["available"] is True                 # chip EV 仍可算
        if cid == "adv08_icm_missing":
            assert r["objective"] == "chip_ev"        # 不冒充 icm
            assert r["icm_available"] is False
        else:
            assert r["bounty_available"] is False     # 不冒充 bounty
    for cid in ("adv11_unknown_equity", "adv12_illegal_to_call",
                "adv15_objective_boundary"):
        r = _rec(rep, cid)
        assert r["available"] is False and r["unavailable_reason"]
        assert r["candidates"] == []                  # 无候选=无伪造动作
    tiny = _rec(rep, "adv13_tiny_samples")
    assert all(c["p_fold"] is None for c in tiny["candidates"])  # 5 样本拒绝
    insuf = _rec(rep, "adv10_opponent_insufficient")
    assert all(c["p_fold"] is None for c in insuf["candidates"])  # 3 样本拒绝


def test_adversarial_multiway_and_sizes():
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    mw = _rec(rep, "adv07_multiway")
    assert mw["available"] and not mw["legality_violations"]
    sg = _rec(rep, "adv14_size_grid")
    bets = [c for c in sg["candidates"] if c["action"] == "bet"]
    assert len(bets) >= 3 and len({c["amount_bb"] for c in bets}) == len(bets)


# ── 8. evaluation report 快照（C6） ──────────────────────────────────
def test_evaluation_snapshot_written(tmp_path):
    rep = asyncio.run(run_evaluation(str(HANDS_DIR)))
    rep.pop("_records", None)                         # 快照仅摘要（可 diff）
    out = ROOT / "docs" / "V3F_P45_EVALUATION_SNAPSHOT.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["overall"] == "PASS"
    assert saved["known_limitations"], "必须显式声明已知限制"
    assert any("不能用于策略有效性证明" in l
               for l in saved["known_limitations"])
    assert saved["dataset"]["sources"]["parsed"].startswith(
        "data/hand_history/synthetic_cash_100")
