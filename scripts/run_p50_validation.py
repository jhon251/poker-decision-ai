"""P50 snapshot 生成（只读真实数据；确定性；产物入 docs/）。

用法: python scripts/run_p50_validation.py [scan_limit]
产物: docs/V3F_P50_REAL_VALIDATION_SNAPSHOT.json
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.config import load_config          # noqa: E402
from poker_decision_ai.real_validation import (                # noqa: E402
    build_evidence,
    collect_real_sample,
    compare_scenario,
    diagnose_saturation,
    manifest,
    marginal_scenarios,
    pick_target_player,
    real_profile,
    three_arm_decision,
)

CHUNK0 = ROOT / "data" / "uoft" / "normalized" / "chunk_000.jsonl"
P49_SNAPSHOT = ROOT / "docs" / "V3F_P49_REAL_VALIDATION_SNAPSHOT.json"


async def main(scan_limit: int) -> dict:
    cfg = load_config()
    hands = collect_real_sample(str(CHUNK0), scan_limit=scan_limit)
    pid = pick_target_player(hands)
    player_hands = hands[pid]
    ev = build_evidence(player_hands, hands)
    prof = real_profile("villain", ev)

    marginal = [await three_arm_decision(cfg, s, "villain", prof)
                for s in marginal_scenarios()]
    p48_results = [await compare_scenario(cfg, s, "villain", prof, None)
                   for s in manifest()["scenarios"]
                   if not s["id"].startswith("marg_")]

    active = [m for m in marginal if m["arms"]["REAL"].get("best")]
    n_change = sum(1 for m in marginal
                   if m["action_change"]["baseline_vs_real"])

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 6) if vals else None

    def fold_of(m, arm):
        f = m["arms"][arm].get("fold_05x")
        return f.get("p_fold") if f else None

    summary = {
        "n_marginal": len(marginal),
        "action_change_rate_real_vs_baseline": round(
            n_change / max(1, len(marginal)), 4),
        "n_action_changes": n_change,
        "changed_scenarios": [m["id"] for m in marginal
                              if m["action_change"]["baseline_vs_real"]],
        "w_stat_real_median": med(
            [m["arms"]["REAL"]["fold_05x"].get("w_stat") for m in active
             if m["arms"]["REAL"]["fold_05x"]]),
        "equity_diff_median": med(
            [m["arms"]["REAL"]["equity"] - m["arms"]["BASELINE"]["equity"]
             for m in active]),
        "fold_diff_median": med(
            [fold_of(m, "REAL") - fold_of(m, "BASELINE") for m in active
             if fold_of(m, "REAL") is not None
             and fold_of(m, "BASELINE") is not None]),
        "counterfactual_fold_differs_from_baseline": sum(
            1 for m in active
            if fold_of(m, "COUNTERFACTUAL") is not None
            and fold_of(m, "BASELINE") is not None
            and fold_of(m, "COUNTERFACTUAL") != fold_of(m, "BASELINE")),
        "counterfactual_vs_real_fold_diff_median": med(
            [fold_of(m, "REAL") - fold_of(m, "COUNTERFACTUAL") for m in active
             if fold_of(m, "REAL") is not None
             and fold_of(m, "COUNTERFACTUAL") is not None]),
    }

    diag = diagnose_saturation(cfg, evidence=ev)

    # 修复前后对照（before = P49 snapshot 的边际汇总）
    before = None
    if P49_SNAPSHOT.exists():
        p49 = json.loads(P49_SNAPSHOT.read_text(encoding="utf-8"))
        before = p49.get("marginal_summary")

    snapshot = {
        "snapshot_version": "p50-real-v1",
        "manifest": manifest(),
        "data": {"chunk": "chunk_000", "scan_limit": scan_limit,
                 "target_player": pid,
                 "target_hands": len(player_hands)},
        "fixes": {
            "F1": ("fold_equity.for_size 融合权重读取兼容 nested "
                   "conditional.samples（P50-1）"),
            "F2": ("P46 continuing 校准 hero-agnostic：baseline continue 与 "
                   "tilt 评分改用 board 条件 + hero 死牌剔除的 vs-random "
                   "强度（P50-2）"),
        },
        "marginal_summary_after_fix": summary,
        "before_after": {"before_P49": before,
                         "after_P50": summary},
        "saturation_diagnosis_after_fix": diag,
        "p48_regression": {
            "n_scenarios": len(p48_results),
            "results": p48_results,
        },
        "scenarios_marginal": marginal,
        "honesty": {
            "pko": "NOT VALIDATABLE WITH CURRENT DATA（语料 cash-only）",
            "mtt": "MECHANISM_CHECK_ONLY（无真实锦标赛手牌）",
            "combo_ground_truth": "NOT VALIDATABLE（观察者视角无 hole cards）",
            "profitability": "本 snapshot 不构成任何盈利性/GTO 证明",
        },
    }
    out = ROOT / "docs" / "V3F_P50_REAL_VALIDATION_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({"target": pid, "hands": len(player_hands),
                      "summary": summary,
                      "before": before,
                      "attribution": diag["attribution"],
                      "hero_conditioning": diag["hero_conditioning"]["verdict"],
                      "w_stat_probe": diag["w_stat_probe"]["finding"][:70]},
                     ensure_ascii=False, indent=1))
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    asyncio.run(main(limit))
