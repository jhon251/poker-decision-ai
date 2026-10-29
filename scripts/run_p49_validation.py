"""P49 snapshot 生成（只读真实数据；确定性；产物入 docs/）。

用法: python scripts/run_p49_validation.py [scan_limit]
产物: docs/V3F_P49_REAL_VALIDATION_SNAPSHOT.json
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


async def main(scan_limit: int) -> dict:
    cfg = load_config()
    hands = collect_real_sample(str(CHUNK0), scan_limit=scan_limit)
    pid = pick_target_player(hands)
    player_hands = hands[pid]
    ev = build_evidence(player_hands, hands)
    prof = real_profile("villain", ev)

    # P48 场景回归（17 个，结果应与 P48 snapshot 逐位一致）
    p48_results = []
    for scen in manifest()["scenarios"]:
        if not scen["id"].startswith("marg_"):
            p48_results.append(await compare_scenario(cfg, scen, "villain",
                                                      prof, None))

    # P49 边际场景三臂对照（27 个）
    marginal = []
    for scen in marginal_scenarios():
        marginal.append(await three_arm_decision(cfg, scen, "villain", prof))

    active = [m for m in marginal if m["arms"]["REAL"].get("best")]
    n_change = sum(1 for m in marginal if m["action_change"]["baseline_vs_real"])

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 6) if vals else None

    summary = {
        "n_marginal": len(marginal),
        "action_change_rate_real_vs_baseline": round(n_change / max(1, len(marginal)), 4),
        "n_action_changes": n_change,
        "changed_scenarios": [m["id"] for m in marginal
                              if m["action_change"]["baseline_vs_real"]],
        "equity_diff_real_vs_base_median": med(
            [m["arms"]["REAL"]["equity"] - m["arms"]["BASELINE"]["equity"]
             for m in active]),
        "fold_diff_real_vs_base_median": med(
            [m["arms"]["REAL"]["fold_05x"]["p_fold"]
             - m["arms"]["BASELINE"]["fold_05x"]["p_fold"]
             for m in active
             if m["arms"]["REAL"]["fold_05x"]
             and m["arms"]["BASELINE"]["fold_05x"]]),
        "counterfactual_equals_baseline_fold": all(
            (m["arms"]["COUNTERFACTUAL"]["fold_05x"] or {}).get("p_fold")
            == (m["arms"]["BASELINE"]["fold_05x"] or {}).get("p_fold")
            for m in active
            if m["arms"]["COUNTERFACTUAL"]["fold_05x"]
            and m["arms"]["BASELINE"]["fold_05x"]),
    }

    diag = diagnose_saturation(cfg, evidence=ev)

    snapshot = {
        "snapshot_version": "p49-real-v1",
        "manifest": manifest(),
        "data": {"chunk": "chunk_000", "scan_limit": scan_limit,
                 "target_player": pid,
                 "target_hands": len(player_hands)},
        "marginal_summary": summary,
        "saturation_diagnosis": diag,
        "p48_regression": {
            "n_scenarios": len(p48_results),
            "note": ("P48 场景在 p49-real-v2 manifest 下复跑；数值应与 "
                     "V3F_P48_REAL_VALIDATION_SNAPSHOT.json 逐位一致"),
            "results": p48_results,
        },
        "scenarios_marginal": marginal,
        "findings": {
            "F1_w_stat_zero": (
                "P42/P43 集成缺陷：conditional_player/conditional_population "
                "统计通道的 samples 嵌套在 prov['conditional'] 内，"
                "fold_equity.for_size 读取顶层 samples → w_stat=0 → "
                "条件 fold 证据完全不进入融合。COUNTERFACTUAL 臂 fold 与 "
                "BASELINE 逐位相同即为该缺陷的决策层表现。只定位，未修复。"),
            "F2_hero_conditioned_calibration": (
                "P46 continuing 校准以 hero 手牌为条件（baseline continue "
                "= 范围×底池赔率 vs hero 精确手牌）：同一真实证据在不同 "
                "hero 手牌下 λ 摆动（见 saturation_diagnosis."
                "hero_conditioning）。villain 范围不应随 hero 手牌变化"
                "（card removal 除外）。只定位，未修复。"),
            "F3_saturation_source": (
                "λ 饱和来自 DEVIATION_CAP（d clip ±1.0）；LAMBDA_CAP=1.2 "
                "不可达（λ=−SENSITIVITY×d ≤ ±1.0）。真实对手的继续率"
                "全部落在饱和区（baseline≈0.08 时线性区仅 0.05–0.3）。"),
            "F4_decision_insensitivity": (
                "在 w_stat=0 + 倾斜效应 ±3pp 的现状下，边际场景的 "
                "REAL vs BASELINE best_action 变化率见 "
                "marginal_summary.action_change_rate_real_vs_baseline；"
                "链条断点定位为 F1（最强证据通道被丢弃）。"),
        },
        "honesty": {
            "pko": "NOT VALIDATABLE WITH CURRENT DATA（语料 cash-only）",
            "mtt": "MECHANISM_CHECK_ONLY（无真实锦标赛手牌）",
            "combo_ground_truth": "NOT VALIDATABLE（观察者视角无 hole cards）",
            "profitability": "本 snapshot 不构成任何盈利性/GTO 证明",
        },
    }
    out = ROOT / "docs" / "V3F_P49_REAL_VALIDATION_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({"target": pid, "hands": len(player_hands),
                      "summary": summary,
                      "attribution": diag["attribution"],
                      "hero_conditioning": diag["hero_conditioning"]["verdict"],
                      "w_stat": diag["w_stat_probe"]["w_stat"]},
                     ensure_ascii=False, indent=1))
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    asyncio.run(main(limit))
