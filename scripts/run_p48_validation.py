"""P48 snapshot 生成（只读真实数据；确定性；产物入 docs/）。

用法: python scripts/run_p48_validation.py [scan_limit]
产物: docs/V3F_P48_REAL_VALIDATION_SNAPSHOT.json
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
    LAMBDA_JUMP_LIMIT,
    build_evidence,
    compare_scenario,
    lambda_curve,
    manifest,
    pick_target_player,
    collect_real_sample,
    real_profile,
)

CHUNK0 = ROOT / "data" / "uoft" / "normalized" / "chunk_000.jsonl"


async def main(scan_limit: int) -> dict:
    cfg = load_config()
    hands = collect_real_sample(str(CHUNK0), scan_limit=scan_limit)
    pid = pick_target_player(hands)
    player_hands = hands[pid]
    ev = build_evidence(player_hands, hands)
    prof = real_profile(pid, ev)

    # 玩家行为画像（解释"为什么变"的证据基础）
    rpi_counts = ev["player_rpi"].to_dict()["counts"]
    position_rates = {k: {"entry_n": v["entry"][0], "entry": round(v["entry"][1] / v["entry"][0], 4) if v["entry"][0] else None,
                          "raise_n": v["raise"][0], "raise": round(v["raise"][1] / v["raise"][0], 4) if v["raise"][0] else None}
                      for k, v in sorted(rpi_counts.items())}

    results = []
    for scen in manifest()["scenarios"]:
        cmp1 = await compare_scenario(cfg, scen, "villain", prof, None)
        results.append(cmp1)

    n = len(player_hands)
    ns = tuple(x for x in (20, 40, 60, min(n, 200), n) if x <= n)
    curve = await lambda_curve(player_hands, hands, cfg, ns,
                               {"street": "preflop", "position": "CO",
                                "route": "open"})

    # 汇总统计（策略对照的可读摘要）
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 6) if vals else None

    cash = [r for r in results if not r["metrics"]["mechanism_only"]]
    active = [r for r in results if r["real"]["posterior_available"]]
    summary = {
        "n_scenarios": len(results),
        "fallback_rate": round(1 - len(active) / len(results), 4),
        "range_tv": {"median": med([r["metrics"]["range_tv"] for r in active]),
                     "max": max((r["metrics"]["range_tv"] for r in active), default=None)},
        "equity_diff": {"median": med([r["metrics"]["equity_diff"] for r in active])},
        "fold_diff_by_size": {s: {"median": med([r["metrics"]["fold"].get(s, {}).get("diff")
                                                 for r in active if r["metrics"]["fold"].get(s)])}
                              for s in ("0.5x_pot", "1.0x_pot", "2.0x_pot")},
        "best_action_change_rate": round(
            sum(1 for r in active if r["metrics"]["best_action_change"]) / max(1, len(active)), 4),
        "lambda_sign": {"loosen(<0)": sum(1 for r in active if (r["real"]["lambda"] or 0) < 0),
                        "tighten(>0)": sum(1 for r in active if (r["real"]["lambda"] or 0) > 0),
                        "zero": sum(1 for r in active if (r["real"]["lambda"] or 0) == 0)},
    }

    # determinism 双跑（首个 cash 场景）
    det = await compare_scenario(cfg, results[0]["id"] and
                                 next(s for s in manifest()["scenarios"]
                                      if s["id"] == results[0]["id"]),
                                 "villain", prof, None)

    snapshot = {
        "snapshot_version": "p48-real-v1",
        "manifest": manifest(),
        "data": {"chunk": "chunk_000", "scan_limit": scan_limit,
                 "n_players_with_hands": len(hands),
                 "target_player": pid,
                 "target_hands": n,
                 "position_rates": position_rates,
                 "gate": ev["gate"].get("evidence_gate")},
        "target_entry_baseline_note": (
            "baseline 隐含率来自 configs/ranges.yaml（静态 book 范围）；"
            "REAL 率来自该玩家在扫描窗口内的真实 preflop 决策"),
        "summary": summary,
        "lambda_curve": curve,
        "determinism_double_run_equal": det == results[0],
        "scenarios": results,
        "honesty": {
            "pko": "NOT VALIDATABLE WITH CURRENT DATA（语料 cash-only）",
            "mtt": "MECHANISM_CHECK_ONLY（无真实锦标赛手牌）",
            "combo_ground_truth": "NOT VALIDATABLE（观察者视角无 hole cards）",
            "profitability": "本 snapshot 不构成任何盈利性/GTO 证明",
        },
    }
    out = ROOT / "docs" / "V3F_P48_REAL_VALIDATION_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({"target": pid, "hands": n, "summary": summary,
                      "lambda_curve": curve["curve"],
                      "det_equal": det == results[0]},
                     ensure_ascii=False, indent=1))
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    asyncio.run(main(limit))
