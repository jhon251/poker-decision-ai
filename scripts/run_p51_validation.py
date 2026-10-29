"""P51 snapshot 生成（只读真实数据；确定性；产物入 docs/）。

用法: python scripts/run_p51_validation.py [scan_limit]
产物: docs/V3F_P51_REAL_VALIDATION_SNAPSHOT.json
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
from poker_decision_ai.domain.poker.conditional_model import (  # noqa: E402
    ConditionalFoldIndex)
from poker_decision_ai.real_validation import (                # noqa: E402
    build_evidence,
    build_loo_split,
    collect_eval_decisions,
    collect_real_sample,
    compare_scenario,
    evaluate_fold_calibration,
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

    # ---- P51 核心：LOO 行为校准（证据段=前 70%，评估段=后 30%） ----
    ev_hands, eval_hands = build_loo_split(player_hands)
    ev_ids = {h["hand_id"] for h in ev_hands}
    eval_ids = {h["hand_id"] for h in eval_hands}
    leakage_free = not (ev_ids & eval_ids)

    player_idx = ConditionalFoldIndex()
    for h in ev_hands:
        player_idx.add_events(h["events"])
    pop_idx = ConditionalFoldIndex()
    for p, hs in hands.items():
        if p == pid:
            continue                    # 评估玩家不进 population（隔离）
        for h in hs:
            pop_idx.add_events(h["events"])
    for h in ev_hands:
        pop_idx.add_events(h["events"])

    records = collect_eval_decisions(eval_hands)
    calibration = evaluate_fold_calibration(records, player_idx, pop_idx)

    # 按 street 分解（breakdown）
    breakdown = {}
    for street in ("flop", "turn", "river"):
        sub = [r for r in records if r["street"] == street]
        if sub:
            breakdown[street] = evaluate_fold_calibration(sub, player_idx,
                                                          pop_idx)

    # ---- 边际三臂（45）+ P48 场景回归（17） ----
    prof = real_profile("villain", build_evidence(player_hands, hands))
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
        "changed_scenarios": [m["id"] for m in marginal
                              if m["action_change"]["baseline_vs_real"]],
        "regret_proxy_median": med(
            [m.get("regret_proxy") for m in active]),
        "regret_proxy_positive_share": round(
            sum(1 for m in active if (m.get("regret_proxy") or 0) > 0)
            / max(1, len(active)), 4),
        "ev_margin_real_median": med(
            [m["ev_margin"].get("REAL") for m in active]),
        "equity_diff_median": med(
            [m["arms"]["REAL"]["equity"] - m["arms"]["BASELINE"]["equity"]
             for m in active]),
        "fold_diff_median": med(
            [fold_of(m, "REAL") - fold_of(m, "BASELINE") for m in active
             if fold_of(m, "REAL") is not None
             and fold_of(m, "BASELINE") is not None]),
    }

    snapshot = {
        "snapshot_version": "p51-real-v1",
        "manifest": manifest(),
        "data": {"chunk": "chunk_000", "scan_limit": scan_limit,
                 "target_player": pid,
                 "target_hands": len(player_hands),
                 "evidence_hands": len(ev_hands),
                 "eval_hands": len(eval_hands),
                 "eval_decisions": len(records),
                 "leakage_free_hand_split": leakage_free},
        "behavioral_calibration": {
            "method": ("LOO 时间序分割；BASELINE=population 同条件率；"
                       "REAL=player 条件估计（population 先验，P43 阶梯）；"
                       "配对评估，REAL 不可用的记录跳过并计入 coverage"),
            "overall": calibration,
            "by_street": breakdown,
            "discipline_note": ("historical agreement（此处为 fold 校准）"
                                "≠ 正确性/盈利性；仅验证证据→行为预测链路"),
        },
        "marginal_summary": summary,
        "p48_regression": {"n_scenarios": len(p48_results),
                           "results": p48_results},
        "scenarios_marginal": marginal,
        "honesty": {
            "pko": "NOT VALIDATABLE WITH CURRENT DATA（语料 cash-only）",
            "mtt": "MECHANISM_CHECK_ONLY（无真实锦标赛手牌）",
            "combo_ground_truth": "NOT VALIDATABLE（观察者视角无 hole cards）",
            "observed_action_agreement": (
                "NOT VALIDATABLE WITH CURRENT DATA——chunk_000 全量 100,979 "
                "手 hole-card 手牌为 0（P51 实测）；历史 action 无法在相同"
                "信息条件下与模型决策对照"),
            "profitability": "本 snapshot 不构成任何盈利性/GTO 证明",
        },
    }
    out = ROOT / "docs" / "V3F_P51_REAL_VALIDATION_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({"target": pid, "hands": len(player_hands),
                      "evidence/eval": [len(ev_hands), len(eval_hands)],
                      "leakage_free": leakage_free,
                      "calibration_overall": calibration,
                      "by_street": {k: v["arms"] if v else None
                                    for k, v in breakdown.items()},
                      "marginal_summary": summary},
                     ensure_ascii=False, indent=1))
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    asyncio.run(main(limit))
