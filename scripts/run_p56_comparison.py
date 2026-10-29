"""P56 snapshot：策略 Core v3.0（P54/P55 基线）vs v3.1（SACA）对照。

只读评估；确定性；产物 docs/V3F_P56_CONTINUUM_UPGRADE_SNAPSHOT.json。
v3.0 基线 = P54 snapshot 的 decision_ev/fold_calibration（同一输入、
同一分割）；v3.1 = 当前数学重跑。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.pluribus_decision_adapter import (  # noqa: E402
    iter_pluribus_rows,
    parse_hand_row,
)
from poker_decision_ai.pluribus_validation import validate  # noqa: E402

P54_SNAPSHOT = ROOT / "docs" / "V3F_P54_DECISION_VALIDATION_SNAPSHOT.json"


def main(decision_hand_limit: int = 400) -> dict:
    hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    print(f"parsed {len(hands)} hands")
    val = validate(hands, decision_hand_limit=decision_hand_limit)

    before = None
    if P54_SNAPSHOT.exists():
        p54 = json.loads(P54_SNAPSHOT.read_text(encoding="utf-8"))
        before = p54.get("validation", {}).get("decision_ev")
        before_fc = p54.get("validation", {}).get("fold_calibration")

    # 行为分布对比（v3.1）
    rows = val["decision_rows_sample"]
    dist: dict[str, int] = {}
    dist_base: dict[str, int] = {}
    for r in val.get("decision_rows_sample", []):
        pass
    # 全行统计（diagnose 模式拿全行）
    val_full = validate(hands, decision_hand_limit=decision_hand_limit,
                        diagnose=True)
    all_rows = val_full.get("decision_rows_all") or []
    for r in all_rows:
        if r.get("real_best"):
            dist[r["real_best"]] = dist.get(r["real_best"], 0) + 1
        if r.get("baseline_best"):
            dist_base[r["baseline_best"]] = dist_base.get(r["baseline_best"],
                                                          0) + 1
    # check→bet rate（v3.1，同口径 P55）
    chg = [r for r in all_rows if r.get("action_change")]
    ctb = sum(1 for r in chg if r.get("baseline_best") == "check"
              and r.get("real_best") == "bet")
    # call survival：observed=call 的行中 REAL 推荐 call 的份额
    obs_call = [r for r in all_rows
                if r.get("observed_human_action") == "call"
                and r.get("real_best")]
    call_survival = (round(sum(1 for r in obs_call
                               if r["real_best"] == "call") / len(obs_call),
                             6) if obs_call else None)
    call_survival_base = (round(sum(1 for r in obs_call
                                    if r.get("baseline_best") == "call")
                                  / len(obs_call), 6) if obs_call else None)

    snapshot = {
        "snapshot_version": "p56-saca-v1",
        "design": {
            "model": "SACA v3.1（street-aware continuation adjustments）",
            "components": [
                "A8 realization R(street,IP/OOP) 作用于 call/check 延续价值",
                "A9 implied odds κ×p_improve×F_future（确定性听牌数牌）",
                "A10 initiative INIT 作用于 bet/raise 被跟分支（OOP=1.0）",
                "competitive_actions：utility 边际 ≤0.10×pot（mixed 表达）",
            ],
            "invariants": [
                "I1 river/preflop 无 CV（逐位 P41-P45）",
                "I2 fold EV=0 恒定",
                "I5 CASH utility==ev 位等",
                "I10 位置/手牌缺失 → CV 中性",
                "I-UNKNOWN/legality 语义不变",
            ],
            "not_solver": "声明式近似——绝不宣称 GTO/solver",
        },
        "before_after": {
            "v3_0_P54_decision_ev": before,
            "v3_1_decision_ev": {
                k: v for k, v in val["decision_ev"].items()},
            "v3_0_action_change_rate": (before or {}).get("action_change_rate"),
            "v3_1_action_change_rate": val["decision_ev"][
                "action_change_rate"],
            "v3_1_check_to_bet_share": (round(ctb / len(chg), 6)
                                        if chg else None),
            "call_survival_observed_call": {
                "baseline": call_survival_base, "real": call_survival},
            "action_distribution_real_v3_1": dict(
                sorted(dist.items(), key=lambda x: -x[1])),
            "action_distribution_baseline": dict(
                sorted(dist_base.items(), key=lambda x: -x[1])),
        },
        "fold_calibration": val["fold_calibration"],
        "p54_fold_calibration_reference": before_fc,
        "labels": val["labels"],
        "legality": val["legality"],
        "unknown": val["unknown"],
        "split": val["split"],
        "honesty": {
            "agreement_not_objective": ("不以 agreement 单一指标优化；"
                                        "本对照报告全部指标"),
            "profitability": "不构成任何盈利性/GTO 证明",
            "research_play": "Pluribus 研究对局——不外推自然线上行为",
            "mtt_pko": "NOT VALIDATABLE WITH CURRENT DATA",
        },
    }
    out = ROOT / "docs" / "V3F_P56_CONTINUUM_UPGRADE_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({
        "headline_v3_0": {k: (before or {}).get(k) for k in
                          ("observed_agreement_baseline",
                           "observed_agreement_real", "action_change_rate",
                           "regret_zero_share", "regret_proxy_median")},
        "headline_v3_1": {k: val["decision_ev"].get(k) for k in
                          ("observed_agreement_baseline",
                           "observed_agreement_real", "action_change_rate",
                           "regret_zero_share", "regret_proxy_median")},
        "check_to_bet_share_v3_1": snapshot["before_after"][
            "v3_1_check_to_bet_share"],
        "call_survival": snapshot["before_after"][
            "call_survival_observed_call"],
        "action_distribution_real_v3_1": snapshot["before_after"][
            "action_distribution_real_v3_1"],
        "fold_calibration_v3_1": val["fold_calibration"]["arms"],
        "legality": val["legality"],
    }, ensure_ascii=False, indent=1))
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(limit)
