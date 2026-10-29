"""P54 snapshot 生成（只读真实数据；确定性；产物入 docs/）。

用法: python scripts/run_p54_validation.py [decision_hand_limit]
产物: docs/V3F_P54_DECISION_VALIDATION_SNAPSHOT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.pluribus_decision_adapter import (  # noqa: E402
    PluribusDecisionAdapter,
    iter_pluribus_rows,
    parse_hand_row,
)
from poker_decision_ai.pluribus_validation import validate  # noqa: E402


def main(decision_hand_limit: int) -> dict:
    # ---- 全量解析（10,000 手） ----
    hands = []
    skipped = 0
    for row in iter_pluribus_rows():
        p = parse_hand_row(row)
        if p is None:
            skipped += 1
            continue
        hands.append(p)
    print(f"parsed {len(hands)} hands, skipped {skipped}")

    # ---- P52 字段完整性（全量记录） ----
    adapter = PluribusDecisionAdapter()
    rep = adapter.evaluate_dataset(limit=0)
    rep["hands_skipped_parse"] = skipped
    rep["n_hands_input"] = len(hands) + skipped

    # ---- 决策级验证（fold 校准全量 eval；EV 有界子集） ----
    val = validate(hands, decision_hand_limit=decision_hand_limit)

    snapshot = {
        "snapshot_version": "p54-decision-v1",
        "field_completeness": rep,
        "validation": val,
        "honesty": {
            "mtt": "NOT VALIDATABLE WITH CURRENT DATA",
            "pko": "NOT VALIDATABLE WITH CURRENT DATA",
            "research_play": ("Pluribus 研究对局——结论不得外推为自然线上"
                              "玩家行为；人类座位为匿名职业玩家"),
            "ai_seat": "solver_label——全程排除出 human validation",
            "profitability": "本 snapshot 不构成任何盈利性/GTO 证明",
            "decision_ev_bound": (f"决策级 EV 在 eval 确定性前缀 "
                                  f"{decision_hand_limit} 手上运行"
                                  "（equity 成本有界，manifest 声明）"),
        },
    }
    out = ROOT / "docs" / "V3F_P54_DECISION_VALIDATION_SNAPSHOT.json"
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    print(json.dumps({
        "field_completeness": {
            "n_records": rep["n_records"],
            "ai_seat_excluded_hands": rep["ai_seat_excluded_hands"],
            "human_decision_points": rep["human_decision_points"],
            "dataset_verdict_ratio": rep["dataset_verdict_ratio"]},
        "validation": {k: val[k] for k in
                       ("split", "labels", "unknown", "legality",
                        "fold_calibration", "decision_ev")},
    }, ensure_ascii=False, indent=1)[:3000])
    return snapshot


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(limit)
