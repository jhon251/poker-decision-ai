"""P55 snapshot 生成（只读诊断；确定性；产物入 docs/）。

用法: python scripts/run_p55_diagnosis.py [decision_hand_limit]
产物: docs/V3F_P55_DIVERGENCE_DIAGNOSIS.json
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
from poker_decision_ai.pluribus_diagnosis import diagnose_full  # noqa: E402


def main(decision_hand_limit: int) -> dict:
    hands = []
    for row in iter_pluribus_rows():
        p = parse_hand_row(row)
        if p is not None:
            hands.append(p)
    print(f"parsed {len(hands)} hands")
    diag = diagnose_full(hands, decision_hand_limit=decision_hand_limit)
    out = ROOT / "docs" / "V3F_P55_DIVERGENCE_DIAGNOSIS.json"
    out.write_text(json.dumps(diag, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("snapshot ->", out)
    cl = diag["classification"]
    print(json.dumps({
        "n_rows": cl["n_rows"], "n_changed": cl["n_changed"],
        "action_change_rate": cl["action_change_rate"],
        "change_categories": cl["change_categories"],
        "lambda_distribution": cl["lambda_distribution"],
        "fold_prob_diff": cl["fold_prob_diff"],
        "equity_diff": cl["equity_diff"],
        "change_rate_by_evidence_bucket": cl["change_rate_by_evidence_bucket"],
        "agreement_by_observed_action": cl["agreement_by_observed_action"],
    }, ensure_ascii=False, indent=1)[:3500])
    return diag


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(limit)
