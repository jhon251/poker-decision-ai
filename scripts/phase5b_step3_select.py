"""Phase 5B Step 3：probe 选择（自动规则，不人为挑选）。

规则：
  ICC 核心通过: ICC(1,k) > 0.10 且 ICC(1,k) CI 下界 > 0（画像信度）——
     注意 ICC(1)（单手信度）仅报告不设通过阈值（低基率探针单手信度天然低）。
  split-half 通过: r > 0.15 且 CI 下界 > 0。
  核心集 = {vpip, pfr, three_bet, position_sensitivity} 中两项均通过的。
  cbet → 恒为 exploratory（低样本机会门控，不参与 MVP 核心判定）。
  未通过核心集的探针 → excluded（如实注明原因，不降阈值）。

输出 data/uoft/models/mvp/probe_selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import ensure_dir  # noqa: E402

CORE_SET = ("vpip", "pfr", "three_bet", "position_sensitivity")
ICC_THRESHOLD = 0.10
R_THRESHOLD = 0.15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc", default="data/uoft/models/mvp/icc.json")
    ap.add_argument("--sh", default="data/uoft/models/mvp/splithalf.json")
    ap.add_argument("--out", default="data/uoft/models/mvp")
    args = ap.parse_args()
    ensure_dir(args.out)

    icc = json.load(open(args.icc, encoding="utf-8"))["probes"]
    sh = json.load(open(args.sh, encoding="utf-8"))["probes"]

    result = {"thresholds": {"icc1_k": ICC_THRESHOLD, "r": R_THRESHOLD},
              "probes": {}}
    for p in ("vpip", "pfr", "three_bet", "position_sensitivity", "cbet",
              "rfi_ep", "rfi_lp"):
        icc_v = icc.get(p, {}).get("icc1_k")
        icc_lo = (icc.get(p, {}).get("icc1_k_ci") or [0, 0])[0]
        r_v = sh.get(p, {}).get("r")
        r_lo = (sh.get(p, {}).get("r_ci95") or [0, 0])[0]
        icc_pass = icc_v is not None and icc_v > ICC_THRESHOLD and icc_lo > 0
        r_pass = r_v is not None and r_v > R_THRESHOLD and r_lo > 0

        if p == "cbet":
            decision = "exploratory"
            reason = "低样本机会门控探针（中位机会 ~17），按设计不参与 MVP 核心判定"
        elif p in CORE_SET:
            if icc_pass and r_pass:
                decision = "core"
                reason = ""
            else:
                decision = "excluded"
                reason = []
                if not icc_pass:
                    reason.append(f"ICC(1,k)={icc_v} (阈值 {ICC_THRESHOLD}) 或 CI 含 0")
                if not r_pass:
                    reason.append(f"split-half r={r_v} (阈值 {R_THRESHOLD}) 或 CI 含 0")
                reason = "; ".join(reason)
        else:
            decision = "info"
            reason = "rfi 分量，仅作 position_sensitivity 的构成报告"

        result["probes"][p] = {
            "icc1": icc.get(p, {}).get("icc1"),
            "icc1_k": icc_v, "icc1_k_ci": icc.get(p, {}).get("icc1_k_ci"),
            "r": r_v, "r_ci95": sh.get(p, {}).get("r_ci95"),
            "icc_pass": icc_pass, "r_pass": r_pass,
            "decision": decision, "reason": reason,
        }
        print(f"  {p}: {decision} (icc_pass={icc_pass}, r_pass={r_pass})",
              flush=True)

    (Path(args.out) / "probe_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
