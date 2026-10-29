"""M4 Phase B3 — GATE 冻结快照（解封 B 组**之前**落盘；防「看过结果再调参」）。

把两场 GATE 战役的全部协议参数（桌面构成、手数、种子、预算档、抽水模型、
代码工件哈希）写入一个不可变快照 JSON，并回写快照自身的 SHA-256 到
进度档案。之后任何参数改动都会与快照哈希冲突 = 可审计。

用法：
  <repo>/.venv/Scripts/python.exe scripts/freeze_m4_gate.py \
      --gate-min-budget soft --gate-1-budget strong
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "configs" / "bench" / "m4_gate_frozen.json"
CODE_ARTIFACTS = [
    "scripts/run_m4_gate_campaign.py",
    "scripts/unseal_group_b_m4.py",
    "external_harness/protocol/cash_match_6max.py",
    "external_harness/protocol/pool_bots.py",
    "external_harness/protocol/rake_model.py",
    "configs/bench/rake_coinpoker.yaml",
    "configs/config.yaml",
]
PROGRESS = ROOT / ".collab" / "state" / "round6_progress.md"


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    from scripts.run_m4_gate_campaign import CAMPAIGNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-min-budget", default=None,
                    choices=["soft", "mixed", "strong"],
                    help="GATE-MIN 预算档（A3 旋钮闸门结论决定；缺省=继承生产 strong）")
    ap.add_argument("--gate-1-budget", default=None,
                    choices=["soft", "mixed", "strong"])
    args = ap.parse_args()

    snapshot = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "campaigns": {},
        "code_artifacts_sha256": {rel: sha(rel) for rel in CODE_ARTIFACTS},
        "note": ("M4 GATE 冻结快照：解封 B 组前落盘。此后任何参数改动"
                 "与本文件冲突 = 可审计（防「看过结果再调参」）。"),
    }
    budget = {"gate_min": args.gate_min_budget, "gate_1": args.gate_1_budget}
    for name, spec in CAMPAIGNS.items():
        snap = {k: v for k, v in spec.items() if k != "seats"}
        snap["seats"] = {str(k): v for k, v in spec["seats"].items()}
        snap["pool_budget_profile"] = budget.get(name) or "strong"
        snapshot["campaigns"][name] = snap
    blob = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    snapshot["self_sha256"] = digest
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    line = (f"{time.strftime('%Y-%m-%d %H:%M')} | B3-FREEZE | GATE 冻结快照"
            f"落盘 sha256={digest[:16]}… gate_min_budget="
            f"{snapshot['campaigns']['gate_min']['pool_budget_profile']} "
            f"gate_1_budget="
            f"{snapshot['campaigns']['gate_1']['pool_budget_profile']}\n")
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
