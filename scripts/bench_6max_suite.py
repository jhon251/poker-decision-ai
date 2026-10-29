#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FI：6-max 保真度**多 offset 聚合基线**（消除单批采样偏差）。

用法：python scripts/bench_6max_suite.py --offsets 0,24,48,72 --nodes 24 --hands 8
输出：各批指标 + 合并指标，并写 docs/bench_6max_suite.json（供回归门使用）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def run_one(offset: int, nodes: int, hands: int) -> dict:
    out = ROOT / ".tmp" / f"b6_offset{offset}.json"
    log = ROOT / ".tmp" / f"b6_offset{offset}.log"
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run([PY, str(ROOT / "scripts" / "bench_6max_library.py"),
                         "--nodes", str(nodes), "--hands", str(hands),
                         "--offset", str(offset), "--out", str(out)],
                        cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, timeout=5400)
    txt = log.read_text(encoding="utf-8", errors="replace")

    def grab(pat: str) -> tuple[int, int] | None:
        m = re.search(pat + r"[^:：]*[:：]\s*(\d+)/(\d+)", txt)
        return (int(m.group(1)), int(m.group(2))) if m else None

    sup = grab("逐手支持率")
    pri = grab("主选一致率")
    anc = grab("锚定率")
    return {"offset": offset,
             "support": sup, "primary": pri, "anchor": anc,
             "sample": (sup[1] if sup else 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="0,24,48,72")
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--hands", type=int, default=8)
    args = ap.parse_args()
    rows = []
    for off in [int(x) for x in args.offsets.split(",") if x.strip()]:
        r = run_one(off, args.nodes, args.hands)
        rows.append(r)
        if r["support"]:
            print(f"offset {off:<4} 支持 {r['support'][0]}/{r['support'][1]} "
                  f"({r['support'][0]/max(r['support'][1],1)*100:.1f}%) | "
                  f"主选 {r['primary'][0]}/{r['primary'][1]} "
                  f"({r['primary'][0]/max(r['primary'][1],1)*100:.1f}%) | "
                  f"锚定 {r['anchor'][0]}/{r['anchor'][1]}")
    agg = {}
    for key in ("support", "primary", "anchor"):
        num = sum(r[key][0] for r in rows if r[key])
        den = sum(r[key][1] for r in rows if r[key])
        agg[key] = round(num / den, 4) if den else None
        agg[key + "_n"] = [num, den]
    print(f"\n合并 {agg['support_n'][1]} 决策：支持 {agg['support']*100:.1f}% | "
          f"主选 {agg['primary']*100:.1f}% | 锚定 {agg['anchor']*100:.1f}%")
    (ROOT / "docs" / "bench_6max_suite.json").write_text(
        json.dumps({"offsets": args.offsets, "nodes": args.nodes, "hands": args.hands,
                     "per_offset": rows, "aggregate": agg}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("已写 docs/bench_6max_suite.json")


if __name__ == "__main__":
    main()
