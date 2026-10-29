#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一批已解 spot 并入 `spots_r1`（追加到 manifest.solved_ids），保留 spot_id 前缀。

用法：python scripts/merge_spots_into_r1.py --src data/solver/spots_r3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/solver/spots_r3")
    ap.add_argument("--dst", default="data/solver/spots_r1")
    args = ap.parse_args()
    src = ROOT / args.src
    dst = ROOT / args.dst
    man_p = dst / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    ids = list(man.get("solved_ids") or [])
    added = []
    for f in sorted(src.glob("*.json")):
        if f.name == "manifest.json":
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        sid = str(rec.get("spot_id") or f.stem)
        shutil.copy2(f, dst / f"{sid}.json")
        added.append(sid)
    new_ids = [i for i in added if i not in ids]
    man["solved_ids"] = ids + new_ids
    man["n_solved"] = len(man["solved_ids"])
    man["n_spots"] = len(man["solved_ids"])
    man["merged_from"] = sorted(set(list(man.get("merged_from") or []) + [args.src]))
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"并入 {len(new_ids)} 个解（{args.src} → {args.dst}）；库现 {len(man['solved_ids'])} 个")
    for i in new_ids:
        print("   +", i)


if __name__ == "__main__":
    main()
