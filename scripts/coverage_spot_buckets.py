#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-ER：**引擎查找键**层面的覆盖度量（而非"唯一牌面"）。

引擎查表维度：`(street, pot_class, texture_class, actor, node)`（+ 声明范围集）。
本脚本枚举 r1 库实际覆盖的桶组合，并列出**缺失/单例**组合，供补解定位。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table import (      # noqa: E402
    _pot_class, _texture_class)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    d = ROOT / "data" / "solver" / "spots_r1"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    buckets: dict[tuple, list[str]] = collections.defaultdict(list)
    for sid in man.get("solved_ids") or []:
        p = d / f"{sid}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        q = r.get("query") or {}
        board = list(q.get("board") or [])
        key = (str(q.get("street")), str(_pot_class(float(q.get("pot_bb") or 0))),
                str(_texture_class(board)), str(r.get("actor") or "oop"),
                str(r.get("node") or "first_in"))
        buckets[key].append(sid)
    print(f"库内解 {sum(len(v) for v in buckets.values())} 个，覆盖桶组合 {len(buckets)} 个\n")
    rows = sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    by_street = collections.Counter(k[0] for k in buckets)
    print("按街的桶数:", dict(by_street))
    print("\n桶组合（解数）：")
    for k, v in rows:
        flag = "  ← 单例" if len(v) == 1 else ""
        print(f"  {k[0]:<6} pot={k[1]:<6} tex={k[2]:<16} actor={k[3]:<4} node={k[4]:<11} n={len(v):<3}{flag}")
    singles = {k: v for k, v in buckets.items() if len(v) == 1}
    print(f"\n单例桶（不同牌面各 1 个解 ⇒ 精确命中依赖该唯一牌面）: {len(singles)}")
    (ROOT / "docs" / "spot_bucket_coverage.json").write_text(
        json.dumps({"n_buckets": len(buckets),
                     "by_street": dict(by_street),
                     "buckets": {str(k): len(v) for k, v in buckets.items()},
                     "samples": {str(k): v[:3] for k, v in buckets.items()}},
                    ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/spot_bucket_coverage.json")


if __name__ == "__main__":
    main()
