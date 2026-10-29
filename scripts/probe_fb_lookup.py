#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FF：fb（facing）查表的**实际命中/未命中分布**（按引擎键）。

直接测量"引擎在 fb 套件的局面里到底缺哪些桶"，用于指导下一步补解（比理论枚举更可靠）。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table import SpotTable   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FB = ROOT / "data" / "solver" / "spots_fb"


def main() -> None:
    st = SpotTable()
    st._ensure_loaded()
    rows = []
    for f in sorted(FB.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        for ex in (rec.get("exports") or []):
            q = {
                "street": str(rec.get("street") or ""),
                "board": list((rec.get("query") or {}).get("board") or []),
                "pot_bb": ex.get("pot_bb"), "node": "facing_bet",
                "bet_ratio": ex.get("bet_ratio"), "to_call_bb": ex.get("to_call_bb"),
                "facing_actor": ex.get("facing_actor") or ex.get("actor"),
                "pot_family": rec.get("family"),
                "ranges": {"oop": (rec.get("query") or {}).get("oop_range"),
                            "ip": (rec.get("query") or {}).get("ip_range")},
            }
            rows.append((str(rec.get("spot_id")), str(ex.get("export_id")), q))
    print(f"导出节点查询数: {len(rows)}")
    stat = collections.Counter()
    miss_keys = collections.Counter()
    for _sid, _eid, q in rows:
        h = st.lookup(q)
        if h is None:
            pot_cls = __import__("poker_decision_ai.core.providers.spot_table",
                                  fromlist=["_pot_class"])._pot_class(float(q["pot_bb"] or 0))
            bucket = __import__("poker_decision_ai.core.providers.spot_table",
                                 fromlist=["_bet_bucket"])._bet_bucket(float(q["bet_ratio"] or 0))
            stat["miss"] += 1
            miss_keys[(q["street"], pot_cls, str(bucket), q["facing_actor"], q["pot_family"])] += 1
        else:
            stat["approx" if h.get("is_approximate") else "exact"] += 1
    print("查表结果:", dict(stat))
    print("\n未命中键（top 12）:")
    for k, v in miss_keys.most_common(12):
        print(f"   {k} → {v}")


if __name__ == "__main__":
    main()
