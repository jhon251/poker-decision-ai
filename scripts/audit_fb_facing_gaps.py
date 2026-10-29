"""P30-9-FE：修正 fb 缺口工具口径（索引键同时含 pot_cls 与 family）。

引擎 facing 索引键 = (street, oop_range, ip_range, pot_cls, bucket, role, **family**, texture)
⇒ 缺口分析必须按**完整键**统计（此前只按 family 过滤 ⇒ shove 节点被误判为 SRP 缺口 ✗）。
本版：按完整键计算已覆盖集合；完整集 = 街 × pot_cls × 纹理 × 尺寸桶 × 角色 × family。
"""
from __future__ import annotations

import collections
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table import SpotTable   # noqa: E402

TEX = {"flop": ["(False, False, 2)", "(False, False, 3)", "(False, True, 1)",
                 "(True, False, 3)", "(True, False, 2)"],
        "turn": ["(False, False, 2)", "(False, False, 3)", "(False, False, 4)",
                  "(False, True, 1)", "(True, False, 2)", "(True, False, 3)"],
        "river": ["(False, False, 2)", "(False, False, 3)", "(False, False, 4)",
                   "(False, True, 1)", "(True, False, 2)", "(True, False, 3)",
                   "(True, False, 4)"]}
POTS = ["SRP", "big"]
FAMS = ["SRP", "3BP"]
BUCKETS = ["mid", "large", "shove"]


def main() -> None:
    st = SpotTable()
    st._ensure_loaded()
    have = collections.Counter()
    for k in st._by_bucket_facing:
        street, _oop, _ip, pot, bucket, role, family, tex = k
        have[(street, pot, str(tex), bucket, role, family)] += 1
    print(f"facing 索引键: {len(st._index_facing)} | 桶: {len(st._by_bucket_facing)}")
    print("按街:", dict(collections.Counter(k[0] for k in have)))

    missing = []
    for street in ("turn", "river"):
        for pot, bucket, role, family, tex in itertools.product(
                POTS, BUCKETS, ("oop", "ip"), FAMS, TEX[street]):
            if (street, pot, tex, bucket, role, family) not in have:
                missing.append((street, pot, tex, bucket, role, family))
    print(f"\n缺失组合（turn/river 完整键）: {len(missing)}")
    g = collections.Counter((m[0], m[3], m[5]) for m in missing)
    for k, v in sorted(g.items()):
        print(f"   {k} → {v}")
    (pathlib.Path(__file__).resolve().parents[1] / "docs"
     / "fb_facing_gaps.json").write_text(
        json.dumps({"have": {str(k): v for k, v in have.items()},
                     "missing": [list(m) for m in missing]},
                    ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/fb_facing_gaps.json")


if __name__ == "__main__":
    main()
