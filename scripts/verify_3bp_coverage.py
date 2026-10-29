#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FY：导入效果验证 + **算力投影**（回答"要解多少牌面才够"）。

1) 逐条验证：48 个导入牌面是否都能被引擎**精确命中**（exact ✓）；
2) 均匀随机覆盖率：3BP first_in 配置下 200 个随机翻牌的 exact 比例（诚实口径）；
3) 算力投影：以实测「单解秒数」推算达到 10% / 25% / 50% 均匀覆盖所需牌面数与时间。
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "solver" / "spots_6max"
FLOPS = 22100          # C(52,3)

Q = {"street": "flop", "pot_bb": 25.5, "node": "first_in",
      "positions": {"oop": "BB", "ip": "BTN"}, "actor": "BB",
      "family": "3bp", "stack_bb": 87.5}


def main() -> None:
    table = SpotTable6max(str(LIB))
    table._ensure_loaded()

    imported = sorted(LIB.glob("3bp-ourhu-*.json"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in imported]
    exact = 0
    for r in recs:
        h = table.lookup({**Q, "board": list(r["board"])})
        if h is not None and not h.get("is_approximate"):
            exact += 1
    print(f"① 导入牌面精确命中: {exact}/{len(recs)}"
          f"{' ✓' if exact == len(recs) else ' ✗'}")

    rnd = random.Random(7)
    deck = [x + y for x in "AKQJT98765432" for y in "shdc"]
    n = 2000
    ex = 0
    for _ in range(n):
        b = rnd.sample(deck, 3)
        h = table.lookup({**Q, "board": b})
        if h is not None and not h.get("is_approximate"):
            ex += 1
    rate = ex / n
    print(f"② 均匀随机翻牌 exact 覆盖率: {ex}/{n} = {rate*100:.2f}%")

    per_sec = 519.3 / 48
    print(f"③ 实测单解 {per_sec:.1f} 秒/牌面（3BP flop）")
    print("   达到目标均匀覆盖所需：")
    for p in (0.05, 0.10, 0.25, 0.50):
        need = int(FLOPS * p)
        hrs = need * per_sec / 3600
        print(f"     {p*100:>4.0f}% ⇒ {need:>6} 牌面 ≈ {hrs:>6.1f} 小时")

    out = ROOT / "docs" / "coverage_projection_3bp.json"
    out.write_text(json.dumps({
        "stage": "P30-9-FY",
        "imported_boards": len(recs), "imported_exact_hits": exact,
        "uniform_random_exact_rate": round(rate, 5), "sampled": n,
        "sec_per_board_measured": round(per_sec, 2),
        "flop_space": FLOPS,
        "projection": {f"{int(p*100)}%": {"boards": int(FLOPS * p),
                                            "hours": round(FLOPS * p * per_sec / 3600, 1)}
                        for p in (0.05, 0.10, 0.25, 0.50)},
        "note": ("均匀随机翻牌是正确口径（牌面与范围独立）；"
                  "要提高均匀覆盖必须解成千上万个牌面 ⇒ 3BP 可行（~10.8s/牌面），SRP 不可行（~10min/牌面）"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
