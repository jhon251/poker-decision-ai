"""P30-9-GH：flop 现场求解的**成本实验**（尺寸菜单/迭代/精度 vs 耗时）。

目标：判断 flop 能否进入实时或"准实时（异步缓存）"路径。
变量：bet_sizes（默认 60%,e,a；简化 33%,a）、raise_sizes、max_iterations、
      target_exploitability_pot_pct。
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai import cli as cli_mod            # noqa: E402

OOP = "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo"
IP = "TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo"
BOARD = ["Ks", "7h", "2c"]

CONFIGS = [
    ("默认菜单 60%,e,a / 300it / 1.0%", {"max_iterations": 300,
                                            "target_exploitability_pot_pct": 1.0}),
    ("简化菜单 33%,a / 300it / 1.0%", {"bet_sizes_oop": "33%,a", "bet_sizes_ip": "33%,a",
                                          "max_iterations": 300,
                                          "target_exploitability_pot_pct": 1.0}),
    ("简化菜单 33%,a / 100it / 2.0%", {"bet_sizes_oop": "33%,a", "bet_sizes_ip": "33%,a",
                                          "max_iterations": 100,
                                          "target_exploitability_pot_pct": 2.0}),
    ("极简 a / 100it / 2.0%", {"bet_sizes_oop": "a", "bet_sizes_ip": "a",
                                 "max_iterations": 100,
                                 "target_exploitability_pot_pct": 2.0}),
]


async def main() -> None:
    rt = cli_mod.build_runtime()
    ctx = rt.new_context("flop_cost", "x")
    prov = ctx.providers.get_solver()
    for label, extra in CONFIGS:
        q = {"street": "flop", "board": BOARD, "ranges": {"oop": OOP, "ip": IP},
              "pot": 40.0, "effective_stack": 100.0, "bb": 1.0}
        q.update(extra)
        t0 = time.time()
        try:
            res = await prov.solve(q)
            el = time.time() - t0
            print(f"{label:<34} {el:>7.1f} s | expl={getattr(res, 'exploitability_bb', None)} "
                  f"| actions={[(a.name, a.size_bb, round(float(a.frequency), 2)) for a in (res.actions or [])][:4]}")
        except Exception as e:                              # noqa: BLE001
            print(f"{label:<34} 失败 {type(e).__name__}: {e}")


asyncio.run(main())
