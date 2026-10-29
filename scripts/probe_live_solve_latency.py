"""P30-9-GE：用**正确键**测 provider live solve 的端到端延迟（turn/river/flop）。

provider.build_cli_query 期望：street / board / ranges{oop,ip} / pot / effective_stack / bb
（筹码口径），可选 bet_sizes_oop/bet_sizes_ip/raise_sizes/max_iterations。
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
CASES = [
    ("turn", ["Js", "8d", "6c", "6s"], 40.0),
    ("river", ["6h", "Ks", "5d", "8d", "5h"], 40.0),
    ("flop", ["Ks", "7h", "2c"], 40.0),
]


def _q(street: str, board: list[str], pot_bb: float) -> dict:
    return {"street": street, "board": board, "ranges": {"oop": OOP, "ip": IP},
             "pot": pot_bb, "effective_stack": 100.0, "bb": 1.0,
             "max_iterations": 300, "target_exploitability_pot_pct": 1.0}


async def main() -> None:
    rt = cli_mod.build_runtime()
    ctx = rt.new_context("lat", "x")
    prov = ctx.providers.get_solver()
    print("provider:", getattr(prov, "name", "?"))
    for street, board, pot_bb in CASES:
        t0 = time.time()
        try:
            res = await prov.solve(_q(street, board, pot_bb))
            el = time.time() - t0
            acts = [(a.name, a.size_bb, round(float(a.frequency), 3))
                    for a in (getattr(res, "actions", None) or [])][:4]
            print(f"{street:<6} {el*1000:>7.0f} ms | approx={res.is_approximate} "
                  f"| name={res.solver_name} | actions={acts}")
        except Exception as e:                              # noqa: BLE001
            print(f"{street:<6} 失败 {type(e).__name__}: {e}（{time.time()-t0:.1f}s）")
    t0 = time.time()
    try:
        await prov.solve(_q("turn", ["Js", "8d", "6c", "6s"], 40.0))
        print(f"二次（同 spot）: {(time.time()-t0)*1000:.0f} ms")
    except Exception as e:                                  # noqa: BLE001
        print("二次失败:", type(e).__name__)


asyncio.run(main())
