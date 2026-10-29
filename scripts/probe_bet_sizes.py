"""P30-9-HB：判定 CLI **是否采纳 `bet_sizes`**（走 solve_spots_fb.py 路径 ✓）。

做法：同一个 spot（river，pot 20，栈 100）用两组 `bet_sizes` 各解一次，比较：
  * 导出节点的 `bet_ratio` 与动作 `amount_bb`（若不同 ⇒ 尺寸被采纳 ✓；若相同 ⇒ 未采纳 ✗）。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")
R = {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
      "ip": "22+,A2s+,K2s+,Q4s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9s+,J9o+,T9o"}
BOARD = ["Qc", "Ad", "Kh", "Ts", "Tc"]


def one(tag: str, sizes: str) -> dict:
    spots = [{"spot_id": f"PROBE-{tag}", "street": "river", "board": BOARD,
               "pot_bb": 20.0, "effective_stack_bb": 100.0, "ranges": dict(R),
               "bet_sizes": sizes, "bet_sizes_ip": sizes,
               "export_nodes": [[1], [2], [0, 1]],
               "exports": [{"export_id": "a", "path": [1]},
                            {"export_id": "b", "path": [2]},
                            {"export_id": "c", "path": [0, 1]}],
               "max_iterations": 300, "target_exploitability_pot_pct": 1.0}]
    cfg = ROOT / ".tmp" / f"probe_bs_{tag}.yaml"
    cfg.write_text(json.dumps({"spot_suite": {"name": f"probe-{tag}",
                                               "cli": "tools/bin/postflop-bench-cli.exe",
                                               "solver": "postflop-solver",
                                               "n_spots": 1}, "spots": spots},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    out = ROOT / ".tmp" / f"probe_bs_out_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
                         "--config", str(cfg), "--out-dir", str(out),
                         "--workers", "1", "--solver-threads", "4"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=1800)
    f = out / f"PROBE-{tag}.json"
    if not f.is_file():
        return {"error": (p.stdout or "")[-200:] + (p.stderr or "")[-200:]}
    rec = json.loads(f.read_text(encoding="utf-8"))
    return {"bet_ratio": rec.get("export_decl_bet_ratio") if "export_decl_bet_ratio" in rec else None,
             "exports": [{"actor": e.get("actor"), "bet_ratio": e.get("bet_ratio"),
                           "amounts": [a.get("amount_bb") for a in (e.get("actions") or [])]}
                          for e in (rec.get("exports") or [])],
             "root_amounts": [a.get("amount_bb") for a in (rec.get("root_actions") or [])]}


def main() -> None:
    for tag, sizes in (("small", "29%,a"), ("large", "75%,a")):
        r = one(tag, sizes)
        ex = r.get("exports") or []
        detail = " ; ".join(
            f"{e['actor']} ratio={e['bet_ratio']} amounts={e['amounts']}" for e in ex)
        print(f"bet_sizes={sizes:<8} → 导出数={len(ex)} | {detail}")
        if not ex:
            print("    原始:", str(r)[:200])


if __name__ == "__main__":
    main()
