#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HE：为**真实会话里的小注 facing 局面**补解并入库（此前判为不可解 ✗ → 现可解 ✓）。

依据（P30-9-HB 实测）：CLI **采纳 `bet_sizes`** ✓ ⇒ 可按实际下注比例构造菜单 ✓。

两个目标局面（来自真实日志，`real_coverage_gate --list-miss` ✓）：
  1) `Qh7d2c`  SB vs BTN  facing  pot 40.4  bet_ratio ≈ 0.1023（小注）
  2) `2h2dTh`  SB vs UTG  facing  pot 3.0   bet_ratio ≈ 0.1765（小注）

做法：解一棵 flop 树（IP 下注尺寸 = 目标比例 ✓），导出「OOP 过牌 → IP 下注」的 facing 节点
（`history=[0,1]`，`actor=oop` ✓），转成 6-max 表 schema（`node=facing_bet` ✓）并入库 ✓。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
R3 = {"oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",
       "ip": "TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo"}
RS = {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
       "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9s+,J9o+,T9o"}

CASES = [
    {"tag": "REALSB1", "family": "3bp", "board": ["Qh", "7d", "2c"], "pot": 40.4,
      "stack": 111.26, "oop": "SB", "ip": "BTN", "ratio": 0.1023, "ranges": R3},
    {"tag": "REALSB2", "family": "srp", "board": ["2h", "2d", "Th"], "pot": 3.0,
      "stack": 48.13, "oop": "SB", "ip": "UTG", "ratio": 0.1765, "ranges": RS},
]


def main() -> None:
    spots = []
    for c in CASES:
        pct = max(5, int(round(c["ratio"] * 100)))         # 目标下注比例（%）✓
        spots.append({"spot_id": c["tag"], "street": "flop", "board": c["board"],
                       "pot_bb": c["pot"], "effective_stack_bb": c["stack"],
                       "ranges": dict(c["ranges"]),
                       "bet_sizes": f"{pct}%,a", "bet_sizes_ip": f"{pct}%,a",
                       "export_nodes": [[0, 1]],
                       "exports": [{"export_id": "facing", "path": [0, 1]}],
                       "max_iterations": 300, "target_exploitability_pot_pct": 1.0})
        c["pct"] = pct
    cfg = ROOT / ".tmp" / "realsb.yaml"
    cfg.write_text(json.dumps({"spot_suite": {"name": "real-smallbet-facing",
                                               "cli": "tools/bin/postflop-bench-cli.exe",
                                               "solver": "postflop-solver", "n_spots": len(spots)},
                                "spots": spots}, ensure_ascii=False, indent=1), encoding="utf-8")
    out = ROOT / ".tmp" / "realsb_out"
    shutil.rmtree(out, ignore_errors=True)
    print(f"求解中（菜单 = 各局面实际下注比例）…")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
                         "--config", str(cfg), "--out-dir", str(out),
                         "--workers", "1", "--solver-threads", "4"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=7200)
    print("   ", " | ".join(x.strip() for x in (p.stdout or "").splitlines()
                             if "n_solved" in x or "n_failed" in x))

    dst = ROOT / "data" / "solver" / "spots_6max"
    written = 0
    for c in CASES:
        f = out / f"{c['tag']}.json"
        if not f.is_file():
            print(f"  {c['tag']}: 无解记录 ✗"); continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        ex = next((e for e in (rec.get("exports") or [])
                    if str(e.get("actor") or "").lower() == c["oop"].lower().lower()
                    or str(e.get("actor") or "").lower() == "oop"), None)
        if ex is None:
            print(f"  {c['tag']}: 无 oop facing 导出 ✗"); continue
        actions = [{"name": str(a.get("name") or "").lower(),
                     "size_bb": a.get("amount_bb"),
                     "frequency": round(float(a.get("frequency") or 0), 4)}
                    for a in (ex.get("actions") or [])]
        sid = f"{c['family']}-ourface-{''.join(c['board'])}"
        rec6 = {"spot_id": sid, "street": "flop", "board": list(c["board"]),
                 "pot_bb": round(float(c["pot"]), 3),
                 "effective_stack_bb": round(float(c["stack"]), 3),
                 "positions": {"oop": c["oop"].upper(), "ip": c["ip"].upper()},
                 "actor_position": c["oop"].upper(), "node": "facing_bet",
                 "to_call_bb": ex.get("to_call_bb"),
                 "bet_ratio": ex.get("bet_ratio"),
                 "actions": actions, "private_cards_actor": [], "freq_per_hand": {},
                 "solved": True, "source": "our_hu_solve_facing",
                 "source_note": (f"真实小注 facing 补解（菜单 {c['pct']}%，实际 bet_ratio≈{c['ratio']}；"
                                  "范围=我们声明范围，假设如实标注）"),
                 "agent": "real_smallbet_facing_v1", "wall_sec": rec.get("wall_sec")}
        (dst / f"{sid}.json").write_text(json.dumps(rec6, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        written += 1
        print(f"  写入 {sid} | bet_ratio={rec6['bet_ratio']} to_call={rec6['to_call_bb']} "
              f"| actions={[(a['name'], a['frequency']) for a in actions]}")
    print(f"共写入 {written} 条 ⇒ {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
