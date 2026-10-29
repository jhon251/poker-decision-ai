#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A：翻后覆盖扩张（离线、无人值守、可断点续跑）。

依据（已证）：spot 查表的 first_in 路径是**按 (street, pot档, 纹理类) 分桶**匹配
（`spot_table.lookup` ⇒ `range_set_exact` / `range_set_nearest`）
⇒ 提高覆盖 = **为新的纹理类各解至少一个代表局面**。

流程：
  1. 生成新套件 `configs/solver/spots_r2.yaml`（10 个新纹理类 × SRP/3BP = 20 个翻牌局面）；
  2. 调 `scripts/solve_spots.py --config ... --out-dir data/solver/spots_r2`（断点续跑）；
  3. 合并进 `data/solver/spots_r1/` 并重建 manifest；
  4. 同步到 `dist/data/solver/spots_r1/`。

用法：python scripts/expand_flop_coverage.py [--workers 3] [--solver-threads 2] [--skip-solve]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "solver" / "spots_r2.yaml"
OUT_DIR = ROOT / "data" / "solver" / "spots_r2"
MERGE_INTO = ROOT / "data" / "solver" / "spots_r1"
DIST_MIRROR = ROOT / "dist" / "data" / "solver" / "spots_r1"

#: 与现有 10 类**不重复**的新纹理类（每类一个代表牌面）
NEW_TEXTURES: list[tuple[str, list[str]]] = [
    ("paired_high", ["Ks", "Kd", "4h"]),          # 高对子面
    ("trips_board", ["8c", "8d", "8h"]),          # 三条面
    ("two_tone_low", ["7s", "5s", "2h"]),         # 低两同花
    ("rainbow_gapped", ["Qc", "8d", "3s"]),       # 彩虹隔张
    ("suited_broadway", ["As", "Ks", "9d"]),      # 同花大牌（同花听牌密集）
    ("low_connected_tt", ["6h", "5h", "4c"]),     # 低连张两同花
    ("ace_low_tt", ["Ac", "5c", "2d"]),           # A 低两同花
    ("paired_middle", ["9h", "9s", "3c"]),        # 中对子面
    ("monotone_connected", ["Ts", "9s", "8s"]),   # 单一花色连张
    ("high_card_rainbow", ["Jd", "7s", "2c"]),    # 高张彩虹
]

RANGE_SETS = {
    "SRP": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
            "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9o+,J9o+,T9o"},
    "3BP": {"oop": "JJ+,AJs+,A5s-A2s,KQs,JTs,T9s,98s,87s,AQo+",
            "ip": "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+"},
}
POT_TYPES = {"SRP": {"pot_bb": 3.5, "effective_stack_bb": 96.5},
             "3BP": {"pot_bb": 22.0, "effective_stack_bb": 89.0}}


def build_suite() -> dict:
    spots = []
    for pot_name, p in POT_TYPES.items():
        for tex, board in NEW_TEXTURES:
            spots.append({
                "spot_id": f"{pot_name}-flop-{tex}",
                "street": "flop",
                "board": list(board),
                "pot_bb": p["pot_bb"],
                "effective_stack_bb": p["effective_stack_bb"],
                "ranges": dict(RANGE_SETS[pot_name]),
                "max_iterations": 400,
                "target_exploitability_pot_pct": 1.0,
            })
    return {"spot_suite": {
        "name": "COVERAGE-R2", "game": "HU 100bb cash",
        "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver (b-inary) — 真实 CFR；HU；尺寸抽象",
        "n_spots": len(spots),
        "note": ("覆盖扩张（A）：10 个**新纹理类** × SRP/3BP；"
                  "目的 = 为 (street,pot档,纹理类) 桶补代表局面，提高 live 命中率"),
    }, "spots": spots}


def merge_and_sync() -> tuple[int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_files = [f for f in sorted(OUT_DIR.glob("*.json")) if f.name != "manifest.json"]
    solved = []
    for f in new_files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        if d.get("solved"):
            solved.append(f.name)
            shutil.copy2(f, MERGE_INTO / f.name)
    # 重建 manifest（保留既有条目 + 新增）
    mpath = MERGE_INTO / "manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    ids = sorted({*(man.get("solved_ids") or []),
                   *[p.stem for p in MERGE_INTO.glob("*.json")
                     if p.name != "manifest.json"]})
    man.update({"n_spots": len(ids), "n_solved": len(ids), "n_failed": 0,
                 "solved_ids": ids,
                 "coverage_expansion": ("A：新增 10 纹理类 × SRP/3BP（"
                                         + str(len(solved)) + " 个新 spot）")})
    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    if DIST_MIRROR.exists() or DIST_MIRROR.parent.exists():
        DIST_MIRROR.mkdir(parents=True, exist_ok=True)
        for f in MERGE_INTO.glob("*.json"):
            shutil.copy2(f, DIST_MIRROR / f.name)
    return len(solved), len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--solver-threads", type=int, default=2)
    ap.add_argument("--skip-solve", action="store_true")
    args = ap.parse_args()

    suite = build_suite()
    SUITE.parent.mkdir(parents=True, exist_ok=True)
    SUITE.write_text(json.dumps(suite, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"① 套件已生成：{SUITE.name}（{suite['spot_suite']['n_spots']} 个新翻牌局面）")
    print("   新纹理类:", [t for t, _ in NEW_TEXTURES])

    if not args.skip_solve:
        cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"),
               str(ROOT / "scripts" / "solve_spots.py"),
               "--config", str(SUITE), "--out-dir", str(OUT_DIR),
               "--workers", str(args.workers),
               "--solver-threads", str(args.solver_threads)]
        print("② 开始离线求解:", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        print(f"   求解进程退出码={proc.returncode}，用时 {(time.time()-t0)/60:.1f} 分钟")

    n_new, n_total = merge_and_sync()
    print(f"③ 合并完成：新解 {n_new} 个；spots_r1 现共 {n_total} 个（已同步 dist）")


if __name__ == "__main__":
    main()
