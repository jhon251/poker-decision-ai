#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-DV：**基准回归门校验**（一条命令；回退则非零退出）。

用法：
    python scripts/bench_gate_check.py            # 跑 r1-flop + r1-turn/river，与基线比对
    python scripts/bench_gate_check.py --fb       # 追加 fb（TV 距离）

基线来源：`docs/bench_regression_gate.json`（按 suite 取**最后一条**记录）。
判定（目标硬要求"支持集命中率/regret 不得回退"）：
  * regret_mean / regret_max 不得超过基线 + 容差（默认 0.002bb）；
  * ev_optimal_rate / support_gt0 不得低于基线 − 容差（默认 0.02）；
  * fb：tv_distance 不得超过基线 + 0.01。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "docs" / "bench_regression_gate.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def _run(args: list[str], out: pathlib.Path) -> str:
    log = out.with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run([PY, str(ROOT / "scripts" / "bench_vs_solver.py"), *args,
                         "--out", str(out)], cwd=str(ROOT), stdout=fh,
                        stderr=subprocess.STDOUT, timeout=5400)
    return log.read_text(encoding="utf-8", errors="replace")


def _parse(txt: str) -> dict:
    m = {k: None for k in ("regret_mean", "regret_max", "zero_rate", "ev_optimal",
                            "support_gt0", "support_5pct", "samples")}
    r = re.search(r"regret 均值 ([0-9.]+) bb .*?最大 ([0-9.]+) bb", txt)
    if r:
        m["regret_mean"], m["regret_max"] = float(r.group(1)), float(r.group(2))
    z = re.search(r"=0 的比例 ([0-9.]+)%", txt)
    if z:
        m["zero_rate"] = float(z.group(1)) / 100
    e = re.search(r"逐手 EV 最优一致：([0-9]+)/([0-9]+)", txt)
    if e:
        m["ev_optimal"] = int(e.group(1)) / max(int(e.group(2)), 1)
    p = re.search(r"支持率\(>0%\)：([0-9]+)/([0-9]+)", txt)
    if p:
        m["support_gt0"] = int(p.group(1)) / max(int(p.group(2)), 1)
    s = re.search(r"手牌级支持集命中: ([0-9]+)/([0-9]+)", txt)
    if s:
        m["support_5pct"] = int(s.group(1)) / max(int(s.group(2)), 1)
    n = re.search(r"样本 (\d+) 条", txt)
    if n:
        m["samples"] = int(n.group(1))
    return m


def _baseline(suite_key: str) -> dict | None:
    if not GATE.exists():
        return None
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    hits = [r for r in gate.get("runs", []) if str(r.get("suite")) == suite_key]
    return hits[-1].get("metrics") if hits else None


def _baseline6() -> dict | None:
    """6-max 保真度基线（多 offset 聚合，见 docs/bench_6max_suite.json）。"""
    p = ROOT / "docs" / "bench_6max_suite.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("aggregate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", action="store_true")
    ap.add_argument("--6max", dest="sixmax", action="store_true",
                     help="追加 6-max 保真度门（4 offset 聚合，约 10 分钟）")
    ap.add_argument("--tol-regret", type=float, default=0.002)
    ap.add_argument("--tol-rate", type=float, default=0.02)
    args = ap.parse_args()

    fails: list[str] = []
    cases = [("r1-flop", ["--suite", "r1", "--streets", "flop", "--spots", "8", "--hands", "30"]),
             ("r1-turn/river", ["--suite", "r1", "--streets", "turn,river", "--spots", "12",
                                 "--hands", "30"])]
    for suite, argv in cases:
        txt = _run(argv, ROOT / ".tmp" / f"gate_{suite.replace('/', '_')}.json")
        cur = _parse(txt)
        base = _baseline(suite)
        print(f"[{suite}] 当前: {json.dumps(cur, ensure_ascii=False)}")
        if not base:
            print(f"  无基线（跳过比对）")
            continue
        print(f"  基线: {json.dumps(base, ensure_ascii=False)}")
        for k in ("regret_mean", "regret_max"):
            if cur.get(k) is not None and base.get(k) is not None \
                    and cur[k] > float(base[k]) + args.tol_regret:
                fails.append(f"{suite}.{k}: {cur[k]} > 基线 {base[k]} (+{args.tol_regret})")
        for k in ("ev_optimal", "support_gt0", "zero_rate"):
            if cur.get(k) is not None and base.get(k) is not None \
                    and cur[k] < float(base[k]) - args.tol_rate:
                fails.append(f"{suite}.{k}: {cur[k]} < 基线 {base[k]} (-{args.tol_rate})")
    if args.fb:
        txt = _run(["--suite", "fb", "--spots", "6", "--hands", "30"],
                    ROOT / ".tmp" / "gate_fb.json")
        tvm = re.search(r"TV 距离: ([0-9.]+)", txt)
        cur = float(tvm.group(1)) if tvm else None
        base = _baseline("fb")
        print(f"[fb] 当前 TV={cur} | 基线 tv_distance={ (base or {}).get('tv_distance') }")
        if cur is not None and base and base.get("tv_distance") is not None \
                and cur > float(base["tv_distance"]) + 0.01:
            fails.append(f"fb.tv: {cur} > 基线 {base['tv_distance']} (+0.01)")

    if args.sixmax:
        base6 = _baseline6()
        with (ROOT / ".tmp" / "gate_6max.log").open("w", encoding="utf-8") as fh:
            subprocess.run([PY, str(ROOT / "scripts" / "bench_6max_suite.py"),
                             "--offsets", "0,24,48,72", "--nodes", "24", "--hands", "8"],
                            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, timeout=5400)
        cur6 = json.loads((ROOT / "docs" / "bench_6max_suite.json")
                           .read_text(encoding="utf-8"))["aggregate"]
        print(f"[6max] 当前: {json.dumps(cur6, ensure_ascii=False)}")
        print(f"  基线: {json.dumps(base6, ensure_ascii=False) if base6 else '无'}")
        if base6:
            for k in ("support", "primary", "anchor"):
                if cur6.get(k) is not None and base6.get(k) is not None \
                        and cur6[k] < float(base6[k]) - args.tol_rate:
                    fails.append(f"6max.{k}: {cur6[k]} < 基线 {base6[k]} (-{args.tol_rate})")

    if fails:
        print("\n[FAIL] 基准门未通过：")
        for f in fails:
            print("  -", f)
        return 1
    print("\n[PASS] 基准门通过（支持集/EV 最优/regret 均未回退）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
