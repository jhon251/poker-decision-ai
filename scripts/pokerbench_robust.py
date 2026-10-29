#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HV：PokerBench 独立验证之**范围稳健性口径**（免费、可解释 ✓）。

问题：此前的 18/40 = 45% **不可解释** ✗ —— 我们的解用的是**声明范围**，
而 PokerBench 不提供范围 ✗（最关键混淆项 ✓）。

做法：对同一局面用**3 套不同范围**各解一次（都按该行实际下注尺寸 ✓）：
  * `base`   = 我们声明的范围 ✓
  * `tight`  = 更紧（去掉弱牌）✓
  * `wide`   = 更宽（加入更多同花/连张）✓
只有当**三套范围给出同一动作**时，才认为该局面的决策是**稳健的** ✓，再与标签比对 ✓。
⇒ 报告"**稳健子集上的一致率**" ✓（把范围敏感性显式量化 ✓，而不是掩盖 ✗）。
仍**不是**能力证明 ✓（标签来自其自身抽象 ✓），但它把"范围"这一主导混淆项**显式化** ✓。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
from pokerbench_validate_aligned import (NAME, board_for, load_rows,   # noqa: E402
                                          menu_for)

VARIANTS = {
    "base": None,                                  # 用脚本内置声明范围 ✓
    "tight": {"3bp": {"oop": "QQ+,JJ,AKs,AQs,AKo",
                       "ip": "TT-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,AQo,KQo"},
               "srp": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,76s,AQo",
                        "ip": ("22+,A2s+,KTs+,QTs+,JTs,T9s,98s,87s,AQo,KQo,QJo")}},
    "wide": {"3bp": {"oop": ("QQ+,JJ,TT,AKs,AQs,AJs,KQs,JTs,T9s,98s,87s,"
                              "AKo,AQo,KQo"),
                      "ip": ("TT-22,AQs-A2s,KQs-K9s,QJs-Q9s,JTs-J8s,T9s-T8s,"
                              "98s-97s,87s-86s,76s,65s,54s,AQo,KQo,QJo,JTo,T9o")},
              "srp": {"oop": ("QQ-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs-J9s,T9s-T8s,"
                               "98s-97s,87s-86s,76s,65s,AQo,KQo,QJo"),
                       "ip": ("22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,"
                               "A2o+,K8o+,Q9o+,J9o+,T9o,98o,87o")}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--streets", default="turn,river")
    args = ap.parse_args()

    rows = load_rows()
    cand = [r for r in rows
            if r["street"] in args.streets.split(",") and len(board_for(r)) >= 4
            and r["hero"] in ("IP", "OOP") and r["label"]
            and ("fold" in " ".join(r["moves"]).lower()
                  or "call" in " ".join(r["moves"]).lower())]
    picks = cand[:args.n]
    print(f"PokerBench 行 {len(rows)} | 可评 {len(cand)} | 取 {len(picks)}"
          f" × 范围变体 {list(VARIANTS)}")

    # 每行 × 每变体 ⇒ 一个 spot（同一牌面/尺寸，不同范围 ✓）
    spots, index = [], []
    for i, r in enumerate(picks):
        sizes, frac = menu_for(r)
        for vname, vmap in VARIANTS.items():
            fam = "3bp" if r["preflop"].count("bb") >= 2 else "srp"
            ranges = (vmap[fam] if vmap else None)
            sid = f"PBR-{i:03d}-{vname}"
            spot = {"spot_id": sid, "street": r["street"], "board": board_for(r),
                     "pot_bb": r["pot"], "effective_stack_bb": 100.0,
                     "bet_sizes": sizes, "bet_sizes_ip": sizes,
                     "export_nodes": [[1], [2], [0, 1]],
                     "exports": [{"export_id": "b1", "path": [1]},
                                  {"export_id": "b2", "path": [2]},
                                  {"export_id": "cb", "path": [0, 1]}],
                     "max_iterations": 300, "target_exploitability_pot_pct": 1.0}
            if ranges:
                spot["ranges"] = dict(ranges)
            else:                                   # base：用已声明范围 ✓
                from pokerbench_validate_aligned import RANGES
                spot["ranges"] = dict(RANGES[fam])
            spots.append(spot)
            index.append((i, vname))
    cfg = ROOT / ".tmp" / "pbr.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"spot_suite": {
        "name": "POKERBENCH-range-robustness", "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver", "n_spots": len(spots)}, "spots": spots},
        ensure_ascii=False, indent=1), encoding="utf-8")
    out = ROOT / ".tmp" / "pbr_out"
    shutil.rmtree(out, ignore_errors=True)
    print(f"求解 {len(spots)} 个（{len(picks)} 局面 × {len(VARIANTS)} 范围）…")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
                         "--config", str(cfg), "--out-dir", str(out),
                         "--workers", "1", "--solver-threads", "4"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=10800)
    print("   ", " | ".join(x.strip() for x in (p.stdout or "").splitlines()
                             if "n_solved" in x or "n_failed" in x))

    per: dict[int, dict] = collections.defaultdict(dict)
    for (i, vname), spot in zip(index, spots):
        f = out / f"{spot['spot_id']}.json"
        if not f.is_file():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        hero = picks[i]["hero"]
        for e in (rec.get("exports") or []):
            if str(e.get("actor") or "").upper() != hero:
                continue
            acts = [(NAME.get(str(a.get("name")).lower(), str(a.get("name"))),
                      float(a.get("frequency") or 0)) for a in (e.get("actions") or [])]
            if acts:
                per[i][vname] = max(acts, key=lambda x: x[1])[0]
            break

    robust, agree = 0, 0
    rows_out = []
    for i, r in enumerate(picks):
        got = per.get(i) or {}
        if len(got) < len(VARIANTS):
            rows_out.append({"board": "".join(board_for(r)), "street": r["street"],
                              "label": r["label"], "variants": got, "robust": False,
                              "reason": "解不全"})
            continue
        vals = set(got.values())
        is_robust = len(vals) == 1
        robust += int(is_robust)
        ok = is_robust and (next(iter(vals)) == r["label"])
        agree += int(ok)
        rows_out.append({"board": "".join(board_for(r)), "street": r["street"],
                          "label": r["label"], "variants": got, "robust": is_robust,
                          "agree": ok})
    n = len(picks)
    print(f"\n范围稳健性：三套范围**动作一致** {robust}/{n} = {robust / max(n, 1) * 100:.0f}%")
    print(f"  稳健子集上与标签一致：**{agree}/{robust}**"
          + (f" = {agree / robust * 100:.0f}%" if robust else "（无稳健样本 ✗）"))
    lab = collections.Counter(r["label"] for r in picks)
    print(f"  标签分布 {dict(lab)}")

    # P30-9-HX：**统计区间**（Wilson）——小样本数字否则不可解释 ✗
    def _wilson(k: int, m: int, z: float = 1.96) -> tuple[float, float] | None:
        if m <= 0:
            return None
        p = k / m
        d = 1 + z * z / m
        c = (p + z * z / (2 * m)) / d
        h = z * ((p * (1 - p) / m + z * z / (4 * m * m)) ** 0.5) / d
        return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))

    # 基线：只看标签分布，恒出出现最多的动作（在**稳健子集**上评估 ✓）
    robust_idxs = [i for i, r in enumerate(rows_out) if r.get("robust")]
    lab_robust = collections.Counter(picks[i]["label"] for i in robust_idxs)
    maj_lab = lab_robust.most_common(1)[0][0] if lab_robust else None
    base_hits = sum(1 for i in robust_idxs if picks[i]["label"] == maj_lab)
    ci_agree = _wilson(agree, robust)
    ci_base = _wilson(base_hits, len(robust_idxs))
    print(f"  一致率 Wilson 95% CI: {ci_agree}")
    print(f"  多数类基线（稳健子集上恒出 '{maj_lab}'）: {base_hits}/{len(robust_idxs)}"
          + (f" = {base_hits / max(len(robust_idxs), 1) * 100:.0f}%" if robust_idxs else "")
          + f" | CI {ci_base}")
    if ci_agree and ci_base:
        _overlap = not (ci_agree[1] < ci_base[0] or ci_base[1] < ci_agree[0])
        print(f"  两者区间{'重叠 ⇒ 统计上不可区分 ✓' if _overlap else '不重叠 ⇒ 有统计差异 ✓'}")

    (ROOT / "docs" / "pokerbench_range_robustness.json").write_text(json.dumps({
        "dataset": "RZ412/PokerBench（Apache-2.0）postflop 10k test",
        "method": "同一局面用 3 套范围各解一次（尺寸按行对齐 ✓）；三套动作一致 = 稳健 ✓",
        "n": n, "robust": robust, "robust_rate": round(robust / max(n, 1), 4),
        "robust_rate_ci95": _wilson(robust, n),
        "agree_on_robust": agree,
        "agree_rate_on_robust": (round(agree / robust, 4) if robust else None),
        "agree_rate_on_robust_ci95": ci_agree,
        "majority_baseline_on_robust": {
            "action": maj_lab, "hits": base_hits, "n": len(robust_idxs),
            "rate": (round(base_hits / len(robust_idxs), 4) if robust_idxs else None),
            "ci95": ci_base,
            "ci_overlaps_ours": (None if not (ci_agree and ci_base) else
                                  not (ci_agree[1] < ci_base[0] or ci_base[1] < ci_agree[0]))},
        "caveats": ["仍非能力证明：标签来自其自身抽象 ✓",
                     "有效栈按 100bb 平铺 ✓",
                     "范围变体是**我们的**构造（非其真实范围 ✗）⇒ 只衡量**范围敏感性** ✓",
                     "抽样非官方评测口径 ✓"],
        "rows": rows_out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/pokerbench_range_robustness.json")


if __name__ == "__main__":
    main()
