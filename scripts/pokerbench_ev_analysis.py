#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HD：PokerBench 对齐版结果的**EV 口径**分析（读已有解记录，不重解 ✓）。

动机：动作名一致率会**低估**策略质量 —— 若分歧行的 EV 差极小（近似无差异），
按"名字是否相同"计分并不公平 ✓。本脚本对每个样本取**标签动作**与**我们的 argmax**，
比较导出节点里两者的 `ev_bb`，报告：
  * 平均 EV 差（bb 与占底池比例 ✓）；
  * EV 差 < 0.02bb / < 0.10bb / < 0.25bb 的比例 ✓；
  * 我们 argmax 的 EV 是否**高于**标签动作（即标签在该节点并非最高 EV ✓）。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, r"<repo>")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"<repo>\scripts")
from pokerbench_validate_aligned import (load_rows, board_for, menu_for,   # noqa: E402
                                            NAME)

ROOT = pathlib.Path(r"<repo>")
OUT = ROOT / ".tmp" / "pbc_out"
DOC = ROOT / "docs" / "pokerbench_validation_aligned.json"


def main() -> None:
    doc = json.loads(DOC.read_text(encoding="utf-8"))
    rows_doc = doc.get("rows") or []
    rows = load_rows()
    cand = [r for r in rows
            if r["street"] in ("turn", "river") and len(board_for(r)) >= 4
            and r["hero"] in ("IP", "OOP") and r["label"]
            and ("fold" in " ".join(r["moves"]).lower()
                  or "call" in " ".join(r["moves"]).lower())]
    picks = cand[:len(rows_doc)]

    diffs, better, missing = [], 0, 0
    detail = []
    for i, r in enumerate(picks):
        f = OUT / f"PBC-{i:03d}.json"
        if not f.is_file():
            missing += 1
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        node = None
        for e in (rec.get("exports") or []):
            if str(e.get("actor") or "").upper() != r["hero"]:
                continue
            node = e
            break
        if node is None:
            missing += 1
            continue
        evs = {}
        for a in (node.get("actions") or []):
            nm = NAME.get(str(a.get("name")).lower(), str(a.get("name")))
            ev = a.get("ev_bb")
            if ev is not None:
                evs[nm] = float(ev)
        if r["label"] not in evs or not evs:
            missing += 1
            continue
        ours = max(evs.items(), key=lambda x: x[1])[0]
        diff = evs[ours] - evs[r["label"]]        # ≥0 ⇒ 我们的选择 EV 不低于标签 ✓
        diffs.append(diff)
        better += int(diff > 1e-9)
        detail.append({"board": "".join(board_for(r)), "street": r["street"],
                        "hero": r["hero"], "label": r["label"], "ours": ours,
                        "ev_label": round(evs[r["label"]], 4),
                        "ev_ours": round(evs[ours], 4), "diff_bb": round(diff, 4),
                        "pot": r["pot"]})

    n = len(diffs)
    if not n:
        print("无可用 EV 数据（缺解记录或标签动作不在导出动作集中）✗")
        return
    diffs_sorted = sorted(diffs)
    mean = sum(diffs) / n
    print(f"可评 {n} 行（缺 {missing}）")
    print(f"  平均 EV 差（我们 − 标签）: {mean:+.4f} bb")
    print(f"  我们 EV **不低于**标签的行数: {better}/{n} = {better/n*100:.0f}%")
    for thr in (0.02, 0.10, 0.25, 1.0):
        frac = sum(1 for d in diffs if abs(d) <= thr) / n
        print(f"  |EV 差| ≤ {thr:<5} bb 的比例: {frac*100:.0f}%")
    print(f"  中位 EV 差: {diffs_sorted[n//2]:+.4f} bb | 最大: {diffs_sorted[-1]:+.4f} bb")
    print("\n样例（前 6）:")
    for d in detail[:6]:
        print(f"   {d['street']:<6} {d['board']:<10} hero={d['hero']:<4} 标签={d['label']:<6} "
              f"我们={d['ours']:<6} EV标签={d['ev_label']:<9} EV我们={d['ev_ours']:<9} "
              f"差={d['diff_bb']:+.4f}bb")

    (ROOT / "docs" / "pokerbench_ev_analysis.json").write_text(json.dumps({
        "n": n, "missing": missing,
        "mean_ev_diff_bb": round(mean, 4),
        "share_ours_not_worse": round(better / n, 4),
        "share_within_bb": {str(t): round(sum(1 for d in diffs if abs(d) <= t) / n, 4)
                             for t in (0.02, 0.10, 0.25, 1.0)},
        "median_ev_diff_bb": round(diffs_sorted[n // 2], 4),
        "note": ("EV 取自我们树的导出节点（尺寸已按行对齐 ✓）；范围仍为声明范围 ✗ ⇒ "
                  "该 EV 是**我们框架内**的 EV，不是真值 ✗"),
        "rows": detail}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/pokerbench_ev_analysis.json")


if __name__ == "__main__":
    main()
