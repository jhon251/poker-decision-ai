#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HC：PokerBench 独立验证（**逐行对齐下注尺寸**版）。

关键解锁（P30-9-HB 实测）：CLI **确实采纳 `bet_sizes`** ✓
（29%,a ⇒ 面对点 bet_ratio=0.29；75%,a ⇒ 0.75 ✓）。
⇒ 可**按每一行的实际下注尺寸**构造求解菜单 ⇒ 消除"尺寸不同导致底池赔率不同"这一主导混淆 ✓。

尺寸来源（按优先级）：
  1) 标签是 `Raise X`：X 为**加注到**的筹码数 ⇒ 以 pot 为基准折算比例 ✓；
  2) 否则取 `postflop_action` 里 hero 决策前**最后一次下注/加注**额 ✓；
  3) 都拿不到 ⇒ 用默认菜单（并标记为未对齐 ✓）。
"""
from __future__ import annotations

import argparse
import ast as _ast
import collections
import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "external" / "pokerbench" / "postflop_10k_test_set_game_scenario_information.csv"
RANGES = {
    "3bp": {"oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",
             "ip": "TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo"},
    "srp": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
             "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9s+,J9o+,T9o"},
}
NAME = {"fold": "Fold", "call": "Call", "raise": "Raise", "check": "Check",
         "bet": "Bet", "allin": "AllIn"}


def load_rows() -> list[dict]:
    out = []
    with CSV.open(encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                moves = [str(m) for m in _ast.literal_eval(r.get("available_moves") or "[]")]
            except Exception:                                      # noqa: BLE001
                moves = []
            out.append({"flop": r.get("board_flop") or "", "turn": r.get("board_turn") or "",
                         "river": r.get("board_river") or "",
                         "street": str(r.get("evaluation_at") or "").lower(),
                         "pot": float(r.get("pot_size") or 0),
                         "hero": str(r.get("hero_position") or "").upper(),
                         "holding": str(r.get("holding") or ""), "moves": moves,
                         "label_full": str(r.get("correct_decision") or ""),
                         "label": str(r.get("correct_decision") or "").split()[0],
                         "preflop": r.get("preflop_action") or "",
                         "line": str(r.get("postflop_action") or "")})
    return out


def board_for(r: dict) -> list[str]:
    b = [r["flop"][i:i + 2] for i in range(0, len(r["flop"]), 2)]
    if r["street"] in ("turn", "river") and r["turn"]:
        b.append(r["turn"])
    if r["street"] == "river" and r["river"]:
        b.append(r["river"])
    return [c for c in b if len(c) == 2]


def family_of(pre: str) -> str:
    segs = [s for s in str(pre or "").split("/") if s]
    return "3bp" if sum(1 for s in segs if s.endswith("bb")) >= 2 else "srp"


def faced_amount(r: dict) -> float | None:
    """hero 决策前最后一次下注/加注额（筹码）——用于折算尺寸比例 ✓。"""
    acts = [a for a in str(r["line"]).split("/") if a and a != "dealcards"]
    for a in reversed(acts):
        m = re.match(r"^(OOP|IP)_(BET|RAISE|ALLIN)_(\d+)$", a.strip().upper())
        if m:
            return float(m.group(3))
    return None


def menu_for(r: dict) -> tuple[str, float | None]:
    """返回 (bet_sizes 菜单, 折算比例)。比例以 pot 为基准 ✓（未对齐则 None ✓）。"""
    pot = max(r["pot"], 1e-9)
    amt = None
    if r["label"].lower() == "raise":
        m = re.search(r"(\d+)\s*$", r["label_full"])
        if m:
            amt = float(m.group(1))
    if amt is None:
        amt = faced_amount(r)
    if amt is None:
        return "60%,a", None
    frac = max(0.05, min(2.0, amt / pot))
    return f"{max(5, int(round(frac * 100)))}%,a", frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--streets", default="turn,river")
    args = ap.parse_args()

    rows = load_rows()
    cand = [r for r in rows
            if r["street"] in args.streets.split(",") and len(board_for(r)) >= 4
            and r["hero"] in ("IP", "OOP") and r["label"]
            and ("fold" in " ".join(r["moves"]).lower() or "call" in " ".join(r["moves"]).lower())]
    picks = cand[:args.n]
    print(f"PokerBench 行 {len(rows)} | 可评 {len(cand)} | 取 {len(picks)}（逐行对齐尺寸 ✓）")

    spots = []
    for i, r in enumerate(picks):
        sizes, frac = menu_for(r)
        spots.append({"spot_id": f"PBC-{i:03d}", "street": r["street"],
                       "board": board_for(r), "pot_bb": r["pot"],
                       "effective_stack_bb": 100.0,
                       "ranges": dict(RANGES[family_of(r["preflop"])]),
                       "bet_sizes": sizes, "bet_sizes_ip": sizes,
                       "export_nodes": [[1], [2], [0, 1]],
                       "exports": [{"export_id": "b1", "path": [1]},
                                    {"export_id": "b2", "path": [2]},
                                    {"export_id": "cb", "path": [0, 1]}],
                       "max_iterations": 300, "target_exploitability_pot_pct": 1.0})
        picks[i]["_sizes"] = sizes
        picks[i]["_frac"] = frac
    cfg = ROOT / ".tmp" / "pbc.yaml"
    cfg.write_text(json.dumps({"spot_suite": {"name": "POKERBENCH-aligned",
                                               "cli": "tools/bin/postflop-bench-cli.exe",
                                               "solver": "postflop-solver", "n_spots": len(spots)},
                                "spots": spots}, ensure_ascii=False, indent=1), encoding="utf-8")
    out_dir = ROOT / ".tmp" / "pbc_out"
    shutil.rmtree(out_dir, ignore_errors=True)
    print("求解中（逐行不同尺寸菜单）…")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
                         "--config", str(cfg), "--out-dir", str(out_dir),
                         "--workers", "1", "--solver-threads", "4"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=7200)
    print("   ", " | ".join(x.strip() for x in (p.stdout or "").splitlines()
                             if "n_solved" in x or "n_failed" in x))

    agree = 0
    labels, ours_dist, aligned = collections.Counter(), collections.Counter(), 0
    res = []
    for i, r in enumerate(picks):
        f = out_dir / f"{spots[i]['spot_id']}.json"
        pick, best = None, None
        if f.is_file():
            rec = json.loads(f.read_text(encoding="utf-8"))
            for e in (rec.get("exports") or []):
                if str(e.get("actor") or "").upper() != r["hero"]:
                    continue
                # 选与该行折算比例最接近的导出（对齐尺寸 ✓）
                ratio = e.get("bet_ratio")
                d = abs(float(ratio) - float(r["_frac"])) if (ratio is not None and r["_frac"]) else 9.0
                acts = [(NAME.get(str(a.get("name")).lower(), str(a.get("name"))),
                          float(a.get("frequency") or 0)) for a in (e.get("actions") or [])]
                if not acts:
                    continue
                cand_name = max(acts, key=lambda x: x[1])[0]
                if best is None or d < best:
                    best, pick = d, cand_name
                    if ratio is not None and r["_frac"]:
                        res_ratio = float(ratio)
        if r["_frac"]:
            aligned += 1
        ok = pick == r["label"]
        agree += int(ok)
        labels[r["label"]] += 1
        ours_dist[pick or "(无)"] += 1
        res.append({"board": "".join(board_for(r)), "street": r["street"], "pot": r["pot"],
                     "hero": r["hero"], "holding": r["holding"], "moves": r["moves"],
                     "label": r["label"], "ours": pick, "agree": ok,
                     "sizes": r["_sizes"], "frac": r["_frac"]})
    n = len(picks)
    maj = labels.most_common(1)[0] if labels else ("-", 0)
    print(f"\n比 {n} 个 facing 局面（**逐行对齐尺寸**，动作名口径）")
    print(f"  我们 == PokerBench 标签: {agree}/{n} = {agree/max(n,1)*100:.0f}%")
    print(f"  多数类基线（恒 '{maj[0]}'）: {maj[1]}/{n} = {maj[1]/max(n,1)*100:.0f}%")
    print(f"  尺寸已按行折算: {aligned}/{n}")
    print("  标签分布:", dict(labels), "| 我们分布:", dict(ours_dist))
    for x in res[:8]:
        print(f"   {x['street']:<6} {x['board']:<10} pot={x['pot']:<6} hero={x['hero']:<4} "
              f"菜单={x['sizes']:<8} frac={x['frac']} 标签={x['label']:<6} 我们={x['ours']}")
    (ROOT / "docs" / "pokerbench_validation_aligned.json").write_text(json.dumps({
        "dataset": "RZ412/PokerBench postflop_10k_test_set（Apache-2.0）",
        "mode": "facing 节点 + **逐行对齐下注尺寸**（CLI 实测采纳 bet_sizes ✓）",
        "n": n, "agree": agree, "agree_rate": round(agree / max(n, 1), 4),
        "majority_baseline": {"action": maj[0], "hits": maj[1], "rate": round(maj[1] / max(n, 1), 4)},
        "size_aligned_rows": aligned,
        "caveats": ["只比动作名（尺寸已对齐到该行折算比例，但仍非精确同尺寸 ✓）",
                     "范围=我们声明的 HU 范围（PokerBench 不提供范围 ⇒ 无法对齐 ✗）",
                     "有效栈按 100bb 平铺（未精确扣减已投注 ✗）",
                     "抽样非全量，非官方评测口径"],
        "rows": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/pokerbench_validation_aligned.json")


if __name__ == "__main__":
    main()
