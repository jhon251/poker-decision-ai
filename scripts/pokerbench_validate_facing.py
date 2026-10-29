#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HA：**PokerBench 独立验证（面对下注节点，口径正确版）**。

首版（P30-9-GY）口径错 ✗：PokerBench 的 postflop 行绝大多数是**面对下注**节点
（`available_moves` = Fold/Call[+Raise] ✗），而我们只取到**街根节点**（Check/Bet ✗）
⇒ 实测 0/40 属**节点错配**，不是策略差异 ✓。

本版走**导出 facing 节点**的正确路径（与 fb 套件同机制 ✓）：
  * 每个样本解一棵树并导出 `[[1],[2],[0,1]]` 三个 facing 节点 ✓；
  * 按导出的 `actor`（oop/ip）**匹配该行 hero 的位置** ✓；
  * 比较**动作名**（Fold/Call/Raise ✓）——尺寸不可比（CLI 忽略 bet_sizes ✗）如实标注 ✓；
  * 同时给**多数类基线**便于解读 ✓。
"""
from __future__ import annotations

import argparse
import ast as _ast
import collections
import csv
import json
import pathlib
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
                         "label": str(r.get("correct_decision") or "").split()[0],
                         "preflop": r.get("preflop_action") or ""})
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--streets", default="turn,river")
    args = ap.parse_args()

    rows = load_rows()
    cand = [r for r in rows
            if r["street"] in args.streets.split(",") and len(board_for(r)) >= 4
            and r["hero"] in ("IP", "OOP") and r["label"] and
            ("fold" in " ".join(r["moves"]).lower() or "call" in " ".join(r["moves"]).lower())]
    picks = cand[:args.n]
    print(f"PokerBench 行 {len(rows)} | 可评（facing + {args.streets}）{len(cand)} | 取 {len(picks)}")

    # ── 建 spot（导出三个 facing 节点；与 fb 套件同机制 ✓）──
    spots = []
    for i, r in enumerate(picks):
        spots.append({"spot_id": f"PB-{i:03d}-{r['street']}-{''.join(board_for(r))}",
                       "street": r["street"], "board": board_for(r),
                       "pot_bb": r["pot"], "effective_stack_bb": 100.0,
                       "ranges": dict(RANGES[family_of(r["preflop"])]),
                       "export_nodes": [[1], [2], [0, 1]],
                       "exports": [{"export_id": "bet_mid", "path": [1]},
                                    {"export_id": "bet_large", "path": [2]},
                                    {"export_id": "check_bet", "path": [0, 1]}],
                       "max_iterations": 300, "target_exploitability_pot_pct": 1.0,
                       "family": family_of(r["preflop"]).upper()})
    cfg = ROOT / ".tmp" / "pb_facing.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"spot_suite": {
        "name": "POKERBENCH-facing-validation", "game": "HU facing-bet nodes",
        "cli": "tools/bin/postflop-bench-cli.exe",
        "solver": "postflop-solver (b-inary) — 真实 CFR；HU；单街树；导出 facing 节点",
        "n_spots": len(spots)}, "spots": spots}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    out_dir = ROOT / ".tmp" / "pb_facing_out"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    print("求解中…（用 solve_spots_fb.py —— 它会向 CLI 传 export_nodes ✓）")
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
                         "--config", str(cfg), "--out-dir", str(out_dir),
                         "--workers", "1", "--solver-threads", "4"],
                        cwd=str(ROOT), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=7200)
    tail = [x for x in (p.stdout or "").splitlines() if "n_solved" in x or "n_failed" in x]
    print("   ", " | ".join(x.strip() for x in tail))

    # ── 比对（按 actor 匹配 hero 位置）──
    agree = base = 0
    labels = collections.Counter()
    ours_dist = collections.Counter()
    res = []
    for i, r in enumerate(picks):
        f = out_dir / f"{spots[i]['spot_id']}.json"
        pick_name = None
        if f.is_file():
            rec = json.loads(f.read_text(encoding="utf-8"))
            for e in (rec.get("exports") or []):
                if str(e.get("actor") or "").upper() == r["hero"]:
                    acts = [(NAME.get(str(a.get("name")).lower(), str(a.get("name"))),
                              float(a.get("frequency") or 0)) for a in (e.get("actions") or [])]
                    if acts:
                        pick_name = max(acts, key=lambda x: x[1])[0]
                    break
        # 我们树上该方向的合法动作名（若无导出 ⇒ 未命中）
        allowed = {NAME.get(str(m).split()[0].lower(), str(m)) for m in r["moves"]}
        ok = pick_name is not None and pick_name in allowed and pick_name == r["label"]
        agree += int(ok)
        labels[r["label"]] += 1
        ours_dist[pick_name or "(无)"] += 1
        res.append({"board": "".join(board_for(r)), "street": r["street"], "pot": r["pot"],
                     "hero": r["hero"], "holding": r["holding"], "moves": r["moves"],
                     "label": r["label"], "ours": pick_name, "agree": ok})
    n = len(picks)
    maj = labels.most_common(1)[0] if labels else ("-", 0)
    print(f"\n比 {n} 个 facing 局面（动作名口径）")
    print(f"  我们 == PokerBench 标签: {agree}/{n} = {agree/max(n,1)*100:.0f}%")
    print(f"  多数类基线（恒 '{maj[0]}'）: {maj[1]}/{n} = {maj[1]/max(n,1)*100:.0f}%")
    print("  标签分布:", dict(labels))
    print("  我们输出分布:", dict(ours_dist))
    for x in res[:8]:
        print(f"   {x['street']:<6} {x['board']:<10} pot={x['pot']:<6} hero={x['hero']:<4} "
              f"{x['holding']:<5} moves={x['moves']} 标签={x['label']:<6} 我们={x['ours']}")
    (ROOT / "docs" / "pokerbench_validation_facing.json").write_text(json.dumps({
        "dataset": "RZ412/PokerBench postflop_10k_test_set（Apache-2.0）",
        "node_type": "facing_bet（按导出的 actor 匹配 hero 位置）",
        "n": n, "agree": agree, "agree_rate": round(agree / max(n, 1), 4),
        "majority_baseline": {"action": maj[0], "hits": maj[1],
                               "rate": round(maj[1] / max(n, 1), 4)},
        "caveats": ["只比动作名——尺寸不可比（CLI 忽略 bet_sizes ⇒ 我们的下注/加注尺寸与其不同 ✗）",
                     "范围=我们声明的 HU 范围（按 preflop_action 判 SRP/3BP），非真实对手范围",
                     "有效栈按 100bb 近似（未精确扣减已投注）",
                     "抽样非全量，非官方评测口径"],
        "rows": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 docs/pokerbench_validation_facing.json")


if __name__ == "__main__":
    main()
