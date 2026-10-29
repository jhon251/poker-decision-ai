#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GY：**PokerBench 独立验证**（第三方标签，非自建 ✓）。

数据集：`RZ412/PokerBench`（Apache-2.0）`postflop_10k_test_set_*`。
每行给：board（flop/turn/river）、`evaluation_at`、`pot_size`、`hero_position`(OOP/IP)、
`holding`、`available_moves`、**`correct_decision`（第三方最优动作标签）**、`preflop_action`。

口径（诚实标注，避免过度声称）：
  * 只比**动作名**（Check/Bet/Call/Fold/Raise/AllIn）——我们的 CLI 忽略 `bet_sizes`，
    尺寸与其动作集不可比 ✗；
  * 只用 **hero=OOP** 的行：我们的现场求解从**街根（OOP 行动）**起解，
    IP 节点取不到 ✗（需 export_nodes 批量路径，本期未用）；
  * 范围：按 `preflop_action` 判定 SRP/3BP，使用**我们声明的 HU 范围**（假设，如实标注 ✓）；
  * 同时报告**多数类基线**（不看牌、恒出该行所在街的出现最多的动作）⇒ 便于解读一致率 ✓。
"""
from __future__ import annotations

import argparse
import ast as _ast
import asyncio
import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "pokerbench" / "postflop_10k_test_set_game_scenario_information.csv"
#: 我们声明的范围（与库内 3BP/SRP 假定一致；**假设**必须如实标注）
RANGES = {
    "3bp": {"oop": "QQ+,JJ,AKs,AQs,AJs,KQs,AKo,AQo",
             "ip": "TT-22,AQs-A2s,KQs-KTs,QJs-QTs,JTs,T9s,98s,87s,AQo,KQo,QJo"},
    "srp": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
             "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9s+,J9o+,T9o"},
}
MAP = {"check": "Check", "bet": "Bet", "call": "Call", "fold": "Fold",
        "raise": "Raise", "allin": "AllIn"}


def norm_action(name: str) -> str:
    n = str(name or "").strip().lower().replace(" ", "")
    if n.startswith("all"):
        return "AllIn"
    for k, v in MAP.items():
        if n.startswith(k):
            return v
    return str(name or "")


def family_of(preflop: str) -> str:
    """`preflop_action` ⇒ SRP/3BP（3bet+ 记 3bp ✓）。"""
    segs = [s for s in str(preflop or "").split("/") if s]
    raises = sum(1 for s in segs if s.endswith("bb"))
    return "3bp" if raises >= 2 else "srp"


def rows() -> list[dict]:
    out = []
    with SRC.open(encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                moves = _ast.literal_eval(r.get("available_moves") or "[]")
            except Exception:                                    # noqa: BLE001
                moves = []
            out.append({"board_flop": r.get("board_flop") or "",
                         "board_turn": r.get("board_turn") or "",
                         "board_river": r.get("board_river") or "",
                         "street": str(r.get("evaluation_at") or "").lower(),
                         "pot": float(r.get("pot_size") or 0),
                         "hero_pos": str(r.get("hero_position") or "").upper(),
                         "holding": str(r.get("holding") or ""),
                         "moves": [str(m) for m in moves],
                         "label": str(r.get("correct_decision") or ""),
                         "preflop": r.get("preflop_action") or "",
                         "aggressor": str(r.get("aggressor_position") or "").upper()})
    return out


def board_for(row: dict) -> list[str]:
    b = [row["board_flop"][i:i + 2] for i in range(0, len(row["board_flop"]), 2)]
    if row["street"] in ("turn", "river") and row["board_turn"]:
        b.append(row["board_turn"])
    if row["street"] == "river" and row["board_river"]:
        b.append(row["board_river"])
    return [c for c in b if len(c) == 2]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽样行数")
    ap.add_argument("--streets", default="turn,river", help="只评这些街（flop 现场解 ~7.5s ✗）")
    args = ap.parse_args()

    from poker_decision_ai import cli as cli_mod
    rt = cli_mod.build_runtime()
    prov = rt.new_context("pb", "x").providers.get_solver()

    all_rows = rows()
    # P30-9-GZ：**必须先过滤节点类型** ✗ —— 我们的现场求解只返回**街根节点**
    # （OOP 首先行动 ⇒ 选项是 Check/Bet ✓）；若拿它去比"面对下注"标签（Fold/Call/Raise ✗）
    # 就是苹果比橘子（首版实测 0/40 ✗ 正是此因）。这里只保留**hero 先行动**的行 ✓。
    def _first_in(r: dict) -> bool:
        mv = " ".join(r["moves"]).lower()
        return ("check" in mv or "bet" in mv) and ("fold" not in mv and "call" not in mv)

    cand = [r for r in all_rows
            if r["hero_pos"] == "OOP" and r["street"] in args.streets.split(",")
            and len(board_for(r)) >= 4 and r["label"] and _first_in(r)]
    print(f"PokerBench 行 {len(all_rows)} | 可评（OOP + {args.streets} + hero 先行动）{len(cand)}")

    picks = cand[:args.n]
    res, agree, base_hits = [], 0, 0
    label_counts = collections.Counter(r["label"] for r in picks)
    majority = label_counts.most_common(1)[0][0] if label_counts else None

    for r in picks:
        board = board_for(r)
        fam = family_of(r["preflop"])
        rg = RANGES[fam]
        ours = None
        try:
            s = await asyncio.wait_for(prov.solve({
                "street": r["street"], "board": board, "ranges": rg,
                "pot": r["pot"], "effective_stack": 100.0, "bb": 1.0,
                "max_iterations": 300, "target_exploitability_pot_pct": 1.0}),
                timeout=30)
            acts = [(norm_action(a.name), float(a.frequency or 0)) for a in (s.actions or [])]
            ours = max(acts, key=lambda x: x[1])[0] if acts else None
        except Exception as e:                                    # noqa: BLE001
            ours = f"ERR:{type(e).__name__}"
        ok = (ours == r["label"])
        agree += int(ok)
        base_hits += int(majority == r["label"])
        res.append({"board": "".join(board), "street": r["street"], "pot": r["pot"],
                     "family": fam, "holding": r["holding"], "moves": r["moves"],
                     "label": r["label"], "ours": ours, "agree": ok})

    n = len(picks)
    print(f"\n比 {n} 个局面（动作名口径）")
    print(f"  我们与 PokerBench 标签一致: {agree}/{n} = {agree/max(n,1)*100:.0f}%")
    print(f"  多数类基线（恒出 '{majority}'）: {base_hits}/{n} = {base_hits/max(n,1)*100:.0f}%")
    print("  标签分布:", dict(label_counts))
    print("  我们输出分布:", dict(collections.Counter(x["ours"] for x in res)))
    print("\n样例（前 8）:")
    for x in res[:8]:
        print(f"   {x['street']:<6} {x['board']:<10} pot={x['pot']:<5} {x['holding']:<5} "
              f"moves={x['moves']} 标签={x['label']:<5} 我们={x['ours']:<6} "
              f"{'✓' if x['agree'] else '✗'}")

    (ROOT / "docs" / "pokerbench_validation.json").write_text(json.dumps({
        "dataset": "RZ412/PokerBench postflop_10k_test_set（Apache-2.0）",
        "n": n, "agree": agree, "agree_rate": round(agree / max(n, 1), 4),
        "majority_baseline": {"action": majority, "hits": base_hits,
                               "rate": round(base_hits / max(n, 1), 4)},
        "caveats": ["只比动作名（尺寸不可比：我们的 CLI 忽略 bet_sizes ✗）",
                     "只用 hero=OOP 的行（我们现场求解只出街根节点）",
                     "范围=我们声明的 HU 范围（按 preflop_action 判 SRP/3BP），非真实对手范围",
                     "抽样非全量，非官方评测口径"],
        "rows": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 docs/pokerbench_validation.json")


if __name__ == "__main__":
    asyncio.run(main())
