#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FX：把**我们自解的 HU 子局**记录转换成 6-max 表 schema 并入库。

来源记录（r1/r3 求解脚本产出）：
  {spot_id, street, board, query{board,pot_bb,effective_stack_bb,oop_range,ip_range},
   actor:"oop", actions:[{name,amount_bb,ev_bb,ev_per_hand_bb,freq_per_hand,frequency}],
   private_cards_actor:[...], solved, ...}

6-max 表需要的字段：
  {spot_id（**首段=family**，如 3bp-…）, street, board, pot_bb, effective_stack_bb,
   positions{oop,ip}, actor_position, node, actions[{name,size_bb,frequency}],
   private_cards_actor, freq_per_hand{动作名:[…]}, solved, source, source_note, agent}

provenance 如实标注：`source=our_hu_solve`，说明是**我们声明的范围**、非外部库。
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(r"<repo>")


def convert(rec: dict, *, family: str, oop: str, ip: str, node: str,
            id_prefix: str, source_note: str) -> dict | None:
    q = rec.get("query") or {}
    board = list(q.get("board") or rec.get("board") or [])
    acts = [a for a in (rec.get("actions") or []) if isinstance(a, dict)]
    hands = [str(h) for h in (rec.get("private_cards_actor") or [])]
    if not board or not acts or not hands:
        return None
    fph: dict[str, list] = {}
    actions_out = []
    for a in acts:
        name = str(a.get("name") or "").lower()
        if not name:
            continue
        size = a.get("amount_bb")
        actions_out.append({"name": name,
                             "size_bb": (round(float(size), 4) if size is not None else None),
                             "frequency": (round(float(a.get("frequency") or 0.0), 4))})
        arr = a.get("freq_per_hand")
        if isinstance(arr, list):
            fph[name] = [round(float(x), 4) if isinstance(x, (int, float)) else 0.0
                          for x in arr]
    if not actions_out:
        return None
    base = str(rec.get("spot_id") or "spot").replace("/", "_")
    return {
        "spot_id": f"{family}-{id_prefix}-{''.join(board)}",
        "street": str(q.get("street") or rec.get("street") or "flop"),
        "board": board,
        "pot_bb": round(float(q.get("pot_bb") or 0.0), 3),
        "effective_stack_bb": round(float(q.get("effective_stack_bb") or 0.0), 3),
        "positions": {"oop": oop.upper(), "ip": ip.upper()},
        "actor_position": oop.upper(),                 # 根节点 = oop 先行动
        "node": node,
        "to_call_bb": None, "bet_ratio": None,
        "actions": actions_out,
        "private_cards_actor": hands,
        "freq_per_hand": fph,
        "n_private_hands_actor": len(hands),
        "wall_sec": rec.get("wall_sec"),
        "exploitability_bb": rec.get("exploitability_bb"),
        "iterations": rec.get("iterations"),
        "solved": True,
        "source": "our_hu_solve",
        "source_note": source_note,
        "agent": "hu_to_6max_v1",
        "src_spot_id": base,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/solver/spots_6max_real3bp")
    ap.add_argument("--dst", default="data/solver/spots_6max")
    ap.add_argument("--family", default="3bp")
    ap.add_argument("--oop", default="BB")
    ap.add_argument("--ip", default="BTN")
    ap.add_argument("--node", default="first_in")
    ap.add_argument("--id-prefix", default="ourhu")
    ap.add_argument("--note", default=("我们自解 HU 子局（BTN 开 2.5 / BB 3bet / BTN 跟，"
                                        "声明范围为我们的 HU 范围）；非外部库合成数据"))
    ap.add_argument("--only", default=None, help="只导 spot_id 含该子串的源记录")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = ROOT / args.src
    dst = ROOT / args.dst
    files = sorted(f for f in src.glob("*.json") if f.name != "manifest.json")
    written, skipped = 0, 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if args.only and args.only not in str(rec.get("spot_id") or ""):
            continue
        out = convert(rec, family=args.family, oop=args.oop, ip=args.ip, node=args.node,
                       id_prefix=args.id_prefix, source_note=args.note)
        if out is None:
            skipped += 1
            continue
        if not args.dry_run:
            (dst / f"{out['spot_id']}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        written += 1
    print(f"转换 {written} 条（跳过 {skipped}）→ {dst.relative_to(ROOT)}"
          + ("（dry-run，未写盘）" if args.dry_run else ""))
    # 抽样展示
    if written and not args.dry_run:
        sample = sorted(dst.glob(f"{args.family}-{args.id_prefix}-*.json"))[:2]
        for s in sample:
            r = json.loads(s.read_text(encoding="utf-8"))
            print(f"   样例 {r['spot_id']} | {r['positions']} actor={r['actor_position']} "
                  f"node={r['node']} pot={r['pot_bb']} actions={[a['name'] for a in r['actions']]} "
                  f"hands={len(r['private_cards_actor'])}")


if __name__ == "__main__":
    main()
