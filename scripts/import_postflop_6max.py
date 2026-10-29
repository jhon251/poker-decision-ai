#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B：把外部 6-max 翻后解（poker-practice）转换为本项目 6-max spot 库格式。

来源与许可（随产物保留）：github.com/jensbaagaard/poker-practice
  README: 「License: MIT. The range data in `data/openSourcePokerData/` is free to use」；
  翻后数据由 TexasSolver（AGPL，作为独立程序运行）离线求解产生，属**数据**。

外部格式（data/postflop/flops/<spot>/<board>.json）：
  {"board":[3 cards], "pot":bb, "stack":bb, "spot":"srp-LJ-BTN", "weight":w,
   "exploitability":x,
   "root":{"p":"oop", "actions":[{"a":"X"}|{"a":"B","bb":s}|{"a":"AI","bb":stack}],
           "s":{hand:[freq%...]}, "c":[子节点(递归)]}}

输出（data/solver/spots_6max/<spot>__<board>.json）：与 spots_r1 同构，便于复用
基准（逐手牌支持集）字段，并**额外**带 6-max 位置信息供位置键查表使用：
  {"spot_id","street","board","pot_bb","effective_stack_bb",
   "positions":{"oop":"LJ","ip":"BTN"},"actor_position","node":"first_in",
   "actions":[{name,size_bb,frequency}], "private_cards_actor":[...],
   "freq_per_hand":{action:[...]}, "weight", "solved":true,
   "source":"external_gto_json", "source_note":<归属>, "agent":"6max_import_v1"}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "external" / "poker_practice" / "postflop"
OUT_DIR = ROOT / "data" / "solver" / "spots_6max"

ATTRIBUTION = ("来源：github.com/jensbaagaard/poker-practice（README 声明 MIT；"
                "data/openSourcePokerData 可自由使用）；翻后数据 = TexasSolver 离线求解输出；"
                "本文件为格式转换，未改动数值")

_ACT = {"X": "check", "B": "bet", "AI": "allin", "C": "call", "F": "fold", "R": "raise"}


def _board_list(fname: str, node: dict) -> list[str]:
    b = node.get("board") or []
    if len(b) == 3:
        return [str(c) for c in b]
    # 文件名形如 7s5h4h.json ⇒ 拆成 3 张
    stem = fname.split(".")[0]
    return [stem[i:i + 2] for i in range(0, len(stem), 2)][:3]


def convert_node(node: dict, board: list[str], spot: str, weight: float,
                  pot_bb: float, stack_bb: float, depth: int = 0,
                  node_kind: str = "first_in", to_call_bb: float | None = None,
                  reached_by: str | None = None) -> dict:
    """把一个解树节点转为我们的 spot 记录。

    node_kind ∈ first_in | facing_bet（锚定契约键）；
    facing_bet 时记录 to_call_bb 与 bet_ratio（供尺寸桶匹配）。
    """
    actions = []
    for a in node.get("actions") or []:
        name = _ACT.get(str(a.get("a") or ""), str(a.get("a") or "").lower())
        size = a.get("bb")
        actions.append({"name": name,
                         "size_bb": (round(float(size), 4) if size is not None else None)})
    per_hand = node.get("s") or {}
    hands = sorted(per_hand.keys())
    freq = {a["name"]: [] for a in actions}
    for h in hands:
        row = per_hand.get(h) or []
        for i, a in enumerate(actions):
            v = row[i] if i < len(row) else 0
            freq[a["name"]].append(round(float(v) / 100.0, 4))
    # 聚合频率（按手牌等权；仅作参考，逐手牌频率才是主指标）
    for a in actions:
        vals = freq.get(a["name"]) or []
        a["frequency"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    actor = str(node.get("p") or "oop")
    # 位置对：翻后行动顺序 SB < BB < LJ/UTG < HJ < CO < BTN ⇒ 先行动者为 OOP
    parts = spot.split("-")
    a, b = (parts[1], parts[2]) if len(parts) >= 3 else ("?", "?")
    order = {"SB": 0, "BB": 1, "LJ": 2, "UTG": 2, "HJ": 3, "MP": 3,
             "CO": 4, "BTN": 5}
    if a in order and b in order:
        oop, ip = (a, b) if order[a] < order[b] else (b, a)
    else:
        oop, ip = a, b
    return {
        "spot_id": f"{spot}-{''.join(board)}",
        "street": "flop",
        "board": list(board),
        "pot_bb": round(float(pot_bb), 3),
        "effective_stack_bb": round(float(stack_bb), 3),
        "positions": {"oop": oop.upper(), "ip": ip.upper()},
        "actor_position": (oop.upper() if actor == "oop" else ip.upper()),
        "node": node_kind,
        "to_call_bb": (round(float(to_call_bb), 4) if to_call_bb is not None else None),
        "bet_ratio": (round(float(to_call_bb) / max(float(pot_bb), 1e-9), 4)
                       if to_call_bb is not None else None),
        "reached_by": reached_by,
        "actions": actions,
        "private_cards_actor": hands,
        "freq_per_hand": freq,
        "weight": round(float(weight), 6),
        "solved": True,
        "source": "external_gto_json",
        "source_note": ATTRIBUTION,
        "agent": "6max_import_v1",
        "children_depth_kept": 0 if depth == 0 else depth,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.glob("*.json") if f.name != "manifest.json")
    n_ok = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                 # noqa: BLE001
            print("跳过", f.name, e)
            continue
        root = d.get("root") or {}
        if not root:
            continue
        board = _board_list(f.name.split("__")[-1], root)
        spot = str(d.get("spot") or f.name.split("__")[0])
        weight = float(d.get("weight") or 0.0)
        pot = float(d.get("pot") or 0.0)
        stack = float(d.get("stack") or 0.0)
        rec = convert_node(root, board, spot, weight, pot, stack)
        (out / f"{rec['spot_id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        n_ok += 1
        # P30-9-CF：导出深度 1 的"面对下注"节点（父节点下注 ⇒ 子节点面对下注）
        kids = root.get("c") or []
        acts = root.get("actions") or []
        for idx, kid in enumerate(kids):
            if not isinstance(kid, dict) or idx >= len(acts):
                continue
            a = acts[idx]
            if str(a.get("a")) not in ("B", "AI") or a.get("bb") is None:
                continue
            bet_bb = float(a.get("bb"))
            kid_pot = pot + bet_bb           # 下注进入底池
            rec2 = convert_node(kid, board, spot, weight, kid_pot, stack,
                                 depth=1, node_kind="facing_bet",
                                 to_call_bb=bet_bb,
                                 reached_by=f"{str(a.get('a'))}{bet_bb:g}")
            rec2["spot_id"] = f"{rec2['spot_id']}-fb-{rec2.get('reached_by') or 'x'}"
            (out / f"{rec2['spot_id']}.json").write_text(
                json.dumps(rec2, ensure_ascii=False, indent=1), encoding="utf-8")
            n_ok += 1
        # P30-9-EZ：**通用递归导出**（逐玩家投入记账，口径与引擎一致；开关控制）
        def _export_deep(node, pot_now, inv, actor_of_node, path, depth_now):
            nonlocal n_ok
            for idx, child in enumerate(node.get("c") or []):
                if not isinstance(child, dict) or not child.get("actions"):
                    continue
                acts_p = node.get("actions") or []
                a_p = acts_p[idx] if idx < len(acts_p) else {}
                nm = str(a_p.get("a") or "")
                sz = a_p.get("bb")
                other = "ip" if actor_of_node == "oop" else "oop"
                inv2 = dict(inv)
                pot2 = pot_now
                new_path = path + [idx]
                if nm in ("B", "AI", "R") and sz is not None:
                    inv2[actor_of_node] = float(inv2.get(actor_of_node, 0.0)) + float(sz)
                    pot2 = pot_now + float(sz)
                    to_call = float(inv2[actor_of_node]) - float(inv2.get(other, 0.0))
                    rec = convert_node(child, board, spot, weight, pot2, stack,
                                        depth=depth_now, node_kind="facing_bet",
                                        to_call_bb=max(to_call, 0.0),
                                        reached_by="B" * len(new_path))
                elif nm == "C":
                    add = max(0.0, float(inv.get(other, 0.0)) - float(inv.get(actor_of_node, 0.0)))
                    inv2[actor_of_node] = float(inv2.get(actor_of_node, 0.0)) + add
                    pot2 = pot_now + add
                    rec = convert_node(child, board, spot, weight, pot2, stack,
                                        depth=depth_now, node_kind="first_in",
                                        reached_by="C" * len(new_path))
                else:                                   # X / F 等：底池不变
                    rec = convert_node(child, board, spot, weight, pot_now, stack,
                                        depth=depth_now, node_kind="first_in",
                                        reached_by="X" * len(new_path))
                tag = "p" + "".join(str(x) for x in new_path)
                rec["spot_id"] = f"{rec['spot_id']}-{tag}"
                (out / f"{rec['spot_id']}.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                n_ok += 1
                if depth_now < 6:
                    _export_deep(child, pot2, inv2,
                                  ("oop" if actor_of_node == "ip" else "ip"),
                                  new_path, depth_now + 1)

        if str(__import__("os").environ.get("SIXMAX_DEEP_EXPORT", "0")).lower() in (
                "1", "on", "true", "yes"):
            _export_deep(root, pot, {"oop": 0.0, "ip": 0.0}, "oop", [], 2)
        # P30-9-EU：c[0] 本身 = **IP 在 OOP 过牌后的首入决策（c-bet）**——翻后最高频
        if kids and isinstance(kids[0], dict):
            _ip_node = kids[0]
            if _ip_node.get("actions"):
                rec_ip = convert_node(_ip_node, board, spot, weight, pot, stack,
                                       depth=1, node_kind="first_in", reached_by="X")
                rec_ip["spot_id"] = f"{rec_ip['spot_id']}-x"
                (out / f"{rec_ip['spot_id']}.json").write_text(
                    json.dumps(rec_ip, ensure_ascii=False, indent=1), encoding="utf-8")
                n_ok += 1
                # 过牌-过牌后 OOP 的下一决策（first_in，actor=oop）
                _xx = (_ip_node.get("c") or [])
                if _xx and isinstance(_xx[0], dict) and _xx[0].get("actions"):
                    rec_xx = convert_node(_xx[0], board, spot, weight, pot, stack,
                                           depth=2, node_kind="first_in", reached_by="XX")
                    rec_xx["spot_id"] = f"{rec_xx['spot_id']}-xx"
                    (out / f"{rec_xx['spot_id']}.json").write_text(
                        json.dumps(rec_xx, ensure_ascii=False, indent=1), encoding="utf-8")
                    n_ok += 1
        # P30-9-EJ：**深度 2** —— c[0]（oop 过牌后 ip 行动）的子节点
        #   c[0].c[0] = 过牌-过牌后 oop 的下一决策（first_in）
        #   c[0].c[1] = oop 过牌后面对 ip 下注（c-bet 防守，单帧可判定）
        #   c[0].c[2] = oop 过牌后面对 ip 全下
        if kids and isinstance(kids[0], dict):
            mid = kids[0]
            mid_acts = mid.get("actions") or []
            mid_kids = mid.get("c") or []
            for j, gk in enumerate(mid_kids):
                if not isinstance(gk, dict) or j >= len(mid_acts):
                    continue
                b = mid_acts[j]
                if str(b.get("a")) == "X":
                    rec3 = convert_node(gk, board, spot, weight, pot, stack,
                                         depth=2, node_kind="first_in",
                                         reached_by="XX")
                    rec3["spot_id"] = f"{rec3['spot_id']}-xx"
                elif str(b.get("a")) in ("B", "AI") and b.get("bb") is not None:
                    bb_bet = float(b.get("bb"))
                    rec3 = convert_node(gk, board, spot, weight, pot + bb_bet, stack,
                                         depth=2, node_kind="facing_bet",
                                         to_call_bb=bb_bet,
                                         reached_by=f"XB{bb_bet:g}")
                    rec3["spot_id"] = f"{rec3['spot_id']}-xb-{bb_bet:g}"
                else:
                    continue
                (out / f"{rec3['spot_id']}.json").write_text(
                    json.dumps(rec3, ensure_ascii=False, indent=1), encoding="utf-8")
                n_ok += 1
    print(f"转换完成：{n_ok} 个 6-max 翻牌节点 → {out}")
    if n_ok:
        first = sorted(out.glob("*.json"))[0]
        r = json.loads(first.read_text(encoding="utf-8"))
        print("样例:", first.name)
        print("  positions:", r["positions"], "| actor:", r["actor_position"],
              "| node:", r["node"], "| pot:", r["pot_bb"], "| stack:", r["effective_stack_bb"])
        print("  actions:", [(a["name"], a["size_bb"], a["frequency"]) for a in r["actions"]])
        print("  手牌数:", len(r["private_cards_actor"]),
              "| 首手牌频率:", {k: v[0] for k, v in r["freq_per_hand"].items()})


if __name__ == "__main__":
    main()
