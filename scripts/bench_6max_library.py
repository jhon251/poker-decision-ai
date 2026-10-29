#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-DX：**6-max 库保真度基准**（用户真实牌局面的均衡遵循度）。

对 6-max 库节点（first_in / facing_bet）：
  * 从该节点的 `private_cards_actor` 用**确定性种子**抽手牌（优先逐手频率 >0 的）；
  * 合成真机风格局面（位置/牌面/底池/筹码/翻前线路），走 **realtime 路径**决策；
  * 指标：
      - 命中模式（查表 exact / approx / miss）；
      - **逐手支持率**：引擎动作在该手牌的求解器支持集内（freq_per_hand > 0）；
      - **主选一致率**：引擎动作 == 该手牌频率最高的动作；
      - 锚定率（策略来源为求解器锚定）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "solver" / "spots_6max"
# 引擎环（button_seat=5 时实测）：1=BB 2=UTG/LJ 3=MP/HJ 4=CO 5=BTN 6=SB
POS_SEAT = {"BB": 1, "LJ": 2, "UTG": 2, "HJ": 3, "MP": 3, "CO": 4, "BTN": 5, "SB": 6}


def _preflop_line(family: str) -> list[dict]:
    if "3bp" in family.lower():
        return [{"street": "preflop", "seat": 5, "action": "raise", "amount": 2.5},
                {"street": "preflop", "seat": 3, "action": "raise", "amount": 10.0},
                {"street": "preflop", "seat": 5, "action": "call", "amount": 10.0}]
    return [{"street": "preflop", "seat": 5, "action": "raise", "amount": 2.5},
            {"street": "preflop", "seat": 3, "action": "call", "amount": 2.5}]


def _observation(rec: dict, hand: list[str]) -> dict:
    pos = rec.get("positions") or {}
    oop, ip = str(pos.get("oop") or "BB"), str(pos.get("ip") or "BTN")
    actor_role = str(rec.get("actor_position") or oop)
    hero_seat = POS_SEAT.get(actor_role, POS_SEAT.get(oop, 3))
    # P30-9-EC：对手 = **另一方**（此前误用 ip 座位 ⇒ actor==BTN 时 hero 与对手同座 ⇒ 状态只剩 1 人）
    _other = oop if actor_role == ip else ip
    opp_seat = POS_SEAT.get(_other, POS_SEAT.get(ip, 5))
    btn = POS_SEAT.get("BTN", 5)
    fam = str(rec.get("spot_id") or "").split("-")[0]
    return {
        "source": "bench_6max", "street": str(rec.get("street") or "flop"),
        "hero_seat": hero_seat, "button_seat": btn,
        "hole_cards": list(hand), "board": list(rec.get("board") or []),
        "seats": [{"seat": i, "name": f"s{i}",
                    "stack": float(rec.get("effective_stack_bb") or 0),
                    "in_hand": i in (hero_seat, opp_seat)} for i in range(1, 7)],
        "pot": float(rec.get("pot_bb") or 0), "blinds": {"sb": 0.5, "bb": 1.0},
        "ante": 0.0, "to_call": float(rec.get("to_call_bb") or 0.0), "confidence": 0.98,
        "action_history": _preflop_line(fam),
        "extra": {"v3f_hero_position": actor_role, "v3f_bets": {}},
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--hands", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0,
                     help="节点采样起点（避免总是取前 N 个）")
    ap.add_argument("--out", default=".tmp/bench_6max.json")
    args = ap.parse_args()

    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner
    from poker_decision_ai.core.providers.spot_table_6max import get_spot_table_6max

    files = sorted(LIB.glob("*.json"))
    by_kind = {"first_in": [], "facing_bet": []}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        by_kind.setdefault(str(d.get("node")), []).append(d)
    pick: list[dict] = []
    for k, lst in by_kind.items():
        _k = max(1, args.nodes // 2)
        _o = args.offset % max(len(lst), 1)
        pick.extend((lst + lst)[_o:_o + _k])
    print(f"采样节点: {len(pick)}（first_in/facing 各半）+ 每节点 {args.hands} 手")

    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    tbl = get_spot_table_6max()
    rows = []
    for rec in pick:
        hands = rec.get("private_cards_actor") or []
        fp = rec.get("freq_per_hand") or {}
        if not hands or not fp:
            continue
        rnd = random.Random(f"{rec.get('spot_id')}|bench6")
        cand = list(range(len(hands)))
        rnd.shuffle(cand)
        picked = 0
        for i in cand:
            if picked >= args.hands:
                break
            # 只取"求解器确实会打"的手牌（逐手频率 >0），保证指标有意义
            if not any((fp.get(k) or [0] * (i + 1))[i] > 0 for k in fp):
                continue
            hand = [hands[i][:2], hands[i][2:4]]
            try:
                rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                        "vision_observation": _observation(rec, hand),
                                        "session_id": "bench6"})
                dd = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
                fd = dd.get("final_decision") or {}
                action = str(fd.get("action") or "")
                reason = str(fd.get("reasoning") or "")
            except Exception as e:                                     # noqa: BLE001
                action, reason = f"ERROR:{type(e).__name__}", ""
            # 逐手支持与主选
            sup = {k: float((fp.get(k) or [0] * (i + 1))[i]) for k in fp}
            in_sup = bool(sup.get(action, 0.0) > 0.0)
            primary = max(sup, key=lambda k: sup[k]) if sup else None
            rows.append({"spot": rec.get("spot_id"), "node": rec.get("node"),
                          "hand": "".join(hand), "engine": action,
                          "in_support_hand": in_sup, "primary": primary,
                          "freq_hand": sup.get(action, 0.0),
                          # P30-9-EH：锚定 **或** 求解器知情的偏离都算"吃到均衡"
                          "anchored": ("求解器锚定" in reason or "6-max GTO 解命中" in reason
                                        or "solver_per_hand" in reason or "偏离" in reason)})
            picked += 1

    n = len(rows)
    if n:
        sup = sum(1 for r in rows if r["in_support_hand"])
        pri = sum(1 for r in rows if r["engine"] == r["primary"])
        anc = sum(1 for r in rows if r["anchored"])
        print(f"\n样本 {n}")
        print(f"  逐手支持率（引擎动作在该手牌支持集内）: {sup}/{n} = {sup/n*100:.1f}%")
        print(f"  主选一致率（该手牌频率最高动作）: {pri}/{n} = {pri/n*100:.1f}%")
        print(f"  锚定率（求解器锚定/6-max 命中）: {anc}/{n} = {anc/n*100:.1f}%")
        import collections
        print("  动作分布:", dict(collections.Counter(r["engine"] for r in rows)))
        print("  不在支持集的样例:")
        for r in [x for x in rows if not x["in_support_hand"]][:6]:
            print(f"    {str(r['spot'])[:26]:<27} {r['hand']:<5} engine={r['engine']:<6} "
                  f"primary={str(r['primary']):<6} freq_hand={r['freq_hand']:.3f}")
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("已存:", args.out)


if __name__ == "__main__":      # P30-9-FQ：加守卫，便于被其它脚本安全导入
    asyncio.run(main())
