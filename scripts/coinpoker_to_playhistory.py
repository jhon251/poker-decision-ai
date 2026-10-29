"""CoinPoker（已转换 V1）牌谱 → play_history 标准会话（进弱点报告/复盘）。

用法：
  python scripts/coinpoker_to_playhistory.py <V1牌谱.txt> [--max-decisions N]
                                             [--session-id 自定义]

流程：逐手解析 → 状态机重建经济账 → 逐 hero 决策点跑 RealtimeRunner
（与对打同管道，算出 AI 建议与一致率）→ PlayHistoryWriter 落盘标准会话。
写入后弱点报告与复盘原生消费。教学展示口径；hero 净收益从牌谱反推
（collected - 投入）。
"""
from __future__ import annotations

import os
import re
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_decision_ai.core.orchestration.runtime import build_runtime
from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.play.history import PlayHistoryWriter
from poker_decision_ai.realtime import RealtimeRunner

HERO_ACT = re.compile(
    r"^(Hero):\s+(folds|checks|calls|bets|raises)"
    r"(?:\s+\$(\d+\.?\d*)(?:\s+to\s+\$(\d+\.?\d*))?"
    r"(?:\s+and is all-in)?)?\s*$")
ACT_RE = re.compile(
    r"^(?P<p>.+?):\s+(?P<a>folds|checks|calls|bets|raises)"
    r"(?:\s+\$(?P<amt>\d+\.?\d*)(?:\s+to\s+\$(?P<to>\d+\.?\d*))?"
    r"(?:\s+and is all-in)?)?\s*$")
BLIND_RE = re.compile(
    r"^(?P<p>.+?):\s+posts\s+(?P<k>small blind|big blind)\s+\$(?P<a>\d+\.?\d*)")
COLLECT_RE = re.compile(
    r"^(?P<p>.+?):?\s+collected\s+\$(?P<a>\d+\.?\d*)\s+from pot")
STREET_MARKERS = ("*** HOLE CARDS ***", "*** FLOP ***", "*** TURN ***",
                  "*** RIVER ***", "*** SHOW DOWN ***", "*** SUMMARY ***")
STREET_OF = {"*** HOLE CARDS ***": "preflop", "*** FLOP ***": "flop",
             "*** TURN ***": "turn", "*** RIVER ***": "river",
             "*** SHOW DOWN ***": "showdown", "*** SUMMARY ***": "summary"}


class HandEcon:
    """一手的经济状态机：座位筹码/本街投入/底池/hero 净收益。"""

    def __init__(self, parsed, hero: str):
        self.bb = parsed.bb or 0.1
        self.hero = hero
        self.seat_by_name = {e["name"]: int(e["seat"])
                             for e in parsed.seats or []}
        self.max_seat = max([int(e["seat"]) for e in parsed.seats or []]
                            or [1])
        self.stacks = {int(e["seat"]): float(e["stack"])
                       for e in parsed.seats or []}
        self.street = "preflop"
        self.committed: dict[int, float] = {}
        self.level = 0.0
        self.pot = 0.0
        self.hero_invest = 0.0
        self.hero_collected = 0.0

    def chips(self, dollars: float) -> int:
        return int(round(dollars / self.bb * 20))

    def street_bets_list(self) -> list[int]:
        return [int(round(self.committed.get(s, 0.0)))
                for s in range(1, self.max_seat + 1)]

    def stacks_list(self) -> list[int]:
        return [int(round(self.stacks.get(s, 0.0)))
                for s in range(1, self.max_seat + 1)]

    def seat_of(self, name: str) -> int | None:
        return self.seat_by_name.get(name)

    def invest(self, seat: int | None, amount: float):
        if amount <= 0:
            return
        if seat is not None:
            self.stacks[seat] = self.stacks.get(seat, 0.0) - amount
            self.committed[seat] = self.committed.get(seat, 0.0) + amount
        self.pot += amount
        if seat == self.seat_of(self.hero):
            self.hero_invest += amount

    def new_street(self, name: str):
        self.street = name
        self.committed = {}
        self.level = 0.0


def process_hand(lines: list[str], parsed, hero: str):
    """状态机重建 + hero 决策点。返回 dict（timeline/points/net/...）。"""
    ec = HandEcon(parsed, hero)
    timeline: list[dict] = []
    points: list[int] = []
    street = "preflop"
    for i, line in enumerate(lines):
        if any(m in line for m in STREET_MARKERS):
            street = STREET_OF.get(next(
                m for m in STREET_MARKERS if m in line), street)
            ec.new_street(street)
            continue
        m_blind = BLIND_RE.match(line)
        if m_blind:
            a = float(m_blind["a"])
            ec.invest(ec.seat_of(m_blind["p"]), a)
            ec.level = max(ec.level, ec.bb)
            continue
        m_col = COLLECT_RE.match(line)
        if m_col:
            if m_col["p"] == hero:
                ec.hero_collected += float(m_col["a"])
            continue
        if line.startswith("Dealt to "):
            continue
        if HERO_ACT.match(line):
            points.append(i)
        m_act = ACT_RE.match(line)
        if not m_act:
            continue                       # shows/未知行：经济上无影响
        p_, a_ = m_act["p"], m_act["a"]
        seat = ec.seat_of(p_)
        amt = float(m_act["amt"]) if m_act["amt"] else None
        to = float(m_act["to"]) if m_act["to"] else None
        invest = 0.0
        if a_ == "calls" and amt:
            invest = amt
        elif a_ == "bets" and amt:
            invest = amt
            ec.level = amt
        elif a_ == "raises" and to is not None:
            put = ec.committed.get(seat, 0.0) if seat is not None else 0.0
            invest = max(to - put, 0.0)
            ec.level = to
        ec.invest(seat, invest)
        timeline.append({
            "street": street, "seat": seat if seat is not None else -1,
            "action": a_,
            "amount_chips": ec.chips(amt) if amt else None,
            "pot_chips": ec.chips(ec.pot), "stacks": ec.stacks_list(),
            "street_bets": ec.street_bets_list()})
    net = ec.hero_collected - ec.hero_invest
    terminal = "showdown" if parsed.shows else "fold_out"
    return {"timeline": timeline, "points": points, "net": net,
            "terminal": terminal, "econ": ec,
            "street_reached": ("river" if len(parsed.board or []) >= 5
                               else "turn" if len(parsed.board or []) == 4
                               else "flop" if len(parsed.board or []) >= 3
                               else "preflop")}


def main() -> int:
    import argparse
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="已转换的 V1 牌谱 txt")
    ap.add_argument("--out-dir", default="data/play_history")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--hero", default="Hero")
    ap.add_argument("--max-decisions", type=int, default=0)
    args = ap.parse_args()

    from pathlib import Path
    text = Path(args.src).read_text(encoding="utf-8")
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.startswith("PokerStars Hand #"):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur is not None and line.strip():
            cur.append(line)
    if cur:
        blocks.append(cur)

    sid = args.session_id or ("coinpoker-" + time.strftime("%Y%m%d_%H%M%S")
                              + "-" + uuid.uuid4().hex[:4])
    writer = PlayHistoryWriter(args.out_dir, sid, stake="nl2",
                               label="CoinPoker 导入牌谱",
                               seed_base=20261012, lineup=[],
                               deterministic=True, table_size=6,
                               stack_bb=100, focus_position="auto",
                               source="external")

    async def run():
        runtime = build_runtime()
        runner = RealtimeRunner(runtime)
        n_dec = n_adv = n_agree = 0
        n_hands = 0
        t0 = time.perf_counter()
        try:
            for bi, block_lines in enumerate(blocks):
                block_text = "\n".join(block_lines)
                parsed = parse_hand_history(block_text)
                res = process_hand(block_lines, parsed, args.hero)
                hero_decisions: list[dict] = []
                for li in res["points"]:
                    if args.max_decisions and n_dec >= args.max_decisions:
                        break
                    street = "preflop"
                    for line in block_lines[:li]:
                        if "*** FLOP ***" in line:
                            street = "flop"
                        elif "*** TURN ***" in line:
                            street = "turn"
                        elif "*** RIVER ***" in line:
                            street = "river"
                    prefix = "\n".join(block_lines[:li])
                    hm = HERO_ACT.match(block_lines[li])
                    act = hm.group(2) if hm else "fold"
                    amt = float(hm.group(3)) if hm and hm.group(3) else None
                    advisor = {"available": False, "reason": "error"}
                    try:
                        rep = await runner.decide({
                            "kind": "cash_hand", "hero_id": args.hero,
                            "hand_history_text": prefix})
                        d = rep.to_dict()
                        fd = d.get("final_decision") or {}
                        advisor = {
                            "available": fd.get("action") is not None,
                            "action": fd.get("action"),
                            "size_bb": fd.get("recommended_size"),
                            "provenance": fd.get("provenance") or {},
                            "latency_ms": d.get("latency_ms")}
                    except Exception as exc:          # noqa: BLE001
                        advisor = {"available": False,
                                   "reason": f"error:{type(exc).__name__}"}
                    base = {"folds": "fold", "checks": "check",
                            "calls": "call", "bets": "bet",
                            "raises": "raise"}.get(act, act)
                    agree = (advisor.get("action") == base
                             if advisor.get("available") else None)
                    hero_decisions.append({
                        "street": street, "pot_bb": None, "to_call_bb": None,
                        "legal_actions": [], "position": None,
                        "action": base, "amount_chips": amt, "size_bb": None,
                        "peeked_advice": False, "advisor": advisor,
                        "advisor_agree": agree})
                    n_dec += 1
                    if advisor.get("available"):
                        n_adv += 1
                        n_agree += 1 if agree else 0
                    if n_dec % 25 == 0:
                        print(f"  决策 {n_dec} ({time.perf_counter() - t0:.0f}s)",
                              flush=True)
                if args.max_decisions and n_dec >= args.max_decisions \
                        and not hero_decisions:
                    break
                net = res["net"]
                bb = parsed.bb or 0.1
                net_bb = (net / bb) if net is not None else None
                net_r = (net_bb - (parsed.rake or 0.0) / bb
                         if net_bb is not None and net_bb > 0 else net_bb)
                seat_by_name = {e["name"]: int(e["seat"])
                                for e in parsed.seats or []}
                revealed = [{"seat": seat_by_name[sh["player"]],
                             "hole": sh["cards"]}
                            for sh in parsed.shows or []
                            if sh["player"] in seat_by_name]
                winners = [seat_by_name[w["player"]]
                           for w in parsed.winners or []
                           if w["player"] in seat_by_name]
                writer.write_hand(
                    hand_index=bi, deal_seed=int(parsed.hand_id or bi),
                    button=parsed.button_seat,
                    hero_hole=list(parsed.hole_cards or []),
                    board=list(parsed.board or []),
                    street_reached=res["street_reached"],
                    n_players_dealt=len(parsed.seats or []),
                    hero_decisions=hero_decisions,
                    net_bb=net_bb, net_bb_raked=net_r, net_bb_raked_full=net_r,
                    rake_chips=round((parsed.rake or 0.0) / bb * 20, 2),
                    pot_chips=round((parsed.pot or 0.0) / bb * 20, 2),
                    terminal_reason=res["terminal"], aborted=None,
                    hand_wall_ms=0.0, timeline=res["timeline"],
                    winners=winners, revealed=revealed)
                n_hands += 1
                if args.max_decisions and n_dec >= args.max_decisions:
                    break
        finally:
            try:
                await runtime.close()
            except Exception:                         # noqa: BLE001
                pass
        print(f"完成：写入 {n_hands} 手 ｜ 决策 {n_dec} ｜ "
              f"AI 建议 {n_adv} ｜ 一致 {n_agree} "
              f"（{n_agree / max(1, n_adv) * 100:.0f}%）", flush=True)

    asyncio.run(run())
    print(f"会话文件: {args.out_dir}/play_session_{sid}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
