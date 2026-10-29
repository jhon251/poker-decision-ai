"""批量牌谱分析：对每手的每个 hero 决策点跑 RealtimeRunner，输出对比报告。

用法：
  python scripts/analyze_batch.py <V1牌谱.txt> <输出报告.txt> [--hero Hero] [--max N]

口径：对每个 hero 决策点，截取"该点之前的完整动作序列"喂决策引擎
（与对打模式同一 RealtimeRunner 管道），对比 AI 建议与 hero 实际动作。
仅教学展示，非 EV 承诺。
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_decision_ai.core.orchestration.runtime import build_runtime
from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.realtime import RealtimeRunner

HERO_ACT = re.compile(
    r"^(?P<hero>Hero):\s+(?P<a>folds|checks|calls|bets|raises)"
    r"(?:\s+\$?(?P<amt>\d+\.?\d*)(?:\s+to\s+\$?(?P<to>\d+\.?\d*))?"
    r"(?:\s+and is all-in)?)?\s*$")
STREET_CN = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌", "river": "河牌"}


def split_hands(text: str) -> list[str]:
    blocks: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.startswith("PokerStars Hand #"):
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        elif cur is not None and line.strip():
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def hero_decision_points(block: list[str]) -> list[tuple[int, str, str | None]]:
    """返回 [(行号, 动作, 金额或None)]（hero 在动作区的决策行）。"""
    pts = []
    in_showdown = False
    for i, line in enumerate(block):
        if line.startswith("*** SHOW DOWN ***") or line.startswith("*** SUMMARY ***"):
            in_showdown = True
        if in_showdown:
            continue
        m = HERO_ACT.match(line)
        if m:
            pts.append((i, m["a"], m["amt"]))
    return pts


def main() -> int:
    import argparse
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--hero", default="Hero")
    ap.add_argument("--max", type=int, default=0, help="最多分析决策数（0=全部）")
    args = ap.parse_args()

    from pathlib import Path
    text = Path(args.src).read_text(encoding="utf-8")
    blocks = split_hands(text)
    rows: list[dict] = []
    n_planned = 0
    for b in blocks:
        n_planned += len(hero_decision_points(b.splitlines()))
    limit = args.max if args.max > 0 else n_planned
    print(f"牌谱 {len(blocks)} 手，hero 决策点 {n_planned} 个，"
          f"本次分析 {min(limit, n_planned)} 个", flush=True)

    async def run():
        runtime = build_runtime()
        runner = RealtimeRunner(runtime)
        done = 0
        t0 = time.perf_counter()
        try:
            for bi, block_text in enumerate(blocks):
                lines = block_text.splitlines()
                parsed = parse_hand_history(block_text)
                hand_id = parsed.hand_id
                hero_seat = next((int(e["seat"]) for e in parsed.seats or []
                                  if e.get("name") == args.hero), None)
                labels = ("BTN", "SB", "BB", "UTG", "MP", "CO")
                position = (labels[(hero_seat - parsed.button_seat)
                                   % parsed.max_players]
                            if hero_seat is not None and parsed.max_players
                            else "?")
                pts = hero_decision_points(lines)
                for (li, act, amt) in pts:
                    if done >= limit:
                        return
                    street = "preflop"
                    for line in lines[:li]:
                        if "*** FLOP ***" in line:
                            street = "flop"
                        elif "*** TURN ***" in line:
                            street = "turn"
                        elif "*** RIVER ***" in line:
                            street = "river"
                    prefix = "\n".join(lines[:li])
                    status = "error"
                    ai_action = ai_size = source = conf = None
                    try:
                        rep = await runner.decide({
                            "kind": "cash_hand", "hero_id": args.hero,
                            "hand_history_text": prefix})
                        d = rep.to_dict()
                        status = d.get("status")
                        fd = d.get("final_decision") or {}
                        ai_action = fd.get("action")
                        ai_size = fd.get("recommended_size")
                        conf = fd.get("confidence")
                        source = (fd.get("provenance") or {}).get("source")
                    except Exception as exc:          # noqa: BLE001
                        ai_action = f"EXC:{type(exc).__name__}"
                    # 词形归一：hero 侧是解析器原词（folds/calls...），AI 侧是基形
                    base = {"folds": "fold", "checks": "check",
                            "calls": "call", "bets": "bet", "raises": "raise"}
                    hero_base = base.get(act, act)
                    agree = (ai_action == hero_base) if ai_action else None
                    rows.append({
                        "hand": hand_id, "street": street, "hero": act,
                        "amount": amt, "ai": ai_action, "ai_size": ai_size,
                        "conf": conf, "source": source, "status": status,
                        "agree": agree, "position": position})
                    done += 1
                    if done % 25 == 0:
                        el = time.perf_counter() - t0
                        print(f"  进度 {done}/{min(limit, n_planned)} "
                              f"({el:.0f}s)", flush=True)
        finally:
            try:
                await runtime.close()
            except Exception:                         # noqa: BLE001
                pass

    asyncio.run(run())

    # ── 聚合 ──────────────────────────────────────────────
    valid = [r for r in rows if r["status"] in ("ok", "degraded")]
    err = [r for r in rows if r not in valid]
    n_adv = sum(1 for r in valid if r["ai"])
    n_agree = sum(1 for r in valid if r["agree"])
    L: list[str] = []
    L.append("══════════ CoinPoker 牌谱 × AI 建议 对比报告 ══════════")
    L.append(f"手数 {len(blocks)} ｜ hero 决策 {len(rows)} 个"
             f"（引擎有效 {len(valid)}，错误 {len(err)}）")
    L.append(f"AI 给出建议 {n_adv} 条 ｜ 你与 AI 一致 {n_agree} 条"
             f"（{n_agree / max(1, n_adv) * 100:.0f}%）")
    L.append("")
    L.append("── 按街一致率 ──")
    for street in ("preflop", "flop", "turn", "river"):
        st = [r for r in valid if r["street"] == street and r["ai"]]
        if not st:
            continue
        ag = sum(1 for r in st if r["agree"])
        L.append(f"  {STREET_CN[street]:　<3} 决策 {len(st):3d} ｜ "
                 f"一致 {ag}/{len(st)}（{ag / len(st) * 100:.0f}%）")
    L.append("")
    L.append("── 不一致明细（前 30 条）──")
    for r in [r for r in valid if r["agree"] is False][:30]:
        size = (f" {r['ai_size']:.1f}bb" if isinstance(r["ai_size"], (int, float))
                else "")
        amt = f" {r['amount']}ch" if r["amount"] else ""
        L.append(f"  手{r['hand']} [{STREET_CN.get(r['street'], r['street'])}] "
                 f"你: {r['hero']}{amt} ｜ AI: {r['ai']}{size}"
                 f"  ({r['source']})")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    import json
    json_path = str(args.out)[:-4] + ".json"
    Path(json_path).write_text(json.dumps({
        "n_hands": len(blocks), "rows": rows,
        "summary": {"decisions": len(rows), "valid": len(valid),
                    "ai_given": n_adv, "agree": n_agree}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"数据: {json_path}")
    print(f"报告: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
