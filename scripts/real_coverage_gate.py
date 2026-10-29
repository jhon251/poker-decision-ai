#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-GP：**真实会话真解率门禁**（确定性、秒级；防回退）。

做法：从 `dist/logs/events-*.jsonl` 重建每个真实翻后局面的**查询键**
（street / board / pot_bb / node / positions / actor / family），直接查 6-max 表：
  * `exact`：库内真解（不近似）✓ —— 这是"吃得到真解"的硬指标；
  * `approx` / `miss`：近似或未命中。
**不走决策引擎**（故无现场解的不确定性 ✓），可与基线比较 ⇒ 回退即失败 ✓。

用法：
  python scripts/real_coverage_gate.py --record     # 记录/更新基线
  python scripts/real_coverage_gate.py --gate       # 与基线比较（回退 ⇒ exit 1）
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "dist" / "logs"
LIB = ROOT / "data" / "solver" / "spots_6max"
BASELINE = ROOT / "docs" / "real_coverage_baseline.json"
NEED = {"flop": 3, "turn": 4, "river": 5}
POS6 = ["BTN", "SB", "BB", "UTG", "MP", "CO"]


def observations() -> list[dict]:
    seen: dict[str, dict] = {}
    for f in sorted(LOG.glob("events-*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            if d.get("event") != "agent_invocation" or d.get("agent") != "vision":
                continue
            vo = ((d.get("input") or {}).get("request") or {}).get("vision_observation")
            if vo and vo.get("street"):
                seen[str(d.get("request_id") or len(seen))] = vo
    return list(seen.values())


def pos_of(seat, btn, n_seats: int) -> str | None:
    if seat is None or btn is None or not n_seats:
        return None
    off = (int(seat) - int(btn)) % n_seats
    return POS6[off] if 0 <= off < len(POS6) else None


def keys_from(vo: dict) -> dict | None:
    street = str(vo.get("street") or "")
    need = NEED.get(street)
    if need is None:
        return None
    board = [str(c) for c in (vo.get("board") or []) if c]
    if len(board) != need:
        return None                                     # 残缺牌面不参与（第②项另计）
    seats = vo.get("seats") or []
    btn = vo.get("button_seat")
    in_hand = [s for s in seats if s.get("in_hand")]
    if len(in_hand) != 2:
        return None                                     # 只评两人池（多人无参考）
    positions = [(pos_of(s.get("seat"), btn, len(seats)), s.get("seat")) for s in in_hand]
    positions = [(p, s) for p, s in positions if p]
    if len(positions) != 2:
        return None
    order = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}
    positions.sort(key=lambda x: order.get(x[0], 9))
    oop, ip = positions[0][0], positions[1][0]
    bb = float(((vo.get("blinds") or {}).get("bb")) or 0.02) or 0.02
    pot_bb = float(vo.get("pot") or 0.0) / bb
    to_call = float(vo.get("to_call") or 0.0)
    stacks = [float(s.get("stack") or 0.0) / bb for s in in_hand if s.get("stack")]
    return {"street": street, "board": board, "pot_bb": round(pot_bb, 3),
             "node": ("facing_bet" if to_call > 1e-9 else "first_in"),
             "positions": {"oop": oop, "ip": ip}, "actor": oop,
             "stack_bb": round((sum(stacks) / len(stacks)) if stacks else 0.0, 2),
             "bet_ratio": (round(to_call / max(pot_bb - to_call, 1e-9), 4)
                            if to_call > 1e-9 else None),
             "to_call_bb": (round(to_call / bb, 3) if to_call > 1e-9 else None)}


def classify(table, q: dict) -> str:
    for fam in ("3bp", "srp"):
        h = table.lookup({**q, "family": fam})
        if h is None:
            continue
        if (h.get("raw") or {}).get("consensus_n"):
            return "approx"
        return "approx" if h.get("is_approximate") else "exact"
    return "miss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="记录/更新基线")
    ap.add_argument("--gate", action="store_true", help="与基线比较（回退 ⇒ exit 1）")
    ap.add_argument("--list-miss", action="store_true", help="打印未精确命中的真实局面")
    ap.add_argument("--tolerance", type=int, default=0, help="允许的绝对下降数")
    args = ap.parse_args()

    from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max
    table = SpotTable6max(str(LIB))
    table._ensure_loaded()

    obs = observations()
    # P30-9-HI：**节点歧义必须排除** ✗ —— `to_call` 未知时无法区分 facing/first_in ✓，
    #   若按 first_in 查询并"命中 first_in 记录"，看起来 exact 却是**节点错配** ✗。
    #   ⇒ 主指标改为"三者兼备"子集（牌面完整 + to_call 可知 + 两人池 ✓），
    #     并把节点歧义的请求单独计数 ✓（不并入主指标 ✓）。
    _obs_all = obs
    obs = [v for v in obs if v.get("to_call") is not None]
    # P30-9-HK：**显式**统计"翻后且 to_call 未知"的请求数 ✓
    #   （此前用 len(全部) − len(过滤后) ✗ ⇒ 把翻前请求也算进去 ⇒ 数字虚高 ✗）
    _amb = sum(1 for v in _obs_all
                if str(v.get("street") or "") in ("flop", "turn", "river")
                and v.get("to_call") is None)
    ks = [k for k in (keys_from(v) for v in obs) if k]
    kinds = collections.Counter(classify(table, k) for k in ks)
    n = len(ks)
    exact = kinds.get("exact", 0)
    misses = [k for k in ks if classify(table, k) != "exact"]
    _amb_n = _amb
    res_amb = {"node_ambiguous_excluded": _amb_n}
    if args.list_miss:
        print(f"未精确命中的真实局面（{len(misses)} 例）——可直接定向补解：")
        for k in misses:
            print(f"  board={''.join(k['board'])} positions={k['positions']} actor={k['actor']} "
                  f"node={k['node']} pot_bb={k['pot_bb']} stack_bb={k['stack_bb']} "
                  f"bet_ratio={k['bet_ratio']} street={k['street']}")
    res = {"n_evaluable_2p_complete": n, "exact": exact,
            "rate": round(exact / max(n, 1), 4), "kinds": dict(kinds),
            "node_ambiguous_excluded": _amb,
            "note": ("只统计**两人池 + 牌面完整 + to_call 可知**的真实翻后局面"
                      "（确定性口径，不走引擎；to_call 未知者**排除**，避免节点错配 ✗）")}
    # P30-9-HI：**输入可信度**汇总（牌面 / to_call / 人数 ✓）——覆盖率的"分母"质量 ✓
    _need = {"flop": 3, "turn": 4, "river": 5}
    _post = [v for v in _obs_all if str(v.get("street") or "") in _need]  # 全量口径 ✓
    if _post:
        _cb = sum(1 for v in _post
                   if len([c for c in (v.get("board") or []) if c]) == _need[str(v["street"])])
        _tk = sum(1 for v in _post if v.get("to_call") is not None)
        _two = sum(1 for v in _post
                    if len([s for s in (v.get("seats") or []) if s.get("in_hand")]) == 2)
        _full = sum(1 for v in _post
                     if len([c for c in (v.get("board") or []) if c]) == _need[str(v["street"])]
                     and v.get("to_call") is not None
                     and len([s for s in (v.get("seats") or []) if s.get("in_hand")]) == 2)
        res["input_integrity"] = {
            "postflop_requests": len(_post), "board_complete": _cb,
            "to_call_known": _tk, "two_player": _two, "fully_evaluable": _full,
            "note": "覆盖率的分母质量：牌面完整 + to_call 可知 + 两人池 ✓"}
        print(f"输入可信度（翻后 {len(_post)} 请求）: 牌面完整 {_cb} | to_call 可知 {_tk} | "
              f"两人池 {_two} | **三者兼备 {_full}**")
    print(f"真实可评估局面 {n} | exact {exact} = {res['rate']*100:.0f}% | {dict(kinds)}")

    if args.record or not BASELINE.is_file():
        BASELINE.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已记录基线 → {BASELINE.relative_to(ROOT)}")
        return
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    drop = int(base.get("exact", 0)) - exact
    print(f"基线 exact={base.get('exact')}（{base.get('rate')}）| 当前 exact={exact} | 变化={-drop:+d}")
    if args.gate and drop > args.tolerance:
        print(f"[FAIL] 真实真解率回退：{base.get('exact')} → {exact}（超过容忍 {args.tolerance}）")
        sys.exit(1)
    if args.gate:
        print("[PASS] 真实真解率未回退 ✓")


if __name__ == "__main__":
    main()
