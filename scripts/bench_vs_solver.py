#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-BM：决策基准（对标离线 CFR 求解结果）。

数据源：`data/solver/spots_r1/*.json`（516 个 HU 已解 spot，真实 CFR，
含**逐手牌 EV** `actions[].ev_per_hand_bb`，与 `private_cards_actor` 一一对应）。

方法：
  * 参考动作（GTO 近似）= 该手牌 EV 最高的动作；
  * regret = 最佳 EV − 引擎所选动作的 EV（**真实数值，不是估计**）；
  * 引擎走**产品路径**（RealtimeRunner，GUI 等价，~1s/次）。

用法：python scripts/bench_vs_solver.py --spots 10 --hands 8 [--out .tmp/bench_vs_solver.json]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                # noqa: BLE001
    pass

SPOT_DIR = ROOT / "data" / "solver" / "spots_r1"
#: solver 动作名 → 引擎动作名（同一决策节点上的合法族）
_SOLVER2ENGINE = {"Check": "check", "Call": "call", "Bet": "bet",
                  "Raise": "raise", "AllIn": "allin", "Fold": "fold"}


def _norm_solver(name: str) -> str:
    n = str(name or "").strip().lower()
    if n in ("all-in", "allin", "shove"):
        return "allin"
    return n


def _map_engine_action(action: str, amount_bb, spot: dict) -> str:
    """把引擎动作映射到该节点的 solver 动作名（仅用该节点**实际存在**的动作）。"""
    a = str(action or "").strip().lower()
    have = {str(x.get("name")) for x in (spot.get("actions") or [])}
    facing = "Call" in have or "Fold" in have          # 该节点是否面对下注
    if a in ("allin", "all-in"):
        return "AllIn" if "AllIn" in have else ""
    if a == "fold":
        return "Fold" if "Fold" in have else ""
    if a == "call":
        return "Call" if "Call" in have else ("Check" if "Check" in have else "")
    if a == "check":
        return "Check" if "Check" in have else ""
    if a in ("bet", "raise"):
        if a == "raise" and "Raise" in have:
            return "Raise"
        if "Bet" in have:
            return "Bet"
        if "Raise" in have:
            return "Raise"
        return ""
    return ""


def load_spots(limit: int | None, streets=("flop",), actor: str | None = "oop") -> list[dict]:
    manifest = json.loads((SPOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for sid in manifest.get("solved_ids") or []:
        p = SPOT_DIR / f"{sid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("solved") or not d.get("actions"):
            continue
        if str(d.get("street")) not in streets:
            continue
        if actor and str(d.get("actor")) != actor:
            continue
        if not (d.get("query") or {}).get("board"):
            continue
        out.append(d)
        if limit and len(out) >= limit:
            break
    return out


def observation_for(spot: dict, hand: str) -> dict:
    """OOP(actor=oop)=BB/seat3；IP(actor=ip)=BTN/seat5。street 根节点 ⇒ to_call=0。"""
    q = spot["query"]
    pot_bb = float(q.get("pot_bb") or 0.0)
    eff = float(q.get("effective_stack_bb") or 0.0)
    bb = 1.0
    # OOP = BB（单加注池：BB 跟注、BTN 加注）⇒ hero 为 OOP，翻后先行动
    actor = str(spot.get("actor") or "oop")
    hero_seat = 3 if actor == "oop" else 5
    pos = "BB" if actor == "oop" else "BTN"
    return {
        "source": "bench_vs_solver", "street": str(spot.get("street") or "flop"),
        "hero_seat": hero_seat, "button_seat": 5,
        "hole_cards": [hand[:2], hand[2:4]], "board": list(q["board"]),
        # P30-9-EE：r1 套件是 **HU 解** ⇒ 基准声明 **2 座 HU 桌**（否则 6-max 外部库会介入）
        "seats": [{"seat": i, "name": f"s{i}", "stack": eff,
                    "in_hand": True} for i in (3, 5)],
        "pot": pot_bb, "blinds": {"sb": 0.5, "bb": bb}, "ante": 0.0,
        "to_call": 0.0, "confidence": 0.98,
        "action_history": [
            {"street": "preflop", "seat": 3, "action": "call", "amount": 1.0},
            {"street": "preflop", "seat": 5, "action": "raise", "amount": 3.5}],
        "extra": {"v3f_hero_position": pos, "v3f_bets": {},
                   "bench_spot_id": spot.get("spot_id"),
                   "v3f_action_history": [
                       {"street": "preflop", "seat": 3, "action": "call", "amount": 1.0},
                       {"street": "preflop", "seat": 5, "action": "raise", "amount": 3.5}]},
    }


FB_DIR = ROOT / "data" / "solver" / "spots_fb"
#: 各位置范围内的代表性具体手牌（用于让引擎在一个真实手牌上决策）
FB_HANDS = {
    "oop": ["QQ", "77", "AQs", "KQs", "T9s", "AQo"],
    "ip": ["QQ", "77", "AQs", "KQs", "T9s", "AQo"],
}


def _concrete(hi_lo: str) -> list[str] | None:
    """把 "AQs"/"77" 这类类别展开成一个具体手牌（如 AdQd）。"""
    try:
        from poker_decision_ai.domain.poker.ranges import parse_range
        r = parse_range(hi_lo)
        for combo, w in r.weights.items():
            if w <= 0:
                continue
            hi, lo, suited = combo
            if hi == lo:
                return [f"{_RN[hi]}h", f"{_RN[hi]}d"]
            return [f"{_RN[hi]}h", f"{_RN[lo]}h" if suited else f"{_RN[lo]}d"]
    except Exception:                                          # noqa: BLE001
        return None
    return None


_RN = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8", 7: "7",
       6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}


def load_fb_spots(limit: int | None, streets=("flop",)) -> list[dict]:
    out = []
    for f in sorted(FB_DIR.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        if not d.get("solved") or str(d.get("street")) not in streets:
            continue
        for i, ex in enumerate(d.get("exports") or []):
            if not isinstance(ex, dict) or not ex.get("actions"):
                continue
            out.append({"spot": d, "export": ex, "export_index": i})
            if limit and len(out) >= limit:
                return out
    return out


def fb_observation(item: dict, hole: list[str]) -> dict:
    spot, ex = item["spot"], item["export"]
    q = spot["query"]
    actor = str(ex.get("actor") or "oop")
    hero_seat = 3 if actor == "oop" else 5
    pos = "BB" if actor == "oop" else "BTN"
    bb = 1.0
    # P30-9-CP：补**翻前线路历史**——引擎的 RangeAgent 依赖 action_history 回放
    # 识别对手线路（open/3bet/4bet/cold_call）；此前 fb 不传历史 ⇒ 引擎退到宽默认
    # 范围 ⇒ 系统性过度跟注（call→GTO:Fold 13/36）。这里按 spot 家族给出真实线路。
    sid = str(spot.get("spot_id") or "")
    hist: list[dict] = []
    if "3BP" in sid:
        hist = [{"street": "preflop", "seat": 5, "action": "raise", "amount": 2.5},
                {"street": "preflop", "seat": 3, "action": "raise", "amount": 10.0},
                {"street": "preflop", "seat": 5, "action": "call", "amount": 10.0}]
    elif "SRP" in sid:
        hist = [{"street": "preflop", "seat": 5, "action": "raise", "amount": 2.5},
                {"street": "preflop", "seat": 3, "action": "call", "amount": 2.5}]
    return {
        "source": "bench_vs_solver_fb", "street": str(spot.get("street") or "flop"),
        "hero_seat": hero_seat, "button_seat": 5,
        "hole_cards": list(hole), "board": list(q["board"]),
        # P30-9-EE：fb 套件是 **HU 解** ⇒ 基准应声明 **2 座 HU 桌**
        # （此前给 6 座 ⇒ 6-max 外部库会介入，范畴不符）
        "seats": [{"seat": i, "name": f"s{i}",
                    "stack": float(q.get("effective_stack_bb") or 0.0),
                    "in_hand": True} for i in (3, 5)],
        "pot": float(ex.get("pot_bb") or 0.0),
        "blinds": {"sb": 0.5, "bb": bb}, "ante": 0.0,
        "to_call": float(ex.get("to_call_bb") or 0.0), "confidence": 0.98,
        # 顶层 action_history（state 的消费口径）+ extra 留档
        "action_history": hist,
        "extra": {"v3f_hero_position": pos, "v3f_bets": {},
                   "v3f_action_history": hist,
                   "bench_spot_id": spot.get("spot_id"),
                   "bench_export": ex.get("history")},
    }


def _fb_sampled_categories(item: dict, actor: str, n: int) -> list[str]:
    """P30-9-DA：按该节点**声明范围**的组合权重抽样 n 个手牌类别（确定性）。

    目标：让"引擎动作分布 vs GTO 聚合频率"成为**同类口径**（聚合本就是范围加权）。
    范围取 `query.<actor>_range`（求解时的声明范围）；不可用时回退固定名单。
    """
    import random as _rnd
    d = item["spot"]
    q = d.get("query") or {}
    rng_str = str(q.get(f"{actor}_range") or q.get("oop_range" if actor == "oop"
                                                     else "ip_range") or "")
    if not rng_str:
        return list(FB_HANDS.get(actor, []))[:n]
    try:
        from poker_decision_ai.domain.poker.ranges import parse_range
        rng = parse_range(rng_str)
        combos = getattr(rng, "combos", None)
        cats: list[str] = []
        weights: list[float] = []
        if isinstance(combos, dict) and combos:
            for k, v in combos.items():
                try:
                    w = float(v)
                except (TypeError, ValueError):
                    w = 1.0
                cats.append(str(k))
                weights.append(max(w, 0.0))
        if not cats:
            return list(FB_HANDS.get(actor, []))[:n]
        seed = f"{d.get('spot_id')}|{actor}"
        r = _rnd.Random(seed)
        picked: list[str] = []
        while len(picked) < n and sum(weights) > 0:
            pick = r.choices(cats, weights=weights, k=1)[0]
            picked.append(pick)
        return picked
    except Exception:                                          # noqa: BLE001
        return list(FB_HANDS.get(actor, []))[:n]


def _fb_sampled_hands(item: dict, actor: str, n: int) -> list[list[str]]:
    """P30-9-DA：按该节点**声明范围**的分布抽样 n 手**具体手牌**（确定性）。

    聚合频率是范围加权的，固定 6 张强牌天然偏高 ⇒ 从范围分布抽样才与聚合同类。
    组合元组约定（ranges.sample_weighted）：(rank1, rank2, suited)，rank 用 2..14。
    失败 ⇒ 回退固定名单（诚实降级，不伪造）。
    """
    import random as _rnd
    board = {str(c) for c in ((item["spot"].get("query") or {}).get("board") or [])}
    ranks = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}
    for v in range(2, 10):
        ranks[v] = str(v)
    suits = ("s", "h", "d", "c")

    def combo_to_cards(combo) -> list[str] | None:
        try:
            a, b, suited = int(combo[0]), int(combo[1]), bool(combo[2])
        except (TypeError, ValueError, IndexError):
            return None
        ra, rb = ranks.get(a), ranks.get(b)
        if ra is None or rb is None:
            return None
        if a == b:
            for s1 in suits:
                for s2 in suits:
                    if s1 == s2:
                        continue
                    c1, c2 = f"{ra}{s1}", f"{rb}{s2}"
                    if c1 not in board and c2 not in board:
                        return [c1, c2]
            return None
        for s1 in suits:
            if suited:
                s2 = s1
            else:
                s2 = next((s for s in suits if s != s1), suits[0])
            c1, c2 = f"{ra}{s1}", f"{rb}{s2}"
            if c1 not in board and c2 not in board:
                return sorted([c1, c2], key=lambda x: "AKQJT98765432".index(x[0]))
        return None

    d = item["spot"]
    q = d.get("query") or {}
    rng_str = str(q.get(f"{actor}_range") or "")
    fallback = [h for h in (_concrete(c) for c in FB_HANDS.get(actor, [])) if h][:n]
    if not rng_str:
        return fallback
    try:
        from poker_decision_ai.domain.poker.ranges import parse_range
        rng = parse_range(rng_str)
        rnd = _rnd.Random(f"{d.get('spot_id')}|{actor}")
        out: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        tries = 0
        while len(out) < n and tries < 40:
            tries += 1
            for combo in rng.sample_weighted(rnd, max(4, n)):
                cards = combo_to_cards(combo)
                if not cards:
                    continue
                key = tuple(cards)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cards)
                if len(out) >= n:
                    break
        return out or fallback
    except Exception:                                          # noqa: BLE001
        return fallback


async def run_fb(args, rr) -> None:
    items = load_fb_spots(args.spots, streets=tuple(
        s.strip() for s in str(args.streets).split(",") if s.strip()))
    print(f"载入 facing-bet 节点: {len(items)} 个")
    rows = []
    for it in items:
        ex = it["export"]
        acts = {str(a.get("name")): a for a in ex.get("actions") or []}
        if not acts:
            continue
        eq_ev = sum(float(a.get("ev_bb") or 0.0) * float(a.get("frequency") or 0.0)
                     for a in acts.values())
        actor = str(ex.get("actor") or "oop")
        # P30-9-DA：**按声明范围加权抽样**手牌——聚合频率是范围加权的，
        # 固定 6 张强牌（FB_HANDS）天然偏高 ⇒ 口径不同类。此处按范围组合权重
        # 抽样（确定性：以 spot_id+actor 为种子），使引擎动作分布与聚合频率可比。
        holes = _fb_sampled_hands(it, actor, max(1, int(args.hands)))
        print(f"  {it['spot']['spot_id'][:26]:<27} actor={actor} 抽样手牌: "
              + ','.join(''.join(h) for h in holes))
        for hole in holes:
            obs = fb_observation(it, hole)
            try:
                rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                        "vision_observation": obs,
                                        "session_id": "benchfb"})
                d = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
                action = (d.get("final_decision") or {}).get("action")
            except Exception as e:                             # noqa: BLE001
                action = f"ERROR:{type(e).__name__}"
            tag = _map_engine_action(action, None, {"actions": list(acts.values())})
            freq = float((acts.get(tag) or {}).get("frequency") or 0.0) if tag else 0.0
            eng_ev = float((acts.get(tag) or {}).get("ev_bb") or 0.0) if tag else None
            closest = max(acts, key=lambda k: float(acts[k].get("frequency") or 0.0))
            rows.append({"spot": it["spot"].get("spot_id"),
                          "export": ex.get("history"), "actor": actor,
                          "board": "".join(it["spot"]["query"]["board"]),
                          "hand": "".join(hole), "hand_cards": list(hole),
                          "to_call_bb": ex.get("to_call_bb"),
                          "gto_support": {k: round(float(v.get("frequency") or 0.0), 3)
                                           for k, v in acts.items()},
                          "engine": action, "mapped": tag,
                          "in_support": bool(freq >= 0.05),
                          "gto_primary": closest,
                          # 注意：导出节点只有"范围级"EV，且 off-path 动作的 ev_bb 与
                          # 均衡 EV 口径不完全可比（实测出现负差）⇒ 仅作参考，不作结论。
                          "ev_gap_bb_ref_only": (round(eq_ev - eng_ev, 4)
                                                  if isinstance(eng_ev, float) else None)})
    ok = [r for r in rows if r.get("mapped")]
    ins = [r for r in ok if r["in_support"]]
    errs = [r for r in ok if not r["in_support"]]
    print(f"\n样本 {len(rows)}（可映射 {len(ok)}）")
    if ok:
        # P30-9-CQ：fb 导出**无逐手数据** ⇒ 逐手比对仅供参考（口径已改名）
        print(f"[informational] 引擎动作落在聚合频率 ≥5% 的比例: "
              f"{len(ins)}/{len(ok)} = {len(ins)/len(ok)*100:.1f}%"
              "（聚合口径，非逐手策略 ⇒ 不作为主指标）")
    # ── 范围级分布比对（fb 的正确口径）────────────────────────────
    if ok:
        eng = collections.Counter(str(r.get("mapped") or "") for r in ok)
        names = sorted({str(r.get("mapped") or "") for r in ok}
                        | {k for r in ok for k in (r.get("gto_support") or {})})
        gto = {}
        for n in names:
            vals = [float((r.get("gto_support") or {}).get(n) or 0.0) for r in ok]
            gto[n] = sum(vals) / len(vals) if vals else 0.0
        n_ok = len(ok)
        eng_share = {n: eng.get(n, 0) / n_ok for n in names}
        tot = sum(eng_share.values()) or 1.0
        eng_share = {n: v / tot for n, v in eng_share.items()}
        tv = 0.5 * sum(abs(eng_share[n] - gto.get(n, 0.0)) for n in names)
        print(f"范围级分布 TV 距离: {tv:.3f}（0=一致，1=完全不同）")
        print("  动作   引擎占比   GTO 聚合频率   偏差")
        for n in sorted(names, key=lambda x: -abs(eng_share[x] - gto.get(x, 0.0))):
            print(f"  {n:<6} {eng_share[n]:>8.3f} {gto.get(n, 0.0):>12.3f} "
                  f"{eng_share[n] - gto.get(n, 0.0):>+8.3f}")
        globals()["_FB_MIX"] = {"tv": round(tv, 4), "engine_share": eng_share,
                                 "gto_aggregate": gto, "n": n_ok}
    if errs:
        from collections import Counter
        pat = Counter(f"{r['mapped'] or '?'}→GTO:{r['gto_primary']}" for r in errs)
        print(f"不在支持集内（支持<5%）: {len(errs)} 条；典型形态（引擎→GTO主选）：")
        for k, v in pat.most_common(6):
            print(f"    {k}: {v} 次")
    for r in ok[:12]:
        print(f"  {r['spot'][:24]:<25} {r['hand']:<4} to_call={r['to_call_bb']:<6} "
              f"GTO={r['gto_support']} engine={r['engine']:<6} "
              f"in_support={r['in_support']}")
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("已存:", args.out)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=10)
    ap.add_argument("--streets", default="flop")
    ap.add_argument("--actor", default="oop")
    ap.add_argument("--suite", default="r1", choices=("r1", "fb"))
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--out", default=".tmp/bench_vs_solver.json")
    args = ap.parse_args()

    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner

    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    if args.suite == "fb":
        await run_fb(args, rr)
        return
    streets = tuple(s.strip() for s in str(args.streets).split(",") if s.strip())
    actor = None if str(args.actor) in ("any", "all", "") else str(args.actor)
    spots = load_spots(args.spots, streets=streets, actor=actor)
    print(f"载入已解 spot: {len(spots)} 个（street={streets}, actor={actor}）")
    rows = []
    for spot in spots:
        hands = spot.get("private_cards_actor") or []
        acts = spot.get("actions") or []
        if not hands or not acts:
            continue
        step = max(1, len(hands) // max(1, args.hands))
        for i in range(0, len(hands), step):
            hand = hands[i]
            if len(hand) != 4 or i >= len(acts[0].get("ev_per_hand_bb") or []):
                continue
            # P30-9-CY：**尺寸感知**的逐手 EV/频率（此前按动作名建字典 ⇒ 同名多尺寸
            # 互相覆盖 ⇒ regret 虚高：536 spot 中 92 个（17.2%）有重名 Bet 行）
            rows_ev: list[tuple[str, float | None, float | None]] = []
            for _a in acts:
                _ev_arr = _a.get("ev_per_hand_bb") or []
                _ev = _ev_arr[i] if i < len(_ev_arr) and isinstance(_ev_arr[i], (int, float)) \
                    else None
                _sz = _a.get("size_bb")
                rows_ev.append((str(_a.get("name")), _ev, float(_sz) if isinstance(_sz, (int, float)) else None))
            evs_rows = [r_ for r_ in rows_ev if r_[1] is not None]
            if not evs_rows:
                continue
            best_row = max(evs_rows, key=lambda r_: r_[1])
            best_ev = best_row[1]
            best_name = best_row[0]
            obs = observation_for(spot, hand)
            t0 = time.perf_counter()
            try:
                rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                        "vision_observation": obs,
                                        "session_id": "bench"})
                d = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
                action = (d.get("final_decision") or {}).get("action")
                size_bb = (d.get("final_decision") or {}).get("recommended_size")
            except Exception as e:                               # noqa: BLE001
                action, size_bb, d = f"ERROR:{type(e).__name__}", None, {}
            dt = (time.perf_counter() - t0) * 1000
            tag = _map_engine_action(action, size_bb, spot)
            # 逐手频率（同样按行，避免重名覆盖；支持集取该动作名下**最高**行）
            _freq_hand: dict = {}
            for _a in acts:
                _fh = _a.get("freq_per_hand") or []
                if i < len(_fh) and isinstance(_fh[i], (int, float)):
                    _n = str(_a.get("name"))
                    _freq_hand[_n] = max(_freq_hand.get(_n, 0.0), float(_fh[i]))
            # 引擎动作 → **同名且尺寸最近**的行（尺寸缺失时取该名下行内最高频行）
            _cand = [r_ for r_ in evs_rows if r_[0] == tag]
            eng_row = None
            if _cand:
                if isinstance(size_bb, (int, float)):
                    _with_size = [r_ for r_ in _cand if r_[2] is not None]
                    if _with_size:
                        eng_row = min(_with_size, key=lambda r_: abs(r_[2] - float(size_bb)))
                if eng_row is None:
                    eng_row = _cand[0]
            eng_ev = eng_row[1] if eng_row is not None else None
            regret = round(best_ev - eng_ev, 4) if isinstance(eng_ev, (int, float)) else None
            rows.append({"spot": spot.get("spot_id"), "board": "".join(spot["query"]["board"]),
                          "hand": hand, "best": best_name, "best_ev": round(best_ev, 4),
                          "engine": action, "mapped": tag,
                          "engine_ev": (round(eng_ev, 4) if isinstance(eng_ev, (int, float)) else None),
                          "regret_bb": regret, "ms": round(dt),
                          "eq_freq_hand": _freq_hand.get(tag) if tag else None,
                          "in_support_hand": (bool(_freq_hand.get(tag, 0.0) >= 0.05)
                                               if tag else None),
                          "eq_argmax_hand": (max(_freq_hand,
                                                  key=lambda k: _freq_hand[k])
                                              if _freq_hand else None)})
            if len(rows) <= 12 or len(rows) % 20 == 0:
                print(f"  {spot.get('spot_id'):<28} {hand} best={best_name:<6} "
                      f"engine={str(action):<7} regret={regret} ({dt:.0f}ms)")

    ok = [r for r in rows if r["regret_bb"] is not None]
    agree = [r for r in ok if r["mapped"] and r["mapped"] == r["best"]]
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n样本 {len(rows)} 条（可算 regret {len(ok)}）")
    if ok:
        regs = sorted(r["regret_bb"] for r in ok)
        print(f"与 solver 首选一致率: {len(agree)}/{len(ok)} = {len(agree)/len(ok)*100:.1f}%")
        print(f"regret 均值 {sum(regs)/len(regs):.3f} bb | 中位 {regs[len(regs)//2]:.3f} bb "
              f"| 最大 {regs[-1]:.3f} bb | =0 的比例 "
              f"{sum(1 for r in regs if r <= 1e-9)/len(regs)*100:.1f}%")
    _hs = [r for r in ok if isinstance(r.get("eq_freq_hand"), float)]
    # P30-9-DS：补充 >0% 支持率与逐手 EV 最优一致率（5% 阈值会惩罚合法混合动作）
    if _hs:
        _pos = [r for r in _hs if float(r.get("eq_freq_hand") or 0.0) > 0.0]
        _evb = [r for r in _hs if str(r.get("best") or "").lower()
                == str(r.get("mapped") or r.get("engine") or "").lower()]
        print(f"  支持率(>0%)：{len(_pos)}/{len(_hs)} = {len(_pos)/len(_hs)*100:.1f}%"
              "（求解器确实会打该动作；5% 阈值口径见上）")
        print(f"  逐手 EV 最优一致：{len(_evb)}/{len(_hs)} = {len(_evb)/len(_hs)*100:.1f}%"
              "（与 regret 同源的保真度主指标）")
    if _hs:
        _ins = [r for r in _hs if r.get("in_support_hand")]
        _ag = [r for r in _hs
               if r.get("eq_argmax_hand") and r.get("mapped") == r["eq_argmax_hand"]]
        print(f"手牌级支持集命中: {len(_ins)}/{len(_hs)} = {len(_ins)/len(_hs)*100:.1f}%"
              f" | 手牌级 argmax 一致: {len(_ag)}/{len(_hs)} = {len(_ag)/len(_hs)*100:.1f}%")
    print("已存:", args.out)


asyncio.run(main())
