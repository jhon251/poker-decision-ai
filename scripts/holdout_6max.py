#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FQ：6-max **留出验证**（非循环评测）。

问题：既有 6-max 基准是"从库里抽节点 → 让引擎决策" ⇒ 引擎读到的是**它自己那张节点**，
99.9% 很大程度只证明"查表/牌面对上了"，**不证明**策略能迁移到**没见过的牌面**。

做法（不动真实库）：
  1. 按族（spot_id 前三段，如 `3bp-BTN-BB`）把**一半牌面**的全部节点（含深层 `-p*`）划为**留出集**；
  2. 用其余节点构建**精简副本目录** `data/solver/spots_6max_holdout_lib/`；
  3. 进程内把 **agent 的** `get_spot_table_6max` 指向精简副本 ⇒ 引擎等价于"从未见过这些牌面"；
  4. 对留出集的节点抽样决策，按该节点**自身逐手频率**评分（支持率/主选一致/频率质量）；
  5. **对照**：同一批状态在**完整库**下再跑一遍（同源口径）⇒ 直接量化"同源高估"幅度；
  6. 同时记录命中模式（exact / approx(桶代表) / miss）与命中的节点。

用法：
    python scripts/holdout_6max.py --hands 8 --level board
    python scripts/holdout_6max.py --hands 8 --level family
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import pathlib
import random
import shutil
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "data" / "solver" / "spots_6max"
REDUCED = ROOT / "data" / "solver" / "spots_6max_holdout_lib"

# 复用基准脚本的观测构造
_spec = importlib.util.spec_from_file_location("b6", ROOT / "scripts" / "bench_6max_library.py")
b6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b6)


def _family(rec: dict) -> str:
    return "-".join(str(rec.get("spot_id") or "").split("-")[:3])


def _board_key(rec: dict) -> str:
    parts = str(rec.get("spot_id") or "").split("-")
    return parts[3] if len(parts) > 3 else ""


def load_records() -> list[dict]:
    out = []
    for f in sorted(LIB.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        if d.get("solved") and d.get("actions"):
            out.append(d)
    return out


def split_holdout(records: list[dict], level: str) -> tuple[list[dict], list[dict]]:
    """返回 (留出集, 保留集)。level=board：每族隔一取一；level=family：留出整族。"""
    if level == "family":
        fams = sorted({_family(r) for r in records})
        held_fams = set(fams[::2])                     # 隔族留出
        hold = [r for r in records if _family(r) in held_fams]
        keep = [r for r in records if _family(r) not in held_fams]
        return hold, keep
    groups: dict[str, list[str]] = collections.defaultdict(set)
    for r in records:
        groups[_family(r)].add(_board_key(r))
    hold_boards: set[tuple[str, str]] = set()
    for fam, boards in groups.items():
        for b in sorted(boards)[::2]:                  # 隔一取一
            hold_boards.add((fam, b))
    hold = [r for r in records if (_family(r), _board_key(r)) in hold_boards]
    keep = [r for r in records if (_family(r), _board_key(r)) not in hold_boards]
    return hold, keep


def build_reduced(keep: list[dict]) -> None:
    if REDUCED.exists():
        shutil.rmtree(REDUCED)
    REDUCED.mkdir(parents=True)
    for r in keep:
        (REDUCED / f"{r['spot_id']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")


def _hands_of(rec: dict, n: int, seed: str) -> list[tuple[int, str]]:
    hands = rec.get("private_cards_actor") or []
    fp = rec.get("freq_per_hand") or {}
    if not hands or not fp:
        return []
    rnd = random.Random(f"{rec.get('spot_id')}|holdout|{seed}")
    cand = list(range(len(hands)))
    rnd.shuffle(cand)
    out = []
    for i in cand:
        if len(out) >= n:
            break
        if any((fp.get(k) or [0] * (i + 1))[i] > 0 for k in fp):
            out.append((i, hands[i]))
    return out


async def run_states(recs: list[dict], hands: int, seed: str, table_dir: pathlib.Path,
                      log: list[dict]) -> list[dict]:
    """在指定库目录下跑一批状态（进程内改写 agent 的表引用）。"""
    import poker_decision_ai.core.providers.spot_table_6max as st6
    import poker_decision_ai.agents.solver.agent as solver_agent
    from poker_decision_ai import cli as cli_mod
    from poker_decision_ai.realtime import RealtimeRunner

    table = st6.SpotTable6max(str(table_dir))
    # 关键：agent 在**调用时**从源模块导入该函数 ⇒ 必须同时改写**源模块**的属性，
    # 否则仍会拿到默认（完整）库（P30-9-FQ 实测：只改 agent 名字会导致留出集仍 exact ✗）。
    st6.get_spot_table_6max = lambda *a, **k: table
    solver_agent.get_spot_table_6max = lambda *a, **k: table
    st6._CACHE.clear()
    print(f"    [表] dir={table.dir.name} 节点数={table.count()}", file=sys.stderr)

    _orig_lookup = st6.SpotTable6max.lookup

    def _lk(self, q):
        h = _orig_lookup(self, q)
        log.append({"hit": (h or {}).get("model"),
                     "approx": (h or {}).get("is_approximate"),
                     "board_approx": (h or {}).get("board_approximate"),
                     "q_family": q.get("family"), "q_actor": q.get("actor"),
                     "q_board": "".join(str(c) for c in (q.get("board") or []))})
        return h

    st6.SpotTable6max.lookup = _lk
    rt = cli_mod.build_runtime()
    rr = RealtimeRunner(rt)
    rows = []
    for rec in recs:
        fp = rec.get("freq_per_hand") or {}
        for i, hand in _hands_of(rec, hands, seed):
            obs = b6._observation(rec, [hand[:2], hand[2:4]])
            try:
                rep = await rr.decide({"kind": "cash_hand", "hero_id": "Hero",
                                        "vision_observation": obs, "session_id": "holdout"})
                dd = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
                fd = dd.get("final_decision") or {}
                action, reason = str(fd.get("action") or ""), str(fd.get("reasoning") or "")
            except Exception as e:                            # noqa: BLE001
                action, reason = f"ERROR:{type(e).__name__}", ""
            sup = {k: float((v[i] if i < len(v) else 0.0)) for k, v in fp.items()}
            prim = max(sup, key=lambda k: sup[k]) if sup else None
            rows.append({"spot": rec.get("spot_id"), "family": _family(rec),
                          "hand": hand, "engine": action, "primary": prim,
                          "freq_hand": sup.get(action, 0.0),
                          "in_support": sup.get(action, 0.0) > 0.0,
                          "anchored": ("求解器锚定" in reason or "偏离" in reason
                                        or "6-max" in reason)})
    st6.SpotTable6max.lookup = _orig_lookup
    return rows


def summarize(rows: list[dict], log: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    sup = sum(1 for r in rows if r["in_support"])
    pri = sum(1 for r in rows if r["engine"] == r["primary"])
    mass = statistics.fmean([r["freq_hand"] for r in rows])
    modes = collections.Counter()
    for entry in log:
        hit = entry.get("hit")                # 注意：None 不可用 `or`（假值陷阱）
        if hit is None:
            modes["miss"] += 1
        elif entry.get("approx"):
            modes["approx(桶代表)"] += 1
        else:
            modes["exact"] += 1
    return {"n": n, "support": round(sup / n, 4), "primary": round(pri / n, 4),
             "freq_mass": round(mass, 4), "hit_modes": dict(modes),
             "anchor_rate": round(sum(1 for r in rows if r["anchored"]) / n, 4)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--level", choices=("board", "family"), default="board")
    ap.add_argument("--max-nodes", type=int, default=40,
                     help="留出集中参与评测的节点数上限（控制耗时）")
    ap.add_argument("--seed", default="holdout")
    args = ap.parse_args()

    recs = load_records()
    hold, keep = split_holdout(recs, args.level)
    print(f"库内节点 {len(recs)} | 留出 {len(hold)} | 保留 {len(keep)}（level={args.level}）")
    build_reduced(keep)
    print(f"精简副本目录: {REDUCED.relative_to(ROOT)}（{len(keep)} 个节点）")

    # 评测集：留出集中节点种类均衡抽样
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for r in hold:
        by_kind[str(r.get("node"))].append(r)
    pick: list[dict] = []
    per_kind = max(1, args.max_nodes // max(len(by_kind), 1))
    for _k, lst in by_kind.items():
        pick.extend(lst[:per_kind])
    print(f"评测节点: {len(pick)}（{ {k: min(len(v), per_kind) for k, v in by_kind.items()} }）")

    log_hold: list[dict] = []
    rows_hold = await run_states(pick, args.hands, args.seed, REDUCED, log_hold)
    log_full: list[dict] = []
    rows_full = await run_states(pick, args.hands, args.seed, LIB, log_full)

    res_hold = summarize(rows_hold, log_hold)
    # 诊断：查询里是否带 family、命中的是否同族
    _with_fam = sum(1 for e in log_hold if e.get('q_family'))
    print(f"    [诊断] 带 family 的查询 {_with_fam}/{len(log_hold)}；样例:")
    for e in log_hold[:3]:
        print('      q_family=', e.get('q_family'), '| q_board=', e.get('q_board'), '| → ', e.get('hit'))
    res_full = summarize(rows_full, log_full)
    print("\n=== 留出（精简库，引擎未见该牌面） ===")
    print(f"  决策 {res_hold.get('n')} | 支持 {res_hold.get('support')} | 主选 {res_hold.get('primary')}"
          f" | 频率质量 {res_hold.get('freq_mass')} | 锚定 {res_hold.get('anchor_rate')}")
    print(f"  命中模式: {res_hold.get('hit_modes')}")
    print("=== 对照（完整库，同源口径） ===")
    print(f"  决策 {res_full.get('n')} | 支持 {res_full.get('support')} | 主选 {res_full.get('primary')}"
          f" | 频率质量 {res_full.get('freq_mass')} | 锚定 {res_full.get('anchor_rate')}")
    print(f"  命中模式: {res_full.get('hit_modes')}")

    out = ROOT / "docs" / "holdout_6max.json"
    out.write_text(json.dumps({"level": args.level, "hands": args.hands,
                                "nodes_total": len(recs), "held_out": len(hold),
                                "kept": len(keep), "evaluated_nodes": len(pick),
                                "holdout": res_hold, "control": res_full,
                                "rows_holdout": rows_hold[:400],
                                "rows_control": rows_full[:400]},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {out.relative_to(ROOT)}")
    print(f"同源高估：支持 +{round((res_full.get('support',0)-res_hold.get('support',0))*100,1)}pp"
          f" | 主选 +{round((res_full.get('primary',0)-res_hold.get('primary',0))*100,1)}pp"
          f" | 频率质量 +{round(res_full.get('freq_mass',0)-res_hold.get('freq_mass',0),3)}")


asyncio.run(main())
