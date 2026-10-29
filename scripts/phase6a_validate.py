"""Phase 6 Step 6A.5：full_pool_suff_stats 验证（仅分析，不改设计）。

5 项验证：
  1. suff_stats 玩家数量统计（pool 分布）
  2. 与原始数据 hand/event 数量一致性（对照 Phase 4 qualification 全量数）
  3. 随机 100 玩家与 Phase 5A event_table 聚合结果交叉验证（逐特征 S,n 一致）
  4. ring/HU 分布检查（n_hands = n_ring + n_hu 守恒）
  5. 各 feature opportunity 分布检查

输出：stdout 报告（不写产物文件）。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

FEATURES = ["vpip", "pfr", "three_bet", "four_bet", "won_hand", "cbet",
            "fold_to_3bet", "fold_to_cbet", "wtsd", "wsd", "agg_freq",
            "cold_call", "rfi_ep", "rfi_lp", "btn_steal", "ep_open"]

# Phase 4 qualification 全量参考值
REF = {"players_all": 278171, "rows_nt": 21616099, "events": 116922023}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--agg", default="data/uoft/stats/population_aggregates.json")
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=== 1. 玩家数量统计 ===")
    n_pool = 0
    h = []
    ring = []
    hu = []
    for line in open(args.stats, encoding="utf-8"):
        rec = json.loads(line)
        n_pool += 1
        h.append(rec["n_hands"])
        ring.append(rec["n_ring_hands"])
        hu.append(rec["n_hu_hands"])
    hs = sorted(h)
    def pct(vals, q):
        if not vals:
            return 0
        i = int(len(vals) * q)
        return vals[i]
    print(f"pool players: {n_pool}")
    print(f"n_hands: min={hs[0]} p25={pct(hs,0.25)} med={pct(hs,0.5)} "
          f"p75={pct(hs,0.75)} p90={pct(hs,0.9)} max={hs[-1]}")

    agg = json.load(open(args.agg, encoding="utf-8"))
    print("=== 2. 与原始数据一致性 ===")
    print(f"players_all: {agg['n_players_all']} (ref {REF['players_all']}) "
          f"{'OK' if agg['n_players_all']==REF['players_all'] else 'MISMATCH'}")
    print(f"rows_nt: {agg['counts']['rows_nt']} (ref {REF['rows_nt']}) "
          f"{'OK' if agg['counts']['rows_nt']==REF['rows_nt'] else 'MISMATCH'}")
    print(f"events: {agg['counts']['events']} (ref {REF['events']}) "
          f"{'OK' if agg['counts']['events']==REF['events'] else 'MISMATCH'}")

    print("=== 3. 随机 100 玩家交叉验证（vs Phase 5A event_table 重聚合）===")
    pool_ids = []
    pool_map = {}
    for line in open(args.stats, encoding="utf-8"):
        rec = json.loads(line)
        pool_ids.append(rec["player_id"])
        pool_map[rec["player_id"]] = rec
    # event table 中实际存在的玩家（5000 MVP 玩家）
    et_ids = set()
    for line in open(args.events, encoding="utf-8"):
        et_ids.add(json.loads(line)["player_id"])
    common = sorted(set(pool_ids) & et_ids)
    rng = random.Random(args.seed)
    sample = rng.sample(common, 100)
    # 从 event table 重聚合这 100 人（ring 口径）
    et_agg = {pid: {"f": {k: [0, 0] for k in FEATURES}, "ring": 0}
              for pid in sample}
    for line in open(args.events, encoding="utf-8"):
        ev = json.loads(line)
        if ev["player_id"] not in et_agg:
            continue
        if ev.get("table_type") == "hu":
            continue
        et_agg[ev["player_id"]]["ring"] += 1
        dest = et_agg[ev["player_id"]]["f"]
        _apply(dest, ev)
    # 分组：et 未封顶玩家（ring 手数 == pool）严格相等；
    # 封顶玩家（et < pool）验证子集关系（S_et<=S_pool 且 n_et<=n_pool）
    n_uncapped = 0
    n_capped = 0
    mism_strict = 0
    mism_subset = 0
    for pid in sample:
        capped = et_agg[pid]["ring"] < pool_map[pid]["n_ring_hands"]
        if capped:
            n_capped += 1
        else:
            n_uncapped += 1
        for k in FEATURES:
            a = pool_map[pid]["features"][k]
            b = et_agg[pid]["f"][k]
            if not capped:
                if a != b:
                    mism_strict += 1
                    if mism_strict <= 3:
                        print(f"  STRICT MISMATCH {pid[:14]} {k}: pool={a} et={b}")
            else:
                if b[0] > a[0] or b[1] > a[1]:
                    mism_subset += 1
                    if mism_subset <= 3:
                        print(f"  SUBSET MISMATCH {pid[:14]} {k}: pool={a} et={b}")
    print(f"sample={len(sample)} uncapped={n_uncapped} capped={n_capped}")
    print(f"strict mismatches={mism_strict} (应为 0) | subset violations={mism_subset} (应为 0)")

    print("=== 4. ring/HU 分布检查 ===")
    viol = 0
    n_mixed = 0
    for line in open(args.stats, encoding="utf-8"):
        rec = json.loads(line)
        if rec["n_hands"] != rec["n_ring_hands"] + rec["n_hu_hands"]:
            viol += 1
        if rec["table_type"] == "mixed":
            n_mixed += 1
    print(f"n_hands == n_ring + n_hu 守恒违反: {viol}")
    print(f"mixed 玩家: {n_mixed}")
    print(f"ring 总手: {sum(ring):,}  hu 总手: {sum(hu):,}")

    print("=== 5. feature opportunity 分布 ===")
    for k in FEATURES:
        ns = sorted(rec["features"][k][1] for rec in pool_map.values())
        zeros = sum(1 for v in ns if v == 0)
        print(f"  {k:14s} med={pct(ns,0.5):6d} p10={pct(ns,0.1):6d} "
              f"zero_players={zeros} ({100*zeros/n_pool:.1f}%)")
    return 0


def _apply(dest: dict, ev: dict) -> None:
    """与 phase6a_suff_stats._feat_delta 相同语义（验证脚本独立重实现，防止同源错误）。"""
    d = dest
    for k in ("vpip", "pfr", "three_bet", "four_bet", "won_hand"):
        d[k][1] += 1
        if ev.get(k):
            d[k][0] += 1
    if ev.get("cbet_opportunity"):
        d["cbet"][1] += 1
        if ev.get("cbet"):
            d["cbet"][0] += 1
    if ev.get("faced_3bet"):
        d["fold_to_3bet"][1] += 1
        if ev.get("folded_to_3bet"):
            d["fold_to_3bet"][0] += 1
    if ev.get("faced_cbet"):
        d["fold_to_cbet"][1] += 1
        if ev.get("folded_to_cbet"):
            d["fold_to_cbet"][0] += 1
    if ev.get("saw_flop"):
        d["wtsd"][1] += 1
        if ev.get("went_to_showdown"):
            d["wtsd"][0] += 1
    if ev.get("went_to_showdown"):
        d["wsd"][1] += 1
        if ev.get("won_at_showdown"):
            d["wsd"][0] += 1
    nb = (ev.get("bets") or 0) + (ev.get("calls") or 0)
    if nb > 0:
        d["agg_freq"][1] += nb
        d["agg_freq"][0] += ev.get("bets") or 0
    pos = ev.get("position")
    if pos is not None and pos not in ("SB", "BB"):
        d["cold_call"][1] += 1
        if ev.get("vpip") and not ev.get("pfr"):
            d["cold_call"][0] += 1
    if pos == "EP" and ev.get("rfi_opp"):
        did = 1 if ev.get("rfi_did") else 0
        d["rfi_ep"][1] += 1
        d["rfi_ep"][0] += did
        d["ep_open"][1] += 1
        d["ep_open"][0] += did
    if pos in ("LP", "BTN") and ev.get("rfi_opp"):
        d["rfi_lp"][1] += 1
        if ev.get("rfi_did"):
            d["rfi_lp"][0] += 1
    if pos == "BTN" and ev.get("rfi_opp"):
        d["btn_steal"][1] += 1
        if ev.get("rfi_did"):
            d["btn_steal"][0] += 1


if __name__ == "__main__":
    raise SystemExit(main())
