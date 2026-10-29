"""Phase 6 Step 6A：全量玩家充分统计聚合（full_pool_suff_stats）。

一次全量扫描 37 parquet（只读），对每个 NT 手用冻结的 map_row_full +
derive_player_events 派生事件行，按玩家累计每特征 (S, n) 充分统计。

输出（data/uoft/stats/）:
  full_pool_suff_stats.jsonl   每玩家一行（n_hands>=50 玩家池）:
     { player_id, venue, stakes_bucket, table_type, n_hands, n_ring_hands,
       n_hu_hands, first_seen, last_seen, features: {name: {S, n}} }
  population_aggregates.json    全局/分层（venue×stakes_bucket）ΣS,Σn（ring 口径）

特征（16 个基础率/差量特征；position_sensitivity 由 rfi_ep/rfi_lp 派生）:
  vpip, pfr, three_bet, four_bet, won_hand        （全手分母）
  cbet (机会门控), fold_to_3bet, fold_to_cbet, wtsd, wsd (机会门控)
  agg_freq (bets/(bets+calls) 门控), cold_call (非盲位门控)
  rfi_ep, rfi_lp, btn_steal, ep_open (位置门控)

冻结边界：只 import 冻结的 uoft_mapping；不修改 Phase 4/5A/5B 任何产物。
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from scripts.uoft_mapping import derive_player_events, map_row_full  # noqa: E402

NEEDED = ["variant", "venue", "time", "time_zone_abbreviation", "currency", "event",
          "source_file", "hand", "hand_number_in_file", "day", "month", "year",
          "min_bet", "small_bet", "big_bet", "actions", "players", "antes",
          "blinds_or_straddles", "starting_stacks", "finishing_stacks", "winnings", "seats"]

FEATURE_NAMES = ["vpip", "pfr", "three_bet", "four_bet", "won_hand", "cbet",
                 "fold_to_3bet", "fold_to_cbet", "wtsd", "wsd", "agg_freq",
                 "cold_call", "rfi_ep", "rfi_lp", "btn_steal", "ep_open"]


def _new_rec() -> dict:
    return {"h": 0, "r": 0, "u": 0, "fs": None, "ls": None,
            "v": collections.Counter(), "sb": collections.Counter(),
            "f": {k: [0, 0] for k in FEATURE_NAMES}}


def _acc(rec: dict, ev: dict) -> None:
    """把一条事件行累加进玩家记录。特征只算 ring 口径；手数计数含 HU。"""
    rec["h"] += 1
    tt = ev.get("table_type")
    if tt == "hu":
        rec["u"] += 1
    else:
        rec["r"] += 1
    ts = ev.get("ts")
    if ts:
        if rec["fs"] is None or ts < rec["fs"]:
            rec["fs"] = ts
        if rec["ls"] is None or ts > rec["ls"]:
            rec["ls"] = ts
    if ev.get("venue"):
        rec["v"][ev["venue"]] += 1
    sb = ev.get("stakes_bucket") or "other"
    rec["sb"][sb] += 1

    if tt == "hu":
        return                      # 特征统计只记 ring
    _apply(rec["f"], ev)


def _scan_shard(path: str) -> dict:
    """每分片：映射 + 事件派生 + 玩家累加。返回可 pickle 结构。"""
    pf = pq.ParquetFile(path)
    players: dict[str, dict] = {}
    glob_f = {k: [0, 0] for k in FEATURE_NAMES}          # ring 全局 ΣS,Σn
    strata = collections.defaultdict(lambda: {k: [0, 0] for k in FEATURE_NAMES})
    counts = {"rows_nt": 0, "events": 0}

    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=NEEDED).to_pydict()
        for i in range(len(t["variant"])):
            if t["variant"][i] != "NT":
                continue
            counts["rows_nt"] += 1
            row = {k: t[k][i] for k in t}
            hand, meta = map_row_full(row, i)
            if hand is None:
                continue
            for ev in derive_player_events(hand):
                counts["events"] += 1
                pid = ev["player_id"]
                rec = players.get(pid)
                if rec is None:
                    rec = players[pid] = _new_rec()
                _acc(rec, ev)
                # 全局/分层聚合（ring）
                if ev.get("table_type") != "hu":
                    _apply(glob_f, ev)
                    key = (ev.get("venue") or "(none)",
                           ev.get("stakes_bucket") or "other")
                    _apply(strata[key], ev)

    # 玩家记录 → JSON 安全（Counter → dict）
    out = {}
    for pid, rec in players.items():
        out[pid] = {
            "h": rec["h"], "r": rec["r"], "u": rec["u"],
            "fs": rec["fs"], "ls": rec["ls"],
            "v": dict(rec["v"]), "sb": dict(rec["sb"]),
            "f": {k: [rec["f"][k][0], rec["f"][k][1]] for k in FEATURE_NAMES},
        }
    return {"players": out, "glob_f": glob_f,
            "strata": {f"{v}|{s}": {k: list(val) for k, val in d.items()}
                       for (v, s), d in strata.items()},
            "counts": counts}


def _feat_delta(ev: dict) -> dict:
    """事件行 → 特征增量 {feature: (ΔS, Δn)}（ring 口径）。单一来源，三处共用。"""
    d = {k: (0, 0) for k in FEATURE_NAMES}
    for k in ("vpip", "pfr", "three_bet", "four_bet", "won_hand"):
        d[k] = (1 if ev.get(k) else 0, 1)
    if ev.get("cbet_opportunity"):
        d["cbet"] = (1 if ev.get("cbet") else 0, 1)
    if ev.get("faced_3bet"):
        d["fold_to_3bet"] = (1 if ev.get("folded_to_3bet") else 0, 1)
    if ev.get("faced_cbet"):
        d["fold_to_cbet"] = (1 if ev.get("folded_to_cbet") else 0, 1)
    if ev.get("saw_flop"):
        d["wtsd"] = (1 if ev.get("went_to_showdown") else 0, 1)
    if ev.get("went_to_showdown"):
        d["wsd"] = (1 if ev.get("won_at_showdown") else 0, 1)
    nb = (ev.get("bets") or 0) + (ev.get("calls") or 0)
    if nb > 0:
        # Aggression Frequency = bets/(bets+calls)：加权分母（冻结 stats.py 同定义）
        d["agg_freq"] = (ev.get("bets") or 0, nb)
    pos = ev.get("position")
    if pos is not None and pos not in ("SB", "BB"):
        d["cold_call"] = (1 if (ev.get("vpip") and not ev.get("pfr")) else 0, 1)
    if pos == "EP" and ev.get("rfi_opp"):
        did = 1 if ev.get("rfi_did") else 0
        d["rfi_ep"] = (did, 1)
        d["ep_open"] = (did, 1)
    if pos in ("LP", "BTN") and ev.get("rfi_opp"):
        d["rfi_lp"] = (1 if ev.get("rfi_did") else 0, 1)
    if pos == "BTN" and ev.get("rfi_opp"):
        d["btn_steal"] = (1 if ev.get("rfi_did") else 0, 1)
    return d


def _apply(dest: dict, ev: dict, ring_only: bool = True) -> None:
    """把事件行的特征增量写入 dest（S,n 对）。"""
    delta = _feat_delta(ev)
    for k in FEATURE_NAMES:
        ds, dn = delta[k]
        dest[k][0] += ds
        dest[k][1] += dn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\PokerDataset\UofT_CPRG\data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-hands", type=int, default=50)
    ap.add_argument("--out", default="data/uoft/stats")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.dir).glob("train-*.parquet"))
    if args.limit_shards:
        files = files[: args.limit_shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"scan: {len(files)} shards ...", flush=True)
    if len(files) == 1 or args.workers <= 1:
        parts = [_scan_shard(f) for f in files]
    else:
        with mp.Pool(args.workers) as pool:
            parts = pool.map(_scan_shard, files)

    # 合并玩家记录（同玩家可能跨分片）
    merged: dict[str, dict] = {}
    glob_f = {k: [0, 0] for k in FEATURE_NAMES}
    strata: dict[str, dict] = collections.defaultdict(
        lambda: {k: [0, 0] for k in FEATURE_NAMES})
    counts = {"rows_nt": 0, "events": 0}
    for part in parts:
        counts["rows_nt"] += part["counts"]["rows_nt"]
        counts["events"] += part["counts"]["events"]
        for k in FEATURE_NAMES:
            glob_f[k][0] += part["glob_f"][k][0]
            glob_f[k][1] += part["glob_f"][k][1]
        for key, d in part["strata"].items():
            for k in FEATURE_NAMES:
                strata[key][k][0] += d[k][0]
                strata[key][k][1] += d[k][1]
        for pid, rec in part["players"].items():
            m = merged.get(pid)
            if m is None:
                merged[pid] = rec
                continue
            m["h"] += rec["h"]
            m["r"] += rec["r"]
            m["u"] += rec["u"]
            if rec["fs"] and (m["fs"] is None or rec["fs"] < m["fs"]):
                m["fs"] = rec["fs"]
            if rec["ls"] and (m["ls"] is None or rec["ls"] > m["ls"]):
                m["ls"] = rec["ls"]
            for v, c in rec["v"].items():
                m["v"][v] = m["v"].get(v, 0) + c
            for s, c in rec["sb"].items():
                m["sb"][s] = m["sb"].get(s, 0) + c
            for k in FEATURE_NAMES:
                m["f"][k][0] += rec["f"][k][0]
                m["f"][k][1] += rec["f"][k][1]

    # 写全量充分统计（n_hands >= min_hands）
    pool_out = []
    n_all = len(merged)
    for pid, rec in merged.items():
        if rec["h"] < args.min_hands:
            continue
        venue = max(rec["v"], key=rec["v"].get) if rec["v"] else None
        sb_bucket = max(rec["sb"], key=rec["sb"].get) if rec["sb"] else "other"
        if rec["u"] > 0 and rec["r"] > 0:
            table_type = "mixed"
        elif rec["u"] > 0:
            table_type = "hu"
        else:
            table_type = "ring"
        pool_out.append({
            "player_id": pid, "venue": venue, "stakes_bucket": sb_bucket,
            "table_type": table_type, "n_hands": rec["h"],
            "n_ring_hands": rec["r"], "n_hu_hands": rec["u"],
            "first_seen": rec["fs"], "last_seen": rec["ls"],
            "features": rec["f"],
        })
    pool_out.sort(key=lambda x: x["player_id"])
    pool_path = out / "full_pool_suff_stats.jsonl"
    with open(pool_path, "w", encoding="utf-8") as w:
        for rec in pool_out:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    agg = {
        "min_hands": args.min_hands,
        "n_players_all": n_all,
        "n_players_pool": len(pool_out),
        "counts": counts,
        "global_features": glob_f,
        "strata_features": dict(strata),
    }
    (out / "population_aggregates.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"players_all={n_all} pool(>= {args.min_hands})={len(pool_out)} "
          f"rows_nt={counts['rows_nt']} events={counts['events']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
