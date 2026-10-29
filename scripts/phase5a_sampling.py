"""Phase 5A Step 1：玩家分层抽样（seed 复现）。

Pass 1 全量扫描 37 分片，建立玩家索引（手数 + 主 venue/stakes_bucket/table_type），
再按 venue × stakes_bucket × table_type(hu/ring) 分层，等比例配额 + 每非空格 min 1，
用固定 seed 确定性选出 5000 名玩家（门槛 n_hands>=200）。

输出（data/uoft/sampling/）:
  player_index.jsonl     全体 >=200 手玩家（主属性 + 计数）
  sampling_config.json   抽样配置（seed/门槛/cap/配额规则）
  selected_players.json  选中的 5000 名玩家（含 cap）

只读源数据；不修改任何冻结文件。开发阶段默认 seed=42。
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from scripts.uoft_mapping import stakes_bucket, table_type  # noqa: E402

NON_RING = {"hu"}  # 环桌口径 = 6max+full


def _scan_shard(path: str) -> dict:
    """Pass 1：每分片累计玩家级计数。返回可 pickle 的中间结构。"""
    p = pq.ParquetFile(path)
    hands = collections.Counter()          # pid -> n_hands
    venues: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stakes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tables: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for rg in range(p.num_row_groups):
        t = p.read_row_group(rg).to_pydict()
        for i in range(len(t["variant"])):
            if t["variant"][i] != "NT":
                continue
            players = t["players"][i] or []
            if not players:
                continue
            venue = t["venue"][i] or ""
            sb_ = stakes_bucket(t["min_bet"][i])
            tt = table_type(len(players))
            for pid in players:
                hands[pid] += 1
                if venue:
                    venues[pid][venue] += 1
                stakes[pid][sb_] += 1
                tables[pid][tt] += 1
    return {
        "hands": dict(hands), "venues": {k: dict(v) for k, v in venues.items()},
        "stakes": {k: dict(v) for k, v in stakes.items()},
        "tables": {k: dict(v) for k, v in tables.items()},
    }


def _merge(parts: list[dict]) -> dict:
    hands = collections.Counter()
    venues: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    stakes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tables: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for part in parts:
        for pid, c in part["hands"].items():
            hands[pid] += c
        for pid, c in part["venues"].items():
            venues[pid].update(c)
        for pid, c in part["stakes"].items():
            stakes[pid].update(c)
        for pid, c in part["tables"].items():
            tables[pid].update(c)
    return {"hands": hands, "venues": venues, "stakes": stakes, "tables": tables}


def _primary(counter: collections.Counter) -> str:
    return counter.most_common(1)[0][0] if counter else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\PokerDataset\UofT_CPRG\data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hands", type=int, default=200)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--n-players", type=int, default=5000)
    ap.add_argument("--out", default="data/uoft/sampling")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.dir).glob("train-*.parquet"))
    if args.limit_shards:
        files = files[: args.limit_shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pass 1 scan: {len(files)} shards ...", flush=True)
    if len(files) == 1 or args.workers <= 1:
        parts = [_scan_shard(f) for f in files]
    else:
        with mp.Pool(args.workers) as pool:
            parts = pool.map(_scan_shard, files)
    agg = _merge(parts)

    # 玩家池：>=min_hands 且主 venue 存在（HandHQ），主 stakes 非 other
    pool = []
    for pid in list(agg["hands"]):
        n = agg["hands"][pid]
        if n < args.min_hands:
            continue
        ven = _primary(agg["venues"][pid])
        if not ven or ven == "unknown":
            continue
        sb_ = _primary(agg["stakes"][pid])
        if sb_ == "other":
            continue
        tt = _primary(agg["tables"][pid])
        ring = "hu" if tt == "hu" else "ring"
        pool.append({
            "player_id": pid, "n_hands": n, "venue": ven,
            "stakes_bucket": sb_, "table_type": tt, "ring": ring,
        })
    print(f"pool (n>={args.min_hands}, venue!=None, stakes!=other): {len(pool)}", flush=True)

    # 分层：venue × stakes_bucket × ring
    strata: dict[tuple, list] = collections.defaultdict(list)
    for pl in pool:
        strata[(pl["venue"], pl["stakes_bucket"], pl["ring"])].append(pl)
    print(f"strata count: {len(strata)}", flush=True)

    # 等比例配额 + 最小 1，最大不超过 strata 规模
    total = len(pool)
    quotas: dict[tuple, int] = {}
    base = {k: len(v) * args.n_players / total for k, v in strata.items()}
    for k in strata:
        quotas[k] = max(1, int(base[k]))
    # 调整到恰好 n_players（超出则从最大 strata 削减；不足则加）
    cur = sum(quotas.values())
    keys = sorted(strata, key=lambda k: -len(strata[k]))
    i = 0
    while cur > args.n_players and i < len(keys):
        k = keys[i % len(keys)]
        if quotas[k] > 1:
            quotas[k] -= 1
            cur -= 1
        i += 1
    i = 0
    while cur < args.n_players and i < len(keys):
        k = keys[i % len(keys)]
        if quotas[k] < len(strata[k]):
            quotas[k] += 1
            cur += 1
        i += 1

    # 确定性选择：strata 内按 player_id 稳定排序 + seed shuffle
    rng = random.Random(args.seed)
    selected = []
    for k, members in strata.items():
        members_sorted = sorted(members, key=lambda x: x["player_id"])
        rng.shuffle(members_sorted)
        for pl in members_sorted[: quotas[k]]:
            pl["cap"] = args.cap
            pl["sampled_n"] = min(pl["n_hands"], args.cap)
            selected.append(pl)
    selected.sort(key=lambda x: x["player_id"])
    print(f"selected players: {len(selected)}", flush=True)

    # 落盘
    with open(out / "player_index.jsonl", "w", encoding="utf-8") as f:
        for pl in pool:
            f.write(json.dumps(pl, ensure_ascii=False) + "\n")
    config = {
        "seed": args.seed, "min_hands": args.min_hands, "cap": args.cap,
        "n_players": args.n_players, "n_shards": len(files),
        "pool_size": total, "strata_count": len(strata),
        "stratify_by": ["venue", "stakes_bucket", "ring"],
        "stakes_bucket_def": {"low": "min_bet<=1.0", "mid": "1.0<min_bet<=4.0",
                              "high": "min_bet>4.0"},
        "quotas": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in quotas.items()},
    }
    (out / "sampling_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "selected_players.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # 摘要
    from collections import Counter
    vc = Counter(pl["venue"] for pl in selected)
    sc = Counter(pl["stakes_bucket"] for pl in selected)
    rc = Counter(pl["ring"] for pl in selected)
    print("selected venue dist:", dict(vc), flush=True)
    print("selected stakes dist:", dict(sc), flush=True)
    print("selected ring dist:", dict(rc), flush=True)
    print(f"total sampled hands (player-hand entries, capped): "
          f"{sum(pl['sampled_n'] for pl in selected):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
