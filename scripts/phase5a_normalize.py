"""Phase 5A Step 2：normalized_full（canonical JSONL）+ feature_ready（event table）。

流程：
  1. mp 并行：每分片只读扫描，凡含选中玩家的 NT 手 → map_row_full + validate_hand
     → 写 canonical chunk + 该分片的 event 临时行（未封顶）
  2. 顺序封顶：按分片顺序读 event 临时行，对每个选中玩家只保留前 cap 手
     → 写 feature_ready/player_event_table.jsonl（最终）

输出：
  data/uoft/normalized/chunk_<idx>.jsonl        canonical + quality（一手一行）
  data/uoft/feature_ready/player_event_table.jsonl   每玩家每手一行（封顶后）
  data/uoft/chunks_manifest.json                chunk 行数 + sha256

只读源数据；不修改冻结文件。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from poker_decision_ai.core.schemas import to_dict  # noqa: E402
from poker_decision_ai.domain.poker.validator import validate_hand  # noqa: E402
from scripts.uoft_mapping import derive_player_events, map_row_full  # noqa: E402

NEEDED = ["variant", "venue", "time", "time_zone_abbreviation", "currency", "event",
          "source_file", "hand", "hand_number_in_file", "day", "month", "year",
          "min_bet", "small_bet", "big_bet", "actions", "players", "antes",
          "blinds_or_straddles", "starting_stacks", "finishing_stacks", "winnings", "seats"]


def _shard_to_chunk(args: tuple) -> dict:
    path, idx, selected, out_norm, out_ev = args
    selected_set = set(selected)
    pf = pq.ParquetFile(path)
    norm_path = Path(out_norm) / f"chunk_{idx:03d}.jsonl"
    ev_path = Path(out_ev) / f"ev_tmp_{idx:03d}.jsonl"
    hands = 0
    events = 0
    nf = open(norm_path, "w", encoding="utf-8")
    ef = open(ev_path, "w", encoding="utf-8")
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=NEEDED).to_pydict()
        for i in range(len(t["variant"])):
            if t["variant"][i] != "NT":
                continue
            players = t["players"][i] or []
            if not any(p in selected_set for p in players):
                continue
            row = {k: t[k][i] for k in t}
            hand, meta = map_row_full(row, i)
            if hand is None:
                continue
            rep = validate_hand(hand)
            rec = {
                "canonical": to_dict(hand),
                "quality": {"status": rep.status,
                            "warnings": [c.name for c in rep.checks
                                         if c.status != "valid"]},
            }
            nf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hands += 1
            for ev in derive_player_events(hand):
                if ev["player_id"] in selected_set:
                    ef.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    events += 1
    nf.close()
    ef.close()
    sha = hashlib.sha256(open(norm_path, "rb").read()).hexdigest()
    return {"idx": idx, "path": path, "hands": hands, "events": events, "sha256": sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\PokerDataset\UofT_CPRG\data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selection", default="data/uoft/sampling/selected_players.json")
    ap.add_argument("--out", default="data/uoft")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    sel = json.load(open(args.selection, encoding="utf-8"))
    selected_ids = [pl["player_id"] for pl in sel]
    cap_by_player = {pl["player_id"]: pl["cap"] for pl in sel}
    files = sorted(str(p) for p in Path(args.dir).glob("train-*.parquet"))
    if args.limit_shards:
        files = files[: args.limit_shards]

    out_norm = Path(args.out) / "normalized"
    out_ev = Path(args.out) / "feature_ready"
    out_norm.mkdir(parents=True, exist_ok=True)
    out_ev.mkdir(parents=True, exist_ok=True)

    tasks = [(f, i, selected_ids, str(out_norm), str(out_ev))
             for i, f in enumerate(files)]
    print(f"normalize: {len(tasks)} shards, {len(selected_ids)} players ...", flush=True)

    results = []
    if len(tasks) == 1 or args.workers <= 1:
        for t in tasks:
            results.append(_shard_to_chunk(t))
            print(f"  shard {results[-1]['idx']}: hands={results[-1]['hands']} "
                  f"events={results[-1]['events']}", flush=True)
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_shard_to_chunk, tasks):
                results.append(r)
                print(f"  shard {r['idx']}: hands={r['hands']} events={r['events']}",
                      flush=True)
    results.sort(key=lambda r: r["idx"])

    # 顺序封顶：按分片顺序读 ev_tmp，每玩家保留前 cap 手，并收集存活 hand_id 集合
    ev_final = out_ev / "player_event_table.jsonl"
    kept = collections.Counter()
    kept_hand_ids: set[str] = set()
    total_events = 0
    n_kept = 0
    with open(ev_final, "w", encoding="utf-8") as w:
        for r in results:
            ev_tmp = out_ev / f"ev_tmp_{r['idx']:03d}.jsonl"
            if not ev_tmp.exists():
                continue
            with open(ev_tmp, encoding="utf-8") as f:
                for line in f:
                    total_events += 1
                    ev = json.loads(line)
                    pid = ev["player_id"]
                    if kept[pid] < cap_by_player[pid]:
                        w.write(line)
                        kept[pid] += 1
                        kept_hand_ids.add(ev["hand_id"])
                        n_kept += 1
            ev_tmp.unlink()  # 清理临时
    print(f"event table: total_events={total_events} kept={n_kept} "
          f"distinct_hands={len(kept_hand_ids)}", flush=True)

    # 规范化层封顶：只保留「至少一个选中玩家仍在 cap 内」的手（与 event table 一致）
    # 逐分片重写 chunk，剔除 hand_key 不在 kept_hand_ids 中的手
    final_hands = 0
    final_chunks = []
    for r in results:
        norm_path = out_norm / f"chunk_{r['idx']:03d}.jsonl"
        tmp_path = out_norm / f"chunk_{r['idx']:03d}.jsonl.tmp"
        kept_lines = 0
        with open(norm_path, encoding="utf-8") as f, \
             open(tmp_path, "w", encoding="utf-8") as w:
            for line in f:
                rec = json.loads(line)
                hk = rec["canonical"]["meta"].get("hand_key")
                if hk in kept_hand_ids:
                    w.write(line)
                    kept_lines += 1
        os.replace(tmp_path, norm_path)
        final_hands += kept_lines
        sha = hashlib.sha256(open(norm_path, "rb").read()).hexdigest()
        final_chunks.append({"idx": r["idx"], "file": f"chunk_{r['idx']:03d}.jsonl",
                             "hands": kept_lines, "sha256": sha})
    print(f"normalized (capped): {final_hands} hands", flush=True)

    # chunks_manifest
    manifest = {
        "n_shards": len(results),
        "chunks": final_chunks,
        "total_hands": final_hands,
        "total_events": total_events,
        "events_kept": n_kept,
        "distinct_hands_kept": len(kept_hand_ids),
        "cap": cap_by_player[next(iter(cap_by_player))],
    }
    (Path(args.out) / "chunks_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: hands={manifest['total_hands']} kept_events={n_kept}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
