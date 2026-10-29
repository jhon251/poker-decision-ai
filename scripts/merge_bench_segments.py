"""SOLVER-2 — 分段 bench 工件合并（每段独立落盘后合并为验收工件）。

本机环境会外杀长跑 python，故 bench 分段跑（bench_b1.py --hand-offset
--out-suffix），本脚本把各段 records 合并重算聚合（aggregate 纯确定性）：

  python scripts/merge_bench_segments.py --matchup solver2_on_vs_pivot \
      --segments s00,s01,s02 [--out docs/bench_r1/solver2_on_vs_pivot.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "bench_r1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", required=True)
    ap.add_argument("--segments", required=True,
                    help="逗号分隔的段后缀（s00,s01,...）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from external_harness.protocol.bench_match import aggregate

    segs = [s.strip() for s in args.segments.split(",") if s.strip()]
    records: list[dict] = []
    latencies: list[float] = []
    counters: dict[str, int] = {}
    parts_meta: dict[str, dict] = {}
    for s in segs:
        agg_p = OUT_DIR / f"{args.matchup}.{s}.json"
        rec_p = OUT_DIR / f"{args.matchup}.{s}.records.jsonl"
        if not rec_p.is_file():
            print(f"missing {rec_p}", file=sys.stderr)
            return 2
        rows = [json.loads(line) for line in
                rec_p.read_text(encoding="utf-8").splitlines() if line.strip()]
        records += rows
        seg_agg = (json.loads(agg_p.read_text(encoding="utf-8"))
                   if agg_p.is_file() else {})
        lat = seg_agg.get("latencies_ms") or []
        latencies += lat
        for k, v in (seg_agg.get("counters") or {}).items():
            counters[k] = counters.get(k, 0) + v
        parts_meta[s] = {"n_hands": len(rows),
                         "bb_per_100": seg_agg.get("bb_per_100"),
                         "hand_offset": seg_agg.get("hand_offset")}

    agg = aggregate(records, latencies, counters)
    action_dist: dict[str, int] = {}
    for s in segs:
        seg_agg = json.loads((OUT_DIR / f"{args.matchup}.{s}.json")
                             .read_text(encoding="utf-8"))
        dist = seg_agg.get("action_distribution") or {}
        for k, v in dist.items():
            action_dist[k] = action_dist.get(k, 0) + int(v)
    agg["action_distribution"] = dict(sorted(action_dist.items()))
    agg["action_distribution_share"] = {
        k: round(v / max(1, sum(action_dist.values())), 4)
        for k, v in sorted(action_dist.items())}
    agg["merged_segments"] = parts_meta
    agg["n_segments"] = len(segs)
    agg["merge_note"] = ("分段跑合并（外杀长跑规避）；aggregate 纯确定性重算；"
                         "hand_offset 分段 seed 推导与单次长跑逐位一致")
    agg["merged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out = pathlib.Path(args.out) if args.out else \
        OUT_DIR / f"{args.matchup}.json"
    out.write_text(json.dumps(agg, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    (OUT_DIR / f"{args.matchup}.records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    print(f"merged {len(segs)} segments, n={agg['n_hands_completed']} "
          f"bb/100={agg['bb_per_100']} CI={agg['ci95_bb_per_100']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
