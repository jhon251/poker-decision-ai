"""Phase 5A Step 3：Dataset Manifest（JSON + Markdown）。

汇总 normalized_full + feature_ready 的元数据，记录 dataset_version / source /
hand_count / player_count / date_range / quality flags / 派生哈希链。

输出:
  data/uoft/manifest.json
  docs/DATASET_MANIFEST_UOFT.md

只读；不修改冻结文件（自行渲染 Markdown，不 import 冻结 dataset_tool 私有函数）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _count_events(ev_path: Path) -> dict:
    players = collections.Counter()
    venues = collections.Counter()
    stakes = collections.Counter()
    table_types = collections.Counter()
    dates = []
    pos = collections.Counter()
    n = 0
    with open(ev_path, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            n += 1
            players[ev["player_id"]] += 1
            if ev.get("venue"):
                venues[ev["venue"]] += 1
            stakes[ev.get("stakes_bucket") or "unknown"] += 1
            table_types[ev.get("table_type") or "unknown"] += 1
            if ev.get("position"):
                pos[ev["position"]] += 1
            if ev.get("ts"):
                dates.append(ev["ts"][:10])
    dates_sorted = sorted(dates)
    return {
        "events": n,
        "distinct_players": len(players),
        "player_hands": sorted(players.values(), reverse=True),
        "venues": dict(venues), "stakes_buckets": dict(stakes),
        "table_types": dict(table_types), "positions": dict(pos),
        "date_min": dates_sorted[0] if dates_sorted else None,
        "date_max": dates_sorted[-1] if dates_sorted else None,
        "n_with_ts": len(dates),
    }


def _percentile(sorted_vals: list, q: float) -> int:
    if not sorted_vals:
        return 0
    i = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * q)))
    return sorted_vals[i]


def _markdown(m: dict) -> str:
    ph = m["player_hand_distribution"]
    return f"""# Dataset Manifest — UofT CPRG（Phase 5A 抽样规范化）

- 生成时间: {m['generated_at']}
- dataset_version: {m['dataset_version']}
- schema_version: {m['schema_version']}

## 来源与许可

- 数据集: {m['source']['dataset']}
- 原始出处: {m['source']['zenodo']}
- 许可: {m['source']['license']}
- 抽样配置: seed={m['sampling']['seed']}, min_hands={m['sampling']['min_hands']}, cap={m['sampling']['cap']}, n_players={m['sampling']['n_players']}

## 规模

- 规范化手数（normalized_full）: {m['hand_count']}
- 事件行数（feature_ready）: {m['event_count']}
- 独立玩家: {m['player_count']}
- 时间范围: {m['date_range']['min']} ~ {m['date_range']['max']}（{m['date_range']['n_with_ts']} 行带时间戳）

## 质量标志

- 校验状态分布（全量 21.6M NT 手，来自 qualification 阶段）: valid=0 / warning={m['quality']['warning']} / invalid={m['quality']['invalid']}
- 抽样子集 invalid: {m['quality']['subset_invalid']}（normalize 阶段复检）
- 已知问题:
  - {m['quality']['known_issues'][0]}
  - {m['quality']['known_issues'][1]}

## 玩家样本量分布（抽样 5000 人）

- min {ph['min']} / p25 {ph['p25']} / 中位 {ph['median']} / p75 {ph['p75']} / p90 {ph['p90']} / max {ph['max']}

## 分布

- venue: {json.dumps(m['distributions']['venues'], ensure_ascii=False)}
- stakes_bucket: {json.dumps(m['distributions']['stakes_buckets'], ensure_ascii=False)}
- table_type: {json.dumps(m['distributions']['table_types'], ensure_ascii=False)}
- 派生哈希链: normalized_full 分块 sha256 见 chunks_manifest.json；event table 由 canonical 确定性派生
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/uoft/sampling/selected_players.json")
    ap.add_argument("--sampling-config", default="data/uoft/sampling/sampling_config.json")
    ap.add_argument("--chunks", default="data/uoft/chunks_manifest.json")
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--out", default="data/uoft")
    ap.add_argument("--doc", default="docs/DATASET_MANIFEST_UOFT.md")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, encoding="utf-8"))
    sel = json.load(open(args.selection, encoding="utf-8"))
    scfg = json.load(open(args.sampling_config, encoding="utf-8"))
    ev_stats = _count_events(Path(args.events))

    # 抽样子集 invalid 计数（来自 normalize 阶段的 quality 字段）
    subset_invalid = 0
    import glob
    for f in sorted(glob.glob(str(Path(args.out) / "normalized" / "chunk_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            q = json.loads(line).get("quality", {})
            if q.get("status") == "invalid":
                subset_invalid += 1

    ph = ev_stats["player_hands"]
    ph_sorted = sorted(ph)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": "uoft-cprg-v2-normalized-5.0",
        "schema_version": "5.0",
        "source": {"dataset": "UofT CPRG Poker Hand Histories",
                   "zenodo": "https://zenodo.org/records/13997158",
                   "license": "cc-by-4.0",
                   "parquet_files": chunks.get("n_shards")},
        "sampling": {"seed": scfg["seed"], "min_hands": scfg["min_hands"],
                     "cap": scfg["cap"], "n_players": scfg["n_players"]},
        "hand_count": chunks["total_hands"],
        "event_count": ev_stats["events"],
        "player_count": ev_stats["distinct_players"],
        "date_range": {"min": ev_stats["date_min"], "max": ev_stats["date_max"],
                       "n_with_ts": ev_stats["n_with_ts"]},
        "quality": {
            "warning": 21616099, "invalid": 0,   # qualification 阶段全量结果
            "subset_invalid": subset_invalid,
            "known_issues": [
                "Ongame winnings/finishing 记账口径不一致（1.43M 手筹码一致性 warning）",
                "96.2% 手牌无开局真牌（观察者视角，无 hero）",
            ],
        },
        "player_hand_distribution": {
            "min": ph_sorted[0] if ph_sorted else 0,
            "p25": _percentile(ph_sorted, 0.25),
            "median": _percentile(ph_sorted, 0.5),
            "p75": _percentile(ph_sorted, 0.75),
            "p90": _percentile(ph_sorted, 0.90),
            "max": ph_sorted[-1] if ph_sorted else 0,
        },
        "distributions": {
            "venues": ev_stats["venues"], "stakes_buckets": ev_stats["stakes_buckets"],
            "table_types": ev_stats["table_types"], "positions": ev_stats["positions"],
        },
    }
    (Path(args.out) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.doc).write_text(_markdown(manifest), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("dataset_version", "hand_count", "event_count",
                       "player_count", "date_range")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
