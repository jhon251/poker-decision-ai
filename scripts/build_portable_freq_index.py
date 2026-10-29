"""EXE-1 — 构建随包分发的便携频率先验索引（population 级桶）。

背景：frequency_prior 的完整源语料 data/uoft/normalized 为 GB 级（不随包）；
磁盘缓存 JSON 26MB / pkl 16MB 且 fingerprint 绑定源 chunk mtime（便携布局
下必然 miss）。本脚本在**构建期**把完整聚合索引中的 population 级桶抽出，
落盘 data/cache/frequency_prior_portable.json（KB 级）：

  {"schema": 2, "level": "NL10:cash:6max",
   "fingerprint": "portable-<UTC时间戳>",
   "index": {"NL10:cash:6max": {"[\"population\",\"BU\",\"flop\",...]": {...}}}}

产品对打的 villain 全部为池 bot（无语料个体 id），population 级先验是
其唯一可达层——个体层不随包 = 诚实的能力边界（provenance.scope 声明）。

用法：.venv/Scripts/python.exe scripts/build_portable_freq_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poker_decision_ai.agents.frequency_prior.agent import (  # noqa: E402
    INDEX_SCHEMA_VERSION, _load_index)


class _Ctx:
    class config:
        @staticmethod
        def frequency_prior_config():
            return {"source_dir": "data/uoft/normalized", "max_chunks": 5,
                    "cache_path": "data/cache/frequency_prior_cache.json",
                    "level": "NL10:cash:6max"}


def main() -> int:
    cfg = {"source_dir": "data/uoft/normalized", "max_chunks": 5,
           "cache_path": "data/cache/frequency_prior_cache.json",
           "level": "NL10:cash:6max"}
    warnings: list = []
    index, prov = _load_index(cfg, _Ctx(), warnings)
    level = "NL10:cash:6max"
    inner = index.get(level) or {}
    if not inner:
        print("FATAL: full index empty（源/缓存缺失？）", file=sys.stderr)
        return 1
    pop = {json.dumps(list(k), ensure_ascii=False): v
           for k, v in inner.items() if k[0] == "population"}
    n_player = len(inner) - len(pop)
    blob = {"schema": INDEX_SCHEMA_VERSION, "level": level,
            "fingerprint": f"portable-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
            "built_at": time.time(),
            "source_provenance": {k: v for k, v in prov.items()
                                  if k in ("chunks", "level",
                                           "index_schema")},
            "index": {level: pop}}
    out = ROOT / "data" / "cache" / "frequency_prior_portable.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(f"portable index written: {out}")
    print(f"  population buckets: {len(pop)}  (player buckets skipped: "
          f"{n_player})  size: {out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
