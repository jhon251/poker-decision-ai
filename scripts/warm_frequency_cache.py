"""SOLVER-1 配套：预热 frequency_prior 磁盘缓存（避免首次 decide 同步重建）。

用法：python scripts/warm_frequency_cache.py [--max-chunks 5]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.config import NestedDict      # noqa: E402
from poker_decision_ai.agents.frequency_prior import agent as fpa  # noqa: E402


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    fp = cfg.get("frequency_prior") or {}
    source_dir = str(fp.get("source_dir") or "data/uoft/normalized")
    max_chunks = int(fp.get("max_chunks") or 5)
    cache_path = str(fp.get("cache_path") or "data/cache/frequency_prior_cache.json")
    level = str(fp.get("level") or fpa.DEFAULT_CORPUS_LEVEL)

    chunks = sorted(pathlib.Path(source_dir).glob("chunk_*.jsonl"))[:max_chunks]
    fingerprint = (f"schema={fpa.INDEX_SCHEMA_VERSION}|level={level}|"
                   f"{source_dir}|{max_chunks}|"
                   + ",".join(f"{c.name}:{int(c.stat().st_mtime)}" for c in chunks))
    cache_file = pathlib.Path(cache_path)
    pkl = cache_file.with_suffix(".pkl")
    if pkl.exists():
        try:
            import pickle
            blob = pickle.loads(pkl.read_bytes())
            if blob.get("fingerprint") == fingerprint:
                print("cache already warm:", pkl)
                return 0
        except Exception:
            pass
    t0 = time.time()
    index = fpa.build_index_from_chunks(chunks, level=level)
    n_buckets = sum(len(v) for v in index.values())
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    tmp = pkl.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({
        "schema": fpa.INDEX_SCHEMA_VERSION,
        "fingerprint": fingerprint, "index": index,
        "built_at": time.time()}, protocol=5))
    tmp.replace(pkl)
    print(f"cache built: level={level} {n_buckets} buckets "
          f"in {time.time()-t0:.1f}s -> {pkl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
