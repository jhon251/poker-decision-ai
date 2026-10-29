#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-EO：HU spot 库**覆盖度量**（街 × 纹理 × 唯一牌面 × 家族）。

用于量化覆盖率补强的效果（例如 flop 覆盖从 40 → 52）。
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from poker_decision_ai.core.providers.spot_table import _texture_class   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def survey(dirname: str) -> dict:
    d = ROOT / "data" / "solver" / dirname
    man = d / "manifest.json"
    ids = []
    if man.is_file():
        ids = json.loads(man.read_text(encoding="utf-8")).get("solved_ids") or []
    files = [d / f"{i}.json" for i in ids] if ids else sorted(d.glob("*.json"))
    street = collections.Counter()
    tex = collections.Counter()
    fam = collections.Counter()
    boards: set[tuple] = set()
    per_street_boards: dict[str, set] = collections.defaultdict(set)
    for f in files:
        if not f.is_file():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        q = r.get("query") or {}
        # 兼容两种形状：r1/fb/r3 用 query.{street,board}；6-max 记录在顶层
        s = str(q.get("street") or r.get("street") or "?")
        b = tuple(q.get("board") or r.get("board") or [])
        street[s] += 1
        tex[str(_texture_class(list(b)))] += 1
        fam[str(r.get("spot_id") or "").split("-")[0]] += 1
        boards.add(b)
        per_street_boards[s].add(b)
    return {"dir": dirname, "n_solutions": sum(street.values()), "unique_boards": len(boards),
             "by_street": dict(street),
             "unique_boards_by_street": {k: len(v) for k, v in per_street_boards.items()},
             "by_texture": dict(tex.most_common(12)), "by_family": dict(fam.most_common(6))}


def main() -> None:
    out = {}
    for dirname in ("spots_r1", "spots_fb", "spots_r3", "spots_6max"):
        d = ROOT / "data" / "solver" / dirname
        if not d.exists():
            continue
        info = survey(dirname)
        out[dirname] = info
        print(f"\n{dirname}: {info['n_solutions']} 个解 | 唯一牌面 {info['unique_boards']}")
        print(f"  街: {info['by_street']}")
        print(f"  各街唯一牌面: {info['unique_boards_by_street']}")
        print(f"  家族: {info['by_family']}")
        print(f"  纹理: {info['by_texture']}")
    (ROOT / "docs" / "spot_library_coverage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 docs/spot_library_coverage.json")


if __name__ == "__main__":
    main()
