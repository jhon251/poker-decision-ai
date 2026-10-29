# -*- coding: utf-8 -*-
"""P30-9-FZ 测试：牌面**顺序无关**（修 exact 漏配 ~5/6）。

背景：真实客户端报的公共牌顺序任意；索引/比较若用原始顺序，
3 张牌只有 1/6 排列能精确命中（实测均匀覆盖 0.055% vs 理论 0.217%）。
"""
from __future__ import annotations

import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import SpotTable6max  # noqa: E402

LIB = ROOT / "data" / "solver" / "spots_6max"


def _our_records() -> list[dict]:
    import json
    out = []
    for p in sorted(LIB.glob("3bp-ourhu-*.json"))[:5]:
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def test_sixmax_board_order_insensitive():
    """同一牌面的任意排列都必须精确命中同一条记录。"""
    recs = _our_records()
    assert recs, "需要 3bp-ourhu-* 记录（自解子局）"
    table = SpotTable6max(str(LIB))
    table._ensure_loaded()
    for rec in recs:
        want = rec["spot_id"]
        hits = set()
        for perm in itertools.permutations(rec["board"]):
            h = table.lookup({"street": rec["street"], "board": list(perm),
                               "pot_bb": rec["pot_bb"], "node": rec["node"],
                               "positions": rec["positions"],
                               "actor": rec["actor_position"], "family": "3bp",
                               "stack_bb": rec["effective_stack_bb"]})
            assert h is not None, f"{perm} 未命中"
            assert h["is_approximate"] is False, f"{perm} 未精确命中"
            hits.add((h.get("raw") or {}).get("spot_id"))
        assert hits == {want}, f"不同排列命中不同记录：{hits}"


def test_canon_helper_exists():
    import poker_decision_ai.core.providers.spot_table as st
    assert hasattr(st, "_canon_board")
    assert st._canon_board(["Td", "9d", "6h"]) == st._canon_board(["6h", "Td", "9d"])
