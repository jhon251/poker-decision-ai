# -*- coding: utf-8 -*-
"""P30-9-FK 测试：6-max 表必须**优先牌面精确匹配**（同纹理桶不得冒名 exact）。

缺陷背景：以记录自身参数查表时，`3bp-BTN-SB-KsJh4h` 返回 `3bp-BTN-SB-7s5h4h`
（同纹理桶、不同牌面）且 `is_approximate=False` ✗ ⇒ 引擎套用了**别的牌面**的策略。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import get_spot_table_6max  # noqa: E402

LIB = ROOT / "data" / "solver" / "spots_6max"


def _query(rec: dict) -> dict:
    """从**表内行**构造引擎查询（行字段：actor/oop/ip，非 actor_position/positions）。"""
    return {"actor": rec["actor"], "street": rec["street"],
             "board": list(rec["board"]), "pot_bb": rec["pot_bb"], "node": rec["node"],
             "positions": {"oop": rec["oop"], "ip": rec["ip"]},
             "bet_ratio": rec.get("bet_ratio"),
             "to_call_bb": rec.get("to_call_bb"), "stack_bb": 97.5}


def _same_texture_boards() -> list[str]:
    """找出同族、同纹理桶但牌面不同的两个记录。"""
    st = get_spot_table_6max()
    seen: dict[tuple, str] = {}
    for r in st._rows:
        key = (r["node"], r["actor"], r["street"], r["pot_bb"], r["texture"])
        board = tuple(r["board"])
        if key in seen and seen[key] != board:
            return [seen[key], board]  # type: ignore[return-value]
        seen.setdefault(key, board)     # type: ignore[arg-type]
    return []


def test_board_exact_preferred_over_texture_bucket():
    st = get_spot_table_6max()
    assert st.count() > 200
    boards = _same_texture_boards()
    assert boards, "需要至少两个同纹理桶不同牌面的节点用于本测试"
    for board in boards:
        rec = next(r for r in st._rows if tuple(r["board"]) == tuple(board))
        hit = st.lookup(_query(rec))
        assert hit is not None
        assert hit["raw"]["spot_id"] == rec["spot_id"], (
            f"牌面 {board} 应命中自身节点，实际 {hit['raw']['spot_id']}")


def test_bucket_representative_is_labeled_approximate():
    """无同牌面候选时，桶代表必须如实标注 board_approximate/近似。"""
    st = get_spot_table_6max()
    rec = next(r for r in st._rows if r["node"] == "facing_bet")
    q = _query(rec)
    q["board"] = ["2c", "2d", "2h"]          # 库内不存在的牌面
    hit = st.lookup(q)
    if hit is not None:
        assert hit["is_approximate"] is True
        assert hit["board_approximate"] is True
