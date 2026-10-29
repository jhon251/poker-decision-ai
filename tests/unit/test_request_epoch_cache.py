# -*- coding: utf-8 -*-
"""P30-9-HP 测试：**请求纪元门控** ⇒ 同一请求内缓存写入不可见（同输入同输出 ✓）。

背景（第 28 轮实测）：真实帧均衡知情率在同一批输入上波动（5/7、5/7、4/7 ✗）——
原因是"遇到即解"的后台缓存写入会**在同一请求内**被表重载 ⇒ 该请求的路径被改变 ✗。
修复：`begin_request()` 推进纪元；表**只在新纪元**接受缓存新增 ✓。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers import spot_table_6max as st  # noqa: E402


def _rec(board, pot=40.0, positions=("SB", "BTN")):
    return {"spot_id": f"3bp-livecache-flop-{positions[0]}{positions[1]}-{''.join(board)}",
             "street": "flop", "board": list(board), "pot_bb": pot,
             "effective_stack_bb": 100.0,
             "positions": {"oop": positions[0], "ip": positions[1]},
             "actor_position": positions[0], "node": "first_in",
             "to_call_bb": None, "bet_ratio": None,
             "actions": [{"name": "check", "size_bb": None, "frequency": 0.5},
                          {"name": "bet", "size_bb": 20.0, "frequency": 0.5}],
             "private_cards_actor": [], "freq_per_hand": {}, "solved": True,
             "source": "live_cache"}


def _q(board):
    return {"street": "flop", "board": list(board), "pot_bb": 40.0,
             "node": "first_in", "positions": {"oop": "SB", "ip": "BTN"},
             "actor": "SB", "family": "3bp", "stack_bb": 100.0}


def test_epoch_helper_monotonic():
    a = st.begin_request()
    b = st.begin_request()
    assert b == a + 1 == st.current_epoch()


def test_cache_write_visible_only_next_request(tmp_path, monkeypatch):
    """**同一表实例**（产品用单例 `get_spot_table_6max` ✓）在同一请求内不得看到缓存新增 ✓。"""
    monkeypatch.setenv("SPOT6_LIVE_CACHE_DIR", str(tmp_path))
    board = ["2c", "2h", "9d"]                     # 该牌面库内不存在（安全用作新缓存 ✓）
    t = st.SpotTable6max(str(ROOT / "data" / "solver" / "spots_6max"))
    st.begin_request()                             # 新纪元：从"空缓存"开始 ✓
    first = t.lookup(_q(board))
    assert not (first and first.get("is_approximate") is False
                 and str(((first.get("raw") or {}).get("spot_id") or "")).startswith("3bp-livecache"))
    # 同一请求内写入缓存记录
    (tmp_path / f"{_rec(board)['spot_id']}.json").write_text(
        json.dumps(_rec(board), ensure_ascii=False), encoding="utf-8")
    t._loaded = False                              # 触发重载判定（真实路径同样会走 ✓）
    same = t.lookup(_q(board))
    assert not (same and same.get("is_approximate") is False
                 and str(((same.get("raw") or {}).get("spot_id") or "")).startswith("3bp-livecache")), \
        "同一请求内缓存写入必须**不可见** ✓（否则同输入不同输出 ✗）"
    # 推进纪元（下一次请求）⇒ 可见 ✓
    st.begin_request()
    t._loaded = False
    seen = t.lookup(_q(board))
    assert seen is not None, "下一次请求应能看到新缓存记录 ✓"
    sid = str(((seen.get("raw") or {}).get("spot_id") or ""))
    assert sid.startswith("3bp-livecache"), f"应命中缓存记录，实得 {sid}"
