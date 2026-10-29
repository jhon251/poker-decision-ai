# -*- coding: utf-8 -*-
"""P30-9-GO 回归测试：**真实会话观测到的牌面必须精确命中**（防回退）。

背景（第 6 轮）：真实会话为 **SB vs BTN / pot ~40bb**，而此前批次硬编码 BB-BTN ✗
⇒ 位置键不匹配 ⇒ 真实翻后精确命中恒为 0 ✗（"真实覆盖 ≈0%"的真因）。
修复后真实翻后真解率 0% → 44%（两人池 58%）。本测试把这两个"真实观测牌面"钉住：
若将来有人改坏位置键/底池容差/牌面规范化，这里会立刻失败 ✓。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import (  # noqa: E402
    POT_EXACT_TOL_REL, SpotTable6max)

LIB = ROOT / "data" / "solver" / "spots_6max"
#: 真实会话中出现过的 2 人池完整牌面（第 7 轮提取：Qh7d2c ×19、2h2dTh ×1）
REAL_BOARDS = [["Qh", "7d", "2c"], ["2h", "2d", "Th"]]
#: 真实查询键（来自日志：SB vs BTN / pot 40.4 / 栈 ~97.5）
REAL_Q = {"street": "flop", "pot_bb": 40.4, "node": "first_in",
           "positions": {"oop": "SB", "ip": "BTN"}, "actor": "SB",
           "family": "3bp", "stack_bb": 97.5}


def test_real_observed_flops_hit_exactly():
    table = SpotTable6max(str(LIB))
    table._ensure_loaded()
    for board in REAL_BOARDS:
        h = table.lookup({**REAL_Q, "board": list(board)})
        assert h is not None, f"{board} 未命中（真实会话牌面必须可命中）"
        assert h["is_approximate"] is False, f"{board} 落入近似（位置键/容差可能被改坏）"
        sid = (h.get("raw") or {}).get("spot_id") or ""
        assert sid.startswith("3bp-"), f"{board} 命中记录异常: {sid}"
        assert "livecache" not in sid, f"{board} 命中的是现场缓存（应为入库真解）: {sid}"


def test_sbbtn_pot40_records_present():
    """真实主线（SB-BTN / pot 40）必须有自解记录（否则真解率会掉回 0%）。"""
    recs = list(LIB.glob("3bp-oursb-*.json"))
    assert len(recs) >= 2, f"缺少 SB-BTN 自解记录：{len(recs)}"
    for p in recs:
        r = json.loads(p.read_text(encoding="utf-8"))
        assert r["positions"] == {"oop": "SB", "ip": "BTN"}, p.name
        assert abs(float(r["pot_bb"]) - 40.4) / 40.4 <= POT_EXACT_TOL_REL, p.name


def test_pot_tolerance_constant_is_defined():
    assert 0 < POT_EXACT_TOL_REL <= 0.5
