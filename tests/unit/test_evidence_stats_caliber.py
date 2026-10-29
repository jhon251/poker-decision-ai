# -*- coding: utf-8 -*-
"""P30-9-HQ 测试：证据统计的**口径正确性**（恢复率只在补读事件上有意义 ✓）。

修正的两个报告缺陷 ✗：
  1. 无补读事件时曾打出"恢复率 0%"（分母为空却给数字 ✗）⇒ 现为 None（**不可读** ✓）；
  2. `to_call_unknown` 曾混进同一句 ⇒ 现**单独统计** ✓；
  3. 只有 `runtime`（真实对局）事件才用于"真实恢复率" ✓（帧回放/门禁事件分开 ✓）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from board_fallback_stats import summarize  # noqa: E402


def test_no_fallback_events_rate_unreadable():
    rows = [{"kind": "to_call_unknown", "source": "runtime"},
             {"kind": "to_call_unknown", "source": "frames_replay"}]
    s = summarize(rows)
    assert s["recovery_rate_all_sources"] is None, "无补读事件 ⇒ 恢复率必须为 None（不可读）✓"
    assert s["to_call_unknown_total"] == 2
    assert s["to_call_unknown_real"] == 1, "真实 to_call 事件须单独计数 ✓"


def test_recovery_rate_only_over_fallback_events():
    rows = [{"kind": "fallback_ok", "source": "runtime"},
             {"kind": "fallback_failed", "source": "runtime"},
             {"kind": "to_call_unknown", "source": "runtime"}]
    s = summarize(rows)
    assert s["recovery_rate_real"] == 0.5, "恢复率分母只含补读事件 ✓"
    assert s["real_fallback_events"] == 2
    assert s["to_call_unknown_real"] == 1


def test_non_runtime_events_excluded_from_real_rate():
    rows = [{"kind": "fallback_ok", "source": "frames_replay"},
             {"kind": "fallback_failed", "source": "frames_replay"},
             {"kind": "fallback_ok", "source": "runtime"}]
    s = summarize(rows)
    assert s["recovery_rate_real"] == 1.0, "帧回放事件不得进入真实恢复率 ✓"
    # 全来源口径**同样只在补读事件上**计算（2 成功 / 1 失败 ⇒ 2/3 ✓）
    assert abs(s["recovery_rate_all_sources"] - 2 / 3) < 1e-9
    assert s["by_source"]["frames_replay"] == 2
