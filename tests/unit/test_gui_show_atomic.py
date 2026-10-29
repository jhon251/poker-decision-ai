# -*- coding: utf-8 -*-
"""P30-9-AU：GUI 显示层回归（大字/详情同源；显示异常不得终止刷新循环）。

背景（用户截图实证）：`_show` 内曾引用裸名 `runner` ⇒ NameError **发生在设置大字之后**
⇒ 大字停留在上一次的「识别不到」，而详情已是新结果；且异常会杀死 `_drain`
⇒ 界面从此不再更新（用户感受为「卡死」）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest                                                            # noqa: E402

tk = pytest.importorskip("tkinter")


def _gui():
    import importlib
    m = importlib.import_module("cash_game_ui.preview_gui")
    root = tk.Tk()
    root.withdraw()
    return m, m.PreviewGUI(root), root


OK_RESULT = {
    "status": "OK", "decision_action": "RAISE", "recommended_size": None,
    "confidence": 0.668, "strategy_mode": "BASELINE",
    "rule_size": {"action": "raise", "bb": 2.0, "chips": 0.04,
                   "bb_chips": 0.02, "source": "rule_floor"},
    "reasoning": "x",
    "gui_stages": {"capture_ms": 42, "pipeline_ms": 2700, "result_ms": 0},
    "vision": {"hero_cards": ["Qc", "6h"], "pot": 0.52, "street": "river",
                "blinds": {"sb": 0.01, "bb": 0.02}, "position": "MP",
                "hero_cards_ok": True},
    "blocking_fields": [], "unknown_fields": [],
}
BLOCKED = {
    "status": "DECISION_UNAVAILABLE", "decision_action": None,
    "reason": "blocking_fields:hero_cards", "blocking_fields": ["hero_cards"],
    "unknown_fields": ["to_call"], "gui_stages": {},
    "vision": {"hero_cards": None,
                "hero_cards_reason": "hero_cards_region_no_match",
                "hero_cards_ok": False},
    "frame_provenance": {},
}


def test_show_has_no_bare_runner_name():
    """`_show` 内不得出现裸名 runner（曾导致 NameError 打断显示）。"""
    src = (ROOT / "cash_game_ui/preview_gui.py").read_text(encoding="utf-8")
    i = src.index("def _show(")
    j = src.index("def _refresh_status(")
    body = src[i:j]
    assert "getattr(runner," not in body
    assert "{runner._last_frame_path}" not in body
    assert 'getattr(self.runner, "_last_frame_path"' in body


def test_label_matches_status_and_shows_rule_size():
    m, app, root = _gui()
    try:
        app._show(1, BLOCKED, "13:00:00")
        assert app.action_lbl.cget("text") == "识别不到"
        app._show(2, OK_RESULT, "13:00:05")
        txt = app.action_lbl.cget("text")
        assert txt.startswith("RAISE")
        assert "≥0.04" in txt                 # 规则下限尺寸必须可见
        assert "OK" in app.meta_lbl.cget("text")
    finally:
        root.destroy()


def test_display_exception_does_not_kill_drain():
    m, app, root = _gui()
    try:
        def boom(*a, **k):
            raise RuntimeError("显示异常")
        app._show = boom
        app._res_q.put((99, OK_RESULT, "13:00:10"))
        app._drain()                          # 不得抛出
        assert app._executed >= 1             # 计数照常推进
        assert app.action_lbl.cget("text") == "⚠ 显示异常"
    finally:
        root.destroy()
