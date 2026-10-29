# -*- coding: utf-8 -*-
"""P30-9-AK：多人保守模式开关（默认关 = 零影响；打开 = 用多人联合胜率）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402
import pytest                                                             # noqa: E402

from poker_decision_ai import multiway_equity as mw                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402

CONTROL = None  # 占位，避免 pytest 误收集为用例


def test_default_off_leaves_equity_untouched():
    """默认关：挂钩必须原样返回（不得改变任何数字）。"""
    mw.clear_conservative()
    assert mw.conservative_active() is False
    assert mw.MULTIWAY_OVERRIDE["value"] is None


def test_apply_override_is_noop_when_off():
    """关闭状态：apply_override 原样返回（不改任何数字）。"""

    class _Out:
        hero_equity = 0.4386
        outcome = {"win": 0.4, "tie": 0.05}

    mw.clear_conservative()
    out = mw.apply_override(_Out())
    assert out.hero_equity == 0.4386
    assert out.outcome == {"win": 0.4, "tie": 0.05}


def test_apply_override_replaces_when_on():
    """开启状态：hero_equity/outcome 被替换为多人估计，且可关闭恢复。"""

    class _Out:
        hero_equity = 0.4386
        outcome = {"win": 0.4, "tie": 0.05}

    mw.set_conservative({"ok": True, "equity": 0.1667, "win": 0.15, "tie": 0.033,
                          "samples": 1500, "n_villains": 5})
    out = mw.apply_override(_Out())
    assert out.hero_equity == 0.1667
    assert out.outcome["method"] == "multiway_joint_mc"
    mw.clear_conservative()
    assert mw.apply_override(_Out()).hero_equity == 0.4386


def test_hook_install_is_idempotent():
    assert mw.install_equity_hook() is True
    assert mw.install_equity_hook() is True
    assert mw._HOOK_INSTALLED is True


def test_on_replaces_equity_with_multiway_estimate():
    """打开：覆盖值 = 多人联合估计（Q8o/5 人 ≈0.17，而非 0.44）。"""
    est = mw.conservative_for(
        {"hole_cards": ["Qs", "8d"], "board": [],
         "seats": [{"seat": i, "in_hand": True} for i in range(1, 7)]},
        samples=1200)
    assert est["ok"] and 0.10 <= est["equity"] <= 0.25
    mw.set_conservative(est)
    assert mw.conservative_active() is True
    assert mw.MULTIWAY_OVERRIDE["value"] == est["equity"]
    mw.clear_conservative()
    assert mw.conservative_active() is False


def test_runner_clears_override_on_entry():
    """入口清空：上一次开启不会泄漏到下一次调用。"""
    run = CashGameRunner()
    mw.set_conservative({"ok": True, "equity": 0.17, "win": 0.15, "tie": 0.04,
                          "samples": 100, "n_villains": 5})
    assert mw.conservative_active() is True
    tiny = np.zeros((40, 40, 4), dtype=np.uint8)
    tiny[..., 3] = 255
    run.analyze(tiny.tobytes(), 40, 40, allow_recapture=False)
    assert mw.conservative_active() is False        # 入口已清空且未开启模式
