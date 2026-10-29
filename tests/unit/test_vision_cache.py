# -*- coding: utf-8 -*-
"""P30-9-IM 回归：**观测缓存**（按帧哈希 ✓）—— 命中即秒回，且**行为完全一致** ✓。

设计边界（关键 ✓）：
  * 只缓存**视觉观测**（纯由像素决定 ✓）⇒ 同一份 RGBA 复用**正确** ✓；
  * **决策永远重算** ✓（决策还依赖求解器缓存/时点 ✓，缓存它就会改变行为 ✗）。
实测（6h6c 夹具）：4.72 秒 → **0.62 秒** ✓，决策同为 `OK/raise` ✓。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIX = ROOT / "tests" / "fixtures" / "real_hero_cards_6h6c_1876x1400.png"


@pytest.mark.skipif(not FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓（不伪造 ✗）")
def test_second_click_on_same_frame_hits_cache(monkeypatch):
    monkeypatch.setenv("VISION_CACHE", "1")
    from cash_game_ui.runner import CashGameRunner
    r = CashGameRunner()
    r._get_runtime()
    im = Image.open(FIX).convert("RGBA")
    rgba = im.tobytes()
    r._vision_cache.clear()
    r.analyze(rgba, im.width, im.height, allow_recapture=False)
    assert r._vision_cache_hit is False, "首次必须未命中 ✓"
    t0 = time.perf_counter()
    r.analyze(rgba, im.width, im.height, allow_recapture=False)
    dt = time.perf_counter() - t0
    assert r._vision_cache_hit is True, "同帧再次点击必须命中 ✓"
    assert dt < 2.0, f"命中后应显著更快（实测 ~0.6 秒 ✓，本次 {dt:.2f} 秒 ✗）"


@pytest.mark.skipif(not FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_cache_disabled_switch(monkeypatch):
    monkeypatch.setenv("VISION_CACHE", "0")
    from cash_game_ui.runner import CashGameRunner
    r = CashGameRunner()
    r._get_runtime()
    im = Image.open(FIX).convert("RGBA")
    r.analyze(im.tobytes(), im.width, im.height, allow_recapture=False)
    r.analyze(im.tobytes(), im.width, im.height, allow_recapture=False)
    assert r._vision_cache_hit is False, "关闭开关后不得命中 ✓（可回退 ✓）"


@pytest.mark.skipif(not FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_cache_preserves_observation(monkeypatch):
    """**观测必须逐字段一致** ✓（缓存只影响速度，不影响内容 ✓）。"""
    monkeypatch.setenv("VISION_CACHE", "1")
    import poker_decision_ai.action_semantics as sem
    from cash_game_ui.runner import CashGameRunner

    seen: list = []
    orig = sem.align_final_decision

    def _spy(*a, **kw):
        obs = kw.get("obs")
        if isinstance(obs, dict):
            seen.append(json.dumps({
                "seats": [(s.get("seat"), s.get("stack"), s.get("in_hand"))
                           for s in (obs.get("seats") or [])],
                "pot": obs.get("pot"), "board": list(obs.get("board") or []),
                "street": obs.get("street"), "to_call": obs.get("to_call"),
                "hole": list(obs.get("hole_cards") or []),
            }, sort_keys=True, default=str))
        return orig(*a, **kw)

    sem.align_final_decision = _spy
    try:
        r = CashGameRunner()
        r._get_runtime()
        im = Image.open(FIX).convert("RGBA")
        rgba = im.tobytes()
        r._vision_cache.clear()
        r.analyze(rgba, im.width, im.height, allow_recapture=False)
        first = seen[-1] if seen else None
        r.analyze(rgba, im.width, im.height, allow_recapture=False)
        second = seen[-1] if seen else None
        assert first is not None and first == second, "命中前后的观测必须一致 ✓"
    finally:
        sem.align_final_decision = orig
