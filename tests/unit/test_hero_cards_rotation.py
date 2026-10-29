# -*- coding: utf-8 -*-
"""P30-9-IB 回归：**hero 底牌倾斜**导致的识别失败（真实失败帧已固化为夹具 ✓）。

背景（用户报告 + 真实帧取证）：
  用户截图显示"识别不到：未识别到你的底牌（hero_cards_region_no_match）"✗，
  帧内底牌其实是 **6♥ 6♣**（肉眼清晰 ✓），而**牌面 6♦7♥9♠ 却识别成功** ✓。
  诊断：`card2`(6♣) 分数 1.00 ✓；`card1`(6♥) 的 rank 在 0° 时
  `6`=0.812 但 `Q`=0.793 ⇒ **判别余量 0.019 < 0.08** ✗ ⇒ `glyph_gate_failed` ✗。
  根因：牌面**扇形/动画使角标倾斜** ✗，而匹配器只对**平移**不变（自带滑动窗口 ✓）✗。
  修法：**小角度扫描**（仅在原角度判失败时启用 ✓），取判别余量最大者 ✓。

本测试锁定该行为（夹具缺失则跳过 ✓，**不伪造帧** ✗）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "real_hero_cards_6h6c_1876x1400.png"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓（不伪造 ✗）")
def test_rotated_hero_cards_are_recognized():
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
        HeroCardsRecognizer1876)

    arr = np.asarray(Image.open(FIXTURE).convert("RGB"))
    assert arr.shape[:2] == (1400, 1876), "夹具尺寸必须是 1876x1400（绝对 ROI 依赖 ✓）"
    out = HeroCardsRecognizer1876().recognize(arr)
    assert out.get("status") == "OK", f"应识别成功，实得 {out}"
    got = {str(c.get("rank", "")).lower() + str(c.get("suit", "")).lower()
            for c in out.get("cards") or []}
    assert got == {"6h", "6c"}, f"画面是真 6♥ 6♣ ⇒ 必须读出这两张，实得 {got}"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_rotation_is_recorded_as_evidence():
    """修复必须**可审计** ✓：结果里要写明用了多少度的小角度扫描 ✓。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
        HeroCardsRecognizer1876)

    arr = np.asarray(Image.open(FIXTURE).convert("RGB"))
    out = HeroCardsRecognizer1876().recognize(arr)
    evs = [(c.get("evidence") or {}) for c in out.get("cards") or []]
    rots = [e.get("rotation_deg") for e in evs if e.get("rotation_deg") is not None]
    assert rots, f"至少要有一张记录 rotation_deg ✓（实得 {evs}）"
    assert any(-20.0 <= float(r) <= 20.0 for r in rots), "角度应在小范围内 ✓"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_thresholds_not_relaxed():
    """**不得降门槛** ✗：MIN_SCORE / MIN_MARGIN 必须保持原值 ✓。"""
    from poker_decision_ai.vision2.pixel.realclient import coinpoker_herocards1876 as hc
    assert hc.MIN_SCORE == 0.7 and hc.MIN_MARGIN == 0.08, "阈值被改动会导致误判 ✗"
    assert hc.HERO_ROTATION_ANGLES and max(abs(a) for a in hc.HERO_ROTATION_ANGLES) <= 20.0
