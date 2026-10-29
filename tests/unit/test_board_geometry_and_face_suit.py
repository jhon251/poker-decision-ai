# -*- coding: utf-8 -*-
"""P30-9-IF 回归：**牌面槽位几何自动校正** + **人头牌花色（角标+颜色）** ✓。

用户实测两处缺口 ✗：
  ① 牌面 **4♣3♥K♥** 三张完整可见，流程只报 2 张 ✗ —— 根因：牌面识别器默认
     `window="1419"`（槽位 X0=**664** ✗），而 1876 渲染真实牌位是 **590/732/875** ✓
     ⇒ 三槽整体错位 ~74px ✗ ⇒ 全部 `rank_gate` ✗。
     修法：`recognize()` **按帧宽自动选几何** ✓ + 用**白卡斑块**实测 x0 校正 ✓。
  ② **K♥** 的角标花色 0.991 ✓ 但**中心**只有 0.272 ✗ —— 人头牌（J/Q/K）卡面中心是
     **王冠/人头图案**而非花色 ✓ ⇒ 中心路径对所有人头牌系统性失效 ✗。
     修法：允许 **角标 ≥0.95 + 红黑颜色一致** 双证据 ⇒ FULL ✓（**不降门槛** ✗）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BOARD_FIX = ROOT / "tests" / "fixtures" / "real_board_4c3hKh_1876x1400.png"
HERO_FIX = ROOT / "tests" / "fixtures" / "real_hero_cards_6h6c_1876x1400.png"


def _recognize(path: pathlib.Path) -> dict:
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
        CoinPokerBoardRecognizer)
    im = Image.open(path).convert("RGBA")
    return CoinPokerBoardRecognizer().recognize(im.tobytes(), im.width, im.height, "test")


@pytest.mark.skipif(not BOARD_FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓（不伪造 ✗）")
def test_geometry_auto_selected_and_corrected():
    """① 自动按帧宽选 1876 几何 ✓，且实测 x0 与配置一致（590 ✓）。"""
    out = _recognize(BOARD_FIX)
    geo = out.get("geometry") or {}
    assert geo.get("window") == "1876", f"应按帧宽自动选 1876 几何 ✓（实得 {geo}）"
    assert geo.get("slot_x0") == 590, f"真实牌位 x0=590 ✓（实得 {geo}）"
    assert geo.get("detected_x0") == 590, "白卡斑块实测应与配置一致 ✓"


@pytest.mark.skipif(not BOARD_FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_full_flop_including_face_card_suit():
    """①+② 三张都要读出（含 **人头牌 K♥** ✓）——画面真值 4♣ 3♥ K♥ ✓。"""
    out = _recognize(BOARD_FIX)
    assert out.get("value") == ["4c", "3h", "Kh"], \
        f"应读出完整三张 4c/3h/Kh ✓（实得 {out.get('value')}）"


@pytest.mark.skipif(not BOARD_FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_face_card_suit_evidence_is_auditable():
    """② 人脸牌花色必须**写明依据**（角标+颜色 ✓），可审计 ✓。"""
    out = _recognize(BOARD_FIX)
    ok = [c for c in (out.get("cards") or []) if c.get("status") == "OK"]
    assert len(ok) >= 3
    face = [c for c in ok if str(c.get("rank") or "").upper() in ("J", "Q", "K", "A")]
    assert face, "应包含人头/尖牌 ✓"
    joined = str([c.get("evidence") for c in face])
    assert "corner" in joined, "必须保留角标证据 ✓"


@pytest.mark.skipif(not HERO_FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓")
def test_other_real_frame_unchanged():
    """**不回归** ✓：另一张真实帧的牌面仍应读出 6♦7♥9♠ ✓。"""
    out = _recognize(HERO_FIX)
    assert out.get("value") == ["6d", "7h", "9s"], f"实得 {out.get('value')}"
