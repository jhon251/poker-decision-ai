# -*- coding: utf-8 -*-
"""V-5.1 Round17 — 微级别桌金额识别完全修复单测。

覆盖：
- stack_gold_1876 暗域亮度阈上限 STACK_GOLD_LUMA_CEIL=36（用户实况
  bg≈45 + 暗金字 l≈41-99 形态：V-5 式 58.5 拒 → V-5.1 捕获）；
- 上限截断不触碰 bg≤27.7 域（逐点不变）与 bg>60 域（冻结）；
- 语料锚定：暗金字帧（bg≈10.1、text p50≈40.6）hero 座读出 + hero
  visible 帧端到端（stack/pot 数值）；
- 机制探针固化：阈值 58.5（bg=45 公式值）→ no_gold_components；
  36 → 全组件 RECOGNIZED；
- 微库 digit 覆盖完整性（stack/pot 0-9 全）；
- 负样本：绿毡 ROI（bg 30-34 域，改动域）gold=0（0 误报）。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    STACK_GOLD_DIM_BG_MAX, STACK_GOLD_G_DELTA_STRICT, STACK_GOLD_LUMA_CEIL,
    STACK_GOLD_LUMA_FLOOR, STACK_GOLD_LUMA_RATIO_DIM,
    STACK_GOLD_R_DELTA_STRICT, STACK_GOLD_RG_RATIO_STRICT,
    SEAT_STACK_ROIS_1876, StackRecognizer1876, stack_comps_8conn,
    stack_gold_1876)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)

ROOT = Path('data/vision2/coinpoker_v3')
MICRO_DIR = Path('data/vision2/coinpoker_assets_v3c_micro')

CORPUS_ANCHORS = {
    'dark_gold_5glyph': 'POT_CHANGED_192_1791377074962_full.png',
    'dark_gold_alt': 'POT_CHANGED_234_1791377245929_full.png',
    'dark_gold_2glyph': 'SCENE_CHANGED_15_1791376391894_full.png',
    'hero_visible': 'POT_CHANGED_109_1791376749721_full.png',
}


def _frame(name):
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png((ROOT / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _mask_at_thr(roi, thr):
    """指定亮度阈的 B 档 gold 掩码（V-5 语义探针用）。"""
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return ((r > b + 15) & (g >= b + 8) & (r >= g * 0.9)
            & (l > thr)).astype(np.uint8)


def _strict_mid_band_mask(roi, ceil, thr):
    """V-5.1 上限带语义：严格暖门 & (ceil, thr]（用户形态模拟用）。"""
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    strict = ((r > b + STACK_GOLD_R_DELTA_STRICT)
              & (g >= b + STACK_GOLD_G_DELTA_STRICT)
              & (r >= g * STACK_GOLD_RG_RATIO_STRICT))
    return (strict & (l > ceil) & (l <= thr)).astype(np.uint8)


# ── 判据：暗域上限（本轮核心修复）─────────────────────────────────
def test_gold_dark_domain_ceiling_constant():
    """上限常量在位且语义带：28 ≤ 36 ≤ 60 域界。"""
    assert STACK_GOLD_LUMA_CEIL == 36
    assert STACK_GOLD_LUMA_FLOOR <= STACK_GOLD_LUMA_CEIL
    assert STACK_GOLD_LUMA_CEIL < STACK_GOLD_DIM_BG_MAX


def test_gold_user_live_form_bg45_dark_text_captured():
    """用户实况形态（暖 pill bg≈45 + 暗金字）：V-5 式（B 档 & l>58）拒、
    V-5.1 上限带严格暖门捕获；暖 pill 底（g-b<18）不得混入。"""
    roi = np.zeros((72, 202, 4), dtype=np.uint8)
    roi[..., :] = (56, 44, 28, 255)           # 暖 pill，l≈44.6（r-b=28 过
    # 严格 r 门但 g-b=16 < 18 拒——语料暖底同构）
    ys, xs = 16, 40
    for dx in range(0, 120, 24):              # 5 个 18px 宽字形
        roi[ys:ys + 34, xs + dx:xs + dx + 18] = (58, 46, 20, 255)
    mask, bg = stack_gold_1876(roi)
    assert 44 <= bg <= 46
    v5_thr = max(bg * STACK_GOLD_LUMA_RATIO_DIM, STACK_GOLD_LUMA_FLOOR)
    assert v5_thr == pytest.approx(59.5, abs=0.5)      # V-5 公式值
    old_mask = _mask_at_thr(roi, v5_thr)
    assert int(old_mask.sum()) == 0                     # V-5：0 捕获
    assert int(mask.sum()) == 5 * 34 * 18               # V-5.1：文本全捕获
    comps = stack_comps_8conn(mask)
    assert len(comps) == 5                              # 组件完整（无底连块）
    assert all(c[3] == 34 for c in comps)


def test_gold_ceiling_does_not_touch_low_bg_domain():
    """bg≤27.7 域：thr≤36 → 上限带为空，掩码与 V-5 式逐像素同。"""
    roi = np.zeros((40, 90, 4), dtype=np.uint8)
    roi[..., :] = (24, 19, 6, 255)            # 近黑 pill，l≈19
    roi[8:30, 10:80] = (70, 55, 18, 255)      # 金字，l≈55.6（>下限 28）
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    assert 0 < bg <= 27.7                     # 上限带为空域
    v5 = ((r > b + 15) & (g >= b + 8) & (r >= g * 0.9)
          & (l > max(bg * STACK_GOLD_LUMA_RATIO_DIM,
                     STACK_GOLD_LUMA_FLOOR))).astype(np.uint8)
    mask, _ = stack_gold_1876(roi)
    assert np.array_equal(mask, v5)


def test_gold_bright_domain_untouched_by_ceiling():
    """bg>60 亮域：冻结 1.6×bg，上限不参与。"""
    roi = np.zeros((30, 60, 4), dtype=np.uint8)
    roi[..., :] = (235, 235, 235, 255)        # 覆盖 UI 亮底
    roi[10:20, 10:40] = (200, 170, 60, 255)   # 亮金字（bg*1.6 之外拒绝）
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    assert bg > 60
    expected = ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
                & (l > bg * 1.6)).astype(np.uint8)
    mask, _ = stack_gold_1876(roi)
    assert np.array_equal(mask, expected)


# ── 语料锚定：暗金字形态 + hero visible 端到端 ────────────────────
@pytest.mark.parametrize('key,expect', [
    ('dark_gold_5glyph', '9.52'),
    ('dark_gold_alt', '9.54'),
    ('dark_gold_2glyph', '10'),
])
def test_corpus_dark_gold_hero_readout(key, expect):
    """语料暗金字帧（bg≈10.1、text p50≈40.6）hero 座数值读出。"""
    fr = _frame(CORPUS_ANCHORS[key])
    r = StackRecognizer1876()._recognize_seat(fr, 'S6')
    assert r['status'] == 'RECOGNIZED'
    assert r['raw_text'] == expect


def test_hero_visible_frame_stack_pot_values():
    """hero visible 帧：S6（hero pill）stack=9.64、pot=0.25（₮ 量级；
    bridge 层座号映射为 V-2K 冻结域，由其自身测试覆盖）。"""
    fr = _frame(CORPUS_ANCHORS['hero_visible'])
    s6 = StackRecognizer1876()._recognize_seat(fr, 'S6')
    assert s6['status'] == 'RECOGNIZED'
    assert s6['stack'] == 9.64 and s6['raw_text'] == '9.64'
    po = PotRecognizer(window='1876').recognize(fr)
    assert po['status'] == 'RECOGNIZED' and po['value'] == '0.25'


def test_mechanism_threshold_sweep_on_real_dark_gold_text():
    """机制探针固化：真实暗金字像素——V-5 公式在 bg=45 的阈 58.5 → 0 组件
    （用户实况失效态）；V-5.1 上限带（严格暖门 & (36, 58.5]）→ 组件完整。"""
    fr = _frame(CORPUS_ANCHORS['dark_gold_5glyph'])
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876['S6']
    roi = fr[y0:y1, x0:x1]
    m_v5 = _mask_at_thr(roi, 58.5)
    m_new = _strict_mid_band_mask(roi, STACK_GOLD_LUMA_CEIL, 58.5)
    assert int(m_v5.sum()) == 0                        # 用户实况失效态
    assert len(stack_comps_8conn(m_v5)) == 0
    comps = stack_comps_8conn(m_new)
    assert len([c for c in comps if c[3] >= 20]) == 3  # 3 数字
    assert any(c[3] <= 9 and c[2] <= 10 for c in comps)  # 小数点在位


# ── 微库 digit 覆盖完整性（stack/pot 0-9 全）─────────────────────
def test_micro_library_digit_completion():
    """v3c_micro：stack_glyph_micro 与 pot_digit_micro 的 label 覆盖 0-9。"""
    m = json.loads((MICRO_DIR / 'manifest.json').read_text(encoding='utf-8'))
    cov = {}
    for a in m['assets']:
        k = a.get('kind')
        if k in ('stack_glyph_micro', 'pot_digit_micro'):
            cov.setdefault(k, set()).add(a['label'])
    for k, labels in cov.items():
        assert set('0123456789') <= labels, (k, sorted('0123456789' - labels))
    assert cov.get('stack_glyph_micro') and cov.get('pot_digit_micro')


# ── 负样本：改动域（bg 30-34 绿毡）0 误报 ─────────────────────────
def test_negative_felt_domain_stays_zero():
    """语料绿毡 ROI（S3 空座，bg≈34——上限生效域）：gold=0（RGB 门拒绿）。"""
    fr = _frame('SCENE_CHANGED_80_1791198452867_full.png')
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876['S3']
    mask, bg = stack_gold_1876(fr[y0:y1, x0:x1])
    assert 30 <= bg <= 35                     # 处于上限生效域
    assert int(mask.sum()) == 0


def test_empty_roi_zero_mask_unchanged():
    """空/截断 ROI 语义不变（UNKNOWN-first）。"""
    z, bg = stack_gold_1876(np.zeros((0, 0, 4), dtype=np.uint8))
    assert z.shape == (0, 0) and bg == 0.0
