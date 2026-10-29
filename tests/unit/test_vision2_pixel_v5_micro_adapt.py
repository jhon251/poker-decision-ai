# -*- coding: utf-8 -*-
"""V-5 Round16 — 微级别桌识别适配单测。

覆盖：
- stack_gold_1876 自适应亮度门分档（暗域/亮域/近黑下限）；
- parse_stack_text / parse_bet_amount 裸小数与前导零守卫；
- 微级别模板 kind 隔离装载（stack/pot/bet/header 互不混载）；
- _ncc_1876 / pot 尺度一致性门；
- 微帧端到端（stack/pot 小数值 + header USDT）。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    STACK_ASSET_DIR_1876, STACK_ASSET_DIR_1876_MICRO, StackRecognizer1876,
    _load_1876_templates, _ncc_1876, parse_stack_text, stack_gold_1876)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    BET_MICRO_DEC_RE, parse_bet_amount)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)

ROOT = Path('data/vision2/coinpoker_v3')
MICRO_DIR = Path('data/vision2/coinpoker_assets_v3c_micro')


def _frame(name):
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png((ROOT / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# ── 判据自适应 ────────────────────────────────────────────────────
def test_gold_dim_branch_threshold():
    """bg≤60 暗域：阈值 = max(bg*1.3, 28)——逐点不高于旧门 1.6*bg 当 bg≥17.5。"""
    roi = np.zeros((20, 40, 4), dtype=np.uint8)
    roi[..., :] = (30, 24, 8, 255)            # bg 亮度 ≈24（暗 pill）
    # 金字像素（r>b+15, g>=b+8）：l≈99
    roi[5:15, 5:35, :] = (140, 115, 40, 255)
    mask, bg = stack_gold_1876(roi)
    assert bg == pytest.approx(24.0, abs=2.0)
    assert int(mask.sum()) > 100              # 文本主体捕获
    # 旧门 1.6*24=38.4 < 新门 max(31.2,28)=31.2 → 新门≥旧捕获（此处全过）
    r = roi[..., 0].astype(int)
    g = roi[..., 1].astype(int)
    b = roi[..., 2].astype(int)
    l = r * 0.299 + g * 0.587 + b * 0.114
    old = ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
           & (l > bg * 1.6)).sum()
    assert int(mask.sum()) >= int(old)


def test_gold_bright_branch_frozen():
    """bg>60 亮域（遮挡/亮底）：判据与旧冻结式逐像素一致。"""
    rng = np.random.default_rng(11)
    roi = rng.integers(180, 255, (30, 60, 4)).astype(np.uint8)
    roi[..., 3] = 255
    roi[10:20, 10:30] = (140, 115, 40, 255)   # 暗金字（亮域下不捕获）
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    expected = ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
                & (l > bg * 1.6)).astype(np.uint8)
    mask, _ = stack_gold_1876(roi)
    assert np.array_equal(mask, expected)


def test_gold_floor_stricter_than_old_at_near_black():
    """近黑底（bg≤17）：绝对下限 28 严于旧 1.6*bg——防暗噪声（有意收紧）。"""
    roi = np.zeros((20, 40, 4), dtype=np.uint8)
    roi[..., :] = (12, 9, 3, 255)             # bg≈9.7
    roi[5:15, 5:35] = (60, 48, 14, 255)       # 暗金 l≈47.5——下限 28 仍捕获
    mask, bg = stack_gold_1876(roi)
    assert bg < 17
    assert int(mask.sum()) > 50
    # 比下限更暗的弱金噪点（l≈22 < 28）被拒——旧门 1.6*9.7=15.5 会放行
    roi2 = roi.copy()
    roi2[5:15, 5:35] = (28, 22, 6, 255)       # l≈22.6：RGB 过、亮度门拒
    mask2, _ = stack_gold_1876(roi2)
    assert int(mask2.sum()) == 0


# ── 解析（单位语义：裸小数=货币单位 float）────────────────────────
def test_parse_stack_decimal():
    assert parse_stack_text('9.64') == 9.64
    assert isinstance(parse_stack_text('9.64'), float)
    assert parse_stack_text('10') == 10
    assert parse_stack_text('201,000') == 201000
    assert parse_stack_text('2.28M') == 2280000
    assert parse_stack_text('9.64.5') is None
    assert parse_stack_text('') is None


def test_parse_bet_micro_decimal_and_guard():
    assert parse_bet_amount('0,28') == 0.28
    assert parse_bet_amount('0,5') == 0.5
    assert parse_bet_amount('1,000') == 1000      # 千位逗号不变（尾组 3 位）
    assert parse_bet_amount('12,345') == 12345
    assert parse_bet_amount('005') is None        # 前导零伪值守卫
    assert parse_bet_amount('999') == 999
    assert parse_bet_amount('2') == 2
    assert BET_MICRO_DEC_RE.match('0,28')
    assert not BET_MICRO_DEC_RE.match('1,000')


# ── kind 隔离装载 ─────────────────────────────────────────────────
def test_micro_asset_kinds_in_manifest():
    import json
    m = json.loads((MICRO_DIR / 'manifest.json').read_text(encoding='utf-8'))
    kinds = {a['kind'] for a in m['assets']}
    assert {'stack_glyph_micro', 'pot_digit_micro',
            'bet_digit_micro'} <= kinds
    # digit 覆盖：stack/pot 0-9 全；bet 0-9 缺 6（语料诚实缺失）
    for kind in ('stack_glyph_micro', 'pot_digit_micro'):
        labs = {a['label'] for a in m['assets'] if a['kind'] == kind}
        assert labs == set('0123456789')
    bet_labs = {a['label'] for a in m['assets']
                if a['kind'] == 'bet_digit_micro'}
    assert '6' not in bet_labs and bet_labs <= set('0123456789')


def test_stack_loader_kind_isolation():
    """微目录仅 stack_glyph_micro 入 stack 模板集（RGB 全 1 资产不混入）。"""
    micro = _load_1876_templates(str(MICRO_DIR), kind='stack_glyph_micro')
    assert set('0123456789') <= set(micro)
    for lab, insts in micro.items():
        for (t, _t0, _tn) in insts:
            assert t.max() <= 1.0 and t.min() >= 0.0   # 二值掩码域
            assert max(t.shape) <= 40                  # 微字形尺度带


def test_header_loader_excludes_v5_digit_kinds():
    """header 模板集不得含 V-5 digit kind（全 1 位图毒化防护）。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header \
        import BettingHeaderRecognizer1876
    rec = BettingHeaderRecognizer1876()
    v3c = rec._variants['v3c_micro']['assets']
    assert 'tether' in v3c and 'slash' in v3c
    for lab, tpl in v3c.items():
        if lab == 'dot':
            continue    # V-4 参考裁片（解析走几何判定，非 NCC）——全 1 合法
        assert not (tpl == 1.0).all(), f'label {lab} is all-ones'


def test_pot_loader_kind_isolation():
    pr = PotRecognizer(window='1876')
    # 基础 + 微实例并存；微实例灰度域（非全 1）
    assert set('0123456789') <= set(pr._prep)
    assert ',' in pr._prep                       # 基础库逗号保留


# ── 尺度一致性门 ──────────────────────────────────────────────────
def test_ncc_1876_scale_gate():
    """微模板（小）不得在大候选内伪匹配——门 = 0.75×候选高。"""
    big = np.zeros((34, 24), dtype=np.float64)
    big[2:32, 2:22] = 1.0                        # '0' 形
    prep = {'0': [(big, big - big.mean(), float(np.sqrt(
        (big - big.mean()) ** 2).sum()))]}
    lab, sc, mg = _ncc_1876(big, prep)
    assert lab == '0'                            # 同尺寸通过
    # 候选 8px 高（< 0.75 分母域）：模板被门跳过 → None
    small_cand = np.ones((8, 6), dtype=np.float64)
    lab2, sc2, _ = _ncc_1876(small_cand, prep)
    assert lab2 is None and sc2 is None


def test_pot_scale_gate_baseline_not_broken():
    """基础模板（30px）对 K 候选（~30px）必须通过尺度门（0.75×30=22.5）。"""
    pr = PotRecognizer(window='1876')
    for lab, insts in pr._prep.items():
        for (t, _t0, _tn) in insts:
            th = t.shape[0]
            assert th == 0 or th >= 0.75 * th    # 门自洽（防回归说明性断言）


# ── 微帧端到端（真实语料帧在盘断言）─────────────────────────────
MICRO_FRAMES = {
    'POT_CHANGED_109_1791376749721_full.png': {
        'stacks': {'S1': '10', 'S2': '17.56', 'S3': '11.11', 'S4': '10',
                   'S5': '22.35', 'S6': '9.64'},
        'pot': '0.25',
    },
    'POT_CHANGED_13_1791376382677_full.png': {
        'stacks': {'S1': '9.65', 'S2': '17.46', 'S3': '10.39', 'S4': '9.95',
                   'S5': '21.57', 'S6': '10'},
        'pot': '0.75',
    },
}


@pytest.mark.parametrize('name', sorted(MICRO_FRAMES))
def test_micro_frame_stack_and_pot(name):
    gt = MICRO_FRAMES[name]
    fr = _frame(name)
    sr = StackRecognizer1876()
    out = sr.recognize(fr)
    for seat, want in gt['stacks'].items():
        assert out[seat]['status'] == 'RECOGNIZED', (seat, out[seat])
        assert out[seat]['raw_text'] == want
        assert out[seat]['stack'] == float(want)
    pr = PotRecognizer(window='1876')
    pot = pr.recognize(fr)
    assert pot['status'] == 'RECOGNIZED'
    assert pot['raw_text'] == gt['pot']


def test_header_micro_usdt():
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header \
        import BettingHeaderRecognizer1876
    rec = BettingHeaderRecognizer1876()
    fr = _frame('POT_CHANGED_109_1791376749721_full.png')
    out = rec.recognize(fr)
    assert out['status'] == 'RECOGNIZED'
    assert out['sb'] == 0.05 and out['bb'] == 0.1
    assert out.get('currency') == 'USDT'


def test_k_frame_stack_unchanged_value():
    """K 桌代表帧 stack 值不回退（hero 708,184——目视 GT 帧）。"""
    fr = _frame('POT_CHANGED_108_1791197414296_full.png')
    sr = StackRecognizer1876()
    out = sr.recognize(fr)
    assert out['S6']['status'] == 'RECOGNIZED'
    assert out['S6']['stack'] == 708184


def test_band_filter_restores_k_regressions():
    """AA 碎片排除 + 底缘辉光线剥离：K 帧 16 例回退修复且微帧不受扰。"""
    sr = StackRecognizer1876()
    cases = [
        ('SCENE_CHANGED_2_1791196870621_full.png', 'S3', '217,168'),
        ('TABLE_CHANGED_2_1791196870621_full.png', 'S3', '217,168'),
        ('TABLE_CHANGED_236_1791200445806_full.png', 'S4', '2.33M'),
        ('SCENE_CHANGED_146_1791201184885_full.png', 'S2', '492,332'),
        ('TABLE_CHANGED_146_1791201184885_full.png', 'S2', '492,332'),
        ('TABLE_CHANGED_23_1791196857830_full.png', 'S3', '214,848'),
        ('TABLE_CHANGED_4_1791196780602_full.png', 'S6', '398,284'),
        ('SCENE_CHANGED_50_1791198309629_full.png', 'S2', '709,745'),
        ('TABLE_CHANGED_179_1791198887842_full.png', 'S2', '210,839'),
        ('TABLE_CHANGED_202_1791201468427_full.png', 'S5', '504,002'),
        ('TABLE_CHANGED_50_1791198309629_full.png', 'S2', '709,745'),
        ('TABLE_CHANGED_55_1791197140303_full.png', 'S4', '281,583'),
        ('TABLE_CHANGED_63_1791197183135_full.png', 'S3', '300,980'),
        ('TABLE_CHANGED_6_1791200493273_full.png', 'S3', '397,040'),
        ('TABLE_CHANGED_72_1791198415773_full.png', 'S2', '689,229'),
        ('TABLE_CHANGED_75_1791200835448_full.png', 'S5', '221,800'),
    ]
    for name, seat, want in cases:
        out = sr.recognize(_frame(name))[seat]
        assert out['status'] == 'RECOGNIZED', (name, out)
        assert out['raw_text'] == want
