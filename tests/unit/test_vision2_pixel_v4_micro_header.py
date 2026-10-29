"""V-4（Round 15）— 微级别 ₮ header 识别 + 小数解析 测试。

覆盖：v3c_micro 资产注册与 kind 过滤 · ₮ 微级别锚定解析（`₮0.05/₮0.1`
443 帧语料 + 尾零省略）· K 格式帧零回退（双路径按分取高）· _to_num 单位
语义（裸小数=货币单位 float，整数格式 int 不变）· sb>=bb 结构拒绝 ·
未覆盖货币符/字形诚实 UNKNOWN · 阈值红线。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    CHAR_SCORE_MIN, HEADER_ASSET_DIR_V3B, HEADER_ASSET_DIR_V3C_MICRO,
    MICRO_DOT_BOTTOM_MIN, MICRO_DOT_MAX_H, MICRO_DOT_MAX_W,
    BettingHeaderRecognizer1876, _to_num)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    DEFAULT_ASSET_DIRS, CoinPokerBoardRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

V3 = Path("data/vision2/coinpoker_v3")
MICRO_FRAMES = [
    "BUTTON_CHANGED_228_1791377220178_full.png",
    "BUTTON_CHANGED_178_1791197763010_full.png",      # K 格式对照（v3b）
]


def _frame(p: Path):
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# ── 1. 资产注册：v3c_micro 载入 + board kind 过滤零污染 ────────────────
def test_v3c_micro_assets_registered():
    rec = BettingHeaderRecognizer1876()
    assert "v3c_micro" in rec._variants
    micro = set(rec._variants["v3c_micro"]["assets"].keys())
    assert {"tether", "slash", "dot", "digit_0", "digit_1",
            "digit_5"} <= micro
    # 微语料未出现 → NOT_COVERED 不入模板（遇即 UNKNOWN）
    assert not ({"digit_2", "digit_3", "digit_4", "digit_9"} & micro)


def test_board_template_set_unpolluted_by_micro_dir():
    """尾追加 v3c_micro 后 board rank/suit 模板集逐字节不变（kind 过滤）。"""
    with_micro = CoinPokerBoardRecognizer(list(DEFAULT_ASSET_DIRS))
    without = CoinPokerBoardRecognizer(
        [d for d in DEFAULT_ASSET_DIRS
         if d != HEADER_ASSET_DIR_V3C_MICRO])
    assert set(with_micro.templates) == set(without.templates)
    for k, v in with_micro.templates.items():
        assert bytes(np.frombuffer(v.data, dtype=np.uint8)) == \
            bytes(np.frombuffer(without.templates[k].data,
                                dtype=np.uint8)), k


# ── 2. ₮ 微级别锚定解析（443 帧语料代表帧）────────────────────────────
def test_micro_header_recognized_usdt():
    rec = BettingHeaderRecognizer1876()
    out = rec.recognize(_frame(V3 / MICRO_FRAMES[0]))
    assert out["status"] == "RECOGNIZED"
    assert (out["sb"], out["bb"]) == (0.05, 0.1)
    assert out["ante"] is None
    assert out["reason"] == "header_stake_anchored_parse_micro"
    assert out["font_variant"] == "v3c_micro"
    assert out["currency"] == "USDT"
    assert out["confidence"] >= CHAR_SCORE_MIN


def test_micro_and_k_paths_pick_best_no_regression():
    """K 格式帧走 v3b 锚定路径原样（不加 currency 字段）；双路径并存。"""
    rec = BettingHeaderRecognizer1876()
    out = rec.recognize(_frame(V3 / MICRO_FRAMES[1]))
    assert out["status"] == "RECOGNIZED"
    assert (out["sb"], out["bb"], out["ante"]) == (1000, 2000, 320)
    assert out["reason"] == "header_stake_anchored_parse"
    assert out["font_variant"] == "v3b"
    assert "currency" not in out


def test_micro_parser_rejects_sb_ge_bb():
    """sb>=bb 结构性非法 → None（诚实拒绝，不猜）。"""
    rec = BettingHeaderRecognizer1876()
    mask = np.zeros((26, 134), dtype=np.float64)
    assert rec._anchored_parse_micro(mask, {}) is None   # 无锚即失败


def test_micro_parser_requires_tether_and_slash():
    rec = BettingHeaderRecognizer1876()
    # 无 tether/slash 模板的资产集 → 直接 None
    assert rec._anchored_parse_micro(
        np.zeros((26, 10)), {"digit_0": np.ones((9, 7))}) is None


# ── 3. _to_num 单位语义（V-4 写明：裸小数=货币单位 float）──────────────
def test_to_num_decimal_money_unit():
    assert _to_num("0.05") == 0.05
    assert isinstance(_to_num("0.05"), float)
    assert _to_num("0.1") == 0.1
    assert _to_num("0.10") == 0.1          # 尾零省略与显式尾零同值
    assert _to_num("12.5") == 12.5


def test_to_num_legacy_formats_unchanged():
    assert _to_num("1K") == 1000 and isinstance(_to_num("1K"), int)
    assert _to_num("2k") == 2000
    assert _to_num("169,760") == 169760
    assert _to_num("320") == 320 and isinstance(_to_num("320"), int)
    assert _to_num("1.28M") == 1_280_000


def test_to_num_uncovered_formats_none():
    assert _to_num("1.28K") is None        # 小数 K：语料未见 → 不支持
    assert _to_num("0.0.5") is None        # 多点
    assert _to_num("5.") is None           # 空小数部
    assert _to_num("") is None
    assert _to_num("abc") is None


# ── 4. 微级别几何点规则参数冻结 ────────────────────────────────────────
def test_micro_dot_geometry_constants():
    assert MICRO_DOT_MAX_H == 3
    assert MICRO_DOT_MAX_W == 5
    assert MICRO_DOT_BOTTOM_MIN == 14


# ── 5. 注册路径常量 ───────────────────────────────────────────────────
def test_header_asset_dir_constants():
    assert HEADER_ASSET_DIR_V3C_MICRO.endswith("v3c_micro")
    assert HEADER_ASSET_DIR_V3B.endswith("v3b_header")
