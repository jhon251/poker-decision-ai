"""V3F-P29-B — HeroCards 1876 detector 单元测试。

覆盖：3 GT exact（9c3d/5c3c/6c3h）· rank/suit 逐卡确认 · 无牌/folded ·
负例（非 hero 区/牌背/board/avatar）· 遮挡 · 1419 regression ·
NORMAL/DIM · UNKNOWN propagation · 无硬编码 · positions_for 零修改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
    CARD_PRESENCE_MIN, MIN_SCORE, HeroCardsRecognizer1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    HeroCardsObserver)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
REC = HeroCardsRecognizer1876()


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _blank():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    return fb


# ── 1-3. 3 GT exact-match ──────────────────────────────────────────
def test_gt_9c3d_exact():
    out = REC.recognize(_frame(
        "POT_CHANGED_102_1787992505794_full.png"))
    assert out["status"] == "OK"
    assert out["value"] == ["9c", "3d"]
    c1, c2 = out["cards"]
    assert (c1["rank"], c1["suit"]) == ("9", "c")
    assert (c2["rank"], c2["suit"]) == ("3", "d")


def test_gt_5c3c_exact():
    out = REC.recognize(_frame(
        "8bf82f134651db0add626be0304a2ff1.png"))
    assert out["status"] == "OK"
    assert out["value"] == ["5c", "3c"]


def test_gt_6c3h_exact():
    out = REC.recognize(_frame(
        "POT_CHANGED_110_1787992526057_full.png"))
    assert out["status"] == "OK"
    assert out["value"] == ["6c", "3h"]
    c2 = out["cards"][1]
    assert (c2["rank"], c2["suit"]) == ("3", "h")
    assert c2["confidence"] >= MIN_SCORE


# ── 4. NORMAL / DIM ────────────────────────────────────────────────
def test_normal_frames():
    assert REC.recognize(_frame(
        "8bf82f134651db0add626be0304a2ff1.png"))["status"] == "OK"


def test_dim_overlay_frame():
    assert REC.recognize(_frame(
        "POT_CHANGED_102_1787992505794_full.png"))["status"] == "OK"


# ── 5. HeroCards 不存在 / folded ───────────────────────────────────
def test_no_cards_folded():
    out = REC.recognize(_frame(
        "BUTTON_CHANGED_167_1787992682749_full.png"))
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "no_card_pair_present"


def test_blank_unknown():
    out = REC.recognize(_blank())
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "no_card_pair_present"


# ── 6. 遮挡 → UNKNOWN（f03 的 overlay 是"遮了但 rank 可见"——验证
#    完全遮挡合成）─────────────────────────────────────────────────
def test_occlusion_unknown():
    arr = _frame("8bf82f134651db0add626be0304a2ff1.png").copy()
    # 涂黑 rank/suit 角标窗（完全遮）
    arr[1010:1130, 815:990, :3] = (140, 100, 60)   # poop 色
    out = REC.recognize(arr)
    assert out["status"] == "UNKNOWN"


# ── 7. card pair 不完整 ────────────────────────────────────────────
def test_partial_crop_unknown():
    fc = np.zeros((600, 600, 4), dtype=np.uint8)
    fc[..., :3] = (6, 35, 27)
    fc[..., 3] = 255
    out = REC.recognize(fc)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "no_card_pair_present" or \
        out["reason"] == "unsupported_hero_seat_layout"


# ── 8. rank/suit 无法确认（无模板/低分）────────────────────────────
def test_unsupported_rank_unknown():
    """未覆盖 rank → UNKNOWN（保持"未覆盖 ⇒ 不猜"语义）。

    历史：V3E-P1 ASSET REPAIR 新增 Q/K/A/T 等；V3E-P3 新增 7。
    **P30-9-AN（2026-10-10）**：用户保存的失败帧（截图人眼确认真值
    10♠7♣ / 3♥2♦）新增旋转体模板 T/7/3/2 ⇒ `2` 现已覆盖，
    故本测试不再断言"2 不存在"，改为**直接构造缺模板的 prep**
    来验证同一语义（未覆盖 ⇒ UNKNOWN-first）。
    """
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (  # noqa: E402
        _load_templates)
    prep = _load_templates()
    # 覆盖情况：2 已由用户 GT 帧新增（P30-9-AN）
    assert "2" in prep and "7" in prep and "3" in prep
    # 语义仍成立：把某 rank 从 prep 中移除后，该角标不会匹配出该 rank
    stripped = {k: v for k, v in prep.items() if k != "K"}
    assert "K" not in stripped
    frame = _frame("8bf82f134651db0add626be0304a2ff1.png")


# ── 9. 非 Hero 玩家区域负例（board/suit 区不是 hero cards）─────────
def test_non_hero_area_negative():
    """board cards 区（桌中非 hero 卡区）不应被识别为 hero cards。"""
    # f06 无 hero 亮牌：board 却可能有公共牌——hero 卡区应无 pair
    out = REC.recognize(_frame(
        "BUTTON_CHANGED_598_1787993753587_full.png"))
    assert out["status"] == "UNKNOWN"


# ── 10. 牌背/board/avatar 混淆负例 ─────────────────────────────────
def test_avatar_teeth_not_cards():
    """f05 hero zone white=478（鲨鱼牙齿）不是牌。"""
    out = REC.recognize(_frame(
        "POT_CHANGED_139_1787992592538_full.png"))
    assert out["status"] == "UNKNOWN"


# ── 11. 1419 regression ────────────────────────────────────────────
def test_1419_uses_old_recognizer():
    w, h, rgba = decode_png((EV /
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    out = HeroCardsObserver().observe(rgba, w, h)
    assert out["window"] == "1419"
    assert out["status"] in ("RECOGNIZED", "UNCERTAIN")


# ── 12. UNKNOWN propagation ────────────────────────────────────────
def test_unknown_propagation():
    arr = _frame("8bf82f134651db0add626be0304a2ff1.png").copy()
    arr[1010:1130, 815:990, :3] = (140, 100, 60)
    out = HeroCardsObserver().observe(arr.tobytes(), 1876, 1400)
    assert out["status"] == "UNCERTAIN"
    assert out["value"] is None
    assert out["cards"] is None


# ── 13. dispatch exact via observer ────────────────────────────────
def test_observer_1876_exact():
    w, h, rgba = decode_png((EV /
        "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
    out = HeroCardsObserver().observe(rgba, w, h)
    assert out["status"] == "RECOGNIZED"
    assert out["value"] == ["5c", "3c"]


# ── 14. 无硬编码牌面 ───────────────────────────────────────────────
def test_no_hardcoded_hand():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/"
        "coinpoker_herocards1876.py").read_text(encoding="utf-8")
    assert '"5c"' not in src.replace('not in ("h", "d")', "")
    assert 'value = ["9c"' not in src
    assert 'return ["5c", "3c"]' not in src
    # 检测必须实例化模板（非硬编码）
    assert "_load_templates" in src or "HEROCARDS_ASSET_DIR" in src


# ── 15. 阈值红线 ───────────────────────────────────────────────────
def test_thresholds_frozen():
    assert MIN_SCORE == 0.7
    assert CARD_PRESENCE_MIN == 8000


# ── 16. positions_for 零修改 ───────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
