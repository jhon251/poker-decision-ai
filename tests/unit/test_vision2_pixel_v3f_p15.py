"""V3F-P15 — 1876 State Calibration Foundation 单元测试。

覆盖：
- render-class 几何门（window_for_frame 正确键控 + 未知几何 → None）；
- Board 1876 正例（真实 1876 帧 → 1876 槽位识别，非 1419 槽位）；
- Board 1419 回归（既有 SETTLED 帧不变）；
- 跨 render class 污染门（1876 帧不得按 1419 槽位/ROI 识别）；
- 未校准字段诚实 EVIDENCE_LIMITED（pot/hero/status @1876；V3F-P29-B 后
  hero_cards 已校准→UNCERTAIN-first）；
- Position 1876 → 1848 锚点派发；Stack 1876 → 1848 全座；
- 未知几何 → EVIDENCE_LIMITED（绝不 fallback 到 1419）；
- 确定性重复。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.render_class import (  # noqa: E402
    EVIDENCE_LIMITED, window_for_frame)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    BoardObserver, HeroCardsObserver, PositionObserver, PotObserver,
    StackObserver, StatusObserver)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (  # noqa: E402
    CoinPokerBoardRecognizer, SLOT_GEOM_BY_WINDOW)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
BOARD_1876 = "BUTTON_CHANGED_167_1787992682749_full.png"
BOARD_1419 = "SETTLED_BOARD_CHECK_961_1787653653866_full.png"


def _frame(fname: str):
    w, h, rgba = decode_png((EV / fname).read_bytes())
    return rgba, w, h


def _blank_1876() -> tuple:
    w, h = 1876, 1400
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[..., :3] = (40, 90, 60)
    frame[..., 3] = 255
    return frame.tobytes(), w, h


# ── 1. render-class 几何门 ─────────────────────────────────────────────
def test_window_for_frame_mapping():
    assert window_for_frame(1419, 1066) == "1419"
    assert window_for_frame(1876, 1400) == "1876"
    assert window_for_frame(1848, 1388) == "1848"
    assert window_for_frame(1280, 779) == "1280"


def test_window_for_frame_unknown_is_none():
    for wh in ((100, 100), (1876, 1066), (1419, 1400), (0, 0),
               (1846, 1388)):
        assert window_for_frame(*wh) is None, wh


def test_board_slot_geometry_dispatches_by_window():
    assert CoinPokerBoardRecognizer(window="1419").slot_x0 == 664
    assert CoinPokerBoardRecognizer(window="1876").slot_x0 == 590
    assert SLOT_GEOM_BY_WINDOW["1876"] == (590, 551, 140, 180)


# ── 2. Board 1876 正例 + 1419 回归 ─────────────────────────────────────
def test_board_1876_recognized_real_frame():
    rgba, w, h = _frame(BOARD_1876)
    assert (w, h) == (1876, 1400)
    out = BoardObserver().observe(rgba, w, h)
    assert out["status"] == "RECOGNIZED"
    # V-2 更新（test-side only）：该帧目视 GT = 3c 7h Qc 5c 共 4 张
    # （docs/vision_v2_board/crops/frame167_1876_verify.png）。V-1 只读
    # 2 张且把 7♥ 误判为 "6c"（红 7 模板缺席 → 族内错配）；V-2 严格全族
    # 复核修复 7h，并新增 5c（slot3，此前 UNCERTAIN）；Qc 中心花色证据
    # 不足 → RANK_ONLY（rank 计数有效、不入牌面）。
    assert out["count"] == 3
    assert set(out["cards"]) == {"3c", "7h", "5c"}
    assert out["rank_only_cards"] == ["Q"]
    assert out["countable_count"] == 4
    assert out["window"] == "1876"


def test_board_1419_regression_unchanged():
    rgba, w, h = _frame(BOARD_1419)
    assert (w, h) == (1419, 1066)
    out = BoardObserver().observe(rgba, w, h)
    assert out["count"] == 5                     # river 5 卡（原 P2 语义）
    assert out["cards"] == ["3c", "6h", "Th", "Ac", "5h"]
    assert out["window"] == "1419"


def test_board_blank_1876_not_fake_recognized():
    """空 1876 帧 → 不按 1419 槽位误识别（空板 ≠ 认可）。"""
    fb, w, h = _blank_1876()
    out = BoardObserver().observe(fb, w, h)
    assert out["status"] != "RECOGNIZED"
    assert out["count"] == 0
    assert out["window"] == "1876"


# ── 3. 跨 render class 污染门（未校准字段） ────────────────────────────
def test_1876_uncalibrated_fields_evidence_limited():
    rgba, w, h = _frame(BOARD_1876)
    # V3F-P22-A：1876 Pot 已校准（P21 视觉修正 ROI）→ RECOGNIZED，不再是 E-L
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        parse_stack_text)
    assert PotObserver().observe(rgba, w, h)["status"] == "RECOGNIZED"
    # V3F-P29-B：1876 HeroCards 已校准（GT-provenance 模板，P29-A 证据）→
    # 不再 EVIDENCE_LIMITED；该帧 hero folded → 诚实 UNCERTAIN（value=None）
    # —— UNKNOWN-first 门保持：不因校准而猜测/伪识别。
    assert HeroCardsObserver().observe(rgba, w, h)["status"] == "UNCERTAIN"
    # V3F-P25-C：1876 Status 已校准（P25-B.3 冻结判据）→ 非 EVIDENCE_LIMITED
    assert StatusObserver().observe(rgba, w, h)["status"] != EVIDENCE_LIMITED
    # 值必须为 None（UNKNOWN ≠ 0/默认/前值）
    assert HeroCardsObserver().observe(rgba, w, h)["cards"] is None


# ── 4. Position/Stack 1876 → 1848 锚点派发 ─────────────────────────────
def test_position_1876_dispatches_to_1848():
    assert PositionObserver._window_for(1876, 1400) == "1848"
    assert PositionObserver._window_for(1419, 1066) == "1419"
    assert PositionObserver._window_for(1280, 779) is None


def test_position_1876_not_evidence_limited():
    """1876 帧 → Position 尝试 1848 锚点识别（非 EVIDENCE_LIMITED 隔离）。"""
    rgba, w, h = _frame(BOARD_1876)
    out = PositionObserver().observe(rgba, w, h)
    assert out.get("status") != EVIDENCE_LIMITED
    assert out["runtime_result"]["uncertainty"] in ("UNKNOWN", "RECOGNIZED")


def test_stack_1876_all_six_seats_no_crop():
    """1876 帧 → Stack 用 1848 全座（S2/S6 不裁剪，6 座全返回）。"""
    rgba, w, h = _frame(BOARD_1876)
    out = StackObserver().observe(rgba, w, h)
    assert out["window"] == "1876"
    assert set(out["seats"]) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    # 任何座不得出现 "cropped" 理由（1848 全见）
    for seat, v in out["seats"].items():
        assert v.get("reason") != "cropped", seat


# ── 5. 未知几何 → EVIDENCE_LIMITED（绝不 fallback 1419） ──────────────
def test_unknown_geometry_evidence_limited_not_fallback():
    fb, w, h = (np.zeros((700, 1000, 4), dtype=np.uint8)
                .tobytes(), 1000, 700)
    assert BoardObserver().observe(fb, w, h)["status"] == EVIDENCE_LIMITED
    assert PotObserver().observe(fb, w, h)["status"] == EVIDENCE_LIMITED
    assert HeroCardsObserver().observe(fb, w, h)["status"] == EVIDENCE_LIMITED
    assert StatusObserver().observe(fb, w, h)["status"] == EVIDENCE_LIMITED
    assert StackObserver().observe(fb, w, h)["status"] == EVIDENCE_LIMITED
    pos = PositionObserver().observe(fb, w, h)
    assert pos["status"] == EVIDENCE_LIMITED


# ── 6. 确定性 ─────────────────────────────────────────────────────────
def test_board_1876_deterministic():
    rgba, w, h = _frame(BOARD_1876)
    a = BoardObserver().observe(rgba, w, h)
    b = BoardObserver().observe(rgba, w, h)
    # V-2：同帧两次识别确定性不变；期望值更新同 test_board_1876_recognized_real_frame
    assert a["cards"] == b["cards"] == ["3c", "7h", "5c"]


def test_gate_is_deterministic():
    assert window_for_frame(1876, 1400) == window_for_frame(1876, 1400)
    assert SLOT_GEOM_BY_WINDOW["1876"] == SLOT_GEOM_BY_WINDOW["1876"]
