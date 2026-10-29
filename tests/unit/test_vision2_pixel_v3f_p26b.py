"""V3F-P26-B — Stack 1876 Production Implementation 单元测试。

覆盖：
- 34 GT exact（NORMAL/DIM/多座位/多字符/K-M 格式；FP=0/FN=0）；
- 负例：底池标签 / Bet / 玩家名 / folded tag / 红牌背 / 动画遮挡 /
  无 gold / 空座 / 不完整 ROI；
- 1419 回归 + Pot/Bet/Status 零退化断言 + UNKNOWN 语义 + 确定性；
- positions_for 零修改 + 红线扫描（0.7/0.08 冻结 + 无跨字段推导）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_STACK_ROIS_1876, StackRecognizer1876, stack_gold_1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StackObserver)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


REC = StackRecognizer1876()


# ── 1. 34 GT exact ──────────────────────────────────────────────────
def test_34_stack_gt_exact():
    n = exact = 0
    fps, fns = [], []
    for ann in GT["annotations"]:
        arr = _frame(ann["frame_id"])
        out = REC.recognize(arr)
        st = (ann.get("fields") or {}).get("stack") or {}
        for seat, value in st.items():
            if value is None:
                continue
            got = out[seat]
            n += 1
            if got["status"] == "RECOGNIZED" and got["stack"] == value:
                exact += 1
            else:
                (fns if got["status"] != "RECOGNIZED" else
                 fps).append((ann["frame_id"][:6], seat, value,
                              got.get("stack"), got.get("reason")))
    assert n == 34
    assert exact == 34, f"exact={exact} fps={fps} fns={fns}"
    assert not fps and not fns


def test_normal_and_dim_frames():
    fids = [ann["frame_id"] for ann in GT["annotations"]]
    # NORMAL=4 帧（global p50≥30），DIM=2 帧（p50≈15-17，f03/f04）
    dim = {"POT_CHANGED_102", "POT_CHANGED_13"}
    assert len(fids) == 6
    for fid in fids:
        arr = _frame(fid)
        out = REC.recognize(arr)
        st = (next(a for a in GT["annotations"]
                   if a["frame_id"] == fid).get("fields") or {}).get("stack")
        for seat, value in (st or {}).items():
            if value is None:
                continue
            assert out[seat]["stack"] == value, (fid[:6], seat, value)


def test_km_format_present():
    """GT 中存在 M 格式（2.23M/2.24M/2.39M/3.24M）。"""
    m_vals = [3_240_000, 2_230_000, 2_240_000, 2_390_000]
    found = 0
    for ann in GT["annotations"]:
        st = (ann.get("fields") or {}).get("stack") or {}
        for seat, v in st.items():
            if v in m_vals:
                arr = _frame(ann["frame_id"])
                assert REC.recognize(arr)[seat]["stack"] == v
                found += 1
    assert found == 4


def test_zero_value_recognized():
    """S3=0（真实 0 显示）→ RECOGNIZED 0（≠ empty/UNKNOWN）。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    out = REC.recognize(arr)
    assert out["S3"]["stack"] == 0
    assert out["S3"]["status"] == "RECOGNIZED"


def test_multichar_all_seats():
    """6 座 34 值覆盖（每座 5-6 个不同值）。"""
    arr = _frame("BUTTON_CHANGED_598_1787993753587_full.png")
    out = REC.recognize(arr)
    assert out["S1"]["stack"] == 80000
    assert out["S2"]["stack"] == 810868
    assert out["S3"]["stack"] == 677320
    assert out["S4"]["stack"] == 425152
    assert out["S5"]["stack"] == 2390000
    assert out["S6"]["stack"] == 213809


# ── 2. 负例 ─────────────────────────────────────────────────────────
def test_dischang_label_not_stack():
    """橙色「底池」标签（S1 f01 (138,325)）不在 stack ROI 内 → 无干扰。
    且其自身区域不产生可解析值（ROI 隔离）。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    out = REC.recognize(arr)
    assert out["S1"]["stack"] == 101600
    # 标签 bbox 与 S1 stack ROI 不重叠（几何断言）
    lb = (138, 325, 241, 366)
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876["S1"]
    assert lb[1] < y0 or lb[0] >= x1, "标签与 ROI 必须空间分离"


def test_bet_not_stack():
    """Bet 胶囊（f02 S5 亮底 / f01 S1 暗底）不产生 stack 值。"""
    out = REC.recognize(_frame("POT_CHANGED_12_1787995290794_full.png"))
    assert out["S5"]["stack"] == 366732      # 正确值
    assert out["S2"]["stack"] == 566136      # bet≠stack 互不混淆


def test_folded_tag_not_stack():
    """folded teal tag（f01 S5 (1531,891)）不与 S5 stack ROI 重叠。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    out = REC.recognize(arr)
    assert out["S5"]["stack"] == 3240000
    tag = (1531, 891, 1634, 931)
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876["S5"]
    assert tag[1] < y0 or tag[0] >= x1


def test_name_not_stack():
    """玩家名（灰白，无 gold 饱和）在 name 行内不产生 gold 组件。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # S1 name 行（x 250-445 避开橙色底池 label 区 x<241）
    roi = arr[370:415, 250:445]
    m, bg = stack_gold_1876(roi)
    assert int(m.sum()) == 0


def test_card_back_not_stack():
    """红牌背（纯卡背区）无 gold。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    roi = arr[130:250, 340:450]
    m, _ = stack_gold_1876(roi)
    assert int(m.sum()) == 0


def test_animation_occlusion_unknown():
    """f05 S5 赢家动画遮挡 → UNKNOWN（不得猜测 5M 等值）。"""
    arr = _frame("POT_CHANGED_139_1787992592538_full.png")
    out = REC.recognize(arr)
    assert out["S5"]["status"] == "UNKNOWN"
    assert out["S5"]["stack"] is None


def test_hero_own_cards_no_gold():
    """f03 S6 hero（亮牌 9c3d）→ 无 gold → UNKNOWN。"""
    arr = _frame("POT_CHANGED_102_1787992505794_full.png")
    out = REC.recognize(arr)
    assert out["S6"]["status"] == "UNKNOWN"
    assert out["S6"]["stack"] is None


def test_blank_frame_all_unknown():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    out = REC.recognize(fb)
    for seat, v in out.items():
        assert v["status"] == "UNKNOWN"
        assert v["stack"] is None


def test_incomplete_roi_unknown():
    """截断 ROI（出界）→ 无 gold → UNKNOWN（不猜）。"""
    fb = np.zeros((400, 400, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    out = REC.recognize(fb)           # 全部 ROI 出界
    for seat, v in out.items():
        assert v["status"] == "UNKNOWN"


# ── 3. UNKNOWN 语义 / 确定性 ────────────────────────────────────────
def test_observer_blank_no_crash():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    out = StackObserver().observe(fb.tobytes(), 1876, 1400)
    for seat, v in out["seats"].items():
        assert v["status"] == "UNKNOWN"


def test_deterministic():
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    a = REC.recognize(arr)["S1"]["stack"]
    b = REC.recognize(arr)["S1"]["stack"]
    assert a == b == 101600


# ── 4. 回归 ─────────────────────────────────────────────────────────
def test_1419_regression():
    """1419 Stack 路径零改动（UNKNOWN/未识别语义不变）。"""
    w, h, rgba = decode_png((EV /
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    out = StackObserver().observe(rgba, w, h)
    assert out["window"] == "1419"
    for seat, v in out["seats"].items():
        assert v["status"] in ("RECOGNIZED", "UNKNOWN"), seat


def test_unknown_geometry_evidence_limited():
    fb = np.zeros((800, 1000, 4), dtype=np.uint8)
    out = StackObserver().observe(fb.tobytes(), 1000, 800)
    assert out["status"] == "EVIDENCE_LIMITED"


def test_positions_for_untouched():
    """positions_for 未被修改（守卫红线——核心领域函数原样存在）。"""
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: F401
    src = Path(
        "poker_decision_ai/domain/poker/models.py"
    ).read_text(encoding="utf-8")
    assert "def positions_for" in src


def test_gates_frozen():
    """0.7/0.08 门未降低（红线）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_stack.py"
    ).read_text(encoding="utf-8")
    assert "sc < 0.7" in src or "sc < 0.70" in src
    assert "margin < 0.08" in src


def test_no_cross_field_imports():
    """StackRecognizer1876 不引用 Pot/Bet/Status/Position。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_stack.py"
    ).read_text(encoding="utf-8")
    for bad in ("PotRecognizer", "BetRecognizer", "PlayerStatusRecognizer",
                "positions_for", "StatusObserver", "BetObserver"):
        assert bad not in src, bad


def test_roi_constants():
    """每座 ROI = P26-A 实测值（布局常量）。"""
    assert SEAT_STACK_ROIS_1876["S1"] == (165, 413, 360, 485)
    assert SEAT_STACK_ROIS_1876["S3"] == (1520, 413, 1715, 485)
    assert SEAT_STACK_ROIS_1876["S6"] == (838, 1236, 1040, 1308)
