"""V3F-P13-B — Bet Recognition Foundation（1280×779 类）单元测试。

范围：
- ROI isolation（几何不相交 + 行为隔离）
- glyph extraction（真实证据）
- 数字/逗号识别闭环（≥5 真实值 ×≥3 独立帧；值矩阵来自 P13-A 像素级分组
  + 全帧上下文目视验证——非 fixture 冒充视觉 GT）
- UNKNOWN-first / partial crop / ambiguous / 格式拒绝
- Bet≠Stack / Pot / Status / Action（行为 + 模块边界）
- 1419 evidence-limited 语义保持
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (
    BET_ROIS_1280, BET_ROIS_1419, BetRecognizer, detect_bet_capsule,
    gray_luma, parse_bet_amount, segment_bet_glyphs, classify_bet_glyphs)

EV = Path("data/vision2/coinpoker_evidence")


def load(name: str) -> np.ndarray:
    w, h, rgba = decode_png((EV / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


REC = BetRecognizer(window="1280")


def blank_with(region: tuple, src: np.ndarray) -> np.ndarray:
    """真实像素区域注入黑帧（anti-confusion：只给 Bet 识别器看该区域）。"""
    x0, y0, x1, y1 = region
    out = np.zeros_like(src)
    out[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return out


# ---------------------------------------------------------------- ROI isolation

def test_roi_geometry_disjoint_from_stack_pot_badge():
    """Bet ROI 与 Stack 文本/Pot 胶囊/Status 徽章区域几何不相交
    （区域框 = 1280 窗口真实证据实测：栈文本 (198,287)-(340,320)；
    pot 胶囊 (843,353)-(1030,394)；跟注徽章 (219,194)-(287,231)）。"""
    stack = (198, 287, 340, 320)
    pot = (843, 353, 1030, 394)
    badge = (219, 194, 287, 231)
    for roi in BET_ROIS_1280.values():
        x0, y0, w, h = roi
        box = (x0, y0, x0 + w, y0 + h)

        def overlap(a, b):
            return not (a[2] <= b[0] or b[2] <= a[0]
                        or a[3] <= b[1] or b[3] <= a[1])
        assert not overlap(box, stack), f"Bet ROI {roi} 与 Stack 区重叠"
        assert not overlap(box, pot), f"Bet ROI {roi} 与 Pot 区重叠"
        assert not overlap(box, badge), f"Bet ROI {roi} 与 徽章区重叠"


# ------------------------------------------------------- closed loop real ×3

CLOSED_LOOP = {
    1000:   [("S_TL", "TABLE_CHANGED_100_1787928276694_full.png"),
             ("S_TL", "TABLE_CHANGED_101_1787928277940_full.png"),
             ("S_TL", "TABLE_CHANGED_102_1787928279169_full.png")],
    2000:   [("S_TL", "TABLE_CHANGED_213_1787928433101_full.png"),
             ("S_TL", "TABLE_CHANGED_215_1787928435301_full.png"),
             ("S_TL", "TABLE_CHANGED_217_1787928437510_full.png"),
             ("S_TR", "TABLE_CHANGED_100_1787928276694_full.png"),
             ("S_TR", "TABLE_CHANGED_101_1787928277940_full.png"),
             ("S_TR", "TABLE_CHANGED_102_1787928279169_full.png")],
    193400: [("S_TL", "TABLE_CHANGED_109_1787928287779_full.png"),
             ("S_TL", "TABLE_CHANGED_117_1787928295861_full.png"),
             ("S_TL", "TABLE_CHANGED_119_1787928298017_full.png")],
    199680: [("S_TR", "TABLE_CHANGED_117_1787928295861_full.png"),
             ("S_TR", "TABLE_CHANGED_119_1787928298017_full.png"),
             ("S_TR", "TABLE_CHANGED_120_1787928299217_full.png")],
    128190: [("S_TL", "TABLE_CHANGED_413_1787928698111_full.png"),
             ("S_TL", "TABLE_CHANGED_420_1787928708495_full.png"),
             ("S_TL", "TABLE_CHANGED_422_1787928711594_full.png"),
             ("S_TR", "TABLE_CHANGED_420_1787928708495_full.png")],
    102000: [("S_TL", "TABLE_CHANGED_185_1787928391857_full.png"),
             ("S_TL", "TABLE_CHANGED_192_1787928402162_full.png"),
             ("S_TL", "TABLE_CHANGED_193_1787928403806_full.png")],
}


def test_closed_loop_real_values_x3():
    """≥5 真实值 ×≥3 独立真实帧：raw frame → 胶囊 → 字形 → 识别 → 解析
    → 期望值（P13-A 已验证值）。0 错误值。"""
    assert len(CLOSED_LOOP) >= 5
    for value, instances in CLOSED_LOOP.items():
        assert len(instances) >= 3
        for zone, frame in instances:
            res = REC.recognize(load(frame))[zone]
            assert res["status"] == "RECOGNIZED", (
                f"{value} {zone} {frame}: {res}")
            assert res["bet"] == value, (
                f"{value} {zone} {frame}: got {res}")


def test_no_recognized_outside_verified_groups():
    """未验证帧：不得出现任何 RECOGNIZED（无伪阳性——全部 UNKNOWN）。"""
    for name in ("TABLE_CHANGED_39_1787928191464_full.png",
                 "TABLE_CHANGED_155_1787928347647_full.png",
                 "TABLE_CHANGED_340_1787928607049_full.png"):
        if not (EV / name).exists():
            continue
        res = REC.recognize(load(name))
        for zone in ("S_TL", "S_TR"):
            assert res[zone]["status"] != "RECOGNIZED", name


# ---------------------------------------------------------------- UNKNOWN-first

def test_unknown_first_no_capsule():
    """无胶囊帧 → UNKNOWN（bet=None，≠0）。"""
    frame = load("TABLE_CHANGED_39_1787928191464_full.png")
    for zone in ("S_TL", "S_TR"):
        res = REC.recognize(frame)[zone]
        assert res["status"] == "UNKNOWN"
        assert res["bet"] is None


def test_unknown_partial_crop_guard():
    """白墨贴 ROI 边（截断）→ None（拒）。"""
    lum = np.full((62, 191), 45.0)          # 胶囊暗底
    lum[0:5, 10:40] = 255.0                 # 贴顶边（截断）
    assert detect_bet_capsule(lum, (0, 0, 191, 62)) is None
    lum2 = np.full((62, 191), 45.0)
    lum2[20:49, 0:80] = 255.0               # 贴右边
    assert detect_bet_capsule(lum2, (0, 0, 191, 62)) is None


def test_unknown_ambiguous_glyph_geometry():
    """歧义字形（非数字非逗号几何）→ 分割/分类拒绝。"""
    mask = np.zeros((29, 60), dtype=np.uint8)
    mask[:, 0:43] = 1                       # 43px 宽超大块（无低墨切点）
    assert segment_bet_glyphs(mask) is None
    # 低但非底部锚定（高度差信号 vs 逗号几何）→ classify None
    glyphs = [{"mask": np.ones((8, 6), dtype=np.uint8), "x": 0, "y": 0,
               "w": 6, "h": 8},
              {"mask": np.ones((29, 17), dtype=np.uint8), "x": 10, "y": 0,
               "w": 17, "h": 29}]
    assert classify_bet_glyphs(glyphs, REC._templates) is None


def test_format_rejection():
    """格式拒绝：非千位格式 → None（不得猜值）。
    通用解析契约：纯数字串可解析（parse）；Bet 层 4-8 token 带 + 千位格式门
    另作拒识（fresh_ok/TOKENS_BAND——识别级保护，与解析器契约分离）。"""
    assert parse_bet_amount("1,000") == 1000
    assert parse_bet_amount("890,401") == 890401
    assert parse_bet_amount("10,45") is None
    assert parse_bet_amount("1,00,0") is None
    assert parse_bet_amount("12") == 12        # 通用解析器契约（非千位也合法）
    assert parse_bet_amount(None) is None


# ---------------------------------------------------------------- anti-confusion

def test_bet_not_stack_real_pixels():
    """Bet 识别器只给 Stack 真实像素 → UNKNOWN（不从 Stack 推导 Bet）。"""
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    only_stack = blank_with((198, 287, 340, 320), frame)
    for zone in ("S_TL", "S_TR"):
        assert REC.recognize(only_stack)[zone]["status"] == "UNKNOWN"


def test_bet_not_pot_real_pixels():
    """只给 Pot 胶囊真实像素 → UNKNOWN（不从 Pot 推导 Bet）。"""
    frame = load("TABLE_CHANGED_100_1787928276694_full.png")
    only_pot = blank_with((843, 353, 1030, 394), frame)
    for zone in ("S_TL", "S_TR"):
        assert REC.recognize(only_pot)[zone]["status"] == "UNKNOWN"


def test_bet_not_status_badge_real_pixels():
    """只给 Status 徽章真实像素（跟注 219,194-287,231）→ UNKNOWN。"""
    frame = load("TABLE_CHANGED_119_1787928298017_full.png")
    only_badge = blank_with((219, 194, 287, 231), frame)
    for zone in ("S_TL", "S_TR"):
        assert REC.recognize(only_badge)[zone]["status"] == "UNKNOWN"


def test_bet_module_has_no_action_dependency():
    """Action 保持 SEALED：Bet 模块零 Action 耦合（无观察者/无动作语义）。"""
    import poker_decision_ai.vision2.pixel.realclient.coinpoker_bet as B
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert "ActionObserver" not in src
    assert "observe(" not in src
    assert "strategy" not in src.lower()


# ----------------------------------------------------------------- regression

def test_1419_evidence_limited_preserved():
    """1419 类 Bet 语义不升级：无 bet 字形 → 恒 UNKNOWN no_bet_scale_glyphs。"""
    rec1419 = BetRecognizer(window="1419")
    assert rec1419.rois == BET_ROIS_1419
    assert not rec1419._has_bet_glyphs
    res = rec1419.recognize(np.zeros((1066, 1419, 4), dtype=np.uint8))
    for seat, r in res.items():
        assert r["status"] == "UNKNOWN"
        assert r["bet"] is None
        assert r["reason"] == "no_bet_scale_glyphs"
