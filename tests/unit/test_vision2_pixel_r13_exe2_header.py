"""R13-EXE-2 — header 锚定滑动识别 + 双模板集自动选择 测试。

覆盖：v3 字重帧锚定识别 · 粗体 GT 帧回归不回退（whole-word 路径零变化）·
双模板集并存自动选择（font_variant provenance）· 变体缺失诚实 UNKNOWN ·
malformed stake 诚实 UNKNOWN · 阈值红线（0.90 门不动）· 一键 runner 链。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    BettingHeaderRecognizer1876, CHAR_SCORE_MIN, HEADER_ASSET_DIR,
    HEADER_ASSET_DIR_V3B, STAKE_NCC_MIN, STAKE_ROI_EXT)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

V3 = Path("data/vision2/coinpoker_v3")
EV = Path("data/vision2/coinpoker_evidence")


def _frame(p: Path):
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# ── 1. v3 字重帧 → 锚定识别（新路径）────────────────────────────────
def test_v3_frame_anchored_recognized():
    rec = BettingHeaderRecognizer1876()
    out = rec.recognize(_frame(
        V3 / "POT_CHANGED_114_1791199840586_full.png"))
    assert out["status"] == "RECOGNIZED"
    assert (out["sb"], out["bb"], out["ante"]) == (1000, 2000, 320)
    assert out["reason"] == "header_stake_anchored_parse"
    assert out["font_variant"] == "v3b"
    assert out["confidence"] >= CHAR_SCORE_MIN


def test_v3_frames_stable_across_corpus():
    rec = BettingHeaderRecognizer1876()
    for fid in ["BUTTON_CHANGED_178_1791197763010_full.png",
                "SETTLED_BOARD_CHECK_115_1791201027720_full.png",
                "TABLE_CHANGED_9_1791201710751_full.png"]:
        out = rec.recognize(_frame(V3 / fid))
        assert out["status"] == "RECOGNIZED", fid
        assert (out["sb"], out["bb"], out["ante"]) == (1000, 2000, 320), fid


# ── 2. 粗体 GT 帧回归不回退（whole-word 0.90 路径零变化）────────────
def test_bold_gt_regression_unchanged():
    rec = BettingHeaderRecognizer1876()
    for fid in ["BUTTON_CHANGED_167_1787992682749_full.png",
                "8bf82f134651db0add626be0304a2ff1.png",
                "POT_CHANGED_110_1787992526057_full.png",
                "POT_CHANGED_102_1787992505794_full.png"]:
        out = rec.recognize(_frame(EV / fid))
        assert out["status"] == "RECOGNIZED", fid
        assert (out["sb"], out["bb"], out["ante"]) == (1000, 2000, 320), fid
        assert out["reason"] == "header_stake_template_match", fid
        assert out["font_variant"] == "v1876_whole", fid
        assert out["confidence"] >= STAKE_NCC_MIN, fid


# ── 3. 双模板集并存自动选择 ─────────────────────────────────────────
def test_variant_selection_requires_both_dirs():
    # 仅 v1876（旧模板集）：v3 字重帧诚实 UNKNOWN（无模板可匹配）
    rec_old = BettingHeaderRecognizer1876(variant_dirs={})
    out = rec_old.recognize(_frame(
        V3 / "POT_CHANGED_114_1791199840586_full.png"))
    assert out["status"] == "UNKNOWN"
    # 双模板集：同一帧 → v3b 锚定识别（自动选择生效）
    rec_both = BettingHeaderRecognizer1876()
    out2 = rec_both.recognize(_frame(
        V3 / "POT_CHANGED_114_1791199840586_full.png"))
    assert out2["status"] == "RECOGNIZED"
    assert out2["font_variant"] == "v3b"


def test_v1876_variant_assets_unchanged():
    rec = BettingHeaderRecognizer1876()
    v1876 = sorted(rec._variants["v1876"]["assets"].keys())
    assert v1876 == ["K", "digit_1", "digit_2", "digit_3", "paren_close",
                     "paren_open", "slash"]
    # v3b 增补 digit{0,4,7,9}（NOT_COVERED: 5/6/8 → 语料未出现）
    v3b = set(rec._variants["v3b"]["assets"].keys())
    assert {"digit_0", "digit_4", "digit_7", "digit_9"} <= v3b
    assert not ({"digit_5", "digit_6", "digit_8"} & v3b)


# ── 4. malformed 诚实降级（识别器只加不降）──────────────────────────
def test_malformed_stake_region_unknown():
    rec = BettingHeaderRecognizer1876()
    fr = _frame(V3 / "POT_CHANGED_114_1791199840586_full.png").copy()
    fr[0:26, 183:216] = (6, 35, 27, 255)      # 抹掉 ante 区 `(320)`
    out = rec.recognize(fr)
    assert out["status"] == "UNKNOWN"
    assert out["sb"] is None and out["bb"] is None and out["ante"] is None


def test_no_header_unknown_unchanged():
    rec = BettingHeaderRecognizer1876()
    blank = np.zeros((1400, 1876, 4), dtype=np.uint8)
    blank[..., :3] = (6, 35, 27)
    blank[..., 3] = 255
    out = rec.recognize(blank)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] in ("no_header_text", "roi_out_of_frame")


# ── 5. 阈值红线（0.90 门不降）───────────────────────────────────────
def test_thresholds_frozen():
    assert STAKE_NCC_MIN == 0.90
    assert CHAR_SCORE_MIN == 0.70
    assert STAKE_ROI_EXT == (90, 0, 224, 26)
    assert HEADER_ASSET_DIR.endswith("v1876betting_header")
    assert HEADER_ASSET_DIR_V3B.endswith("v3b_header")


# ── 6. 一键 runner 链（不截屏——直接喂帧；三态结构化）───────────────
def test_runner_analyze_ok_frame_smoke():
    """8bf8（READY 帧）→ runner.analyze 出建议；置信/尺寸/provenance
    从 final_decision 如实透传。"""
    from cash_game_ui.runner import CashGameRunner
    w, h, rgba = decode_png(
        (EV / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
    runner = CashGameRunner()
    res = runner.analyze(rgba, w, h)
    d = res.to_dict()
    assert d["status"] in ("OK", "DECISION_UNAVAILABLE", "UNAVAILABLE")
    if d["status"] == "OK":
        assert d["decision_action"]
        assert d["vision"].get("blinds") == {"sb": 1000, "bb": 2000}
        assert d["strategy_mode"] in ("REAL", "RULES", "BASELINE")


def test_runner_analyze_incomplete_frame_honest():
    """部分帧（无完整牌局证据）→ 三态如实：缺失字段清单，不编造。"""
    from cash_game_ui.runner import CashGameRunner
    blank = np.zeros((1400, 1876, 4), dtype=np.uint8)
    blank[..., :3] = (6, 35, 27)
    blank[..., 3] = 255
    runner = CashGameRunner()
    res = runner.analyze(blank.tobytes(), 1876, 1400)
    d = res.to_dict()
    assert d["status"] in ("DECISION_UNAVAILABLE", "UNAVAILABLE")
    assert d["blocking_fields"] or d["reason"]


# ── 7. GUI 三态显示冒烟（不起 mainloop；EXE Preview GUI 同函数）─────
def test_preview_gui_three_states():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    try:
        from cash_game_ui.preview_gui import PreviewGUI
        from cash_game_ui.runner import CashGameRunner as _CGR
        gui = PreviewGUI(root, frame_path=None,
                         runner=object.__new__(_CGR))
        # 态 a：OK → 建议动作+尺寸
        gui._show(1, {"status": "OK", "decision_action": "call",
                      "recommended_size": None, "confidence": 0.95,
                      "strategy_mode": "BASELINE",
                      "vision": {"blinds": {"sb": 1000, "bb": 2000},
                                 "pot": 8920}}, "12:00:00")
        assert "CALL" in gui.action_lbl.cget("text")
        # 态 b：MANDATORY 缺失 → 识别不到 + 缺失清单
        gui._show(2, {"status": "DECISION_UNAVAILABLE",
                      "reason": "blocking_fields:hero_cards,blinds",
                      "decision_action": None,
                      "blocking_fields": ["hero_cards", "blinds"],
                      "unknown_fields": ["to_call"],
                      "vision": {}}, "12:00:01")
        assert gui.action_lbl.cget("text") == "识别不到"
        assert "缺手牌" in gui.meta_lbl.cget("text")
        assert "盲注未识别" in gui.meta_lbl.cget("text")
        # 态 c：部分识别（有 blocking → 标题"识别不到"+已识别/缺失并列；
        # 无 blocking 无动作 → "信息不足"+已识别/缺失清单）
        gui._show(3, {"status": "DECISION_UNAVAILABLE",
                      "reason": "blocking_fields:hero_cards",
                      "decision_action": None,
                      "blocking_fields": ["hero_cards"],
                      "unknown_fields": [],
                      "vision": {"pot": 8920, "blinds": {"sb": 1000,
                                                         "bb": 2000}}},
                  "12:00:02")
        assert gui.action_lbl.cget("text") == "识别不到"
        assert "pot=8920" in gui.meta_lbl.cget("text")
        gui._show(4, {"status": "DECISION_UNAVAILABLE",
                      "reason": "judge_safety_gate:legal_actions_unknown",
                      "decision_action": None,
                      "blocking_fields": [],
                      "unknown_fields": ["to_call", "current_player"],
                      "vision": {"blinds": {"sb": 1000, "bb": 2000}}},
                  "12:00:03")
        assert "信息不足" in gui.action_lbl.cget("text")
        assert "to_call" in gui.meta_lbl.cget("text")
    finally:
        root.destroy()
