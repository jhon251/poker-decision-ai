"""Vision V3F-P4 — Hero Cards Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    HeroCardsObserver)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _frame(fname: str):
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    return rgba, w, h


def _blank():
    w, h = 1419, 1066
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    frame[..., :3] = (40, 90, 60)
    frame[..., 3] = 255
    return frame.tobytes(), w, h


# ── A：真实 Qs/9d 帧 ─────────────────────────────────────────────
def test_case_a_real_qs_9d():
    rgba, w, h = _frame("SCENE_CHANGED_188_1787489345211_full.png")
    out = HeroCardsObserver().observe(rgba, w, h)
    assert out["status"] == "RECOGNIZED"
    assert out["cards"] == ["Qs", "9d"]


# ── B：空帧 → UNKNOWN（≠ 伪造手牌）──────────────────────────────
def test_case_b_blank_unknown():
    fb, w, h = _blank()
    out = HeroCardsObserver().observe(fb, w, h)
    assert out["status"] == "UNCERTAIN"
    assert out["cards"] is None
    assert out["value"] is None               # 未识别 ≠ 无手牌（None 非 []）


# ── C：帧隔离 ────────────────────────────────────────────────────
def test_case_c_frame_isolation():
    obs = HeroCardsObserver()
    a, w, h = _frame("SCENE_CHANGED_188_1787489345211_full.png")
    b, w2, h2 = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    obs.observe(a, w, h)
    out = obs.observe(b, w2, h2)
    # 961 帧无 hero 卡区域内容 → 诚实（不继承 Qs/9d）
    assert out["cards"] is None or out["cards"] == []  # 视为未识别/空


# ── D：UNKNOWN 不继承 ───────────────────────────────────────────
def test_case_d_unknown_no_inherit():
    obs = HeroCardsObserver()
    a, w, h = _frame("SCENE_CHANGED_188_1787489345211_full.png")
    obs.observe(a, w, h)
    fb, w2, h2 = _blank()
    out = obs.observe(fb, w2, h2)
    assert out["cards"] is None
    assert obs.summary()["last_cards"] is None


# ── E：无 fixture 泄漏 ───────────────────────────────────────────
def test_case_e_no_fixture():
    obs1, obs2 = HeroCardsObserver(), HeroCardsObserver()
    assert obs1.summary()["last_cards"] is None
    assert obs2.summary()["last_cards"] is None


# ── F：无 cache / previous-frame ─────────────────────────────────
def test_case_f_no_cache():
    obs = HeroCardsObserver()
    a, w, h = _frame("SCENE_CHANGED_188_1787489345211_full.png")
    obs.observe(a, w, h)
    assert obs.summary()["recognized"] == 1
    fb, w2, h2 = _blank()
    obs.observe(fb, w2, h2)
    assert obs.summary()["unknown_frames"] == 1    # 无 cache 回填


# ── G：capture 接线完整性 ───────────────────────────────────────
def test_case_g_wiring_complete():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("HeroCardsObserver", "hero_cards_observer.observe(sub.tobytes(),"
              " cw, chh)", "hero_cards_runtime"):
        assert b in src, b


# ── H：capture 原有行为未改变 ────────────────────────────────────
def test_case_h_capture_untouched():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "stream.accept", "engine.process_frame", "--seconds",
              "--fps", "coinpoker_auto_capture.json"):
        assert b in src, b


# ── I：Recognizer 源码冻结检查 ───────────────────────────────────
def test_case_i_recognizer_frozen():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_hero.py"
    ).read_text(encoding="utf-8")
    # 关键语义未变（rank/suit 双门 + status OK/UNCERTAIN）
    assert '"status": "OK"' in src or "status: \"OK\"" in src or \
        '"status": "OK"' in src.replace("'", '"')
    assert "UNCERTAIN" in src
    assert "min(" in src                       # 置信取 min（rank/suit）


# ── J：Observer 独立性（与 Position/Pot/Board/Street 同帧并行）────
def test_case_j_observer_independent():
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        PositionObserver, PotObserver, BoardObserver, StreetObserver)
    rgba, w, h = _frame("SCENE_CHANGED_188_1787489345211_full.png")
    hc = HeroCardsObserver()
    hc_res = hc.observe(rgba, w, h)
    pt = PotObserver()
    pt_res = pt.observe(rgba, w, h)
    assert hc_res["cards"] == ["Qs", "9d"]
    # Pot 独立（该帧 pot 为其自身结果——不受 hero 影响）
    assert pt_res is not None
    # 各自 summary 独立
    assert hc.summary()["frames"] == pt.summary()["frames"] == 1
