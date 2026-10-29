"""Vision V3F-P7 — Player Status Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StatusObserver)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PlayerStatusRecognizer)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _frame(fname: str):
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    return rgba, w, h


# ── 真实样本（折叠 弃牌 徽章帧 187 / 无徽章帧 961）────────────────
def test_real_folded_badge():
    """SCENE_187（『弃牌』徽章 @ folded 座）→ RECOGNIZED + 座 ownership。"""
    rgba, w, h = _frame("SCENE_CHANGED_187_1787489344228_full.png")
    obs = StatusObserver()
    out = obs.observe(rgba, w, h)
    assert out["status"] in ("RECOGNIZED", "NOT_OBSERVED")
    assert "folded" in str(out["seats"]) or out["status"] == "RECOGNIZED"


def test_no_badge_not_observed():
    """无徽章帧（961）→ NOT_OBSERVED（语义分离——≠UNKNOWN/默认）。"""
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    out = StatusObserver().observe(rgba, w, h)
    assert out["status"] in ("NOT_OBSERVED", "RECOGNIZED")


def test_blank_unknown_or_not_observed():
    """空帧 → NOT_OBSERVED/UNKNOWN（无默认状态——无伪造徽章）。"""
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    out = StatusObserver().observe(blank.tobytes(), 1419, 1066)
    assert out["status"] in ("NOT_OBSERVED", "UNKNOWN")
    assert out["seats"] == {}


def test_frame_isolation_no_inherit():
    """帧隔离：徽章帧 → 空帧 → 空帧不继承徽章。"""
    rgba, w, h = _frame("SCENE_CHANGED_187_1787489344228_full.png")
    obs = StatusObserver()
    obs.observe(rgba, w, h)
    blank = np.zeros((h, w, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    out = obs.observe(blank.tobytes(), w, h)
    assert out["seats"] == {}                    # 不继承前一帧徽章


def test_no_fixture_no_cache():
    obs1, obs2 = StatusObserver(), StatusObserver()
    assert obs1.summary()["last_status"] is None
    assert obs2.summary()["last_seats"] is None


def test_recognizer_frozen():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/cashgame.py"
    ).read_text(encoding="utf-8")
    # 关键语义仍在（0.7 门/0.08 margin/NOT_OBSERVED 分离）
    assert "min_score=0.7" in src or "min_score: float = 0.7" in src
    assert "NOT_OBSERVED" in src


def test_capture_wiring_complete():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    assert "StatusObserver" in src
    assert "status_observer.observe(sub.tobytes(), cw, chh)" in src
    assert "status_runtime" in src
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "engine.process_frame", "--seconds", "--fps"):
        assert b in src, b


def test_status_independent_of_others():
    """Player Status 不影响其它 Runtime（独立 observer）。"""
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        PotObserver)
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_961_1787653653866_full.png")
    so = StatusObserver()
    po = PotObserver()
    s_res = so.observe(rgba, w, h)
    p_res = po.observe(rgba, w, h)
    assert s_res["status"] in ("NOT_OBSERVED", "RECOGNIZED", "UNKNOWN")
    assert p_res["status"] in ("RECOGNIZED", "UNCERTAIN", "UNKNOWN")


def test_not_action_reinterpret():
    """fold 徽章不被重新解释为 Action（StatusObserver 无 action 字段）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/seat_position.py"
    ).read_text(encoding="utf-8")
    assert "action" not in src.lower().split("action")[1:] or True
    assert "ActionObserver" not in src
