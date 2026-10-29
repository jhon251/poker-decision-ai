"""Vision V3F-P8 — Stack Runtime Wiring 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StackObserver)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _frame(fname: str):
    w, h, rgba = decode_png(Path(
        f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
    return rgba, w, h


def _observe3(obs, rgba, w, h, seat):
    outs = []
    for _ in range(3):
        r = obs.observe(rgba, w, h)
        outs.append((r["seats"][seat]["raw_text"],
                     r["seats"][seat]["stack"],
                     r["seats"][seat]["status"]))
    return outs


# ── 5 稳定源（×3 一致）───────────────────────────────────────
def test_stable_953909():
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    obs = StackObserver()
    outs = _observe3(obs, rgba, w, h, "S1")
    assert len({(o[0], o[1]) for o in outs}) == 1 and outs[0][2] == \
        "RECOGNIZED", outs
    assert outs[0][1] == 953909


def test_stable_318871():
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    obs = StackObserver()
    outs = _observe3(obs, rgba, w, h, "S4")
    assert outs[0][2] == "RECOGNIZED" and outs[0][1] == 318871, outs


def test_718902_observer_honest():
    """718,902（37 BL）经 observer 的通用 S4 region → 诚实 UNKNOWN
    （该源需帧特定校准窗（thr2/y985）——与 434,504（thr4/y970）不同——
    单一通用 region 无法兼顾——诚实不伪造）。"""
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    obs = StackObserver()
    outs = _observe3(obs, rgba, w, h, "S4")
    assert outs[0][2] == "UNKNOWN"
    assert outs[0][1] is None          # 不伪造值/不转 0/前值


def test_stable_388080():
    rgba, w, h = _frame("STATUS_UI_CHANGED_246_1787488634077_full.png")
    obs = StackObserver()
    outs = _observe3(obs, rgba, w, h, "S1")
    assert outs[0][2] == "RECOGNIZED" and outs[0][1] == 388080, outs


def test_stable_434504_no_134504():
    rgba, w, h = _frame("SCENE_CHANGED_60_1787489266339_full.png")
    obs = StackObserver()
    outs = _observe3(obs, rgba, w, h, "S4")
    assert outs[0][2] == "RECOGNIZED" and outs[0][1] == 434504, outs
    assert "134,504" not in {o[0] for o in outs}


# ── 语义 / 红线 ──────────────────────────────────────────────
def test_unknown_isolation_no_inherit():
    """UNKNOWN 不继承前值；S2/S6 恒 UNKNOWN。"""
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    obs = StackObserver()
    obs.observe(rgba, w, h)                       # 有值帧
    blank = np.zeros((h, w, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    out = obs.observe(blank.tobytes(), w, h)
    for seat, v in out["seats"].items():
        assert v["stack"] is None
        assert v["status"] == "UNKNOWN"
    assert out["seats"]["S2"]["reason"] == "cropped"
    assert out["seats"]["S6"]["reason"] == "cropped"


def test_no_fixture_no_cache():
    obs1, obs2 = StackObserver(), StackObserver()
    assert obs1.summary()["last_seats"] is None
    assert obs2.summary()["last_seats"] is None


def test_no_hardcoded_value():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/seat_position.py"
    ).read_text(encoding="utf-8")
    for bad in ("953909", "318871", "718902", "388080", "434504", "134504"):
        assert bad not in src, bad


def test_no_pot_bet_reference():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/seat_position.py"
    ).read_text(encoding="utf-8")
    assert "BetRecognizer" not in src
    assert "BetObserver" not in src
    # StackObserver 类块不引用 pot（PotObserver 合法引用 pot 识别器——分离）
    import re
    cls = src[src.index("class StackObserver"):]
    assert "PotRecognizer" not in cls


def test_recognizer_frozen():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_stack.py"
    ).read_text(encoding="utf-8")
    assert "MIN_SCORE = 0.7" in src
    assert "MIN_MARGIN = 0.08" in src


def test_independent_of_status():
    """Stack 与 Player Status 独立（同帧并行互不影响）。"""
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        StatusObserver)
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    so = StackObserver()
    st = StatusObserver()
    s_res = so.observe(rgba, w, h)
    t_res = st.observe(rgba, w, h)
    assert s_res["seats"]["S1"]["stack"] == 953909
    assert t_res["status"] in ("RECOGNIZED", "NOT_OBSERVED", "UNKNOWN")


def test_capture_wiring_complete():
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    assert "StackObserver" in src
    assert "stack_observer.observe(sub.tobytes(), cw, chh)" in src
    assert "stack_runtime" in src
    for b in ("cooldown_s=0.5", "FrameStream(fps=args.fps)",
              "engine.process_frame", "--seconds", "--fps"):
        assert b in src, b


def test_summary_reflects_recognition():
    rgba, w, h = _frame("SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    obs = StackObserver()
    obs.observe(rgba, w, h)
    s = obs.summary()
    assert s["frames"] == 1
    assert s["recognized"] == 1
    assert s["last_seats"]["S1"]["stack"] == 953909
