"""V3F-P30-4 — Decision / Solver 第一阶段测试。

覆盖：Action Contract（derive/validate/illegal/negative）· 合法动作空间
（fold/call/check/bet/raise/allin）· UNKNOWN 输入不推导 · Judgy 安全门
（非法 action 拒绝）· baseline strategy（solver mock 不复刻 GTO 声明）·
真实 8bf8 回归 · frame 状态保持（f01/pc110/f03）· positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.decision_contract import (  # noqa: E402
    derive_legal_actions, validate_action, is_negative, LEGAL_ACTIONS)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── Action Contract：派生公开例子 ─────────────────────────────────────
def test_derive_no_bet_to_call_zero():
    assert derive_legal_actions(0.0, 100000.0, 8920.0) == ["check", "bet"]


def test_derive_check_bet_allin():
    assert derive_legal_actions(0.0, 5000.0, 8920.0) == \
        ["check", "bet", "allin"]


def test_derive_fold_call_raise():
    assert derive_legal_actions(2000.0, 197680.0, 8920.0) == \
        ["fold", "call", "raise"]


def test_derive_fold_call_short_stack():
    assert derive_legal_actions(2000.0, 1500.0, 8920.0) == ["fold", "call"]


def test_derive_unknown_inputs_none():
    assert derive_legal_actions(None, 1000.0, 8920.0) is None
    assert derive_legal_actions(2000.0, None, 8920.0) is None
    assert derive_legal_actions(2000.0, 1000.0, None) is None
    assert derive_legal_actions(2000.0, 0.0, 8920.0) is None   # 非正栈
    assert derive_legal_actions(-1.0, 1000.0, 8920.0) is None  # 负 to_call


# ── 合法动作全集 ─────────────────────────────────────────────────────
def test_legal_actions_enum():
    assert set(LEGAL_ACTIONS) == {"fold", "check", "call", "bet", "raise",
                                  "allin"}


# ── validate ─────────────────────────────────────────────────────────
def test_validate_ok():
    ok, why = validate_action("call", ["fold", "call", "raise"])
    assert ok and why == ""


def test_validate_illegal():
    ok, why = validate_action("bet", ["fold", "call", "raise"])
    assert not ok and why == "illegal_action:bet"


def test_validate_unknown_moveset():
    ok, why = validate_action("call", None)
    assert not ok and why == "legal_actions_unknown"


def test_validate_unknown_action():
    ok, why = validate_action("surrender", ["fold", "call", "raise"])
    assert not ok and why == "unknown_action:surrender"


# ── 负金额检查 ────────────────────────────────────────────────────────
def test_is_negative():
    assert is_negative(-5.0) is True
    assert is_negative(0.0) is False
    assert is_negative(5.0) is False
    assert is_negative(None) is False          # UNKNOWN 非负（不判非法）


# ── JUDGE 安全门（frame_decision 层）────────────────────────────────
def test_judge_gate_action_valid():
    """8bf8（legal=[fold,call,raise] + call 输出）→ 通过。"""
    import asyncio
    from poker_decision_ai.frame_decision import frame_to_decision_async
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    runtime = build_runtime(load_config())
    res = asyncio.run(frame_to_decision_async(
        runtime, np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4),
        w, h))
    assert res.status == "OK"
    fd = res.report.get("final_decision") or {}
    legal = (res.provenance or {}).get("legal_actions")
    assert fd.get("action") in (legal or [])


def test_judge_gate_provenance():
    import asyncio
    from poker_decision_ai.frame_decision import frame_to_decision_async
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    runtime = build_runtime(load_config())
    res = asyncio.run(frame_to_decision_async(
        runtime, np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4),
        w, h))
    prov = res.provenance or {}
    assert prov.get("action_validated") is True
    assert prov.get("legal_actions") == ["fold", "call", "raise"]
    assert prov.get("decision_engine") == "realtime"


# ── BASELINE 标记（solver 不冒充 GTO）────────────────────────────────
def test_solver_baseline_marker():
    """SolverAgent 输出 must 标 is_approximate（mock=baseline 非 GTO）。"""
    src = Path(
        "poker_decision_ai/agents/solver/agent.py").read_text(encoding="utf-8")
    assert "is_approximate=True" in src or "is_approximate=bool" in src


# ── 真实帧回归（D4-7）──────────────────────────────────────────────
def test_real_8bf8_readiness():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "OK"


def test_real_f03_unchanged():
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"


def test_real_f01_unchanged():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝（原
    # DECISION_UNAVAILABLE 建立在伪造 street=preflop 之上）
    assert res["status"] == "UNAVAILABLE"


def test_real_pc110_unchanged():
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "DECISION_UNAVAILABLE"
    assert "stack" in res["bridge"]["blocking_fields"]


# ── 1419 regression ──────────────────────────────────────────────────
def test_1419_regression():
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
        ToCallObserver)
    out = ToCallObserver().observe(rgba, w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
