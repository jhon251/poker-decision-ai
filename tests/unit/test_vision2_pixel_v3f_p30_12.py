"""V3F-P30-12 — Remaining Production Blockers Closure Audit 测试。

覆盖：无真实 Solver · executable 不存在 · timeout/malformed/failure ·
BASELINE/REAL provenance · action_history UNKNOWN · EVENT_STREAM 正常 ·
VISUAL_FACT 不得产生行为 · side_pot/current_player UNKNOWN · Hero
current_player 可见性（面板证据——weak signal 记录）· capture→decision
loop（--decide 接线审计）· idempotency · readiness gate · 4 真实帧回归 ·
positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.solver_contract import freeze_mode  # noqa: E402
from poker_decision_ai.strategy_provider import (  # noqa: E402
    classify_solver_result, strategy_failure)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-7. Solver ─────────────────────────────────────────────────────
def test_no_real_solver_available():
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    p = PostflopSolverCLIProvider()
    assert p.is_configured() is False           # 本机无 executable（诚实）


def test_executable_absent():
    assert strategy_failure("postflop", "executable not found") \
        .strategy_mode == "UNKNOWN"


def test_timeout_unknown():
    assert strategy_failure("postflop", "timeout").strategy_mode == "UNKNOWN"


def test_malformed_unknown():
    assert strategy_failure("postflop", "malformed").strategy_mode == "UNKNOWN"


def test_failure_unknown():
    assert strategy_failure("postflop", "crash").strategy_mode == "UNKNOWN"


def test_baseline_provenance():
    r = classify_solver_result("mock", True, gto_action="call")
    assert r.strategy_mode == "BASELINE"
    assert r.to_dict()["provenance"]["is_approximate"] is True


def test_real_provenance_only_real_solver():
    """仅当真实 executable 成功返回 → REAL（模式冻结）。"""
    r = freeze_mode("postflop", False, gto_action="call")
    assert r.mode == "REAL"
    # 但本机 executable 不存在 → 当前实际为 false（非 REAL 环境）
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    assert PostflopSolverCLIProvider().is_configured() is False


# ── 8-10. 行为事件源 ───────────────────────────────────────────────
def test_action_history_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["bridge"]["observation"]["extra"][
        "v3f_action_history_status"] == "UNKNOWN"


def test_event_stream_normal():
    from poker_decision_ai.action_event_source import from_hand_history
    src = from_hand_history([{"street": "preflop", "player": "BTN",
                              "action": "raise", "amount": 30}])
    assert src.status == "RECOGNIZED"
    assert src.source == "HAND_HISTORY"


def test_visual_fact_no_behavior():
    from poker_decision_ai.event_stream_consumer import EventStreamConsumer
    c = EventStreamConsumer()
    ev = c.ingest_visual_fact("BET_CHANGED", 20, {"from": 0, "to": 100})
    assert ev.action == ""                        # 无行为归因
    assert ev.is_behavior() is False


# ── 11-13. side_pot / current_player ───────────────────────────────
def test_side_pot_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert "side_pot" not in obs                  # 不创建/不推导


def test_current_player_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["bridge"]["observation"]["extra"][
        "v3f_current_player"] is None


def test_hero_current_player_visual_panel_evidence():
    """动作面板 = hero 当前可行动的弱信号（审计：非强保证——
    面板可见 ⇒ hero 有动作选项；但面板也出现在 hero 折叠/等待时——
    仅记录为弱信号，不建 detector）。"""
    # 审计断言：生产 bridge 不消费面板 → current_player 保持 UNKNOWN
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["bridge"]["observation"]["extra"][
        "v3f_current_player_status"] == "UNKNOWN"


# ── 14-16. capture → decision loop ─────────────────────────────────
def test_capture_decision_loop_wiring_audit():
    """--decide 接线脚本存在且只输出 DecisionResult（不自动点击）。"""
    src = Path("scripts/phase_v2s_auto_capture.py").read_text(
        encoding="utf-8")
    assert "--decide" in src
    assert "ProductionDecisionLoop" in src
    assert "decision_log" in src
    # 无自动操作调用（代码体不含 click/send_bet 等——排除禁止声明行）
    body = "\n".join(
        ln for ln in src.splitlines()
        if "no auto-click" not in ln and "不自动点击" not in ln)
    for word in ("click", "send_bet", "submit", "mouse_event",
                 "keyboard", "pyautogui"):
        assert word.lower() not in body.lower()


def test_idempotency():
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r1 = loop.tick(400, rgba, w, h)
    r2 = loop.tick(400, rgba, w, h)
    assert r1["decision_id"] == r2["decision_id"]
    assert len(loop.decisions) == 1


def test_readiness_gate():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝（结构性）
    assert res["status"] == "UNAVAILABLE"
    assert res["request"] is None


# ── 17. 4 真实帧回归 ─────────────────────────────────────────────────
def test_four_real_frames_regression():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # V-1 FM-1 语义更新：f01 帧不再静默 preflop → 结构性 UNAVAILABLE
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "UNAVAILABLE"
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "DECISION_UNAVAILABLE"
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "UNAVAILABLE"


# ── 18. positions_for 回归 ──────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
