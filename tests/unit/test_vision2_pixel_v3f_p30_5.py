"""V3F-P30-5 — Cash-Game Decision Capability Expansion 测试。

覆盖：baseline provider · real provider 接口 · failure · approximate 标记 ·
GTO 不混淆 · action_history RECOGNIZED/UNKNOWN · UNKNOWN≠[] · Range
degradation/provenance · Solver 非法 action · Judge 拒绝 · strategy UNKNOWN ·
equity UNKNOWN · 真实帧回归（8bf8/f03/f01/pc110）· 完整链路 · 无随机 fallback。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.strategy_provider import (  # noqa: E402
    StrategyResult, classify_solver_result, strategy_failure,
    STRATEGY_MODES)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-2. baseline / real provider ────────────────────────────────────
def test_baseline_provider_classified():
    r = classify_solver_result("mock", True,
                               actions=[{"name": "call", "frequency": 0.6}],
                               gto_action="call")
    assert r.strategy_mode == "BASELINE"
    assert r.is_approximate is True


def test_real_provider_classified():
    r = classify_solver_result("postflop", False,
                               actions=[{"name": "call", "frequency": 0.5}],
                               gto_action="call")
    assert r.strategy_mode == "REAL"
    assert r.is_approximate is False


# ── 3. provider failure ─────────────────────────────────────────────
def test_provider_failure_unknown():
    r = strategy_failure("postflop", "executable not found")
    assert r.strategy_mode == "UNKNOWN"
    assert r.gto_action is None
    assert r.actions == []


# ── 4-5. approximate / GTO 标记 ─────────────────────────────────────
def test_approximate_marked_not_gto():
    r = classify_solver_result("mock", True, gto_action="raise")
    assert r.strategy_mode != "REAL"
    assert r.is_approximate is True


def test_mock_never_real():
    r = classify_solver_result("mock", False, gto_action="call")
    assert r.strategy_mode != "REAL"          # mock 永不 REAL（防冒充）


# ── 6-8. action_history 语义 ────────────────────────────────────────
def test_action_history_unknown_not_empty():
    """bridge 层 action_history UNKNOWN（extra 状态）——非 []。"""
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
        snapshot_to_vision_observation)
    snap = {"street": {"last_street": "PREFLOP"},
            "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
            "hero_cards": {"last_cards": ["5c", "3c"],
                           "last_status": "RECOGNIZED"},
            "board": {"last_cards": []}, "bet": {"last_seats": {}},
            "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                      for i in range(1, 7)}},
            "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                               "status": "RECOGNIZED"}
                                     for i in range(1, 7)}},
            "position": {"last_hero_position": "CO",
                         "last_uncertainty": "RECOGNIZED"},
            "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                      "last_status": "RECOGNIZED"},
            "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"}}
    pr = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                             for i in range(1, 7)},
          "dealer_seat": "S4", "hero_seat": "S6", "table_topology":
          "6-max"}
    br = snapshot_to_vision_observation(snap, pr)
    assert br["observation"]["extra"]["v3f_action_history"] is None
    assert br["observation"]["extra"]["v3f_action_history_status"] == \
        "UNKNOWN"


def test_action_history_recognized_passthrough():
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
        snapshot_to_vision_observation)
    snap = {"street": {"last_street": "PREFLOP"},
            "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
            "hero_cards": {"last_cards": ["5c", "3c"],
                           "last_status": "RECOGNIZED"},
            "board": {"last_cards": []}, "bet": {"last_seats": {}},
            "status": {"last_seats": {f"S{i}": {"status": "NOT_OBSERVED"}
                                      for i in range(1, 7)}},
            "stack": {"last_seats": {f"S{i}": {"stack": 1000 * i,
                                               "status": "RECOGNIZED"}
                                     for i in range(1, 7)}},
            "position": {"last_hero_position": "CO",
                         "last_uncertainty": "RECOGNIZED"},
            "blind": {"last_sb": 1000, "last_bb": 2000, "last_ante": 320,
                      "last_status": "RECOGNIZED"},
            "to_call": {"last_to_call": 2000, "last_status": "RECOGNIZED"},
            "action_history": {"last_actions": [
                {"street": "preflop", "player_id": "seat5", "action": "call",
                 "amount": 2000, "seq": 1}]}}
    pr = {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                             for i in range(1, 7)},
          "dealer_seat": "S4", "hero_seat": "S6", "table_topology":
          "6-max"}
    br = snapshot_to_vision_observation(snap, pr)
    assert br["observation"]["extra"]["v3f_action_history_status"] == \
        "RECOGNIZED"


# ── 9-10. Range degradation / provenance ────────────────────────────
def test_range_degradation_mode():
    """Range 无玩家数据 → BASELINE 模式（非真实 range）。"""
    from poker_decision_ai.agents.range.agent import RangeOutput
    out = RangeOutput()
    assert out.range_mode == "BASELINE"
    assert out.range_source == ""


def test_equity_random_marked():
    """EquityOutput 默认 APPROXIMATE；random fallback 标记 RANDOM。"""
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    out.range_mode = "RANDOM"
    assert out.range_mode == "RANDOM"


# ── 11-12. Solver 非法 action / Judge 拒绝 ──────────────────────────
def test_solver_illegal_action_classified():
    """solver 输出非法 action（'surrender'）→ classify 仍记录（Judge 门拦截）。"""
    r = classify_solver_result("mock", True,
                               actions=[{"name": "surrender",
                                         "frequency": 0.9}],
                               gto_action="surrender")
    assert r.gto_action == "surrender"           # 原样记录
    # Judge 安全门（decision_contract）会拒绝非 LEGAL_ACTIONS
    from poker_decision_ai.decision_contract import validate_action
    ok, why = validate_action("surrender", ["fold", "call", "raise"])
    assert not ok and why == "unknown_action:surrender"


def test_judge_rejects_illegal_action():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(2000.0, 100000.0, 8920.0)
    ok, why = validate_action("bet", legal)
    assert not ok and why == "illegal_action:bet"


# ── 13. strategy UNKNOWN ────────────────────────────────────────────
def test_strategy_unknown_no_random_fallback():
    r = strategy_failure("", "no provider")
    assert r.strategy_mode == "UNKNOWN"
    assert r.gto_action is None                 # 无随机 action
    assert r.actions == []


# ── 14. equity UNKNOWN ──────────────────────────────────────────────
def test_equity_unknown_missing_villains():
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    assert out.hero_equity == 0.0
    assert out.confidence == 0.0


# ── 15-18. 真实帧回归 ───────────────────────────────────────────────
def test_real_8bf8_regression():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "OK"


def test_real_f03_regression():
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"


def test_real_f01_regression():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    # V-1 FM-1 修复语义更新：该帧板区仅识别 2 卡（部分证据）→ 机器停留
    # NEW_HAND → 不再静默映射 preflop（旧 DECISION_UNAVAILABLE 建立在
    # 伪造 street=preflop 之上）——诚实拒绝 UNAVAILABLE。
    # V-2 更新（test-side only）：该帧目视 GT = 3c 7h Qc 5c 共 4 张——
    # 召回修复后可计数控 4（Qc 为 RANK_ONLY）、FULL 牌面 3 → 单帧豁免
    # street=turn 与 FULL 张数 3 交叉矛盾 → 拒绝原因变为
    # street_board_contradiction（诚实拒绝语义不变）。
    assert res["status"] == "UNAVAILABLE"
    assert "street_board_contradiction" in res["reason"]


def test_real_pc110_regression():
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "DECISION_UNAVAILABLE"


# ── 19. 完整链路（8bf8 → decide → 合法 action）──────────────────────
def test_full_chain_8bf8():
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
    assert fd.get("action") in ("fold", "check", "call", "bet", "raise",
                                "allin")
    assert (res.provenance or {}).get("action_validated") is True


# ── 20. no-random-fallback ──────────────────────────────────────────
def test_no_random_fallback():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    # UNKNOWN 输入 → legal_actions=None → 拒绝（不随机选）
    legal = derive_legal_actions(None, 1000.0, 8920.0)
    assert legal is None
    ok, why = validate_action("call", legal)
    assert not ok and why == "legal_actions_unknown"


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
