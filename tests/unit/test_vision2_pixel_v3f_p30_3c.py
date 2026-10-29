"""V3F-P30-3-C — Decision Readiness & UNAVAILABLE Gap Closure 测试。

覆盖：readiness matrix · real 1876 frame readiness · bet/status UNKNOWN
(UNKNOWN_ALLOWED) · current_bet/current_player/action_history UNKNOWN 不阻断
· 多 UNKNOWN 安全 · no UNKNOWN→0/[]/默认 · RealtimeRunner gate ·
production CLI regression · 1419/1848 · positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation, frame_to_decision_request,
    DECISION_GAPS)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    rgba, w, h = None, None, None
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


def _full_snapshot():
    return {
        "street": {"last_street": "PREFLOP"},
        "pot": {"last_pot": 8920, "last_status": "RECOGNIZED"},
        "hero_cards": {"last_cards": ["5c", "3c"],
                       "last_status": "RECOGNIZED"},
        "board": {"last_cards": []},
        "bet": {"last_seats": {}},
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
    }


def _full_position():
    return {"occupied_seats": {f"S{i}": {"status": "OCCUPIED"}
                               for i in range(1, 7)},
            "dealer_seat": "S4", "hero_seat": "S6", "table_topology":
            "6-max"}


# ── 1. Readiness contract：MANDATORY 阻断 vs UNKNOWN_ALLOWED ────────
def test_all_ready_ok_with_unknown_fields_reported():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    assert br["blocking_fields"] == []
    # current_bet/current_player/action_history 不在 blocking（允许未知）
    assert "current_bet" not in br["blocking_fields"]
    assert "current_player" not in br["blocking_fields"]
    assert "action_history" not in br["blocking_fields"]


def test_status_no_evidence_unknown_allowed():
    s = _full_snapshot()
    s["status"] = {"last_seats": {f"S{i}": {"status": "UNKNOWN",
                                            "reason": "no_status_evidence"}
                                  for i in range(1, 7)}}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "OK"                 # UNKNOWN_ALLOWED
    for seat in br["observation"]["seats"]:
        assert seat["in_hand"] is True          # 保守


def test_status_conflict_blocks():
    s = _full_snapshot()
    s["status"] = {"last_seats": {"S1": {"status": "UNKNOWN",
                                         "reason": "conflicting_evidence"}}}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert "status" in br["blocking_fields"]


def test_bet_partial_unknown_ok():
    s = _full_snapshot()
    s["bet"] = {"last_seats": {"S1": {"bet": 1000, "status": "RECOGNIZED"},
                               "S5": {"bet": None, "status": "UNKNOWN"}}}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "OK"                 # UNKNOWN_ALLOWED
    assert br["observation"]["extra"]["v3f_bets"]["S5"] is None


# ── 2. current_bet / pot UNKNOWN 语义 ────────────────────────────────
def test_current_bet_unknown_no_zero_no_block():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert "current_bet" not in br["observation"]       # 不创建
    assert "current_bet" in DECISION_GAPS               # gap 保持
    assert "current_bet" not in br["blocking_fields"]   # 允许未知


def test_pot_unknown_blocks():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert "pot" in br["blocking_fields"]
    assert br["observation"]["pot"] is None


def test_to_call_unknown_blocks():
    # R13-EXE-2 语义修正（所有者确认）：to_call UNKNOWN 不再阻断
    # （UNKNOWN_ALLOWED——未轮到 hero 时 None 属正常；其余 MANDATORY 不变）。
    s = _full_snapshot()
    s["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "OK"
    assert br["observation"]["to_call"] is None
    assert "to_call" not in br["blocking_fields"]


def test_hero_cards_unknown_blocks():
    s = _full_snapshot()
    s["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert br["observation"]["hole_cards"] is None


# ── 3. 多 UNKNOWN：fabricated state 检查 ────────────────────────────
def test_multi_unknown_no_fabrication():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    s["status"] = {"last_seats": {f"S{i}": {"status": "UNKNOWN",
                                            "reason": "no_status_evidence"}
                                  for i in range(1, 7)}}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    obs = br["observation"]
    assert obs["pot"] is None                    # 非 0
    assert obs["street"] == "preflop"            # 已知值不受影响
    # 无默认 hero/current_player 伪造
    assert obs["extra"]["v3f_current_player"] is None
    assert obs["extra"]["v3f_current_bet_status"] == "UNKNOWN"


# ── 4. 真实帧 readiness ──────────────────────────────────────────────
def test_real_1876_frame_ok():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "OK"
    obs = res["bridge"]["observation"]
    assert obs["to_call"] == 2000.0                # 跟注 2,000（已识别）
    assert obs["blinds"] == {"sb": 1000.0, "bb": 2000.0}
    assert obs["hero_seat"] == 5


def test_real_frame_unknown_field_lists():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    br = res["bridge"]
    assert "status" not in br["blocking_fields"]    # no_status_evidence 允许
    assert "bet" not in br["blocking_fields"]       # 部分 UNKNOWN 允许
    # status 值不用于决策（extra 标注 UNKNOWN_ALLOWED）
    assert br["observation"]["extra"]["v3f_status"] is not None


# ── 5. no UNKNOWN→0 / [] / 默认 ─────────────────────────────────────
def test_no_unknown_to_zero_or_default():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    s["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["pot"] is None
    assert br["observation"]["hole_cards"] is None
    assert br["observation"]["extra"]["v3f_current_player"] is None


# ── 6. RealtimeRunner gate（frame_to_decision_request 门控）──────────
def test_frame_gate_unknown_no_request():
    """f03（occupied UNKNOWN）→ 结构性拒绝 → request=None（不进入 decide）。"""
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    assert res["request"] is None


# ── 7. production CLI regression（frame → decide 可用性）─────────────
def test_cli_frame_reaches_decision():
    """8bf8 真实帧 → CLI --frame → FrameDecisionResult OK（决策产生）。
    直接调用 async adapter（无 subprocess——快）。"""
    import asyncio
    from poker_decision_ai.frame_decision import frame_to_decision_async
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    runtime = build_runtime(load_config())
    res = asyncio.run(frame_to_decision_async(runtime, arr, w, h))
    assert res.status == "OK"
    assert res.report is not None
    fd = res.report.get("final_decision") or {}
    assert fd.get("action") in ("fold", "check", "call", "bet", "raise",
                                "allin")


# ── 8. 1419/1848 regression ─────────────────────────────────────────
def test_1419_regression():
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
        ToCallObserver)
    out = ToCallObserver().observe(rgba, w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


# ── 9. positions_for 零修改 ─────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
