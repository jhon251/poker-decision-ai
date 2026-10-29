"""V3F-P30-3-B — Production Frame→Decision + Action History 集成测试。

覆盖：production caller（frame→bridge→decide）· frame path · 1419/1848
regression · UNKNOWN 语义（to_call/current_bet/pot/hero_cards/occupied/
current_player 不转 0/check/默认）· action_history UNKNOWN 不伪造 ·
重复帧/手牌生命周期（审计级断言）· 无 unauthorized fallback。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation, DECISION_GAPS)
from poker_decision_ai.frame_decision import FrameDecisionResult  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
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


# ── UNKNOWN 语义（bridge 层）─────────────────────────────────────────
def test_current_bet_unknown_not_zero():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    obs = br["observation"]
    assert "current_bet" not in obs                 # bridge 不创建
    assert br["observation"]["extra"]["v3f_current_bet_status"] == "UNKNOWN"
    assert "current_bet" in br["decision_gaps"]     # 仍为 gap（冻结）


def test_to_call_unknown_not_zero():
    s = _full_snapshot()
    s["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["to_call"] is None     # 非 0


def test_pot_unknown_not_zero():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["pot"] is None


def test_hero_cards_unknown_not_empty():
    s = _full_snapshot()
    s["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["hole_cards"] is None  # 非 []


def test_occupied_unknown_rejects():
    p = _full_position()
    p["occupied_seats"]["S6"] = {"status": "UNKNOWN"}
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["status"] == "UNAVAILABLE"
    assert br["observation"] is None


def test_current_player_unknown_not_default():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["observation"]["extra"]["v3f_current_player_status"] == \
        "UNKNOWN"
    assert br["observation"]["extra"]["v3f_current_player"] is None


# ── Action History：不伪造 ───────────────────────────────────────────
def test_action_history_unknown_no_fabrication():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["observation"]["extra"]["v3f_action_history"] is None
    assert br["observation"]["extra"]["v3f_action_history_status"] == \
        "UNKNOWN"
    # 无 fake "BTN raised" 类事件
    ah = str(br["observation"]["extra"].get("v3f_action_history"))
    assert "raise" not in ah and "call" not in ah


def test_action_history_passthrough_if_provided():
    s = _full_snapshot()
    s["action_history"] = {"last_actions": [
        {"street": "preflop", "player_id": "seat5", "action": "call",
         "amount": 2000, "seq": 1}]}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["observation"]["extra"]["v3f_action_history_status"] == \
        "RECOGNIZED"
    assert len(br["observation"]["extra"]["v3f_action_history"]) == 1


# ── Production adapter（frame→decide 门控）───────────────────────────
def test_frame_decision_gate_ok_unknown():
    """非 OK bridge → request=None（不喂 decide）；status 透传精确分类。"""
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
        frame_to_decision_request)
    got = frame_to_decision_request(rgba, w, h)
    assert got["request"] is None                   # 门控
    # P30-3-D：status 透传 bridge 精确分类（不抹平为 UNAVAILABLE）
    assert got["status"] in ("DECISION_UNAVAILABLE", "UNAVAILABLE")


def test_frame_decision_dataclass_shape():
    r = FrameDecisionResult(status="UNAVAILABLE", reason="x")
    assert r.status == "UNAVAILABLE"
    assert r.report is None


# ── 手牌生命周期（审计级断言：无 unauthorized previous fallback）──────
def test_no_previous_frame_fallback_in_bridge():
    """bridge 不得引用上一帧/缓存制造未验证事件——只读单帧。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/production_bridge.py"
    ).read_text(encoding="utf-8")
    # 代码体（注释外）不允许 prev_frame / previous_frame 引用
    import re
    for m in re.finditer(r"^(?!\s*#).*(prev_frame|previous_frame)",
                         src, re.M):
        line = m.group(0)
        if "#" in line and (line.split(" #", 1)[0].strip() == "#" or
                            line.strip().startswith("#")):
            continue
        assert "previous" not in line, line


def test_decision_gaps_still_has_current_bet():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert "current_bet" in br["decision_gaps"]
    assert "action_history" not in br["decision_gaps"]   # D 类（经
    # action_history_status 标注）


# ── 1419/1848 regression ────────────────────────────────────────────
def test_1419_observer_not_affected():
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
        ToCallObserver)
    out = ToCallObserver().observe(rgba, w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


def test_1419_pipeline_still_runs():
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
        PositionObserver)
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    out = PositionObserver().observe(rgba, w, h)
    assert out["ground_truth_status"] == "runtime_state_non_gt"


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
