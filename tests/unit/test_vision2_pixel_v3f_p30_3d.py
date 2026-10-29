"""V3F-P30-3-D — Decision Engine Boundary / Safety 审计测试。

覆盖：DecisionInput 字段矩阵 · 隐式补值（or 0/[] 扫描断言）·
Case A-J 决策安全性 · 决策 provenance · 真实帧回归（8bf8 READY /
f01 拒绝 / pc110 READY / f03 保持拒绝）· 无 previous 泄漏 ·
positions_for 零修改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    snapshot_to_vision_observation, frame_to_decision_request)
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


# ── Case A-J 决策安全性 ──────────────────────────────────────────────
def test_case_a_all_ready_ok():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"


def test_case_b_pot_unknown_rejects():
    s = _full_snapshot()
    s["pot"] = {"last_pot": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert "pot" in br["blocking_fields"]


def test_case_c_to_call_unknown_allowed():
    # R13-EXE-2 语义修正（所有者确认）：未轮到 hero 时 to_call=None 属
    # 正常——MANDATORY → UNKNOWN_ALLOWED（None/0 均合法，不阻断）。
    # 仅此一项放宽；blinds 等其它 MANDATORY 语义不变。
    s = _full_snapshot()
    s["to_call"] = {"last_to_call": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "OK"
    assert "to_call" not in br["blocking_fields"]
    assert "to_call" in br["unknown_fields"]
    assert br["observation"]["to_call"] is None      # 非 0 非默认


def test_case_d_hero_cards_unknown_rejects():
    s = _full_snapshot()
    s["hero_cards"] = {"last_cards": None, "last_status": "UNKNOWN"}
    br = snapshot_to_vision_observation(s, _full_position())
    assert br["status"] == "DECISION_UNAVAILABLE"
    assert "hero_cards" in br["blocking_fields"]


def test_case_e_hero_seat_unknown_rejects():
    p = _full_position()
    p["hero_seat"] = None
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["status"] == "DECISION_UNAVAILABLE"


def test_case_f_occupied_unknown_rejects():
    p = _full_position()
    p["occupied_seats"]["S2"] = {"status": "UNKNOWN"}
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["status"] == "UNAVAILABLE"      # 结构性（无法表示座存在性）


def test_case_g_current_bet_unknown_allowed_no_zero():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    assert "current_bet" not in br["observation"]      # 不转 0


def test_case_h_action_history_unknown_allowed_no_empty():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    assert br["observation"]["extra"]["v3f_action_history_status"] == \
        "UNKNOWN"                                       # 不伪造 [] 已识别


def test_case_i_current_player_unknown_allowed_no_infer():
    br = snapshot_to_vision_observation(_full_snapshot(), _full_position())
    assert br["status"] == "OK"
    assert br["observation"]["extra"]["v3f_current_player"] is None


def test_case_j_dealer_unknown_mandatory_rejects():
    p = _full_position()
    p["dealer_seat"] = None
    br = snapshot_to_vision_observation(_full_snapshot(), p)
    assert br["status"] == "DECISION_UNAVAILABLE"       # dealer MANDATORY
    assert br["observation"]["button_seat"] is None


# ── DecisionInput 字段矩阵（真实 8bf8：bridge→observation 完整）──────
def test_decision_input_matrix_8bf8():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    # MANDATORY 全部已知
    assert obs["pot"] == 8920.0
    assert obs["to_call"] == 2000.0
    assert obs["blinds"] == {"sb": 1000.0, "bb": 2000.0}
    assert obs["ante"] == 320.0
    assert obs["hero_seat"] == 5
    assert obs["button_seat"] == 6
    assert obs["street"] == "preflop"
    assert obs["hole_cards"] == ["5c", "3c"]
    assert obs["board"] == []
    # 无 current_bet 键（不转 0）
    assert "current_bet" not in obs


# ── D2 隐式补值扫描（决策链核心文件断言）──────────────────────────────
def test_implicit_defaults_audited():
    src = Path(
        "poker_decision_ai/agents/state/agent.py").read_text(encoding="utf-8")
    # 既有 V2 语义记录：pot/current_bet/to_call = float(x or 0.0)（记录不修）
    assert "float(obs.get(\"pot\") or 0.0)" in src
    assert "float(obs.get(\"current_bet\") or 0.0)" in src
    # 无默认 hero/dealer seat（无 hero_seat=1 / dealer_seat=1 硬编码）
    assert "hero_seat = 1" not in src
    assert "dealer_seat = 1" not in src


def test_no_previous_frame_leak():
    for fp in ["poker_decision_ai/frame_decision.py",
               "poker_decision_ai/vision2/pixel/realclient/production_bridge.py"]:
        src = Path(fp).read_text(encoding="utf-8")
        assert "prev_frame" not in src
        assert "previous_obs" not in src


# ── D4 provenance ────────────────────────────────────────────────────
def test_decision_provenance_fields():
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
    prov = res.provenance or {}
    assert prov.get("source_frame_geometry") == "1876x1400"
    assert prov.get("observation_source") == "v3f_snapshot"
    assert prov.get("decision_engine") == "realtime"
    assert "hero_cards" in prov.get("mandatory_fields_ready", [])
    assert prov.get("unknown_allowed_rationale")


# ── D5 真实帧回归 ────────────────────────────────────────────────────
def test_real_frame_8bf8_ready():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "OK"


def test_real_frame_pc110_ready():
    """pc110：S1 stack glyph_gate_failed → stack MANDATORY 阻断
    （诚实：非全 READY——非放宽）。其余字段（to_call=2000）正确。"""
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "DECISION_UNAVAILABLE"
    assert "stack" in res["bridge"]["blocking_fields"]
    obs = res["bridge"]["observation"]
    assert obs["to_call"] == 2000.0
    assert obs["blinds"] == {"sb": 1000.0, "bb": 2000.0}


def test_real_frame_f01_rejected():
    """f01（hero folded，无动作面板）→ 拒绝。（V-1 FM-1 语义更新：原
    DECISION_UNAVAILABLE 建立在静默 street=preflop 之上；该帧板区仅识别
    2 卡 → NEW_HAND 诚实拒绝 UNAVAILABLE——street_new_hand_unconfirmed。）"""
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    # V-2 更新（test-side only）：召回修复后该帧可计数控 4 / FULL 3 →
    # 拒绝原因 street_board_contradiction（同帧诚实拒绝，语义不变）。
    assert "street_board_contradiction" in res["reason"]


def test_real_frame_f03_still_rejected():
    """f03（S6 occupied UNKNOWN）→ 结构性拒绝——不得放宽 gate。"""
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"


# ── 1419/1848 regression ────────────────────────────────────────────
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
