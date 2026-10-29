"""R14-EXE-3 — 参考建议模式（方案 Y）测试。

覆盖：advice_reference 通道契约（advice_type/legal_actions_status 恒携带）·
6-max 开局表查表（in-range raise / out-of-range fold）· HU chart 场景 ·
诚实 miss（BB 首入 / 6-max facing / 缺输入 / postflop 卡数矛盾）·
postflop spot 首入查表（首入假设标注）· frame_decision 接线（ADVICE_REFERENCE
+ judge_action_not_consumed）· 正常路径零回退（to_call 已知 → OK 不变；非法
动作 gate 不变）· production_decision_loop 映射（与 READY 严格可区分）·
runner.analyze ADVICE_REFERENCE 分支（judge final_decision 不混入）·
GUI 三态显示（app/preview 参考打法标注）· 真实帧端到端（corpus skipif）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.advice_reference import (  # noqa: E402
    ADVICE_REFERENCE_VERSION, ADVICE_TYPE_REFERENCE, ADVICE_WARNING_ZH,
    DECISION_STATUS_ADVICE_REFERENCE, LEGAL_ACTIONS_STATUS_UNKNOWN,
    advice_from_observation, reference_advice)
from poker_decision_ai.frame_decision import FrameDecisionResult  # noqa: E402

V3 = Path("data/vision2/coinpoker_v3")
EV = Path("data/vision2/coinpoker_evidence")
BLINDS = {"sb": 1000.0, "bb": 2000.0}
BB = 2000.0


def _ref(**kw):
    base = dict(street="preflop", hole_cards=["Ah", "Kd"], board=[], pot=4920.0,
                blinds=BLINDS, ante=320.0, hero_position="SB", n_seated=6)
    base.update(kw)
    return reference_advice(**base)


def _obs(*, to_call=None, street="preflop", hole_cards=("5c", "4s"),
         board=(), pot=4920.0, position="SB", n_seats=6, stack=40000.0):
    seats = [{"seat": i, "name": f"seat{i}", "stack": stack, "in_hand": True}
             for i in range(1, n_seats + 1)]
    return {
        "source": "v3f_snapshot", "street": street, "hero_seat": 5,
        "button_seat": 5, "hole_cards": list(hole_cards), "board": list(board),
        "seats": seats, "pot": pot, "blinds": dict(BLINDS), "ante": 320.0,
        "to_call": to_call, "confidence": 0.9,
        "field_confidences": {"street": 1.0, "pot": 1.0, "hero_cards": 1.0,
                              "blinds": 1.0, "to_call": 0.0 if to_call is None
                              else 1.0},
        "warnings": [], "extra": {"v3f_hero_position": position},
    }


class _FakeReport:
    def __init__(self, action="check"):
        self._action = action

    def to_dict(self):
        return {"final_decision": {"action": self._action, "amount": None},
                "agents_ran": [], "degraded": [], "skipped": []}


def _patch_bridge(obs):
    """frame_decision._run_impl 的 bridge 入口（函数内 import → 模块属性补丁）。"""
    from poker_decision_ai.vision2.pixel.realclient import production_bridge
    got = {"status": "OK",
           "request": {"kind": "cash", "hero_id": "Hero",
                       "vision_observation": obs},
           "bridge": {"status": "OK", "observation": obs,
                      "unknown_fields": [], "warnings": [],
                      "field_confidences": obs["field_confidences"]},
           "street_resolution": {}}
    return mock.patch.object(production_bridge, "frame_to_decision_request",
                             return_value=got)


def _patch_decide(action="check"):
    from poker_decision_ai import realtime as rt

    class _FakeRR:
        def __init__(self, runtime):
            pass

        async def decide(self, request):
            return _FakeReport(action)

    return mock.patch.object(rt, "RealtimeRunner", _FakeRR)


def _run_frame(obs, action="check"):
    from poker_decision_ai.frame_decision import frame_to_decision
    with _patch_bridge(obs), _patch_decide(action):
        return frame_to_decision(object(), np.zeros((4, 4, 4), np.uint8),
                                 4, 4)


# ══ 1. advice_reference 通道（契约 + 场景 + miss）══════════════════════
def test_contract_fields_always_present():
    """命中/miss 均恒携带 advice_type=reference + legal_actions_status=unknown。"""
    for r in (_ref(), _ref(hero_position="BB")):
        assert r["advice_type"] == "reference"
        assert r["legal_actions_status"] == "unknown"
        assert r["version"] == ADVICE_REFERENCE_VERSION
        assert ADVICE_WARNING_ZH in r["warnings"]


def test_sixmax_first_in_in_range_open_raise():
    r = _ref()                                  # SB + AKo + 盲注+ante 池
    assert r["available"] is True
    assert r["action"] == "raise"
    assert r["size_bb"] == 2.5
    assert r["source"] == "ranges_open_table"
    assert r["scenario"] == "open.SB"
    assert r["hand_class"] == "AKo"
    assert r["basis"]["scenario_family"] == "first_in"


def test_sixmax_first_in_out_of_range_fold():
    r = _ref(hole_cards=["7d", "2c"], hero_position="MP")   # 72o
    assert r["available"] is True
    assert r["action"] == "fold"
    assert r["size_bb"] is None
    assert r["scenario"] == "open.MP"


def test_bb_first_in_honest_miss():
    r = _ref(hero_position="BB")
    assert r["available"] is False
    assert "bb_first_in" in r["reason"]
    assert r["action"] is None


def test_sixmax_facing_honest_miss():
    """pot 高于首入带 + 6-max → 声明表不含 facing → 诚实 miss（不套 HU）。"""
    r = _ref(pot=8920.0, hero_position="CO")    # 4.46bb > 2.46+0.5
    assert r["available"] is False
    assert "facing_6max" in r["reason"]
    assert r["basis"]["scenario_family"] == "facing"


def test_hu_chart_scenarios():
    hu_blinds = {"sb": 500.0, "bb": 1000.0}
    r = _ref(pot=1500.0, blinds=hu_blinds, ante=None, n_seated=2,
             hero_position="SB", hole_cards=["Ah", "Kh"])
    assert r["available"] and r["action"] == "raise"
    assert r["size_bb"] == 2.5 and r["scenario"] == "sb_open"
    r2 = _ref(pot=3500.0, blinds=hu_blinds, ante=None, n_seated=2,
              hero_position="BB", hole_cards=["Ah", "Kh"])
    assert r2["available"] and r2["action"] == "raise"
    assert r2["size_bb"] == 11.0 and r2["scenario"] == "bb_vs_sb_open"


@pytest.mark.parametrize("kw,reason_part", [
    (dict(street=None), "street"),
    (dict(hole_cards=None), "hole_cards"),
    (dict(pot=None), "pot_or_blinds"),
    (dict(blinds=None), "pot_or_blinds"),
    (dict(n_seated=None), "n_seated"),
])
def test_missing_inputs_honest_miss(kw, reason_part):
    r = _ref(**kw)
    assert r["available"] is False
    assert reason_part in r["reason"]
    assert r["action"] is None


def test_postflop_spot_first_in_and_cardinality_gate():
    r = reference_advice(street="flop", hole_cards=["Ah", "Kh"],
                         board=["Td", "9d", "6h"], pot=3500.0,
                         blinds={"sb": 500.0, "bb": 1000.0}, ante=None,
                         hero_position="BTN", n_seated=2)
    if r["available"]:                          # spot 库在位时
        assert r["action"] in ("check", "bet", "allin")
        assert r["source"] == "solver_spot_table_first_in"
        assert "首入假设" in (r["basis"].get("assumption_first_in") or "")
    else:                                       # 库缺失 → 诚实 miss
        assert "spot_miss" in r["reason"]
    r2 = reference_advice(street="turn", hole_cards=["Ah", "Kh"],
                          board=["Td", "9d", "6h"], pot=3500.0,
                          blinds={"sb": 500.0, "bb": 1000.0}, ante=None,
                          hero_position="BTN", n_seated=2)
    assert r2["available"] is False
    assert "board_cardinality" in r2["reason"]


def test_advice_from_observation_reads_bridge_fields():
    r = advice_from_observation(_obs())
    assert r["advice_type"] == "reference"
    assert r["hand_class"] == "54o"
    # 54s 在 open.SB（75s+ 含 54s），54o 不在任何声明 open 范围 → fold
    assert r["available"] is True and r["action"] == "fold"


# ══ 2. frame_decision 接线 ═════════════════════════════════════════════
def test_frame_decision_advice_reference_wiring():
    res = _run_frame(_obs(to_call=None), action="check")
    assert res.status == "ADVICE_REFERENCE"
    prov = res.provenance or {}
    assert prov["advice_type"] == "reference"
    assert prov["legal_actions_status"] == "unknown"
    assert prov["legal_actions"] is None            # 显式 UNKNOWN（不推导）
    assert prov["judge_action_not_consumed"] == "check"   # judge 不作决策消费
    adv = prov["advice_reference"]
    assert adv["available"] is True
    assert prov["action"] == adv["action"]
    assert res.reason == "advice_reference:legal_actions_unknown"
    assert any("不是针对当前下注的决策" in w for w in res.warnings)


def test_frame_decision_advice_miss_stays_unavailable():
    res = _run_frame(_obs(to_call=None, position="BB"), action="check")
    assert res.status == "DECISION_UNAVAILABLE"
    assert "advice_reference_miss" in (res.reason or "")
    prov = res.provenance or {}
    assert prov["advice_type"] == "reference"       # 语义状态仍如实可审计
    assert prov["advice_reference"]["available"] is False


def test_frame_decision_normal_path_to_call_known():
    """to_call 已知（legal_actions 可派生）→ 正常决策零回退（自动升级红线）。"""
    res = _run_frame(_obs(to_call=2000.0), action="call")
    assert res.status == "OK"
    prov = res.provenance or {}
    assert prov["action_validated"] is True
    assert prov["legal_actions"] == ["fold", "call", "raise"]
    assert "advice_type" not in prov                # 正常路径无参考建议字段
    assert res.warnings == []


def test_frame_decision_postflop_to_call_known_normal_path():
    """postflop + to_call 已知（=0 过牌局）→ 正常 OK（语料无此形态真实帧，
    合成 observation 验证接线——分支条件只看 legal is None、与街无关）。"""
    res = _run_frame(_obs(to_call=0.0, street="flop",
                          board=("Td", "9d", "6h"), pot=8920.0,
                          position="BTN", hole_cards=("Ah", "Kh")),
                     action="check")
    assert res.status == "OK"
    prov = res.provenance or {}
    assert prov["action_validated"] is True
    assert prov["legal_actions"] == ["check", "bet"]   # stack(40000)>pot → 无 allin
    assert "advice_type" not in prov


def test_frame_decision_illegal_action_still_gated():
    """legal_actions 已知 + 越界动作 → judge 硬门不变。"""
    res = _run_frame(_obs(to_call=2000.0), action="bet_the_moon")
    assert res.status == "DECISION_UNAVAILABLE"
    assert (res.reason or "").startswith("judge_safety_gate:")


# ══ 3. production_decision_loop 映射（与 READY 严格可区分）═════════════
def _loop_with(monkeypatch, result: FrameDecisionResult):
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    loop = ProductionDecisionLoop(object(), throttle_s=0.0)
    monkeypatch.setattr("poker_decision_ai.production_decision_loop."
                        "frame_to_decision", lambda *a, **k: result)
    return loop


def test_loop_tick_advice_reference_mapping(monkeypatch):
    adv = {"available": True, "action": "raise", "size_bb": 2.5,
           "source": "ranges_open_table", "scenario": "open.SB",
           "hand_class": "AKo", "advice_type": "reference",
           "legal_actions_status": "unknown", "warnings": [ADVICE_WARNING_ZH]}
    res = FrameDecisionResult(
        status="ADVICE_REFERENCE", reason="advice_reference:legal_actions_unknown",
        provenance={"advice_reference": adv, "advice_type": "reference",
                    "legal_actions_status": "unknown", "legal_actions": None,
                    "judge_action_not_consumed": "check",
                    "hand_id": "h1", "street": "preflop"})
    loop = _loop_with(monkeypatch, res)
    out = loop.tick(1, np.zeros((4, 4, 4), np.uint8), 4, 4)
    assert out["decision_status"] == "ADVICE_REFERENCE"
    assert out["advice_type"] == "reference"
    assert out["legal_actions_status"] == "unknown"
    assert out["action"] == "raise"
    assert out["action_validated"] is False         # 恒 False（红线）
    assert out["strategy_mode"] == "BASELINE"       # 绝不 REAL/RULES
    assert "不是针对当前下注的决策" in (out["warning"] or "")
    assert out["solver_provenance"]["solver_status"] == "advice_reference"
    # judge 动作不得泄漏为决策
    assert out["provenance"]["judge_action_not_consumed"] == "check"


def test_loop_ready_vs_advice_strictly_distinguishable(monkeypatch):
    """红线：ADVICE_REFERENCE 与正常 READY 在契约字段上严格可区分。"""
    adv_prov = {"advice_reference": {"available": True, "action": "raise",
                                     "warnings": [ADVICE_WARNING_ZH]},
                "advice_type": "reference",
                "legal_actions_status": "unknown",
                "judge_action_not_consumed": "check",
                "street": "preflop"}
    loop = _loop_with(monkeypatch, FrameDecisionResult(
        status="ADVICE_REFERENCE", provenance=dict(adv_prov)))
    advice_out = loop.tick(2, np.zeros((4, 4, 4), np.uint8), 4, 4)
    loop2 = _loop_with(monkeypatch, FrameDecisionResult(
        status="OK", provenance={"action_validated": True,
                                 "legal_actions": ["check", "bet"],
                                 "street": "flop"}))
    ready_out = loop2.tick(3, np.zeros((4, 4, 4), np.uint8), 4, 4)
    assert advice_out["decision_status"] == DECISION_STATUS_ADVICE_REFERENCE
    assert ready_out["decision_status"] == "READY"
    assert advice_out["decision_status"] != ready_out["decision_status"]
    assert ready_out["advice_type"] is None
    assert ready_out["legal_actions_status"] == "known"
    assert ready_out["action_validated"] is True
    assert advice_out["action_validated"] is False


# ══ 4. runner.analyze + GUI 显示 ════════════════════════════════════════
def test_runner_analyze_advice_reference_branch(monkeypatch):
    """runner.analyze：ADVICE_REFERENCE → action 只读 advice_reference 通道；
    judge final_decision 不混入 raw（红线）。"""
    from cash_game_ui.runner import AnalysisResult, CashGameRunner
    from poker_decision_ai import frame_decision as fd_mod
    from poker_decision_ai.vision2.pixel.realclient import production_bridge
    obs = _obs(to_call=None)
    adv = {"available": True, "action": "raise", "size_bb": 2.5,
           "confidence": 0.6, "source": "ranges_open_table",
           "scenario": "open.SB", "hand_class": "AKo",
           "advice_type": "reference", "legal_actions_status": "unknown",
           "warnings": [ADVICE_WARNING_ZH], "version": ADVICE_REFERENCE_VERSION}
    got = {"status": "OK",
           "request": {"kind": "cash", "hero_id": "Hero",
                       "vision_observation": obs},
           "bridge": {"status": "OK", "observation": obs,
                      "unknown_fields": ["to_call"], "warnings": [],
                      "field_confidences": obs["field_confidences"]},
           "street_resolution": {}}
    res = FrameDecisionResult(
        status="ADVICE_REFERENCE",
        reason="advice_reference:legal_actions_unknown",
        provenance={"advice_reference": adv, "advice_type": "reference",
                    "legal_actions_status": "unknown",
                    "judge_action_not_consumed": "check",
                    "unknown_allowed_fields": ["to_call"],
                    "unknown_allowed_rationale": "x", "action_history": {
                        "status": "UNKNOWN"}})
    monkeypatch.setattr(production_bridge, "frame_to_decision_request",
                        lambda *a, **k: got)
    monkeypatch.setattr(fd_mod, "frame_to_decision",
                        lambda *a, **k: res)
    runner = CashGameRunner()
    monkeypatch.setattr(runner, "_get_runtime", lambda: object())
    out = runner.analyze(b"\x00" * 16, 2, 2).to_dict()
    assert out["status"] == "ADVICE_REFERENCE"
    assert out["decision_action"] == "raise"
    assert out["advice_type"] == "reference"
    assert out["legal_actions_status"] == "unknown"
    assert "不是针对当前下注的决策" in (out["reasoning"] or "")
    assert out["raw"].get("advice_reference", {}).get("action") == "raise"
    assert "final_decision" not in out["raw"]       # judge 输出不得混入
    assert any(a["agent"] == "Advice" and a["status"] == "REFERENCE"
               for a in out["agents"])


def test_gui_app_advice_reference_display():
    """app._show_recognize：参考打法标注（未识别动作选项）+ 诚实警告。"""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    try:
        from cash_game_ui.app import CashGameApp
        app = CashGameApp.__new__(CashGameApp)
        app.root = root
        app.btn_rec = tk.Button(root)
        app.btn = tk.Button(root)
        app.status_lbl = tk.Label(root)
        app.action_lbl = tk.Label(root)
        app.meta_lbl = tk.Label(root)
        app.agents_txt = tk.Text(root)
        app.detail_panel = None
        d = {"status": "ADVICE_REFERENCE", "decision_action": "raise",
             "recommended_size": 2.5, "confidence": 0.6,
             "advice_type": "reference", "legal_actions_status": "unknown",
             "reasoning": ADVICE_WARNING_ZH,
             "reason": "advice_reference:legal_actions_unknown",
             "blocking_fields": [], "unknown_fields": ["to_call"],
             "vision": {"blinds": {"sb": 1000, "bb": 2000}, "pot": 4920},
             "raw": {"advice_reference": {"source": "ranges_open_table",
                                          "scenario": "open.SB"}},
             "agents": []}
        app._show_recognize(d)
        assert "参考打法" in app.status_lbl.cget("text")
        assert "未识别动作选项" in app.status_lbl.cget("text")
        assert "RAISE" in app.action_lbl.cget("text")
        assert "参考" in app.action_lbl.cget("text")
        assert "不是针对当前下注的决策" in app.meta_lbl.cget("text")
    finally:
        root.destroy()


def test_gui_preview_advice_reference_display():
    """preview_gui._show：参考打法标注 + runlog。"""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    try:
        from cash_game_ui.preview_gui import PreviewGUI
        gui = PreviewGUI.__new__(PreviewGUI)
        gui.root = root
        gui.action_lbl = tk.Label(root)
        gui.meta_lbl = tk.Label(root)
        gui._log_q = __import__("queue").Queue()
        gui._res_q = __import__("queue").Queue()
        gui._pending = 0
        gui._executed = 0
        gui._refresh_status = lambda: None
        gui._stages = {}
        d = {"status": "ADVICE_REFERENCE", "decision_action": "fold",
             "recommended_size": None, "confidence": 0.6,
             "reason": "advice_reference:legal_actions_unknown",
             "vision": {"blinds": {"sb": 1000, "bb": 2000}, "pot": 4920,
                        "street": "preflop"},
             "raw": {"advice_reference": {"source": "ranges_open_table",
                                          "scenario": "open.MP",
                                          "hand_class": "72o"}}}
        gui._show(1, d, "12:00:00")
        assert "FOLD" in gui.action_lbl.cget("text")
        assert "参考" in gui.action_lbl.cget("text")
    finally:
        root.destroy()


# ══ 5. 真实帧端到端（corpus 在位时）═════════════════════════════════════
def _decode(p: Path):
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def test_e2e_preflop_frame_advice_reference():
    """真实 preflop 帧（bridge OK + to_call UNKNOWN）→ ADVICE_REFERENCE。"""
    f = V3 / "POT_CHANGED_114_1791199840586_full.png"
    if not f.is_file():
        pytest.skip("corpus frame missing")
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    from poker_decision_ai.frame_decision import frame_to_decision
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
        frame_to_decision_request)
    runtime = build_runtime()
    arr = _decode(f)
    got = frame_to_decision_request(arr, arr.shape[1], arr.shape[0])
    obs = got["bridge"].get("observation") or {}
    if got["status"] != "OK" or obs.get("to_call") is not None:
        pytest.skip("frame not in legal-unknown state")
    res = frame_to_decision(runtime, arr, arr.shape[1], arr.shape[0])
    assert res.status == "ADVICE_REFERENCE"
    prov = res.provenance or {}
    adv = prov["advice_reference"]
    assert adv["available"] is True
    assert adv["action"] in ("raise", "fold", "call", "check")
    assert adv["advice_type"] == "reference"
    assert prov["legal_actions_status"] == "unknown"
    assert any("不是针对当前下注的决策" in w for w in adv["warnings"])


def test_e2e_tocall_known_frame_normal_path():
    """to_call 已知帧（8bf8，tc=2000）→ 正常 OK 决策（零回退）。"""
    f = EV / "8bf82f134651db0add626be0304a2ff1.png"
    if not f.is_file():
        pytest.skip("corpus frame missing")
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    from poker_decision_ai.frame_decision import frame_to_decision
    runtime = build_runtime()
    arr = _decode(f)
    res = frame_to_decision(runtime, arr, arr.shape[1], arr.shape[0])
    assert res.status == "OK"
    prov = res.provenance or {}
    assert prov["action_validated"] is True
    assert "advice_type" not in prov
