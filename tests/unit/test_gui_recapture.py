# -*- coding: utf-8 -*-
"""P30-9-J：hero 牌缺失时的只读重捕获策略（GUI 路径必须生效）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cash_game_ui.runner as R                                            # noqa: E402
import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb  # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


def _patch(monkeypatch, blocking):
    state = {"captures": 0, "bridges": 0}

    def fake_capture(self):
        state["captures"] += 1
        return b"x" * 16, 100, 100, (0, 0, 100, 100)

    def fake_bridge(rgba, w, h, **kw):
        state["bridges"] += 1
        return {"status": "UNAVAILABLE", "reason": "blocked",
                "bridge": {"blocking_fields": list(blocking),
                            "observation": {}, "unknown_fields": []},
                "request": {}}

    monkeypatch.setattr(CashGameRunner, "capture_frame", fake_capture)
    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    return state


def test_recapture_runs_when_gui_passes_precaptured_frame(monkeypatch):
    """GUI 自截图后传入帧（allow_recapture=True）⇒ 重捕获必须执行。"""
    st = _patch(monkeypatch, ["blinds", "hero_cards"])
    run = CashGameRunner()
    res = run.analyze(b"frame", 100, 100, allow_recapture=True)
    # P30-9-N：重试预算改为 2（每轮桥 ~1.2s；避免一次点击等 15s）
    # 传帧时首发不截图 ⇒ 捕获全部来自重捕获循环（2 次）；bridge 1 首发 + 2 重跑
    assert st["captures"] == 2
    assert st["bridges"] == 3
    assert run._last_recapture == 2
    assert res.status == "UNAVAILABLE"
    assert res.frame_provenance.get("recapture_attempts") == 2


def test_recapture_not_run_for_fixture_frame(monkeypatch):
    st = _patch(monkeypatch, ["hero_cards"])
    run = CashGameRunner()
    run.analyze(b"frame", 100, 100, allow_recapture=False)
    assert st["captures"] == 0            # 固定帧 ⇒ 不读屏
    assert st["bridges"] == 1
    assert run._last_recapture == 0


def test_recapture_triggers_for_all_hero_blocker_keys(monkeypatch):
    for key in ("hero", "hero_cards", "hero_position"):
        st = _patch(monkeypatch, ["blinds", key])
        run = CashGameRunner()
        run.analyze(b"frame", 100, 100, allow_recapture=True)
        assert run._last_recapture == 2, key      # P30-9-N 预算 2


def test_no_recapture_when_blocking_is_not_hero_related(monkeypatch):
    st = _patch(monkeypatch, ["blinds"])
    run = CashGameRunner()
    run.analyze(b"frame", 100, 100, allow_recapture=True)
    assert st["captures"] == 0 and run._last_recapture == 0


def test_recapture_stops_early_when_hero_cards_recovered(monkeypatch):
    state = {"captures": 0, "bridges": 0}

    def fake_capture(self):
        state["captures"] += 1
        return b"x" * 16, 100, 100, (0, 0, 100, 100)

    def fake_bridge(rgba, w, h, **kw):
        state["bridges"] += 1
        blocking = ["hero_cards"] if state["bridges"] == 1 else ["blinds"]
        return {"status": "UNAVAILABLE", "reason": "blocked",
                "bridge": {"blocking_fields": blocking,
                            "observation": {"seats": [{"seat": 1}]},
                            "unknown_fields": []},
                "request": {}}

    monkeypatch.setattr(CashGameRunner, "capture_frame", fake_capture)
    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    run = CashGameRunner()
    run.analyze(b"frame", 100, 100, allow_recapture=True)
    assert state["captures"] == 1         # hero 牌恢复 ⇒ 立即停止重捕获
    assert run._last_recapture == 1
