# -*- coding: utf-8 -*-
"""P30-9-O-2：一致性贯穿所有盲注来源 + 阻止误读值回写。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb   # noqa: E402
from cash_game_ui.runner import CashGameRunner                              # noqa: E402


OBS_BAD = {"street": "flop", "pot": 0.41, "hero_seat": 5,
           "seats": [{"seat": i} for i in range(1, 7)],
           "extra": {"v3f_bets": {"S2": None, "S6": 0.05}}}
OBS_OK = {"street": "flop", "pot": 21.0, "hero_seat": 5,
          "seats": [{"seat": i} for i in range(1, 7)],
          "extra": {"v3f_bets": {"S2": 5.0, "S6": 10.0}}}


def test_vision_summary_exposes_consistency(monkeypatch):
    run = CashGameRunner()
    assert run._blinds_consistent({"blinds": {"sb": 0.02, "bb": 0.05},
                                    "pot": 0.12,
                                    "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05,
                                                            "S5": 0.05}}}) is True
    assert run._blinds_consistent({"blinds": {"sb": 2.0, "bb": 5.0},
                                    "pot": 0.41,
                                    "extra": {"v3f_bets": {"S6": 0.05}}}) is False
    # 无证据 ⇒ 不算反证
    assert run._blinds_consistent({}) is True


def _run_blocking(monkeypatch, obs, override=None):
    """驱动 analyze 到"仅 blinds 阻断"，返回 runner（供检查 rejected/结果）。"""
    state = {"n": 0}

    def fake_bridge(rgba, w, h, **kw):
        state["n"] += 1
        return {"status": "UNAVAILABLE", "reason": "blocking_fields:blinds",
                "bridge": {"blocking_fields": ["blinds"], "observation": obs,
                            "unknown_fields": []},
                "request": {"vision_observation": obs}}

    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    run = CashGameRunner()
    run.analyze(b"f", 100, 100, allow_recapture=False, blinds_override=override)
    return run


def test_inconsistent_manual_override_is_rejected(monkeypatch):
    """兜底框里的 2/5（误读后被回写）在 pot=0.41 的帧上必须被拒绝。"""
    run = _run_blocking(monkeypatch, OBS_BAD, override={"sb": 2.0, "bb": 5.0})
    rej = getattr(run, "_last_blinds_rejected", [])
    assert any(r["source"] == "user_setting" for r in rej), rej
    assert all(r["reason"] in ("pot_less_than_half_blind_round",
                                "observed_amounts_far_below_blinds") for r in rej)


def test_consistent_manual_override_is_accepted(monkeypatch):
    run = _run_blocking(monkeypatch, OBS_OK, override={"sb": 2.0, "bb": 5.0})
    rej = getattr(run, "_last_blinds_rejected", [])
    assert rej == [], rej
