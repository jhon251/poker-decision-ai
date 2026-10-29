# -*- coding: utf-8 -*-
"""P30-9-S：盲注证据不被其他阻断项挡住（用户 22:23 帧 + 真实标题）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb    # noqa: E402
from cash_game_ui.runner import CashGameRunner                              # noqa: E402
from poker_decision_ai import facing_evidence as fe                         # noqa: E402

TITLE = 'NLH 1524173 - \u00a50.05/\u00a50.1 Must Show'

OBS_MULTI = {"street": "preflop", "pot": 0.15, "hero_seat": None,
              "button_seat": None, "blinds": None,
              "seats": [{"seat": i} for i in range(1, 7)],
              "extra": {"v3f_bets": {"S1": None, "S2": None, "S6": None,
                                      "S5": 0.02}}}


def test_title_parsing_on_real_title():
    ev = fe.parse_stakes_from_title(TITLE)
    assert ev["source"] == "window_title"
    assert (ev["sb"], ev["bb"]) == (0.05, 0.1)


def _patch(monkeypatch, blocking, obs, title=TITLE):
    def fake_bridge(rgba, w, h, **kw):
        return {"status": "UNAVAILABLE", "reason": "blocking_fields",
                "bridge": {"blocking_fields": list(blocking),
                            "observation": obs, "unknown_fields": []},
                "request": {"vision_observation": obs}}

    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    run = CashGameRunner()
    monkeypatch.setattr(run, "_blinds_from_window_title",
                        lambda: fe.parse_stakes_from_title(title))
    return run


def test_blinds_resolved_even_with_hero_blockers(monkeypatch):
    """blinds + hero 类同时阻断 ⇒ 仍应解析盲注并把 blinds 移出阻断清单。"""
    run = _patch(monkeypatch, ["blinds", "hero", "hero_cards", "hero_position"],
                 OBS_MULTI)
    res = run.analyze(b"f", 100, 100, allow_recapture=False)
    assert res.status == "UNAVAILABLE"
    assert "blinds" not in (res.blocking_fields or []), res.blocking_fields
    assert set(res.blocking_fields) == {"hero", "hero_cards", "hero_position"}
    pv = res.frame_provenance or {}
    assert pv.get("blinds_resolved", {}).get("source") == "window_title"
    assert pv["blinds_resolved"]["sb"] == 0.05 and pv["blinds_resolved"]["bb"] == 0.1
    # 解析出的盲注也进 vision 摘要（供 GUI 显示/回写）
    assert res.vision.get("blinds") == {"sb": 0.05, "bb": 0.1}
    assert res.vision.get("blinds_resolved_source") == "window_title"


def test_blinds_only_still_runs_decision(monkeypatch):
    """仅 blinds 阻断 + 标题可解析 ⇒ 仍走决策（不再直接 UNAVAILABLE）。"""
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    frame = ROOT / "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"
    if not frame.exists():
        return
    import copy
    w, h, rgba = decode_png(frame.read_bytes())
    got = pb.frame_to_decision_request(rgba, w, h)
    sim = copy.deepcopy(got)
    vo = sim["request"]["vision_observation"]
    vo["blinds"] = None
    sim["bridge"] = {**sim["bridge"],
                      "observation": {**sim["bridge"]["observation"],
                                       "blinds": None},
                      "blocking_fields": ["blinds"]}
    run = CashGameRunner()
    runtime = run._get_runtime()
    ev = fe.parse_stakes_from_title(TITLE)
    res = run._analyze_with_derived_blinds(runtime, sim, ev, w, h)
    assert res is not None and res.status == "OK"
    assert (res.frame_provenance or {}).get("blinds_source") == "window_title"


def test_no_title_match_keeps_blinds_blocking(monkeypatch):
    run = _patch(monkeypatch, ["blinds", "hero_cards"], OBS_MULTI,
                 title="NLH 1524173 Must Show")
    res = run.analyze(b"f", 100, 100, allow_recapture=False)
    assert "blinds" in (res.blocking_fields or [])
