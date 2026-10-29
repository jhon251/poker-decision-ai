# -*- coding: utf-8 -*-
"""P30-9-F：盲注证据推导 + 边界重跑决策（用户现场：blocking=['blinds']）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                       # noqa: E402

FRAME = ROOT / "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"


def _obs_preflop(bets=None, button=6, blinds=None):
    return {"street": "preflop", "button_seat": button, "hero_seat": 5,
            "blinds": blinds,
            "seats": [{"seat": i} for i in range(1, 7)],
            "extra": {"v3f_bets": bets if bets is not None else
                      {"S1": 1000, "S2": 2000, "S6": 1920, "S5": None}}}


def test_observed_blinds_win():
    d = fe.derive_blinds(_obs_preflop(blinds={"sb": 1000, "bb": 2000}))
    assert d["source"] == "observed" and d["sb"] == 1000 and d["bb"] == 2000


def test_derived_from_posted_blinds():
    d = fe.derive_blinds(_obs_preflop(blinds=None))
    assert d["source"] == "derived_from_posted_blinds"
    assert (d["sb"], d["bb"]) == (1000.0, 2000.0)
    assert d["detail"]["sb_seat"] == 1 and d["detail"]["bb_seat"] == 2


def test_postflop_does_not_derive():
    o = _obs_preflop(blinds=None)
    o["street"] = "flop"
    assert fe.derive_blinds(o)["source"] == "no_evidence"


def test_missing_button_does_not_derive():
    o = _obs_preflop(blinds=None)
    o["button_seat"] = None
    assert fe.derive_blinds(o)["source"] == "no_evidence"


def test_inconsistent_blinds_are_rejected():
    # BB < SB ⇒ 证据不自洽 ⇒ 不推导（维持阻断）
    o = _obs_preflop(bets={"S1": 2000, "S2": 1000}, blinds=None)
    d = fe.derive_blinds(o)
    assert d["sb"] is None and d["bb"] is None
    assert d["source"] == "derived_insufficient"


def test_real_frame_derivation_matches_header_recognition():
    """真帧：仅用盲注座投入推导 → 必须等于 header 识别出的盲注。"""
    import pytest
    pytest.importorskip("numpy")
    if not FRAME.exists():
        import pytest as _p
        _p.skip("evidence frame missing")
    from poker_decision_ai.vision2.pixel.realclient import production_bridge as pb
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png(FRAME.read_bytes())
    obs = pb.frame_to_decision_request(rgba, w, h)["bridge"]["observation"]
    header = obs.get("blinds") or {}
    stripped = dict(obs)
    stripped["blinds"] = None
    d = fe.derive_blinds(stripped)
    assert d["source"] == "derived_from_posted_blinds"
    assert d["sb"] == header.get("sb") and d["bb"] == header.get("bb")
