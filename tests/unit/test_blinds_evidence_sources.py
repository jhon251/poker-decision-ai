# -*- coding: utf-8 -*-
"""P30-9-K：盲注证据补强（窗口标题解析 + 手动输入兜底）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


# ── 窗口标题解析 ────────────────────────────────────────────────────
def test_title_parsing_common_forms():
    for title, sb, bb in (
            ("CoinPoker NL5 0.02/0.05 - Table 3", 0.02, 0.05),
            ("$0.02/$0.05 Hold'em", 0.02, 0.05),
            ("CoinPoker 0,02/0,05 (6-max)", 0.02, 0.05),
            ("NL2 0.01 / 0.02", 0.01, 0.02),
            ("Table 5 - 1/2", 1.0, 2.0)):
        ev = fe.parse_stakes_from_title(title)
        assert ev["source"] == "window_title", title
        assert (ev["sb"], ev["bb"]) == (sb, bb), title


def test_title_parsing_rejects_unsafe_or_absent():
    # 比例过大（疑似日期/版本）、倒置、无数字对 ⇒ 一律不采用
    for title in ("CoinPoker 2024/10/09", "0.05/0.02", "CoinPoker Table 7",
                   "v1.2/3.4.5", "", None, "6/9 6-max"):   # 6/9：9 不在盲注白名单
        ev = fe.parse_stakes_from_title(title)
        assert ev["source"] == "no_evidence", title
        assert ev["sb"] is None and ev["bb"] is None


def test_title_parsing_sanity_ratio_bound():
    assert fe.parse_stakes_from_title("0.02/0.05")["source"] == "window_title"
    assert fe.parse_stakes_from_title("0.02/0.05")["bb"] == 0.05
    # bb/sb = 5 > 上限 ⇒ 拒绝
    assert fe.parse_stakes_from_title("1/5")["source"] == "no_evidence"
    # BB 不在常见盲注表内 ⇒ 拒绝（防误读）
    assert fe.parse_stakes_from_title("9/9")["source"] == "no_evidence"
    assert fe.parse_stakes_from_title("0.02/0.07")["source"] == "no_evidence"


# ── 手动输入（user_setting） ────────────────────────────────────────
def test_manual_override_evidence_valid_and_invalid():
    ok = CashGameRunner._blinds_override_ev({"sb": 0.02, "bb": 0.05})
    assert ok["source"] == "user_setting" and (ok["sb"], ok["bb"]) == (0.02, 0.05)
    for bad in (None, {}, {"sb": "x", "bb": "y"}, {"sb": 0.05, "bb": 0.02},
                {"sb": 0, "bb": 0.05}, {"sb": -1, "bb": 0.05}, {"sb": 0.02}):
        assert CashGameRunner._blinds_override_ev(bad) == {}


def test_evidence_priority_declared():
    c = fe.evidence_contract()
    pri = c["blinds_priority"]
    assert pri[0] == "observed" and pri[1] == "derived_from_posted_blinds"
    assert any("user_setting" in p for p in pri)
    assert any("window_title" in p for p in pri)
    assert pri[-1] == "no_evidence"


# ── 端到端：标题/手动证据都能驱动决策（用真实帧 + 置空 blinds） ────────
def test_manual_override_unblocks_real_frame(monkeypatch):
    import copy
    import pytest
    pytest.importorskip("numpy")
    frame = ROOT / "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"
    if not frame.exists():
        pytest.skip("evidence frame missing")
    from poker_decision_ai.vision2.pixel.realclient import production_bridge as pb
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png(frame.read_bytes())
    got = pb.frame_to_decision_request(rgba, w, h)
    sim = copy.deepcopy(got)
    vo = sim["request"]["vision_observation"]
    vo["blinds"] = None
    sim["bridge"] = {**sim["bridge"],
                     "observation": {**(sim["bridge"]["observation"]), "blinds": None},
                     "blocking_fields": ["blinds"]}
    run = CashGameRunner()
    runtime = run._get_runtime()
    ev = run._blinds_override_ev({"sb": 0.02, "bb": 0.05})
    assert ev
    res = run._analyze_with_derived_blinds(runtime, sim, ev, w, h)
    assert res is not None
    assert res.status == "OK"
    assert (res.frame_provenance or {}).get("blinds_source") == "user_setting"
    assert res.decision_action in {"fold", "call", "raise", "check", "bet", "allin"}
