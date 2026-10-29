# -*- coding: utf-8 -*-
"""P30-9-AD：自校准发现器 + 标签映射 + 只补不改合并。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import bet_capsule_finder as bcf                     # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


# ── 标签映射（锚点表座位 ↔ bet 环标签） ─────────────────────────────
def test_anchor_to_bet_ring_label_mapping():
    got = bcf.to_bet_ring_labels({"S1": 1, "S4": 4, "S6": 6})
    assert got == {"S1": 1, "S6": 4, "S4": 6}       # 底中/左下互换
    assert bcf.ANCHOR_TO_BET_LABEL_1876["S6"] == "S4"   # 底中（hero）→ bet 环 S4
    assert bcf.ANCHOR_TO_BET_LABEL_1876["S4"] == "S6"   # 左下 → bet 环 S6


# ── 发现器：真实帧上必须找到全部"裸台面"投注位 ────────────────────
def _load(rel: str):
    import numpy as np
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png(pathlib.Path(ROOT / rel).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def test_finder_covers_all_visible_bets_user_2250():
    frame = _load("data/vision2/coinpoker_evidence/user_2250_preflop_q6s.png")
    out = bcf.find_bet_capsules(frame, bcf.__dict__.get("SEAT_ANCHORS_1876") or
                                 __import__("poker_decision_ai.vision2.pixel."
                                            "realclient.hero_seat",
                                            fromlist=["SEAT_ANCHORS_1876"])
                                 .SEAT_ANCHORS_1876,
                                 cross_check={"pot": 0.24,
                                               "stacks": {"S1": 4.98, "S6": 4.96}})
    ring = bcf.to_bet_ring_labels(out["seats"])
    assert ring.get("S1") == 0.02 and ring.get("S2") == 0.05
    assert ring.get("S3") == 0.12
    assert ring.get("S4") == 0.05            # hero 自己的投入（原缺失位）


def test_finder_excludes_pot():
    frame = _load("data/vision2/coinpoker_evidence/user_2250_preflop_q6s.png")
    from poker_decision_ai.vision2.pixel.realclient.hero_seat import (
        SEAT_ANCHORS_1876)
    out = bcf.find_bet_capsules(frame, SEAT_ANCHORS_1876,
                                 cross_check={"pot": 0.24, "stacks": {}})
    assert all(c["value"] != 0.24 for c in out["capsules"])   # 底池不得被当下注


# ── 合并：只补不改 ─────────────────────────────────────────────────
def test_merge_fills_only_missing_and_never_overrides(monkeypatch):
    run = CashGameRunner()
    obs = {"pot": 0.24, "seats": [], "extra": {"v3f_bets": {"S1": 0.02, "S2": None}}}
    monkeypatch.setattr(
        run, "_fill_bets_from_finder", run._fill_bets_from_finder)
    # 直接验证"只补不改"语义：伪造发现器结果
    import poker_decision_ai.bet_capsule_finder as m
    monkeypatch.setattr(m, "find_bet_capsules",
                        lambda *a, **k: {"seats": {"S1": 9.99, "S2": 0.05, "S3": 0.12}})
    monkeypatch.setattr(m, "to_bet_ring_labels", lambda s: dict(s))
    out = run._fill_bets_from_finder(b"x" * 16, 2, 2, obs)
    bets = out["extra"]["v3f_bets"]
    assert bets["S1"] == 0.02                 # 已有值不得被覆盖
    assert bets["S2"] == 0.05                 # 空缺被补齐
    assert bets["S3"] == 0.12
    assert out["extra"]["bets_filled_by_finder"] == {"S2": 0.05, "S3": 0.12}
    assert out["extra"]["bets_conflicts"] == {"S1": {"observer": 0.02,
                                                       "finder": 9.99}}


def test_merge_is_noop_without_frame():
    run = CashGameRunner()
    obs = {"extra": {"v3f_bets": {"S1": None}}}
    assert run._fill_bets_from_finder(None, None, None, obs) is obs
