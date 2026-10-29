"""Vision V3E-P1 — Runtime Output Contract Validation 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    OCCUPIED, EMPTY, UNKNOWN, PositionRuntimeResult, compute_positions,
    to_assembly_contract)
from poker_decision_ai.domain.poker.models import positions_for  # noqa: E402
from poker_decision_ai.vision2.assembly import _positions_for  # noqa: E402


def _occ(**kw):
    st = {s: EMPTY for s in ("S1", "S2", "S3", "S4", "S5", "S6")}
    st.update(kw)
    return st


# ── P1-2：adapter 契约 ─────────────────────────────────────────────
def test_adapter_maps_to_int_domain():
    a = to_assembly_contract(_occ(S1="OCCUPIED", S3="OCCUPIED",
                                  S4="OCCUPIED"), "S4", "S1")
    assert a["button_seat"] == 6            # S4 → 环序号 6
    assert a["hero_seat"] == 1              # S1 → 1
    assert set(a["occupied_ringed"]) == {1, 3, 6}


def test_adapter_unknown_propagates():
    a = to_assembly_contract(_occ(S1="OCCUPIED", S4="OCCUPIED"), None, "S1")
    assert a["button_seat"] is None         # dealer 缺失 → None（不 fallback）
    a2 = to_assembly_contract(_occ(S1="OCCUPIED"), "S1", None)
    assert a2["hero_seat"] is None


def test_adapter_feeds_frozen_positions_for():
    """adapter 输出 → 冻结 positions_for → hero.position 语义正确。
    （3 人局对照 1328 实况：dealer S4 → 环 6；S1=SB, S3=BB。）"""
    a = to_assembly_contract(_occ(S1="OCCUPIED", S3="OCCUPIED",
                                  S4="OCCUPIED"), "S4", "S1")
    players = [{"seat": rn} for rn in sorted(a["occupied_ringed"])]
    pm = positions_for(players, a["button_seat"])
    assert pm[a["hero_seat"]] == "SB"
    assert pm[3] == "BB"                     # S3 环序号 3


def test_adapter_full_6max_hero_bb():
    a = to_assembly_contract(
        _occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED",
             S5="OCCUPIED", S6="OCCUPIED"), "S3", "S6")
    players = [{"seat": rn} for rn in sorted(a["occupied_ringed"])]
    pm = positions_for(players, a["button_seat"])
    assert pm[a["hero_seat"]] == "BB"        # S6 环 5 → BB（与 V3C 一致）


# ── P1-4：contract 测试矩阵 ──────────────────────────────────────────
def test_valid_matrix():
    cases = [
        (_occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED",
              S5="OCCUPIED", S6="OCCUPIED"), "S3", "S6", "BB"),
        (_occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED",
              S5="OCCUPIED"), "S3", "S5", "SB"),
        (_occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED"),
         "S4", "S1", "SB"),
        (_occ(S1="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED"), "S4", "S3",
         "BB"),
        (_occ(S1="OCCUPIED", S4="OCCUPIED"), "S4", "S4",
         "BTN/SB"),
    ]
    for occ, dealer, hero, expect in cases:
        r = compute_positions(occ, dealer, hero)
        assert r["uncertainty"] == "RECOGNIZED", (occ, dealer, hero, r)
        assert r["hero_position"] == expect, (occ, dealer, hero, r)


def test_invalid_unknown_matrix():
    """任何无法唯一确定 → UNKNOWN/None。"""
    assert compute_positions(_occ(S1="OCCUPIED"), None, "S1")[
        "uncertainty"] == "UNKNOWN"                  # dealer 不可见
    assert compute_positions(_occ(S1="OCCUPIED"), "S1", None)[
        "uncertainty"] == "UNKNOWN"                  # hero 不可见
    assert compute_positions(_occ(S1="OCCUPIED", S2="OCCUPIED"), "S3",
                             "S1")["uncertainty"] == "UNKNOWN"   # dealer 不在座
    assert compute_positions(_occ(S1="OCCUPIED", S3="OCCUPIED",
                                  S4="UNKNOWN"), "S4", "S1")[
        "uncertainty"] == "UNKNOWN"                  # occupied 不完整
    # 1419 crop：S2 不可见 → 若输入含 UNKNOWN → 不完整 → UNKNOWN
    assert compute_positions(_occ(S1="OCCUPIED", S3="OCCUPIED",
                                  S4="OCCUPIED", S2="UNKNOWN"), "S4",
                             "S1")["uncertainty"] == "UNKNOWN"
    # 非法座
    assert compute_positions(_occ(S7="OCCUPIED"), "S7", "S7")[
        "uncertainty"] == "UNKNOWN"
    # hero 与 occupied 冲突（hero 不在座）
    assert compute_positions(_occ(S1="OCCUPIED", S4="OCCUPIED"), "S4",
                             "S3")["uncertainty"] == "UNKNOWN"


def test_multi_hero_conflict_not_supported_by_contract():
    """本轮无多 hero 检测（C3 单座证据门）——contract 对 hero 冲突的
    表达=UNKNOWN（compute_positions 仅接受单 hero_seat 或 None）。"""
    r = compute_positions(_occ(S1="OCCUPIED", S6="OCCUPIED"), "S6", None)
    assert r["uncertainty"] == "UNKNOWN"        # hero 缺失 → UNKNOWN


# ── P1-3 / P1-5：fixture 分离 + provenance ───────────────────────────
def test_no_production_fallback():
    """runtime 路径无 table_size/hero 默认值/固定 dealer fallback。"""
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient" / "seat_position.py").read_text(
        encoding="utf-8")
    assert "table_size" not in src
    assert 'hero_seat = "S6"' not in src
    assert "hero_seat = 0" not in src
    assert "hero_seat = 3" not in src
    assert 'dealer_seat = "S3"' not in src
    assert "dealer_seat = 0" not in src


def test_provenance_rich_enough():
    r = PositionRuntimeResult(occupied_seats={"S6": OCCUPIED},
                              dealer_seat=None, hero_seat="S6")
    d = r.to_dict()
    assert d["uncertainty"] == "UNKNOWN"
    assert d["provenance"] == "runtime_state_non_gt"
    assert d["ground_truth_status"] == "runtime_state_non_gt"
    # 消费方可区分：visual/derived/UNKNOWN（uncertainty + provenance + reason）
    assert d["hero_position"] is None
