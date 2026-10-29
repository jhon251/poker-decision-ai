# -*- coding: utf-8 -*-
"""P30-9-H：盲注会话记忆（同桌同局复用）+ hero 阻断键覆盖。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402


def test_memo_store_and_lookup_fresh():
    memo: dict = {}
    ev = {"sb": 20.0, "bb": 50.0, "source": "observed",
          "detail": {"blinds": {"sb": 20, "bb": 50}}}
    fe.memo_store(memo, "bbox|6", ev, now=1000.0)
    got = fe.memo_lookup(memo, "bbox|6", now=1010.0)
    assert got is not None
    assert got["source"] == "session_memo"
    assert (got["sb"], got["bb"]) == (20.0, 50.0)
    assert got["detail"]["memo_origin"] == "observed"
    assert got["detail"]["memo_age_s"] == 10.0


def test_memo_expires_after_ttl():
    memo: dict = {}
    fe.memo_store(memo, "k", {"sb": 1, "bb": 2, "source": "observed"}, now=0.0)
    assert fe.memo_lookup(memo, "k", now=100.0) is not None
    assert fe.memo_lookup(memo, "k", now=fe.BLINDS_MEMO_TTL_S + 1) is None


def test_memo_never_crosses_tables():
    memo: dict = {}
    fe.memo_store(memo, "tableA|6", {"sb": 1, "bb": 2, "source": "observed"},
                  now=0.0)
    assert fe.memo_lookup(memo, "tableB|6", now=1.0) is None


def test_memo_ignores_invalid_evidence():
    memo: dict = {}
    fe.memo_store(memo, "k", {"sb": None, "bb": 2, "source": "no_evidence"},
                  now=0.0)
    fe.memo_store(memo, "k", {"sb": 0, "bb": 2, "source": "no_evidence"}, now=0.0)
    assert memo == {}


def test_derived_preflop_evidence_feeds_memo_then_serves_postflop():
    """preflop 推导 → 记忆 → flop（无法推导）复用。"""
    preflop = {"street": "preflop", "button_seat": 6, "hero_seat": 5,
               "seats": [{"seat": i} for i in range(1, 7)],
               "extra": {"v3f_bets": {"S1": 20, "S2": 50, "S6": 40, "S5": None}}}
    ev = fe.derive_blinds(preflop)
    assert ev["source"] == "derived_from_posted_blinds"
    memory: dict = {}
    fe.memo_store(memory, "bbox|6", ev, now=100.0)
    flop = {"street": "flop", "button_seat": 6, "hero_seat": 5,
            "seats": [{"seat": i} for i in range(1, 7)],
            "extra": {"v3f_bets": {"S6": 32, "S5": None}}}
    assert fe.derive_blinds(flop)["source"] == "no_evidence"      # flop 推不出
    reuse = fe.memo_lookup(memory, "bbox|6", now=140.0)
    assert reuse is not None and (reuse["sb"], reuse["bb"]) == (20.0, 50.0)


def test_contract_declares_memo_ttl():
    c = fe.evidence_contract()
    assert c["blinds_memo_ttl_s"] == fe.BLINDS_MEMO_TTL_S
    assert any("session_memo" in p for p in c["blinds_priority"])
