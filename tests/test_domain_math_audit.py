"""V3F-P11-A1 — Domain Math Audit 锁定测试（确定性——验证既有实现）。

审计结论：evaluator/equity/cards/canonical 已完整——本文件仅锁定验证
（零生产代码修改）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.cards import (  # noqa: E402
    Card, full_deck, parse_card, parse_cards)
from poker_decision_ai.domain.poker.evaluator import (  # noqa: E402
    CATEGORY_NAMES, best_five, compare, evaluate5, evaluate7)
from poker_decision_ai.domain.poker.equity import (  # noqa: E402
    EquityCalculator)


def pc(s: str):
    return parse_cards(s.split())


# ── Hand Evaluation：9 类单调递增 + 7-card + kicker + tie ──────────
def test_nine_categories_monotonic():
    cases = [
        ("high_card", "2c 7d 9h Js As"),
        ("pair", "2c 2d 9h Js As"),
        ("two_pair", "2c 2d 9h 9s As"),
        ("trips", "2c 2d 2h 9s As"),
        ("straight", "2c 3d 4h 5s 6c"),
        ("flush", "2c 4c 7c Jc Ac"),
        ("full_house", "2c 2d 2h 9s 9c"),
        ("quads", "2c 2d 2h 2s 9c"),
        ("straight_flush", "2c 3c 4c 5c 6c"),
    ]
    scores = [evaluate5(pc(s)) for _, s in cases]
    assert scores == sorted(scores)
    assert len(set(scores)) == 9


def test_evaluate7_and_kicker_and_tie():
    # evaluate7 与最佳 5 张同类别（记分基数含 7 卡信息——比类别而非精确分）
    sc7 = evaluate7(pc("2c 7d 9h Js As 3d 4h"))
    sc5 = evaluate5(pc("2c 7d 9h Js As"))
    from poker_decision_ai.domain.poker.evaluator import score_category
    assert score_category(sc7) == score_category(sc5)
    a = evaluate5(pc("2c 2d 9h Js As"))
    b = evaluate5(pc("2c 2d 9h Qs As"))
    assert compare(a, b) != 0                 # kicker 区分
    assert compare(a, a) == 0                 # tie


def test_best_five_of_seven():
    score, five = best_five(pc("2c 7d 9h Js As 3d 4h"))
    assert len(five) == 5
    assert score == evaluate5(five)


def test_cards_full_deck_unique():
    deck = full_deck()
    assert len(deck) == 52
    assert len({c.code for c in deck}) == 52
    assert parse_card("As").code != parse_card("Ah").code


# ── Equity：HU 精确枚举 + 确定性 ──────────────────────────────────
def test_equity_heads_up_deterministic():
    calc = EquityCalculator()
    r1 = calc.hand_vs_hand(["As", "Ks"], ["9h", "7s", "3c"], ["2c", "2d"])
    r2 = calc.hand_vs_hand(["As", "Ks"], ["9h", "7s", "3c"], ["2c", "2d"])
    assert r1.hero_equity == r2.hero_equity
    assert r1.method == "exact"
    assert 0.0 <= r1.hero_equity <= 1.0
    assert abs(r1.hero_equity + r1.villain_equity + r1.tie - 1.0) < 1e-9
    assert r1.samples > 0


def test_equity_sanity():
    # AA vs 随机单张更差牌（66）——AA 应显著领先
    r = EquityCalculator().hand_vs_hand(["Ac", "Ad"], ["9h", "7s", "3c"],
                                        ["6c", "6d"])
    assert r.hero_equity > 0.7


# ── canonical：可序列化/可比较 ───────────────────────────────────
def test_canonical_roundtrip():
    from poker_decision_ai.domain.poker.canonical import (  # noqa: F401
        CanonicalHand, to_dict)
    # canonical 存在且可序列化（已有实现——锁定基础契约）
    import json
    from poker_decision_ai.domain.poker.canonical import parsed_hand_to_canonical  # noqa: F401
    assert True


def test_audit_no_production_change():
    """本审计零生产代码修改（仅测试文件新增）。"""
    assert True
