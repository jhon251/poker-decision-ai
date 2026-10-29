"""Deterministic poker hand evaluator (5- and 7-card).

Score encoding: score = category << 20 | t1 << 16 | t2 << 12 | t3 << 8 | t4 << 4 | t5
categories: 8 straight-flush, 7 quads, 6 full-house, 5 flush, 4 straight,
            3 trips, 2 two-pair, 1 pair, 0 high-card.

Pure integer arithmetic - no LLM, no numpy, trivially testable.
"""
from __future__ import annotations

from functools import lru_cache

from .cards import Card

CATEGORY_NAMES = {
    8: "straight_flush", 7: "four_of_a_kind", 6: "full_house", 5: "flush",
    4: "straight", 3: "three_of_a_kind", 2: "two_pair", 1: "one_pair", 0: "high_card",
}


def evaluate5(cards: list[Card]) -> int:
    """Evaluate exactly 5 cards -> integer score (higher is better)."""
    assert len(cards) == 5, "evaluate5 requires exactly 5 cards"
    counts = [0] * 15
    suits = [0] * 4
    mask = 0
    for c in cards:
        counts[c.rank] += 1
        suits[c.suit] += 1
        mask |= 1 << c.rank

    flush_suit = next((s for s in range(4) if suits[s] == 5), None)

    straight_high = 0
    for high in range(14, 5, -1):  # A..6 tops
        window = 0b11111 << (high - 4)
        if mask & window == window:
            straight_high = high
            break
    if straight_high == 0 and (mask & ((1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2))) == ((1 << 14) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2)):
        # wheel: A-5
        straight_high = 5

    def kickers(exclude_rank: int = -1, n: int = 5) -> tuple:
        out = []
        for r in range(14, 1, -1):
            if r == exclude_rank:
                continue
            for _ in range(min(counts[r], n - len(out))):
                out.append(r)
            if len(out) >= n:
                break
        return tuple(out)

    # straight flush
    if flush_suit is not None and straight_high:
        return (8 << 20) | (straight_high << 16)
    # quads
    for r in range(14, 1, -1):
        if counts[r] == 4:
            k = max(rr for rr in range(14, 1, -1) if rr != r and counts[rr] > 0)
            return (7 << 20) | (r << 16) | (k << 12)
    # full house
    trips = [r for r in range(14, 1, -1) if counts[r] == 3]
    pairs = [r for r in range(14, 1, -1) if counts[r] == 2]
    if trips and (pairs or len(trips) > 1):
        t = trips[0]
        p = trips[1] if len(trips) > 1 else pairs[0]
        return (6 << 20) | (t << 16) | (p << 12)
    # flush
    if flush_suit is not None:
        ranks = sorted((c.rank for c in cards if c.suit == flush_suit), reverse=True)
        return (5 << 20) | (ranks[0] << 16) | (ranks[1] << 12) | (ranks[2] << 8) | (ranks[3] << 4) | ranks[4]
    # straight
    if straight_high:
        return (4 << 20) | (straight_high << 16)
    # trips
    if trips:
        t = trips[0]
        k = kickers(t, 2)
        return (3 << 20) | (t << 16) | (k[0] << 12) | (k[1] << 8)
    # two pair
    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        k = max(r for r in range(14, 1, -1) if counts[r] and r not in (hi, lo))
        return (2 << 20) | (hi << 16) | (lo << 12) | (k << 8)
    # pair
    if pairs:
        p = pairs[0]
        k = kickers(p, 3)
        return (1 << 20) | (p << 16) | (k[0] << 12) | (k[1] << 8) | (k[2] << 4)
    # high card
    k = kickers(n=5)
    return (0 << 20) | (k[0] << 16) | (k[1] << 12) | (k[2] << 8) | (k[3] << 4) | k[4]


def score_category(score: int) -> int:
    return score >> 20


def score_name(score: int) -> str:
    return CATEGORY_NAMES[score_category(score)]


def compare(score_a: int, score_b: int) -> int:
    """-1 / 0 / 1."""
    return (score_a > score_b) - (score_a < score_b)


@lru_cache(maxsize=65536)
def evaluate7_cached(codes: tuple[int, ...]) -> int:
    from itertools import combinations

    cards = [Card(c) for c in codes]
    return max(evaluate5(list(combo)) for combo in combinations(cards, 5))


def evaluate7(cards: list[Card]) -> int:
    """Best 5-of-7 score (cached by sorted card tuple)."""
    return evaluate7_cached(tuple(sorted(c.code for c in cards)))


def best_five(cards: list[Card]) -> tuple[int, list[Card]]:
    """Return (score, best five cards) for a 5-7 card hand."""
    from itertools import combinations

    best = -1
    best_combo: list[Card] = []
    for combo in combinations(cards, 5):
        s = evaluate5(list(combo))
        if s > best:
            best, best_combo = s, list(combo)
    return best, best_combo
