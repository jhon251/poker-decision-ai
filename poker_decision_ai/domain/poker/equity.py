"""Deterministic equity calculation: exact enumeration + Monte Carlo.

Rule (spec §十一): equity is ALWAYS computed, never LLM-guessed.
  - hand vs hand, river or small enumeration -> exact
  - otherwise -> Monte Carlo (seeded, reproducible)
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from .cards import Card, parse_card, remaining_deck
from .evaluator import evaluate5, evaluate7
from .ranges import Range, combo_to_cards, sample_concrete_pair


@dataclass
class EquityResult:
    hero_equity: float = 0.0
    villain_equity: float = 0.0
    tie: float = 0.0
    samples: int = 0
    method: str = "exact"          # exact | monte_carlo
    win_samples: int = 0
    tie_samples: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hero_equity": round(self.hero_equity, 4),
            "villain_equity": round(self.villain_equity, 4),
            "tie": round(self.tie, 4),
            "samples": self.samples,
            "method": self.method,
            "details": self.details,
        }


def _eval(hole: list[Card], board: list[Card]) -> int:
    cards = hole + board
    return evaluate7(cards) if len(cards) == 7 else evaluate5(cards)


def _to_cards(cards: Iterable[str] | Iterable[Card]) -> list[Card]:
    return [c if isinstance(c, Card) else parse_card(str(c)) for c in cards]


def _finalize(win: int, tie: int, total: int, method: str, details: dict) -> EquityResult:
    if total == 0:
        return EquityResult(0.5, 0.5, 0.0, 0, method, 0, 0, {"error": "no samples"})
    hero = (win + tie / 2) / total
    return EquityResult(
        hero_equity=hero, villain_equity=1 - hero, tie=tie / total,
        samples=total, method=method, win_samples=win, tie_samples=tie, details=details,
    )


class EquityCalculator:
    def __init__(self, iterations: int = 20000, seed: int = 42,
                 exact_limit: int = 400_000) -> None:
        """exact_limit: max enumerations before switching to Monte Carlo."""
        self.iterations = iterations
        self.seed = seed
        self.exact_limit = exact_limit

    # ---------------------------------------------------------------- hand vs hand
    def hand_vs_hand(self, hero: list[str] | list[Card], board: list[str] | list[Card],
                     villain: list[str] | list[Card]) -> EquityResult:
        hero_c = _to_cards(hero)
        board_c = _to_cards(board)
        vill_c = _to_cards(villain)
        remaining = remaining_deck(hero_c + board_c + vill_c)
        needed = 5 - len(board_c)
        if needed < 0:
            raise ValueError("board cannot have more than 5 cards")

        if needed == 0:
            hs, vs = _eval(hero_c, board_c), _eval(vill_c, board_c)
            win = tie = 0
            if hs > vs:
                win = 1
            elif hs == vs:
                tie = 1
            return _finalize(win, tie, 1, "exact", {"board": [str(c) for c in board_c]})

        n_combos = math.comb(len(remaining), needed)
        if n_combos <= self.exact_limit:
            win = tie = 0
            for combo in itertools.combinations(remaining, needed):
                b = board_c + list(combo)
                hs, vs = _eval(hero_c, b), _eval(vill_c, b)
                if hs > vs:
                    win += 1
                elif hs == vs:
                    tie += 1
            return _finalize(win, tie, n_combos, "exact", {"enumerations": n_combos})

        rng = random.Random(self.seed)
        win = tie = 0
        for _ in range(self.iterations):
            b = board_c + rng.sample(remaining, needed)
            hs, vs = _eval(hero_c, b), _eval(vill_c, b)
            if hs > vs:
                win += 1
            elif hs == vs:
                tie += 1
        return _finalize(win, tie, self.iterations, "monte_carlo",
                         {"iterations": self.iterations, "seed": self.seed})

    # ---------------------------------------------------------------- hand vs range
    def hand_vs_range(self, hero: list[str] | list[Card], board: list[str] | list[Card],
                      villain_range: Range, iterations: int | None = None,
                      seed: int | None = None) -> EquityResult:
        hero_c = _to_cards(hero)
        board_c = _to_cards(board)
        iters = iterations or self.iterations
        rng = random.Random(self.seed if seed is None else seed)

        # Build the weighted option list ONCE (O(#combos)), then sample each
        # iteration with a binary search - this is the hot path for MC equity.
        from bisect import bisect_left

        combos, weights = villain_range.collision_free_options(hero_c + board_c)
        if not combos:
            return EquityResult(0.5, 0.5, 0.0, 0, "monte_carlo", 0, 0, {"error": "empty range"})
        cum: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w
            cum.append(acc)
        total_w = acc

        needed = 5 - len(board_c)
        win = tie = evaluated = 0
        for _ in range(iters):
            combo = combos[bisect_left(cum, rng.random() * total_w)]
            vill = sample_concrete_pair(rng, combo[0], combo[1], combo[2], hero_c + board_c)
            if vill is None:      # defensive: option list guarantees availability
                continue
            remaining = remaining_deck(hero_c + board_c + vill)
            b = board_c + (rng.sample(remaining, needed) if needed > 0 else [])
            hs, vs = _eval(hero_c, b), _eval(vill, b)
            evaluated += 1
            if hs > vs:
                win += 1
            elif hs == vs:
                tie += 1
        return _finalize(win, tie, evaluated, "monte_carlo",
                         {"iterations": evaluated, "seed": self.seed if seed is None else seed})

    # ---------------------------------------------------------------- range vs range
    def range_vs_range(self, hero_range: Range, board: list[str] | list[Card],
                       villain_range: Range, iterations: int | None = None,
                       seed: int | None = None) -> EquityResult:
        board_c = _to_cards(board)
        iters = iterations or self.iterations
        rng = random.Random(self.seed if seed is None else seed)
        win = tie = evaluated = 0
        for _ in range(iters):
            hc = hero_range.sample_non_colliding(rng, board_c)
            if hc is None:
                continue
            h_hole = sample_concrete_pair(rng, hc[0], hc[1], hc[2], board_c)
            if h_hole is None:
                continue
            vc = villain_range.sample_non_colliding(rng, board_c + h_hole)
            if vc is None:
                continue
            v_hole = sample_concrete_pair(rng, vc[0], vc[1], vc[2], board_c + h_hole)
            if v_hole is None:
                continue
            remaining = remaining_deck(board_c + h_hole + v_hole)
            needed = 5 - len(board_c)
            b = board_c + (rng.sample(remaining, needed) if needed > 0 else [])
            hs, vs = _eval(h_hole, b), _eval(v_hole, b)
            evaluated += 1
            if hs > vs:
                win += 1
            elif hs == vs:
                tie += 1
        return _finalize(win, tie, evaluated, "monte_carlo",
                         {"iterations": evaluated, "seed": self.seed if seed is None else seed})
