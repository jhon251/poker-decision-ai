"""Range model: weighted combo distributions over the 1326 preflop combos.

A combo is (hi_rank, lo_rank, suited) with hi >= lo; pairs use hi == lo and
suited=True by convention.  Ranges support:
  - notation parsing ("22+", "AKs", "KQo", "A5s-A2s", "76s+", "AJs+:0.5", ...)
  - card removal (hero hole cards + board) -> dead combos get weight 0
  - weighted sampling for Monte Carlo equity
  - 169-bucket categorical distributions for output/reporting
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from .cards import Card, RANKS, RANK_VALUES, VALUE_RANKS

#: (hi, lo, suited) ; pairs use hi == lo, suited = True
Combo = tuple[int, int, bool]

PAIR_MARK = True  # pairs are stored with suited=True


def combo_key(hi: int, lo: int, suited: bool) -> Combo:
    if hi < lo:
        hi, lo = lo, hi
    if hi == lo:
        return (hi, lo, True)
    return (hi, lo, bool(suited))


def combo_name(combo: Combo) -> str:
    hi, lo, suited = combo
    h, l = VALUE_RANKS[hi], VALUE_RANKS[lo]
    if hi == lo:
        return f"{h}{l}"
    return f"{h}{l}{'s' if suited else 'o'}"


def combo_to_cards(combo: Combo) -> list[Card]:
    """A canonical card pair for the combo (suit assignment matters only for
    removal checks; exact suits are handled by the caller via sampling)."""
    hi, lo, suited = combo
    if hi == lo:
        return [Card((hi - 2) * 4), Card((hi - 2) * 4 + 1)]
    if suited:
        return [Card((hi - 2) * 4), Card((lo - 2) * 4)]
    return [Card((hi - 2) * 4), Card((lo - 2) * 4 + 1)]


def _iter_combo_ranks(hi: int, lo: int) -> Iterator[tuple[int, int]]:
    yield hi, lo


@dataclass
class Range:
    """Weighted set of combos. weight 0 == excluded."""

    weights: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        self.weights = {k: float(v) for k, v in self.weights.items() if v > 0}

    # ---- construction ----
    def set_weight(self, hi: int, lo: int, suited: bool, weight: float) -> None:
        key = combo_key(hi, lo, suited)
        if weight <= 0:
            self.weights.pop(key, None)
        else:
            self.weights[key] = weight

    def add(self, hi: int, lo: int, suited: bool, weight: float = 1.0) -> "Range":
        self.set_weight(hi, lo, suited, weight)
        return self

    # ---- queries ----
    def combos(self) -> list[Combo]:
        return list(self.weights)

    def total_weight(self) -> float:
        return sum(self.weights.values())

    def weight_of(self, combo: Combo) -> float:
        return self.weights.get(combo_key(*combo), 0.0)

    def contains(self, hi: int, lo: int, suited: bool) -> bool:
        return self.weight_of(combo_key(hi, lo, suited)) > 0

    def is_empty(self) -> bool:
        return not self.weights

    # ---- removal ----
    def remove_cards(self, cards: list[Card]) -> "Range":
        """Zero-out combos that use any dead card (hero hole / board)."""
        dead_ranks: dict[int, set[int]] = {}
        for c in cards:
            dead_ranks.setdefault(c.rank, set()).add(c.suit)
        out = Range(name=self.name)
        for (hi, lo, suited), w in self.weights.items():
            available = 0
            for combo in _suit_assignments(hi, lo, suited):
                c1, c2 = combo
                if not _collides(c1, dead_ranks) and not _collides(c2, dead_ranks):
                    available += 1
            if available > 0:
                out.weights[(hi, lo, suited)] = w * available / _total_assignments(hi, lo, suited)
        return out

    # ---- sampling ----
    def sample_weighted(self, rng: random.Random, n: int) -> list[Combo]:
        """Sample n combos with replacement, weighted."""
        if not self.weights:
            return []
        items = list(self.weights.items())
        total = sum(w for _, w in items)
        r = rng.random
        out = []
        for _ in range(n):
            x = r() * total
            acc = 0.0
            for combo, w in items:
                acc += w
                if x <= acc:
                    out.append(combo)
                    break
        return out

    def sample_non_colliding(self, rng: random.Random, dead_cards: list[Card],
                             preferred: Combo | None = None) -> Combo | None:
        """Sample one combo whose cards do not collide with dead cards."""
        combos, weights = self.collision_free_options(dead_cards)
        if not combos:
            return None
        total = sum(weights)
        x = rng.random() * total
        acc = 0.0
        for combo, w in zip(combos, weights):
            acc += w
            if x <= acc:
                return combo
        return combos[-1]

    def collision_free_options(self, dead_cards: list[Card]) -> tuple[list[Combo], list[float]]:
        """(combos, weights) for all combos with at least one non-colliding pair.

        Weights are scaled by available/total concrete pairs (card removal).
        Computed in O(#combos) with O(1) arithmetic per combo - this is the
        hot path for Monte Carlo equity.
        """
        if not self.weights:
            return [], []
        dead_ranks = _dead_rank_map(dead_cards)
        combos: list[Combo] = []
        weights: list[float] = []
        for (hi, lo, suited), w in self.weights.items():
            avail = _available_count(hi, lo, suited, dead_ranks)
            if avail > 0:
                combos.append((hi, lo, suited))
                weights.append(w * avail / _total_assignments(hi, lo, suited))
        return combos, weights

    # ---- distribution over 169 buckets ----
    def distribution(self) -> dict[str, float]:
        """Normalized weight per 169 category name."""
        out: dict[str, float] = {}
        for combo, w in self.weights.items():
            out[combo_name(combo)] = out.get(combo_name(combo), 0.0) + w
        total = sum(out.values()) or 1.0
        return {k: round(v / total, 6) for k, v in sorted(out.items())}

    def stats(self) -> dict:
        total = self.total_weight()
        return {
            "name": self.name,
            "combo_count": len(self.weights),
            "total_weight": round(total, 3),
            "pairs_pct": round(sum(w for (h, l, _), w in self.weights.items() if h == l) / total * 100, 2) if total else 0.0,
            "suited_pct": round(sum(w for (h, l, s), w in self.weights.items() if h != l and s) / total * 100, 2) if total else 0.0,
            "broadway_pct": round(sum(w for (h, l, _), w in self.weights.items() if h >= 10 and l >= 10) / total * 100, 2) if total else 0.0,
        }

    def __len__(self) -> int:
        return len(self.weights)

    def __or__(self, other: "Range") -> "Range":
        out = Range(dict(self.weights), name=self.name)
        for k, v in other.weights.items():
            out.weights[k] = max(out.weights.get(k, 0.0), v)
        return out

    def __and__(self, other: "Range") -> "Range":
        keys = set(self.weights) & set(other.weights)
        return Range({k: min(self.weights[k], other.weights[k]) for k in keys}, name=self.name)


# ---------------------------------------------------------------- helpers
def _total_assignments(hi: int, lo: int, suited: bool) -> float:
    if hi == lo:
        return 6.0
    return 4.0 if suited else 12.0


def _suit_assignments(hi: int, lo: int, suited: bool) -> list[tuple[Card, Card]]:
    """All concrete card pairs for a combo category."""
    if hi == lo:
        suits = list(range(4))
        return [(Card((hi - 2) * 4 + s1), Card((hi - 2) * 4 + s2))
                for i, s1 in enumerate(suits) for s2 in suits[i + 1:]]
    if suited:
        return [(Card((hi - 2) * 4 + s), Card((lo - 2) * 4 + s)) for s in range(4)]
    return [(Card((hi - 2) * 4 + s1), Card((lo - 2) * 4 + s2))
            for s1 in range(4) for s2 in range(4) if s1 != s2]


def _collides(card: Card, dead_ranks: dict[int, set[int]]) -> bool:
    return card.rank in dead_ranks and card.suit in dead_ranks[card.rank]


def _available_count(hi: int, lo: int, suited: bool, dead_ranks: dict[int, set[int]]) -> float:
    """Number of concrete card pairs of the combo that avoid all dead cards.

    O(1) arithmetic (no card-object construction):
      - pair:   C(4-k, 2) with k dead cards of that rank
      - suited: 4 - |blocked suits| (both cards share the suit)
      - offsuit: a*b - overlap, a/b = free suits per rank, overlap = shared free suits
    """
    if hi == lo:
        k = len(dead_ranks.get(hi, ()))
        return 3.0 if k == 1 else (1.0 if k == 2 else (0.0 if k >= 3 else 6.0))
    dead_a = dead_ranks.get(hi, set())
    dead_b = dead_ranks.get(lo, set())
    if suited:
        return float(4 - len(dead_a | dead_b))
    a = 4 - len(dead_a)
    b = 4 - len(dead_b)
    if a <= 0 or b <= 0:
        return 0.0
    overlap = 4 - len(dead_a | dead_b)
    return float(a * b - overlap)


def _dead_rank_map(dead_cards: list[Card]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for c in dead_cards:
        out.setdefault(c.rank, set()).add(c.suit)
    return out


def sample_concrete_pair(rng: random.Random, hi: int, lo: int, suited: bool,
                         dead_cards: list[Card]) -> list[Card] | None:
    """Uniformly sample ONE concrete card pair of the combo that avoids dead cards.

    Fixes the canonical-pair collision bug: combo_to_cards() returns a fixed
    pair which may collide with hero hole cards / board; sampling must instead
    pick uniformly among the *free* concrete pairs.
    """
    dead_ranks = _dead_rank_map(dead_cards)
    if hi == lo:
        free = [s for s in range(4) if s not in dead_ranks.get(hi, set())]
        if len(free) < 2:
            return None
        s1, s2 = rng.sample(free, 2)
        return [Card((hi - 2) * 4 + s1), Card((hi - 2) * 4 + s2)]
    if suited:
        free = [s for s in range(4)
                if s not in dead_ranks.get(hi, set()) and s not in dead_ranks.get(lo, set())]
        if not free:
            return None
        s = rng.choice(free)
        return [Card((hi - 2) * 4 + s), Card((lo - 2) * 4 + s)]
    free_a = [s for s in range(4) if s not in dead_ranks.get(hi, set())]
    free_b = [s for s in range(4) if s not in dead_ranks.get(lo, set())]
    pairs = [(a, b) for a in free_a for b in free_b if a != b]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    return [Card((hi - 2) * 4 + a), Card((lo - 2) * 4 + b)]


def _rank_index(ch: str) -> int:
    if ch.upper() not in RANK_VALUES:
        raise ValueError(f"invalid rank '{ch}'")
    return RANK_VALUES[ch.upper()]


# ---------------------------------------------------------------- parser
def parse_range(text: str, name: str = "") -> Range:
    """Parse a range string like 'AA,KK,AKs,AQo,JJ-77,A5s-A2s,KTs+:0.5'.

    Tokens:
      22+        pairs 22..AA            A5s-A2s   connected wheel aces
      AKs / AKo  single combo            KQo+      KQo, AQo, AKo
      AKs+       AKs, AQs                JTs+      JTs, QJs, KQs, AKs
      22-77      pair band               Q9s+      Q9s..QKs? (no: Q9s, QTs, QJs, QKs, QAs)
      suffix :w  weight (e.g. AKs:0.5)   random:w  all combos at weight w
    """
    rng = Range(name=name)
    for token in text.replace(" ", "").split(","):
        token = token.strip()
        if not token:
            continue
        weight = 1.0
        if ":" in token:
            token, w = token.split(":", 1)
            weight = float(w)
        if token.lower() == "random":
            for hi in range(2, 15):
                for lo in range(2, hi + 1):
                    if hi == lo:
                        rng.add(hi, lo, True, weight)
                    else:
                        rng.add(hi, lo, True, weight)
                        rng.add(hi, lo, False, weight)
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            r1 = _parse_single(a)
            r2 = _parse_single(b)
            hi1, lo1, s1 = r1
            hi2, lo2, s2 = r2
            if hi1 == lo1 and hi2 == lo2:                    # pair band 77-22
                for r in range(min(hi1, hi2), max(hi1, hi2) + 1):
                    rng.add(r, r, True, weight)
            elif hi1 == hi2 and s1 == s2 and lo1 > lo2:      # A5s-A2s / KQs-K9s
                for r in range(lo2, lo1 + 1):
                    rng.add(hi1, r, s1, weight)
            elif hi1 == hi2 and s1 == s2 and lo1 < lo2:      # A2s-A5s
                for r in range(lo1, lo2 + 1):
                    rng.add(hi1, r, s1, weight)
            elif lo1 == lo2 and s1 == s2 and hi1 > hi2:      # KQo-ATo
                for r in range(hi2, hi1 + 1):
                    rng.add(r, lo1, s1, weight)
            elif lo1 == lo2 and s1 == s2 and hi1 < hi2:      # ATo-KQo
                for r in range(hi1, hi2 + 1):
                    rng.add(r, lo1, s1, weight)
            elif hi1 > hi2 and lo1 > lo2 and s1 == s2 and (hi1 - lo1) == (hi2 - lo2):
                gap = hi1 - lo1                                # connector chain JTs-76s
                for r in range(hi2, hi1 + 1):
                    rng.add(r, r - gap, s1, weight)
            else:
                raise ValueError(f"cannot parse range token '{token}'")
            continue
        if token.endswith("+"):
            r = _parse_single(token[:-1])
            hi, lo, suited = r
            if hi == lo:
                for rr in range(hi, 15):
                    rng.add(rr, rr, True, weight)
            else:
                for rr in range(hi, 15):
                    rng.add(rr, lo, suited, weight)
            continue
        r = _parse_single(token)
        rng.add(r[0], r[1], r[2], weight)
    return rng


def _parse_single(token: str) -> Combo:
    """Parse 'AKs' | 'KQo' | '76s' | 'AA' | 'T9s' -> (hi, lo, suited)."""
    if len(token) == 2 and token[0] == token[1]:
        r = _rank_index(token[0])
        return combo_key(r, r, True)
    if len(token) == 3:
        r1, r2 = _rank_index(token[0]), _rank_index(token[1])
        if token[2] == "s":
            return combo_key(r1, r2, True)
        if token[2] == "o":
            return combo_key(r1, r2, False)
    raise ValueError(f"cannot parse range token '{token}'")


def range_from_cards(card_pairs: list[tuple[str, str] | list[str]], name: str = "") -> Range:
    """Build a range from concrete hole-card strings (hand-history style)."""
    rng = Range(name=name)
    for pair in card_pairs:
        c1, c2 = parse_single_card(pair[0]), parse_single_card(pair[1])
        hi = max(c1.rank, c2.rank)
        lo = min(c1.rank, c2.rank)
        suited = c1.suit == c2.suit
        rng.add(hi, lo, suited, 1.0)
    return rng


def parse_single_card(text: str) -> Card:
    from .cards import parse_card

    return parse_card(text)
