"""Card representation and deck utilities (deterministic, no LLM)."""
from __future__ import annotations

import random
from dataclasses import dataclass

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUES = {r: i + 2 for i, r in enumerate(RANKS)}          # '2'->2 ... 'A'->14
VALUE_RANKS = {v: r for r, v in RANK_VALUES.items()}


class Card:
    """Immutable card as an int code 0..51: (rank-2) * 4 + suit."""

    __slots__ = ("_code",)

    def __init__(self, code: int):
        self._code = int(code)

    @property
    def code(self) -> int:
        return self._code

    @property
    def rank(self) -> int:
        return (self._code >> 2) + 2

    @property
    def suit(self) -> int:
        return self._code & 3

    @property
    def rank_char(self) -> str:
        return VALUE_RANKS[self.rank]

    @property
    def suit_char(self) -> str:
        return SUITS[self.suit]

    def __str__(self) -> str:
        return f"{self.rank_char}{self.suit_char}"

    def __repr__(self) -> str:
        return f"Card('{self}')"

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self._code == other._code

    def __hash__(self) -> int:
        return hash(self._code)

    def __lt__(self, other: "Card") -> bool:
        return self._code < other._code


def parse_card(text: str) -> Card:
    """Parse 'As', 'Th', '2d' ... case-insensitive."""
    t = text.strip()
    if len(t) != 2:
        raise ValueError(f"invalid card '{text}'")
    r, s = t[0].upper(), t[1].lower()
    if r not in RANK_VALUES or s not in SUITS:
        raise ValueError(f"invalid card '{text}'")
    return Card((RANK_VALUES[r] - 2) * 4 + SUITS.index(s))


def parse_cards(texts: list[str]) -> list[Card]:
    return [parse_card(t) for t in texts]


def format_cards(cards: list[Card] | list[int]) -> list[str]:
    return [str(c if isinstance(c, Card) else Card(c)) for c in cards]


def full_deck() -> list[Card]:
    return [Card(i) for i in range(52)]


def shuffled_deck(seed: int | None = None) -> list[Card]:
    deck = full_deck()
    random.Random(seed).shuffle(deck)
    return deck


def remaining_deck(dead: list[Card]) -> list[Card]:
    dead_set = {c.code for c in dead}
    return [Card(i) for i in range(52) if i not in dead_set]
