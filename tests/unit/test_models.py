"""Unit tests for poker domain models (typed state coercion etc.)."""
from __future__ import annotations

import pytest

from poker_decision_ai.domain.poker import (
    CashState,
    TournamentState,
    coerce_state,
    effective_stack,
    hero,
    satellite_seat_ev,
)


def _state(game_type: str) -> dict:
    return {
        "hand_id": "h1", "game_type": game_type, "street": "flop",
        "board": ["Qh", "7d", "2c"], "pot": 10.1, "current_bet": 3.75,
        "to_call": 3.75, "min_raise_to": 7.5, "sb": 0.1, "bb": 0.25, "ante": 0.0,
        "players": [
            {"player_id": "Bob", "name": "Bob", "seat": 2, "stack": 31.25, "status": "active",
             "hole_cards": [], "position": "BB", "is_hero": False},
            {"player_id": "Hero", "name": "Hero", "seat": 3, "stack": 24.38, "status": "active",
             "hole_cards": ["As", "Kd"], "position": "BTN", "is_hero": True},
        ],
        "button_seat": 3, "hero_id": "Hero", "hero_seat": 3,
        "action_history": [], "street_actions": {}, "available_actions": ["fold", "call", "raise"],
        "last_aggressor": "Bob", "rake": 0.0, "source": "hand_history",
        "cash": None, "tournament": None,
    }


def test_coerce_cash_state_with_null_tournament():
    """STATE_SCHEMA allows `tournament: null`; coerce must accept it."""
    s = _state("cash")
    s["cash"] = {"stakes": "NL10", "sb": 0.1, "bb": 0.25, "currency": "USD", "rake_percent": 5.0}
    st = coerce_state(s)
    assert isinstance(st, CashState)
    assert st.cash.stakes == "NL10"
    assert st.hero_id == "Hero"
    assert hero(st).name == "Hero"
    assert effective_stack(st) == 24.38


def test_coerce_tournament_state():
    s = _state("mtt")
    s["tournament"] = {"tournament_id": "T1", "name": "T", "total_players": 100,
                       "players_remaining": 45, "total_chips": 100000.0,
                       "payouts": [50.0, 30.0, 20.0], "next_payout": 2.0,
                       "itm_players": 10, "satellite_seats": None,
                       "stage": "bubble", "avg_stack": 2222.0}
    st = coerce_state(s)
    assert isinstance(st, TournamentState)
    assert st.tournament.players_remaining == 45


def test_coerce_rejects_bad_input():
    from poker_decision_ai.core.errors import SchemaError

    with pytest.raises(SchemaError):
        coerce_state("not a dict")
    with pytest.raises(SchemaError):
        coerce_state({"game_type": "cash", "unknown_field": 1, "hand_id": "h", "street": "preflop"})


def test_satellite_seat_ev_guards_seats():
    with pytest.raises(ValueError):
        satellite_seat_ev({"A": 100.0, "B": 50.0}, seats=0)
