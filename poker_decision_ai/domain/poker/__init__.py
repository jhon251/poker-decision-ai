"""Poker domain: deterministic computation engine (no LLM in this layer)."""
from .bounty import (
    BountyInfo, bounty_ev_chips, bounty_pressure,
    bounty_share_of_buyin, bounty_to_chips, call_ev_with_bounty,
    chip_dollar_value, effective_required_equity, knockout_probability,
    normal_required_equity, progressive_bounty_growth,
)
from .cards import Card, full_deck, parse_card, parse_cards, shuffled_deck
from .evaluator import evaluate5, evaluate7, best_five, score_name, CATEGORY_NAMES
from .equity import EquityCalculator, EquityResult
from .hand_history import ParsedHand, parse_hand_history
from .icm import (
    ICMResult, bubble_factor, c_ev_to_d_ev, finish_probabilities, icm, icm_exact,
    icm_monte_carlo, risk_premium, satellite_seat_ev,
)
from .models import (
    STATE_SCHEMA, ActionRecord, ActionType, CashInfo, CashState, GameType,
    PlayerState, PokerStateBase, Street, TournamentInfo, TournamentStage,
    TournamentState, active_players, coerce_state, effective_stack, hero,
    position_of, positions_for, pot_bb, state_to_dict,
)
from .pot_odds import (
    PotOddsResult, breakeven_bluff_frequency, ev_of_call, implied_odds_factor,
    minimum_defense_frequency, pot_odds, reverse_implied_odds_flag, spr,
)
from .ranges import Combo, Range, combo_name, combo_to_cards, parse_range, range_from_cards
from .stats import (
    HAND_RECORD_FIELDS, Metric, StatsBundle, aggression_factor, aggression_frequency,
    bayes_shrink, bb_per_100, cbet, compute_all, fold_to_3bet, fold_to_cbet,
    four_bet, itm, pfr, roi, three_bet, time_decay_weights, vpip,
    wilson_interval, won_at_showdown, wtsd, wwsf,
)
from .state_engine import (
    PokerStateEngine, StateEngineError, StateEngineEvent, events_from_parsed_hand,
)

__all__ = [
    "BountyInfo", "bounty_ev_chips", "bounty_pressure",
    "bounty_share_of_buyin", "bounty_to_chips", "call_ev_with_bounty",
    "chip_dollar_value", "effective_required_equity", "knockout_probability",
    "normal_required_equity", "progressive_bounty_growth",
    "Card", "full_deck", "parse_card", "parse_cards", "shuffled_deck",
    "evaluate5", "evaluate7", "best_five", "score_name", "CATEGORY_NAMES",
    "EquityCalculator", "EquityResult",
    "ParsedHand", "parse_hand_history",
    "ICMResult", "bubble_factor", "c_ev_to_d_ev", "finish_probabilities", "icm",
    "icm_exact", "icm_monte_carlo", "risk_premium", "satellite_seat_ev",
    "STATE_SCHEMA", "ActionRecord", "ActionType", "CashInfo", "CashState",
    "GameType", "PlayerState", "PokerStateBase", "Street", "TournamentInfo",
    "TournamentStage", "TournamentState", "active_players", "coerce_state",
    "effective_stack", "hero", "position_of", "positions_for", "pot_bb",
    "state_to_dict",
    "PotOddsResult", "breakeven_bluff_frequency", "ev_of_call",
    "implied_odds_factor", "minimum_defense_frequency", "pot_odds",
    "reverse_implied_odds_flag", "spr",
    "Combo", "Range", "combo_name", "combo_to_cards", "parse_range", "range_from_cards",
    "HAND_RECORD_FIELDS", "Metric", "StatsBundle", "aggression_factor",
    "aggression_frequency", "bayes_shrink", "bb_per_100", "cbet", "compute_all", "fold_to_3bet",
    "fold_to_cbet", "four_bet", "itm", "pfr", "roi", "three_bet",
    "time_decay_weights", "vpip", "wilson_interval", "won_at_showdown",
    "wtsd", "wwsf",
    "PokerStateEngine", "StateEngineError", "StateEngineEvent", "events_from_parsed_hand",
]
