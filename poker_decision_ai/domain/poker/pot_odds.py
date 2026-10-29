"""Deterministic pot-odds / EV / pressure math (spec §十一: never LLM-guessed)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PotOddsResult:
    pot: float
    to_call: float
    pot_odds_ratio: str
    pot_odds_pct: float          # call amount / final pot
    required_equity: float       # == pot_odds_pct
    mdf: float | None = None     # minimum defense frequency when facing a bet
    implied_odds_factor: float | None = None
    reverse_implied_flag: bool = False
    spr: float | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pot": self.pot, "to_call": self.to_call,
            "pot_odds_ratio": self.pot_odds_ratio, "pot_odds_pct": round(self.pot_odds_pct, 4),
            "required_equity": round(self.required_equity, 4),
            "mdf": None if self.mdf is None else round(self.mdf, 4),
            "implied_odds_factor": self.implied_odds_factor,
            "reverse_implied_flag": self.reverse_implied_flag,
            "spr": None if self.spr is None else round(self.spr, 2),
            "notes": self.notes,
        }


def pot_odds(pot_before_call: float, to_call: float) -> PotOddsResult:
    """pot_odds = to_call / (pot + 2 * to_call); required equity equals it."""
    if to_call <= 0:
        return PotOddsResult(pot=pot_before_call, to_call=0.0, pot_odds_ratio="inf",
                             pot_odds_pct=0.0, required_equity=0.0,
                             notes=["no bet to call"])
    ratio = pot_before_call / to_call
    pct = to_call / (pot_before_call + 2 * to_call)
    return PotOddsResult(
        pot=pot_before_call, to_call=to_call,
        pot_odds_ratio=f"{ratio:.1f}:1", pot_odds_pct=pct, required_equity=pct,
    )


def minimum_defense_frequency(pot_before_bet: float, bet_size: float) -> float:
    """MDF = pot / (pot + bet). Defender must defend >= MDF to make bluffs indifferent."""
    if bet_size <= 0:
        return 1.0
    return pot_before_bet / (pot_before_bet + bet_size)


def spr(effective_stack: float, pot: float) -> float:
    """Stack-to-pot ratio."""
    return effective_stack / pot if pot > 0 else float("inf")


def required_equity_from_pot_odds(pot_before_call: float, to_call: float) -> float:
    return pot_odds(pot_before_call, to_call).required_equity


def implied_odds_factor(draw_outs: int, to_call: float, est_future_win: float) -> float:
    """Simple deterministic implied-odds multiplier: (to_call + future)/(to_call).

    est_future_win is an explicit input (e.g. from stack/pot analysis); the
    system never invents it.  draw_outs is informational (for notes).
    """
    if to_call <= 0:
        return 1.0
    return (to_call + est_future_win) / to_call


def reverse_implied_odds_flag(board_texture: dict) -> bool:
    """Heuristic flag: paired/monotone boards raise reverse implied odds.

    board_texture comes from the Board Agent (deterministic features), not
    from the LLM.
    """
    return bool(board_texture.get("paired") or board_texture.get("monotone")
                or board_texture.get("flush_draw_present"))


def ev_of_call(equity: float, pot_before_call: float, to_call: float) -> float:
    """EV(call) = equity * (pot + to_call) - (1 - equity) * to_call."""
    return equity * (pot_before_call + to_call) - (1 - equity) * to_call


def breakeven_bluff_frequency(bet_size: float, pot_before_bet: float) -> float:
    """Minimum fold frequency a bluff of this size needs to break even."""
    if bet_size <= 0:
        return 0.0
    return bet_size / (pot_before_bet + bet_size)
