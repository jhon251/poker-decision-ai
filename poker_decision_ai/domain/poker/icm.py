"""Deterministic ICM / tournament equity / satellite seat EV.

Exact ICM for small fields (<= max_exact players, memoized recursion over
finish orders); Monte Carlo ICM otherwise.  Bubble factor and risk premium
are derived deterministically from the payout structure (spec §十一).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable


@dataclass
class ICMResult:
    ev: dict                      # player_id -> tournament equity ($ or %)
    method: str = "exact"         # exact | monte_carlo
    samples: int = 0
    prize_pool: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ev": {k: round(v, 6) for k, v in self.ev.items()},
            "method": self.method, "samples": self.samples,
            "prize_pool": round(self.prize_pool, 2), "details": self.details,
        }


def _check_inputs(stacks: dict, payouts: list) -> None:
    if not stacks:
        raise ValueError("stacks is empty")
    if any(s < 0 for s in stacks.values()):
        raise ValueError("negative stack")
    if not payouts:
        raise ValueError("payouts is empty")
    # 不足人数的名次由 icm_exact / icm_monte_carlo 内部补零（此函数只做校验）


def icm_exact(stacks: dict, payouts: list) -> dict:
    """Exact ICM via memoized top-down recursion over finish orders (n <= ~9).

    The next player to finish (winner of the current sub-race) is drawn with
    probability proportional to stack; the winner takes pay[n - len(remaining)].
    """
    _check_inputs(stacks, payouts)
    n = len(stacks)
    pay = list(payouts) + [0.0] * (n - len(payouts))

    def rec(remaining: tuple) -> dict:
        if len(remaining) == 1:
            return {remaining[0]: pay[n - 1]}       # last survivor finishes 1st
        total = sum(stacks[p] for p in remaining)
        out: dict = {p: 0.0 for p in remaining}
        for w in remaining:                     # w finishes next among `remaining`
            if stacks[w] == 0:
                continue
            prob = stacks[w] / total
            sub = rec(tuple(x for x in remaining if x != w))
            out[w] += prob * pay[n - len(remaining)]      # position of next finish
            for q in sub:
                out[q] += prob * sub[q]
        return out

    players = tuple(sorted(stacks, key=lambda p: (stacks[p], p)))
    return rec(players)


def icm_monte_carlo(stacks: dict, payouts: list, samples: int = 20000, seed: int = 42) -> dict:
    """Monte Carlo ICM: random weighted finish orders."""
    _check_inputs(stacks, payouts)
    n = len(stacks)
    pay = list(payouts) + [0.0] * (n - len(payouts))
    rng = random.Random(seed)
    acc = {p: 0.0 for p in stacks}
    for _ in range(samples):
        remaining = list(stacks)
        for pos in range(n):
            total = sum(stacks[p] for p in remaining)
            x = rng.random() * total
            acc_s = 0.0
            pick = remaining[-1]
            for p in remaining:
                acc_s += stacks[p]
                if x <= acc_s:
                    pick = p
                    break
            acc[pick] += pay[pos]
            remaining.remove(pick)
    return {p: v / samples for p, v in acc.items()}


def icm(stacks: dict, payouts: list, max_exact: int = 9, samples: int = 20000,
        seed: int = 42) -> ICMResult:
    _check_inputs(stacks, payouts)
    n = len(stacks)
    if n <= max_exact:
        ev = icm_exact(stacks, payouts)
        return ICMResult(ev=ev, method="exact", samples=1, prize_pool=float(sum(payouts)))
    ev = icm_monte_carlo(stacks, payouts, samples=samples, seed=seed)
    return ICMResult(ev=ev, method="monte_carlo", samples=samples,
                     prize_pool=float(sum(payouts)), details={"seed": seed})


def bubble_factor(stacks: dict, payouts: list, hero: str, villain: str,
                  step: float | None = None, max_exact: int = 9) -> float:
    """Bubble factor: how much losing a chip to `villain` hurts vs winning one.

    BF = (EV(hero) - EV(hero - step, villain + step)) /
         (EV(hero + step, villain - step) - EV(hero))
    BF > 1 means survival pressure (chips lost cost more than chips won gain).
    """
    if hero not in stacks or villain not in stacks or hero == villain:
        raise ValueError("bubble_factor requires two distinct players in stacks")
    if step is None:
        step = max(1.0, min(stacks[hero], stacks[villain]) * 0.01)
    base = icm(stacks, payouts, max_exact=max_exact).ev[hero]
    lose = dict(stacks)
    lose[hero] -= step
    lose[villain] += step
    ev_lose = icm(lose, payouts, max_exact=max_exact).ev[hero]
    win = dict(stacks)
    win[hero] += step
    win[villain] -= step
    ev_win = icm(win, payouts, max_exact=max_exact).ev[hero]
    loss_delta = base - ev_lose
    win_delta = ev_win - base
    if win_delta <= 0:
        return 1.0
    return loss_delta / win_delta


def risk_premium(bubble_factor_value: float) -> float:
    """Required-equity premium (additive) implied by a bubble factor.

    premium = (BF - 1) / BF  -> e.g. BF=2 -> +0.5 required equity.
    This is the standard linear approximation; document it wherever used.
    """
    if bubble_factor_value <= 1:
        return 0.0
    return (bubble_factor_value - 1) / bubble_factor_value


def c_ev_to_d_ev(chip_ev: float, bubble_factor_value: float) -> float:
    """Convert chip EV to $-EV using bubble factor (linear approximation)."""
    return chip_ev / bubble_factor_value if bubble_factor_value > 0 else chip_ev


def satellite_seat_ev(stacks: dict, seats: int, max_exact: int = 9,
                      samples: int = 20000, seed: int = 42) -> ICMResult:
    """Satellite equity: each ticket is worth 1 unit, paid to the top `seats`.

    This is the ICM model applied to satellite payouts [1]*seats - the correct
    satellite objective (maximize seat probability, NOT chip EV).
    """
    if seats <= 0:
        raise ValueError(f"seats must be >= 1, got {seats}")
    payouts = [1.0] * seats
    result = icm(stacks, payouts, max_exact=max_exact, samples=samples, seed=seed)
    result.details["satellite_seats"] = seats
    return result


def finish_probabilities(stacks: dict, max_exact: int = 9, samples: int = 20000,
                         seed: int = 42) -> dict:
    """P(player, position) matrix via finish-order sampling (deterministic seed)."""
    n = len(stacks)
    pay_for_pos = [0.0] * n
    out = {p: [0.0] * n for p in stacks}
    if n <= max_exact:
        # exact: run icm n times with degenerate payout marking position k
        for pos in range(n):
            pay = [0.0] * n
            pay[pos] = 1.0
            ev = icm_exact(stacks, pay)
            for p in stacks:
                out[p][pos] = ev[p]
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            remaining = list(stacks)
            for pos in range(n):
                total = sum(stacks[p] for p in remaining)
                x = rng.random() * total
                acc_s = 0.0
                pick = remaining[-1]
                for p in remaining:
                    acc_s += stacks[p]
                    if x <= acc_s:
                        pick = p
                        break
                out[pick][pos] += 1
                remaining.remove(pick)
        for p in stacks:
            out[p] = [v / samples for v in out[p]]
    return out
