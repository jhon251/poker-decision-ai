"""Equity Agent (胜率): deterministic hand-vs-range equity (never LLM-guessed).

Uses `EquityCalculator` from domain/poker:
  - hero hole cards vs each active villain's range (from the Range Agent's
    distribution output, rebuilt into a Range);
  - fallback to a full random range when no range input is supplied (+warning);
  - river (5 board cards) -> exact enumeration over the villain range;
  - otherwise Monte Carlo (seeded from config for reproducibility).

Pot odds / EV / implied odds come from domain/poker/pot_odds.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import (
    Card,
    EquityCalculator,
    Range,
    ev_of_call,
    evaluate7,
    implied_odds_factor,
    parse_card,
    parse_range,
    pot_odds,
    reverse_implied_odds_flag,
)


@dataclass
class EquityOutput:
    hero_equity: float = 0.0
    hero_equity_vs_each: dict = field(default_factory=dict)   # pid -> hero equity
    villains: dict = field(default_factory=dict)              # pid -> {equity, method, samples}
    pot_odds: dict = field(default_factory=dict)              # {pot_odds_pct, required_equity, ratio}
    ev_call_bb: float = 0.0
    ev_raise_est_bb: float | None = None
    implied_odds: dict = field(default_factory=dict)          # {factor, note}
    reverse_implied: bool = False
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # P30-5-D：范围来源诚实标记（random fallback 不得冒充真实 range）
    range_mode: str = "APPROXIMATE"      # REAL | APPROXIMATE | RANDOM
    # P42-C1：结果分布（win/lose/tie 概率；来自同一 seeded MC 的样本计数，
    # hero_equity = win + tie/2 严格一致；不可用时为 {} —— 不伪造分布）
    outcome: dict = field(default_factory=dict)
    # P30-9-BO：弃牌权益（EV(raise) 用到的 fold_pct 及其来源，可审计）
    fold_equity: dict = field(default_factory=dict)


def _resolve_fold_pct(*, ctx: Any, state: dict, range_in: dict,
                      hero_hole: list, board: list, pot_bb: float,
                      to_call_bb: float, current_bet_bb: float,
                      raise_to_bb: float) -> tuple[float, dict]:
    """EV(raise) 里的 fold_pct：按配置取"估计值"或"假设值"。

    * `equity.fold_equity_source`：`assumption`（默认，0.5）| `estimator`
    * 估计不可用 ⇒ 回退 `equity.fold_pct_default`（默认 0.5）并**标注来源**。
    """
    default = 0.5
    try:
        default = float(ctx.config.root.get("equity.fold_pct_default", 0.5))
        src = str(ctx.config.root.get("equity.fold_equity_source", "assumption"))
    except Exception:                                          # noqa: BLE001
        src = "assumption"
    if src != "estimator":
        return default, {"p_fold": default, "source": "assumption",
                          "mode": "assumption", "note": "配置为 assumption（中性 0.5）"}
    try:
        from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
        vmap = (range_in or {}).get("villains") or {}
        dist = None
        for p in (state.get("players") or []):
            if not isinstance(p, dict) or p.get("is_hero"):
                continue
            if p.get("status") not in ("active", "allin"):
                continue
            info = vmap.get(p.get("player_id")) or {}
            if isinstance(info, dict) and info.get("distribution"):
                dist = info["distribution"]
                break
        facing = "raise" if to_call_bb > 0 else "bet"
        est = FoldEstimator(pot_bb=pot_bb, facing=facing,
                             street=str(state.get("street") or "flop"),
                             villain_dist=dist, hero_cards=hero_hole,
                             board=board)
        call_bb = max(0.0, raise_to_bb - current_bet_bb)
        res = est.for_size(call_bb if call_bb > 0 else raise_to_bb)
        if getattr(res, "available", False) and res.p_fold is not None:
            return float(res.p_fold), {
                "p_fold": round(float(res.p_fold), 4), "source": "estimator",
                "mode": res.mode, "provenance": res.provenance,
                "villain_call_bb": round(call_bb, 4)}
        reason = getattr(res, "unavailable_reason", "") or "estimator unavailable"
        return default, {"p_fold": default, "source": "config_default",
                          "mode": "fallback",
                          "note": f"估计不可用（{reason}）⇒ 回退默认值",
                          "warnings": [{"code": "FOLD_EQUITY_UNAVAILABLE",
                                         "message": reason}]}
    except Exception as e:                                     # noqa: BLE001
        return default, {"p_fold": default, "source": "config_default",
                          "mode": "fallback", "note": f"估计异常：{e}"}


def _find_hero(players: list) -> dict | None:
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _effective_stack(state: dict) -> float:
    players = state.get("players", []) or []
    stacks: list[float] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        if p.get("is_hero") or p.get("status") in ("active", "allin"):
            stacks.append(float(p.get("stack", 0.0) or 0.0))
    return min(stacks) if stacks else 0.0


def _range_from_distribution(dist: dict) -> Range:
    """Rebuild a Range from the Range Agent's `distribution` ({cat: weight}).

    Category names match `combo_name` (e.g. "AA", "AKs", "T9o"), so each is
    parseable back into a single combo via parse_range(cat).
    """
    r = Range(name="from_distribution")
    for cat, w in dist.items():
        try:
            sub = parse_range(str(cat))
        except (ValueError, Exception):  # noqa: BLE001 - ignore unparseable cats
            continue
        for (hi, lo, suited), _ in sub.weights.items():
            r.add(hi, lo, suited, float(w))
    return r


def _concrete_pairs(hi: int, lo: int, suited: bool) -> list[tuple[Card, Card]]:
    """All concrete card pairs for a combo category (mirrors ranges.py logic)."""
    if hi == lo:
        suits = list(range(4))
        return [(Card((hi - 2) * 4 + s1), Card((hi - 2) * 4 + s2))
                for i, s1 in enumerate(suits) for s2 in suits[i + 1:]]
    if suited:
        return [(Card((hi - 2) * 4 + s), Card((lo - 2) * 4 + s)) for s in range(4)]
    return [(Card((hi - 2) * 4 + s1), Card((lo - 2) * 4 + s2))
            for s1 in range(4) for s2 in range(4) if s1 != s2]


def _exact_vs_range(hero_cards: list[Card], board_cards: list[Card], vr: Range
                    ) -> tuple[float, int, str, float | None, float | None]:
    """Exact river equity: enumerate every non-colliding concrete combo.

    For each category with weight w and `available` non-colliding concrete
    pairs, the category contributes w * (wins + 0.5*ties)/available to the
    numerator and w to the denominator (each available combo is equally likely
    given the category).  P42: also returns weighted win/tie fractions
    (None when the range is empty — honest unavailable).
    """
    dead = {c.code for c in hero_cards + board_cards}
    hs = evaluate7(hero_cards + board_cards)
    total_w = 0.0
    win_w = 0.0
    win_raw = 0.0
    tie_raw = 0.0
    count = 0
    for (hi, lo, suited), w in vr.weights.items():
        pairs = [(c1, c2) for c1, c2 in _concrete_pairs(hi, lo, suited)
                 if c1.code not in dead and c2.code not in dead]
        if not pairs:
            continue
        wins = ties = 0
        for c1, c2 in pairs:
            vs = evaluate7([c1, c2] + board_cards)
            if hs > vs:
                wins += 1
            elif hs == vs:
                ties += 1
        count += len(pairs)
        total_w += w
        win_w += w * (wins + 0.5 * ties) / len(pairs)
        win_raw += w * wins / len(pairs)
        tie_raw += w * ties / len(pairs)
    if total_w <= 0:
        return 0.5, 0, "exact", None, None
    return (win_w / total_w, count, "exact",
            win_raw / total_w, tie_raw / total_w)


def _draw_structure(board_cards: list[Card]) -> list[str]:
    """Minimal local draw detection (flush / straight draws) from the board."""
    out: list[str] = []
    ranks = [c.rank for c in board_cards]
    suits = [c.suit for c in board_cards]
    sc = {s: suits.count(s) for s in set(suits)}
    if len(board_cards) == 3 and max(sc.values()) >= 2:
        out.append("flush_draw")
    rset = set(ranks)
    windows = 0
    for high in range(5, 15):
        needed = {r for r in range(high - 4, high + 1)}
        if len(board_cards) == 3 and len(rset & needed) >= 3:
            windows += 1
        if len(board_cards) >= 4 and len(rset & needed) >= 4:
            windows += 1
    if windows:
        out.append("straight_draw")
    return out


def _board_texture(board_cards: list[Card]) -> dict:
    ranks = [c.rank for c in board_cards]
    suits = [c.suit for c in board_cards]
    sc = {s: suits.count(s) for s in set(suits)}
    max_suit = max(sc.values()) if sc else 0
    return {
        "paired": len(set(ranks)) < len(ranks),
        "monotone": max_suit >= 3 and len(board_cards) >= 3,
        "flush_draw_present": (len(board_cards) == 3 and max_suit >= 2)
                               or (max_suit >= 3 and len(board_cards) >= 3),
    }


class EquityAgent(BaseAgent):
    name: ClassVar[str] = "equity"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "胜率 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = EquityOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for equity calculation"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> EquityOutput:
        state = inp.get("state") or {}
        range_in = inp.get("range") or {}
        out = EquityOutput()
        warnings: list[dict] = []

        eq_cfg = ctx.config.equity_config()
        iterations = int(eq_cfg.get("iterations", 20000))
        seed = int(eq_cfg.get("seed", 42))
        exact_limit = int(eq_cfg.get("exact_limit", 400000))
        # V2 Realtime 快路径：低迭代覆盖（ctx.extra["realtime"]["equity_iterations"]）
        realtime = (ctx.extra or {}).get("realtime") or {}
        if realtime.get("equity_iterations"):
            iterations = int(realtime["equity_iterations"])
        calc = EquityCalculator(iterations=iterations, seed=seed, exact_limit=exact_limit)

        players = state.get("players", []) or []
        hero_p = _find_hero(players)
        if not hero_p:
            warnings.append({"code": "NO_HERO", "message": "hero not found in state.players"})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        hero_hole = [str(c) for c in (hero_p.get("hole_cards") or [])]
        board = [str(c) for c in state.get("board", []) or []]
        if not hero_hole or len(hero_hole) < 2:
            warnings.append({"code": "NO_HOLE_CARDS", "message": "hero hole cards missing; "
                                                                 "equity undefined"})
            out.warnings = warnings
            out.confidence = 0.0
            out.hero_equity = 0.0
            return out

        try:
            hero_cards = [parse_card(c) for c in hero_hole]
            board_cards = [parse_card(c) for c in board]
        except ValueError as e:
            warnings.append({"code": "INVALID_CARD", "message": str(e)})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        bb = float(state.get("bb") or 0.0) or 1.0
        pot = float(state.get("pot") or 0.0)
        to_call = float(state.get("to_call") or 0.0)
        pot_bb = pot / bb
        to_call_bb = to_call / bb

        active_villains = [p for p in players
                           if isinstance(p, dict) and not p.get("is_hero")
                           and p.get("status") in ("active", "allin")]
        if not active_villains:
            warnings.append({"code": "NO_VILLAINS", "message": "no active villains in state"})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        villain_range_map = (range_in or {}).get("villains") or {}
        used_range = bool(villain_range_map)
        warned_no_range = False

        for v in active_villains:
            pid = str(v.get("player_id", ""))
            if not pid:
                continue
            vinfo = villain_range_map.get(pid) or {}
            dist = vinfo.get("distribution")
            if isinstance(dist, dict) and dist:
                vr = _range_from_distribution(dist)
            else:
                if not warned_no_range and not used_range:
                    warnings.append({"code": "NO_RANGE",
                                     "message": "no range input; using full random range for villains"})
                    warned_no_range = True
                vr = parse_range("random:1.0", name="random")
                out.range_mode = "RANDOM"      # P30-5-D：诚实标记随机退化
            if vr.is_empty():
                warnings.append({"code": "EMPTY_RANGE", "message": f"villain {pid} range is empty"})
                continue

            # ---- V2 EquityCache：命中则跳过 MC（键含范围签名/迭代数/种子） ----
            from poker_decision_ai.core.cache import digest

            eq_cache = getattr(ctx, "caches", None)
            range_sig = digest(vr.weights) if vr.weights else "empty"
            cache_key = None
            if eq_cache is not None and len(board_cards) < 5:
                cache_key = type(eq_cache.equity).key(
                    [str(c) for c in hero_cards], [str(c) for c in board_cards],
                    range_sig, iterations, seed)
                cached = eq_cache.equity.get(cache_key)
                if cached is not None:
                    eq = float(cached.get("hero_equity", 0.5))
                    samples = int(cached.get("samples", iterations))
                    method = str(cached.get("method", "monte_carlo"))
                    out.hero_equity_vs_each[pid] = round(eq, 4)
                    ventry = {"equity": round(eq, 4), "method": method,
                              "samples": samples, "cached": True}
                    # P42：缓存命中带分布时透传（旧缓存无 win/tie → 诚实缺失）
                    if cached.get("win") is not None:
                        ventry["win"] = float(cached["win"])
                        ventry["tie"] = float(cached.get("tie") or 0.0)
                    out.villains[pid] = ventry
                    continue

            if len(board_cards) == 5:
                eq, samples, method, win_f, tie_f = _exact_vs_range(
                    hero_cards, board_cards, vr)
            else:
                res = calc.hand_vs_range(hero_cards, board_cards, vr,
                                         iterations=iterations, seed=seed)
                eq, samples, method = res.hero_equity, res.samples, res.method
                # P42-C1：同一 seeded MC 的样本计数 → win/tie 分布（零额外计算）
                if res.samples > 0:
                    win_f = res.win_samples / res.samples
                    tie_f = res.tie_samples / res.samples
                else:
                    win_f = tie_f = None

            if eq_cache is not None and cache_key is not None:
                eq_cache.equity.put(cache_key, {
                    "hero_equity": eq, "samples": samples, "method": method,
                    "win": win_f, "tie": tie_f,
                })

            out.hero_equity_vs_each[pid] = round(eq, 4)
            ventry = {"equity": round(eq, 4), "method": method, "samples": int(samples)}
            if win_f is not None:
                ventry["win"] = round(win_f, 6)
                ventry["tie"] = round(tie_f, 6)
            out.villains[pid] = ventry

        # ---- aggregate hero equity: weighted mean across villains ----
        if out.hero_equity_vs_each:
            hero_equity_agg = sum(out.hero_equity_vs_each.values()) / len(out.hero_equity_vs_each)
        else:
            hero_equity_agg = 0.0
        out.hero_equity = round(hero_equity_agg, 4)

        # ---- P42-C1: aggregate outcome distribution (mean across villains
        #      that carry win/tie; hero_equity = win + tie/2 严格一致) ----
        win_vals = [v["win"] for v in out.villains.values() if "win" in v]
        tie_vals = [v["tie"] for v in out.villains.values() if "tie" in v]
        if win_vals and tie_vals and len(win_vals) == len(tie_vals):
            win_p = sum(win_vals) / len(win_vals)
            tie_p = sum(tie_vals) / len(tie_vals)
            win_p = min(max(win_p, 0.0), 1.0)
            tie_p = min(max(tie_p, 0.0), 1.0 - win_p)
            out.outcome = {
                "win": round(win_p, 6),
                "tie": round(tie_p, 6),
                "lose": round(1.0 - win_p - tie_p, 6),
                "samples": int(samples) if out.villains else 0,
                "seed": int(seed),
                "method": method,
            }

        # ---- pot odds (domain deterministic) ----
        po = pot_odds(pot, to_call)
        out.pot_odds = {"pot_odds_pct": round(po.pot_odds_pct, 4),
                        "required_equity": round(po.required_equity, 4),
                        "ratio": po.pot_odds_ratio}

        # ---- EV(call) = equity*(pot+to_call) - (1-equity)*to_call (in bb) ----
        out.ev_call_bb = round(ev_of_call(hero_equity_agg, pot_bb, to_call_bb), 4)

        # ---- simplified EV of raising to 2x current bet ----
        # Assumption: villain folds with prob fold_pct (population fold rate; the
        # equity contract has no population input, so we use a neutral 0.5) and
        # otherwise calls and we realize equity against raise_to added to the pot.
        current_bet_bb = float(state.get("current_bet") or 0.0) / bb
        raise_to_bb = 2.0 * current_bet_bb if current_bet_bb > 0 else 0.5 * pot_bb
        # P30-9-BO：fold_pct 由配置决定来源（默认 assumption=0.5 ⇒ 行为与旧版一致）
        fold_pct, fold_info = _resolve_fold_pct(
            ctx=ctx, state=state, range_in=range_in, hero_hole=hero_hole,
            board=board, pot_bb=pot_bb, to_call_bb=to_call_bb,
            current_bet_bb=current_bet_bb, raise_to_bb=raise_to_bb)
        out.fold_equity = fold_info
        if fold_info.get("warnings"):
            warnings.extend(fold_info["warnings"])
        # EV(raise) = fold% * pot + (1-fold%) * [equity*(pot + 2*raise_to) - raise_to]
        ev_raise = fold_pct * pot_bb + (1 - fold_pct) * (
            hero_equity_agg * (pot_bb + 2 * raise_to_bb) - raise_to_bb)
        out.ev_raise_est_bb = round(ev_raise, 4)

        # ---- implied odds: flush/straight draw + deep stack ----
        draw = _draw_structure(board_cards)
        eff_stack = _effective_stack(state)
        deep = (eff_stack / bb) > 40
        if ("flush_draw" in draw or "straight_draw" in draw) and deep and to_call > 0:
            factor = implied_odds_factor(9, to_call_bb, eff_stack / bb * 0.3)
            out.implied_odds = {"factor": round(factor, 4),
                                "note": f"draw present {draw}; deep stack; "
                                        f"est_future = 30% of effective stack"}
        else:
            out.implied_odds = {"factor": None,
                                "note": "no strong draw or not deep enough for implied odds"}

        # ---- reverse implied odds (paired/monotone/flush-draw boards) ----
        out.reverse_implied = reverse_implied_odds_flag(_board_texture(board_cards))

        conf = 0.5
        if used_range:
            conf = 0.8
        if len(board_cards) == 5:
            conf = min(1.0, conf + 0.1)
        out.confidence = round(conf, 3)

        out.evidence = [{"kind": "equity",
                         "text": f"hero {hero_hole} vs {len(out.hero_equity_vs_each)} villain(s) "
                                 f"on {board}: hero_equity={out.hero_equity}",
                         "source": "equity", "weight": 1.0}]
        out.warnings = warnings
        return out
