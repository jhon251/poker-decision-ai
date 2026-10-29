"""Range Agent (范围分析): reconstruct + adjust preflop ranges for every villain.

Deterministic core (spec §2): starting ranges come from `ctx.config.default_ranges()`
(configs/ranges.yaml), never from an LLM.  For each active villain we:
  1. replay `action_history` to detect their line (open / 3bet / 4bet / cold_call
     / cbet / river_value);
  2. pick the base range string by position;
  3. `parse_range(...)` then `remove_cards(hero hole + board)` for card removal;
  4. adjust by `opponent.feature_vector` (tightness / bluff tendency) with the
     formulas documented inline, falling back to population stats when the
     individual vector is absent.

Output follows contract §4 (range): villains / hero_range / positions +
confidence / evidence / warnings meta fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker import Range, parse_card, parse_range
from poker_decision_ai.domain.poker.range_posterior import (
    RangePosteriorIndex, RangePosteriorResult, build_range_posterior)


@dataclass
class RangeOutput:
    villains: dict = field(default_factory=dict)    # pid -> villain range report
    hero_range: dict = field(default_factory=dict)  # hero's own range report
    positions: dict = field(default_factory=dict)   # position -> base range name
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # P30-5-D：范围来源诚实标记（不冒充真实 population/GTO range）
    range_mode: str = "BASELINE"        # REAL | APPROXIMATE | BASELINE
    range_source: str = ""              # 来源 provenance（base/population/individual/random）


def _find_hero(players: list) -> dict | None:
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _position_of(state: dict, pid: str) -> str:
    """Position label: player dict field first, else derive from button seat."""
    for p in state.get("players", []):
        if isinstance(p, dict) and p.get("player_id") == pid:
            return str(p.get("position") or "")
    return ""


def _detect_route(state: dict, pid: str) -> str:
    """Replay action_history to classify a player's line.

    Preflop raising order decides open (1st raise) / three_bet (2nd) /
    four_bet (3rd+); a preflop call facing a raise is a cold_call.  A preflop
    aggressor who keeps betting postflop is further refined to cbet (any
    non-river street) or river_value (river).
    """
    ah = state.get("action_history", []) or []
    street = str(state.get("street", "preflop"))
    pf = [a for a in ah if a.get("street") == "preflop"]
    raises = [a for a in pf if a.get("action") in ("raise", "bet")]
    own_raises = [a for a in raises if a.get("player_id") == pid]

    if own_raises:
        idx = raises.index(own_raises[0])  # identical dict objects, index works
        if idx == 0:
            route = "open"
        elif idx == 1:
            route = "three_bet"
        else:
            route = "four_bet"

        # postflop continuation refinement
        post = [a for a in ah if a.get("player_id") == pid and a.get("street") != "preflop"]
        bets = [a for a in post if a.get("action") in ("bet", "raise")]
        if route == "open" and bets:
            route = "river_value" if street == "river" else "cbet"
        return route

    # no raise of our own: a preflop call facing a raise is a cold call
    own_calls = [a for a in pf if a.get("player_id") == pid and a.get("action") == "call"]
    if own_calls and raises:
        return "cold_call"
    return "open"


def _base_range_string(defaults: dict, route: str, pos: str) -> str | None:
    """Resolve the base range string for a route (+position when applicable)."""
    if route == "open":
        open_map = defaults.get("open") or {}
        if isinstance(open_map, dict) and pos in open_map:
            return open_map[pos]
        # unknown position -> fall back to the most generic open range
        return open_map.get("BB") or open_map.get("BTN") or open_map.get("CO")
    section = defaults.get(route)
    if isinstance(section, dict):
        return section.get("default")
    return section if isinstance(section, str) else None


def _feature_vector(opponent: dict | None, pid: str) -> dict:
    players = (opponent or {}).get("players") or {}
    p = players.get(pid)
    fv = p.get("feature_vector") if isinstance(p, dict) else None
    return fv if isinstance(fv, dict) else {}


def _population_tightness_proxy(population: dict | None) -> float | None:
    """Loose population tightness proxy: tightness ~ 1 - VPIP.

    Only used when the individual feature_vector has no tightness.  It is an
    approximation (VPIP is a hand-selection stat, not a tightness measure) and
    is recorded as such in `assumptions`.
    """
    stats = (population or {}).get("stats") or {}
    m = stats.get("vpip")
    if isinstance(m, dict) and int(m.get("samples", 0) or 0) >= 30:
        return 1.0 - float(m.get("value", 0.0) or 0.0)
    return None


def _adjust_range(vr: Range, tightness: float | None, bluff_tendency: float | None,
                  street: str, dead_cards: list) -> tuple[Range, list[str]]:
    """Deterministic range adjustment by feature vector.  Returns assumptions.

    - tightness > 0.65 : tighten - marginal combos (weight < mean weight) * 0.5
    - tightness < 0.35 : loosen  - marginal combos (weight < mean weight) * 1.5
      ("marginal" = a combo whose weight is below the range's mean weight, i.e.
       non-premium combos under a roughly uniform base range).
    - bluff_tendency > 0.6 and street == river: add air combos at 0.2 weight
      from a full random range (card-removed).
    """
    assumptions: list[str] = []
    if tightness is not None:
        total = vr.total_weight()
        mean_w = total / max(1, len(vr.weights))
        if tightness > 0.65:
            for combo, w in list(vr.weights.items()):
                if w < mean_w:
                    vr.weights[combo] = w * 0.5
            assumptions.append("tightness>0.65: marginal combos (weight<mean) *0.5")
        elif tightness < 0.35:
            for combo, w in list(vr.weights.items()):
                if w < mean_w:
                    vr.weights[combo] = w * 1.5
            assumptions.append("tightness<0.35: marginal combos (weight<mean) *1.5")

    if bluff_tendency is not None and bluff_tendency > 0.6 and street == "river":
        air = parse_range("random:1.0", name="air").remove_cards(dead_cards)
        added = 0
        for combo, w in air.weights.items():
            if combo not in vr.weights:
                vr.weights[combo] = w * 0.2
                added += 1
        if added:
            assumptions.append(f"bluff_tendency>0.6 at river: {added} air combos *0.2 added")
    return vr, assumptions


def _rpi_from(d: Any) -> RangePosteriorIndex | None:
    """profile 字典 → RangePosteriorIndex（空/非法 → None）。"""
    if isinstance(d, dict) and d.get("counts"):
        return RangePosteriorIndex.from_dict(d)
    return None


class RangeAgent(BaseAgent):
    name: ClassVar[str] = "range"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "范围分析 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = RangeOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for range analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> RangeOutput:
        state = inp.get("state") or {}
        opponent = inp.get("opponent") or {}
        population = inp.get("population") or {}
        out = RangeOutput()
        warnings: list[dict] = []

        defaults = ctx.config.default_ranges()
        if not defaults:
            warnings.append({"code": "NO_DEFAULT_RANGES",
                             "message": "config default_ranges() is empty (configs/ranges.yaml)"})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        players = state.get("players", []) or []
        hero_p = _find_hero(players)
        hero_id = state.get("hero_id") or (hero_p.get("player_id") if hero_p else None)
        hero_hole = [str(c) for c in (hero_p.get("hole_cards") or [])] if hero_p else []
        board = [str(c) for c in state.get("board", []) or []]
        street = str(state.get("street", "preflop"))

        # ---- parse dead cards (hero hole + board) for card removal ----
        try:
            dead_hero = [parse_card(c) for c in hero_hole]
            dead_board = [parse_card(c) for c in board]
        except ValueError as e:
            warnings.append({"code": "INVALID_CARD", "message": str(e)})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        # ---- individual feature vectors are the primary adjustment source ----
        opponent_present = bool((opponent or {}).get("players"))
        population_present = bool((population or {}).get("stats") or (population or {}).get("segments"))

        villains: dict[str, dict] = {}
        confidences: list[float] = []
        for p in players:
            if not isinstance(p, dict):
                continue
            if p.get("is_hero"):
                continue
            if p.get("status") not in ("active", "allin"):
                continue
            pid = str(p.get("player_id", ""))
            if not pid:
                continue
            pos = _position_of(state, pid)
            route = _detect_route(state, pid)
            base = _base_range_string(defaults, route, pos)
            range_name = f"{pos}_{route}" if pos else route
            out.positions[pos] = range_name if pos else route
            if base is None:
                warnings.append({"code": "NO_BASE_RANGE",
                                 "message": f"no base range for villain {pid} (route={route}, pos={pos})"})
                continue

            vr_full = parse_range(base, name=range_name)
            vr = vr_full.remove_cards(dead_hero + dead_board)

            # ---- P46：数据驱动范围 posterior（证据足够时替代启发式） ----
            # 证据源与 P43 同管（opponent 挂接的 parsed-hand 条件事件索引）；
            # 无证据 → 下方启发式路径逐位保持 P30-P45 语义。
            profile = ((opponent or {}).get("players") or {}).get(pid) or {}
            rpi_player = _rpi_from(profile.get("range_posterior"))
            rpi_pop = _rpi_from(profile.get("range_posterior_population"))
            post_kind = "starting" if street == "preflop" else "continuing"
            gate = (profile.get("evidence_gate")
                    if isinstance(profile, dict) else None)
            # P47：REAL 数据的 posterior 必须被 evidence gate 明确允许
            # （legacy 无 gate 记录维持 P46 语义）；被排除 → 诚实 baseline。
            need = "starting_range" if post_kind == "starting" else "conditional_fold"
            gate_block = eg.record_allowed({"evidence_gate": gate}, need) \
                if isinstance(gate, dict) else True
            if not gate_block:
                vr_post = None
                post = RangePosteriorResult(
                    available=False, status="baseline", kind=post_kind,
                    fallback_reason=(f"P47 evidence gate excludes {need} "
                                     f"(profile data_kind="
                                     f"{profile.get('data_kind')})"))
                post.baseline_distribution = vr.distribution()
            else:
                vr_post, post = build_range_posterior(
                    vr_full, vr, kind=post_kind, position=pos, street=street,
                    route=route, hero_cards=hero_hole, board=board,
                    player_index=rpi_player, population_index=rpi_pop,
                    cond_player=(profile.get("conditional_model")
                                 if isinstance(profile, dict) else None),
                    cond_population=(profile.get("conditional_population")
                                     if isinstance(profile, dict) else None))
                if post.available:
                    # P47：posterior 的数据真实性随 provenance 传播
                    post.provenance["data_kind"] = profile.get("data_kind")
                    if isinstance(profile.get("dataset"), dict):
                        post.provenance["dataset"] = dict(profile["dataset"])
            posterior_block = post.to_dict()

            fv = _feature_vector(opponent, pid)
            tightness = fv.get("tightness")
            tightness_source = "individual feature_vector"
            if not isinstance(tightness, (int, float)):
                # population proxy only when individual data is absent
                prox = _population_tightness_proxy(population)
                tightness = prox
                tightness_source = "population vpip proxy (tightness ~ 1-VPIP)"

            bluff = fv.get("bluff_tendency")
            if not isinstance(bluff, (int, float)):
                bluff = None

            assumptions = [f"base range {range_name} = {base!r}",
                           f"card removal applied for hero hole {hero_hole} + board {board}"]
            if isinstance(tightness, (int, float)):
                assumptions.append(f"tightness={round(float(tightness), 3)} (source: {tightness_source})")

            if post.available and vr_post is not None:
                # P46：posterior 生效——替代 0.5/1.5 启发式（同一底层数据，
                # posterior 带收缩与 provenance；启发式不再叠加，避免双重计数）
                vr = vr_post
                assumptions.append(
                    f"P46 posterior active ({post.status}): "
                    f"{post.kind}/{post.condition.get('metric') or 'continue_rate'} "
                    f"target={post.target.get('posterior_rate')} vs baseline "
                    f"{post.target.get('baseline_implied')} → tilt "
                    f"lambda={post.tilt.get('lambda')} (support=baseline)")
                stats = vr.stats()
                # confidence: player posterior > population posterior > heuristic
                conf = 0.8 if post.status == "player" else 0.65
                villains[pid] = {
                    "range_name": range_name,
                    "distribution": vr.distribution(),
                    "baseline_distribution": post.baseline_distribution,
                    "posterior": posterior_block,
                    "combo_count": int(stats["combo_count"]),
                    "total_weight": float(stats["total_weight"]),
                    "confidence": conf,
                    "assumptions": assumptions,
                    "evidence": [{"kind": "range", "text": f"{pid} {range_name}: "
                                                           f"{stats['combo_count']} combos, "
                                                           f"total_weight={stats['total_weight']} "
                                                           f"(P46 posterior: {post.status})",
                                  "source": "range", "weight": 1.0}],
                }
                confidences.append(conf)
                continue

            vr, adj_assumptions = _adjust_range(vr, tightness, bluff, street, dead_hero + dead_board)
            assumptions.extend(adj_assumptions)

            stats = vr.stats()
            # confidence: individual vector beats population proxy beats base-only
            conf = 0.7 if tightness_source == "individual feature_vector" else 0.5
            villains[pid] = {
                "range_name": range_name,
                "distribution": vr.distribution(),
                "posterior": posterior_block,
                "combo_count": int(stats["combo_count"]),
                "total_weight": float(stats["total_weight"]),
                "confidence": conf,
                "assumptions": assumptions,
                "evidence": [{"kind": "range", "text": f"{pid} {range_name}: "
                                                       f"{stats['combo_count']} combos, "
                                                       f"total_weight={stats['total_weight']}",
                              "source": "range", "weight": 1.0}],
            }
            confidences.append(conf)

        # ---- hero's own range (board removal only; keep hero hole cards) ----
        if hero_p and hero_id:
            hero_pos = _position_of(state, hero_id)
            hero_route = _detect_route(state, hero_id)
            hero_base = _base_range_string(defaults, hero_route, hero_pos)
            hero_name = f"{hero_pos}_{hero_route}" if hero_pos else hero_route
            out.positions[hero_pos] = hero_name if hero_pos else hero_route
            if hero_base:
                hr = parse_range(hero_base, name=hero_name)
                # NOTE: hero's own hole cards must NOT be removed from hero's range
                hr = hr.remove_cards(dead_board)
                hstats = hr.stats()
                out.hero_range = {
                    "range_name": hero_name,
                    "distribution": hr.distribution(),
                    "combo_count": int(hstats["combo_count"]),
                    "total_weight": float(hstats["total_weight"]),
                    "assumptions": [f"hero route {hero_route} position {hero_pos} -> {hero_base!r}",
                                    "board removal only; hero hole cards kept in own range"],
                }
            else:
                warnings.append({"code": "NO_HERO_RANGE",
                                 "message": f"no base range for hero (route={hero_route}, pos={hero_pos})"})

        out.villains = villains
        out.confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        # P30-5-D：范围来源诚实标记——P46 posterior > individual feature_vector
        # > population proxy > base-only；绝不冒充真实/GTO range。
        post_active = [v.get("posterior") or {} for v in villains.values()
                       if (v.get("posterior") or {}).get("available")]
        if any((p.get("status") == "player") for p in post_active):
            out.range_mode = "POSTERIOR"         # P46：数据驱动 posterior
            out.range_source = "player_history"
        elif post_active:
            out.range_mode = "POSTERIOR"         # P46：population 级 posterior
            out.range_source = "population_history"
        elif any(v.get("confidence", 0.0) >= 0.7 for v in villains.values()):
            out.range_mode = "APPROXIMATE"       # 个体特征向量——近似
            out.range_source = "individual_feature_vector"
        elif opponent_present or population_present:
            out.range_mode = "APPROXIMATE"       # population proxy——近似
            out.range_source = "population_proxy"
        else:
            out.range_mode = "BASELINE"          # 仅 base range（无玩家调整）
            out.range_source = "config_default_ranges"
        out.evidence = [{"kind": "range",
                         "text": f"{len(villains)} villain range(s) reconstructed; "
                                 f"positions={list(out.positions)}; "
                                 f"mode={out.range_mode}",
                         "source": "range", "weight": 1.0}]

        if not opponent_present and not population_present:
            warnings.append({"code": "NO_PROFILE_DATA",
                             "message": "no opponent/population input; using base ranges without "
                                        "player adjustment"})
        out.warnings = warnings
        return out
