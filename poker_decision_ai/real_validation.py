"""P48 — Real-data Strategy Validation harness（只读评估；确定性；无 Agent 新增）。

目标（P48 授权范围）：在相同牌局状态下对照
  1. BASELINE（config 静态范围 + 无对手证据）
  2. REAL（UofT CPRG 经 P47 evidence gate 的 Population/Conditional/Range 证据）
验证接入真实数据后 Range / Fold / Equity / EV-utility 的变化是否
**可解释、可重复、方向合理**——重点回答"为什么变"，不输出单一准确率。

纪律：
  - 复用冻结数学（RangeAgent / build_range_posterior / FoldEstimator /
    compute_ev），不修改任何策略语义；
  - starting_range 只作 behavior-rate evidence（P46 语义）；
  - 无 hole cards → hole_card_range_gt excluded（P47 gate）；
  - 不伪造 tournament/bounty 数据：PKO/KO 无真实 context →
    NOT VALIDATABLE WITH CURRENT DATA；MTT 仅做 objective/provenance
    机制检查（cash 衍生行为证据的锦标赛迁移性不可验证）；
  - 确定性：固定扫描顺序、固定截断、固定种子；同输入同输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poker_decision_ai.agents.opponent.agent import OpponentAgent
from poker_decision_ai.agents.range.agent import RangeAgent
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker.conditional_model import (
    ConditionalFoldIndex, extract_events)
from poker_decision_ai.domain.poker.equity import EquityCalculator
from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
from poker_decision_ai.domain.poker.range_posterior import (
    RangePosteriorIndex, combo_strength_preflop)
from poker_decision_ai.domain.poker.ranges import Range, parse_range
from poker_decision_ai.domain.poker.strategy_core import compute_ev
from poker_decision_ai.uoft_adapter import UOFT_DATASET_META

#: harness 评估常量（声明式；非策略阈值）
EQUITY_ITERATIONS = 2000
FOLD_SIZES = (0.5, 1.0, 2.0)        # 占底池比（postflop fold 对照尺寸）
LAMBDA_JUMP_LIMIT = 0.35            # 样本递增时 |Δλ| 的异常跳变上限
POSITIONS = ("CO", "BTN", "SB", "BB")


class _RangeCtx:
    """RangeAgent.handle 的最小 ctx shim（仅 default_ranges 被使用）。"""

    def __init__(self, config: Any) -> None:
        self.config = config


# ---------------------------------------------------------------- 数据采集
def collect_real_sample(canonical_path: str, scan_limit: int
                        ) -> dict[str, list[dict]]:
    """扫描 canonical 前缀 → {player_id: [extract_events 事件按手分组]}。

    确定性：按文件顺序扫描前 scan_limit 手；每手一次 extract_events；
    事件按 (hand_id, player) 分组保留（支持按手数子采样）。
    """
    import json

    hands: dict[str, list[dict]] = {}
    with open(canonical_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan_limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            canonical = rec.get("canonical") or {}
            quality = rec.get("quality") or {}
            verdicts = eg.classify_canonical(canonical, quality)
            if verdicts["action_level"]["verdict"] not in eg.ALLOWED_VERDICTS:
                continue
            from poker_decision_ai.uoft_adapter import canonical_to_parsed
            parsed = canonical_to_parsed(canonical)
            if not parsed:
                continue
            try:
                evs = extract_events(parsed)
            except Exception:
                continue
            by_player: dict[str, list[dict]] = {}
            for ev in evs:
                by_player.setdefault(str(ev.get("player") or ""), []).append(ev)
            hid = str(parsed.get("hand_id") or i)
            for pid, pev in by_player.items():
                if pev:
                    hands.setdefault(pid, []).append(
                        {"hand_id": hid, "events": pev, "quality": quality,
                         "verdicts": verdicts})
    return hands


def pick_target_player(hands: dict[str, list[dict]]) -> str | None:
    """确定性选择目标玩家：preflop 决策机会最多者（id 字典序 tiebreak）。"""
    best, best_n = None, -1
    for pid, hs in sorted(hands.items()):
        n = sum(1 for h in hs if any(e.get("street") == "preflop"
                                     for e in h["events"]))
        if n > best_n:
            best, best_n = pid, n
    return best


def build_evidence(player_hands: list[dict], all_hands: dict[str, list[dict]]
                   ) -> dict:
    """目标玩家的 P43/P46 证据索引（player + population 双索引 + gate）。"""
    player_rpi = RangePosteriorIndex()
    player_cfi = ConditionalFoldIndex()
    for h in player_hands:
        player_rpi.add_events(h["events"])
        player_cfi.add_events(h["events"])
    pop_rpi = RangePosteriorIndex()
    pop_cfi = ConditionalFoldIndex()
    for pid, hs in all_hands.items():
        for h in hs:
            pop_rpi.add_events(h["events"])
            pop_cfi.add_events(h["events"])
    rep = eg.EvidenceReport()
    for h in player_hands[:1] or []:
        rep.add(h["verdicts"])
    gate = OpponentAgent()._gate_summary(
        [{"data_kind": "REAL", "dataset": UOFT_DATASET_META,
          "evidence_gate": (player_hands[0]["verdicts"]
                            if player_hands else {})}])
    return {"player_rpi": player_rpi, "player_cfi": player_cfi,
            "pop_rpi": pop_rpi, "pop_cfi": pop_cfi,
            "gate": gate, "n_hands": len(player_hands)}


def real_profile(pid: str, evidence: dict) -> dict:
    """REAL 对手 profile（与 OpponentAgent 挂接字段同形）。"""
    return {
        "player_id": pid, "name": pid,
        "range_posterior": evidence["player_rpi"].to_dict(),
        "range_posterior_population": evidence["pop_rpi"].to_dict(),
        "conditional_model": evidence["player_cfi"].to_dict(),
        "conditional_population": evidence["pop_cfi"].to_dict(),
        "data_kind": "REAL", "dataset": dict(UOFT_DATASET_META),
        "evidence_gate": evidence["gate"].get("evidence_gate") or {},
        "behavior_statistics": {},
    }


# ---------------------------------------------------------------- 场景
def make_state(street: str, villain_pos: str, route: str,
               game_type: str = "cash", payouts: list | None = None) -> dict:
    """固定最小现金状态（hero BTN；villain 指定位置/route）。"""
    pot, to_call = {"preflop": (1.5, 0.0), "flop": (10.0, 0.0),
                    "turn": (24.0, 0.0), "river": (52.0, 0.0)}[street]
    boards = {"preflop": [], "flop": ["Ah", "Kd", "7c"],
              "turn": ["Ah", "Kd", "7c", "2s"],
              "river": ["Ah", "Kd", "7c", "2s", "9d"]}
    hero_hole = ["As", "Kd"]
    history: list[dict] = [{"street": "preflop", "player_id": "hero",
                            "action": "raise", "amount": 2.5, "seq": 1}]
    if street != "preflop":
        history.append({"street": "flop", "player_id": "villain",
                        "action": "bet", "amount": 6.0, "seq": 2})
    if street in ("turn", "river"):
        history.append({"street": "flop", "player_id": "hero",
                        "action": "call", "amount": 6.0, "seq": 3})
        history.append({"street": street, "player_id": "villain",
                        "action": "bet", "amount": 14.0, "seq": 4})
    if route == "three_bet" and street == "preflop":
        history.insert(1, {"street": "preflop", "player_id": "villain",
                           "action": "raise", "amount": 8.0, "seq": 1.5})
        to_call = 8.0
    players = [
        {"player_id": "hero", "is_hero": True, "status": "active",
         "position": "BTN", "hole_cards": hero_hole, "stack": 100.0},
        {"player_id": "villain", "status": "active",
         "position": villain_pos, "hole_cards": [], "stack": 100.0},
    ]
    state = {"game_type": game_type, "street": street, "board": boards[street],
             "pot": pot, "to_call": to_call, "bb": 1.0, "sb": 0.5,
             "hero_id": "hero", "button_seat": 1, "players": players,
             "action_history": history,
             "available_actions": (["fold", "call", "raise"] if to_call > 0
                                   else ["fold", "check", "bet"])}
    if payouts is not None:
        state["tournament"] = {"payouts": payouts, "stage": "mid"}
    return state


def scenarios() -> list[dict]:
    """固定场景 manifest（确定性顺序；CASH 策略对照 + MTT 机制检查）。"""
    out: list[dict] = []
    for street in ("preflop", "flop", "turn", "river"):
        for pos in POSITIONS:
            route = "three_bet" if (street == "preflop" and pos == "BB") \
                else "open"
            out.append({"id": f"cash_{street}_{pos}", "street": street,
                        "position": pos, "route": route, "game_type": "cash"})
    # MTT：仅 objective/provenance 机制检查（行为证据来自 cash 语料——
    # 锦标赛迁移性不可验证，报告显式声明）
    out.append({"id": "mtt_mech_flop_CO", "street": "flop", "position": "CO",
                "route": "open", "game_type": "mtt",
                "payouts": [50, 30, 20, 15, 10, 8, 6, 5, 4, 3],
                "mechanism_only": True})
    return out


# ---------------------------------------------------------------- 指标
def _range_from_distribution(dist: dict[str, float]) -> Range:
    out = Range(name="from_dist")
    for name, w in (dist or {}).items():
        try:
            sub = parse_range(str(name))
        except ValueError:
            continue
        for combo in sub.weights:
            out.weights[combo] = out.weights.get(combo, 0.0) + w
    return out


def total_variation(d1: dict, d2: dict) -> float:
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(float(d1.get(k, 0.0)) - float(d2.get(k, 0.0)))
                     for k in keys)


def _weighted_strength(vr: Range, preflop: bool, hero_cards=None,
                       board=None) -> float | None:
    """后验/基线范围的平均强度（方向 sanity 用；与 P46 评分同源）。"""
    total = sum(vr.weights.values())
    if total <= 0:
        return None
    if preflop:
        return sum(w * combo_strength_preflop(c)
                   for c, w in vr.weights.items()) / total
    # P50-2：与后验校准同源的 hero-agnostic 强度（board 条件 + 死牌剔除）
    from poker_decision_ai.domain.poker.range_posterior import (
        _combo_strength_vs_random)
    from poker_decision_ai.domain.poker.cards import parse_card
    strg = _combo_strength_vs_random(vr, [parse_card(c) for c in board],
                                     [parse_card(c) for c in
                                      (list(hero_cards) + list(board))])
    if not strg:
        return None
    return sum(w * strg.get(c, 0.0) for c, w in vr.weights.items()) / total


async def _run_side(agent: RangeAgent, ctx: _RangeCtx, state: dict, pid: str,
                    profile: dict | None) -> dict:
    inp = {"state": state}
    if profile is not None:
        inp["opponent"] = {"players": {pid: profile}}
    out = await agent.handle(inp, ctx)
    v = (out.villains or {}).get(pid) or {}
    return {"v": v, "range_mode": out.range_mode,
            "range_source": out.range_source}


async def compare_scenario(cfg: Any, scen: dict, pid: str, profile: dict,
                           evidence: dict) -> dict:
    """单场景 BASELINE vs REAL 全链路对照（Range→Equity→Fold→EV）。"""
    agent = RangeAgent()
    ctx = _RangeCtx(cfg)
    state = make_state(scen["street"], scen["position"], scen["route"],
                       scen.get("game_type", "cash"), scen.get("payouts"))
    base = await _run_side(agent, ctx, state, pid, None)
    real = await _run_side(agent, ctx, state, pid, profile)
    bv, rv = base["v"], real["v"]
    bd, rd = bv.get("distribution") or {}, rv.get("distribution") or {}
    hero = state["players"][0]["hole_cards"]
    board = state["board"]
    preflop = state["street"] == "preflop"

    # ---- equity（同映射、同种子；仅分布不同） ----
    calc = EquityCalculator(iterations=EQUITY_ITERATIONS, seed=42)
    eq = {}
    for tag, dist in (("base", bd), ("real", rd)):
        if not dist:
            eq[tag] = None
            continue
        er = calc.hand_vs_range(hero, board, _range_from_distribution(dist),
                                iterations=EQUITY_ITERATIONS, seed=42)
        eq[tag] = {"hero_equity": er.hero_equity, "tie": er.tie}

    # ---- fold probability（postflop；3 个参考尺寸） ----
    folds: dict[str, dict] = {}
    if not preflop and bd and rd:
        for frac in FOLD_SIZES:
            s = frac * (state["pot"] / state["bb"])
            row = {}
            for tag, dist in (("base", bd), ("real", rd)):
                fe = FoldEstimator(pot_bb=state["pot"] / state["bb"],
                                   facing="bet", street=state["street"],
                                   villain_dist=dist,
                                   behavior=({"conditional_model":
                                              profile["conditional_model"],
                                              "conditional_population":
                                              profile["conditional_population"]}
                                             if tag == "real" else {}),
                                   hero_cards=hero, board=board, seed=42)
                r = fe.for_size(s)
                row[tag] = ({"p_fold": r.p_fold, "mode": r.mode}
                            if r.available else None)
            row["diff"] = (None if row["base"] is None or row["real"] is None
                           or row["base"]["p_fold"] is None
                           or row["real"]["p_fold"] is None
                           else round(row["real"]["p_fold"]
                                      - row["base"]["p_fold"], 6))
            folds[f"{frac}x_pot"] = row

    # ---- EV / utility ranking（各侧用自己的 equity + 分布） ----
    ev = {}
    for tag, dist, e in (("base", bd, eq["base"]), ("real", rd, eq["real"])):
        if not dist or e is None:
            ev[tag] = None
            continue
        hero_eq = e["hero_equity"]
        tie = e["tie"]
        outcome = {"win": round(hero_eq - tie / 2, 6), "tie": round(tie, 6),
                   "lose": round(1.0 - hero_eq - tie / 2, 6)}
        res = compute_ev(state, {"hero_equity": hero_eq, "outcome": outcome},
                         range_src={"villains": {pid: {"distribution": dist}}})
        ev[tag] = ({"available": res.available,
                    "objective": res.objective, "mode": res.mode,
                    "icm_available": res.icm_available,
                    "best": ({"action": res.best_action,
                              "amount_bb": res.best_amount_bb}
                             if res.available else None),
                    "ranking": [c.to_dict() for c in res.candidates]}
                   if res.available else
                   {"available": False,
                    "reason": res.unavailable_reason})

    post = rv.get("posterior") or {}
    return {
        "id": scen["id"], "street": scen["street"], "position": scen["position"],
        "route": scen["route"], "game_type": scen.get("game_type", "cash"),
        "baseline": {"range_mode": base["range_mode"],
                     "combo_count": bv.get("combo_count")},
        "real": {"range_mode": real["range_mode"],
                 "posterior_status": post.get("status"),
                 "posterior_available": post.get("available"),
                 "fallback_reason": post.get("fallback_reason"),
                 "data_kind": (post.get("provenance") or {}).get("data_kind"),
                 "lambda": (post.get("tilt") or {}).get("lambda"),
                 "target": post.get("target"),
                 "confidence": post.get("confidence")},
        "metrics": {
            "range_tv": round(total_variation(bd, rd), 6),
            "equity": eq,
            "equity_diff": (None if not eq["base"] or not eq["real"] else
                            round(eq["real"]["hero_equity"]
                                  - eq["base"]["hero_equity"], 6)),
            "fold": folds,
            "best_action": {"base": (ev["base"] or {}).get("best"),
                            "real": (ev["real"] or {}).get("best")},
            "best_action_change": (
                None if not (ev["base"] or {}).get("best")
                or not (ev["real"] or {}).get("best") else
                ev["base"]["best"]["action"] != ev["real"]["best"]["action"]),
            "ev_objective": {"base": (ev["base"] or {}).get("objective"),
                             "real": (ev["real"] or {}).get("objective")},
            "mechanism_only": bool(scen.get("mechanism_only")),
        },
    }


# ---------------------------------------------------------------- sanity
async def lambda_curve(player_hands: list[dict], all_hands: dict[str, list[dict]],
                       cfg: Any, ns: tuple[int, ...], scen: dict) -> dict:
    """样本递增 → posterior λ 曲线（跳变检测 + 方向一致性）。"""
    agent = RangeAgent()
    ctx = _RangeCtx(cfg)
    state = make_state(scen["street"], scen["position"], scen["route"])
    curve: list[dict] = []
    for n in ns:
        sub = player_hands[:n]
        ev = build_evidence(sub, all_hands)
        prof = real_profile("villain", ev)
        real = await _run_side(agent, ctx, state, "villain", prof)
        post = (real["v"].get("posterior") or {})
        curve.append({"n": n, "status": post.get("status"),
                      "lambda": (post.get("tilt") or {}).get("lambda"),
                      "target_rate": (post.get("target") or {}).get(
                          "posterior_rate")})
    jumps = [abs(curve[i + 1]["lambda"] - curve[i]["lambda"])
             for i in range(len(curve) - 1)
             if curve[i]["lambda"] is not None
             and curve[i + 1]["lambda"] is not None]
    return {"curve": curve,
            "max_jump": round(max(jumps), 6) if jumps else None,
            "no_abnormal_jump": (max(jumps) <= LAMBDA_JUMP_LIMIT
                                 if jumps else True)}


async def direction_check(cfg: Any, scen: dict, evidence_tight: dict,
                          evidence_loose: dict) -> dict:
    """强收紧/强宽松行为 → 后验范围强度方向必须一致（不反向）。"""
    agent = RangeAgent()
    ctx = _RangeCtx(cfg)
    state = make_state(scen["street"], scen["position"], scen["route"])
    preflop = state["street"] == "preflop"
    hero = state["players"][0]["hole_cards"]
    board = state["board"]
    out: dict[str, Any] = {}
    base = await _run_side(agent, ctx, state, "villain", None)
    b_dist = base["v"].get("distribution") or {}
    b_strength = _weighted_strength(_range_from_distribution(b_dist), preflop,
                                    hero, board)
    for tag, ev in (("tight", evidence_tight), ("loose", evidence_loose)):
        real = await _run_side(agent, ctx, state, "villain", real_profile("villain", ev))
        post = real["v"].get("posterior") or {}
        dist = real["v"].get("distribution") or {}
        r_strength = _weighted_strength(_range_from_distribution(dist), preflop,
                                        hero, board)
        lam = (post.get("tilt") or {}).get("lambda")
        out[tag] = {"lambda": lam, "status": post.get("status"),
                    "strength_base": b_strength, "strength_real": r_strength}
        if lam is not None and b_strength is not None and r_strength is not None:
            out[tag]["direction_ok"] = (
                (lam < 0 and r_strength < b_strength)
                or (lam > 0 and r_strength > b_strength)
                or lam == 0)
    return out


# ─────────────────────────── P49：边际场景与诊断 ───────────────────────────
#: P49 边际 hero 三档（固定 board AhKd7c 上相对真实后验范围的 equity 分档）
MARGINAL_HERO_TIERS = {
    "strong": ["Ah", "Kd"],          # 顶两对级别
    "medium": ["Qc", "Jc"],          # 两高牌无对
    "weak": ["6c", "5c"],            # 无对无听牌
}
#: P51 facing -> to_call(bb)（strategy_core 语义：to_call>0 → facing=raise）
#: check = villain 让牌（hero 首个行动，bet/size 候选）；bet/raise 多尺寸
MARGINAL_FACINGS = {
    "check": 0.0,                    # villain check → hero 可 check/bet
    "bet_05": 5.0,                   # 面对 0.5x pot 下注
    "bet_10": 10.0,                  # 面对 1.0x pot 下注
    "raise_15": 15.0,                # 面对 1.5x pot 加注
    "raise_25": 25.0,                # 面对 2.5x pot 加注
}
#: P49 诊断常量（声明式；非策略阈值）
SAT_GRID_RATES = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)


def marginal_scenarios() -> list[dict]:
    """P49 边际场景：CASH，3 街 × 3 档 hero equity × 3 facing = 27。

    MTT/PKO 维持 P48 不可验证声明（不伪造 tournament/bounty context）。
    """
    out = []
    for street in ("flop", "turn", "river"):
        for tier in MARGINAL_HERO_TIERS:
            for facing in MARGINAL_FACINGS:
                out.append({"id": f"marg_{street}_{tier}_{facing}",
                            "street": street, "position": "CO",
                            "route": "open", "game_type": "cash",
                            "hero_tier": tier,
                            "hero_cards": list(MARGINAL_HERO_TIERS[tier]),
                            "facing": facing,
                            "to_call_bb": MARGINAL_FACINGS[facing]})
    return out


def marginal_state(scen: dict) -> dict:
    """边际场景状态（在 make_state 基础上覆写 hero 手牌与 to_call）。"""
    st = make_state(scen["street"], scen["position"], scen["route"],
                    scen.get("game_type", "cash"))
    st["players"][0]["hole_cards"] = list(scen["hero_cards"])
    bb = st["bb"]
    st["to_call"] = round(scen["to_call_bb"] * bb, 6)
    st["current_bet"] = st["to_call"]
    if st["to_call"] > 0:
        st["available_actions"] = ["fold", "call", "raise"]
    else:
        st["available_actions"] = ["fold", "check", "bet"]
    return st


async def three_arm_decision(cfg: Any, scen: dict, pid: str,
                             profile: dict) -> dict:
    """三臂决策对照：BASELINE / REAL / COUNTERFACTUAL。

    COUNTERFACTUAL = baseline 分布 + REAL 行为证据（隔离 P46 倾斜的贡献）；
    REAL = posterior 分布 + 行为证据。每臂输出 equity、fold 概率
    （0.5x pot）与 EV 候选排序/最佳动作。
    """
    agent = RangeAgent()
    ctx = _RangeCtx(cfg)
    state = marginal_state(scen)
    hero = state["players"][0]["hole_cards"]
    board = state["board"]
    calc = EquityCalculator(iterations=EQUITY_ITERATIONS, seed=42)

    base = await _run_side(agent, ctx, state, pid, None)
    real = await _run_side(agent, ctx, state, pid, profile)
    bd = base["v"].get("distribution") or {}
    rd = real["v"].get("distribution") or {}

    def _ev_arm(dist: dict, with_behavior: bool) -> dict:
        if not dist:
            return {"available": False, "reason": "no_distribution"}
        er = calc.hand_vs_range(hero, board, _range_from_distribution(dist),
                                iterations=EQUITY_ITERATIONS, seed=42)
        eq_src = {"hero_equity": er.hero_equity,
                  "outcome": {"win": round(er.hero_equity - er.tie / 2, 6),
                              "tie": round(er.tie, 6),
                              "lose": round(1.0 - er.hero_equity - er.tie / 2,
                                            6)}}
        opp = {"players": {pid: profile}} if with_behavior else None
        res = compute_ev(state, eq_src,
                         range_src={"villains": {pid: {"distribution": dist}}},
                         opponent_src=opp)
        fe = FoldEstimator(pot_bb=state["pot"] / state["bb"], facing="bet",
                           street=state["street"], villain_dist=dist,
                           behavior=(profile if with_behavior else {}),
                           hero_cards=hero, board=board, seed=42)
        fr = fe.for_size(0.5 * state["pot"] / state["bb"])
        return {
            "equity": round(er.hero_equity, 6),
            "fold_05x": ({"p_fold": fr.p_fold, "mode": fr.mode,
                          "w_stat": fr.provenance.get("w_stat"),
                          "stat_source": fr.provenance.get("source")}
                         if fr.available else None),
            "best": ({"action": res.best_action,
                      "amount_bb": res.best_amount_bb,
                      "ev_bb": round(res.best_ev_bb, 6)}
                     if res.available else None),
            "ranking": [{"action": c.action, "amount_bb": c.amount_bb,
                         "utility_bb": round(c.utility_bb, 6)}
                        for c in res.candidates] if res.available else [],
        }

    arms = {"BASELINE": _ev_arm(bd, False),
            "COUNTERFACTUAL": _ev_arm(bd, True),
            "REAL": _ev_arm(rd, True)}
    best_of = lambda arm: (arms[arm].get("best") or {}).get("action")
    return {
        "id": scen["id"], "hero_tier": scen["hero_tier"],
        "facing": scen["facing"], "street": scen["street"],
        "arms": arms,
        "posterior": {"lambda": ((real["v"].get("posterior") or {})
                                 .get("tilt") or {}).get("lambda"),
                      "status": (real["v"].get("posterior") or {})
                      .get("status")},
        "action_change": {"baseline_vs_real":
                          best_of("BASELINE") != best_of("REAL")},
        # P51：内部决策质量指标（同一模型族下的 EV 一致性——非盈利性证明）
        "regret_proxy": (
            None if not (arms["REAL"].get("best")
                         and arms["BASELINE"].get("best"))
            else round(arms["REAL"]["best"]["ev_bb"]
                       - arms["BASELINE"]["best"]["ev_bb"], 6)),
        "ev_margin": {arm: (round(arms[arm]["ranking"][0]["utility_bb"]
                                  - arms[arm]["ranking"][1]["utility_bb"], 6)
                            if len(arms[arm].get("ranking") or []) >= 2
                            else None)
                      for arm in arms},
    }


def _synthetic_continuing_events(n: int, fold_frac: float, prefix: str
                                 ) -> list[dict]:
    """诊断用合成条件事件（仅用于 λ 响应映射；非真实数据，显式标注）。"""
    out = []
    for i in range(n):
        f = 1 if (i % 100) / 100.0 < fold_frac else 0
        out.append({"player": "diag", "street": "flop", "facing": "bet",
                    "size_bucket": "m<=1.0", "position": "CO",
                    "texture": "wet", "action": "fold" if f else "call",
                    "fold": f, "hand_id": f"{prefix}{i}"})
    return out


def diagnose_saturation(cfg: Any, evidence: dict | None = None) -> dict:
    """P46 极端响应诊断（只读；不修改 P46 数学）。

    三个问题：
      1. λ 饱和由哪个常量造成（DEVIATION_CAP vs LAMBDA_CAP）——
         合成率网格映射 λ(response)（synthetic diagnostic，显式标注）；
      2. hero-conditioned 校准伪影——同一真实证据、不同 hero 手牌的 λ；
      3. 融合权重缺陷探针——conditional 统计通道的 w_stat（当前实现下
         conditional 路径的 samples 嵌套在 prov["conditional"] 内，
         for_size 读取顶层 samples → w_stat=0；P42/P43 集成缺陷，
         本阶段只定位不修改）。
    """
    from poker_decision_ai.domain.poker.conditional_model import (
        ConditionalFoldIndex)
    from poker_decision_ai.domain.poker.range_posterior import (
        build_range_posterior)
    from poker_decision_ai.domain.poker.ranges import parse_range

    base = parse_range(
        "AA-22,AKs-ATs,KQs-K9s,Q9s+,J9s+,T9s,98s,87s,76s,65s,AKo-AJo,KQo")
    live = base.remove_cards([])
    hero = ["Ah", "Kd"]
    board = ["Ah", "Kd", "7c"]

    grid = []
    for rate in SAT_GRID_RATES:
        cfi = ConditionalFoldIndex()
        for ev in _synthetic_continuing_events(200, 1.0 - rate, "d"):
            cfi.add_events([ev])
        _, res = build_range_posterior(
            base, live, kind="continuing", position="CO", street="flop",
            hero_cards=hero, board=board, cond_player=cfi.to_dict())
        d = (res.target or {}).get("deviation")
        lam = (res.tilt or {}).get("lambda")
        binding = ("none" if lam is None or abs(lam) < 1e-9 else
                   "deviation_cap" if d is not None and abs(d) >= 1.0 - 1e-9
                   else "lambda_cap" if lam is not None
                   and abs(lam) >= 1.2 - 1e-9 else "linear_response")
        grid.append({"target_continue_rate": rate, "deviation": d,
                     "lambda": lam, "binding": binding,
                     "posterior_available": res.available})
    sat_points = [g for g in grid if g["binding"] == "deviation_cap"]
    lambda_cap_binding = any(g["binding"] == "lambda_cap" for g in grid)

    # hero-conditioning 探针：同一真实证据，不同 hero 手牌的 λ
    hero_probe = []
    if evidence is not None:
        for tier, hc in MARGINAL_HERO_TIERS.items():
            _, res = build_range_posterior(
                base, live, kind="continuing", position="CO", street="flop",
                hero_cards=hc, board=board,
                cond_player=evidence["player_cfi"].to_dict(),
                cond_population=evidence["pop_cfi"].to_dict())
            hero_probe.append({
                "hero_tier": tier,
                "lambda": (res.tilt or {}).get("lambda"),
                "baseline_implied": (res.target or {}).get("baseline_implied"),
                "raw_continue": (res.provenance.get("conditional_estimate")
                                 or {}).get("raw")})
    lam_vals = [p["lambda"] for p in hero_probe if p["lambda"] is not None]
    raws = {round(p["raw_continue"], 6) for p in hero_probe
            if p["raw_continue"] is not None}
    bases = {round(p["baseline_implied"], 6) for p in hero_probe
             if p["baseline_implied"] is not None}
    spread = (round(max(lam_vals) - min(lam_vals), 6)
              if len(lam_vals) >= 2 else None)
    # 结构判据：raw 行为率与 hero 无关、baseline 随 hero 变、λ 随之摆动
    # （阈值 0.3：λ 的摆动超过收缩后的中等响应量级——声明式诊断阈值）
    hero_conditioning = {
        "probe": hero_probe,
        "lambda_spread": spread,
        "raw_continue_hero_invariant": len(raws) <= 1 and bool(raws),
        "baseline_implied_varies": len(bases) >= 2,
        # P50-2 修复后：baseline_implied 仅随 card removal 微变 → spread 小
        "verdict": ("hero-conditioned calibration artifact CONFIRMED"
                    if spread is not None and spread > 0.3
                    and len(raws) <= 1 and len(bases) >= 2
                    else "ambiguous"
                    if spread is not None and spread > 0.1
                    else "hero-invariant (P50-2 fix verified)"),
    }

    # w_stat 探针：conditional 统计通道在融合中的实际权重
    w_stat_probe = None
    if evidence is not None:
        fe = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                           villain_dist=live.distribution(),
                           behavior={"conditional_model":
                                     evidence["player_cfi"].to_dict(),
                                     "conditional_population":
                                     evidence["pop_cfi"].to_dict()},
                           hero_cards=hero, board=board, seed=42)
        fr = fe.for_size(5.0)
        w_stat_probe = {
            "available": fr.available, "mode": fr.mode,
            "w_stat": fr.provenance.get("w_stat"),
            "stat_source": fr.provenance.get("source"),
            "finding": ("REGRESSION: conditional stat channel contributes "
                        "zero fusion weight — P50-1 fix broken")
            if fr.available and fr.provenance.get("w_stat") == 0.0
            and fr.provenance.get("source") == "conditional_player"
            else ("P50-1 fix verified: conditional evidence carries fusion "
                  "weight w_stat=" + str(fr.provenance.get("w_stat")))
            if fr.available and fr.provenance.get("source")
            == "conditional_player" else "no conditional stat channel"}

    return {"grid_synthetic": grid,
            "deviation_cap_share": (len(sat_points) / len(grid)
                                    if grid else None),
            "lambda_cap_ever_binds": lambda_cap_binding,
            "attribution": ("saturation source = DEVIATION_CAP (d clipped at "
                            "±1.0); LAMBDA_CAP=1.2 unreachable because "
                            "λ=−SENSITIVITY×d ≤ ±1.0")
            if sat_points and not lambda_cap_binding else
            ("no saturation for rates in grid (P50-2 hero-agnostic "
             "baseline removed the pathology); DEVIATION_CAP retained as "
             "guardrail")
            if not sat_points and not lambda_cap_binding else "mixed",
            "hero_conditioning": hero_conditioning,
            "w_stat_probe": w_stat_probe}


# ─────────────── P51：历史行为校准验证（LOO；无需 hole cards） ───────────────
#: 时间序分割（文件序≈时间序）：前段=证据，后段=评估（确定性；无泄漏）
EVAL_SPLIT_FRAC = 0.3
#: 中性 fold 先验（population 估计不可用时的 BASELINE 臂）
NEUTRAL_FOLD = 0.5


def build_loo_split(player_hands: list[dict],
                    eval_frac: float = EVAL_SPLIT_FRAC
                    ) -> tuple[list[dict], list[dict]]:
    """确定性时间序分割：前 (1−eval_frac) 手 = 证据，后段 = 评估。

    同一手牌不得同时出现在两侧（P51 泄漏纪律；测试断言）。
    """
    k = int(len(player_hands) * (1.0 - eval_frac))
    return player_hands[:k], player_hands[k:]


def collect_eval_decisions(eval_hands: list[dict]) -> list[dict]:
    """评估手的 facing≠none 决策 → observed fold 记录（真实历史行为）。"""
    out = []
    for h in eval_hands:
        for ev in h.get("events") or []:
            if ev.get("facing") in (None, "", "none"):
                continue
            out.append({"hand_id": str(h.get("hand_id") or ""),
                        "street": str(ev.get("street") or ""),
                        "facing": str(ev.get("facing") or ""),
                        "size_bucket": ev.get("size_bucket"),
                        "position": str(ev.get("position") or ""),
                        "texture": str(ev.get("texture") or ""),
                        "observed_fold": int(ev.get("fold") or 0)})
    return out


def evaluate_fold_calibration(eval_records: list[dict],
                              player_index, pop_index,
                              min_samples: int = 30) -> dict:
    """LOO 行为校准对照：预测 fold vs 真实 observed fold。

    BASELINE = population 同条件率（无 player 证据；缺失 → 中性 0.5）；
    REAL     = player 条件估计（population 作先验，P43 阶梯）——评估手
    不在 player 索引内（LOO）。指标：Brier / agreement（预测≥0.5 对齐
    observed）/ 配对覆盖率。历史 action 一致率 ≠ 正确性/盈利性（声明）。
    """
    arms = {"BASELINE": {"p": [], "y": []},
            "REAL": {"p": [], "y": []}}
    n_total = len(eval_records)
    n_real_available = 0
    for rec in eval_records:
        y = rec["observed_fold"]
        args = (rec["street"], rec["facing"], rec["size_bucket"],
                rec["position"], rec["texture"])
        pb = (pop_index.estimate(*args, min_samples=min_samples)
              if pop_index is not None else None)
        p_base = float(pb["p_fold"]) if pb is not None else NEUTRAL_FOLD
        pr = (player_index.estimate(*args, population=pop_index,
                                    min_samples=min_samples)
              if player_index is not None else None)
        if pr is None:
            continue                          # 配对评估：REAL 不可用 → 跳过
        n_real_available += 1
        p_real = float(pr["p_fold"])
        arms["BASELINE"]["p"].append(p_base)
        arms["BASELINE"]["y"].append(y)
        arms["REAL"]["p"].append(p_real)
        arms["REAL"]["y"].append(y)

    def _metrics(pairs):
        if not pairs or not pairs["p"]:
            return None
        p = pairs["p"]
        y = pairs["y"]
        brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)
        agr = sum(1 for pi, yi in zip(p, y)
                  if (pi >= 0.5) == bool(yi)) / len(p)
        return {"n": len(p),
                "brier": round(brier, 6),
                "agreement": round(agr, 6)}

    out = {"n_eval_decisions": n_total,
           "n_paired_real_available": n_real_available,
           "real_coverage": (round(n_real_available / n_total, 6)
                             if n_total else None),
           "arms": {a: _metrics(d) for a, d in arms.items()}}
    b, r = out["arms"]["BASELINE"], out["arms"]["REAL"]
    if b and r:
        out["brier_delta_real_minus_baseline"] = round(
            r["brier"] - b["brier"], 6)
        out["agreement_delta_real_minus_baseline"] = round(
            r["agreement"] - b["agreement"], 6)
    return out


def manifest() -> dict:
    """P48 固定 evaluation manifest（版本化；无时间戳；可 diff）。"""
    return {
        "manifest_version": "p51-real-v3",
        "purpose": ("验证 P32/P43/P46 接入真实 UofT evidence 后的策略变化是否"
                    "可解释/可重复/方向合理；不是 accuracy 竞赛"),
        "data_source": {
            "dataset": UOFT_DATASET_META,
            "collection": ("collect_real_sample：canonical 前缀确定性扫描，"
                           "逐手 extract_events，按 (hand, player) 分组"),
            "gate": "P47 evidence_gate（REAL 记录必须被允许才进入证据）",
        },
        "comparison_arms": [
            {"arm": "BASELINE", "definition": "config 静态范围，无对手证据"},
            {"arm": "REAL", "definition": ("P46 RangePosterior + P43 "
                                           "ConditionalFold（player→population 阶梯）"
                                           "+ P47 gate provenance")},
        ],
        "metrics": ["range_tv", "equity_diff", "fold_p_diff@{0.5,1.0,2.0}x_pot",
                    "best_action_change", "ev_objective", "fallback_status",
                    "provenance_correctness"],
        "fixed_constants": {"equity_iterations": EQUITY_ITERATIONS,
                            "fold_sizes_pot_frac": list(FOLD_SIZES),
                            "lambda_jump_limit": LAMBDA_JUMP_LIMIT,
                            "positions": list(POSITIONS)},
        "behavioral_calibration": {
            "method": ("LOO 时间序分割（前 70% 手=证据，后 30%=评估）；"
                       "预测 fold vs 真实 observed fold；Brier/agreement"),
            "arms": {"BASELINE": "population 同条件率（无 player 证据）",
                     "REAL": "player 条件估计（population 作先验，P43 阶梯）"},
            "discipline": ("同一历史手牌不得同时成为 evidence 与 evaluation "
                           "ground truth；historical agreement ≠ 正确性/盈利性"),
        },
        "not_validatable": [
            {"evidence_type": "bounty_context",
             "status": "NOT VALIDATABLE WITH CURRENT DATA",
             "reason": ("UofT CPRG 语料为 cash-only（manifest venue 全为现金桌），"
                        "无真实 tournament/bounty context；不伪造")},
            {"evidence_type": "mtt_strategy_transfer",
             "status": "MECHANISM_CHECK_ONLY",
             "reason": ("无真实锦标赛手牌；MTT 场景仅验证 objective/icm 机制与"
                        "provenance 一致性，cash 行为证据的锦标赛迁移性不可验证")},
            {"evidence_type": "combo_level_range_ground_truth",
             "status": "NOT VALIDATABLE WITH CURRENT DATA",
             "reason": ("观察者视角无真实 hole cards（P47 gate："
                        "hole_card_range_gt excluded）")},
            {"evidence_type": "observed_action_agreement",
             "status": "NOT VALIDATABLE WITH CURRENT DATA",
             "reason": ("chunk_000 全量扫描 100,979 手 hole-card 手牌为 0"
                        "（P51 实测）——真实 cash 决策无 hole-card ground "
                        "truth，observed action 无法与模型决策在相同信息"
                        "条件下对照；行为校准走 LOO fold 通道（见 "
                        "behavioral_calibration）")},
        ],
        "scenarios": scenarios() + marginal_scenarios(),
        "p49_addendum": {"marginal_hero_tiers": {k: v for k, v in MARGINAL_HERO_TIERS.items()}, "marginal_facings_to_call_bb": {k: v for k, v in MARGINAL_FACINGS.items()}, "diagnostics": ["saturation_grid_synthetic", "hero_conditioning_probe", "w_stat_probe"], "purpose": "边际场景验证 REAL evidence 是否影响 best action；P46 极端响应只诊断不修改"}
    }
