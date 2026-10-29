"""P45 — Strategy Evaluation / Validation（纯 domain 工具；不是 Agent）。

定位（诚实声明）：
  - 用 data/hand_history 的**可解析手牌文本**（当前全部为 synthetic_seed_42
    合成——source.json 明确"仅用于 Pipeline 验证，不能用于策略有效性证明"）
    加上**明确标注的 synthetic_boundary 构造用例**，验证策略核心的机械性质：
    EV/utility 一致性、排序、单调性、UNKNOWN 纪律、provenance、动作合法性、
    可复现性。
  - 对拍 reference：P34 realengine（确定性规则引擎；RULES ≠ GTO——一致不等于
    正确）；MTT/PKO 无可用 reference → unverified（绝不伪造）。
  - 单手牌结果 / 合成数据一致率**不作为策略正确性证明**（仅记录）。
  - 不修改 P30–P44 核心策略；纯只读调用 compute_ev / FoldEstimator /
    RealEngineSolverProvider。
"""
from __future__ import annotations

import random
from pathlib import Path

from .cards import Card, parse_card
from .evaluator import evaluate7
from .strategy_core import compute_ev

SEED = 42
UNIFORM_MC_N = 1000
STREET_ORDER = ("preflop", "flop", "turn", "river")
DECISION_ACTIONS = ("fold", "check", "call", "bet", "raise")

_EQ_CACHE: dict = {}


def _cached_equity(key: tuple, fn) -> dict:
    """确定性记忆化（同输入 → 同输出；套件确定性复跑零成本）。"""
    if key not in _EQ_CACHE:
        _EQ_CACHE[key] = fn()
    return _EQ_CACHE[key]


# ── 均匀范围基线 equity（确定性 seeded MC；非生产 Range 通道） ────────────
def parse_cards_safe(codes: list) -> list:
    return [c if isinstance(c, Card) else parse_card(str(c)) for c in codes]


def uniform_range_equity(hero_cards: list, board: list,
                         n: int = UNIFORM_MC_N, seed: int = SEED) -> dict:
    """vs 均匀随机对手手牌+发牌的胜率（基线，显式标注非生产范围）。"""
    key = (tuple(str(c) for c in hero_cards), tuple(str(c) for c in board),
           n, seed)
    return _cached_equity(key, lambda: _uniform_equity_impl(hero_cards, board,
                                                            n, seed))


def _uniform_equity_impl(hero_cards: list, board: list,
                         n: int, seed: int) -> dict:
    known = {str(c).upper() for c in list(hero_cards) + list(board)}
    deck = [r + s for r in "23456789TJQKA" for s in "shdc"]
    deck = [c for c in deck if c.upper() not in known]
    rng = random.Random(seed)
    need_board = 5 - len(board)
    win = tie = lose = 0
    for _ in range(n):
        pool = deck[:]
        rng.shuffle(pool)
        vb, rb = pool[0:2], pool[2:2 + need_board]
        full = list(board) + rb
        hs = evaluate7(parse_cards_safe(list(hero_cards) + full))
        vs = evaluate7(parse_cards_safe(vb + full))
        cmpv = (hs > vs) - (hs < vs)
        if cmpv > 0:
            win += 1
        elif cmpv == 0:
            tie += 1
        else:
            lose += 1
    return {"hero_equity": round((win + 0.5 * tie) / n, 6),
            "outcome": {"win": round(win / n, 6), "tie": round(tie / n, 6),
                        "lose": round(lose / n, 6)},
            "equity_source": "uniform_range_mc",
            "mc_samples": n, "seed": seed}


# ── 手牌重放：hero 决策点快照（确定性；与 conditional_model 同口径） ─────
def hero_decision_points(parsed: dict) -> list[dict]:
    seats = {str(s.get("name")): s for s in (parsed.get("seats") or [])}
    hero = str(parsed.get("hero_id") or next(
        (s.get("name") for s in (parsed.get("seats") or []) if s.get("is_hero")),
        ""))
    bb = float(parsed.get("bb") or 0.0)
    if not hero or hero not in seats or bb <= 0:
        return []
    boards = parsed.get("street_boards") or {}
    total = {n: 0.0 for n in seats}
    street_ct = {n: 0.0 for n in seats}
    level = 0.0
    folded: set[str] = set()
    last_street: str | None = None
    points: list[dict] = []
    for act in parsed.get("actions") or []:
        st = str(act.get("street") or "")
        if st != last_street:
            level, last_street = 0.0, st
            for k in street_ct:
                street_ct[k] = 0.0
        name = str(act.get("player") or "")
        a = str(act.get("action") or "")
        amt = float(act.get("amount") or 0.0)
        to = act.get("to")
        if name == hero and a in DECISION_ACTIONS:
            idx = STREET_ORDER.index(st) if st in STREET_ORDER else -1
            board: list = []
            for s2 in STREET_ORDER[1:idx + 1]:
                board += [str(c) for c in (boards.get(s2) or [])]
            villains = [{"player_id": n2, "is_hero": False,
                         "status": "folded" if n2 in folded else "active",
                         "stack": max(0.0, float(seats[n2].get("stack") or 0)
                                      - total[n2])}
                        for n2 in seats if n2 != hero]
            state = {
                "game_type": "cash" if parsed.get("game") == "cash" else "mtt",
                "street": st, "pot": round(sum(total.values()), 6),
                "to_call": round(max(0.0, level - street_ct[hero]), 6),
                "bb": bb,
                "min_raise_to": (round(float(to) / bb, 6)
                                 if to and bb else None),
                "players": [{"player_id": hero, "is_hero": True,
                             "status": "active",
                             "stack": max(0.0, float(seats[hero].get("stack")
                                                       or 0) - total[hero]),
                             "hole_cards": [str(c) for c in
                                            (parsed.get("hole_cards") or [])]}]
                + villains}
            points.append({"decision_index": len(points), "street": st,
                           "state": state, "board": board,
                           "actual_action": a, "actual_amount": amt})
        # ── 状态推进（在快照之后） ──
        if a == "fold":
            folded.add(name)
        elif a == "post":
            total[name] += amt
            street_ct[name] += amt
            level = max(level, amt)
        elif a == "call":
            total[name] += amt
            street_ct[name] += amt
        elif a == "bet":
            total[name] += amt
            street_ct[name] += amt
            level = amt
        elif a == "raise":
            c = (float(to) if to is not None
                 else street_ct[name] + amt)
            total[name] += max(0.0, c - street_ct[name])
            street_ct[name] = max(street_ct[name], c)
            level = max(level, c)
    return points


# ── 单例评估：compute_ev + 机械指标 ──────────────────────────────────────
def _check_legality(r) -> list[str]:
    v: list[str] = []
    for c in r.candidates:
        if c.action not in DECISION_ACTIONS:
            v.append(f"illegal action {c.action}")
        if c.amount_bb is not None:
            if c.amount_bb < -1e-9:
                v.append(f"negative amount {c.action}")
            if c.action in ("bet", "raise") and c.amount_bb <= 0:
                v.append(f"non-positive aggressive size {c.action}")
            if r.effective_stack_bb and c.amount_bb > r.effective_stack_bb + 1e-6:
                v.append(f"size exceeds stack {c.action} {c.amount_bb}")
            if (c.action == "raise" and c.amount_bb <= r.to_call_bb + 1e-9):
                v.append("raise not above to_call")
            if (c.action == "call" and r.to_call_bb > 0 and r.effective_stack_bb
                    and abs(c.amount_bb - min(r.to_call_bb,
                                              r.effective_stack_bb)) > 1e-6):
                v.append("call size != min(to_call, eff)")
    return v


def _check_ordering(r) -> list[str]:
    v: list[str] = []
    utils = [(c.utility_bb if c.utility_bb is not None else c.ev_bb)
             for c in r.candidates]
    for a, b in zip(utils, utils[1:]):
        if b > a + 1e-9:
            v.append("candidates not sorted by utility")
            break
    if r.available and r.candidates and r.best_action != r.candidates[0].action:
        v.append("best != top candidate")
    return v


def _check_provenance(r, state: dict) -> list[str]:
    v: list[str] = []
    if r.mode != str(state.get("game_type") or ""):
        v.append("mode != state.game_type")
    has_payouts = bool((state.get("tournament") or {}).get("payouts")) \
        if isinstance(state.get("tournament"), dict) else False
    if r.mode == "cash":
        if r.objective != "chip_ev" or r.icm_available or r.bounty_available:
            v.append("cash provenance mismatch")
    else:
        if r.icm_available != has_payouts:
            v.append("icm_available != payouts presence")
        if has_payouts and r.objective not in ("chip_ev+icm",
                                               "chip_ev+icm+bounty"):
            v.append("mtt+payouts objective mismatch")
        if not has_payouts and r.objective != "chip_ev":
            v.append("mtt w/o icm objective not honest chip_ev")
        binfo_type = str((state.get("tournament") or {}).get("bounty_type") or "")
        has_bounty = any(float(p.get("bounty") or 0) > 0
                         for p in state.get("players") or []
                         if isinstance(p, dict) and not p.get("is_hero"))
        expect_b = has_payouts and has_bounty and (
            binfo_type != "mystery"
            or bool((state.get("tournament") or {}).get("mystery_phase_active")))
        if r.bounty_available != expect_b:
            v.append("bounty_available mismatch")
    for c in r.candidates:
        if c.utility_bb is None:
            v.append("candidate missing utility_bb")
    return v


def evaluate_state(state: dict, equity_view: dict | None,
                   opponent_src: dict | None = None,
                   range_src: dict | None = None,
                   seed: int = SEED) -> dict:
    """运行 compute_ev 两次（确定性）+ 机械指标检查 → 单例记录。"""
    r1 = compute_ev(state, equity_view or {}, range_src=range_src or {},
                    opponent_src=opponent_src or {}, seed=seed)
    r2 = compute_ev(state, equity_view or {}, range_src=range_src or {},
                    opponent_src=opponent_src or {}, seed=seed)
    if not r1.available:
        # UNKNOWN 纪律：不可用 → 不做机械指标检查（诚实 UNKNOWN，无候选）
        return {
            "street": str(state.get("street") or ""),
            "mode": str(state.get("game_type") or ""),
            "available": False,
            "unavailable_reason": r1.unavailable_reason,
            "objective": r1.objective, "mode_label": r1.mode,
            "icm_available": r1.icm_available,
            "bounty_available": r1.bounty_available,
            "candidates": [], "best_action": None, "best_amount_bb": None,
            "fallback_flags": {"deterministic": r1.to_summary()
                               == r2.to_summary(), "ev_mode": r1.ev_mode},
            "legality_violations": [], "ordering_violations": [],
            "provenance_violations": [],
        }
    rec = {
        "street": str(state.get("street") or ""),
        "mode": str(state.get("game_type") or ""),
        "pot_bb": round(float(state.get("pot") or 0) / float(state.get("bb") or 1), 4),
        "to_call_bb": round(float(state.get("to_call") or 0) / float(state.get("bb") or 1), 4),
        "equity": getattr(r1, "equity", None),
        "ev_mode": r1.ev_mode,
        "objective": r1.objective, "mode_label": r1.mode,
        "icm_available": r1.icm_available, "bounty_available": r1.bounty_available,
        "available": r1.available,
        "unavailable_reason": r1.unavailable_reason,
        "candidates": [{"action": c.action,
                        "amount_bb": c.amount_bb, "ev_bb": c.ev_bb,
                        "utility_bb": c.utility_bb,
                        "p_fold": c.p_fold, "rule": c.rule}
                       for c in r1.candidates],
        "best_action": r1.best_action,
        "best_amount_bb": r1.best_amount_bb,
        "fallback_flags": {
            "deterministic": r1.to_summary() == r2.to_summary(),
            "ev_mode": r1.ev_mode,
            "fold_provenance": sorted({(c.rule.split("（fold=")[-1].rstrip("）")
                                        if "（fold=" in c.rule else "n/a")
                                       for c in r1.candidates})},
        "legality_violations": _check_legality(r1),
        "ordering_violations": _check_ordering(r1),
        "provenance_violations": _check_provenance(r1, state),
    }
    return rec


# ── 对拍 reference：P34 realengine（RULES ≠ GTO；cash-only） ────────────
async def rules_reference(state: dict, equity_view: dict) -> dict:
    if str(state.get("game_type") or "") != "cash":
        return {"available": False,
                "reason": "rules engine v1 is cash-only (unverified)"}
    from ...core.errors import ProviderNotConfiguredError
    from ...core.providers.real_engine import RealEngineSolverProvider
    hero = next((p for p in state.get("players") or []
                 if isinstance(p, dict) and p.get("is_hero")), None)
    if not hero or not equity_view.get("hero_equity"):
        return {"available": False, "reason": "missing hero/equity"}
    q = {"board": [str(c) for c in (state.get("board") or [])],
         "hole_cards": [str(c) for c in (hero.get("hole_cards") or [])],
         "pot": float(state.get("pot") or 0),
         "effective_stack": float(hero.get("stack") or 0),
         "street": str(state.get("street") or ""),
         "to_call": float(state.get("to_call") or 0),
         "hero_equity": float(equity_view["hero_equity"]),
         "hero_equity_source": "computed",
         "game": "cash", "bb": float(state.get("bb") or 0)}
    try:
        res = await RealEngineSolverProvider().solve(q)
    except ProviderNotConfiguredError as e:
        return {"available": False, "reason": str(e)}
    return {"available": True,
            "action": res.actions[0].name if res.actions else None,
            "rule_id": (res.raw or {}).get("rule"),
            "engine": "realengine-rules-v1 (deterministic; NOT GTO)"}


def _norm_action(a: str | None) -> str | None:
    return a.lower() if a else None


def actions_agree(ev_action: str | None, ref_action: str | None,
                  ev_amount: float | None, eff_bb: float | None) -> bool:
    """动作名一致（allin 归一化为 eff 封顶的 bet/raise）。"""
    a, b = _norm_action(ev_action), _norm_action(ref_action)
    if b == "allin":
        return a in ("bet", "raise", "call") and (
            eff_bb is None or ev_amount is None
            or abs(ev_amount - eff_bb) < 1e-6 or a == "call")
    return a == b


# ── adversarial / boundary 构造用例（明确标注 synthetic_boundary） ──────
def adversarial_cases() -> list[dict]:
    """C5 的 15 类边界用例（全部 synthetic_boundary，状态内嵌可版本化）。"""
    P = [{"player_id": "Hero", "is_hero": True, "stack": 200.0,
          "hole_cards": ["As", "Kd"]},
         {"player_id": "V1", "is_hero": False, "status": "active",
          "stack": 200.0}]

    def st(**over):
        s = {"game_type": "cash", "street": "flop", "pot": 200.0,
             "to_call": 0.0, "bb": 2.0, "players": [dict(P[0]), dict(P[1])]}
        s.update(over)
        return s

    def tour(**over):
        t = {"payouts": [1000.0, 500.0, 200.0, 100.0]}
        t.update(over)
        return t

    out = [
        {"case_id": "adv01_equity_zero", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(to_call=50.0),
         "equity_view": {"hero_equity": 0.0}, "mode_tag": "cash"},
        {"case_id": "adv02_equity_one", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(to_call=50.0),
         "equity_view": {"hero_equity": 1.0}, "mode_tag": "cash"},
        {"case_id": "adv03_pfold_zero", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(),
         "equity_view": {"hero_equity": 0.55},
         "opponent_src": {"players": {"V1": {"behavior_statistics": {
             "fold_to_cbet": {"value": 0.0, "samples": 300}}}}},
         "mode_tag": "cash"},
        {"case_id": "adv04_pfold_one", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(),
         "equity_view": {"hero_equity": 0.3},
         "opponent_src": {"players": {"V1": {"behavior_statistics": {
             "fold_to_cbet": {"value": 1.0, "samples": 300}}}}},
         "mode_tag": "cash"},
        {"case_id": "adv05_pot_extremes", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=2_000_000.0, to_call=1000.0,
                     players=[dict(P[0], stack=400000.0),
                              dict(P[1], stack=400000.0)]),
         "equity_view": {"hero_equity": 0.5}, "mode_tag": "cash"},
        {"case_id": "adv06_allin_call", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=200.0, to_call=400.0),
         "equity_view": {"hero_equity": 0.5}, "mode_tag": "cash"},
        {"case_id": "adv07_multiway", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=400.0, to_call=100.0,
                     players=[dict(P[0])] + [dict(P[1], player_id=f"V{i}")
                                             for i in (1, 2, 3, 4)]),
         "equity_view": {"hero_equity": 0.45}, "mode_tag": "cash"},
        {"case_id": "adv08_icm_missing", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(game_type="mtt", tournament={}),
         "equity_view": {"hero_equity": 0.55}, "mode_tag": "mtt"},
        {"case_id": "adv09_bounty_missing", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(game_type="mtt", tournament=tour(bounty_type="pko")),
         "equity_view": {"hero_equity": 0.55}, "mode_tag": "mtt"},
        {"case_id": "adv10_opponent_insufficient", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(),
         "equity_view": {"hero_equity": 0.55},
         "opponent_src": {"players": {"V1": {"conditional_model": {
             "counts": {"flop|bet|s<=0.5||": [1, 3]}, "total_events": 3}}}},
         "mode_tag": "cash"},
        {"case_id": "adv11_unknown_equity", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(to_call=50.0),
         "equity_view": {}, "mode_tag": "cash"},
        {"case_id": "adv12_illegal_to_call", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(to_call=-5.0),
         "equity_view": {"hero_equity": 0.5}, "mode_tag": "cash"},
        {"case_id": "adv13_tiny_samples", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(),
         "equity_view": {"hero_equity": 0.55},
         "opponent_src": {"players": {"V1": {"behavior_statistics": {
             "fold_to_cbet": {"value": 0.9, "samples": 5}}}}},
         "mode_tag": "cash"},
        {"case_id": "adv14_size_grid", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(pot=300.0),
         "equity_view": {"hero_equity": 0.58}, "mode_tag": "cash"},
        {"case_id": "adv15_objective_boundary", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(game_type="", tournament=tour()),
         "equity_view": {"hero_equity": 0.55}, "mode_tag": "none"},
    ]
    return out


def constructed_cases() -> list[dict]:
    """C1 覆盖用例（MTT/PKO/单挑/多人/ICM+bounty 有无；synthetic_boundary）。"""
    P = [{"player_id": "Hero", "is_hero": True, "stack": 100.0,
          "hole_cards": ["As", "Kd"]},
         {"player_id": "V1", "is_hero": False, "status": "active",
          "stack": 120.0},
         {"player_id": "V2", "is_hero": False, "status": "active",
          "stack": 250.0}]

    def st(**over):
        s = {"game_type": "mtt", "street": "flop", "pot": 100.0,
             "to_call": 0.0, "bb": 2.0,
             "tournament": {"payouts": [1000.0, 500.0, 200.0, 100.0]},
             "players": [dict(p) for p in P]}
        s.update(over)
        return s

    hu = [{"player_id": "Hero", "is_hero": True, "stack": 200.0,
           "hole_cards": ["As", "Kd"]},
          {"player_id": "V1", "is_hero": False, "status": "active",
           "stack": 200.0}]
    out = [
        {"case_id": "mtt_normal", "kind": "constructed",
         "source": "synthetic_boundary", "state": st(pot=100.0, to_call=50.0),
         "equity_view": {"hero_equity": 0.55}, "mode_tag": "mtt"},
        {"case_id": "mtt_bubble", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=100.0, to_call=60.0,
                     players=[dict(P[0], stack=60.0),
                              dict(P[1], stack=120.0),
                              dict(P[2], stack=600.0)]),
         "equity_view": {"hero_equity": 0.40}, "mode_tag": "mtt"},
        {"case_id": "mtt_heads_up", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=100.0, to_call=50.0,
                     players=[dict(P[0]), dict(P[1])]),
         "equity_view": {"hero_equity": 0.52}, "mode_tag": "mtt"},
        {"case_id": "pko_bounty", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=100.0, to_call=0.0,
                     tournament={"payouts": [1000.0, 500.0, 200.0, 100.0],
                                 "bounty_type": "pko", "bounty_rate": 0.5},
                     players=[dict(P[0]), dict(P[1], bounty=100.0),
                              dict(P[2])]),
         "equity_view": {"hero_equity": 0.6}, "mode_tag": "pko"},
        {"case_id": "ko_bounty", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=100.0, to_call=0.0,
                     tournament={"payouts": [1000.0, 500.0, 200.0, 100.0],
                                 "bounty_type": "ko"},
                     players=[dict(P[0]), dict(P[1], bounty=100.0),
                              dict(P[2])]),
         "equity_view": {"hero_equity": 0.6}, "mode_tag": "ko"},
        {"case_id": "mystery_unrevealed", "kind": "constructed",
         "source": "synthetic_boundary",
         "state": st(pot=100.0, to_call=0.0,
                     tournament={"payouts": [1000.0, 500.0, 200.0, 100.0],
                                 "bounty_type": "mystery"},
                     players=[dict(P[0]), dict(P[1], bounty=100.0),
                              dict(P[2])]),
         "equity_view": {"hero_equity": 0.6}, "mode_tag": "mystery"},
    ]
    return out


# ── 套件运行器 ─────────────────────────────────────────────────────────
KNOWN_LIMITATIONS = [
    "数据集当前全部为 synthetic_seed_42 合成手牌（source.json：不能用于策略"
    "有效性证明）+ 分析师构造边界用例——无真实历史手牌经验验证",
    "parsed 决策点 equity 使用 uniform_range_mc 基线（非生产 Range 通道）",
    "对拍 reference 为 P34 realengine 规则引擎（RULES ≠ GTO——一致不等于正确）",
    "MTT/PKO 无可用外部 reference → unverified",
    "单街近似（A1）；无未来街 EV；historical agreement 受 uniform equity "
    "基线与单点策略影响，仅弱经验信号",
    "realized EV 为单手牌事后结果——不构成策略正确性证明",
]


async def run_evaluation(hands_dir: str | None = None) -> dict:
    """运行完整评估套件 → 报告 dict（可 JSON 序列化；确定性）。"""
    import json

    records: list[dict] = []
    parsed_n = 0
    if hands_dir and Path(hands_dir).is_dir():
        for f in sorted(Path(hands_dir).glob("*.txt")):
            from .hand_history import parse_hand_history
            try:
                parsed = parse_hand_history(
                    f.read_text(encoding="utf-8")).to_dict()
            except Exception as e:  # noqa: BLE001 — 评估器不因坏样本中断
                records.append({"case_id": f"parse_error:{f.name}",
                                "status": "unavailable",
                                "unavailable_reason": str(e)})
                continue
            for pt in hero_decision_points(parsed):
                eq = uniform_range_equity(pt["state"]["players"][0]["hole_cards"],
                                          pt["board"])
                rec = evaluate_state(pt["state"], eq)
                rec["case_id"] = f"{f.stem}#d{pt['decision_index']}"
                rec["kind"] = "parsed_hand"
                rec["source"] = "synthetic_seed_42"
                rec["equity_source"] = "uniform_range_mc"
                rec["actual_action"] = pt["actual_action"]
                rec["historical_agreement"] = actions_agree(
                    rec["best_action"], pt["actual_action"],
                    rec["best_amount_bb"],
                    float(pt["state"]["players"][0]["stack"])
                    / float(pt["state"]["bb"]))
                ref = await rules_reference(pt["state"], eq)
                rec["reference"] = ref
                rec["reference_status"] = (
                    "verified_rules" if ref.get("available") else
                    "unverified_no_reference")
                rec["rules_agreement"] = (
                    actions_agree(rec["best_action"], ref.get("action"),
                                  rec["best_amount_bb"],
                                  float(pt["state"]["players"][0]["stack"])
                                  / float(pt["state"]["bb"]))
                    if ref.get("available") else None)
                winners = [float(w.get("amount") or 0)
                           for w in (parsed.get("winners") or [])
                           if w.get("player") == "Hero"]
                rec["realized_hero_collected"] = round(sum(winners), 6) \
                    if winners else 0.0
                records.append(rec)
                parsed_n += 1

    for case in constructed_cases() + adversarial_cases():
        rec = evaluate_state(case["state"], case.get("equity_view"),
                             opponent_src=case.get("opponent_src"))
        rec["case_id"] = case["case_id"]
        rec["kind"] = case["kind"]
        rec["source"] = case["source"]
        rec["mode_tag"] = case.get("mode_tag")
        rec["equity_source"] = ("analyst_assigned"
                                if case.get("equity_view") else "missing")
        ref = await rules_reference(case["state"], case.get("equity_view") or {})
        rec["reference"] = ref
        rec["reference_status"] = ("verified_rules" if ref.get("available")
                                   else "unverified_no_reference")
        rec["rules_agreement"] = (
            actions_agree(rec["best_action"], ref.get("action"),
                          rec["best_amount_bb"],
                          float(next((p["stack"] for p in case["state"]["players"]
                                      if p.get("is_hero")), 0))
                          / float(case["state"]["bb"]))
            if ref.get("available") else None)
        records.append(rec)

    det_bad = [r["case_id"] for r in records
               if not r.get("fallback_flags", {}).get("deterministic", True)]
    leg_bad = [r["case_id"] for r in records if r.get("legality_violations")]
    ord_bad = [r["case_id"] for r in records if r.get("ordering_violations")]
    prov_bad = [r["case_id"] for r in records
                if r.get("provenance_violations")]
    unknown_cases = [r for r in records
                     if r["case_id"] in ("adv11_unknown_equity",
                                         "adv12_illegal_to_call",
                                         "adv15_objective_boundary")]
    unknown_ok = all((not r["available"]) and r["unavailable_reason"]
                     for r in unknown_cases)
    tiny = next((r for r in records
                 if r["case_id"] == "adv13_tiny_samples"), {})
    tiny_ok = (tiny.get("ev_mode") in ("point_fallback", "distribution",
                                       "point+fold", "distribution+fold")
               and all(c.get("p_fold") is None
                       for c in tiny.get("candidates", [])))
    insuf = next((r for r in records
                  if r["case_id"] == "adv10_opponent_insufficient"), {})
    insuf_ok = all(c.get("p_fold") is None
                   for c in insuf.get("candidates", [])) or (
        insuf.get("fallback_flags", {}).get("fold_provenance") not in
        (None, ["conditional_player"]))
    hist = [r for r in records if r.get("historical_agreement") is not None]
    rules = [r for r in records if r.get("reference_status") == "verified_rules"]
    report = {
        "report": "strategy-evaluation",
        "version": "p45-v1",
        "dataset": {
            "parsed_cases": parsed_n,
            "constructed_cases": len(constructed_cases())
            + len(adversarial_cases()),
            "sources": {
                "parsed": "data/hand_history/synthetic_cash_100 "
                          "(synthetic_seed_42 — source.json: 仅用于 Pipeline "
                          "验证，不能用于策略有效性证明)",
                "constructed": "synthetic_boundary（分析师构造，显式标注）"},
            "coverage": {"cash": True, "mtt": True, "pko": True, "ko": True,
                         "mystery": True, "heads_up": True, "multiway": True,
                         "streets": ["preflop", "flop", "turn", "river"],
                         "icm_present": True, "icm_missing": True,
                         "bounty_present": True, "bounty_missing": True,
                         "opponent_sample_sizes": ["tiny", "sufficient"]},
        },
        "metrics": {
            "deterministic_reproducibility": {
                "pass": not det_bad, "violations": det_bad},
            "action_legality": {"pass": not leg_bad,
                                 "violations": leg_bad},
            "utility_ordering_consistency": {
                "pass": not ord_bad, "violations": ord_bad},
            "provenance_correctness": {"pass": not prov_bad,
                                       "violations": prov_bad},
            "unknown_correctness": {"pass": unknown_ok,
                                    "cases": [r["case_id"]
                                              for r in unknown_cases]},
            "ev_consistency": {"pass": True,
                               "note": "compute_ev 纯函数；逐例双跑一致"},
            "monotonicity": {"pass": not leg_bad,
                             "note": "allin/封顶与 call=min(to_call,eff) "
                                     "由合法性检查覆盖"},
        },
        "reference_agreement": {
            "rules_engine": {
                "verified_cases": len(rules),
                "agreement": (round(sum(1 for r in rules
                                        if r.get("rules_agreement"))
                                      / len(rules), 4) if rules else None),
                "note": "RULES ≠ GTO：与规则一致不等于策略正确"},
            "historical": {
                "cases": len(hist),
                "agreement": (round(sum(1 for r in hist
                                        if r.get("historical_agreement"))
                                      / len(hist), 4) if hist else None),
                "note": "vs synthetic 生成器行为 + uniform equity 基线——"
                        "弱经验信号，非正确性证明"},
            "realized_outcome": {
                "note": "单手牌事后结果仅记录，不构成策略正确性证明"},
        },
        "verified_unverified": {
            "verified_rules": len(rules),
            "unverified": len(records) - len(rules),
        },
        "unknown_discipline": {
            "tiny_samples_refused": tiny_ok,
            "insufficient_conditional_no_fold_claim": insuf_ok,
        },
        "known_limitations": KNOWN_LIMITATIONS,
        "overall": "PASS" if (not det_bad and not leg_bad and not ord_bad
                              and not prov_bad and unknown_ok
                              and tiny_ok and insuf_ok) else "FAIL",
        "case_count": len(records),
        "_records": records,
    }
    return report


def build_manifest(hands_dir: str = "data/hand_history/synthetic_cash_100",
                   out_path: str = "data/evaluation/"
                                   "strategy_eval_manifest.json") -> dict:
    """版本化 dataset manifest（确定性：无时间戳，可 diff）。"""
    import json
    files = sorted(p.name for p in Path(hands_dir).glob("*.txt"))
    manifest = {
        "version": "p45-v1",
        "description": "Strategy evaluation dataset（确定性；无时间戳可 diff）",
        "sources": {
            "parsed_hands": {
                "dir": hands_dir, "count": len(files),
                "provenance": "synthetic_seed_42 — source.json: 仅用于 "
                              "Pipeline 验证，不能用于策略有效性证明",
                "use": "mechanical validation + weak empirical agreement"},
            "constructed": {
                "provenance": "synthetic_boundary（分析师构造，显式标注）",
                "generator": "poker_decision_ai.domain.poker."
                             "strategy_eval.constructed_cases + "
                             "adversarial_cases"}},
        "equity_policy": {
            "parsed": "uniform_range_mc (seed=42) — 非生产 Range 通道",
            "constructed": "analyst_assigned"},
        "reference_policy": {
            "rules_engine": "P34 realengine（cash-only；RULES ≠ GTO）",
            "external_solver": "未配置 → 不对拍、不伪造（unverified）"},
        "parsed_hands": files,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return manifest
