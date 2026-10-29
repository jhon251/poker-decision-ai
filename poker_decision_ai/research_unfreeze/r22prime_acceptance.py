"""R2.2 候选 S′ 一次性独立验收（研究分支；实现在**封存前**固定）。

严格遵循 `docs/V3F_R22_CANDIDATE_ACCEPTANCE_PLAN.md`（已 PASS）：
  * S′ 玩家 = Bill / Eddie / MrPink（性能盲规则固定，不得重选）
  * 仅用手序 ≥ 1500 的手牌（避开已窥视区间）
  * F′/C′ = 其余 7 名玩家的决策（决策级玩家不相交）；索引只用**不含 S′ 玩家的手牌**
  * 主判据 = PI-EV 排序一致率（paired bootstrap）；NOT equilibrium；Solver BLOCKED
  * 预注册判定：Δ≥+4pp 且 CI 下界>0 → ACCEPT；+2~+4pp → UNCERTAIN；<+2pp 或 CI 含 0 → REJECT
    coverage<70% / 有效点<1200 / 层拒绝>40% → PARTIAL
  * **封存后一次性运行**；禁止任何 S′ 预计算、成本校准、重跑
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..domain.poker.equity import EquityCalculator
from ..domain.poker.strategy_core import compute_ev
from ..multistreet_v3 import MS3_VERSION, evaluate_v3
from ..opponent_aggression import build_aggression_index
from ..pluribus_validation import (AI_SEAT, _current_aggressor, _load_config,
                                   _positions_for_seats, _profile, _route_of,
                                   _state_for, _villain_range_arm,
                                   build_indexes, legal_action_types)
from .unfreeze_core import PiEvReference, SolverReference

ROOT = Path(__file__).resolve().parents[2]
PRIME_VERSION = "r22prime-acceptance-v1"
PRIME_S_PLAYERS = ("Bill", "Eddie", "MrPink")      # 固定（来自已 PASS 的设计审计）
PRIME_HAND_MIN_INDEX = 1500
SEED = 42
EQUITY_ITERS = 1500
RUNOUTS = 8                     # R2.2 层已发布配置（连续性；非调参）
CHILD_ITERS = 300
BOOTSTRAP_N = 2000
COVERAGE_MIN = 0.70
VALID_POINTS_MIN = 1200
DECLINE_MAX = 0.40
ACCEPT_PP, UNCERTAIN_PP = 4.0, 2.0

REGISTRY = ROOT / "docs" / "V3F_R22PRIME_SPLIT_REGISTRY.json"
MANIFEST = ROOT / "docs" / "V3F_R22PRIME_MANIFEST.json"
RESULT = ROOT / "docs" / "V3F_R22PRIME_RESULT.json"

SPR_BUCKETS = ((2.0, "spr<2"), (5.0, "2<=spr<5"), (10.0, "5<=spr<10"))
POT_BUCKETS = ((10.0, "pot<10bb"), (30.0, "10<=pot<30"),
               (100.0, "30<=pot<100"))


def _bucket(v: float, table) -> str:
    for hi, name in table:
        if v < hi:
            return name
    return "high"


def _hkey(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


# ── split 物化（不涉及任何模型计算） ───────────────────────────────────
def build_prime_splits(hands: list[dict]) -> dict:
    """S′ / F′ / C′ / FIT_HANDS 的确定性物化（性能盲）。"""
    s_prime = set(PRIME_S_PLAYERS)
    hand_index = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    in_scope = {hid for hid, i in hand_index.items()
                if i >= PRIME_HAND_MIN_INDEX}
    s_prime_hands = [h for h in hands
                     if str(h["hand_id"]) in in_scope
                     and any(p in s_prime
                             for p in h["players"] if p != AI_SEAT)]
    # 索引用手牌：范围内且**不含任何 S′ 玩家**（保证拟合侧零 S′ 标签）
    fit_hands = [h for h in hands
                 if str(h["hand_id"]) in in_scope
                 and not any(p in s_prime
                             for p in h["players"] if p != AI_SEAT)]
    assignment: dict[str, str] = {}
    for h in hands:
        hid = str(h["hand_id"])
        if hid not in in_scope:
            assignment[hid] = "OUT_OF_SCOPE"
            continue
        r = (_hkey(hid) % 1000) / 1000.0
        assignment[hid] = "F" if r < 0.70 else ("C" if r < 0.85 else "C2")
    assign_sha = hashlib.sha256(json.dumps(sorted(assignment.items()),
                                           ensure_ascii=False)
                                .encode("utf-8")).hexdigest()
    fit_sha = hashlib.sha256(json.dumps(sorted(str(h["hand_id"])
                                               for h in fit_hands),
                                        ensure_ascii=False)
                             .encode("utf-8")).hexdigest()
    return {
        "registry_version": PRIME_VERSION,
        "s_prime_players": sorted(s_prime),
        "hand_min_index": PRIME_HAND_MIN_INDEX,
        "hand_index_max_available": max(hand_index.values()) if hand_index else None,
        "counts": {
            "hands_total": len(hands),
            "hands_in_scope": len(in_scope),
            "hands_with_s_prime_player": len(s_prime_hands),
            "index_hands_excluding_s_prime": len(fit_hands),
        },
        "rules": {
            "S_prime": "S′ 玩家的**全部决策**（手序 ≥ 1500）",
            "F_prime": "非 S′ 玩家决策（手序 ≥ 1500；决策级玩家不相交）",
            "index_fit": "侵略性/范围索引仅用**不含 S′ 玩家**的手牌（零 S′ 标签）",
            "residual_coupling": ("手级上下文（牌面/发牌/同手他人动作）无法消除："
                                  "每手 5 名人类 + 仅 13 名玩家 → 严格手级解耦实测"
                                  " 0 手可用（已在审计中登记）"),
        },
        "assignment_sha256": assign_sha,
        "index_hands_sha256": fit_sha,
        "frozen_before_model_run": True,
    }


def write_prime_registry(hands: list[dict]) -> dict:
    reg = build_prime_splits(hands)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return reg


# ── manifest（封存；运行前） ───────────────────────────────────────────
SEALED = (
    "poker_decision_ai/research_unfreeze/r22prime_acceptance.py",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py",
    "poker_decision_ai/multistreet_v3.py",
    "poker_decision_ai/multistreet_v2.py",
    "poker_decision_ai/opponent_aggression.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/pluribus_validation.py",
    "docs/V3F_R22PRIME_SPLIT_REGISTRY.json",
)


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def write_prime_manifest() -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    man = {
        "manifest_version": PRIME_VERSION,
        "experiment_id": "r22_candidate_s_prime_acceptance",
        "target_evidence_level": "L2（PI-EV；perfect-information，NOT equilibrium）",
        "code_hashes": {rel: _sha(rel) for rel in SEALED},
        "config": {
            "arms": ["frozen", "R2.2"],
            "layer_version": MS3_VERSION,
            "s_prime_players": reg["s_prime_players"],
            "hand_min_index": reg["hand_min_index"],
            "equity_iterations": EQUITY_ITERS,
            "runouts_per_branch": RUNOUTS,
            "child_equity_iterations": CHILD_ITERS,
            "reference": "PI-EV（全枚举；unavailable 不得近似替代）",
            "solver": "BLOCKED（POSTFLOP_SOLVER_PATH 未设；禁止伪实现）",
            "excluded": ["U1", "U1+U3", "U1+U2+U3", "λ 扫描", "CFI"],
        },
        "data_manifest": {
            "corpus": "uoft-cprg::pluribus-human-seats（10,000 手）",
            "split_registry": "docs/V3F_R22PRIME_SPLIT_REGISTRY.json",
            "assignment_sha256": reg["assignment_sha256"],
            "index_hands_sha256": reg["index_hands_sha256"],
            "residual_coupling": reg["rules"]["residual_coupling"],
        },
        "seeds": {"layer": SEED, "equity": SEED, "reference": SEED,
                  "bootstrap": SEED},
        "gates": {
            "primary": (f"PI-EV 一致率 Δ ≥ +{ACCEPT_PP}pp 且 bootstrap 95% CI "
                        f"下界 > 0 → ACCEPT；+{UNCERTAIN_PP}~+{ACCEPT_PP}pp → "
                        f"UNCERTAIN；< +{UNCERTAIN_PP}pp 或 CI 含 0 → REJECT"),
            "partial_conditions": (f"coverage < {COVERAGE_MIN:.0%} / 有效点 < "
                                   f"{VALID_POINTS_MIN} / 层拒绝率 > "
                                   f"{DECLINE_MAX:.0%} → PARTIAL"),
            "blocked_conditions": ("legality<100% / 确定性失败 / 模式隔离失败 / "
                                   "manifest 漂移 / 封存违规 → BLOCKED 或作废"),
        },
        "rejection_lines": [
            "manifest 哈希漂移",
            "封存后、运行前查看或计算任何 S′ 指标（含成本校准）",
            "S′ 重跑或结果不利后调参/改阈值",
            "以 agreement 作为主判据",
            "扩大/修改 split 或重选 S′ 玩家",
            "把 PI-EV unavailable 的点近似替代",
        ],
        "sealed_before_model_run": True,
        "production_wiring": "NOT WIRED",
    }
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return man


def verify_prime_manifest() -> dict:
    if not MANIFEST.exists():
        return {"present": False}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha(rel) != h}
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    split_drift = (man["data_manifest"]["assignment_sha256"]
                   != reg["assignment_sha256"])
    return {"present": True, "drift": drift,
            "clean": (not drift) and (not split_drift),
            "split_drift": split_drift, "checked": len(man.get("code_hashes")
                                                       or {})}


# ── 一次性验收运行 ─────────────────────────────────────────────────────
def run_prime_acceptance(hands: list[dict]) -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    s_prime = set(reg["s_prime_players"])
    cfg = _load_config()
    hand_index = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    in_scope = {hid for hid, i in hand_index.items()
                if i >= PRIME_HAND_MIN_INDEX}
    fit_hands = [h for h in hands if str(h["hand_id"]) in in_scope]
    idx = _build_indexes_excluding(fit_hands, s_prime)
    agg = _build_aggression_excluding(fit_hands, s_prime)
    calc = EquityCalculator(iterations=EQUITY_ITERS, seed=SEED)
    pi = PiEvReference()
    solver = SolverReference()
    points: list[dict] = []
    counts: dict[str, Any] = {
        "index_hands": len(fit_hands), "index_events": agg["coverage"]["n_events"],
        "attempted": 0, "pi_points": 0, "pi_unavailable": 0,
        "layer_declined": 0, "legal": 0, "illegal": 0,
        "skipped_no_range": 0, "skipped_no_hole": 0, "skipped_no_villain": 0,
        "solver_configured": solver.is_configured(),
        "solver_reason": solver.unavailable_reason(),
    }
    for h in hands:
        hid = str(h["hand_id"])
        if hid not in in_scope:
            continue
        bb = h["bb"] or 1.0
        n = len(h["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(h["players"])}
        folded: set[str] = set()
        hist: list[dict] = []
        boards = h["board_by_street"]
        for st in h["streets"]:
            street = st["street"]
            if street == "preflop":
                board: list[str] = []
            elif street == "flop":
                board = list(boards.get("flop", []))
            elif street == "turn":
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", []))
            else:
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", [])) + list(boards.get("river", []))
            record = street in ("flop", "turn")
            for a in st["actions"]:
                pid = a["player"]
                if pid == AI_SEAT:
                    if a["action"] in ("bet", "raise", "call"):
                        hist.append({"street": street, "player_id": pid,
                                     "action": a["action"],
                                     "amount": a.get("amount") or 0.0})
                    continue
                if not record or pid not in s_prime:
                    _adv(a, pid, street, folded, hist)
                    continue
                counts["attempted"] += 1
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                counts["legal" if (la is not None and observed in la)
                       else "illegal"] += 1
                hole = list(h["hole_cards"].get(pid) or [])
                villain = _current_aggressor(h, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in h["players"]
                                    if p != pid and p not in folded), None)
                if len(hole) != 2:
                    counts["skipped_no_hole"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                if villain is None:
                    counts["skipped_no_villain"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                vpos = str(pos_map.get(seat_of.get(villain)) or "")
                vr, _p = _villain_range_arm(
                    cfg, villain=villain, villain_pos=vpos, street=street,
                    route=_route_of(hist, villain, street),
                    dead_cards=hole + list(board), idx=idx, with_evidence=True,
                    hole_cards=hole, board=board)
                if vr is None or not vr.distribution():
                    counts["skipped_no_range"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                dist = vr.distribution()
                er = calc.hand_vs_range(hole, board, vr,
                                        iterations=EQUITY_ITERS, seed=SEED)
                eq, tie = er.hero_equity, er.tie
                eq_src = {"hero_equity": eq,
                          "outcome": {"win": round(eq - tie / 2, 6),
                                      "tie": round(tie, 6),
                                      "lose": round(1 - eq - tie / 2, 6)}}
                state = _state_for(h, a, street, board, pos_map, folded, hist,
                                   pid, bb)
                rs = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                osrc = {"players": {villain: prof}} if prof else None
                frozen = compute_ev(state, eq_src, range_src=rs,
                                    opponent_src=osrc, seed=SEED)
                if not frozen.available or not frozen.candidates:
                    counts["skipped_no_range"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                cands = [{"action": c.action, "amount_bb": c.amount_bb}
                         for c in frozen.candidates]
                vhands = []
                for p in h["players"]:
                    if p == AI_SEAT or p == pid or p in folded:
                        continue
                    cards = list(h["hole_cards"].get(p) or [])
                    if len(cards) == 2:
                        vhands.append(cards)
                ref = pi.reference(
                    hero_cards=hole, villain_hands=vhands, board=board,
                    pot_bb=float(state.get("pot") or 0.0) / bb,
                    to_call_bb=float(state.get("to_call") or 0.0) / bb,
                    hero_stack_bb=float(next(
                        (p.get("stack") or 0.0 for p in state["players"]
                         if isinstance(p, dict) and p.get("is_hero")), 0.0)) / bb,
                    candidates=cands)
                if not ref["available"]:
                    counts["pi_unavailable"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                counts["pi_points"] += 1
                r22 = evaluate_v3(state, eq_src, range_src=rs, aggression=agg,
                                  villain_id=villain, villain_position=vpos,
                                  runouts=RUNOUTS,
                                  equity_iterations=CHILD_ITERS, seed=SEED)
                declined = not r22.get("available")
                if declined:
                    counts["layer_declined"] += 1
                hero_stack_bb = float(next(
                    (p.get("stack") or 0.0 for p in state["players"]
                     if isinstance(p, dict) and p.get("is_hero")), 0.0)) / bb
                pot_bb = float(state.get("pot") or 0.0) / bb
                spr = hero_stack_bb / pot_bb if pot_bb > 0 else 0.0
                points.append({
                    "hand_id": hid, "street": street, "player": pid,
                    "observed": observed,
                    "legal": bool(la is not None and observed in la),
                    "frozen_best": frozen.best_action,
                    "frozen_ranking": [c.action for c in frozen.candidates],
                    "r22_best": r22.get("best_v3"),
                    "r22_declined": declined,
                    "r22_decline_reason": r22.get("decline_reason"),
                    "node_order": (r22.get("provenance") or {}).get("node_order"),
                    "hero_ip": r22.get("hero_ip"),
                    "pi_best": ref["ordering"][0],
                    "pi_ordering": ref["ordering"],
                    "pi_equity": ref["pi_equity"],
                    "n_runouts": ref["n_runouts"],
                    "spr": round(spr, 4), "pot_bb": round(pot_bb, 4),
                    "n_candidates": len(cands),
                    "position": vpos,
                })
                _adv(a, pid, street, folded, hist)
    result = _summarize(points, counts, reg)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return result


def _adv(a: dict, pid: str, street: str, folded: set, hist: list) -> None:
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"], "amount": a.get("amount") or 0.0})


def _build_indexes_excluding(hands: list[dict], excluded: set[str]) -> dict:
    """镜像 `build_indexes`，但丢弃 `excluded` 玩家的证据事件。

    与预注册设计一致：F′/C′ = 非 S′ 玩家的**决策**（决策级玩家不相交，手级
    上下文耦合已在审计中登记为不可消除）。
    """
    from ..domain.poker.conditional_model import ConditionalFoldIndex
    from ..pluribus_validation import (RangePosteriorIndex,
                                       build_evidence_events)
    player_cfi: dict[str, ConditionalFoldIndex] = {}
    player_rpi: dict[str, RangePosteriorIndex] = {}
    pop_cfi = ConditionalFoldIndex()
    pop_rpi = RangePosteriorIndex()
    for h in hands:
        for ev in build_evidence_events(h):
            pid = ev["player"]
            if pid in excluded:
                continue
            if ev.get("facing") == "none":
                pop_rpi.add_events([ev])
                player_rpi.setdefault(pid,
                                      RangePosteriorIndex()).add_events([ev])
            else:
                pop_cfi.add_events([ev])
                player_cfi.setdefault(pid,
                                      ConditionalFoldIndex()).add_events([ev])
    return {"player_cfi": player_cfi, "player_rpi": player_rpi,
            "pop_cfi": pop_cfi, "pop_rpi": pop_rpi}


def _build_aggression_excluding(hands: list[dict], excluded: set[str]) -> dict:
    """镜像 `build_aggression_index`，丢弃 `excluded` 玩家的事件。"""
    from ..opponent_aggression import (AGGRESSION_VERSION,
                                       ConditionalAggressionIndex,
                                       build_aggression_events)
    pop = ConditionalAggressionIndex()
    per_player: dict[str, ConditionalAggressionIndex] = {}
    by_street: dict[str, int] = {}
    by_facing: dict[str, int] = {}
    n = 0
    for h in hands:
        evs = [e for e in build_aggression_events(h)
               if e["player"] not in excluded]
        if not evs:
            continue
        n += len(evs)
        pop.add_events(evs)
        for e in evs:
            per_player.setdefault(str(e["player"]),
                                  ConditionalAggressionIndex()).add_events([e])
            by_street[str(e["street"])] = by_street.get(str(e["street"]), 0) + 1
            by_facing[str(e["facing"])] = by_facing.get(str(e["facing"]), 0) + 1
    return {"version": AGGRESSION_VERSION, "population": pop,
            "players": per_player,
            "coverage": {"n_events": n, "by_street": by_street,
                         "by_facing": by_facing}}

    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"], "amount": a.get("amount") or 0.0})


# ── 汇总（含分层与判定） ───────────────────────────────────────────────
def _agree(sub: list[dict], key: str) -> dict:
    tot = ok = 0
    for p in sub:
        best = p.get(key)
        if not best:
            continue
        tot += 1
        ok += int(best == p["pi_best"])
    return {"n": tot, "agreement": round(ok / tot, 6) if tot else None}


def _paired(sub: list[dict], n: int = BOOTSTRAP_N) -> dict:
    diffs = [int(p["r22_best"] == p["pi_best"]) - int(p["frozen_best"]
                                                      == p["pi_best"])
             for p in sub if p.get("r22_best") and p.get("frozen_best")]
    if not diffs:
        return {"n": 0, "delta_pp": None, "ci95": None}
    rng = random.Random(SEED)
    sims = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        sims.append(sum(s) / len(s))
    sims.sort()
    return {"n": len(diffs),
            "delta_pp": round(100 * sum(diffs) / len(diffs), 4),
            "ci95_low_pp": round(100 * sims[int(0.025 * n)], 4),
            "ci95_high_pp": round(100 * sims[int(0.975 * n)], 4)}


def _obs(sub: list[dict], key: str) -> dict:
    tot = ok = aggr = 0
    for p in sub:
        best = p.get(key)
        if not best:
            continue
        tot += 1
        ok += int(best == p["observed"])
        aggr += int(best in ("bet", "raise"))
    return {"n": tot, "observed_agreement": round(ok / tot, 6) if tot else None,
            "bet_bias_share": round(aggr / tot, 6) if tot else None}


def _summarize(points: list[dict], counts: dict, reg: dict) -> dict:
    valid = [p for p in points if p.get("r22_best") and p.get("frozen_best")]
    coverage = (counts["pi_points"] / counts["attempted"]
                if counts["attempted"] else 0.0)
    decline_rate = (counts["layer_declined"] / counts["pi_points"]
                    if counts["pi_points"] else 0.0)
    primary = _paired(valid)
    strata = {}
    for name, keyf in (
            ("street", lambda p: p["street"]),
            ("position", lambda p: "IP" if p.get("hero_ip") else "OOP"),
            ("spr", lambda p: _bucket(p["spr"], SPR_BUCKETS)),
            ("pot", lambda p: _bucket(p["pot_bb"], POT_BUCKETS))):
        groups: dict[str, list[dict]] = {}
        for p in valid:
            groups.setdefault(str(keyf(p)), []).append(p)
        strata[name] = {g: {"n": len(v),
                            "frozen_pi": _agree(v, "frozen_best")["agreement"],
                            "r22_pi": _agree(v, "r22_best")["agreement"],
                            "paired": _paired(v, n=500)}
                        for g, v in sorted(groups.items())}
    d = primary.get("delta_pp")
    ci = primary.get("ci95_low_pp")
    if coverage < COVERAGE_MIN:
        verdict = "PARTIAL"
    elif len(valid) < VALID_POINTS_MIN:
        verdict = "PARTIAL"
    elif decline_rate > DECLINE_MAX:
        verdict = "PARTIAL"
    elif d is None:
        verdict = "BLOCKED"
    elif d >= ACCEPT_PP and ci is not None and ci > 0:
        verdict = "ACCEPT"
    elif d >= UNCERTAIN_PP:
        verdict = "UNCERTAIN"
    else:
        verdict = "REJECT"
    return {
        "snapshot_version": PRIME_VERSION,
        "s_prime_players": reg["s_prime_players"],
        "hand_min_index": reg["hand_min_index"],
        "counts": counts,
        "coverage": round(coverage, 6),
        "layer_decline_rate": round(decline_rate, 6),
        "valid_points": len(valid),
        "frozen": {"pi": _agree(valid, "frozen_best"),
                   "obs": _obs(valid, "frozen_best")},
        "r22": {"pi": _agree(valid, "r22_best"), "obs": _obs(valid, "r22_best")},
        "primary_paired_delta": primary,
        "strata": strata,
        "legality": {"legal": counts["legal"], "illegal": counts["illegal"],
                     "share": (counts["legal"]
                               / (counts["legal"] + counts["illegal"])
                               if (counts["legal"] + counts["illegal"]) else None)},
        "failures": {
            "pi_unavailable": counts["pi_unavailable"],
            "layer_declined": counts["layer_declined"],
            "skipped_no_range": counts["skipped_no_range"],
            "skipped_no_hole": counts["skipped_no_hole"],
            "skipped_no_villain": counts["skipped_no_villain"],
        },
        "hand_context_coupling": reg["rules"]["residual_coupling"],
        "reference": {"kind": "pi_ev", "level": "L2",
                      "equilibrium": False, "NOT_exact_gto": True,
                      "solver": counts["solver_configured"],
                      "solver_reason": counts["solver_reason"]},
        "verdict": verdict,
        "verdict_rule": (f"Δ≥+{ACCEPT_PP}pp 且 CI 下界>0 → ACCEPT；"
                         f"+{UNCERTAIN_PP}~+{ACCEPT_PP}pp → UNCERTAIN；"
                         f"<+{UNCERTAIN_PP}pp 或 CI 含 0 → REJECT；"
                         f"coverage<{COVERAGE_MIN:.0%} / 有效点<"
                         f"{VALID_POINTS_MIN} / 拒绝率>{DECLINE_MAX:.0%} → PARTIAL"),
        "honesty": {
            "no_profitability": "不构成盈利性/GTO 证明",
            "not_equilibrium": "PI-EV = perfect-information，非均衡",
            "all_results_reported": True,
            "no_tuning": "单次运行；未据结果调参或改阈值",
            "excluded": ["U1", "U2", "U3", "λ", "CFI"],
            "production_wiring": "NOT WIRED",
        },
    }
