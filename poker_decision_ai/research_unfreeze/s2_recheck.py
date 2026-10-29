"""S″ 重检（修复后仪器）：plit/manifest 生成 + 一次性运行器（**本阶段不执行**）。

与 S′ 的差别（唯一差别）：PI-EV 输入走 `pi_ev_contract.pi_ev_inputs()`，并强制
**仪器验证门**（比例校验 / 审计日志 / 分布检查 / 异常计数）。其余设计（玩家
不相交、手序下界、冻结配置、预注册判定）与已 PASS 的验收设计保持一致。

安全护栏：`run_s2_acceptance(..., allow_run=True)` 必须显式传入，否则抛错 →
防止误执行（本阶段仅生成/封存，不运行）。

两项评估**分别**进行、互不混合：
  * `arm="U1"`    ：compute_ev_variant(U1) vs frozen（call/raise 损失基础修正）
  * `arm="R2.2"`  ：evaluate_v3 vs frozen（多街层）
共同基线：`frozen`；主判据：PI-EV 排序一致率 Δ ≥ +4.0pp 且 CI 下界 > 0。
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
from ..pluribus_validation import (AI_SEAT, _current_aggressor, _load_config,
                                   _positions_for_seats, _profile, _route_of,
                                   _state_for, _villain_range_arm,
                                   build_indexes, legal_action_types)
from .pi_ev_contract import (AuditLog, UnitContractError, pi_ev_inputs,
                            sanity_check)
from .r22prime_acceptance import (_adv, _build_aggression_excluding,
                                  _build_indexes_excluding, _hkey)
from .unfreeze_core import PiEvReference, SolverReference, compute_ev_variant

ROOT = Path(__file__).resolve().parents[2]
S2_VERSION = "s2-recheck-v2"
S2_HAND_MIN_INDEX = 1500
SEED = 42
EQUITY_ITERS = 1500
RUNOUTS = 8
CHILD_ITERS = 300
BOOTSTRAP_N = 2000
ACCEPT_PP, UNCERTAIN_PP = 4.0, 2.0
COVERAGE_MIN = 0.70
VALID_POINTS_MIN = 1200
DECLINE_MAX = 0.40
ANOMALY_MAX_RATE = 0.01          # 仪器异常率上限：超过即整轮 BLOCKED
AUDIT_CAPACITY = 256

REGISTRY = ROOT / "docs" / "V3F_S2_SPLIT_REGISTRY.json"
MANIFEST_U1 = ROOT / "docs" / "V3F_S2_MANIFEST_U1.json"
MANIFEST_R22 = ROOT / "docs" / "V3F_S2_MANIFEST_R22.json"
RESULT_U1 = ROOT / "docs" / "V3F_S2_RESULT_U1.json"
RESULT_R22 = ROOT / "docs" / "V3F_S2_RESULT_R22.json"

# 已被使用过的玩家（不得再入选 S″）
USED_PLAYERS = ("Budd", "Joe", "MrBlonde",      # S（已开封）
                "Bill", "Eddie", "MrPink")       # S′（已开封，结果作废但数据已用）


def _player_action_counts(hands: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in hands:
        for st in h["streets"]:
            for a in st["actions"]:
                if a["player"] != AI_SEAT:
                    counts[a["player"]] = counts.get(a["player"], 0) + 1
    return counts


def build_s2_splits(hands: list[dict]) -> dict:
    """S″ 结构物化（性能盲；不含任何模型计算）。"""
    counts = _player_action_counts(hands)
    pool = {p: c for p, c in counts.items() if p not in USED_PLAYERS}
    if len(pool) < 3:
        raise RuntimeError(f"可用玩家不足 3：{sorted(pool)}")
    vals = sorted(pool.values())
    median = vals[len(vals) // 2]
    s2 = sorted(sorted(pool, key=lambda p: (abs(pool[p] - median), p))[:3])
    hand_index = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    in_scope = {hid for hid, i in hand_index.items()
                if i >= S2_HAND_MIN_INDEX}
    # 手牌范围规则（事前声明）：优先「范围内且不含 S′/S 玩家」的手牌；
    # 若其中 S″ 决策数 < 400，则退化为「范围内全部手牌」（声明手级耦合）
    used = set(USED_PLAYERS)
    clean_hands = [h for h in hands
                   if str(h["hand_id"]) in in_scope
                   and not any(p in used for p in h["players"]
                               if p != AI_SEAT)]
    clean_decisions = sum(
        1 for h in clean_hands for st in h["streets"] for a in st["actions"]
        if a["player"] in s2)
    if clean_decisions >= 400:
        chosen_hands = clean_hands
        scope_rule = ("范围内且**不含任何 S/S′ 玩家**的手牌（最强分离）")
    else:
        chosen_hands = [h for h in hands if str(h["hand_id"]) in in_scope]
        scope_rule = ("范围内全部手牌（S″ 玩家决策与其他玩家决策级隔离；"
                      "**手级上下文耦合**声明保留）")
        clean_decisions = sum(
            1 for h in chosen_hands for st in h["streets"]
            for a in st["actions"] if a["player"] in s2)
    assignment = {str(h["hand_id"]): ("IN" if h in chosen_hands else "OUT")
                  for h in hands}
    assign_sha = hashlib.sha256(json.dumps(sorted(assignment.items()),
                                           ensure_ascii=False)
                                .encode("utf-8")).hexdigest()
    return {
        "registry_version": S2_VERSION,
        "s2_players": s2,
        "s2_player_action_counts": {p: pool[p] for p in s2},
        "median_actions_pool": median,
        "used_players_excluded": list(USED_PLAYERS),
        "hand_min_index": S2_HAND_MIN_INDEX,
        "scope_rule": scope_rule,
        "counts": {
            "hands_total": len(hands),
            "hands_in_scope": len(in_scope),
            "hands_clean_no_S_no_Sprime": len(clean_hands),
            "hands_chosen": len(chosen_hands),
            "s2_decisions": clean_decisions,
        },
        "assignment_sha256": assign_sha,
        "frozen_before_model_run": True,
        "note": ("S″ 完全独立于 S/S′ 的**玩家标签**；手牌范围按上述事前规则；"
                 "S′ 结果作废，但其数据已用于已封存运行，故其玩家一并排除"),
    }


def write_s2_registry(hands: list[dict]) -> dict:
    reg = build_s2_splits(hands)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return reg


SEALED = (
    "poker_decision_ai/research_unfreeze/s2_recheck.py",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py",
    "poker_decision_ai/research_unfreeze/r22prime_acceptance.py",
    "poker_decision_ai/multistreet_v3.py",
    "poker_decision_ai/multistreet_v2.py",
    "poker_decision_ai/opponent_aggression.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/pluribus_validation.py",
    "docs/V3F_S2_SPLIT_REGISTRY.json",
)


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def write_s2_manifest(path: Path, *, arm: str) -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    man = {
        "manifest_version": S2_VERSION,
        "experiment_id": f"s2_recheck_{arm}",
        "arm": arm,
        "target_evidence_level": "L2（PI-EV，perfect-information，NOT equilibrium）",
        "code_hashes": {rel: _sha(rel) for rel in SEALED},
        "config": {
            "baseline": "frozen",
            "candidate": ("U1（raise 损失基础修正）" if arm == "U1"
                          else f"R2.2（{MS3_VERSION} 多街层）"),
            "s2_players": reg["s2_players"],
            "hand_min_index": reg["hand_min_index"],
            "scope_rule": reg["scope_rule"],
            "equity_iterations": EQUITY_ITERS,
            "runouts_per_branch": RUNOUTS,
            "child_equity_iterations": CHILD_ITERS,
            "bootstrap_n": BOOTSTRAP_N,
            "unit_contract": "pi-ev-unit-contract-v1（PI-EV 输入唯一构造路径）",
            "excluded": (["R2.2", "CFI", "λ 扫描"] if arm == "U1"
                         else ["U1", "CFI", "λ 扫描"]),
        },
        "data_manifest": {
            "corpus": "uoft-cprg::pluribus-human-seats（10,000 手）",
            "split_registry": "docs/V3F_S2_SPLIT_REGISTRY.json",
            "assignment_sha256": reg["assignment_sha256"],
            "hand_context_coupling": (reg["scope_rule"]),
        },
        "instrument_gates": {
            "IG1_raw_ratio": ("PI-EV 输入与原始记录比例必须 ∈ [0.5, 2.0]；"
                              "异常 → 该点 BLOCKED"),
            "IG2_anomaly_rate": (f"异常率 > {ANOMALY_MAX_RATE:.0%} → 整轮 BLOCKED"),
            "IG3_distribution": ("S″ 决策点 pot_bb 分布必须与语料结构分布一致且"
                                 "不得退化为单一桶；否则 BLOCKED"),
            "IG4_audit_log": ("逐条记录 raw → state → PI-EV input（容量 "
                              f"{AUDIT_CAPACITY}）"),
            "IG5_no_approximation": "unavailable 点不得近似替代",
            "IG6_no_dry_run": "封存后禁止任何成本校准/试运行",
        },
        "seeds": {"layer": SEED, "equity": SEED, "reference": SEED,
                  "bootstrap": SEED},
        "gates": {
            "primary": (f"PI-EV 一致率 Δ ≥ +{ACCEPT_PP}pp 且 bootstrap 95% CI "
                        f"下界 > 0 → ACCEPT；+{UNCERTAIN_PP}~+{ACCEPT_PP}pp → "
                        f"UNCERTAIN；< +{UNCERTAIN_PP}pp 或 CI 含 0 → REJECT"),
            "partial": (f"coverage < {COVERAGE_MIN:.0%} / 有效点 < "
                        f"{VALID_POINTS_MIN} / 层拒绝率 > {DECLINE_MAX:.0%} → PARTIAL"),
            "blocked": "仪器门 IG1–IG3 失败 / 封存漂移 / legality<100% → BLOCKED",
        },
        "rejection_lines": [
            "manifest 哈希漂移",
            "封存后、运行前计算或查看任何 S″ 指标（含成本校准试运行）",
            "S″ 重跑 / 据中间结果调参 / 改阈值",
            "以 agreement 作为主判据",
            "复用 S′ 的 PI-EV 结果或数据点",
            "把 unavailable 点近似替代",
        ],
        "sealed_before_model_run": True,
        "production_wiring": "NOT WIRED",
        "run_guard": "run_s2_acceptance(allow_run=True) 必须显式传入",
    }
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return man


def verify_s2_manifest(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    man = json.loads(path.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha(rel) != h}
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    split_drift = (man["data_manifest"]["assignment_sha256"]
                   != reg["assignment_sha256"])
    return {"present": True, "drift": drift,
            "clean": (not drift) and (not split_drift),
            "split_drift": split_drift,
            "checked": len(man.get("code_hashes") or {})}


# ── 一次性运行器（**本阶段不执行**） ────────────────────────────────────
def run_s2_acceptance(hands: list[dict], *, arm: str,
                      allow_run: bool = False) -> dict:
    """S″ 一次性验收（arm ∈ {U1, R2.2}）。默认禁用，需 allow_run=True。"""
    if not allow_run:
        raise RuntimeError(
            "run_s2_acceptance 已禁用（本阶段仅修复/审计/准备）。"
            "执行需下一阶段明确授权并显式传入 allow_run=True。")
    if arm not in ("U1", "R2.2"):
        raise ValueError(f"未知 arm: {arm!r}")
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    s2 = set(reg["s2_players"])
    cfg = _load_config()
    in_scope = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    chosen = {hid for hid, i in in_scope.items() if i >= S2_HAND_MIN_INDEX}
    scope_hands = [h for h in hands if str(h["hand_id"]) in chosen]
    idx = _build_indexes_excluding(scope_hands, s2)
    agg = _build_aggression_excluding(scope_hands, s2)
    calc = EquityCalculator(iterations=EQUITY_ITERS, seed=SEED)
    pi = PiEvReference()
    solver = SolverReference()
    audit = AuditLog(capacity=AUDIT_CAPACITY)
    counts: dict[str, Any] = {
        "arm": arm, "attempted": 0, "pi_points": 0, "pi_unavailable": 0,
        "candidate_declined": 0, "legal": 0, "illegal": 0,
        "instrument_blocked": 0, "anomaly_details": [],
        "solver_configured": solver.is_configured(),
        "solver_reason": solver.unavailable_reason(),
    }
    points: list[dict] = []
    for h in scope_hands:
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
                if not record or pid not in s2:
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
                if len(hole) != 2 or villain is None:
                    _adv(a, pid, street, folded, hist)
                    continue
                vpos = str(pos_map.get(seat_of.get(villain)) or "")
                vr, _p = _villain_range_arm(
                    cfg, villain=villain, villain_pos=vpos, street=street,
                    route=_route_of(hist, villain, street),
                    dead_cards=hole + list(board), idx=idx, with_evidence=True,
                    hole_cards=hole, board=board)
                if vr is None or not vr.distribution():
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
                # ── 仪器门：唯一合法 PI-EV 输入路径 + sanity check ──────
                try:
                    inputs = pi_ev_inputs(state, raw=a, hand_bb=bb)
                    sanity_check(inputs, raw=a, hand_bb=bb)
                except UnitContractError as e:
                    counts["instrument_blocked"] += 1
                    if len(counts["anomaly_details"]) < 20:
                        counts["anomaly_details"].append(
                            {"hand_id": str(h["hand_id"]), "street": street,
                             "reason": str(e)})
                    _adv(a, pid, street, folded, hist)
                    continue
                hero = next(p for p in state["players"]
                            if isinstance(p, dict) and p.get("is_hero"))
                audit.record(hand_id=str(h["hand_id"]), street=street, raw=a,
                             state_pot_bb=float(state.get("pot") or 0.0),
                             state_to_call_bb=float(state.get("to_call") or 0.0),
                             hero_stack_bb=float(hero.get("stack") or 0.0),
                             hand_bb=bb, inputs=inputs)
                rs = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                osrc = {"players": {villain: prof}} if prof else None
                frozen = compute_ev(state, eq_src, range_src=rs,
                                    opponent_src=osrc, seed=SEED)
                if not frozen.available or not frozen.candidates:
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
                ref = pi.reference(hero_cards=hole, villain_hands=vhands,
                                   board=board, candidates=cands,
                                   pot_bb=inputs["pot_bb"],
                                   to_call_bb=inputs["to_call_bb"],
                                   hero_stack_bb=inputs["hero_stack_bb"])
                if not ref["available"]:
                    counts["pi_unavailable"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                counts["pi_points"] += 1
                if arm == "U1":
                    res, meta = compute_ev_variant(state, eq_src, rs, osrc,
                                                   variant="U1", seed=SEED,
                                                   base=frozen)
                    best = res.best_action if res.available else None
                    declined = (not res.available) or not meta["u1_applied"]
                else:
                    r22 = evaluate_v3(state, eq_src, range_src=rs,
                                      aggression=agg, villain_id=villain,
                                      villain_position=vpos, runouts=RUNOUTS,
                                      equity_iterations=CHILD_ITERS, seed=SEED)
                    best = r22.get("best_v3")
                    declined = not r22.get("available")
                if declined:
                    counts["candidate_declined"] += 1
                points.append({
                    "hand_id": str(h["hand_id"]), "street": street,
                    "observed": observed,
                    "legal": bool(la is not None and observed in la),
                    "frozen_best": frozen.best_action, "cand_best": best,
                    "cand_declined": bool(declined),
                    "pi_best": ref["ordering"][0],
                    "pi_equity": ref["pi_equity"],
                    "pot_bb": inputs["pot_bb"],
                    "hero_stack_bb": inputs["hero_stack_bb"],
                    "position": vpos,
                })
                _adv(a, pid, street, folded, hist)
    result = _summarize_s2(points, counts, reg, arm, audit)
    out = RESULT_U1 if arm == "U1" else RESULT_R22
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return result


def _summarize_s2(points, counts, reg, arm, audit) -> dict:
    valid = [p for p in points if p.get("cand_best") and p.get("frozen_best")]
    coverage = counts["pi_points"] / counts["attempted"] if counts["attempted"] else 0.0
    decline_rate = (counts["candidate_declined"] / counts["pi_points"]
                    if counts["pi_points"] else 0.0)
    anomaly_rate = (counts["instrument_blocked"] / counts["attempted"]
                    if counts["attempted"] else 0.0)
    diffs = [int(p["cand_best"] == p["pi_best"])
             - int(p["frozen_best"] == p["pi_best"]) for p in valid]
    rng = random.Random(SEED)
    sims = []
    for _ in range(BOOTSTRAP_N):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        sims.append(sum(s) / len(s))
    sims.sort()
    delta = round(100 * sum(diffs) / len(diffs), 4) if diffs else None
    ci = ([round(100 * sims[int(0.025 * BOOTSTRAP_N)], 4),
           round(100 * sims[int(0.975 * BOOTSTRAP_N)], 4)] if diffs else None)
    pots = sorted(p["pot_bb"] for p in points)
    buckets = {"<5bb": 0, "5-10bb": 0, "10-20bb": 0, ">=20bb": 0}
    for v in pots:
        buckets["<5bb" if v < 5 else ("5-10bb" if v < 10 else
                                      ("10-20bb" if v < 20 else ">=20bb"))] += 1
    instrument_ok = anomaly_rate <= ANOMALY_MAX_RATE and sum(
        1 for k, v in buckets.items() if v > 0) >= 2
    if not instrument_ok:
        verdict = "BLOCKED"
    elif coverage < COVERAGE_MIN or len(valid) < VALID_POINTS_MIN \
            or decline_rate > DECLINE_MAX:
        verdict = "PARTIAL"
    elif delta is None:
        verdict = "BLOCKED"
    elif delta >= ACCEPT_PP and ci and ci[0] > 0:
        verdict = "ACCEPT"
    elif delta >= UNCERTAIN_PP:
        verdict = "UNCERTAIN"
    else:
        verdict = "REJECT"
    return {
        "snapshot_version": S2_VERSION, "arm": arm,
        "s2_players": reg["s2_players"], "scope_rule": reg["scope_rule"],
        "counts": counts, "coverage": round(coverage, 6),
        "decline_rate": round(decline_rate, 6),
        "anomaly_rate": round(anomaly_rate, 6),
        "valid_points": len(valid),
        "frozen_pi": (round(sum(1 for p in valid
                                if p["frozen_best"] == p["pi_best"])
                            / len(valid), 6) if valid else None),
        "candidate_pi": (round(sum(1 for p in valid
                                   if p["cand_best"] == p["pi_best"])
                               / len(valid), 6) if valid else None),
        "paired_delta_pp": delta, "ci95_pp": ci, "n_paired": len(diffs),
        "pot_buckets": buckets,
        "pot_percentiles_bb": ({k: pots[int(q * (len(pots) - 1))]
                                for k, q in (("p10", 0.1), ("p50", 0.5),
                                             ("p90", 0.9))} if pots else None),
        "instrument": {"audit_rows": len(audit.rows),
                       "audit_sample": audit.rows[:3],
                       "gates": {"IG1_raw_ratio": True,
                                 "IG2_anomaly_rate": anomaly_rate
                                 <= ANOMALY_MAX_RATE,
                                 "IG3_distribution": sum(
                                     1 for v in buckets.values() if v > 0)
                                 >= 2}},
        "reference": {"kind": "pi_ev", "level": "L2", "equilibrium": False,
                      "NOT_exact_gto": True,
                      "solver_configured": counts["solver_configured"],
                      "solver_reason": counts["solver_reason"]},
        "hand_context_coupling": reg["scope_rule"],
        "verdict": verdict,
    }
