"""R2.2 基准：B1′ 同点对照 + B2′ 跨语料 + B3′ 位置分层（各只运行一次）。

纪律（沿用 R2.1，全部**只读引用** R1/R2/R2.1）：
  * B1′：侵略性索引拟合 = Pluribus evidence 7,000 手；评估 = 同 2,950 点。
  * B2′：拟合 = UofT canonical 非 Pluribus 20,000 手；评估 = Pluribus eval。
  * B3′：位置分层留出（IP / OOP 分层），验证 F1 位置对称性。
  * 配置在 `docs/V3F_R22_MANIFEST.json` **运行前哈希封存**；运行后校验零漂移。
  * **预声明预期**：F1/F2 属语义正确性修正，**不承诺** call agreement 上升。
  * F4 诊断常驻：per-action 分位数（中位/p10/p90）+ 三消融
    （frozen / v3 baseline / no-aggression）。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .domain.poker.equity import EquityCalculator
from .domain.poker.strategy_core import compute_ev
from .multistreet_v3 import MS3_VERSION, evaluate_v3
from .opponent_aggression import (AGGRESSION_VERSION, MIN_SAMPLES,
                                  build_aggression_events_canonical,
                                  build_aggression_index)

ROOT = Path(__file__).resolve().parents[1]
B1V3 = "r22-bench-b1-v1"
B2V3 = "r22-bench-b2-v1"
B3V3 = "r22-bench-b3-position-v1"
DECISION_HAND_LIMIT = 400
EQUITY_ITERATIONS = 1500
RUNOUTS = 8
CHILD_ITERS = 300
SEED = 42
B2_FIT_SOURCE = "data/uoft/normalized/chunk_000.jsonl"
B2_FIT_HANDS = 20000
B2_FIT_SHA = ("bee05c45ba9bf16dcfd4b48f87c64f14411b9265df62832"
              "b2da1a68d93590442")
SNAP_B1 = ROOT / "docs" / "V3F_R22_B1.json"
SNAP_B2 = ROOT / "docs" / "V3F_R22_B2.json"
SNAP_B3 = ROOT / "docs" / "V3F_R22_B3_POSITION.json"
REF_R1 = ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json"
REF_R2 = ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json"
REF_R21 = ROOT / "docs" / "V3F_R21_B1_BENCH_pass2.json"
ARMS = ("R1_RANGE_ONLY", "R1_REAL", "R2_2")


def config(version: str) -> dict:
    return {
        "bench_version": version, "layer": MS3_VERSION,
        "aggression": AGGRESSION_VERSION,
        "split_rule": "hand_id 字典序；前 70% evidence / 后 30% evaluation",
        "decision_hand_limit": DECISION_HAND_LIMIT,
        "equity_iterations": EQUITY_ITERATIONS, "runouts_per_branch": RUNOUTS,
        "child_equity_iterations": CHILD_ITERS, "min_samples": MIN_SAMPLES,
        "seed": SEED, "depth_cap_streets": 1,
        "villain_bet_frac": 0.66, "villain_raise_frac": 1.0,
        "pre_registered_expectation": (
            "F1/F2 是语义正确性修正（位置对称 + 对手加注风险），**不承诺** "
            "call agreement 上升；R2.2 诊断已表明 call 天花板主因位于层外"
            "（冻结核 raise>call 排序），若该诊断成立则 call 预期基本不变"),
        "no_tuning": "预算/配置运行前封存；B1′/B2′/B3′ 各只运行一次",
    }


def collect_rows(hands: list[dict], *, aggression_override: dict | None = None,
                 decision_hand_limit: int = DECISION_HAND_LIMIT,
                 equity_iterations: int = EQUITY_ITERATIONS,
                 runouts: int = RUNOUTS, child_iters: int = CHILD_ITERS,
                 max_points: int = 0, seed: int = SEED) -> dict:
    from .pluribus_validation import (AI_SEAT, _current_aggressor,
                                      _positions_for_seats, _profile,
                                      _route_of, _state_for,
                                      _villain_range_arm, build_indexes,
                                      legal_action_types, split_hands,
                                      _load_config)
    cfg = _load_config()
    train, eval_hands = split_hands(hands)
    idx = build_indexes(train)
    agg = (aggression_override if aggression_override is not None
           else build_aggression_index(train))
    calc = EquityCalculator(iterations=equity_iterations, seed=seed)
    rows: list[dict] = []
    counts = {"points": 0, "no_range": 0, "no_hole": 0, "no_villain": 0,
              "v3_declined": 0, "legal": 0, "illegal": 0,
              "ip_points": 0, "oop_points": 0,
              "agg_source_player": 0, "agg_source_population": 0,
              "node_villain_first": 0, "node_hero_first": 0,
              "allin_guarded_candidates": 0, "position_unknown": 0}
    for hand in eval_hands[:decision_hand_limit]:
        bb = hand["bb"] or 1.0
        n = len(hand["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(hand["players"])}
        folded: set[str] = set()
        hist: list[dict] = []
        boards = hand["board_by_street"]
        for st in hand["streets"]:
            street = st["street"]
            board = [] if street == "preflop" else list(boards.get("flop", []))
            if street in ("turn", "river"):
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", []))
            if street == "river":
                board = list(boards.get("flop", [])) + list(
                    boards.get("turn", [])) + list(boards.get("river", []))
            for a in st["actions"]:
                pid = a["player"]
                if pid == AI_SEAT:
                    if a["action"] in ("bet", "raise", "call"):
                        hist.append({"street": street, "player_id": pid,
                                     "action": a["action"],
                                     "amount": a.get("amount") or 0.0})
                    continue
                raw = a["action"]
                observed = raw if not (raw == "call"
                                       and (a.get("to_call") or 0) <= 1e-9) \
                    else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                counts["legal" if (la is not None and observed in la)
                       else "illegal"] += 1
                hole = list(hand["hole_cards"].get(pid) or [])
                villain = _current_aggressor(hand, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in hand["players"]
                                    if p != pid and p not in folded), None)
                if len(hole) != 2 or villain is None:
                    counts["no_hole" if len(hole) != 2 else "no_villain"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                vpos = str(pos_map.get(seat_of.get(villain)) or "")
                vr, _ = _villain_range_arm(
                    cfg, villain=villain, villain_pos=vpos, street=street,
                    route=_route_of(hist, villain, street),
                    dead_cards=hole + list(board), idx=idx, with_evidence=True,
                    hole_cards=hole, board=board)
                if vr is None or not vr.distribution():
                    counts["no_range"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                dist = vr.distribution()
                er = calc.hand_vs_range(hole, board, vr,
                                        iterations=equity_iterations, seed=seed)
                eq, tie = er.hero_equity, er.tie
                eq_src = {"hero_equity": eq,
                          "outcome": {"win": round(eq - tie / 2, 6),
                                      "tie": round(tie, 6),
                                      "lose": round(1 - eq - tie / 2, 6)}}
                state = _state_for(hand, a, street, board, pos_map, folded,
                                   hist, pid, bb)
                rs = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                r_range = compute_ev(state, eq_src, range_src=rs, seed=seed)
                r_real = compute_ev(state, eq_src, range_src=rs,
                                    opponent_src=({"players": {villain: prof}}
                                                  if prof else None), seed=seed)
                r22 = evaluate_v3(state, eq_src, range_src=rs, aggression=agg,
                                  villain_id=villain, villain_position=vpos,
                                  runouts=runouts,
                                  equity_iterations=child_iters, seed=seed)
                if not r22.get("available"):
                    counts["v3_declined"] += 1
                    if "position unverifiable" in str(
                            r22.get("decline_reason") or ""):
                        counts["position_unknown"] += 1
                else:
                    prov = r22.get("provenance") or {}
                    if prov.get("node_order") == "hero_first":
                        counts["node_hero_first"] += 1
                        counts["oop_points"] += 1
                    else:
                        counts["node_villain_first"] += 1
                        counts["ip_points"] += 1
                    if prov.get("aggression_source") == "player":
                        counts["agg_source_player"] += 1
                    else:
                        counts["agg_source_population"] += 1
                arms: dict[str, Any] = {}
                for key, res in (("R1_RANGE_ONLY", r_range), ("R1_REAL", r_real)):
                    if res.available:
                        arms[key] = {"best_action": res.best_action,
                                     "ranking": [{"action": c.action,
                                                  "utility_bb": round(float(
                                                      c.utility_bb
                                                      if c.utility_bb is not None
                                                      else c.ev_bb or 0.0), 6)}
                                                 for c in res.candidates]}
                cands = r22.get("candidates") or []
                scored = [c for c in cands if c.get("utility_v3") is not None]
                best = (max(scored, key=lambda c: c["utility_v3"])["action"]
                        if scored else None)
                counts["allin_guarded_candidates"] += sum(
                    1 for c in cands if c.get("allin_guard"))
                arms["R2_2"] = {
                    "best_action": best or (r_real.best_action
                                            if not r22.get("available")
                                            else None),
                    "declined": not r22.get("available"),
                    "decline_reason": r22.get("decline_reason"),
                    "best_ss": r22.get("best_ss"),
                    "hero_ip": r22.get("hero_ip"),
                    "node_order": (r22.get("provenance") or {}).get("node_order"),
                    "aggression_source": r22.get("aggression_source"),
                    "p_aggress": r22.get("p_aggress"),
                    "components": (max(scored, key=lambda c: c["utility_v3"])
                                   if scored else {}),
                    "candidates": [_slim(c) for c in cands],
                }
                pos_key = ("ip" if r22.get("hero_ip") else
                           "oop" if r22.get("available") else "na")
                rows.append({"hand_id": hand["hand_id"], "street": street,
                             "observed": observed, "position": pos_key,
                             "legal": bool(la is not None and observed in la),
                             "equity": round(eq, 6), "arms": arms})
                counts["points"] += 1
                if max_points and counts["points"] >= max_points:
                    return {"rows": rows, "counts": counts, "aggression": agg,
                            "split": {"evidence_hands": len(train),
                                      "evaluation_hands": len(eval_hands)}}
                _adv(a, pid, street, folded, hist)
    return {"rows": rows, "counts": counts, "aggression": agg,
            "split": {"evidence_hands": len(train),
                      "evaluation_hands": len(eval_hands)}}


def _slim(c: dict) -> dict:
    return {k: c.get(k) for k in
            ("action", "amount_bb", "immediate_ev", "future_realization",
             "aggression_risk", "utility_chip_v3", "utility_v3", "delta",
             "utility_ss", "frozen_ev_bb", "p_fold", "allin_guard",
             "opponent_response_distribution")}


def _adv(a, pid, street, folded, hist):
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"], "amount": a.get("amount") or 0.0})


def _quantiles(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    s = sorted(vals)

    def q(p: float) -> float:
        if len(s) == 1:
            return round(s[0], 6)
        idx = p * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return round(s[lo] * (1 - frac) + s[hi] * frac, 6)

    return {"n": len(s), "median": q(0.5), "p10": q(0.1), "p90": q(0.9)}


def _arm_stats(rows: list[dict], arm: str) -> dict:
    acts = ("fold", "check", "call", "bet", "raise")
    conf = {o: {a: 0 for a in acts} for o in acts}
    n = agree = aggr = declines = 0
    for r in rows:
        a = r["arms"].get(arm) or {}
        got, obs = a.get("best_action"), r["observed"]
        if a.get("declined"):
            declines += 1
        if obs not in conf or got not in acts:
            continue
        conf[obs][got] += 1
        n += 1
        agree += int(got == obs)
        aggr += int(got in ("bet", "raise"))
    by_obs = {o: {"n": sum(conf[o].values()),
                  "agreement": round(conf[o][o] / sum(conf[o].values()), 6)}
              for o in acts if sum(conf[o].values())}
    return {"n": n, "observed_action_agreement":
            round(agree / n, 6) if n else None,
            "agreement_by_observed": by_obs, "confusion": conf,
            "bet_bias_share": round(aggr / n, 6) if n else None,
            "declines": declines}


def diagnostics(rows: list[dict]) -> dict:
    """F4：per-action 分位数 + 三消融 agreement（不改变既有指标）。"""
    per_action: dict[str, list[dict]] = {}
    for r in rows:
        for c in (r["arms"]["R2_2"].get("candidates") or []):
            per_action.setdefault(c["action"], []).append(c)

    def collect(act: str, key: str) -> list:
        return [c.get(key) for c in per_action.get(act, [])]

    out = {"per_action_quantiles": {}, "ablation_agreement": {}}
    for act in ("fold", "check", "call", "bet", "raise"):
        if act not in per_action:
            continue
        out["per_action_quantiles"][act] = {
            "n": len(per_action[act]),
            "immediate_ev": _quantiles(collect(act, "immediate_ev")),
            "future_realization": _quantiles(collect(act, "future_realization")),
            "aggression_risk": _quantiles(collect(act, "aggression_risk")),
            "utility_chip_v3": _quantiles(collect(act, "utility_chip_v3")),
            "utility_ss": _quantiles(collect(act, "utility_ss")),
            "delta": _quantiles(collect(act, "delta")),
        }

    def best_by(key: str, derive=None) -> dict:
        best: dict[tuple, tuple[float, str]] = {}
        obs_of: dict[tuple, str] = {}
        for r in rows:
            k = (r["hand_id"], r["street"], r["observed"])
            obs_of[k] = r["observed"]
            for c in (r["arms"]["R2_2"].get("candidates") or []):
                v = (derive(c) if derive else c.get(key))
                if v is None:
                    continue
                cur = best.get(k)
                if cur is None or v > cur[0]:
                    best[k] = (v, c["action"])
        if not best:
            return {"n_points": 0, "agreement": None}
        agree = sum(1 for k, (v, a) in best.items() if a == obs_of[k])
        return {"n_points": len(best),
                "agreement": round(agree / len(best), 6)}

    out["ablation_agreement"] = {
        "frozen_ss": best_by("utility_ss"),
        "v3_baseline": best_by("utility_chip_v3"),
        "no_aggression": best_by(None, derive=lambda c: (
            None if (c.get("immediate_ev") is None
                     or c.get("future_realization") is None)
            else c["immediate_ev"] + c["future_realization"])),
    }
    ident = [abs(c["utility_chip_v3"] - (c["immediate_ev"]
                                         + c["future_realization"]
                                         - c["aggression_risk"]))
             for c in [c for cs in per_action.values() for c in cs]
             if c.get("utility_chip_v3") is not None
             and c.get("immediate_ev") is not None
             and c.get("future_realization") is not None
             and c.get("aggression_risk") is not None]
    out["identity_check"] = {"n": len(ident),
                             "max_abs_gap": max(ident) if ident else None,
                             "ok_within_rounding": (max(ident) <= 2e-6
                                                    if ident else None)}
    return out


def run_b1(hands=None, *, snapshot=SNAP_B1, label="b1", **kw) -> dict:
    from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    col = collect_rows(hands, **kw)
    rows = col["rows"]
    snap = {
        "snapshot_version": B1V3, "run_label": label,
        "config": config(B1V3), "split": col["split"], "counts": col["counts"],
        "arms": {a: _arm_stats(rows, a) for a in ARMS},
        "diagnostics": diagnostics(rows),
        "references": {"r1": _read(REF_R1, ("snapshot_version",)),
                       "r2": _read(REF_R2, ("snapshot_version", "arms")),
                       "r21": _read(REF_R21, ("snapshot_version", "arms"))},
        "aggression_coverage": {k: v for k, v in
                                col["aggression"]["coverage"].items()
                                if k != "players"},
        "honesty": _honesty(),
    }
    snapshot.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return snap


def _iter_canonical(path: str, limit: int):
    with open(path, encoding="utf-8") as fh:
        n = 0
        for line in fh:
            if n >= limit:
                break
            try:
                c = json.loads(line).get("canonical")
            except Exception:  # noqa: BLE001
                continue
            if not c or "pluribus" in str(c.get("source") or "").lower():
                continue
            n += 1
            yield c


def run_b2(hands=None, *, snapshot=SNAP_B2, fit_source=B2_FIT_SOURCE,
           fit_hands=B2_FIT_HANDS, **kw) -> dict:
    from .opponent_aggression import ConditionalAggressionIndex
    from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    events: list[dict] = []
    n_fit = 0
    for c in _iter_canonical(fit_source, fit_hands):
        n_fit += 1
        events.extend(build_aggression_events_canonical(c))
    pop = ConditionalAggressionIndex()
    pop.add_events(events)
    agg = {"version": AGGRESSION_VERSION, "population": pop, "players": {},
           "coverage": {"n_events": len(events), "fit_hands": n_fit,
                        "source": fit_source}}
    col = collect_rows(hands, aggression_override=agg, **kw)
    rows = col["rows"]
    snap = {
        "snapshot_version": B2V3,
        "config": {**config(B2V3), "fit_source": fit_source,
                   "fit_hands": fit_hands, "fit_chunk_sha256": B2_FIT_SHA,
                   "player_index": "unavailable（跨语料 id 域不同）→ 仅 population"},
        "fit": {"hands_scanned": n_fit, "events": len(events),
                "population_cells": len(pop.counts)},
        "split": col["split"], "counts": col["counts"],
        "arms": {a: _arm_stats(rows, a) for a in ARMS},
        "diagnostics": diagnostics(rows),
        "domain_note": ("B2′：拟合 = handhq/ABS 现金桌（2009-07，观察者视角，"
                        "无 hole cards）；评估 = Pluribus 研究对局 → 存在 "
                        "venue/年代/对手池/赛制 域差异；仅用于稳健性判断"),
        "honesty": _honesty(),
    }
    snapshot.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return snap


def run_b3(rows: list[dict], *, snapshot=SNAP_B3) -> dict:
    """B3′：位置分层（IP / OOP）——验证 F1 位置对称性（同一运行的再分层）。"""
    ip = [r for r in rows if r["position"] == "ip"]
    oop = [r for r in rows if r["position"] == "oop"]
    na = [r for r in rows if r["position"] == "na"]

    def per_position(sub: list[dict]) -> dict:
        return {a: _arm_stats(sub, a) for a in ARMS}

    snap = {
        "snapshot_version": B3V3,
        "config": {**config(B3V3),
                   "note": "B3′ = B1′ 同一运行的按位置再分层（不额外采样）"},
        "counts": {"ip": len(ip), "oop": len(oop), "na_declined": len(na)},
        "arms_ip": per_position(ip), "arms_oop": per_position(oop),
        "diagnostics_ip": diagnostics(ip) if ip else {},
        "diagnostics_oop": diagnostics(oop) if oop else {},
        "symmetry": _symmetry(ip, oop),
        "honesty": _honesty(),
    }
    snapshot.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return snap


def _symmetry(ip: list[dict], oop: list[dict]) -> dict:
    """位置对称性检查：两侧分量方向一致性与 node_order 计数。"""
    def comp(sub: list[dict]) -> dict:
        out = {"future_realization": [], "aggression_risk": [],
               "utility_chip_v3": []}
        for r in sub:
            c = r["arms"]["R2_2"].get("components") or {}
            for k in out:
                if c.get(k) is not None:
                    out[k].append(float(c[k]))
        return {k: (round(statistics.median(v), 6) if v else None)
                for k, v in out.items()}

    def orders(sub: list[dict]) -> dict:
        d: dict[str, int] = {}
        for r in sub:
            o = r["arms"]["R2_2"].get("node_order") or "na"
            d[o] = d.get(o, 0) + 1
        return d

    return {"ip_components_median": comp(ip),
            "oop_components_median": comp(oop),
            "ip_node_order": orders(ip), "oop_node_order": orders(oop),
            "expectation": ("IP 必须全为 villain_first、OOP 必须全为 hero_first"
                            "（F1 断言）")}


def _read(path: Path, keys: tuple) -> dict:
    if not path.exists():
        return {"present": False}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"present": True, **{k: d.get(k) for k in keys}}


def _honesty() -> dict:
    return {
        "no_profitability": "不构成任何盈利性/GTO 证明",
        "no_single_metric_strength": "任一单独指标不得作为强度证明",
        "no_tuning": "配置运行前封存；B1′/B2′ 各只运行一次；结果不利不重跑",
        "no_lookahead": "侵略性特征严格决策时点截断；未来牌仅未见牌采样",
        "strength_conditioned_aggression": ("NOT MODELED（无证据）——不得声称"
                                            "真实 opponent model 已完成"),
        "bounty_semantics": ("MTT/PKO bounty 项沿用冻结核候选值；"
                             "bounty 公式未改（F5）"),
        "layer_scope": ("R2.2 仅层内语义修正；**未解冻** strategy_core/CFI/"
                        "λ/fold·raise 排序"),
        "r1_r2_r21_preserved": "R1/R2/R2.1 代码、快照、manifest 只读未覆盖",
        "mtt_pko": "无真实语料 → 仅机制/隔离；B1′/B2′/B3′ 仅 CASH",
    }


# ── 预注册 manifest（运行前封存；运行后校验） ────────────────────────────
MANIFEST = ROOT / "docs" / "V3F_R22_MANIFEST.json"
SEALED = (
    "poker_decision_ai/multistreet_v3.py",
    "poker_decision_ai/multistreet_v3_benchmark.py",
    "poker_decision_ai/multistreet_v2.py",
    "poker_decision_ai/opponent_aggression.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/pluribus_validation.py",
)


def _sha(rel: str) -> str:
    import hashlib
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def write_manifest() -> dict:
    man = {
        "manifest_version": "r22-manifest-v1",
        "phase": "R2.2（层内语义正确性修正 + 诊断常驻化；不接入生产链）",
        "code_hashes": {rel: _sha(rel) for rel in SEALED},
        "config": {"b1": config(B1V3), "b2": {**config(B2V3),
                                              "fit_source": B2_FIT_SOURCE,
                                              "fit_hands": B2_FIT_HANDS,
                                              "fit_chunk_sha256": B2_FIT_SHA},
                   "b3": config(B3V3)},
        "fixes": {"F1": "节点方向位置感知（IP 对手先行 / OOP hero 先行）",
                  "F2": "hero 下注后对手响应三分支 fold/call/raise",
                  "F3": "all-in/退化守卫（跳过节点，保留冻结 continuing-equity）",
                  "F4": "诊断常驻化（per-action 分位数 + 三消融）",
                  "F5": "bounty 语义标注（公式未改）"},
        "data_manifest": {
            "b1_eval": {"source": "uoft-cprg::pluribus-human-seats",
                        "split": "hand_id 字典序 70/30",
                        "decision_hand_limit": DECISION_HAND_LIMIT},
            "b2_fit": {"path": B2_FIT_SOURCE, "hands": B2_FIT_HANDS,
                       "sha256": B2_FIT_SHA,
                       "excludes": "source 含 pluribus 的手牌"},
            "b3": {"derivation": "B1′ 同一运行按 IP/OOP 再分层"}},
        "seeds": {"layer_seed": SEED, "equity_seed": SEED, "runout_seed": SEED},
        "run_policy": {"b1_runs": 1, "b2_runs": 1, "b3_derived": True,
                       "no_tuning": "运行后不得调整配置重跑以迎合指标",
                       "verify_after": "运行后校验 code_hashes 未漂移"},
        "production_wiring": "NOT WIRED（需后续独立授权）",
    }
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return man


def verify_manifest() -> dict:
    if not MANIFEST.exists():
        return {"present": False}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha(rel) != h}
    return {"present": True, "drift": drift, "clean": not drift,
            "checked": len(man.get("code_hashes") or {})}
