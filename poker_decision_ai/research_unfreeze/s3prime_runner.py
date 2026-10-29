"""S‴′ runner（`s3prime-validation-v2`）—— **一次性重新验收**用。

相对 v1 的唯一变化：把逐点处理抽成可复用函数，并接入**已证明等价**的
S1/S2/S5 性能路径（`perf_optimize.patched_s1_s2` + `evaluate_v3_s1s2` +
按手连续分片的进程池），使一次性运行在可接受时间内完成。

不变量（与 v1 完全一致）
------------------------
* 参照：`pi_reference_complete.reference_complete`（**全部在场对手，含 AI 座**；
  在场对手手牌未知 → unavailable，绝不伪造）；
* 决策集：S‴ 冻结 registry（手序 ≥1500 + S‴ 玩家）——**NOT INDEPENDENT**；
* PI-EV 定义 / 候选动作 / seed / 枚举预算 / 阈值 / UNKNOWN 语义：取自 sealed 模块，未改；
* 被评系统侧（主对手选择、`_state_for`、`_route_of`、range、aggression、frozen、U1、R2.2）
  沿用 S‴ 遗留口径 → **Defect B 只披露、不夹带修复**；
* Solver BLOCKED；旧封存产物只读 + 硬护栏拒写；`allow_run=True` 必须显式传入。

模式：`baseline1`（单进程无缓存，等价性基准）/ `s1s2_1` / `s1s2_8`（默认）/ `s1s2_16`
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import time
from pathlib import Path

from ..domain.poker.equity import EquityCalculator
from ..domain.poker.strategy_core import _hero_in_position, compute_ev
from ..multistreet_v3 import evaluate_v3
from ..pluribus_decision_adapter import _positions_for_seats
from ..pluribus_validation import (_current_aggressor, _load_config, _profile,
                                   _route_of, _state_for, _villain_range_arm,
                                   legal_action_types)
from . import s3_validation as s3
from .pi_ev_contract import UnitContractError, pi_ev_inputs, sanity_check
from .pi_reference_complete import (REFERENCE_VERSION, build_villain_hands,
                                    reference_complete)
from .unfreeze_core import compute_ev_variant

S3P_VERSION = "s3prime-validation-v2"
S3P_HAND_MIN_INDEX = s3.S3_HAND_MIN_INDEX
S3P_SEED = s3.SEED
S3P_EQUITY_ITERS = s3.EQUITY_ITERS
S3P_RUNOUTS = s3.RUNOUTS
S3P_CHILD_ITERS = s3.CHILD_ITERS
S3P_BOOTSTRAP_N = s3.BOOTSTRAP_N
S3P_ARMS = ("U1", "R2.2")
S3P_MODES = ("baseline1", "s1s2_1", "s1s2_8", "s1s2_16")
S3P_WORKERS = {"baseline1": 1, "s1s2_1": 1, "s1s2_8": 8, "s1s2_16": 16}
S3P_INDEPENDENCE = ("NOT INDEPENDENT：与 S‴ 同一冻结决策集；本阶段为器械校正重跑，"
                    "不主张样本独立性")
COMPLETE_SRC = "all_live_opponents_incl_ai_seat"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs"
MANIFEST = OUT / "V3F_S3PRIME_MANIFEST.json"
RESULT = {a: OUT / ("V3F_S3PRIME_RESULT_%s.json" % a.replace(".", ""))
          for a in S3P_ARMS}
PER_POINT = {a: OUT / ("V3F_S3PRIME_PER_POINT_%s.jsonl" % a.replace(".", ""))
             for a in S3P_ARMS}
FORBIDDEN_PATHS = ("V3F_S3_SPLIT_REGISTRY.json", "V3F_S3_MANIFEST_U1.json",
                   "V3F_S3_MANIFEST_R22.json", "V3F_S3_PER_POINT_U1.jsonl",
                   "V3F_S3_PER_POINT_R22.jsonl", "V3F_S3_RESULT_U1.json",
                   "V3F_S3_RESULT_R22.json")
S3P_CODE = ("poker_decision_ai/research_unfreeze/s3prime_runner.py",
            "poker_decision_ai/research_unfreeze/pi_reference_complete.py")
S3P_READONLY_DEPS = ("poker_decision_ai/research_unfreeze/s3_validation.py",
                     "poker_decision_ai/research_unfreeze/pi_ev_contract.py",
                     "poker_decision_ai/research_unfreeze/unfreeze_core.py",
                     "poker_decision_ai/research_unfreeze/perf_optimize.py",
                     "poker_decision_ai/multistreet_v3.py",
                     "poker_decision_ai/domain/poker/strategy_core.py",
                     "poker_decision_ai/pluribus_validation.py",
                     "docs/V3F_S3_SPLIT_REGISTRY.json",
                     "docs/V3F_PERF_V2_MANIFEST.json")


class S3PGuardError(RuntimeError):
    pass


def _sha(rel) -> str:
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _assert_not_sealed(path: Path) -> None:
    if path.name in FORBIDDEN_PATHS or "V3F_S3_" in path.name:
        raise S3PGuardError(f"拒绝写入封存路径: {path}")


def write_manifest() -> dict:
    _assert_not_sealed(MANIFEST)
    man = {"runner_version": S3P_VERSION, "arms": list(S3P_ARMS),
           "modes": list(S3P_MODES), "workers_by_mode": dict(S3P_WORKERS),
           "default_mode": "s1s2_8",
           "reference_version": REFERENCE_VERSION, "villain_source": COMPLETE_SRC,
           "code_hashes": {rel: _sha(rel) for rel in S3P_CODE},
           "readonly_dependency_hashes": {rel: _sha(rel)
                                          for rel in S3P_READONLY_DEPS},
           "frozen_decision_set": {
               "registry": "docs/V3F_S3_SPLIT_REGISTRY.json",
               "assignment_sha256": json.loads(
                   s3.REGISTRY.read_text(encoding="utf-8"))[
                       "assignment_sha256"],
               "hand_min_index": S3P_HAND_MIN_INDEX,
               "independence": S3P_INDEPENDENCE,
               "resample": "NONE（不新增、不删除、不重采样）"},
           "semantics_unchanged": {
               "pi_ev_definition": "sealed PiEvReference（L2 / equilibrium=False / NOT_exact_gto）",
               "candidates": "sealed frozen candidates（未改）",
               "seed": S3P_SEED, "equity_iters": S3P_EQUITY_ITERS,
               "runouts": S3P_RUNOUTS, "child_iters": S3P_CHILD_ITERS,
               "bootstrap_n": S3P_BOOTSTRAP_N,
               "runout_budget": "PI_EV_MAX_RUNOUTS=2048（未改）",
               "thresholds_readonly": {
                   "accept_pp": s3.ACCEPT_PP, "uncertain_pp": s3.UNCERTAIN_PP,
                   "coverage_min": s3.COVERAGE_MIN,
                   "valid_points_min": s3.VALID_POINTS_MIN,
                   "decline_max": s3.DECLINE_MAX},
               "unknown_semantics": "无在场对手 或 在场对手手牌未知 → unavailable；禁止伪造可用",
               "solver": "BLOCKED（未配置；不伪造 GTO/REAL/equilibrium）"},
           "perf_path": {"S1": "combo_equities 跨实例记忆化（不可变存储）",
                         "S2": "child_equity 记忆化",
                         "S5": "手级连续分片进程池（按提交序收集）",
                         "excluded": ["S3", "S4", "S6"],
                         "equivalence": "docs/V3F_PERF_V2_RESULT.json（205 点 × 26 字段 0 不一致）"},
           "fixed": {"A_reference_omission": "villain_hands 纳入全部在场对手（含 AI 座）"},
           "known_unfixed": {"B_ai_fold_tracking": (
               "遗留 runner 不记录 AI 座弃牌 → 主对手回退可能选中已弃牌 AI 座"
               "（527 点）；本阶段只披露、不修复，不夹带进结果")},
           "old_sealed_artifacts": "READ ONLY（不覆盖、不改哈希）",
           "run_guard": "run_s3prime(..., allow_run=True) 必须显式传入；仅运行一次",
           "production_wiring": "NOT WIRED", "sealed_before_run": True,
           "audit": "docs/V3F_S3PRIME_REFERENCE_AUDIT.json"}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return man


def verify_manifest() -> dict:
    if not MANIFEST.exists():
        return {"present": False}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = {r: {"sealed": h, "now": _sha(r)}
             for r, h in (man.get("code_hashes") or {}).items() if _sha(r) != h}
    dep = {r: {"sealed": h, "now": _sha(r)}
           for r, h in (man.get("readonly_dependency_hashes") or {}).items()
           if _sha(r) != h}
    return {"present": True, "drift": drift, "dep_drift": dep,
            "clean": not drift and not dep,
            "checked": len(man.get("code_hashes") or {}),
            "checked_deps": len(man.get("readonly_dependency_hashes") or {})}


def sealed_s3_clean() -> dict:
    return {"U1": s3.verify_s3_manifest(s3.MANIFEST_U1)["clean"],
            "R2.2": s3.verify_s3_manifest(s3.MANIFEST_R22)["clean"]}


def perf_v2_clean() -> dict:
    from . import perf_runner_v2 as v2
    return v2.verify_v2_manifest()


def assemble_gates(rows, summary, counters, wall_s) -> dict:
    pot = sorted(r["pot_bb"] for r in rows if r.get("pot_bb") is not None)
    rp = [r.get("ratio_pot_input_over_raw") for r in rows]
    rs = [r.get("ratio_stack_input_over_state") for r in rows]
    nonempty = sum(1 for v in (summary["strata"].get("pot") or {}).values()
                   if v["n"] > 0)
    g = {"IG1_raw_ratio": bool(rows) and all(
             x is not None and 0.5 <= x <= 2.0 for x in rp + rs),
         "IG2_anomaly_rate": (counters["instrument_blocked"]
                              / max(1, counters["attempted"])
                              <= s3.ANOMALY_MAX_RATE),
         "IG3_distribution": nonempty >= 3 and len(set(pot)) > 1,
         "IG4_audit_log": all(r.get("raw_pot_before_chips") is not None
                              and r.get("state_pot_bb") is not None
                              and r.get("ratio_pot_input_over_raw") is not None
                              for r in rows),
         "IG5_no_approximation": all(int(r.get("n_runouts") or 0) > 0
                                     for r in rows),
         "IG6_no_dry_run": True,
         "IG7_per_point_schema": summary["schema_violations"] == 0,
         "IG8_strata_completeness": bool(
             summary["strata_completeness"]["complete"]) and all(
             v == 0 for v in summary["strata_completeness"]["missing"].values()),
         "IG9_self_consistency": (summary["pi_points"] == len(rows)
                                  and summary["unavailable"]
                                  == counters["unavailable"]
                                  and summary["legality"]["illegal"] == 0),
         "IG10_traceability": all(len(r) >= len(s3.POINT_SCHEMA) for r in rows)}
    return {"gates": g, "all_pass": all(g.values()), "wall_s": round(wall_s, 4),
            "ratio_pot_min": min(rp) if rp else None,
            "ratio_pot_max": max(rp) if rp else None,
            "ratio_stack_min": min(rs) if rs else None,
            "ratio_stack_max": max(rs) if rs else None,
            "pot_percentiles_bb": ({k: pot[min(len(pot) - 1, int(
                math.ceil(q / 100 * len(pot))) - 1)]
                for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}
                if pot else {})}


def _adv_both(a, pid, street, folded, folded_all, hist) -> None:
    if a["action"] == "fold":
        folded.add(pid)
        folded_all.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"],
                     "amount": a.get("amount") or 0.0})


def process_hands(hs: list, arm: str, *, idx, agg, cfg, optimized: bool,
                  players: set) -> dict:
    if optimized:
        from .perf_optimize import (evaluate_v3_s1s2 as _eval_v3,
                                    patched_s1_s2)
    else:
        _eval_v3 = evaluate_v3
        patched_s1_s2 = None
    calc = EquityCalculator(iterations=S3P_EQUITY_ITERS, seed=S3P_SEED)
    counters = {"attempted": 0, "unavailable": 0, "instrument_blocked": 0,
                "unavailable_incomplete": 0, "unavailable_no_opponent": 0,
                "recovered_by_fix": 0, "villain_complete": 0,
                "ai_seat_included": 0}
    rows: list = []
    for h in hs:
        bb = h["bb"] or 1.0
        pos_map = _positions_for_seats(len(h["players"]), len(h["players"]))
        seat_of = {p: i + 1 for i, p in enumerate(h["players"])}
        folded: set = set()
        folded_all: set = set()
        hist: list = []
        boards = h["board_by_street"]
        for st in h["streets"]:
            street = st["street"]
            if street == "preflop":
                board: list = []
            elif street == "flop":
                board = list(boards.get("flop", []))
            elif street == "turn":
                board = list(boards.get("flop", [])) + list(boards.get("turn", []))
            else:
                board = (list(boards.get("flop", []))
                         + list(boards.get("turn", []))
                         + list(boards.get("river", [])))
            record = street in ("flop", "turn")
            for a in st["actions"]:
                pid = a["player"]
                if not record or pid not in players:
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                counters["attempted"] += 1
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                hole = list(h["hole_cards"].get(pid) or [])
                villain = _current_aggressor(h, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in h["players"]
                                    if p != pid and p not in folded), None)
                if len(hole) != 2 or villain is None:
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                vpos = str(pos_map.get(seat_of.get(villain)) or "")
                vr, _p = _villain_range_arm(
                    cfg, villain=villain, villain_pos=vpos, street=street,
                    route=_route_of(hist, villain, street),
                    dead_cards=hole + list(board), idx=idx, with_evidence=True,
                    hole_cards=hole, board=board)
                if vr is None or not vr.distribution():
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                dist = vr.distribution()
                er = calc.hand_vs_range(hole, board, vr,
                                        iterations=S3P_EQUITY_ITERS,
                                        seed=S3P_SEED)
                eq, tie = er.hero_equity, er.tie
                eq_src = {"hero_equity": eq,
                          "outcome": {"win": round(eq - tie / 2, 6),
                                      "tie": round(tie, 6),
                                      "lose": round(1 - eq - tie / 2, 6)}}
                state = _state_for(h, a, street, board, pos_map, folded, hist,
                                   pid, bb)
                try:
                    inputs = pi_ev_inputs(state, raw=a, hand_bb=bb)
                    sanity_check(inputs, raw=a, hand_bb=bb)
                except UnitContractError:
                    counters["instrument_blocked"] += 1
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                hero = next(p for p in state["players"]
                            if isinstance(p, dict) and p.get("is_hero"))
                hero_ip = _hero_in_position(state, villain)
                rs = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                osrc = {"players": {villain: prof}} if prof else None
                frozen = compute_ev(state, eq_src, range_src=rs,
                                    opponent_src=osrc, seed=S3P_SEED)
                if not frozen.available or not frozen.candidates:
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                cands = [{"action": c.action, "amount_bb": c.amount_bb}
                         for c in frozen.candidates]
                facts = build_villain_hands(h, pid, folded_all,
                                            include_ai_seat=True)
                leg = build_villain_hands(h, pid, folded, include_ai_seat=False)
                ref = reference_complete(
                    hand=h, hero_pid=pid, folded=folded_all, hero_cards=hole,
                    board=board, pot_bb=inputs["pot_bb"],
                    to_call_bb=inputs["to_call_bb"],
                    hero_stack_bb=inputs["hero_stack_bb"], candidates=cands)
                if not ref["available"]:
                    counters["unavailable"] += 1
                    if facts["unknown_live"]:
                        counters["unavailable_incomplete"] += 1
                    else:
                        counters["unavailable_no_opponent"] += 1
                    _adv_both(a, pid, street, folded, folded_all, hist)
                    continue
                counters["villain_complete"] += 1
                if facts["ai_seat_live"]:
                    counters["ai_seat_included"] += 1
                if not leg["villains_used"]:
                    counters["recovered_by_fix"] += 1
                if arm == "U1":
                    res, meta = compute_ev_variant(state, eq_src, rs, osrc,
                                                   variant="U1", seed=S3P_SEED,
                                                   base=frozen)
                    cand_best = res.best_action if res.available else None
                    applicable = bool(meta["u1_applied"])
                    declined = (not res.available) or not applicable
                elif patched_s1_s2 is not None:
                    with patched_s1_s2():
                        r22 = _eval_v3(state, eq_src, range_src=rs,
                                       aggression=agg, villain_id=villain,
                                       villain_position=vpos,
                                       runouts=S3P_RUNOUTS,
                                       equity_iterations=S3P_CHILD_ITERS,
                                       seed=S3P_SEED)
                    cand_best = r22.get("best_v3")
                    applicable = bool(r22.get("available"))
                    declined = not applicable
                else:
                    r22 = evaluate_v3(state, eq_src, range_src=rs,
                                      aggression=agg, villain_id=villain,
                                      villain_position=vpos,
                                      runouts=S3P_RUNOUTS,
                                      equity_iterations=S3P_CHILD_ITERS,
                                      seed=S3P_SEED)
                    cand_best = r22.get("best_v3")
                    applicable = bool(r22.get("available"))
                    declined = not applicable
                ratios = inputs.get("ratios_vs_raw") or {}
                rows.append({
                    "arm": arm, "hand_id": str(h["hand_id"]), "street": street,
                    "observed": observed,
                    "legal": bool(la is not None and observed in la),
                    "hero_ip": bool(hero_ip) if hero_ip is not None else None,
                    "spr": round(inputs["hero_stack_bb"] / inputs["pot_bb"], 6)
                    if inputs["pot_bb"] > 0 else None,
                    "pot_bb": inputs["pot_bb"], "to_call_bb": inputs["to_call_bb"],
                    "hero_stack_bb": inputs["hero_stack_bb"],
                    "raw_pot_before_chips": float(a.get("pot_before") or 0.0),
                    "raw_to_call_chips": float(a.get("to_call") or 0.0),
                    "hand_bb_chips": float(bb),
                    "state_pot_bb": round(float(state.get("pot") or 0.0), 9),
                    "state_to_call_bb": round(float(state.get("to_call") or 0.0), 9),
                    "ratio_pot_input_over_raw": ratios.get("pot_bb"),
                    "ratio_stack_input_over_state": round(
                        inputs["hero_stack_bb"]
                        / float(hero.get("stack") or 1.0), 9),
                    "frozen_best": frozen.best_action, "cand_best": cand_best,
                    "cand_applicable": bool(applicable),
                    "cand_declined": bool(declined),
                    "pi_best": ref["ordering"][0],
                    "pi_ordering": ref["ordering"],
                    "pi_equity": round(float(ref["pi_equity"]), 6),
                    "n_runouts": int(ref["n_runouts"]),
                    "position": vpos, "villains_used": facts["villains_used"],
                    "villain_source": COMPLETE_SRC})
                _adv_both(a, pid, street, folded, folded_all, hist)
    return {"rows": rows, "counters": counters}


def _worker(payload: dict) -> dict:
    t0 = time.perf_counter()
    c0 = time.process_time()
    out = process_hands(payload["hands"], payload["arm"], idx=payload["idx"],
                        agg=payload["agg"], cfg=payload["cfg"],
                        optimized=payload["optimized"],
                        players=set(payload["players"]))
    return {"rows": out["rows"], "counters": out["counters"],
            "worker_wall_s": time.perf_counter() - t0,
            "worker_cpu_s": time.process_time() - c0}


def scope_hands(hands: list, *, max_hands: int | None = None) -> list:
    sc = [h for i, h in enumerate(hands) if i >= S3P_HAND_MIN_INDEX]
    return sc[:max_hands] if max_hands else sc


def run_core(hands: list, arm: str, *, mode: str = "s1s2_8",
             max_hands: int | None = None, players=None) -> dict:
    if mode not in S3P_MODES:
        raise ValueError(f"未知 mode: {mode!r}")
    reg = json.loads(s3.REGISTRY.read_text(encoding="utf-8"))
    players = set(players) if players else set(reg["s3_players"])
    scope = scope_hands(hands, max_hands=max_hands)
    idx = s3.build_indexes(scope, exclude_players=players)
    agg = s3.build_aggression_index_excluding(scope, self_exclude=players)
    cfg = _load_config()
    workers = S3P_WORKERS[mode]
    optimized = mode != "baseline1"
    counters = {"attempted": 0, "unavailable": 0, "instrument_blocked": 0,
                "unavailable_incomplete": 0, "unavailable_no_opponent": 0,
                "recovered_by_fix": 0, "villain_complete": 0,
                "ai_seat_included": 0}
    wcpu = wwall = 0.0
    t0 = time.perf_counter()
    if workers == 1:
        res = process_hands(scope, arm, idx=idx, agg=agg, cfg=cfg,
                            optimized=optimized, players=players)
        rows = res["rows"]
        for k, v in res["counters"].items():
            counters[k] = counters.get(k, 0) + v
        eff = None
    else:
        step = max(1, math.ceil(len(scope) / workers))
        chunks = [scope[i:i + step] for i in range(0, len(scope), step)]
        payloads = [{"hands": c, "arm": arm, "idx": idx, "agg": agg,
                     "cfg": cfg, "optimized": optimized,
                     "players": sorted(players)} for c in chunks]
        res_list = []
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_worker, payloads):
                res_list.append(r)
        rows = []
        for r in res_list:
            rows.extend(r["rows"])
            for k, v in r["counters"].items():
                counters[k] = counters.get(k, 0) + v
            wcpu += r["worker_cpu_s"]
            wwall += r["worker_wall_s"]
        eff = (wcpu / (time.perf_counter() - t0) / len(payloads)) if payloads \
            else None
    wall = time.perf_counter() - t0
    summary = s3.recompute_summary(rows, arm=arm, reg=reg,
                                   attempted=counters["attempted"],
                                   unavailable=counters["unavailable"])
    return {"rows": rows, "summary": summary, "counters": counters,
            "wall_s": wall, "mode": mode, "workers": workers,
            "optimized": optimized, "scope_hands": len(scope),
            "gates": assemble_gates(rows, summary, counters, wall),
            "worker_cpu_s_sum": round(wcpu, 4),
            "worker_wall_s_sum": round(wwall, 4),
            "worker_efficiency": (round(eff, 4) if eff else None),
            "cache_stats": _cache_stats(optimized)}


def _cache_stats(optimized: bool):
    if not optimized:
        return None
    try:
        from .perf_optimize import STATS
        return {k: dict(v) for k, v in STATS.items()}
    except Exception:  # noqa: BLE001
        return None


def run_s3prime(hands: list, arm: str, *, allow_run: bool = False,
                mode: str = "s1s2_8", max_hands: int | None = None,
                write_outputs: bool = True, run_id: str | None = None) -> dict:
    if not allow_run:
        raise S3PGuardError("run_s3prime 已禁用；需显式 allow_run=True（授权后）")
    if arm not in S3P_ARMS:
        raise ValueError(f"未知 arm: {arm!r}")
    man = verify_manifest()
    if not man.get("present") or not man.get("clean"):
        raise S3PGuardError(f"S‴′ 清单未封存或已漂移: {man}")
    before = sealed_s3_clean()
    if not all(before.values()):
        raise S3PGuardError(f"旧封存清单不干净: {before}")
    if write_outputs:
        _assert_not_sealed(RESULT[arm])
        _assert_not_sealed(PER_POINT[arm])
    core = run_core(hands, arm, mode=mode, max_hands=max_hands)
    out = {"runner_version": S3P_VERSION, "arm": arm, "mode": mode,
           "run_id": run_id, "pid": os.getpid(),
           "reference_version": REFERENCE_VERSION,
           "villain_source": COMPLETE_SRC,
           "independence": S3P_INDEPENDENCE,
           "points": len(core["rows"]), "wall_s": round(core["wall_s"], 4),
           "scope_hands": core["scope_hands"], "workers": core["workers"],
           "optimized": core["optimized"], "counters": core["counters"],
           "summary": core["summary"], "gates": core["gates"],
           "worker_efficiency": core["worker_efficiency"],
           "worker_cpu_s_sum": core["worker_cpu_s_sum"],
           "cache_stats": core["cache_stats"],
           "sealed_s3_clean_before": before,
           "sealed_s3_clean_after": sealed_s3_clean(),
           "perf_v2_manifest_clean": perf_v2_clean()["clean"],
           "solver": "BLOCKED（未配置；未伪造 GTO/REAL/equilibrium）"}
    if write_outputs:
        PER_POINT[arm].write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n"
                    for r in core["rows"]), encoding="utf-8")
        RESULT[arm].write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        out["per_point_file"] = str(PER_POINT[arm].relative_to(ROOT))
    return out


def load_rows(arm: str) -> list:
    return [json.loads(l) for l in
            PER_POINT[arm].read_text(encoding="utf-8").splitlines() if l.strip()]
