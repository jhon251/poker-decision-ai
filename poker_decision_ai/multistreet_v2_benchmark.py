"""R2.1 基准：B1 同点对照 + B2 跨语料独立留出（各只运行一次）。

纪律
----
* B1：侵略性索引拟合于 **Pluribus evidence 7000 手**；评估于 R1/R2 同点
  （eval 3000 手内的 400 手决策前缀 → 2,950 点）。**手级不相交**（沿用
  R1 切分），但**玩家身份在两侧重叠**（研究对局同一批职业玩家）——如实登记。
* B2：索引拟合于 **UofT canonical 非 Pluribus 语料**（声明前缀）；评估于
  **Pluribus eval 同一 2,950 点**；玩家级索引不可用（id 域不同）→ 仅
  population 率 → 衡量**跨语料域差异**。
* 两基准的配置在 `docs/V3F_R21_MANIFEST.json` **运行前哈希封存**；结果不利
  不得调参重跑。
* R1/R2 快照**只读引用**，绝不覆盖。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .domain.poker.equity import EquityCalculator
from .domain.poker.strategy_core import compute_ev
from .multistreet_v2 import MS2_VERSION, child_equity, evaluate_v2
from .opponent_aggression import (AGGRESSION_VERSION, MIN_SAMPLES,
                                  build_aggression_events_canonical,
                                  build_aggression_index)

ROOT = Path(__file__).resolve().parents[1]

B1_VERSION = "r21-bench-b1-v1"
B2_VERSION = "r21-bench-b2-v1"
B1_DECISION_HAND_LIMIT = 400
B1_EQUITY_ITERATIONS = 1500
B1_RUNOUTS = 8
B1_CHILD_EQUITY_ITERATIONS = 300
B1_SEED = 42
B2_FIT_SOURCE = "data/uoft/normalized/chunk_000.jsonl"
B2_FIT_HANDS = 20000
B2_FIT_CHUNK_SHA256 = ("bee05c45ba9bf16dcfd4b48f87c64f14411b9265df62832"
                       "b2da1a68d93590442")
B1_SNAPSHOT = ROOT / "docs" / "V3F_R21_B1_BENCH.json"
B2_SNAPSHOT = ROOT / "docs" / "V3F_R21_B2_CROSS_CORPUS.json"
R1_SNAPSHOT = ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json"
R2_SNAPSHOT = ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json"

ARMS = ("R1_RANGE_ONLY", "R1_REAL", "R2_1")


def _config(version: str) -> dict:
    return {
        "bench_version": version,
        "layer": MS2_VERSION,
        "aggression": AGGRESSION_VERSION,
        "split_rule": "hand_id 字典序；前 70% evidence / 后 30% evaluation",
        "decision_hand_limit": B1_DECISION_HAND_LIMIT,
        "equity_iterations": B1_EQUITY_ITERATIONS,
        "runouts_per_branch": B1_RUNOUTS,
        "child_equity_iterations": B1_CHILD_EQUITY_ITERATIONS,
        "min_samples": MIN_SAMPLES,
        "seed": B1_SEED,
        "depth_cap_streets": 1,
        "villain_bet_frac": 0.66,
        "note": "配置运行前封存；结果不利不得调参重跑（B1/B2 各只运行一次）",
    }


def _arm_stats(rows: list[dict], arm: str) -> dict:
    acts = ("fold", "check", "call", "bet", "raise")
    conf = {o: {a: 0 for a in acts} for o in acts}
    n = agree = 0
    aggressive = 0
    comp = {"immediate_ev": [], "future_realization": [],
            "aggression_risk": [], "utility_chip_v2": []}
    declines = 0
    for r in rows:
        got = (r["arms"].get(arm) or {}).get("best_action")
        obs = r["observed_human_action"]
        if obs not in conf or got not in acts:
            if (r["arms"].get(arm) or {}).get("declined"):
                declines += 1
            continue
        conf[obs][got] += 1
        n += 1
        agree += int(got == obs)
        aggressive += int(got in ("bet", "raise"))
        for k in comp:
            v = (r["arms"].get(arm) or {}).get("components", {}).get(k)
            if v is not None:
                comp[k].append(v)
    by_obs = {}
    for o in acts:
        row = conf[o]
        tot = sum(row.values())
        if tot:
            by_obs[o] = {"n": tot,
                         "agreement": round(row[o] / tot, 6)}
    return {
        "n": n,
        "observed_action_agreement": round(agree / n, 6) if n else None,
        "agreement_by_observed": by_obs,
        "confusion": conf,
        "bet_bias_share": round(aggressive / n, 6) if n else None,
        "passive_share": (round(sum(1 for o in acts for a in ("check", "call")
                                    if conf[o][a]) / n, 6) if n else None),
        "declines": declines,
        "components_mean": {k: (round(sum(v) / len(v), 6) if v else None)
                            for k, v in comp.items()},
    }


def collect_rows_b1(hands: list[dict], *,
                    decision_hand_limit: int = B1_DECISION_HAND_LIMIT,
                    equity_iterations: int = B1_EQUITY_ITERATIONS,
                    runouts: int = B1_RUNOUTS,
                    child_equity_iterations: int = B1_CHILD_EQUITY_ITERATIONS,
                    max_points: int = 0, seed: int = B1_SEED,
                    aggression_override: dict | None = None) -> dict:
    """B1/B2 共用重放：收集 R1_REAL（冻结）与 R2_1（新层）两臂。

    `aggression_override`：B2 用外部拟合的索引（跨语料）；None → 用本语料
    evidence 分割拟合并报告覆盖率。
    """
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
           else build_aggression_index(train))   # 拟合：仅 evidence 手
    calc = EquityCalculator(iterations=equity_iterations, seed=seed)
    rows: list[dict] = []
    counts = {"points": 0, "no_range": 0, "no_hole_cards": 0, "no_villain": 0,
              "r21_declined": 0, "legal": 0, "illegal": 0,
              "agg_source_player": 0, "agg_source_population": 0}
    for hand in eval_hands[:decision_hand_limit]:
        bb = hand["bb"] or 1.0
        n = len(hand["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(hand["players"])}
        folded: set[str] = set()
        action_history: list[dict] = []
        boards = hand["board_by_street"]
        for st in hand["streets"]:
            street = st["street"]
            if street == "preflop":
                board: list[str] = []
            else:
                board = list(boards.get("flop", []))
                if street in ("turn", "river"):
                    board += list(boards.get("turn", []))
                if street == "river":
                    board += list(boards.get("river", []))
            for a in st["actions"]:
                pid = a["player"]
                if pid == AI_SEAT:
                    if a["action"] in ("bet", "raise", "call"):
                        action_history.append(
                            {"street": street, "player_id": pid,
                             "action": a["action"],
                             "amount": a.get("amount") or 0.0})
                    continue
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                counts["legal" if (la is not None and observed in la)
                       else "illegal"] += 1
                hero_hole = list(hand["hole_cards"].get(pid) or [])
                villain = _current_aggressor(hand, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in hand["players"]
                                    if p != pid and p not in folded), None)
                if len(hero_hole) != 2 or villain is None:
                    counts["no_hole_cards" if len(hero_hole) != 2
                           else "no_villain"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
                villain_pos = str(pos_map.get(seat_of.get(villain)) or "")
                route = _route_of(action_history, villain, street)
                vr, _prov = _villain_range_arm(
                    cfg, villain=villain, villain_pos=villain_pos,
                    street=street, route=route,
                    dead_cards=hero_hole + list(board), idx=idx,
                    with_evidence=True, hole_cards=hero_hole, board=board)
                if vr is None or not vr.distribution():
                    counts["no_range"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
                dist = vr.distribution()
                er = calc.hand_vs_range(hero_hole, board, vr,
                                        iterations=equity_iterations, seed=seed)
                hero_eq, tie = er.hero_equity, er.tie
                eq_src = {"hero_equity": hero_eq,
                          "outcome": {"win": round(hero_eq - tie / 2, 6),
                                      "tie": round(tie, 6),
                                      "lose": round(1 - hero_eq - tie / 2, 6)}}
                state = _state_for(hand, a, street, board, pos_map, folded,
                                   action_history, pid, bb)
                range_src = {"villains": {villain: {"distribution": dist}}}
                prof = _profile(villain, idx)
                opponent_src = ({"players": {villain: prof}} if prof
                                else None)
                # 参考臂（与 R1 发布口径一致）：
                #   R1_RANGE_ONLY = REAL 范围 + 无行为证据（**不是** R1 的
                #                   BASELINE 臂；后者另用静态 config 范围）
                #   R1_REAL       = REAL 范围 + 行为证据（= R1 发布 REAL 臂）
                r_range = compute_ev(state, eq_src, range_src=range_src,
                                     seed=seed)
                r1 = compute_ev(state, eq_src, range_src=range_src,
                                opponent_src=opponent_src, seed=seed)
                r21 = evaluate_v2(
                    state, eq_src, range_src=range_src,
                    aggression=agg, villain_id=villain,
                    villain_position=villain_pos, runouts=runouts,
                    equity_iterations=child_equity_iterations, seed=seed)
                if not r21.get("available"):
                    counts["r21_declined"] += 1
                else:
                    src = r21.get("aggression_source")
                    counts["agg_source_player" if src == "player"
                           else "agg_source_population"] += 1
                arms: dict[str, Any] = {}
                for key, res in (("R1_RANGE_ONLY", r_range), ("R1_REAL", r1)):
                    if res.available:
                        arms[key] = {
                            "best_action": res.best_action,
                            "ranking": [{"action": c.action,
                                         "utility_bb": round(float(
                                             c.utility_bb if c.utility_bb
                                             is not None else c.ev_bb or 0.0), 6)}
                                        for c in res.candidates],
                        }
                cands = r21.get("candidates") or []
                scored = [c for c in cands if c.get("utility_v2") is not None]
                best21 = (max(scored, key=lambda c: c["utility_v2"])["action"]
                          if scored else None)
                arms["R2_1"] = {
                    "best_action": best21 or (r1.best_action
                                              if not r21.get("available")
                                              else None),
                    "declined": not r21.get("available"),
                    "decline_reason": r21.get("decline_reason"),
                    "best_ss": r21.get("best_ss"),
                    "p_aggress": r21.get("p_aggress"),
                    "aggression_source": r21.get("aggression_source"),
                    "components": (max(scored,
                                       key=lambda c: c["utility_v2"])
                                   if scored else {}),
                }
                rows.append({"hand_id": hand["hand_id"], "street": street,
                             "observed_human_action": observed,
                             "legal": bool(la is not None and observed in la),
                             "hero_equity": round(hero_eq, 6),
                             "arms": arms})
                counts["points"] += 1
                if max_points and counts["points"] >= max_points:
                    return {"rows": rows, "counts": counts, "aggression": agg,
                            "split": {"evidence_hands": len(train),
                                      "evaluation_hands": len(eval_hands)}}
                _advance(a, pid, street, folded, action_history)
    return {"rows": rows, "counts": counts, "aggression": agg,
            "split": {"evidence_hands": len(train),
                      "evaluation_hands": len(eval_hands)}}


def _advance(a: dict, pid: str, street: str, folded: set,
             action_history: list) -> None:
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        action_history.append({"street": street, "player_id": pid,
                               "action": a["action"],
                               "amount": a.get("amount") or 0.0})


def run_b1(hands: list[dict] | None = None, *,
           snapshot_path: Path | None = None, run_label: str = "pass1",
           **kw) -> dict:
    from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    col = collect_rows_b1(hands, **kw)
    rows = col["rows"]
    arms = {a: _arm_stats(rows, a) for a in ARMS}
    cov = col["aggression"]["coverage"]
    snap = {
        "snapshot_version": B1_VERSION,
        "run_label": run_label,
        "config": _config(B1_VERSION),
        "counts": col["counts"],
        "split": col["split"],
        "arms": arms,
        "aggression_coverage": {k: v for k, v in cov.items()
                                if k != "players"},
        "aggression_players": cov.get("players"),
        "r1_reference": _read(R1_SNAPSHOT, ("cash",)),
        "r2_reference": _read(R2_SNAPSHOT, ("snapshot_version", "arms",
                                            "r2_vs_r1")),
        "domain_note": ("B1：拟合与评估均来自 Pluribus；**手级不相交**，"
                        "但玩家身份重叠（同一批职业玩家）——不可解读为"
                        "完全独立留出"),
        "honesty": _honesty(),
    }
    out_path = snapshot_path or B1_SNAPSHOT
    out_path.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return snap


def _iter_canonical_hands(path: str, limit: int):
    with open(path, encoding="utf-8") as fh:
        n = 0
        for line in fh:
            if n >= limit:
                break
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            c = d.get("canonical")
            if not c:
                continue
            src = str(c.get("source") or "")
            if "pluribus" in src.lower():
                continue
            n += 1
            yield c


def run_b2(hands: list[dict] | None = None,
           fit_path: str = B2_FIT_SOURCE, fit_hands: int = B2_FIT_HANDS,
           **kw) -> dict:
    """B2：跨语料独立留出（拟合=UofT canonical 非 Pluribus；评估=Pluribus eval）。"""
    from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    fit_events: list[dict] = []
    n_fit = 0
    for c in _iter_canonical_hands(fit_path, fit_hands):
        n_fit += 1
        fit_events.extend(build_aggression_events_canonical(c))
    from .opponent_aggression import ConditionalAggressionIndex
    pop = ConditionalAggressionIndex()
    pop.add_events(fit_events)
    agg = {"version": AGGRESSION_VERSION, "population": pop, "players": {},
           "coverage": {"n_events": len(fit_events), "fit_hands": n_fit,
                        "source": fit_path}}
    col = collect_rows_b1(hands, aggression_override=agg, **kw)
    rows = col["rows"]
    arms = {a: _arm_stats(rows, a) for a in ARMS}
    snap = {
        "snapshot_version": B2_VERSION,
        "config": {**_config(B2_VERSION), "fit_source": fit_path,
                   "fit_hands": fit_hands,
                   "fit_chunk_sha256": B2_FIT_CHUNK_SHA256,
                   "player_index": "unavailable（跨语料 id 域不同）→ 仅 population"},
        "fit": {"hands_scanned": n_fit, "events": len(fit_events),
                "population_cells": len(pop.counts)},
        "counts": col["counts"],
        "split": col["split"],
        "arms": arms,
        "r1_reference": _read(R1_SNAPSHOT, ("cash",)),
        "domain_note": ("B2：拟合语料 = handhq/ABS 现金桌（2009-07，观察者视角，"
                        "无 hole cards）；评估语料 = Pluribus 研究对局 → "
                        "存在**venue/年代/对手池/赛制**域差异，结论只用于"
                        "稳健性判断，不与 B1 数值直接等同"),
        "honesty": _honesty(),
    }
    B2_SNAPSHOT.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return snap


def _read(path: Path, keys: tuple) -> dict:
    if not path.exists():
        return {"present": False}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"present": True, **{k: d.get(k) for k in keys}}


# ── 预注册 manifest（运行前哈希封存；运行后校验未漂移） ──────────────────
MANIFEST_PATH = ROOT / "docs" / "V3F_R21_MANIFEST.json"
_SEALED_FILES = (
    "poker_decision_ai/opponent_aggression.py",
    "poker_decision_ai/multistreet_v2.py",
    "poker_decision_ai/multistreet_v2_benchmark.py",
    "poker_decision_ai/pluribus_validation.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/domain/poker/conditional_model.py",
)


def _sha256(rel: str) -> str:
    import hashlib
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() \
        else "MISSING"


def write_manifest() -> dict:
    """**运行前**封存：代码/配置/数据清单/split/seed/阈值/版本。"""
    man = {
        "manifest_version": "r21-manifest-v1",
        "phase": "R2.1（独立层 + 基准；不接入生产决策链）",
        "code_hashes": {rel: _sha256(rel) for rel in _SEALED_FILES},
        "config": {
            "b1": _config(B1_VERSION),
            "b2": {**_config(B2_VERSION), "fit_source": B2_FIT_SOURCE,
                   "fit_hands": B2_FIT_HANDS,
                   "fit_chunk_sha256": B2_FIT_CHUNK_SHA256},
            "depth_cap_streets": 1,
            "villain_bet_frac": 0.66,
            "ladder_levels": 4,
            "min_samples": MIN_SAMPLES,
        },
        "data_manifest": {
            "b1_fit_eval": {
                "source": "uoft-cprg::pluribus-human-seats（raw parquet）",
                "dir": "C:/PokerDataset/UofT_CPRG/data",
                "files": sorted(p.name for p in
                                Path("C:/PokerDataset/UofT_CPRG/data")
                                .glob("*.parquet")) if
                Path("C:/PokerDataset/UofT_CPRG/data").is_dir() else [],
                "split": "hand_id 字典序；前 70% evidence / 后 30% evaluation",
                "decision_hand_limit": B1_DECISION_HAND_LIMIT,
            },
            "b2_fit": {"path": B2_FIT_SOURCE, "hands": B2_FIT_HANDS,
                       "sha256": B2_FIT_CHUNK_SHA256,
                       "excludes": "source 含 'pluribus' 的手牌（跨语料分离）"},
            "b2_eval": "Pluribus eval 分割（同 B1 评估侧）",
        },
        "seeds": {"layer_seed": B1_SEED, "equity_seed": B1_SEED,
                  "runout_seed": B1_SEED},
        "run_policy": {
            "b1_runs": 1, "b2_runs": 1,
            "no_tuning": "运行后不得调整配置重跑以迎合指标",
            "verify_after": "运行后必须校验 code_hashes 未漂移",
        },
        "production_wiring": "NOT WIRED（生产接线需后续独立授权）",
    }
    MANIFEST_PATH.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return man


def verify_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"present": False}
    man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha256(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha256(rel) != h}
    return {"present": True, "drift": drift, "clean": not drift,
            "checked": len(man.get("code_hashes") or {})}


def _honesty() -> dict:
    return {
        "no_profitability": "不构成任何盈利性/GTO 证明",
        "no_tuning": "配置运行前封存；B1/B2 各只运行一次；结果不利不重跑",
        "no_lookahead": "侵略性特征严格决策时点截断；未来牌仅未见牌采样",
        "strength_conditioned_aggression": ("NOT MODELED（无证据）——不得声称"
                                            "真实 opponent model 已完成"),
        "mechanism_scope": ("MTT/PKO 无真实锦标赛/bounty 语料 → 仅机制与隔离；"
                            "本基准仅 CASH"),
        "r1_r2_preserved": "R1/R2 快照只读引用，未被覆盖",
        "agreement_not_objective": ("agreement 不是优化目标；同时报告 bet 偏置、"
                                    "confusion matrix 与分量分解"),
    }
