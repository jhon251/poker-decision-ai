"""R2 评测 — 多街前瞻层 vs R1 冻结 baseline（同切分、同决策点、同种子）。

设计原则（对应 R2 验收条款）：
  1. **同点对比**：复用 R1 的切分（70/30 字典序）、同 400 手决策前缀、同
     equity 迭代数（1500）与 seed（42）→ 与 R1 指标可直接比较。
  2. **旧指标不覆盖**：本模块输出到**独立**快照
     `docs/V3F_R2_MULTISTREET_BASELINE.json`；R1 快照只读引用。
  3. **等价性自证**：内部复算 `R1_REAL` 臂；其结果必须与 R1 快照一致
     （测试断言），否则说明评测口径漂移 → 报告为不可信。
  4. **不调参**：预算（runouts / child equity 迭代数）在 manifest 中**预先声明**，
     不随结果调整；不以 agreement 为优化目标。
  5. **UNKNOWN-first**：层拒绝前瞻的点记为 `ms_declined`，该臂回退冻结单街动作
     （不得猜测），并单独计数。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .domain.poker.equity import EquityCalculator
from .domain.poker.strategy_core import compute_ev
from .multistreet import (MS_VERSION, multistreet_evaluate)

ROOT = Path(__file__).resolve().parents[1]

R2_BENCH_VERSION = "r2-multistreet-bench-v1"
R2_DECISION_HAND_LIMIT = 400     # 与 R1 相同（同点可比）
R2_EQUITY_ITERATIONS = 1500      # 与 R1 相同
R2_RUNOUTS = 8                   # 预算（预先声明；不随结果调整）
R2_CHILD_EQUITY_ITERATIONS = 300
R2_SEED = 42
R1_SNAPSHOT = ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json"
R2_SNAPSHOT = ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json"

#: B7-2/B7-8：运行期工件登记清单（fail closed；禁止 fallback）
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"
_GUARD = None


def _guard():
    """历史工件读取的唯一入口守卫（惰性单例；每次 require 重算真实哈希）。"""
    global _GUARD
    if _GUARD is None:
        from .artifact_guard import ArtifactGuard
        _GUARD = ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)
    return _GUARD


def r2_manifest(*, runouts: int | None = None,
                child_equity_iterations: int | None = None,
                decision_hand_limit: int | None = None) -> dict:
    budget = {
        "equity_iterations": R2_EQUITY_ITERATIONS,
        "runouts_per_branch": runouts if runouts is not None else R2_RUNOUTS,
        "child_equity_iterations": (child_equity_iterations
                                    if child_equity_iterations is not None
                                    else R2_CHILD_EQUITY_ITERATIONS),
        "seed": R2_SEED,
        "note": ("预算为**事前声明**：R2 结果不得以调整预算/阈值获得；"
                 "若结果不利，如实报告"),
    }
    return {
        "bench_version": R2_BENCH_VERSION,
        "layer_version": MS_VERSION,
        "purpose": ("在 R1 冻结切分/点集上测量多街前瞻层（R2）相对冻结单街"
                    "（R1）的决策差异；**不覆盖** R1 指标，不调参"),
        "data_source": "uoft-cprg::pluribus-human-seats（REAL_OBSERVED，cc-by-4.0）",
        "split": {"rule": "hand_id 字典序；前 70% evidence / 后 30% evaluation",
                  "evidence_frac": 0.7,
                  "decision_hand_limit": (decision_hand_limit
                                          if decision_hand_limit is not None
                                          else R2_DECISION_HAND_LIMIT),
                  "ai_seat": "Pluribus = solver_label（排除）"},
        "budget_declared": budget,
        "r1_reference_snapshot": "docs/V3F_R1_QUALITY_BASELINE.json（只读）",
        "evidence_grades": {
            "observed_action": "REAL_OBSERVED（人类座位真实动作）",
            "R1_BASELINE / R1_REAL": "HISTORICAL_REPLAY + COUNTERFACTUAL",
            "R2_MULTISTREET": ("HISTORICAL_REPLAY + COUNTERFACTUAL + "
                               "DECLARATIVE_LOOKAHEAD（叶子=冻结单街核）"),
        },
        "honesty": {
            "no_solver": "未接入 Solver；叶子为冻结单街核（非 GTO）",
            "no_tuning": "未因指标结果修改任何常数/公式/预算",
            "no_profitability": "不构成任何盈利性证明",
            "anti_lookahead": "未来牌仅用未见牌确定性采样（测试覆盖置换不变性）",
            "mtt_pko": "NOT VALIDATABLE WITH CURRENT DATA（本评测仅 CASH）",
        },
    }


def _profile(idx: dict, villain: str) -> dict | None:
    from .pluribus_validation import _profile as _p
    return _p(villain, idx)


def _arm_summary(res) -> dict:
    return {
        "best_action": res.best_action,
        "best_amount_bb": res.best_amount_bb,
        "objective": res.objective,
        "ranking": [{"action": c.action,
                     "utility_bb": round(float(
                         c.utility_bb if c.utility_bb is not None else c.ev_bb), 6)}
                    for c in res.candidates],
    }


def collect_rows(hands: list[dict], *,
                 decision_hand_limit: int = R2_DECISION_HAND_LIMIT,
                 equity_iterations: int = R2_EQUITY_ITERATIONS,
                 runouts: int = R2_RUNOUTS,
                 child_equity_iterations: int = R2_CHILD_EQUITY_ITERATIONS,
                 max_points: int = 0, seed: int = R2_SEED) -> dict:
    """单遍重放（与 R1 同走法）：逐点计算 R1_BASELINE / R1_REAL / R2 三臂。"""
    from .pluribus_validation import (AI_SEAT, _current_aggressor,
                                      _positions_for_seats, _route_of,
                                      _state_for, _villain_range_arm,
                                      build_indexes, legal_action_types,
                                      split_hands, _load_config)
    cfg = _load_config()
    train, eval_hands = split_hands(hands)
    idx = build_indexes(train)
    calc = EquityCalculator(iterations=equity_iterations, seed=seed)
    rows: list[dict] = []
    counts = {"observed_human_action": 0, "pluribus_ai_action_excluded": 0,
              "points": 0, "no_range": 0, "no_hole_cards": 0,
              "no_villain": 0, "ms_declined": 0, "illegal": 0, "legal": 0}

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
                    counts["pluribus_ai_action_excluded"] += 1
                    if a["action"] in ("bet", "raise", "call"):
                        action_history.append(
                            {"street": street, "player_id": pid,
                             "action": a["action"],
                             "amount": a.get("amount") or 0.0})
                    continue
                counts["observed_human_action"] += 1
                observed = a["action"] if not (
                    a["action"] == "call"
                    and (a.get("to_call") or 0) <= 1e-9) else "check"
                la = legal_action_types(a.get("to_call") or 0,
                                        a.get("remaining_stack") or 0)
                if la is None or observed not in la:
                    counts["illegal" if la is not None else "unknown_legal"] = \
                        counts.get("illegal" if la is not None else "unknown_legal", 0) + 1
                else:
                    counts["legal"] += 1
                hero_hole = list(hand["hole_cards"].get(pid) or [])
                if len(hero_hole) != 2:
                    counts["no_hole_cards"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
                villain = _current_aggressor(hand, street, a)
                if villain is None or villain == pid:
                    villain = next((p for p in hand["players"]
                                    if p != pid and p not in folded), None)
                if villain is None:
                    counts["no_villain"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
                villain_pos = str(pos_map.get(seat_of.get(villain)) or "")
                route = _route_of(action_history, villain, street)
                dead = hero_hole + list(board)
                vr, prov = _villain_range_arm(
                    cfg, villain=villain, villain_pos=villain_pos,
                    street=street, route=route, dead_cards=dead, idx=idx,
                    with_evidence=True, hole_cards=hero_hole, board=board)
                if vr is None:
                    counts["no_range"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
                dist = vr.distribution()
                if not dist:
                    counts["no_range"] += 1
                    _advance(a, pid, street, folded, action_history)
                    continue
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
                prof = _profile(idx, villain)
                opponent_src = {"players": {villain: prof}} if prof else None

                r_base = compute_ev(state, eq_src, range_src=range_src,
                                    seed=seed)
                r_real = compute_ev(state, eq_src, range_src=range_src,
                                    opponent_src=opponent_src, seed=seed)
                r2 = multistreet_evaluate(
                    state, eq_src, range_src=range_src,
                    opponent_src=opponent_src, runouts=runouts,
                    child_equity_iterations=child_equity_iterations,
                    seed=seed)
                if not r2.get("available"):
                    counts["ms_declined"] += 1

                arms = {}
                # 注：本臂 = REAL 范围 + **无**行为证据。R1 发布的 BASELINE 臂
                # 另用「静态 config 范围」，故两者不可混称——此处诚实命名为
                # R1_RANGE_ONLY（不作为 R1 baseline 引用）。
                for name, res in (("R1_RANGE_ONLY", r_base), ("R1_REAL", r_real)):
                    if res.available:
                        arms[name] = _arm_summary(res)
                    else:
                        arms[name] = {"best_action": None, "ranking": [],
                                      "unavailable": res.unavailable_reason}
                arms["R2_MULTISTREET"] = {
                    "best_action": r2.get("best_ms") or (
                        arms["R1_REAL"]["best_action"]
                        if not r2.get("available") else None),
                    "declined": not r2.get("available"),
                    "decline_reason": r2.get("decline_reason"),
                    "best_ss": r2.get("best_ss"),
                    "changed_vs_ss": (r2.get("comparison") or {}).get("changed"),
                    "candidates": [
                        {"action": c["action"], "amount_bb": c["amount_bb"],
                         "utility_ss": c["utility_ss"],
                         "utility_ms": c["utility_ms"], "delta": c["delta"]}
                        for c in r2.get("candidates") or []],
                    "call_delta": (r2.get("comparison") or {}).get("call_delta"),
                    "check_delta": (r2.get("comparison") or {}).get("check_delta"),
                }
                arms["R2_CHECKDOWN"] = {
                    "best_action": r2.get("best_ms_checkdown") or (
                        arms["R1_REAL"]["best_action"]
                        if not r2.get("available") else None),
                    "declined": not r2.get("available"),
                    "changed_vs_ss": (r2.get("comparison") or {})
                    .get("changed_checkdown"),
                    "call_delta": (r2.get("comparison") or {})
                    .get("call_delta_checkdown"),
                    "check_delta": (r2.get("comparison") or {})
                    .get("check_delta_checkdown"),
                    "ranking": [],       # checkdown 变体不产出新排名
                }
                row = {"hand_id": hand["hand_id"], "street": street,
                       "observed_human_action": observed,
                       "facing": "bet" if (a.get("to_call") or 0) > 1e-9 else "check",
                       "legal": (la is not None and observed in la),
                       "hero_equity": round(hero_eq, 6),
                       "villain_position": villain_pos,
                       "arms": arms, "range_provenance": prov}
                rows.append(row)
                counts["points"] += 1
                if max_points and counts["points"] >= max_points:
                    return {"rows": rows, "counts": counts,
                            "split": {"evidence_hands": len(train),
                                      "evaluation_hands": len(eval_hands),
                                      "decision_hand_limit": decision_hand_limit}}
                _advance(a, pid, street, folded, action_history)
    return {"rows": rows, "counts": counts,
            "split": {"evidence_hands": len(train),
                      "evaluation_hands": len(eval_hands),
                      "decision_hand_limit": decision_hand_limit}}


def _advance(a: dict, pid: str, street: str, folded: set,
             action_history: list) -> None:
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        action_history.append({"street": street, "player_id": pid,
                               "action": a["action"],
                               "amount": a.get("amount") or 0.0})


def _arm_metrics(rows: list[dict], arm: str) -> dict:
    with_best = [r for r in rows if (r["arms"].get(arm) or {}).get("best_action")]
    if not with_best:
        return {"n": 0}
    agree = sum(1 for r in with_best
                if r["arms"][arm]["best_action"] == r["observed_human_action"])
    by_obs: dict[str, dict] = {}
    for obs in ("call", "check", "bet", "raise", "fold"):
        sub = [r for r in with_best if r["observed_human_action"] == obs]
        if sub:
            by_obs[obs] = {
                "n": len(sub),
                "agreement": round(sum(
                    1 for r in sub
                    if r["arms"][arm]["best_action"] == obs) / len(sub), 6)}
    regrets = []
    for r in with_best:
        ranking = r["arms"][arm].get("ranking") or []
        obs_u = [c["utility_bb"] for c in ranking
                 if c["action"] == r["observed_human_action"]]
        if obs_u and ranking:
            regrets.append(round(max(obs_u) - ranking[0]["utility_bb"], 6))
    return {
        "n": len(with_best),
        "observed_action_agreement": round(agree / len(with_best), 6),
        "agreement_by_observed": by_obs,
        "regret_zero_share": (round(sum(1 for x in regrets if x == 0)
                                    / len(regrets), 6) if regrets else None),
        "regret_proxy_median": (round(statistics.median(regrets), 6)
                                if regrets else None),
    }


def run_benchmark(hands: list[dict] | None = None, **kw) -> dict:
    from .pluribus_decision_adapter import iter_pluribus_rows, parse_hand_row
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    collected = collect_rows(hands, **kw)
    rows = collected["rows"]
    arms = {}
    for arm in ("R1_RANGE_ONLY", "R1_REAL", "R2_MULTISTREET", "R2_CHECKDOWN"):
        arms[arm] = _arm_metrics(rows, arm)
    changed = sum(1 for r in rows
                  if r["arms"]["R2_MULTISTREET"].get("changed_vs_ss"))
    decided = [r for r in rows
               if r["arms"]["R2_MULTISTREET"].get("best_action")
               and r["arms"]["R1_REAL"].get("best_action")]
    changed_real = sum(1 for r in decided
                       if r["arms"]["R2_MULTISTREET"]["best_action"]
                       != r["arms"]["R1_REAL"]["best_action"])
    deltas = [r["arms"]["R2_MULTISTREET"]["call_delta"] for r in rows
              if r["arms"]["R2_MULTISTREET"].get("call_delta") is not None]
    return {
        "manifest": r2_manifest(),
        "split": collected["split"],
        "counts": collected["counts"],
        "arms": arms,
        "r2_vs_r1": {
            "changed_vs_single_street": changed,
            "changed_vs_R1_REAL_arm": changed_real,
            "n_compared": len(decided),
            "call_delta_median": (round(statistics.median(deltas), 6)
                                  if deltas else None),
            "call_delta_mean": (round(sum(deltas) / len(deltas), 6)
                                if deltas else None),
            "call_delta_positive_share": (round(sum(1 for d in deltas if d > 0)
                                                / len(deltas), 6)
                                          if deltas else None),
        },
        "rows": rows,
    }


def r1_reference() -> dict:
    """R1 冻结指标（只读；不重算、不覆盖）。

    B7-2/B7-8：读取前经 `ArtifactGuard.require()` fail closed 校验；未登记 /
    缺失 / stale / 哈希不一致 → `ArtifactGuardError`，**不回退**到
    `{"present": False}`，也不读取任何未校验文件。
    """
    _guard().require("docs/V3F_R1_QUALITY_BASELINE.json")
    d = json.loads(R1_SNAPSHOT.read_text(encoding="utf-8"))
    cash = d.get("cash") or {}
    return {
        "present": True,
        "snapshot_version": d.get("snapshot_version"),
        "fold_calibration": cash.get("fold_calibration"),
        "decision_ev": cash.get("decision_ev"),
        "split": cash.get("split"),
        "legality": cash.get("legality"),
    }


def snapshot_from_collected(collected: dict) -> dict:
    """由已收集的 rows 组装快照（确定性；避免重跑昂贵收集）。"""
    rows = collected["rows"]
    arms = {}
    for arm in ("R1_RANGE_ONLY", "R1_REAL", "R2_MULTISTREET", "R2_CHECKDOWN"):
        arms[arm] = _arm_metrics(rows, arm)
    changed = sum(1 for r in rows
                  if r["arms"]["R2_MULTISTREET"].get("changed_vs_ss"))
    decided = [r for r in rows
               if r["arms"]["R2_MULTISTREET"].get("best_action")
               and r["arms"]["R1_REAL"].get("best_action")]
    changed_real = sum(1 for r in decided
                       if r["arms"]["R2_MULTISTREET"]["best_action"]
                       != r["arms"]["R1_REAL"]["best_action"])
    deltas = [r["arms"]["R2_MULTISTREET"]["call_delta"] for r in rows
              if r["arms"]["R2_MULTISTREET"].get("call_delta") is not None]
    ref = r1_reference()
    snap = {
        "snapshot_version": R2_BENCH_VERSION,
        "manifest": r2_manifest(),
        "split": collected.get("split"),
        "counts": collected.get("counts"),
        "arms": arms,
        "r2_vs_r1": {
            "changed_vs_single_street": changed,
            "changed_vs_R1_REAL_arm": changed_real,
            "n_compared": len(decided),
            "call_delta_median": (round(statistics.median(deltas), 6)
                                  if deltas else None),
            "call_delta_mean": (round(sum(deltas) / len(deltas), 6)
                                if deltas else None),
            "call_delta_positive_share": (round(sum(1 for d in deltas if d > 0)
                                                / len(deltas), 6)
                                          if deltas else None),
        },
        "r1_reference": ref,
        "equivalence_check": {
            "R1_REAL_reproduced": (arms["R1_REAL"]
                                   .get("observed_action_agreement")),
            "R1_published_real": (ref.get("decision_ev") or {})
            .get("observed_agreement_real"),
            "note": ("R1_REAL 臂必须与 R1 发布值逐位一致——否则评测口径漂移，"
                     "结论不可信（测试断言）"),
        },
        "arm_naming_note": ("R1_RANGE_ONLY = REAL 范围 + 无行为证据——**不是** "
                            "R1 发布的 BASELINE 臂（后者另用静态 config 范围）；"
                            "与 R1 的可比对照以 R1_REAL 臂为准（逐位复现）"),
        "verdict_inputs": {
            "call_agreement_r1_real": (arms["R1_REAL"]
                                       .get("agreement_by_observed", {})
                                       .get("call", {}).get("agreement")),
            "call_agreement_r2": (arms["R2_MULTISTREET"]
                                  .get("agreement_by_observed", {})
                                  .get("call", {}).get("agreement")),
            "check_agreement_r1_real": (arms["R1_REAL"]
                                        .get("agreement_by_observed", {})
                                        .get("check", {}).get("agreement")),
            "check_agreement_r2": (arms["R2_MULTISTREET"]
                                   .get("agreement_by_observed", {})
                                   .get("check", {}).get("agreement")),
        },
        "honesty": {
            "no_profitability": "不构成任何盈利性/GTO 证明",
            "no_single_metric_strength": ("agreement/regret 任一单独指标不得作为"
                                          "强度证明；须与证据等级同读"),
            "no_tuning": "预算与切分事前声明；未因结果调整任何常数",
            "anti_lookahead": "未来牌仅未见牌采样（置换不变性有测试）",
            "declarative": "R2 为声明式前瞻（叶子=冻结单街核），非 solver",
            "mtt_pko": "NOT VALIDATABLE WITH CURRENT DATA",
            "r1_preserved": "R1 快照与指标未被覆盖（独立文件）",
            "goal_status": ("R2 未改善 call/check 一致率（实测恶化）——A1 天花板"
                            "未解决；详见报告 §6/§9"),
        },
    }
    R2_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    R2_SNAPSHOT.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return snap


def build_r2_snapshot(hands: list[dict] | None = None, **kw) -> dict:
    return snapshot_from_collected(run_benchmark(hands, **kw))
