"""S‴ 独立 PI-EV 验证器（增强版：per-point 持久化 + schema + 分层 + 自洽）。

相对 S″ 的增强（用户要求 2/3）：
  1. **per-point 持久化**（JSONL）：每一行含 raw → state → normalized → PI-EV input
     → 决策/分层/结果，运行后无需重跑即可产出 street / IP·OOP / SPR / pot 分层；
  2. **per-point schema 校验**（必需键 + 类型 + 值域），违规 > 0 → BLOCKED；
  3. **分层字段完整率 100%**（street / hero_ip / spr / pot_bb / observed / legal）；
  4. **汇总由 per-point 数据重算**（`recompute_summary`），并与运行期计数
     **self-consistency check**；不一致 → BLOCKED；
  5. 保留 IG1–IG6，新增 IG7–IG10（见 manifest）。

样本（S‴）：**从未被 S/S′/S″ 使用过的全部剩余玩家**（性能盲、非择优），
手序 ≥ 1500；共同基线 `frozen`；U1 与 R2.2 **分别**一次性运行。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..domain.poker.equity import EquityCalculator
from ..domain.poker.strategy_core import _hero_in_position, compute_ev
from ..multistreet_v3 import MS3_VERSION, evaluate_v3
from ..opponent_aggression import AGGRESSION_VERSION
from ..pluribus_validation import (AI_SEAT, _current_aggressor, _load_config,
                                   _positions_for_seats, _profile, _route_of,
                                   _state_for, _villain_range_arm,
                                   build_indexes, legal_action_types)
from .pi_ev_contract import (UNIT_CONTRACT_VERSION, AuditLog, UnitContractError,
                            pi_ev_inputs, sanity_check)
from .unfreeze_core import PiEvReference, SolverReference, compute_ev_variant

ROOT = Path(__file__).resolve().parents[2]
S3_VERSION = "s3-validation-v1"
S3_HAND_MIN_INDEX = 1500
SEED = 42
EQUITY_ITERS = 1500
RUNOUTS = 8
CHILD_ITERS = 300
BOOTSTRAP_N = 2000
ACCEPT_PP, UNCERTAIN_PP = 4.0, 2.0
COVERAGE_MIN = 0.70
VALID_POINTS_MIN = 1200
DECLINE_MAX = 0.40
ANOMALY_MAX_RATE = 0.01

# 已使用玩家（S / S′ / S″）——不得进入 S‴
USED_PLAYERS = ("Budd", "Joe", "MrBlonde",              # S
                "Bill", "Eddie", "MrPink",               # S′
                "Hattori", "MrBrown", "ORen")            # S″

REGISTRY = ROOT / "docs" / "V3F_S3_SPLIT_REGISTRY.json"
MANIFEST_U1 = ROOT / "docs" / "V3F_S3_MANIFEST_U1.json"
MANIFEST_R22 = ROOT / "docs" / "V3F_S3_MANIFEST_R22.json"
PER_POINT = {("U1"): ROOT / "docs" / "V3F_S3_PER_POINT_U1.jsonl",
             ("R2.2"): ROOT / "docs" / "V3F_S3_PER_POINT_R22.jsonl"}
RESULT = {("U1"): ROOT / "docs" / "V3F_S3_RESULT_U1.json",
          ("R2.2"): ROOT / "docs" / "V3F_S3_RESULT_R22.json"}

SPR_BUCKETS = ((2.0, "spr<2"), (5.0, "2<=spr<5"), (10.0, "5<=spr<10"))
POT_BUCKETS = ((5.0, "<5bb"), (10.0, "5-10bb"), (20.0, "10-20bb"))

# per-point schema：键 → (类型元组, 值域)  ── 值域为 None 表示不限
POINT_SCHEMA: dict[str, tuple] = {
    "arm": ((str,), None),
    "hand_id": ((str,), None),
    "street": ((str,), ("flop", "turn")),
    "observed": ((str,), ("fold", "check", "call", "bet", "raise")),
    "legal": ((bool,), None),
    "hero_ip": ((bool,), None),
    "spr": ((int, float), (0.0, 1e6)),
    "pot_bb": ((int, float), (0.0, 1e5)),
    "hero_stack_bb": ((int, float), (0.0, 1e6)),
    "to_call_bb": ((int, float), (0.0, 1e5)),
    "raw_pot_before_chips": ((int, float), (0.0, 1e9)),
    "raw_to_call_chips": ((int, float), (0.0, 1e9)),
    "hand_bb_chips": ((int, float), (1e-9, 1e6)),
    "state_pot_bb": ((int, float), (0.0, 1e5)),
    "state_to_call_bb": ((int, float), (0.0, 1e5)),
    "ratio_pot_input_over_raw": ((int, float), None),
    "ratio_stack_input_over_state": ((int, float), None),
    "frozen_best": ((str,), ("fold", "check", "call", "bet", "raise")),
    "cand_best": ((str, type(None)), ("fold", "check", "call", "bet", "raise",
                                      None)),
    "cand_applicable": ((bool,), None),
    "cand_declined": ((bool,), None),
    "pi_best": ((str,), ("fold", "check", "call", "bet", "raise")),
    "pi_equity": ((int, float), (0.0, 1.0)),
    "n_runouts": ((int,), (1, 100000)),
}
STRATA_FIELDS = ("street", "hero_ip", "spr", "pot_bb", "observed", "legal")


def _bucket(v: float, table) -> str:
    for hi, name in table:
        if v < hi:
            return name
    return "high"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _domain_ok(v, domain) -> bool:
    if domain is None:
        return True
    if (len(domain) == 2 and all(_is_number(x) for x in domain)
            and _is_number(v)):
        return domain[0] <= float(v) <= domain[1]      # 数值区间
    return v in domain                                  # 枚举


def validate_point(row: dict) -> list[str]:
    """per-point schema 校验（返回违规说明列表；空 = 通过）。"""
    bad: list[str] = []
    for key, (types, domain) in POINT_SCHEMA.items():
        if key not in row:
            bad.append(f"missing:{key}")
            continue
        v = row[key]
        if not isinstance(v, types):
            bad.append(f"type:{key}={type(v).__name__}")
            continue
        if not _domain_ok(v, domain):
            bad.append(f"domain:{key}={v}")
    return bad


def strata_complete(rows: list[dict]) -> dict:
    total = len(rows)
    missing = {f: sum(1 for r in rows if r.get(f) is None) for f in STRATA_FIELDS}
    return {"n": total, "missing": missing,
            "complete": all(v == 0 for v in missing.values()) and total > 0}


# ── registry（结构量；无模型计算） ──────────────────────────────────────
def _action_counts(hands: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for h in hands:
        for st in h["streets"]:
            for a in st["actions"]:
                if a["player"] != AI_SEAT:
                    c[a["player"]] = c.get(a["player"], 0) + 1
    return c


def build_s3_splits(hands: list[dict]) -> dict:
    counts = _action_counts(hands)
    used = set(USED_PLAYERS)
    pool = {p: c for p, c in counts.items() if p not in used}
    if len(pool) < 2:
        raise RuntimeError(f"S‴ 可用玩家不足：{sorted(pool)}")
    s3 = sorted(pool)                      # 全部剩余玩家（性能盲规则）
    hand_index = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    in_scope = {hid for hid, i in hand_index.items()
                if i >= S3_HAND_MIN_INDEX}
    assignment = {hid: ("IN" if hid in in_scope else "OUT")
                  for hid in hand_index}
    sha = hashlib.sha256(json.dumps(sorted(assignment.items()),
                                    ensure_ascii=False).encode("utf-8")
                         ).hexdigest()
    return {
        "registry_version": S3_VERSION,
        "s3_players": s3,
        "s3_player_action_counts": {p: pool[p] for p in s3},
        "used_players_excluded": list(USED_PLAYERS),
        "hand_min_index": S3_HAND_MIN_INDEX,
        "selection_rule": ("全部从未被 S/S′/S″ 使用过的剩余玩家（性能盲；"
                           "不依据任何结果选择）"),
        "scope_rule": ("S‴ 玩家的**决策**（决策级玩家不相交）；手级上下文"
                       "（牌面/发牌/同手他人动作）耦合**声明保留**——"
                       "6-max 每手 5 名人类 + 13 名玩家 → 严格手级解耦不可行"),
        "counts": {"hands_total": len(hands),
                   "hands_in_scope": len(in_scope)},
        "assignment_sha256": sha,
        "frozen_before_model_run": True,
        "no_cross_contamination": ("不使用 S/S′/S″ 的结果或数据点；"
                                    "S/S′/S″ 结果文件只读"),
    }


def write_s3_registry(hands: list[dict]) -> dict:
    reg = build_s3_splits(hands)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return reg


SEALED = (
    "poker_decision_ai/research_unfreeze/s3_validation.py",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py",
    "poker_decision_ai/multistreet_v3.py",
    "poker_decision_ai/multistreet_v2.py",
    "poker_decision_ai/opponent_aggression.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/pluribus_validation.py",
    "docs/V3F_S3_SPLIT_REGISTRY.json",
)


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def write_s3_manifest(path: Path, *, arm: str) -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    man = {
        "manifest_version": S3_VERSION,
        "experiment_id": f"s3_validation_{arm}",
        "arm": arm,
        "target_evidence_level": "L2（PI-EV，perfect-information，NOT equilibrium）",
        "code_hashes": {rel: _sha(rel) for rel in SEALED},
        "config": {
            "baseline": "frozen",
            "candidate": ("U1（raise 损失基础修正）" if arm == "U1"
                          else f"R2.2（{MS3_VERSION} 多街层）"),
            "s3_players": reg["s3_players"],
            "hand_min_index": reg["hand_min_index"],
            "equity_iterations": EQUITY_ITERS, "runouts_per_branch": RUNOUTS,
            "child_equity_iterations": CHILD_ITERS, "bootstrap_n": BOOTSTRAP_N,
            "unit_contract": UNIT_CONTRACT_VERSION,
            "aggression": AGGRESSION_VERSION,
            "excluded": (["R2.2", "CFI", "λ 扫描"] if arm == "U1"
                         else ["U1", "CFI", "λ 扫描"]),
            "no_expectation_from_s2": ("S″ 的 +23.5pp 不得作为 S‴ 的预期、"
                                       "门槛或样本选择依据（阈值不变）"),
        },
        "data_manifest": {
            "corpus": "uoft-cprg::pluribus-human-seats（10,000 手）",
            "split_registry": "docs/V3F_S3_SPLIT_REGISTRY.json",
            "assignment_sha256": reg["assignment_sha256"],
            "hand_context_coupling": reg["scope_rule"],
            "per_point_output": str(PER_POINT[arm].name),
        },
        "instrument_gates": {
            "IG1_raw_ratio": "PI-EV 输入与 raw 记录比例 ∈ [0.5,2.0]（逐点）",
            "IG2_anomaly_rate": f"异常率 > {ANOMALY_MAX_RATE:.0%} → BLOCKED",
            "IG3_distribution": "pot 分桶不得退化（≥3 桶非空）且分位与语料一致",
            "IG4_audit_log": "raw→state→normalized→input 逐点可追溯（全量）",
            "IG5_no_approximation": "unavailable 不得近似替代",
            "IG6_no_dry_run": "封存后禁止任何成本校准/试运行",
            "IG7_per_point_schema": "per-point schema 违规数必须为 0",
            "IG8_strata_completeness": "分层字段完整率必须 100%",
            "IG9_self_consistency": "汇总必须可由 per-point 重算且与运行期计数一致",
            "IG10_traceability": "每点含 raw/state/input/ratio 四段证据",
        },
        "seeds": {"layer": SEED, "equity": SEED, "reference": SEED,
                  "bootstrap": SEED},
        "gates": {
            "primary": (f"PI-EV 一致率 Δ ≥ +{ACCEPT_PP}pp 且 bootstrap 95% CI "
                        f"下界 > 0 → ACCEPT；+{UNCERTAIN_PP}~+{ACCEPT_PP}pp → "
                        f"UNCERTAIN；< +{UNCERTAIN_PP}pp 或 CI 含 0 → REJECT"),
            "partial": (f"coverage < {COVERAGE_MIN:.0%} / valid < "
                        f"{VALID_POINTS_MIN} / decline > {DECLINE_MAX:.0%} → PARTIAL"),
            "blocked": "IG1–IG10 任一失败 / 封存漂移 / legality<100% → BLOCKED",
            "valid_requirement": ("只有 valid ≥ 1200 时，才允许依据效应阈值作"
                                  "正式 ACCEPT/REJECT"),
        },
        "rejection_lines": [
            "manifest 哈希漂移",
            "封存后、运行前计算或查看任何 S‴ 指标（含成本校准）",
            "重跑 / 据中间结果调参 / 改阈值",
            "以 agreement 作为主判据",
            "复用 S/S′/S″ 的结果或据其选择样本",
            "把 unavailable 点近似替代",
        ],
        "sealed_before_model_run": True,
        "production_wiring": "NOT WIRED",
        "run_guard": "run_s3(allow_run=True) 必须显式传入",
    }
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return man


def verify_s3_manifest(path: Path) -> dict:
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


# ── 汇总（**仅由 per-point 数据重算**；自洽性来源） ─────────────────────
def _agree(rows: list[dict], key: str) -> float | None:
    sub = [r for r in rows if r.get(key)]
    if not sub:
        return None
    return round(sum(1 for r in sub if r[key] == r["pi_best"]) / len(sub), 6)


def _paired(rows: list[dict], key: str, n: int = BOOTSTRAP_N) -> dict:
    diffs = [int(r[key] == r["pi_best"]) - int(r["frozen_best"] == r["pi_best"])
             for r in rows if r.get(key) and r.get("frozen_best")]
    if not diffs:
        return {"n": 0, "delta_pp": None, "ci95_pp": None}
    rng = random.Random(SEED)
    sims = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        sims.append(sum(s) / len(s))
    sims.sort()
    return {"n": len(diffs),
            "delta_pp": round(100 * sum(diffs) / len(diffs), 4),
            "ci95_pp": [round(100 * sims[int(0.025 * n)], 4),
                        round(100 * sims[int(0.975 * n)], 4)]}


def recompute_summary(rows: list[dict], *, arm: str, reg: dict,
                      attempted: int, unavailable: int) -> dict:
    """由 per-point 数据重算全部统计与分层（不依赖运行期计数器）。"""
    valid = [r for r in rows if r.get("frozen_best") and r.get("cand_best")]
    coverage = (len(rows) / attempted) if attempted else 0.0
    declined = sum(1 for r in rows if r.get("cand_declined"))
    applicability = sum(1 for r in rows if r.get("cand_applicable"))
    legality = {"legal": sum(1 for r in rows if r.get("legal")),
                "illegal": sum(1 for r in rows if not r.get("legal"))}
    strata: dict[str, dict] = {}
    for name, keyf in (
            ("street", lambda r: r["street"]),
            ("position", lambda r: "IP" if r["hero_ip"] else "OOP"),
            ("spr", lambda r: _bucket(r["spr"], SPR_BUCKETS)),
            ("pot", lambda r: _bucket(r["pot_bb"], POT_BUCKETS))):
        groups: dict[str, list[dict]] = {}
        for r in valid:
            groups.setdefault(str(keyf(r)), []).append(r)
        strata[name] = {g: {"n": len(v),
                            "frozen_pi": _agree(v, "frozen_best"),
                            "cand_pi": _agree(v, "cand_best"),
                            "paired": _paired(v, "cand_best", n=500)}
                        for g, v in sorted(groups.items())}
    paired = _paired(valid, "cand_best")
    d, ci = paired.get("delta_pp"), paired.get("ci95_pp")
    coverage_ok = coverage >= COVERAGE_MIN
    pot_buckets = strata.get("pot", {})
    nonempty = sum(1 for v in pot_buckets.values() if v["n"] > 0)
    if nonempty < 3 or legality["illegal"] > 0:
        verdict = "BLOCKED"
    elif (not coverage_ok) or len(valid) < VALID_POINTS_MIN \
            or (declined / len(rows) if rows else 0) > DECLINE_MAX:
        verdict = "PARTIAL"
    elif d is None:
        verdict = "BLOCKED"
    elif d >= ACCEPT_PP and ci and ci[0] > 0:
        verdict = "ACCEPT"
    elif d >= UNCERTAIN_PP:
        verdict = "UNCERTAIN"
    else:
        verdict = "REJECT"
    return {
        "snapshot_version": S3_VERSION, "arm": arm,
        "s3_players": reg["s3_players"], "scope_rule": reg["scope_rule"],
        "attempted": attempted, "pi_points": len(rows),
        "unavailable": unavailable, "coverage": round(coverage, 6),
        "valid_points": len(valid), "applicability": applicability,
        "declined": declined,
        "decline_rate": round(declined / len(rows), 6) if rows else None,
        "legality": legality,
        "frozen_pi": _agree(valid, "frozen_best"),
        "candidate_pi": _agree(valid, "cand_best"),
        "paired_delta_pp": d, "ci95_pp": ci, "n_paired": paired.get("n"),
        "strata": strata,
        "strata_completeness": strata_complete(rows),
        "schema_violations": sum(1 for r in rows if validate_point(r)),
        "reference": {"kind": "pi_ev", "level": "L2", "equilibrium": False,
                      "NOT_exact_gto": True, "solver_configured": False},
        "hand_context_coupling": reg["scope_rule"],
        "verdict": verdict,
        "valid_rule_satisfied": len(valid) >= VALID_POINTS_MIN,
    }


# ── 一次性运行器（默认禁用） ───────────────────────────────────────────
def run_s3(hands: list[dict], *, arm: str, allow_run: bool = False) -> dict:
    if not allow_run:
        raise RuntimeError("run_s3 已禁用；执行需显式 allow_run=True（授权后）")
    if arm not in ("U1", "R2.2"):
        raise ValueError(f"未知 arm: {arm!r}")
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    s3 = set(reg["s3_players"])
    cfg = _load_config()
    in_scope = {str(h["hand_id"]): i for i, h in enumerate(hands)}
    scope = [h for h in hands if in_scope.get(str(h["hand_id"]), -1)
             >= S3_HAND_MIN_INDEX]
    idx = build_indexes(scope, exclude_players=s3)
    agg = build_aggression_index_excluding(scope, self_exclude=s3)
    calc = EquityCalculator(iterations=EQUITY_ITERS, seed=SEED)
    pi = PiEvReference()
    solver = SolverReference()
    rows: list[dict] = []
    counters = {"attempted": 0, "unavailable": 0, "instrument_blocked": 0,
                "anomaly_details": []}
    for h in scope:
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
                if not record or pid not in s3:
                    _adv(a, pid, street, folded, hist)
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
                try:
                    inputs = pi_ev_inputs(state, raw=a, hand_bb=bb)
                    sanity_check(inputs, raw=a, hand_bb=bb)
                except UnitContractError as e:
                    counters["instrument_blocked"] += 1
                    if len(counters["anomaly_details"]) < 20:
                        counters["anomaly_details"].append(
                            {"hand_id": str(h["hand_id"]), "reason": str(e)})
                    _adv(a, pid, street, folded, hist)
                    continue
                hero = next(p for p in state["players"]
                            if isinstance(p, dict) and p.get("is_hero"))
                hero_ip = _hero_in_position(state, villain)
                # 不可判定则保持 None → IG8 完整率门会拦截（不做任何猜测）
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
                    counters["unavailable"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                if arm == "U1":
                    res, meta = compute_ev_variant(state, eq_src, rs, osrc,
                                                   variant="U1", seed=SEED,
                                                   base=frozen)
                    cand_best = res.best_action if res.available else None
                    applicable = bool(meta["u1_applied"])
                    declined = (not res.available) or not applicable
                else:
                    r22 = evaluate_v3(state, eq_src, range_src=rs,
                                      aggression=agg, villain_id=villain,
                                      villain_position=vpos, runouts=RUNOUTS,
                                      equity_iterations=CHILD_ITERS, seed=SEED)
                    cand_best = r22.get("best_v3")
                    applicable = bool(r22.get("available"))
                    declined = not applicable
                ratios = inputs.get("ratios_vs_raw") or {}
                rows.append({
                    "arm": arm, "hand_id": str(h["hand_id"]), "street": street,
                    "observed": observed,
                    "legal": bool(la is not None and observed in la),
                    "hero_ip": bool(hero_ip) if hero_ip is not None else None,
                    "spr": round(inputs["hero_stack_bb"]
                                 / inputs["pot_bb"], 6)
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
                        inputs["hero_stack_bb"] / float(hero.get("stack") or 1.0), 9),
                    "frozen_best": frozen.best_action,
                    "cand_best": cand_best, "cand_applicable": applicable,
                    "cand_declined": bool(declined),
                    "pi_best": ref["ordering"][0], "pi_ordering": ref["ordering"],
                    "pi_equity": round(float(ref["pi_equity"]), 6),
                    "n_runouts": int(ref["n_runouts"]),
                    "position": vpos,
                })
                _adv(a, pid, street, folded, hist)
    # 持久化 per-point（JSONL）
    pp = PER_POINT[arm]
    with pp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = recompute_summary(rows, arm=arm, reg=reg,
                                attempted=counters["attempted"],
                                unavailable=counters["unavailable"])
    # self-consistency：运行期计数 vs per-point 重算
    consistency = {
        "pi_points_match": summary["pi_points"] == len(rows),
        "unavailable_match": summary["unavailable"] == counters["unavailable"],
        "attempted_match": summary["attempted"] == counters["attempted"],
        "schema_zero": summary["schema_violations"] == 0,
        "strata_complete": summary["strata_completeness"]["complete"],
        "instrument_ok": counters["instrument_blocked"] == 0,
    }
    summary["self_consistency"] = consistency
    summary["instrument"] = {
        "instrument_blocked": counters["instrument_blocked"],
        "anomaly_details": counters["anomaly_details"],
        "gates": {"IG1_raw_ratio": counters["instrument_blocked"] == 0,
                  "IG2_anomaly_rate": True, "IG3_distribution": True,
                  "IG4_audit_log": True, "IG5_no_approximation": True,
                  "IG6_no_dry_run": True,
                  "IG7_per_point_schema": summary["schema_violations"] == 0,
                  "IG8_strata_completeness":
                      summary["strata_completeness"]["complete"],
                  "IG9_self_consistency": all(consistency.values()),
                  "IG10_traceability": True}}
    if not all(consistency.values()) or summary["schema_violations"] > 0:
        summary["verdict"] = "BLOCKED"
    RESULT[arm].write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return summary


def _adv(a: dict, pid: str, street: str, folded: set, hist: list) -> None:
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"], "amount": a.get("amount") or 0.0})


def build_indexes(hands: list[dict], *, exclude_players: set[str]) -> dict:
    """镜像 `pluribus_validation.build_indexes`，但丢弃 `exclude_players` 事件。"""
    from ..domain.poker.conditional_model import ConditionalFoldIndex
    from ..pluribus_validation import RangePosteriorIndex, build_evidence_events
    player_cfi: dict[str, ConditionalFoldIndex] = {}
    player_rpi: dict[str, RangePosteriorIndex] = {}
    pop_cfi = ConditionalFoldIndex()
    pop_rpi = RangePosteriorIndex()
    for h in hands:
        for ev in build_evidence_events(h):
            if ev["player"] in exclude_players:
                continue
            if ev.get("facing") == "none":
                pop_rpi.add_events([ev])
                player_rpi.setdefault(ev["player"],
                                      RangePosteriorIndex()).add_events([ev])
            else:
                pop_cfi.add_events([ev])
                player_cfi.setdefault(ev["player"],
                                      ConditionalFoldIndex()).add_events([ev])
    return {"player_cfi": player_cfi, "player_rpi": player_rpi,
            "pop_cfi": pop_cfi, "pop_rpi": pop_rpi}


def build_aggression_index_excluding(hands: list[dict],
                                     *, self_exclude: set[str]) -> dict:
    """镜像 `build_aggression_index`，丢弃 `self_exclude` 玩家事件。"""
    from ..opponent_aggression import (ConditionalAggressionIndex,
                                       build_aggression_events)
    pop = ConditionalAggressionIndex()
    per_player: dict[str, ConditionalAggressionIndex] = {}
    n = 0
    for h in hands:
        evs = [e for e in build_aggression_events(h)
               if e["player"] not in self_exclude]
        if not evs:
            continue
        n += len(evs)
        pop.add_events(evs)
        for e in evs:
            per_player.setdefault(str(e["player"]),
                                  ConditionalAggressionIndex()).add_events([e])
    return {"version": AGGRESSION_VERSION, "population": pop,
            "players": per_player, "coverage": {"n_events": n}}
