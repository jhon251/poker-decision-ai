"""第一次解冻实验：预注册矩阵 + PI-EV 主判据 + G1–G6（研究分支）。

纪律
----
* **先封存后运行**：manifest（代码/配置/数据/split/种子/门槛/拒绝线）在**任何
  S 集计算之前**写入并哈希化；运行后校验零漂移。
* **S 集一次性**：S = 决策级玩家不相交（3 名保留玩家）；不得据 S 结果迭代。
* **主判据**：S 集 **PI-EV 排序一致率**提升 ≥ +3.0pp 且 bootstrap 95% CI
  下界 > 0；若 S 有效决策点 < 1,200 → 判 PARTIAL（功效不足），**不得扩大 S**。
* **全配置上报**：矩阵 {frozen, U1, U1+U3, U1+U2+U3} × λ 敏感性，禁止择优隐藏。
* **不修改**：strategy_core / CFI / λ / fold·raise 排序、生产链、Vision/State 等。
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from ..domain.poker.equity import EquityCalculator
from ..domain.poker.strategy_core import compute_ev
from ..pluribus_validation import (AI_SEAT, _current_aggressor, _load_config,
                                   _positions_for_seats, _profile, _route_of,
                                   _state_for, _villain_range_arm,
                                   build_indexes, legal_action_types)
from ..research_unfreeze import (LAMBDA_SCALES, PiEvReference, SolverReference,
                                 UNFREEZE_VERSION, compute_ev_variant)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "V3F_UNFREEZE_SPLIT_REGISTRY.json"
MANIFEST_U = ROOT / "docs" / "V3F_UNFREEZE_MANIFEST_UMATRIX.json"
MANIFEST_CFI = ROOT / "docs" / "V3F_UNFREEZE_MANIFEST_CFI.json"
SNAP_R1P = ROOT / "docs" / "V3F_R1P_REBASELINE.json"
SNAP_R2P = ROOT / "docs" / "V3F_R2P_REBASELINE.json"
SNAP_U = ROOT / "docs" / "V3F_UNFREEZE_UMATRIX_RESULT.json"
SNAP_CFI = ROOT / "docs" / "V3F_UNFREEZE_CFI_GRID.json"
SNAP_DELTA = ROOT / "docs" / "V3F_UNFREEZE_DELTA.json"

SEED = 42
EQUITY_ITERS = 1500
BOOTSTRAP_N = 2000
MIN_S_POINTS = 1200                 # 预注册：低于此值 → PARTIAL
G2_MIN_GAIN_PP = 3.0                # 预注册主判据
G3_BRIER_TOL = 0.002
G4_AGREEMENT_TOL_PP = 0.5
G4_BETBIAS_TOL_PP = 2.0
VARIANTS = ("frozen", "U1", "U1+U3", "U1+U2+U3")


# ── split（与冻结 registry 完全一致；运行时校验哈希） ───────────────────
def _hkey(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def assign_split(hands: list[dict], registry: dict) -> dict[str, str]:
    s_players = set(registry["assignment"]["s_players"])
    f_frac = 0.70
    d_frac = 0.15
    out: dict[str, str] = {}
    for h in hands:
        r = (_hkey(str(h["hand_id"])) % 1000) / 1000.0
        hs = "F" if r < f_frac else ("D" if r < f_frac + d_frac else "C")
        out[str(h["hand_id"])] = hs
    return out, s_players                                  # type: ignore[return-value]


# ── 采点（frozen 管线；S/C/D 由调用方选择） ─────────────────────────────
def collect_points(hands: list[dict], *, registry: dict, splits: tuple[str, ...],
                   max_points: int = 0, seed: int = SEED,
                   with_r2p: bool = False, r2p_split: str = "S",
                   r2p_cap: int = 0) -> dict:
    cfg = _load_config()
    hand_split, s_players = assign_split(hands, registry)
    f_hands = [h for h in hands if hand_split[str(h["hand_id"])] == "F"]
    idx = build_indexes(f_hands)
    calc = EquityCalculator(iterations=EQUITY_ITERS, seed=seed)
    pi = PiEvReference()
    solver = SolverReference()
    points: list[dict] = []
    counts: dict[str, Any] = {
        "files": {"registry": str(REGISTRY.name)},
        "solver_configured": solver.is_configured(),
        "solver_unavailable_reason": solver.unavailable_reason(),
        "pi_points": 0, "pi_unavailable": 0, "points": 0,
        "legal": 0, "illegal": 0, "skipped_no_range": 0,
        "skipped_no_hole": 0, "skipped_no_villain": 0,
        "skipped_street": 0, "runouts_total": 0, "r2p_points": 0}
    agg_idx = None
    if with_r2p:
        from ..opponent_aggression import build_aggression_index
        agg_idx = build_aggression_index(f_hands)
    for h in hands:
        hid = str(h["hand_id"])
        hs = hand_split[hid]
        humans = [p for p in h["players"] if p != AI_SEAT]
        boards = h["board_by_street"]
        bb = h["bb"] or 1.0
        n = len(h["players"])
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(h["players"])}
        folded: set[str] = set()
        hist: list[dict] = []
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
            if not record:
                counts["skipped_street"] += 1
            for a in st["actions"]:
                pid = a["player"]
                if pid == AI_SEAT:
                    if a["action"] in ("bet", "raise", "call"):
                        hist.append({"street": street, "player_id": pid,
                                     "action": a["action"],
                                     "amount": a.get("amount") or 0.0})
                    continue
                if not record:
                    _adv(a, pid, street, folded, hist)
                    continue
                # split 归属：S 玩家决策 → S；否则按手牌哈希
                pt_split = "S" if pid in s_players else hs
                if pt_split not in splits:
                    _adv(a, pid, street, folded, hist)
                    continue
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
                vr, _prov = _villain_range_arm(
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
                                        iterations=EQUITY_ITERS, seed=seed)
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
                base = compute_ev(state, eq_src, range_src=rs,
                                  opponent_src=osrc, seed=seed)
                if not base.available or not base.candidates:
                    counts["skipped_no_range"] += 1
                    _adv(a, pid, street, folded, hist)
                    continue
                cands = [{"action": c.action, "amount_bb": c.amount_bb}
                         for c in base.candidates]
                # 活跃对手手牌（PI-EV 需要；Pluribus 全明牌）
                villain_hands = []
                for p in humans:
                    if p == pid or p not in h["hole_cards"]:
                        continue
                    cards = list(h["hole_cards"].get(p) or [])
                    if len(cards) == 2 and p not in folded:
                        villain_hands.append(cards)
                ref = pi.reference(
                    hero_cards=hole, villain_hands=villain_hands, board=board,
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
                counts["runouts_total"] += int(ref.get("n_runouts") or 0)
                variants: dict[str, dict] = {}
                for v in VARIANTS:
                    res, meta = compute_ev_variant(
                        state, eq_src, rs, osrc, variant=v, seed=seed,
                        base=base)
                    if not res.available:
                        continue
                    variants[v] = {
                        "best": res.best_action,
                        "best_amount_bb": res.best_amount_bb,
                        "ranking": [{"action": c.action,
                                     "ev_bb": round(float(c.ev_bb or 0.0), 6),
                                     "utility_bb": (None if c.utility_bb is None
                                                    else round(float(c.utility_bb), 6))}
                                    for c in res.candidates],
                        "u1_applied": meta["u1_applied"],
                    }
                # λ 敏感性（仅对 U1+U2+U3 变体扫描）
                lam: dict[str, str] = {}
                for ls in LAMBDA_SCALES:
                    res, _ = compute_ev_variant(state, eq_src, rs, osrc,
                                                variant="U1+U2+U3",
                                                lambda_scale=ls, seed=seed,
                                                base=base)
                    if res.available:
                        lam[str(ls)] = res.best_action
                # R2′ 参照行（R2.2 v3 层；仅指定 split + 声明上限内计算）
                r2p: dict | None = None
                if (with_r2p and pt_split == r2p_split
                        and (not r2p_cap or counts["r2p_points"] < r2p_cap)):
                    from ..multistreet_v3 import evaluate_v3
                    r2 = evaluate_v3(state, eq_src, range_src=rs,
                                     aggression=agg_idx, villain_id=villain,
                                     villain_position=vpos, runouts=4,
                                     equity_iterations=150, seed=seed)
                    r2p = {"available": bool(r2.get("available")),
                           "best": r2.get("best_v3"),
                           "decline_reason": r2.get("decline_reason"),
                           "node_order": (r2.get("provenance") or {})
                           .get("node_order")}
                    counts["r2p_points"] += 1
                points.append({
                    "hand_id": hid, "split": pt_split, "street": street,
                    "player": pid, "observed": observed,
                    "position": vpos, "legal": bool(la is not None
                                                    and observed in la),
                    "variants": variants,
                    "pi": {"ordering": ref["ordering"],
                           "best": ref["ordering"][0],
                           "eq": ref["pi_equity"],
                           "n_runouts": ref["n_runouts"],
                           "provenance": ref["provenance"]},
                    "lambda_best": lam, "r2p": r2p,
                    "n_candidates": len(cands)})
                counts["points"] += 1
                if max_points and counts["points"] >= max_points:
                    return {"points": points, "counts": counts,
                            "hand_split": hand_split,
                            "s_players": sorted(s_players)}
                _adv(a, pid, street, folded, hist)
    return {"points": points, "counts": counts, "hand_split": hand_split,
            "s_players": sorted(s_players)}


def _adv(a: dict, pid: str, street: str, folded: set, hist: list) -> None:
    if a["action"] == "fold":
        folded.add(pid)
    if a["action"] in ("bet", "raise", "call"):
        hist.append({"street": street, "player_id": pid,
                     "action": a["action"], "amount": a.get("amount") or 0.0})


# ── 指标 ───────────────────────────────────────────────────────────────
def pi_agreement(points: list[dict], variant: str) -> dict:
    ok = tot = 0
    for p in points:
        v = p["variants"].get(variant)
        if not v:
            continue
        tot += 1
        ok += int(v["best"] == p["pi"]["best"])
    return {"n": tot, "pi_agreement": round(ok / tot, 6) if tot else None}


def paired_bootstrap(points: list[dict], a: str, b: str,
                     n: int = BOOTSTRAP_N) -> dict:
    """a 相对 b 的 PI-EV 一致率差（配对 bootstrap，点级重采样）。"""
    diffs: list[int] = []
    for p in points:
        va, vb = p["variants"].get(a), p["variants"].get(b)
        if not va or not vb:
            continue
        diffs.append(int(va["best"] == p["pi"]["best"])
                     - int(vb["best"] == p["pi"]["best"]))
    if not diffs:
        return {"n": 0, "delta_pp": None, "ci95": None}
    rng = random.Random(SEED)
    deltas = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        deltas.append(sum(sample) / len(sample))
    deltas.sort()
    return {"n": len(diffs),
            "delta_pp": round(100.0 * sum(diffs) / len(diffs), 4),
            "ci95_low_pp": round(100.0 * deltas[int(0.025 * n)], 4),
            "ci95_high_pp": round(100.0 * deltas[int(0.975 * n)], 4)}


def observed_metrics(points: list[dict], variant: str) -> dict:
    ok = tot = aggr = 0
    for p in points:
        v = p["variants"].get(variant)
        if not v:
            continue
        tot += 1
        ok += int(v["best"] == p["observed"])
        aggr += int(v["best"] in ("bet", "raise"))
    return {"n": tot, "observed_agreement": round(ok / tot, 6) if tot else None,
            "bet_bias_share": round(aggr / tot, 6) if tot else None}


def legality(points: list[dict]) -> dict:
    legal = sum(1 for p in points if p["legal"])
    return {"legal": legal, "illegal": len(points) - legal,
            "legal_share": round(legal / len(points), 6) if points else None}


def lambda_sensitivity(points: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for ls in LAMBDA_SCALES:
        flips = tot = 0
        for p in points:
            if str(ls) not in p["lambda_best"]:
                continue
            tot += 1
            flips += int(p["lambda_best"][str(ls)] != p["lambda_best"]["1.0"])
        out[str(ls)] = {"n": tot,
                        "flip_rate_vs_1.0": round(flips / tot, 6) if tot else None}
    return out


# ── manifest（封存） ───────────────────────────────────────────────────
def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


SEALED = (
    "poker_decision_ai/research_unfreeze/unfreeze_core.py",
    "poker_decision_ai/research_unfreeze/unfreeze_experiment.py",
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/domain/poker/conditional_model.py",
    "poker_decision_ai/pluribus_validation.py",
    "docs/V3F_UNFREEZE_SPLIT_REGISTRY.json",
)


def write_manifest(path: Path, *, experiment_id: str, config: dict) -> dict:
    reg = load_registry()
    man = {
        "manifest_version": "unfreeze-manifest-v1",
        "experiment_id": experiment_id,
        "target_evidence_level": "L2（PI-EV 参照；L4 需真 solver）",
        "code_hashes": {rel: _sha(rel) for rel in SEALED},
        "config": config,
        "data_manifest": {
            "corpus": "uoft-cprg::pluribus-human-seats（10,000 手）",
            "split_registry": "docs/V3F_UNFREEZE_SPLIT_REGISTRY.json",
            "split_assignment_sha256": reg["assignment"][
                "hand_assignment_sha256"],
            "s_assignment_sha256": reg["assignment"]["s_assignment_sha256"],
            "s_players": reg["assignment"]["s_players"],
        },
        "seeds": {"layer": SEED, "equity": SEED, "ref": SEED,
                  "bootstrap": SEED},
        "budget": {"equity_iterations": EQUITY_ITERS,
                   "pi_ev_runout_budget": 2048, "bootstrap_n": BOOTSTRAP_N},
        "gates": {
            "G1": "完整性（零变体等价 / 确定性 / legality 100% / 模式隔离 / "
                  "anti-lookahead）",
            "G2": f"S 集 PI-EV 一致率 ≥ +{G2_MIN_GAIN_PP}pp 且 bootstrap 95% "
                  f"CI 下界 > 0",
            "G3": f"fold Brier 不劣化 > +{G3_BRIER_TOL}",
            "G4": f"决策侧容忍带：observed agreement ±{G4_AGREEMENT_TOL_PP}pp、"
                  f"bet 偏置 ±{G4_BETBIAS_TOL_PP}pp、legality 100%",
            "G5": "重基线 delta 表（R1′/R2′）+ 受影响测试清单",
            "G6": "诚实性（L2 上限声明 / 无伪造 / 全配置上报）",
        },
        "rejection_lines": [
            "manifest 哈希漂移（代码/配置/数据/种子任一）",
            "S 集在封存前被计算，或据 S 结果迭代",
            "以 agreement 作为唯一或主验收标准",
            "参照不可用却宣称 L4 / 使用伪 solver",
            "用被测模型自身输出充当参照或标签",
            "隐藏劣化项或选择性上报",
            "修改 split 或 P 分层以求更好结果",
        ],
        "s_peek_declaration": ("本 manifest 在**任何 S 集计算之前**写入；"
                              "S 集只运行一次"),
        "production_wiring": "NOT WIRED（研究分支）",
    }
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return man


def verify_manifest(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    man = json.loads(path.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha(rel) != h}
    reg = load_registry()
    split_drift = (man["data_manifest"]["split_assignment_sha256"]
                   != reg["assignment"]["hand_assignment_sha256"])
    return {"present": True, "drift": drift,
            "clean": (not drift) and (not split_drift),
            "split_drift": split_drift,
            "checked": len(man.get("code_hashes") or {})}


# ── CFI 网格（**仅 C 集**；目标 = fold 校准） ──────────────────────────
CFI_MIN_SAMPLES_GRID = (20, 30, 50)


def run_cfi_grid(hands: list[dict], *, registry: dict, seed: int = SEED) -> dict:
    """C 集上评估 fold 校准（Brier）；仅 `min_samples` 维度可参数化。

    **审核发现**：先验强度 N0 与阶梯衰减位于冻结 `FoldEstimator` 内部，未暴露
    → 在当前「不解冻」边界内**不可扫描**（如实登记为 BLOCKED 维度）。
    """
    from ..domain.poker.conditional_model import ConditionalFoldIndex
    from ..pluribus_validation import (AI_SEAT, _agg_n_before, size_bucket,
                                       texture_tag)
    hand_split, s_players = assign_split(hands, registry)
    f_hands = [h for h in hands if hand_split[str(h["hand_id"])] == "F"]
    pop = ConditionalFoldIndex()
    per_player: dict[str, ConditionalFoldIndex] = {}
    for h in f_hands:
        for st in h["streets"]:
            for a in st["actions"]:
                if a["player"] == AI_SEAT:
                    continue
                to_call = float(a.get("to_call") or 0.0)
                if to_call <= 1e-9:
                    continue
                boards = h["board_by_street"]
                street = st["street"]
                board = [] if street == "preflop" else list(
                    boards.get("flop", []))
                if street in ("turn", "river"):
                    board = list(boards.get("flop", [])) + list(
                        boards.get("turn", []))
                if street == "river":
                    board = list(boards.get("flop", [])) + list(
                        boards.get("turn", [])) + list(boards.get("river", []))
                ev = {"player": a["player"], "street": street,
                      "facing": ("raise" if _agg_n_before(h, street, a) >= 2
                                 else "bet"),
                      "size_bucket": size_bucket(to_call,
                                                 a.get("pot_before") or 1.0),
                      "position": "", "texture": (texture_tag(board) if board
                                                  else "preflop"),
                      "action": a["action"],
                      "fold": 1 if a["action"] == "fold" else 0}
                pop.add_events([ev])
                per_player.setdefault(a["player"],
                                      ConditionalFoldIndex()).add_events([ev])
    # 评估集 = C 集（非 S 玩家 + hand 哈希 C）
    rows: list[tuple] = []
    n = len(hands[0]["players"]) if hands else 6
    for h in hands:
        if hand_split[str(h["hand_id"])] != "C":
            continue
        pos_map = _positions_for_seats(n, n)
        seat_of = {p: i + 1 for i, p in enumerate(h["players"])}
        boards = h["board_by_street"]
        for st in h["streets"]:
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
                    continue
                to_call = float(a.get("to_call") or 0.0)
                if to_call <= 1e-9 or a["action"] not in ("fold", "call"):
                    continue
                args = (street,
                        ("raise" if _agg_n_before(h, street, a) >= 2 else "bet"),
                        size_bucket(to_call, a.get("pot_before") or 1.0),
                        str(pos_map.get(seat_of.get(pid)) or ""),
                        texture_tag(board) if board else "preflop")
                y = 1 if a["action"] == "fold" else 0
                rows.append((pid, args, y))
    grid: dict[str, dict] = {}
    for ms in CFI_MIN_SAMPLES_GRID:
        briers: list[float] = []
        for pid, args, y in rows:
            pb = pop.estimate(*args)
            p_base = float(pb["p_fold"]) if pb is not None else 0.5
            pc = per_player.get(pid)
            pr = (pc.estimate(*args, population=pop, min_samples=ms)
                  if pc is not None else None)
            p = float(pr["p_fold"]) if pr is not None else p_base
            briers.append((p - y) ** 2)
        b = sum(briers) / len(briers) if briers else None
        grid[str(ms)] = {"n": len(briers),
                         "brier": round(b, 6) if b is not None else None}
    base_b = grid.get("30", {}).get("brier")
    chosen = min(grid, key=lambda k: (grid[k]["brier"] is None,
                                      grid[k]["brier"]))
    return {
        "grid_dimension": "min_samples（唯一可在不解冻前提下扫描的维度）",
        "blocked_dimensions": {
            "N0_prior_strength": ("位于冻结 FoldEstimator 内部，未暴露 → "
                                  "需单独授权方可解冻扫描"),
            "ladder_decay": "同上（阶梯权重在 ConditionalFoldIndex 内部）",
        },
        "evaluation_split": "C（选择集；未使用 S）",
        "fit_split": "F",
        "results": grid,
        "baseline_min_samples": 30,
        "baseline_brier": base_b,
        "chosen_min_samples": int(chosen),
        "chosen_brier": grid[chosen]["brier"],
        "improvement_vs_baseline": (round(base_b - grid[chosen]["brier"], 6)
                                    if base_b is not None else None),
        "g3_tolerance": G3_BRIER_TOL,
        "g3_pass": (base_b is not None and grid[chosen]["brier"] is not None
                    and (base_b - grid[chosen]["brier"]) >= -G3_BRIER_TOL),
        "honesty": ("仅 min_samples 维度可扫描；其余维度 BLOCKED（不解冻）；"
                    "CFI 的 S 集验收留待下一次一次性授权（S 不可重复使用）"),
    }
