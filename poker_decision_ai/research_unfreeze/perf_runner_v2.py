"""PI-EV 性能优化 **接入版 runner v2**（独立分支；不覆盖、不触碰旧 runner）。

定位
----
* 旧 runner（`s3_validation.run_s3`）与其封存产物 **完全不动**（只读依赖）；
* 本模块是**独立**的性能 runner：同样调用 sealed 的
  `_state_for / pi_ev_inputs / _villain_range_arm / compute_ev /
  compute_ev_variant / evaluate_v3 / PiEvReference`，只是把
  **S1（`_combo_equities` 跨实例记忆化，不可变存储）**、
  **S2（`child_equity` 记忆化）** 与 **S5（手级分片进程池）** 接入；
* 汇总/门/分层 **复用 sealed `recompute_summary` / `validate_point` /
  `strata_complete`**，因此 IG1–IG10、legality、coverage、unavailable
  的语义与旧 runner 同源；
* 输出全部写到 `docs/V3F_PERF_V2_*`（**新文件**），并有硬护栏拒绝写任何
  封存路径；`run_s3` 从不被调用（测试用 AST 断言）。

模式
----
`baseline1`   单进程、无缓存（与旧 runner 同路径的基线）
`s1s2_1`      单进程 + S1/S2
`s1s2_8`      8 workers + S1/S2（**默认**）
`s1s2_16`     16 workers + S1/S2（可选）
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

from ..pluribus_validation import _load_config
from . import perf_profile as pp
from . import s3_validation as s3

V2_VERSION = "pi-ev-perf-runner-v2"
V2_HAND_MAX_INDEX = 1500                 # 等价集：手序 < 1500（任何阶段均未使用）
V2_DEFAULT_WORKERS = 8
V2_MODES = ("baseline1", "s1s2_1", "s1s2_8", "s1s2_16")
V2_WORKERS = {"baseline1": 1, "s1s2_1": 1, "s1s2_8": 8, "s1s2_16": 16}

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs"
EQUIV_SET = OUT / "V3F_PERF_V2_EQUIV_SET.json"
MANIFEST = OUT / "V3F_PERF_V2_MANIFEST.json"
RESULT = OUT / "V3F_PERF_V2_RESULT.json"
PER_POINT = {(m, a): OUT / ("V3F_PERF_V2_PER_POINT_%s_%s.jsonl"
                            % (m.upper(), a.replace(".", "")))
             for m in V2_MODES for a in ("U1", "R2.2")}

# 硬护栏：绝不允许写这些路径（旧 runner 的封存产物）
FORBIDDEN_PATHS = ("V3F_S3_SPLIT_REGISTRY.json", "V3F_S3_MANIFEST_U1.json",
                   "V3F_S3_MANIFEST_R22.json", "V3F_S3_PER_POINT_U1.jsonl",
                   "V3F_S3_PER_POINT_R22.jsonl", "V3F_S3_RESULT_U1.json",
                   "V3F_S3_RESULT_R22.json")


class V2GuardError(RuntimeError):
    pass


def _sha(rel_or_abs) -> str:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _assert_not_sealed(path: Path) -> None:
    if path.name in FORBIDDEN_PATHS:
        raise V2GuardError(f"拒绝写入封存路径: {path}")
    if "V3F_S3_" in path.name:
        raise V2GuardError(f"拒绝写入 V3F_S3_* 路径: {path}")


# ── 等价集（与既有阶段不相交；只读语料） ──────────────────────────────
def build_equiv_set(hands: list) -> dict:
    ids = [str(h["hand_id"]) for i, h in enumerate(hands)
           if i < V2_HAND_MAX_INDEX]
    payload = "\n".join(ids).encode("utf-8")
    return {"equiv_version": V2_VERSION,
            "hand_max_index": V2_HAND_MAX_INDEX,
            "hands": len(ids),
            "assignment_sha256": hashlib.sha256(payload).hexdigest(),
            "player_rule": ("决策者 = 任意非 AI 座位；`build_indexes("
                            "exclude_players=set())`；不含任何阶段选样规则"),
            "disjoint_from": {"S": ">=1500", "S'": ">=1500", "S''": ">=1500",
                              "S'''": ">=1500"},
            "purpose": "仅用于 v2 性能 runner 的等价性/门一致性验证（无验收判定）",
            "first_hand_id": ids[0] if ids else None,
            "last_hand_id": ids[-1] if ids else None}


def write_equiv_set(hands: list) -> dict:
    _assert_not_sealed(EQUIV_SET)
    reg = build_equiv_set(hands)
    EQUIV_SET.write_text(json.dumps(reg, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return reg


def _equiv_hands(hands: list) -> list:
    return [h for i, h in enumerate(hands) if i < V2_HAND_MAX_INDEX]


# ── 封存（先封存，后运行） ────────────────────────────────────────────
V2_CODE = ("poker_decision_ai/research_unfreeze/perf_runner_v2.py",
           "poker_decision_ai/research_unfreeze/perf_profile.py",
           "poker_decision_ai/research_unfreeze/perf_optimize.py")
V2_READONLY_DEPS = tuple(s3.MANIFEST_U1.parent.joinpath("..").resolve()
                         .joinpath(p).as_posix() for p in ())  # 占位（见下）


def _dep_relpaths() -> tuple:
    return ("poker_decision_ai/research_unfreeze/s3_validation.py",
            "poker_decision_ai/research_unfreeze/pi_ev_contract.py",
            "poker_decision_ai/research_unfreeze/unfreeze_core.py",
            "poker_decision_ai/multistreet_v3.py",
            "poker_decision_ai/multistreet_v2.py",
            "poker_decision_ai/domain/poker/strategy_core.py",
            "poker_decision_ai/pluribus_validation.py",
            "docs/V3F_PERF_V2_EQUIV_SET.json")


def write_v2_manifest(arms=("U1", "R2.2"), modes=V2_MODES) -> dict:
    _assert_not_sealed(MANIFEST)
    man = {
        "runner_version": V2_VERSION,
        "arms": list(arms), "modes": list(modes),
        "default_workers": V2_DEFAULT_WORKERS,
        "workers_by_mode": {m: V2_WORKERS[m] for m in modes},
        "code_hashes": {rel: _sha(rel) for rel in V2_CODE},
        "readonly_dependency_hashes": {rel: _sha(rel)
                                       for rel in _dep_relpaths()},
        "sealed_s3_hashes_at_seal": {
            "U1": s3.verify_s3_manifest(s3.MANIFEST_U1),
            "R2.2": s3.verify_s3_manifest(s3.MANIFEST_R22)},
        "equiv_set": {"path": "docs/V3F_PERF_V2_EQUIV_SET.json",
                      "assignment_sha256": json.loads(
                          EQUIV_SET.read_text(encoding="utf-8")
                      )["assignment_sha256"]},
        "semantics": {
            "decision_set": "V3F_PERF_V2_EQUIV_SET.json（手序 < 1500）",
            "seeds": {"layer": 42, "equity": 42, "reference": 42,
                      "bootstrap": 42},
            "enumeration": "精确全枚举（combinations；无采样替代）",
            "runout_budget": "PI_EV_MAX_RUNOUTS = 2048（未修改）",
            "thresholds_readonly": {
                "accept_pp": s3.ACCEPT_PP, "uncertain_pp": s3.UNCERTAIN_PP,
                "coverage_min": s3.COVERAGE_MIN,
                "valid_points_min": s3.VALID_POINTS_MIN,
                "decline_max": s3.DECLINE_MAX,
                "anomaly_max_rate": s3.ANOMALY_MAX_RATE},
            "pi_ev_definition": "sealed PiEvReference（L2 / NOT_exact_gto）",
            "optimizations": {
                "S1": "FoldEstimator._combo_equities 跨实例记忆化（不可变存储 + 返回副本）",
                "S2": "child_equity 记忆化（float 值）",
                "S5": "手级分片进程池（结果按提交序收集）",
                "excluded": {"S4": "compute_ev 缓存（0 命中且返回可变对象）",
                             "S6": "evaluate7 容量扩容（实测无增益）"}},
            "unchanged": ["strategy_core", "CFI", "λ", "frozen kernel",
                          "Vision", "State", "生产链路", "PI-EV 定义",
                          "decision set(旧)", "seed", "budget", "阈值"]},
        "run_guard": "run_v2(..., allow_run=True) 必须显式传入",
        "production_wiring": "NOT WIRED",
        "no_old_overwrite": "FORBIDDEN_PATHS 硬护栏 + 运行前后 sealed 清单复验",
        "sealed_before_run": True,
        "host": {"python": platform.python_version(), "cpu": os.cpu_count()},
    }
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return man


def verify_v2_manifest() -> dict:
    if not MANIFEST.exists():
        return {"present": False}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drift = {rel: {"sealed": h, "now": _sha(rel)}
             for rel, h in (man.get("code_hashes") or {}).items()
             if _sha(rel) != h}
    dep_drift = {rel: {"sealed": h, "now": _sha(rel)}
                 for rel, h in (man.get("readonly_dependency_hashes")
                                or {}).items() if _sha(rel) != h}
    return {"present": True, "drift": drift, "dep_drift": dep_drift,
            "clean": not drift and not dep_drift,
            "checked": len(man.get("code_hashes") or {}),
            "checked_deps": len(man.get("readonly_dependency_hashes") or {})}


def sealed_s3_clean() -> dict:
    return {"U1": s3.verify_s3_manifest(s3.MANIFEST_U1)["clean"],
            "R2.2": s3.verify_s3_manifest(s3.MANIFEST_R22)["clean"]}


# ── 门（IG1–IG10；全部基于 sealed 语义） ──────────────────────────────
def assemble_gates(rows: list, summary: dict, counters: dict,
                   wall_s: float) -> dict:
    pot = sorted(r["pot_bb"] for r in rows if r.get("pot_bb") is not None)
    pct = ({k: pot[min(len(pot) - 1, int(math.ceil(q / 100 * len(pot))) - 1)]
            for k, q in (("p10", 10), ("p50", 50), ("p90", 90))} if pot else {})
    ratios_pot = [r.get("ratio_pot_input_over_raw") for r in rows]
    ratios_stk = [r.get("ratio_stack_input_over_state") for r in rows]
    ig1 = bool(rows) and all(
        r is not None and 0.5 <= r <= 2.0
        for r in ratios_pot + ratios_stk)
    ig2 = (counters["instrument_blocked"] / max(1, counters["attempted"])
           <= s3.ANOMALY_MAX_RATE)
    nonempty = sum(1 for v in (summary["strata"].get("pot") or {}).values()
                   if v["n"] > 0)
    ig3 = nonempty >= 3 and len(set(pot)) > 1
    ig4 = all(r.get("raw_pot_before_chips") is not None
              and r.get("state_pot_bb") is not None
              and r.get("ratio_pot_input_over_raw") is not None
              for r in rows)
    ig5 = all(int(r.get("n_runouts") or 0) > 0 for r in rows)
    ig7 = summary["schema_violations"] == 0
    ig8 = bool(summary["strata_completeness"]["complete"]) and \
        all(v == 0 for v in summary["strata_completeness"]["missing"].values())
    ig9 = (summary["pi_points"] == len(rows)
           and summary["unavailable"] == counters["unavailable"]
           and summary["legality"]["illegal"] == 0
           and summary["attempted"] == counters["attempted"])
    ig10 = all(len(r) >= len(s3.POINT_SCHEMA) for r in rows)
    gates = {"IG1_raw_ratio": ig1, "IG2_anomaly_rate": ig2,
             "IG3_distribution": ig3, "IG4_audit_log": ig4,
             "IG5_no_approximation": ig5, "IG6_no_dry_run": True,
             "IG7_per_point_schema": ig7, "IG8_strata_completeness": ig8,
             "IG9_self_consistency": ig9, "IG10_traceability": ig10}
    return {"gates": gates, "all_pass": all(gates.values()),
            "pot_percentiles_bb": pct,
            "ratio_pot_min": min(ratios_pot) if ratios_pot else None,
            "ratio_pot_max": max(ratios_pot) if ratios_pot else None,
            "ratio_stack_min": min(ratios_stk) if ratios_stk else None,
            "ratio_stack_max": max(ratios_stk) if ratios_stk else None,
            "anomaly_rate": round(counters["instrument_blocked"]
                                  / max(1, counters["attempted"]), 6),
            "wall_s": round(wall_s, 4)}


# ── 运行器 ───────────────────────────────────────────────────────────
def run_v2(hands: list, arm: str, mode: str, *, max_points: int,
           allow_run: bool = False, write_per_point: bool = True,
           attempt_window: tuple | None = None,
           include_rows: bool = False) -> dict:
    """运行一个 (arm, mode)。`include_rows=True` 时把 per-point 行随返回值
    一起给出（供测试做内存内逐字段比较，**不写任何文件**）。"""
    if not allow_run:
        raise V2GuardError("run_v2 已禁用；执行需显式 allow_run=True（本阶段授权）")
    if arm not in ("U1", "R2.2"):
        raise ValueError(f"未知 arm: {arm!r}")
    if mode not in V2_MODES:
        raise ValueError(f"未知 mode: {mode!r}")
    man = verify_v2_manifest()
    if not man.get("present") or not man.get("clean"):
        raise V2GuardError(f"v2 清单未封存或已漂移: {man}")
    before = sealed_s3_clean()
    if not all(before.values()):
        raise V2GuardError(f"旧封存清单不干净: {before}")

    out_path = PER_POINT[(mode, arm)]
    if write_per_point:
        _assert_not_sealed(out_path)

    scope = _equiv_hands(hands)
    t_idx = time.perf_counter()
    idx = s3.build_indexes(scope, exclude_players=set())
    agg = s3.build_aggression_index_excluding(scope, self_exclude=set())
    index_build_s = time.perf_counter() - t_idx
    cfg = _load_config()

    workers = V2_WORKERS[mode]
    optimized = mode != "baseline1"
    t0 = time.perf_counter()
    if workers > 1:
        res = pp.profile_arm_pool(scope, arm, max_points=max_points,
                                  workers=workers, optimized=optimized,
                                  idx=idx, agg=agg, cfg=cfg,
                                  attempt_window=attempt_window)
    else:
        res = pp.profile_arm(scope, arm, max_points=max_points,
                             optimized=optimized, idx=idx, agg=agg, cfg=cfg,
                             attempt_window=attempt_window)
    wall = time.perf_counter() - t0

    rows = res["rows"]
    counters = res["counters"]
    reg = json.loads(s3.REGISTRY.read_text(encoding="utf-8"))
    summary = s3.recompute_summary(rows, arm=arm, reg=reg,
                                   attempted=counters["attempted"],
                                   unavailable=counters["unavailable"])
    gates = assemble_gates(rows, summary, counters, wall)

    persist_s = 0.0
    if write_per_point:
        t1 = time.perf_counter()
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        out_path.write_text(blob, encoding="utf-8")
        persist_s = time.perf_counter() - t1

    after = sealed_s3_clean()
    if not all(after.values()):
        raise V2GuardError(f"运行后旧封存清单不干净: {after}")

    out = {"runner_version": V2_VERSION, "arm": arm, "mode": mode,
           "workers": workers, "optimized": optimized,
           "equiv_set": "hand_index < %d" % V2_HAND_MAX_INDEX,
           "points": len(rows), "wall_s": round(wall, 4),
           "index_build_s": round(index_build_s, 4),
           "persist_s": round(persist_s, 4),
           "wall_with_index_s": round(wall + index_build_s, 4),
           "per_point_s": round(wall / len(rows), 6) if rows else None,
           "point_times_s": res.get("point_times_s") or [],
           "percentiles_s": pp.percentiles(res.get("point_times_s") or []),
           "timers": res["timers"], "counters": counters,
           "runout_stats": res["runout_stats"],
           "summary": summary, "gates": gates,
           "cache_stats": res.get("cache_stats"),
           "worker_cpu_s_sum": res.get("worker_cpu_s_sum"),
           "worker_efficiency": res.get("worker_efficiency"),
           "sealed_s3_clean_before": before, "sealed_s3_clean_after": after,
           "per_point_file": (str(out_path.relative_to(ROOT))
                              if write_per_point else None)}
    if include_rows:
        out["rows"] = rows
    return out


def load_rows(mode: str, arm: str) -> list:
    return [json.loads(l) for l in
            PER_POINT[(mode, arm)].read_text(encoding="utf-8").splitlines()
            if l.strip()]


def compare_rows(a: list, b: list, keys=None) -> dict:
    """逐点逐字段比较（0 容差）；按提交序对齐。"""
    if len(a) != len(b):
        return {"aligned": False, "n_a": len(a), "n_b": len(b)}
    if keys is None:
        keys = sorted(set().union(*[set(r) for r in a + b]) - {"mode"})
    diff_fields: dict = {}
    bad = 0
    for ra, rb in zip(a, b):
        d = [k for k in keys if ra.get(k) != rb.get(k)]
        if d:
            bad += 1
            for k in d:
                diff_fields[k] = diff_fields.get(k, 0) + 1
    return {"aligned": True, "points": len(a), "mismatched_points": bad,
            "diff_fields": diff_fields, "fields_compared": len(keys),
            "zero_tolerance": True}
