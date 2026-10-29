"""S‴′ 前的 PI-EV 性能审计：**语义等价回归测试**。

本文件只验证「性能优化候选 == 现行实现」，不改变任何验收语义、不写封存路径：

* instrumented profiling 副本 == sealed `PiEvReference.reference`；
* 缓存版 PI 参照 == sealed（逐字段）；
* `compute_ev_cached` == sealed `compute_ev`（dataclass 深比较）；
* `optimized_evaluate_v3` == sealed `evaluate_v3`（整字典深比较）；
* `child_equity` / `FoldEstimator._combo_equities` 记忆化等值；
* 逐点结果与执行顺序无关（进程池并行合法性的前提）；
* 进程池 worker 与单进程结果一致；
* 封存清单仍未漂移（seal 完整性守卫）。
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from poker_decision_ai.pluribus_decision_adapter import (
    iter_pluribus_rows, parse_hand_row)
from poker_decision_ai.pluribus_validation import _load_config
from poker_decision_ai.research_unfreeze import perf_optimize as po
from poker_decision_ai.research_unfreeze import perf_profile as pp
from poker_decision_ai.research_unfreeze import s3_validation as s3
from poker_decision_ai.research_unfreeze.s3_validation import (
    build_aggression_index_excluding, build_indexes)

SCOPE_HANDS = 140          # 小切片：够产生真实点位，又让测试保持秒级
N_POINTS = 3


def _scope():
    hands = [p for row in itertools.islice(iter_pluribus_rows(), SCOPE_HANDS)
             if (p := parse_hand_row(row))]
    assert hands, "corpus unavailable"
    return hands[:SCOPE_HANDS]


@pytest.fixture(scope="module")
def ctx():
    hands = _scope()
    idx = build_indexes(hands, exclude_players=set())
    agg = build_aggression_index_excluding(hands, self_exclude=set())
    cfg = _load_config()
    u1 = pp.run_points(hands, "U1", N_POINTS, idx=idx, agg=agg, cfg=cfg,
                       fixture_cap=N_POINTS, case_cap=N_POINTS)
    r22 = pp.run_points(hands, "R2.2", N_POINTS, idx=idx, agg=agg, cfg=cfg,
                        fixture_cap=N_POINTS, case_cap=N_POINTS)
    return {"hands": hands, "idx": idx, "agg": agg, "cfg": cfg,
            "u1": u1, "r22": r22}


def test_profiler_is_read_only_by_construction():
    """perf 模块不得调用 `run_s3`、不得引用封存输出路径（AST 静态检查）。"""
    import ast
    sealed_attrs = {"PER_POINT", "RESULT", "MANIFEST_U1", "MANIFEST_R22",
                    "REGISTRY"}
    for mod in (pp, po):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        names, attrs, calls = set(), set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                attrs.add(node.attr)
            elif isinstance(node, ast.Call):
                fn = node.func
                calls.add(fn.id if isinstance(fn, ast.Name)
                          else getattr(fn, "attr", ""))
        assert "run_s3" not in names and "run_s3" not in calls, mod.__name__
        assert not (sealed_attrs & attrs), \
            f"{mod.__name__} touches sealed attrs: {sealed_attrs & attrs}"


PERF_TOOLCHAIN = ("perf_profile.py", "perf_optimize.py", "perf_runner_v2.py")
# 显式声明的**非生产**研究 runner（各自 manifest 内 `production_wiring="NOT WIRED"`）；
# 该名单必须逐一列举，新增即需有意修改本行。
NON_PRODUCTION = PERF_TOOLCHAIN + ("s3prime_runner.py",)


def _imports_perf_toolchain(path):
    """AST 判定：该模块是否**真的导入**了 perf 工具链（文档串/注释不算）。"""
    import ast as _ast
    tree = _ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    needles = {n[:-3] for n in PERF_TOOLCHAIN}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            if any(a.name.split(".")[-1] in needles for a in node.names):
                return True
        elif isinstance(node, _ast.ImportFrom):
            if (node.module or "").split(".")[-1] in needles:
                return True
            if any(a.name in needles for a in node.names):
                return True
    return False


def test_perf_modules_not_imported_by_production():
    """性能工具链（perf_profile/perf_optimize/perf_runner_v2）不得被生产模块**导入**。

    工具链内部互相导入是允许的（它们**本身**就是非生产代码：v2 清单里
    `production_wiring = "NOT WIRED"`）。判定用 AST 解析真实 import：
    其它模块在文档串里**引用**工具链名字（例如审计报告源码注释里点名缺陷
    调用点）不构成生产接线，不应判失败。
    """
    pkg = Path(pp.__file__).resolve().parents[1]      # poker_decision_ai/
    offenders = [str(p.relative_to(pkg)) for p in pkg.rglob("*.py")
                 if p.name not in NON_PRODUCTION and _imports_perf_toolchain(p)]
    assert offenders == [], f"perf modules imported by: {offenders}"
    # 反向守卫：只有已声明的非生产模块可以导入工具链
    assert not any(_imports_perf_toolchain(p) for p in pkg.rglob("*.py")
                   if p.name not in NON_PRODUCTION)


def test_sealed_manifests_still_clean():
    """S‴ 封存清单（9 文件哈希）在性能阶段后仍无漂移。"""
    for manifest in (s3.MANIFEST_U1, s3.MANIFEST_R22):
        rep = s3.verify_s3_manifest(manifest)
        assert rep["present"] and rep["clean"] and rep["checked"] == 9
        assert rep["drift"] == {} and rep["split_drift"] is False


def test_instrumented_pi_reference_matches_sealed(ctx):
    for arm in ("u1", "r22"):
        fixtures = ctx[arm]["fixtures"]
        assert fixtures, "no fixtures collected"
        assert pp.equivalence_check(fixtures)["mismatches"] == 0


def test_optimized_pi_reference_matches_sealed(ctx):
    for arm in ("u1", "r22"):
        rep = po.prove_pi_equivalence(ctx[arm]["fixtures"])
        assert rep["checked"] > 0 and rep["mismatches"] == 0


def test_cached_compute_ev_matches_sealed(ctx):
    rep = po.prove_compute_ev_equivalence(ctx["r22"]["cases"])
    assert rep["checked"] > 0 and rep["mismatches"] == 0


def test_optimized_evaluate_v3_matches_sealed(ctx):
    rep = po.prove_evaluate_v3_equivalence(ctx["r22"]["cases"])
    assert rep["checked"] > 0 and rep["mismatches"] == 0


def test_child_equity_memo_returns_identical_value(ctx):
    case = ctx["r22"]["cases"][0]
    dist = case["rs"]["villains"][case["vid"]]["distribution"]
    hero = [p for p in case["state"]["players"]
            if isinstance(p, dict) and p.get("is_hero")][0]["hole_cards"]
    board = list(case["state"]["board"])
    po.reset_stats()
    a = po.child_equity_cached(hero, board, dist, 300, 42)
    b = po.child_equity_cached(hero, board, dist, 300, 42)
    assert a == b
    assert po.STATS["child_equity"]["hits"] >= 1


def test_combo_equities_memo_is_value_identical(ctx):
    """记忆化 `_combo_equities` 与 sealed 方法返回同一对象内容。"""
    from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
    case = ctx["r22"]["cases"][0]
    dist = case["rs"]["villains"][case["vid"]]["distribution"]
    hero = [p for p in case["state"]["players"]
            if isinstance(p, dict) and p.get("is_hero")][0]["hole_cards"]
    board = list(case["state"]["board"])

    def _mk(pot):
        return FoldEstimator(pot_bb=pot, facing="bet",
                             street=str(case["state"].get("street") or "flop"),
                             villain_dist=dict(dist), hero_cards=list(hero),
                             board=list(board), seed=42)

    baseline = list(FoldEstimator._combo_equities(_mk(10.0)) or [])
    po._COMBO_MEMO.clear()
    orig = FoldEstimator._combo_equities
    FoldEstimator._combo_equities = po.combo_equities_cached
    try:
        first = list(_mk(10.0)._combo_equities() or [])
        second = list(_mk(99.0)._combo_equities() or [])
    finally:
        FoldEstimator._combo_equities = orig
    assert first == baseline == second
    assert po.STATS["combo_equities"]["hits"] >= 1


@pytest.mark.parametrize("arm", ["U1", "R2.2"])
def test_point_results_are_order_independent(arm, ctx):
    """逐点结果与执行顺序无关 → 进程池并行的合法性前提。"""
    rep = pp.prove_order_independence(ctx["hands"], arm, 4)
    assert rep["common_hand_street_keys"] > 0
    assert rep["order_mismatches"] == 0
    assert rep["repeat_run_identical"] is True


def test_pool_worker_matches_single_process(ctx):
    """同一分片：worker 函数（进程内直接调用）与单进程逐行一致。"""
    a = pp.run_points(ctx["hands"], "U1", 2, idx=ctx["idx"], agg=ctx["agg"],
                      cfg=ctx["cfg"])
    payload = {"hands": ctx["hands"], "arm": "U1", "cap": 2,
               "idx": ctx["idx"], "agg": ctx["agg"], "cfg": ctx["cfg"]}
    b = pp._pool_worker(payload)
    assert b["rows"] == a["rows"]


def test_process_pool_end_to_end_matches_single_process(ctx):
    """真进程池（2 workers）与单进程结果一致。"""
    a = pp.run_points(ctx["hands"], "R2.2", 2, idx=ctx["idx"], agg=ctx["agg"],
                      cfg=ctx["cfg"])
    step = max(1, len(ctx["hands"]) // 2)
    chunks = [ctx["hands"][:step], ctx["hands"][step:]]
    payloads = [{"hands": c, "arm": "R2.2", "cap": 2, "idx": ctx["idx"],
                 "agg": ctx["agg"], "cfg": ctx["cfg"]} for c in chunks]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(pp._pool_worker, payloads):
            rows.extend(r["rows"])
    assert rows[:2] == a["rows"], "pool rows diverged from single-process rows"
    json.dumps(rows)          # 结果可序列化（可持久化 → 可复现）


def test_enumeration_precision_unchanged(ctx):
    """优化只复用结果、不降低精度：枚举数与非缓存实现一致。"""
    fixtures = ctx["u1"]["fixtures"]
    from poker_decision_ai.research_unfreeze.unfreeze_core import PiEvReference
    for f in fixtures:
        sealed = PiEvReference().reference(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"],
            to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        opt = po.pi_reference_optimized(
            hero_cards=f["hole"], villain_hands=f["vhands"], board=f["board"],
            pot_bb=f["inputs"]["pot_bb"],
            to_call_bb=f["inputs"]["to_call_bb"],
            hero_stack_bb=f["inputs"]["hero_stack_bb"], candidates=f["cands"])
        assert opt["n_runouts"] == sealed["n_runouts"]
        assert opt["pi_equity"] == sealed["pi_equity"]
