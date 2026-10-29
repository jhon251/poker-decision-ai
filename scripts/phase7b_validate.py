"""Phase 7B：Exploit Advisor Evidence Validation（只验证 P1.5）。

严格按 docs/PHASE7B_STEP0_VALIDATION_AUDIT.md（已通过）+ 预注册 spec 快照执行。
管线（只读冻结产物，只调用冻结 7A decide()）：
  history(N) → 7A decide → ExploitDecision → future(50 手) 评分
    → 方向持久性（玩家级 bootstrap）/ confidence 校准（C1 修正口径）/ 安全契约扫描

红线（违规 = AssertionError 硬失败，绝不自动降级/修复）：
  - train/test 互斥；prior train-only（hb_fit fit_on=task_a_train + 方案 B 测试玩家跳过）
  - history 末 ts ≤ as_of ≤ future 首 ts；future 只作评分，不进 profile/prior/阈值
  - 无任何拟合：阈值/bucket/bootstrap 全部冻结常量（post-hoc tuning 被 config echo 封死）
  - outcome 字段不进 advisor input（7A 黑白名单硬断言；7B context 只含情境字段）
  - cluster=disabled、time_decay=none（7A config 硬断言）
  - strength_applied ≡ 0、adjustment_applied ≡ false、ev_claim ≡ false（全量扫描）
  - verdict 词汇仅 {advisor_evidence_validated, advisor_utility_unproven,
    insufficient_evidence}；禁止 exploit_utility_proven / EV / ROI / profitable

输出: data/uoft/models/phase7/phase7b/validation.json（+ DEGRADE_LOG.json 若失败）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase5b_math import beta_binomial_pmf, bootstrap_pct  # noqa: E402
from phase6f_common import (  # noqa: E402
    feature_suff, fit_fold_to_3bet_prior, load_test_players)
from phase7a_decision import DECISION_FEATURES, decide  # noqa: E402

# ---- 冻结阈值（Audit §7，全部预注册，禁止修改）----
FROZEN = {
    "horizons": [50, 100, 200],
    "min_future": 50,
    "tail_sig": 0.975,
    "p7_effect_size": 0.02,
    "persistence_lower": 0.5,
    "utility_gate_min": 3,
    "feature_universe": 5,
    "min_units": 30,
    "min_evaluable_buckets": 2,
    "min_opp": {"default": 10, "cbet": 2},
    "conf_strong": 0.95,
    "bootstrap": {"unit": "player", "paired": True, "n": 1000, "seed": 42},
    "buckets": [[0.5, 0.7], [0.7, 0.85], [0.85, 0.95], [0.95, 1.0]],
}
BUCKET_NAMES = ("b1", "b2", "b3", "b4")
N_TRAIN, N_TEST = 3999, 1001

VERDICT_ALLOWED = ("advisor_evidence_validated", "advisor_utility_unproven",
                   "insufficient_evidence")


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _ci(records: list) -> tuple[float, float]:
    """玩家级配对 bootstrap 95% CI（冻结 bootstrap_pct, n=1000, seed=42）。"""
    lo, hi, _ = bootstrap_pct(lambda xs: sum(xs) / len(xs), records,
                              n_resample=FROZEN["bootstrap"]["n"],
                              seed=FROZEN["bootstrap"]["seed"])
    return round(lo, 6), round(hi, 6)


def _suff(hands: list, feature: str) -> tuple[int, int]:
    """(S, n) 充分统计（冻结 6F 口径 + cbet n_eff=max(S,n)，6.5 冻结修正）。"""
    S, n = feature_suff(hands, feature)
    if feature == "cbet" and S > n:
        n = S
    return S, n


def _bucket_of(c: float) -> int:
    for i, (lo, hi) in enumerate(FROZEN["buckets"]):
        if lo <= c <= hi:
            return i
    raise AssertionError(f"confidence {c} 不在冻结 bucket 范围")


def _predicted_persistence(direction: str, n_f: int, a: float, b: float,
                           mu: float) -> float:
    """C1 修正口径：history-only advisor 预测持久率
    p̂ = Σ beta_binomial_pmf(k, n_f, a', b')（k/n_f 与 μ 同侧）；
    n_f 是机会计数而非 outcome（条件化语义同冻结 predict_probe）。
    """
    tot = 0.0
    for k in range(n_f + 1):
        r = k / n_f
        if direction == "over" and r > mu:
            tot += beta_binomial_pmf(k, n_f, a, b)
        elif direction == "under" and r < mu:
            tot += beta_binomial_pmf(k, n_f, a, b)
    return tot


def _spot_context(feature: str) -> dict:
    """决策点之前的情境字段（Audit §8-5：outcome 字段绝不进入）。"""
    if feature == "fold_to_3bet":
        return {"faced_3bet": True}
    if feature == "cbet":
        return {"cbet_opportunity": True}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--splits", default="data/uoft/splits.json")
    ap.add_argument("--hb", default="data/uoft/models/mvp/hb_fit.json")
    ap.add_argument("--out", default="data/uoft/models/phase7/phase7b")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 泄漏硬断言 1：train/test 互斥（冻结 splits）----
    train, test = load_test_players(args.splits)
    assert len(train) == N_TRAIN and len(test) == N_TEST, "splits 规模与冻结不符"
    assert not (train & test), "train/test 玩家重叠（泄漏）"

    # ---- 泄漏硬断言 5/6：prior train-only provenance ----
    hb = json.load(open(args.hb, encoding="utf-8"))
    assert hb["config"]["fit_on"] == "task_a_train_players", \
        "hb_fit 必须为 task_a_train_players（train-only）"
    priors = {}
    for f in ("vpip", "pfr", "three_bet", "cbet"):
        fit = hb["probes"][f]
        a, b = fit["a"], fit["b"]
        # mu 由冻结 a,b 精确重算 a/(a+b)：hb_fit 的 a/b/mu 为 4/6 位小数舍入值，
        # 7A 冻结的内部一致性硬断言（|mu−a/(a+b)|≤1e-9）要求参数自洽；
        # 重算 mu 与冻结 6 位 mu 的偏差 ≤1.5e-5（纯舍入差，≪ δ=0.02 且 ≪ rate 粒度
        # 1/50，方向判定无影响）。参考来源仍是 train-only prior，未改任何阈值。
        priors[f] = {"a": a, "b": b, "mu": a / (a + b), "provenance": "hb_fit"}
    fb = fit_fold_to_3bet_prior(args.stats, train, test)
    assert fb["n_test_players_skipped"] == N_TEST, \
        f"fold_to_3bet 方案 B 应跳过全部 {N_TEST} 个测试玩家，" \
        f"实际 {fb['n_test_players_skipped']}（泄漏）"
    priors["fold_to_3bet"] = {"a": fb["a"], "b": fb["b"],
                              "mu": fb["a"] / (fb["a"] + fb["b"]),
                              "provenance": "scheme_b"}
    pop_mu = {f: p["mu"] for f, p in priors.items()}

    # ---- 测试玩家 ring 序列（ts 升序）----
    seq: dict[str, list] = {}
    for line in open(args.events, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("table_type") == "hu":
            continue
        pid = ev["player_id"]
        if pid in test:
            seq.setdefault(pid, []).append(ev)
    for pid in seq:
        seq[pid].sort(key=lambda x: x["ts"] or "")
    assert set(seq) <= test, "出现非测试玩家（泄漏）"

    decisions: list[dict] = []
    safety = {"scanned": 0, "violations": []}

    for N in FROZEN["horizons"]:
        for pid, hands in seq.items():
            if len(hands) < N + FROZEN["min_future"]:
                continue
            hist, fut = hands[:N], hands[N:N + FROZEN["min_future"]]
            # ---- 泄漏硬断言 2/3：边界 ----
            assert hist[-1]["ts"] <= fut[0]["ts"], f"{pid} history/future 重叠"
            as_of = hist[-1]["ts"]
            for f in DECISION_FEATURES:
                inp = {
                    "opponent_id": pid, "as_of": as_of, "history": hist,
                    "priors": priors, "population_mu": pop_mu,
                    "spot": {"feature": f, "trigger": True,
                             "context": _spot_context(f)},
                    "baseline_decision": {"action": "check", "size": None},
                    "config": {"p7_effect_size": FROZEN["p7_effect_size"],
                               "strength_arm": "k=0",
                               "cluster": "disabled", "time_decay": "none"},
                }
                out = decide(inp)   # 7A 内部已硬断言 as_of/ts/prior/黑白名单

                # ---- Safety contract 全量扫描（违规 = AssertionError，不降级）----
                safety["scanned"] += 1
                try:
                    assert out["adjustment_applied"] is False
                    assert out["meta"]["ev_claim"] is False
                    assert out["meta"]["advisor_mode"] is True
                    assert out["meta"]["cluster_probs"] is None
                    for c in out["candidates"]:
                        assert c["strength_applied"] == 0.0
                        if out["exploit_status"] == "none":
                            assert c["confidence"] is None
                            assert c["strength_recommended"] == 0.0
                    if out["tier"] in ("unknown", "T4"):
                        assert out["candidates"] == []   # population_only：无个体行
                    if out["fallback_level"] == "population_prior" \
                            and out["candidates"]:
                        # 特征级 zero_opportunity：仅注释候选，方向 none 且无置信
                        assert all(c["direction"] == "none"
                                   and c["confidence"] is None
                                   for c in out["candidates"])
                except AssertionError as e:
                    safety["violations"].append(
                        {"player": pid, "feature": f, "horizon": N,
                         "reason": str(e)})
                    raise AssertionError(
                        f"SAFETY CONTRACT 违规: {pid}/{f}/N={N}: {e}")

                if out["exploit_status"] not in ("strong", "weak"):
                    continue
                cand = out["candidates"][0]
                d, c = cand["direction"], cand["confidence"]
                Sh, nh = _suff(hist, f)
                Sf, nf = _suff(fut, f)
                min_opp = FROZEN["min_opp"]["cbet" if f == "cbet" else "default"]
                if nf < min_opp:
                    continue   # 未来机会不足：不参与指标（预注册），原因记录
                a, b, mu = priors[f]["a"], priors[f]["b"], priors[f]["mu"]
                ap, bp = a + Sh, b + nh - Sh
                rate_f = Sf / nf
                d_f = ("over" if rate_f > mu else
                       ("under" if rate_f < mu else "none"))
                match = 1 if d_f == d else 0
                p_hat = _predicted_persistence(d, nf, ap, bp, mu)
                decisions.append({
                    "player": pid, "feature": f, "horizon": N,
                    "direction": d, "confidence": c,
                    "n_history_opp": nh, "n_future_opp": nf,
                    "match": match, "p_hat": round(p_hat, 6)})

    # ---- 指标（主判定 N=200；50/100 为学习曲线）----
    per_horizon: dict = {}
    for N in FROZEN["horizons"]:
        decs = [d for d in decisions if d["horizon"] == N]
        per_feature: dict = {}
        for f in DECISION_FEATURES:
            recs = [d for d in decs if d["feature"] == f]
            players = {d["player"] for d in recs}
            assert len(players) == len(recs), "每玩家每特征至多 1 决策"
            buckets = [0, 0, 0, 0]
            for d in recs:
                buckets[_bucket_of(d["confidence"])] += 1
            if not recs:
                per_feature[f] = {
                    "n_full_pass": 0, "persistence_rate": None,
                    "persistence_ci95": None, "status": "insufficient_evidence",
                    "reason": "无 full-pass 决策（冻结 gate 未通过或未来机会不足）",
                    "confidence_buckets": dict(zip(BUCKET_NAMES, buckets)),
                    "n_future_opp_median": 0}
                continue
            matches = [d["match"] for d in recs]
            lo, hi = _ci(matches)
            n = len(recs)
            status = ("insufficient_evidence" if n < FROZEN["min_units"] else
                      ("validated" if lo > FROZEN["persistence_lower"]
                       else "unproven"))
            med_nf = sorted(d["n_future_opp"] for d in recs)[n // 2]
            per_feature[f] = {
                "n_full_pass": n,
                "persistence_rate": round(_mean(matches), 6),
                "persistence_ci95": [lo, hi],
                "lower_bound": lo,
                "pass_criterion": f"lower_bound > {FROZEN['persistence_lower']}",
                "passed": lo > FROZEN["persistence_lower"],
                "status": status,
                "reason": ("" if status != "insufficient_evidence"
                           else f"n_full_pass={n} < {FROZEN['min_units']}"),
                "confidence_buckets": dict(zip(BUCKET_NAMES, buckets)),
                "n_future_opp_median": med_nf}
        per_horizon[str(N)] = {"per_feature": per_feature}

    # ---- N=200 主判定 ----
    decs200 = [d for d in decisions if d["horizon"] == 200]
    pf200 = per_horizon["200"]["per_feature"]

    # pooled G-PERSIST（玩家先聚合，再 bootstrap）
    by_player: dict[str, list] = {}
    for d in decs200:
        by_player.setdefault(d["player"], []).append(d["match"])
    pooled_vals = [_mean(v) for v in by_player.values()]
    plo, phi = _ci(pooled_vals) if pooled_vals else (float("nan"), float("nan"))
    g_persist = {
        "n_players": len(by_player), "pooled_persistence": round(_mean(pooled_vals), 6),
        "ci95": [plo, phi], "lower_bound": plo,
        "pass": (len(by_player) >= FROZEN["min_units"]
                 and plo > FROZEN["persistence_lower"])}

    # G-CALIB（C1 修正口径：经验持久率 vs history-only 预测 p̂，玩家级配对）
    calib_buckets: dict = {}
    for bi, name in enumerate(BUCKET_NAMES):
        lo_b, hi_b = FROZEN["buckets"][bi]
        e_map: dict[str, list] = {}
        q_map: dict[str, list] = {}
        for d in decs200:
            if not (lo_b <= d["confidence"] <= hi_b):
                continue
            pid = d["player"]
            e_map.setdefault(pid, []).append(d["match"])
            q_map.setdefault(pid, []).append(d["p_hat"])
        n_players = len(e_map)
        if n_players < FROZEN["min_units"]:
            calib_buckets[name] = {
                "range": [lo_b, hi_b], "evaluable": False, "low_power": True,
                "n_players": n_players, "n_decisions": sum(len(v) for v in e_map.values()),
                "mean_empirical": None, "mean_predicted": None,
                "diff_ci95": None, "contains_zero": None, "passed": None,
                "reason": f"玩家数 {n_players} < {FROZEN['min_units']}（low_power，"
                          f"排除出通过判定并如实报告）"}
            continue
        pairs = [( _mean(e_map[p]), _mean(q_map[p])) for p in e_map]
        diffs = [e - q for e, q in pairs]
        dlo, dhi = _ci(diffs)
        contains_zero = dlo <= 0.0 <= dhi
        calib_buckets[name] = {
            "range": [lo_b, hi_b], "evaluable": True, "low_power": False,
            "n_players": n_players, "n_decisions": sum(len(v) for v in e_map.values()),
            "mean_empirical": round(_mean([p[0] for p in pairs]), 6),
            "mean_predicted": round(_mean([p[1] for p in pairs]), 6),
            "diff_ci95": [dlo, dhi], "contains_zero": contains_zero,
            "passed": contains_zero, "reason": ""}
    evaluable_buckets = [b for b in calib_buckets.values() if b["evaluable"]]
    g_calib_insufficient = len(evaluable_buckets) < FROZEN["min_evaluable_buckets"]
    g_calib = {
        "buckets": calib_buckets,
        "evaluable_buckets": [n for n, b in calib_buckets.items() if b["evaluable"]],
        "low_power_buckets": [n for n, b in calib_buckets.items() if b["low_power"]],
        "insufficient": g_calib_insufficient,
        "pass": (not g_calib_insufficient
                 and all(b["passed"] for b in evaluable_buckets)),
        "criterion": "每可评估 bucket 的配对差(经验率−p̂)玩家级 bootstrap 95% CI 含 0；"
                     "可评估 bucket ≥ 2（<30 玩家 → low_power，排除并报告）"}

    ev_features = [f for f in DECISION_FEATURES
                   if pf200[f]["n_full_pass"] >= FROZEN["min_units"]]
    insuf_features = [f for f in DECISION_FEATURES if f not in ev_features]
    passed_features = [f for f in ev_features if pf200[f]["passed"] is True]
    failed_features = [f for f in ev_features if pf200[f]["passed"] is False]

    # ---- 冻结 verdict 逻辑（Audit §7/§8；绝不调阈值）----
    if len(ev_features) < 3:
        verdict = "insufficient_evidence"
    elif g_calib_insufficient:
        verdict = "insufficient_evidence"
    elif g_persist["n_players"] < FROZEN["min_units"]:
        verdict = "insufficient_evidence"
    elif (g_persist["pass"] and g_calib["pass"]
          and len(passed_features) >= FROZEN["utility_gate_min"]):
        verdict = "advisor_evidence_validated"
    else:
        verdict = "advisor_utility_unproven"
    assert verdict in VERDICT_ALLOWED

    results = {
        "phase": "7B", "proposition": "P1.5（advisor evidence 安全/方向持久/校准）",
        "config": {"frozen_thresholds": FROZEN,
                   "decision_features": list(DECISION_FEATURES),
                   "prior_provenance": ("train-only: hb_fit.json(fit_on="
                                        "task_a_train_players) + fold_to_3bet "
                                        "方案 B(train suff stats, test 跳过 "
                                        f"{fb['n_test_players_skipped']})"),
                   "no_fitting": ("7B 零拟合：阈值/bucket/bootstrap 全部冻结常量；"
                                  "p̂ 为每决策 history-only 确定性计算"),
                   "mu_recomputed_note": ("hb_fit 的 a/b/mu 为冻结舍入值（4/6 位小数）；"
                                          "为满足 7A 冻结的一致性硬断言，7B 以冻结 a,b "
                                          "精确重算 mu=a/(a+b)（与冻结 6 位 mu 偏差 "
                                          "≤1.5e-5，纯舍入差，方向判定无影响）"),
                   "cbet_n_eff_note": "cbet S>n 行以 n_eff=max(S,n) 修正（6.5 冻结）",
                   "cluster": "disabled（6D 降级，cluster_probs 恒 null）",
                   "time_decay": "none（time_sensitive 仅状态标记）",
                   "verdict_vocabulary": {"allowed": list(VERDICT_ALLOWED),
                                          "prohibited": ["exploit_utility_proven",
                                                         "EV proven", "ROI proven",
                                                         "profitable", "profitability",
                                                         "winnings improvement"]}},
        "leakage_assertions": [
            "train/test 互斥（3999/1001，硬断言）",
            "hb_fit fit_on=task_a_train_players（文件级断言）",
            "fold_to_3bet 方案 B 测试玩家跳过=1001（硬断言）",
            "history 末 ts ≤ as_of ≤ future 首 ts（7A+7B 双重硬断言）",
            "future 不进入 profile（7A history-only 构造；7B 无 6G 导入）",
            "outcome 字段不进入 advisor input（7A 黑白名单硬断言）",
            "无 hero/equity 数据路径（事件表观察者视角；源码红线测试）",
            "cluster=disabled / time_decay=none（7A config 硬断言）",
            "无阈值/bucket 拟合（零拟合 + config echo 冻结）"],
        "safety_contract": {"scanned_decisions": safety["scanned"],
                            "violations": len(safety["violations"]),
                            "all_clean": len(safety["violations"]) == 0,
                            "invariants": [
                                "strength_applied == 0",
                                "adjustment_applied == false",
                                "ev_claim == false", "advisor_mode == true",
                                "cluster_probs == null",
                                "gate fail → confidence == null",
                                "zero opportunity → 不消费",
                                "non-consumable → 不消费",
                                "unknown/T4 → population-only"]},
        "per_horizon": per_horizon,
        "calibration": g_calib,
        "pooled_persistence": g_persist,
        "main_judgment": {
            "horizon": 200,
            "evaluable_features": ev_features,
            "insufficient_features": insuf_features,
            "features_passed": passed_features,
            "features_failed": failed_features,
            "utility_gate": f">= {FROZEN['utility_gate_min']}/{len(ev_features)} "
                           f"可评估特征（insufficient 排除出分母，逐项报告）",
            "g_persist_pass": g_persist["pass"],
            "g_calib_pass": g_calib["pass"],
            "g_calib_insufficient": g_calib_insufficient,
            "verdict": verdict,
            "conclusion": (
                "advisor_evidence_validated：安全性 100%、方向持久性 CI 下界>0.5、"
                "置信度校准通过（C1 口径），且 ≥3 个可评估特征通过——证据链在留出"
                "测试集上成立（P1.5）。注意：本结论不涉及任何 EV/收益主张（P2 专属）。"
                if verdict == "advisor_evidence_validated" else
                ("insufficient_evidence：可评估单元不足（<3 可评估特征 / pooled<30 "
                 "/ 可评估 bucket<2），无法判定——如实报告，不把样本不足写成通过或失败。"
                 if verdict == "insufficient_evidence" else
                 "advisor_utility_unproven：证据链未达预注册门槛（方向持久/校准/≥3 特征"
                 " 之一失败），如实报告，不调阈值。strength 保持 0，无 EV 主张。"))},
    }

    (out_dir / "validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if verdict != "advisor_evidence_validated":
        entries = [{
            "component": "phase7b_advisor_evidence",
            "path": "advisor_utility_unproven" if verdict == "advisor_utility_unproven"
                    else "insufficient_evidence",
            "reason": results["main_judgment"]["conclusion"],
            "evidence": {
                "verdict": verdict,
                "evaluable_features": ev_features,
                "insufficient_features": insuf_features,
                "features_passed": passed_features,
                "features_failed": failed_features,
                "g_persist": g_persist, "g_calib_pass": g_calib["pass"],
                "g_calib_insufficient": g_calib_insufficient,
            }}]
        (out_dir / "DEGRADE_LOG.json").write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print(json.dumps(results["main_judgment"], ensure_ascii=False))
    for N in FROZEN["horizons"]:
        d = per_horizon[str(N)]["per_feature"]
        line = {f: (d[f]["n_full_pass"], d[f]["status"], d[f]["persistence_rate"])
                for f in DECISION_FEATURES}
        print(f"N={N}: {json.dumps(line, ensure_ascii=False)}")
    print(f"calibration: pass={g_calib['pass']} insufficient={g_calib_insufficient} "
          f"evaluable_buckets={g_calib['evaluable_buckets']}")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
