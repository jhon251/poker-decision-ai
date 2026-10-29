"""Phase 5B Step 6：生成 MVP_STRUCTURE_REPORT + summary.json。

汇总 icc/splithalf/probe_selection/hb_fit/task_a/task_b 全部结果，
逐 probe 给出 PASS/FAIL 与证据；cbet 探索通道单独报告；能力边界声明。

输出 docs/MVP_STRUCTURE_REPORT.md + data/uoft/models/mvp/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase5b_common import ensure_dir  # noqa: E402

CORE_PROBES = ("vpip", "pfr", "three_bet", "position_sensitivity")


def _pass_vs_fail(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/uoft/models/mvp")
    ap.add_argument("--doc", default="docs/MVP_STRUCTURE_REPORT.md")
    args = ap.parse_args()
    d = Path(args.model_dir)

    icc = json.load(open(d / "icc.json", encoding="utf-8"))["probes"]
    sh = json.load(open(d / "splithalf.json", encoding="utf-8"))["probes"]
    sel = json.load(open(d / "probe_selection.json", encoding="utf-8"))
    hb = json.load(open(d / "hb_fit.json", encoding="utf-8"))
    ta = json.load(open(d / "task_a.json", encoding="utf-8"))["probes"]
    tb = json.load(open(d / "task_b.json", encoding="utf-8"))

    summary = {"core_probes": {}, "exploratory": {}, "overall": {}}
    core_pass = []
    lines = []
    lines.append("# MVP 玩家行为结构验证报告（Phase 5B）")
    lines.append("")
    lines.append("> 验证问题：**玩家行为是否存在稳定、可重复识别的结构**。")
    lines.append("> 数据：UofT CPRG 抽样 5000 玩家（seed=42，ring 口径 6-max+）。")
    lines.append("> 能力边界：本报告只验证结构存在性与层级收缩增益；**不声称** 读牌 / EV 最大化 / GTO exploit / Bluff 识别。")
    lines.append("")

    lines.append("## 0. 判定规则（冻结）")
    lines.append("")
    lines.append("- ICC：ICC(1,k) > 0.10 且 bootstrap 95% CI 下界 > 0（ICC(1) 仅报告，单手信度天然低，不设阈值）")
    lines.append("- split-half：r > 0.15 且 CI 下界 > 0")
    lines.append("- Task A：95% 预测区间覆盖率 >= 0.90")
    lines.append("- Task B：HB log-loss < 朴素个体频率（bootstrap CI 上界 < 0）")
    lines.append("- **不因部分指标失败而强行解释为成功**")
    lines.append("")

    lines.append("## 1. 结果总览")
    lines.append("")
    lines.append("| probe | 通道 | ICC(1) | ICC(1,k) | split-half r | Task A 预测覆盖 | Task B HB vs naive | 结论 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in ("vpip", "pfr", "three_bet", "position_sensitivity", "cbet"):
        s = sel["probes"][p]
        i = icc.get(p, {})
        shp = sh.get(p, {})
        tap = ta.get(p, {})
        tbp = tb.get(p, {}).get("probes", {}).get(p, {}) \
            if isinstance(tb.get("probes", {}).get(p, {}), dict) else {}
        icc1 = f"{i.get('icc1', 0):.3f}" if i.get("icc1") is not None else "—"
        icck = f"{i.get('icc1_k', 0):.3f}" if i.get("icc1_k") is not None else "—"
        r = f"{shp.get('r', 0):.3f}" if shp.get("r") is not None else "—"
        cov = tap.get("coverage_predictive")
        cov_s = f"{cov:.3f}" if cov is not None else "—"
        tbb = tb.get("probes", {}).get(p, {})
        diff = tbb.get("diff_hb_vs_naive")
        diff_s = f"{diff:+.4f}" if diff is not None else "—"
        if p == "cbet":
            verdict = "EXPLORATORY（不计核心判定）"
        elif p == "position_sensitivity":
            # ps 为派生差量：专用判定 = ICC ✓ + split-half ✓ + TaskB RMSE(hb)<RMSE(naive) + TaskA RMSE(hb)<RMSE(pop)
            tbps = tb.get("probes", {}).get(p, {})
            taps = ta.get(p, {})
            ps_ok = (s["decision"] == "core"
                     and (tbps.get("rmse_hb") or 1.0) < (tbps.get("rmse_naive") or 0.0)
                     and (taps.get("rmse_hb") or 1.0) < (taps.get("rmse_pop") or 0.0))
            verdict = "PASS" if ps_ok else "FAIL"
            if ps_ok:
                core_pass.append(p)
        else:
            verdict = "PASS" if (s["decision"] == "core"
                                 and tap.get("coverage_pass")
                                 and tbb.get("hb_better_than_naive")) else "FAIL"
            if verdict == "PASS":
                core_pass.append(p)
        lines.append(f"| {p} | {s['decision']} | {icc1} | {icck} | {r} | "
                     f"{cov_s} | {diff_s} | {verdict} |")
    lines.append("")

    # 每 probe 详细节
    for p in ("vpip", "pfr", "three_bet", "position_sensitivity", "cbet"):
        s = sel["probes"][p]
        lines.append(f"## 2. {p}（{s['decision']}）")
        lines.append("")
        if s["reason"]:
            lines.append(f"- 通道原因: {s['reason']}")
        i = icc.get(p, {})
        if i:
            lines.append(f"- ICC(1)={i.get('icc1')} (CI {i.get('icc1_ci')}) | "
                         f"ICC(1,k)={i.get('icc1_k')} (CI {i.get('icc1_k_ci')}) | "
                         f"n0={i.get('n0')} 玩家={i.get('k')} 手={i.get('N')}")
        shp = sh.get(p, {})
        if shp:
            lines.append(f"- split-half r={shp.get('r')} (CI {shp.get('r_ci95')}) | "
                         f"Spearman-Brown 全长信度={shp.get('r_spearman_brown')} | "
                         f"n={shp.get('n_players')}")
            if shp.get("note"):
                lines.append(f"- 注: {shp['note']}")
        tbb = tb.get("probes", {}).get(p, {})
        if tbb and isinstance(tbb, dict) and tbb.get("ll_hb") is not None:
            lines.append(f"- Task B: ll_hb={tbb['ll_hb']} vs ll_naive={tbb['ll_naive']} "
                         f"(diff {tbb['diff_hb_vs_naive']:+.4f}, CI "
                         f"{tbb['diff_hb_vs_naive_ci']}) | "
                         f"n={tbb.get('n_evaluated')}")
        tap = ta.get(p, {})
        if tap and tap.get("coverage_predictive") is not None:
            lines.append(f"- Task A: 预测覆盖率={tap['coverage_predictive']} "
                         f"(θ 后验区间={tap.get('coverage_theta_ci')}) | "
                         f"RMSE hb={tap.get('rmse_hb')} pop={tap.get('rmse_pop')} | "
                         f"n={tap.get('n_evaluated')}")
        if p == "position_sensitivity" and tbb.get("rmse_hb") is not None:
            lines.append(f"- Task B(ps 差分 RMSE): hb={tbb.get('rmse_hb')} "
                         f"pop={tbb.get('rmse_pop')} naive={tbb.get('rmse_naive')}")
        if p != "cbet" and p in core_pass:
            lines.append(f"- **结论: PASS**")
        elif p == "cbet":
            lines.append("- 结论: 探索通道（低样本机会门控），单独报告不计核心判定")
        else:
            lines.append("- **结论: FAIL（见原因，不强行解释）**")
        lines.append("")

    lines.append("## 3. HB 先验（经验贝叶斯矩估计，train 玩家拟合）")
    lines.append("")
    lines.append("| probe | mu | a | b | nu | 数值保护 |")
    lines.append("|---|---|---|---|---|---|")
    for p in ("vpip", "pfr", "three_bet", "cbet", "rfi_ep", "rfi_lp"):
        h = hb["probes"][p]
        prot = f"触发({h['protection_reason']})" if h["protection_triggered"] else "无"
        lines.append(f"| {p} | {h['mu']} | {h['a']} | {h['b']} | {h['effective_nu']} | {prot} |")
    lines.append("")
    lines.append("- 数值保护: NU_MAX=100000；raw_sigma2_theta/raw_nu 已在 hb_fit.json 完整记录，")
    lines.append("  无静默 clip。")
    lines.append("")

    lines.append("## 4. 时间审计（Task B as_of 语义）")
    lines.append("")
    time_audit = tb.get("time_audit", {})
    n_ok = sum(1 for v in time_audit.values() if v.get("as_of_ok"))
    lines.append(f"- 审计玩家数: {len(time_audit)}；as_of_ok=True: {n_ok}；违反: "
                 f"{len(time_audit) - n_ok}")
    lines.append("- 字段: history_end_timestamp / future_start_timestamp（可在 task_b.json 逐玩家人工核验）")
    lines.append("")

    lines.append("## 5. 总结")
    lines.append("")
    if core_pass:
        lines.append(f"- **结构验证通过（核心探针）**: {', '.join(core_pass)}")
    else:
        lines.append("- **核心探针全部失败或部分失败（见上）**")
    lines.append("- CBet: 探索通道（不计核心判定）")
    lines.append("- GMM/DPMM: 按计划保留到 Phase 6，本阶段未引入。")
    lines.append("")
    lines.append("> 诚实声明：本报告验证的是「玩家行为存在稳定、可重复结构」且「层级收缩优于朴素频率」。")
    lines.append("> 它不构成对读牌/EV/GTO/Bluff 能力的任何声明。")

    doc = "\n".join(lines) + "\n"
    Path(args.doc).write_text(doc, encoding="utf-8")

    summary["core_probes"] = {p: {"decision": sel["probes"][p]["decision"],
                                  "task_a_coverage_pass": ta[p].get("coverage_pass"),
                                  "task_b_hb_better": tb["probes"][p].get(
                                      "hb_better_than_naive")}
                              for p in CORE_PROBES}
    summary["exploratory"] = {"cbet": sel["probes"]["cbet"]}
    summary["overall"] = {
        "core_passed": core_pass,
        "all_core_pass": len(core_pass) == len(CORE_PROBES),
        "as_of_violations": len(time_audit) - n_ok,
        "hb_numerical_protection_triggered": {
            p: hb["probes"][p]["protection_triggered"] for p in hb["probes"]},
    }
    (d / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
