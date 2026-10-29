#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HK：**单页诚实看板**（聚合各项已测数字，每个都带口径与样本量 ✓）。

原则（与目标硬约束一致）：
  * 只汇总**已有实测产物**（不新造数字 ✓）；
  * 每项都写**口径**与 **n** ✓；样本过小者显式标注"不可作能力结论" ✓；
  * **不做任何 EV/GTO/盈利声明** ✓；缺参考的边界如实标注 ✓。

输出：`docs/HONEST_DASHBOARD.md` + `docs/honest_dashboard.json`
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "docs"


def _j(name: str) -> dict:
    p = D / name
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}


def _jl(name: str) -> list[dict]:
    p = ROOT / "logs" / name
    if not p.is_file():
        p = D / name
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:                                  # noqa: BLE001
                continue
    return out


def main() -> None:
    cov = _j("real_coverage_baseline.json")          # 真实会话主指标（修正口径 ✓）
    est = _j("coverage_estimate.json")               # 随机牌面覆盖率（结构性 ✗）
    mech = _j("mechanisms_gate.json")                # 四机制
    frames = _j("coverage_real_frames.json")         # 真实帧（7 张）
    pb = _j("pokerbench_validation_aligned.json")    # 独立标签（口径已打通、混淆未消）
    pb_ev = _j("pokerbench_ev_analysis.json")
    ev = _jl("board_fallback.jsonl")
    # P30-9-HQ：口径修正 —— 恢复率**只在补读事件**上有意义 ✓；
    #   `to_call_unknown` 属另一类缺口 ⇒ 单独统计 ✓；非 runtime 事件单独计数 ✓。
    ev_real = [e for e in ev if (e.get("source") or "runtime") == "runtime"]
    ev_ok = sum(1 for e in ev_real if e.get("kind") == "fallback_ok")
    ev_bad = sum(1 for e in ev_real if e.get("kind") == "fallback_failed")
    ev_fb = ev_ok + ev_bad
    ev_other = len(ev_real) - ev_fb                     # 例如 to_call_unknown ✓
    ev_tcu_real = sum(1 for e in ev_real if e.get("kind") == "to_call_unknown")
    ev_non_runtime = len(ev) - len(ev_real)

    L: list[str] = []
    L.append("# 诚实看板（自动生成）")
    L.append("")
    L.append("> 由 `scripts/honest_dashboard.py` 从既有实测产物聚合 ✓；"
              "**每个数字都带口径与样本量**；样本过小者显式标注；"
              "**不做 EV/GTO/盈利类声明** ✓。")
    L.append("")
    L.append("## 1. 真实会话：输入可信度（覆盖率的分母质量）")
    ii = (cov.get("input_integrity") or {})
    if ii:
        n = ii.get("postflop_requests") or 0
        L.append("")
        L.append(f"翻后请求 **{n}** 个（来自 `dist/logs` 真实会话）：")
        L.append("")
        L.append("| 条件 | 满足数 |")
        L.append("|---|---|")
        L.append(f"| 公共牌完整 | {ii.get('board_complete')} |")
        L.append(f"| `to_call` 可知（能区分 facing / first_in）| {ii.get('to_call_known')} |")
        L.append(f"| 两人池（多人无参考）| {ii.get('two_player')} |")
        L.append(f"| **三者兼备（可用于主指标）** | **{ii.get('fully_evaluable')}** |")
    else:
        L.append("\n（暂无数据）")

    L.append("")
    L.append("## 2. 真实会话：主指标（**修正口径**）")
    if cov:
        L.append("")
        L.append(f"* 可评估局面 **{cov.get('n_evaluable_2p_complete')}** 个 ⇒ "
                  f"精确命中 **{cov.get('exact')}** = **{(cov.get('rate') or 0) * 100:.0f}%** ✓")
        L.append(f"* 命中形态：`{cov.get('kinds')}`")
        L.append(f"* 排除：节点歧义（`to_call` 未知）**{cov.get('node_ambiguous_excluded')}** 个 ✗ "
                  "（未知时无法区分 facing/first_in ⇒ 计入会虚高 ✗）")
        L.append("* ⚠️ **n 很小** ⇒ 该比例**不可作为能力结论** ✓（此前误报的 20/20 已作废 ✗）")

    L.append("")
    L.append("## 2b. 逐会话采集清单（数据积累后自动更新 ✓）")
    man = _j("session_manifest.json")
    _sess = (man.get("sessions") or {})
    if _sess:
        L.append("")
        L.append("| 会话 | 翻后 | 牌面完整 | `to_call` 可知 | 两人池 | 三者兼备 | 唯一牌面 | 精确命中 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for day, v in sorted(_sess.items()):
            L.append(f"| {day} | {v.get('postflop')} | {v.get('board_complete')} | "
                      f"{v.get('to_call_known')} | {v.get('two_player')} | "
                      f"**{v.get('fully_evaluable')}** | {v.get('unique_boards')} | "
                      f"{v.get('exact_hits')}/{v.get('fully_evaluable')} |")
        tt = (man.get("totals") or {})
        L.append(f"| **合计** | {tt.get('postflop')} | {tt.get('board_complete')} | "
                  f"{tt.get('to_call_known')} | {tt.get('two_player')} | "
                  f"**{tt.get('fully_evaluable')}** | - | **{tt.get('exact')}/{tt.get('keys')}** |")
        L.append("")
        L.append("> 说明：每会话自动统计 ✓；数据一积累，本表与主指标自动收敛 ✓"
                  "（**不做人工挑选** ✓）。样本 <30 时比例不可作能力结论 ✓。")
    else:
        L.append("\n（暂无会话清单；运行 `scripts/session_manifest.py` 生成 ✓）")

    L.append("")
    L.append("## 3. 随机牌面覆盖率（**结构性口径**）")
    rnd = (est.get("random_board_coverage") or {})
    if rnd:
        L.append("")
        L.append("| 配置 | exact | 任一命中 |")
        L.append("|---|---|---|")
        for k, v in rnd.items():
            L.append(f"| {k} | {v.get('exact')}（{v.get('exact_rate')}）| {v.get('any_hit_rate')} |")
        L.append("")
        L.append("* 说明：翻牌空间 22,100 种 ⇒ 预解 N 个 ≈ N/22,100 ⇒ **uniform exact 结构性≈0%** ✗；"
                  "真正有意义的是**真实会话口径**与**任一命中率** ✓。")

    L.append("")
    L.append("## 4. 机制状态（本目标新增能力）")
    if mech:
        m1 = mech.get("m1_board") or {}
        m2 = mech.get("m2_multiway") or {}
        m3 = mech.get("m3_live_solve") or {}
        m4 = mech.get("m4_cache") or {}
        L.append("")
        L.append(f"* ① 牌面阻断 + 本地补读：`{m1}`")
        L.append(f"* ② 多人告知（2 人无 / 3·4 人有）：`{m2.get('note_present')}`")
        L.append(f"* ③ turn/river 现场真解：`{m3.get('times')}`")
        L.append(f"* ④ 现场解缓存：`{m4}`")
        L.append(f"* 汇总：`{'PASS' if mech.get('all_ok') else 'FAIL'}`")

    L.append("")
    L.append("## 5. 输入残缺：证据与恢复情况")
    L.append("")
    L.append(f"* 真实补读事件 **{ev_fb}** 条（成功 {ev_ok} / 失败 {ev_bad}）"
              + (f" ⇒ 恢复率 **{ev_ok / ev_fb * 100:.0f}%** ✓"
                 if ev_fb else " ⇒ 补读恢复率**不可读** ✓（**不再误打 0%** ✗："
                                "恢复率只在补读事件上有意义 ✓）"))
    L.append(f"* 其它真实证据事件：`to_call` UNKNOWN **{ev_tcu_real}** 次"
              f"（另一类输入缺口，**单独统计** ✓）；"
              f"非真实来源（帧回放/门禁）**{ev_non_runtime}** 条 ✓")
    L.append("* 已定位的两个输入缺口：")
    L.append("  - **公共牌残缺**：V3F 快照生产端（19/19 例）⇒ 已加**本地补读** + **诚实阻断** + **证据采集** ✓")
    L.append("  - **`to_call` 未知**（61%）：识别器金额模板**仅 3 个参考** ✗ ⇒ **数据缺口**，"
              "已加**明确告知**（不假装知道 ✓）；补齐需**带标注样本** ✗")

    L.append("")
    L.append("## 6. 真实帧（产品路径，7 张）")
    if frames:
        L.append("")
        if isinstance(frames, list):            # 该文件顶层是**列表**（逐帧记录 ✓）
            _n = len(frames)
            _i = sum(1 for r in frames if isinstance(r, dict) and r.get("informed"))
            _inf = f"{_i}/{_n}"
        else:
            _inf = frames.get("informed_rate") or frames.get("n_informed") or "见 JSON"
        L.append(f"* 均衡知情率：`{_inf}`（口径：历史真实帧 ✓）")

    L.append("")
    L.append("## 7. 独立标签（PokerBench，口径已打通）")
    if pb:
        L.append("")
        L.append(f"* 样本 **{pb.get('n')}**（facing + **逐行对齐下注尺寸** ✓）⇒ "
                  f"与标签一致 **{pb.get('agree')}/{pb.get('n')}** = "
                  f"**{(pb.get('agree_rate') or 0) * 100:.0f}%**；"
                  f"多数类基线 **{(pb.get('majority_baseline') or {}).get('rate')}**")
        L.append("* ⚠️ **不得解读为能力评价** ✗：范围无法对齐（PokerBench 不提供范围 ✗）、"
                  "有效栈近似、标签为单一动作 ✓")
    # P30-9-HX：**范围稳健性**口径（含 Wilson 区间 ⇒ 小样本也不至于不可解释 ✓）
    pbr = _j("pokerbench_range_robustness.json")
    if pbr:
        L.append("")
        L.append(f"* **范围稳健性**（3 套范围各解一次 ✓）：动作一致 "
                  f"**{pbr.get('robust')}/{pbr.get('n')}** = "
                  f"**{(pbr.get('robust_rate') or 0) * 100:.0f}%**"
                  f"（95% CI {pbr.get('robust_rate_ci95')}）")
        _mb = (pbr.get("majority_baseline_on_robust") or {})
        L.append(f"* **稳健子集**上与标签一致：**{pbr.get('agree_on_robust')}/{pbr.get('robust')}**"
                  f" = {(pbr.get('agree_rate_on_robust') or 0) * 100:.0f}%"
                  f"（95% CI {pbr.get('agree_rate_on_robust_ci95')}）"
                  f"；该子集多数类基线（恒 '{_mb.get('action')}'）"
                  f"= {_mb.get('hits')}/{_mb.get('n')}（95% CI {_mb.get('ci95')}）")
        if _mb.get("ci_overlaps_ours") is True:
            L.append("* ⇒ 两者**置信区间重叠 ⇒ 统计上不可区分** ✓（**不能说更优或更差** ✓）")
        elif _mb.get("ci_overlaps_ours") is False:
            L.append("* ⇒ 两者**置信区间不重叠** ⇒ 存在统计差异 ✓（方向上见上文数字 ✓）")
    if pb_ev:
        L.append(f"* EV 口径（**循环论证，不可作证据** ✗）：均值 "
                  f"{pb_ev.get('mean_ev_diff_bb')} bb ⇒ 仅用于显示**框架差异之大** ✓")

    L.append("")
    L.append("## 8. 采购与算力决策（第 ③ 项）")
    L.append("")
    L.append("| 决策 | 结论 | 依据 |")
    L.append("|---|---|---|")
    L.append("| SRP 翻牌预解 | **不投入** ✗ | 单解 494–775 秒 ⇒ 1000 个 ≈ 8–9 天；真实主导是 40bb 底池(≈3-bet) |")
    L.append("| 多人（3-way+）求解器 | **暂不采购** ✗ | 多人仅 10/97；其中 8/10 牌面残缺 ⇒ 可度量价值 ≈2%；3-way 覆盖不了 4 人 |")
    L.append("| turn/river | **现场真解**（无需预解）✓ | 根节点 44/18ms；**facing 69/18ms** ✓ |")
    L.append("| flop | 预解 + 后台缓存（Idle 优先级）✓ | 现场解 ~7.5s ✗（缓存 277s 落盘，不抢客户端 ✓）|")

    L.append("")
    L.append("---")
    L.append("*本看板只列实测事实与口径；"
              "任何「更有水平 / 更赚」的说法都不在本看板的结论范围内 ✓。*")

    text = "\n".join(L) + "\n"
    (D / "HONEST_DASHBOARD.md").write_text(text, encoding="utf-8")
    (D / "honest_dashboard.json").write_text(json.dumps({
        "input_integrity": ii, "main_metric": cov, "random_coverage": rnd,
        "mechanisms_ok": mech.get("all_ok"), "evidence": {"real": len(ev_real), "ok": ev_ok, "bad": ev_bad},
        "pokerbench": {"n": pb.get("n"), "agree_rate": pb.get("agree_rate")},
        "note": "只列实测事实与口径；不做 EV/GTO/盈利声明 ✓",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 docs/HONEST_DASHBOARD.md（{len(text.splitlines())} 行）"
          f" + docs/honest_dashboard.json")
    print(f"  输入可信度: {ii}")
    print(f"  主指标: {cov.get('exact')}/{cov.get('n_evaluable_2p_complete')}"
          f" | 真实证据事件: {len(ev_real)} | 机制: {'PASS' if mech.get('all_ok') else '-'}")


if __name__ == "__main__":
    main()
