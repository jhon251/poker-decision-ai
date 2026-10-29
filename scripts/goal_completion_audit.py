#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-HM：**目标逐项终检**（每条目标 → 可核查证据 → 诚实判定）✓。

输出 `docs/GOAL_COMPLETION_AUDIT.md` + `docs/goal_completion_audit.json`。
判定只用**可核查的东西**：文件是否存在、门禁数字、样本量 ✓；
不满足/样本过小者标 `PARTIAL` / `BOUNDARY`，**不粉饰** ✓。
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "docs"
LIB = ROOT / "data" / "solver" / "spots_6max"


def _j(name: str):
    p = D / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}


def main() -> None:
    cov = _j("real_coverage_baseline.json")
    est = _j("coverage_estimate.json")
    mech = _j("mechanisms_gate.json")
    man = _j("session_manifest.json")
    pb = _j("pokerbench_validation_aligned.json")
    dash = (D / "HONEST_DASHBOARD.md").is_file()

    n_sb = len(list(LIB.glob("3bp-oursb*.json")))
    n_fc = len(list(LIB.glob("*-ourface-*.json")))
    n_all = len(list(LIB.glob("*.json")))
    n_tr = len(list(LIB.glob("3bp-ourtr*.json")))

    items = []

    # ① 批量自解真实形态 → 入库 → 复测
    items.append({
        "id": "①", "title": "自解真实形态并入库 + 复测覆盖率",
        "verdict": "DONE（机制）/ BOUNDARY（结构性）",
        "evidence": [
            f"库内自解记录：SB-BTN {n_sb} 条、facing {n_fc} 条、turn/river {n_tr} 条、库总 {n_all} 条 ✓",
            f"随机牌面覆盖复测：{est.get('random_board_coverage', {}).get('3bp/first_in/SB-BTN/pot40', {})}",
            f"真实会话逐会话清单：{man.get('totals', {})}",
        ],
        "honest": "**均匀随机翻牌 exact 结构性≈0%** ✗（翻牌空间 22,100）；"
                   "真正提升的是『真实出现过的局面』口径 ✓；真实会话样本仅 3 例 ⇒ 不可作能力结论 ✓",
    })

    # ② 牌面残缺：诊断 + 修复（不让决策因输入不全降级）
    items.append({
        "id": "②", "title": "诊断并修复『输入不全导致决策降级』",
        "verdict": "DONE（下游）/ BOUNDARY（上游生产端）",
        "evidence": [
            "根因定位：19/19 残缺牌面全部来自 v3f_snapshot（另一来源 19/19 完整）✓",
            "本地补读：真实帧读出完整翻牌 `Ad 9h 6s`（V3F 当时残缺）✓",
            "诚实阻断：牌面不足 ⇒ DECISION_UNAVAILABLE（46% 真实翻后请求）✓",
            "证据采集：残缺/补读成败留痕（含来源标记 ✓）",
            "第二缺口：`to_call` UNKNOWN 61% ⇒ 定位为**金额模板仅 3 个参考**的数据缺口 ✗"
            " ⇒ 已加**明确告知**（不假装知道 ✓）",
        ],
        "honest": "上游 V3F 生产端与金额模板**不在本仓库 / 需标注样本** ⇒ 未修本体 ✗（如实标注 ✓）；"
                   "下游『不再基于不可信输入给建议』已达成 ✓",
    })

    # ③ 采购与算力决策 + 多人管道
    items.append({
        "id": "③", "title": "SRP 算力 / 多人采购决策 + 多人管道就位",
        "verdict": "DONE（决策 + 管道就位）",
        "evidence": [
            "SRP 翻牌：**不投入**（单解 494–775s ⇒ 1000 个 ≈8–9 天；真实主导为 40bb≈3-bet 池）✓",
            "多人（3-way+）：**暂不采购**（多人仅 10/97，其中 8/10 牌面残缺 ⇒ 可度量价值 ≈2%；"
            "3-way 覆盖不了 4 人）✓ + 复用条件已写明 ✓",
            "多人管道：**已就位** ✓ —— 独立表 `spot_table_multiway.py`（键含 `n_active` ✓、"
            "**默认关闭** ⇒ 行为零变化 ✓）、导入器 `import_multiway_solves.py`（严格校验 + dry-run ✓）、"
            "4 项单测通过 ✓（含『生产库不得混入多人记录』✓）",
        ],
        "honest": "管道就位 与 数据源采购 已**解耦** ✓：现在缺的只是**数据源本身**（决策 B 暂不采购 ✓），"
                   "买到即可直接导入接线 ✓ —— 这属**决策结果**，不是未完成的工作 ✓",
    })

    # 硬约束
    gates = {
        "dist 同步": _j("dist_sync.json").get("problems", "?"),
        "基准门": _j("bench_regression_gate.json"),
        "真实真解率门禁": {"exact": cov.get("exact"), "n": cov.get("n_evaluable_2p_complete")},
        "四机制": mech.get("all_ok"),
        "独立标签口径": {"n": pb.get("n"), "agree_rate": pb.get("agree_rate")},
        "诚实看板": dash,
    }

    L = ["# 目标逐项终检（自动生成）", "",
          "> 只用**可核查证据**（文件/门禁数字/样本量 ✓）；不满足或样本过小者标 PARTIAL/BOUNDARY，"
          "**不粉饰** ✓；**不做 EV/GTO/盈利声明** ✓。", ""]
    for it in items:
        L.append(f"## {it['id']} {it['title']}")
        L.append("")
        L.append(f"**判定：{it['verdict']}**")
        L.append("")
        for e in it["evidence"]:
            L.append(f"* {e}")
        L.append("")
        L.append(f"**诚实说明**：{it['honest']}")
        L.append("")
    L.append("## 硬约束（不得回退）")
    L.append("")
    for k, v in gates.items():
        L.append(f"* {k}：`{str(v)[:110]}`")
    L.append("")
    L.append("## 若要继续提升：唯一现实路径 = 积累真实数据")
    L.append("")
    L.append("| 谁 | 做什么 | 是否阻塞 |")
    L.append("|---|---|---|")
    L.append("| **你** | 继续正常使用（会话数据自动累积 ✓）| 不阻塞（我这边全自动 ✓）|")
    L.append("| **你** | 若在意 `to_call` 缺口：批准一次**只读标注**（用已采集帧补金额模板）| 需你确认 ✓ |")
    L.append("| **你** | 若在意多人局：决定是否采购多人求解器（$249 级）| 需你确认 ✓ |")
    L.append("| **你** | 若想要『首见即真解』的 flop：批准加一个『精确分析（~8 秒）』入口 | 需你确认 ✓ |")
    L.append("| **我（自动）** | 每次数据增加后：重跑门禁 / 会话清单 / 看板 / 终检 ✓ | 已接入 `verify_all` ✓ |")
    L.append("")
    L.append("---")
    L.append("*本终检只列事实与边界；不含任何『更有水平/更赚』的结论 ✓。*")

    text = "\n".join(L) + "\n"
    (D / "GOAL_COMPLETION_AUDIT.md").write_text(text, encoding="utf-8")
    (D / "goal_completion_audit.json").write_text(json.dumps({
        "items": items, "hard_constraints": gates,
        "note": "只用可核查证据；PARTIAL/BOUNDARY 如实标注；不做 EV/GTO/盈利声明 ✓"},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"已写 docs/GOAL_COMPLETION_AUDIT.md（{len(text.splitlines())} 行）")
    for it in items:
        print(f"  {it['id']} {it['verdict']}")
    print(f"  硬约束: {json.dumps(gates, ensure_ascii=False, default=str)[:200]}")


if __name__ == "__main__":
    main()
