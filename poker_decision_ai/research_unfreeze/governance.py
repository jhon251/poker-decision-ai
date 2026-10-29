"""M6 治理层：RESULT 限制声明（B6）+ 工件 fail-closed 守卫（B7）。

不含任何策略逻辑；不修改任何历史封存工件；不修改 strategy_core/CFI/λ。

M7：工件守卫的实现移至中性位置 `poker_decision_ai/artifact_guard.py`，
本模块 re-export 同名对象（对外 API 不变）。原因：benchmark 的真实 artifact
读取路径需要 fail closed 校验，而 B7-1 隔离约束禁止生产包 import
research_unfreeze —— 迁移后两个约束同时成立。
"""
from __future__ import annotations

import json

GOVERNANCE_VERSION = "v3f-result-governance-v1"

# B6：下一版本 RESULT 必须内嵌的限制区块（六项，缺一即不合法）
REQUIRED_LIMITATIONS = ("independence", "solver", "profit_unidentifiable",
                        "lambda_u2", "defect_b_meaning", "accept_scope")

LIMITATIONS = {
    "independence": {
        "value": "NOT INDEPENDENT",
        "detail": "S‴′ 与 S‴ 使用同一冻结决策集；为器械校正重跑，不主张样本独立性。",
        "b1": "BLOCKED（独立样本不可达：13/13 玩家已用尽）"},
    "solver": {
        "value": "BLOCKED",
        "detail": "真实 postflop solver 未配置（POSTFLOP_SOLVER_PATH 未设置）。",
        "reference": "L2 / perfect-information check-down；equilibrium=false；NOT_exact_gto=true"},
    "profit_unidentifiable": {
        "value": "UNIDENTIFIABLE",
        "detail": ("语料中的结果是**人类策略**的实现值；系统所选动作在 54.6% 的点上与"
                   "已实现动作不同，且无随机指派/执行环境 → 收益效应不可识别（M3 BLOCKED）。"),
        "forbidden": "不得把 PI-best 一致率描述为收益、GTO 或均衡证明"},
    "lambda_u2": {
        "value": "BLOCKED",
        "detail": ("λ/U2 未在完整参照下重算；历史 λ 敏感性基于**旧参照**（1,770 点漏算在场"
                   "AI 座、1,356 点被计 unavailable）→ 旧结果不得替代（M4 BLOCKED）。")},
    "defect_b_meaning": {
        "value": "CLOSED (scope-limited)",
        "detail": ("Defect B 指**遗留 sealed runner** 不记录 AI 座弃牌；S‴′ 所用 runner 以统一"
                   "记账覆盖 AI 座，residual mismatch = 0。该结论**不**表示历史 sealed "
                   "S‴/S″/S′ 的结果被重算。"),
        "corrects": "早期“只披露未修”的表述对实际使用 runner 而言过度保守。"},
    "accept_scope": {
        "value": "R2.2 ACCEPT（限本冻结决策集）",
        "detail": ("仅适用于 S‴′ 7,483 个冻结决策点、完整参照 `pi-ev-reference-complete-v1`、"
                   "L2 非均衡；不外推至其它玩家池/语料/模式。"),
        "delta_pp": 20.9809, "ci95_pp": [19.4441, 22.5979]},
}


def govern_result(arm_result: dict) -> dict:
    """把一份研究 RESULT 包装成**受治理**版本（不修改原文件）。"""
    out = {"governance_version": GOVERNANCE_VERSION,
           "source_runner_version": arm_result.get("runner_version"),
           "arm": arm_result.get("arm"), "run_id": arm_result.get("run_id"),
           "summary": arm_result.get("summary"),
           "gates": arm_result.get("gates"),
           "counters": arm_result.get("counters"),
           "limitations": LIMITATIONS,
           "claims": {"may_claim": [
               "在该冻结决策集上，R2.2 的 PI-best 一致率显著高于 frozen（paired，CI 下界>0）",
               "器械完整性/门禁/封存/一致性测试通过"],
               "may_not_claim": [
                   "盈利能力或任何经济性结论", "独立性/可复现性（B1 BLOCKED）",
                   "GTO/均衡一致性（Solver BLOCKED）", "对其它语料/玩家池/模式的外推",
                   "λ/U2 的稳健性（M4 BLOCKED）"]}}
    missing = [k for k in REQUIRED_LIMITATIONS if k not in out["limitations"]]
    if missing:
        raise ValueError(f"治理层缺少限制声明: {missing}")
    return out


def validate_governed(doc: dict) -> list:
    """返回违规列表（空 = 合法）。"""
    bad = []
    if doc.get("governance_version") != GOVERNANCE_VERSION:
        bad.append("governance_version 不匹配")
    lim = doc.get("limitations") or {}
    for k in REQUIRED_LIMITATIONS:
        if k not in lim:
            bad.append(f"缺少限制声明: {k}")
    text = json.dumps(doc, ensure_ascii=False)
    for banned in ("具有盈利能力", "达到 GTO", "均衡最优", "独立验证通过"):
        if banned in text:
            bad.append(f"出现禁止性表述: {banned}")
    return bad


# ── B7-2：工件 fail-closed 守卫（stale / wrong-manifest / 缺失 一律拒绝） ──
# M7 迁移：guarded 实现移至中性位置 `poker_decision_ai/artifact_guard.py`，
# 使 benchmark **真实读取路径**可 fail closed 校验，同时不违反 B7-1
# 「生产包不得 import research_unfreeze」的隔离约束。
# 此处保留同名 re-export：调用点 `governance.ArtifactGuard(...).require(rel)`
# 的语义、签名、异常类型完全不变（无行为改动）。
from ..artifact_guard import (ArtifactGuard,  # noqa: F401  (re-export)
                              ArtifactGuardError, sha256_file)
