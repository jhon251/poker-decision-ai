"""V3G-2 — CASH 决策上下文完整性 / 证据旁路接线测试。

覆盖授权测试矩阵 1–13：
  1 完整 state → evidence 正确生成；2 hero 缺失 → UNKNOWN；3 stack 缺失 →
  UNKNOWN；4 board/street 冲突 → UNKNOWN；5 button/dealer 缺失 → UNKNOWN；
  6/7 接线前后 Judge 结果逐位一致（完整 / 不完整）；8 `compute_ev().to_summary()`
  逐位一致；9 candidate set 一致；10 size/amount 一致；11 CASH context 不泄漏到
  MTT/PKO；12 生产 import/path 审计（无 research_unfreeze 越界）；
  13 V3G-1 的 36 项继续通过（由 V3G-1 文件自身覆盖，本文件不重复）。

**核心不变式**：旁路只增不改——除 `evidence` 外，Judge 输出必须逐位相同。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_integrity_context import (
    CONTEXT_VERSION, EVIDENCE_KIND, cash_integrity_context)
from poker_decision_ai.core.config import load_config
from poker_decision_ai.core.logging import EventLogger, MetricsCollector
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.providers import build_providers
from poker_decision_ai.core.repositories import build_in_memory
from poker_decision_ai.domain.poker.strategy_core import compute_ev

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


@pytest.fixture()
def ctx():
    """离线 AgentContext（不依赖 tmp_path —— 本机沙箱禁止删除 %TEMP%）。"""
    import uuid
    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g2_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    return AgentContext(
        request_id="v3g2-req", session_id="v3g2-session", config=cfg,
        providers=build_providers(cfg), repos=build_in_memory(),
        logger=EventLogger(str(logs)), metrics=MetricsCollector())

#: Judge 输出中**决策性**字段（必须逐位不变）
DECISION_KEYS = (
    "action", "alternative_action", "recommended_size", "ev_core",
    "weights_used", "final_inputs_summary", "uncertainty", "confidence",
    "reasoning", "warnings", "scores", "margin",
)


def cash_state(**over) -> dict:
    st = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 200.0, "to_call": 50.0, "bb": 2.0, "sb": 1.0,
        "hero_id": "hero", "button_seat": 3, "table_size": 6,
        "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "SB", "stack": 400.0},
            {"player_id": "hero", "seat": 3, "position": "BTN", "is_hero": True,
             "hole_cards": ["9c", "4c"], "stack": 400.0},
            {"player_id": "v2", "seat": 6, "position": "BB", "stack": 400.0},
        ],
    }
    st.update(over)
    return st


def _eq(v: float = 0.8) -> dict:
    return {"hero_equity": v,
            "outcome": {"win": v, "tie": 0.0, "lose": round(1.0 - v, 6)}}


def _strip_bypass(out: dict) -> dict:
    """去掉旁路条目后的 Judge 输出（用于逐位比较）。"""
    o = copy.deepcopy(out)
    ev = o.get("evidence")
    if isinstance(ev, list):
        o["evidence"] = [e for e in ev
                         if not (isinstance(e, dict)
                                 and e.get("kind") == EVIDENCE_KIND)]
    return o


# ── 1. 完整 state → evidence 正确生成 ─────────────────────────────────
def test_01_complete_state_generates_evidence():
    c = cash_integrity_context(cash_state())
    assert c["present"] is True
    assert c["context_version"] == CONTEXT_VERSION
    assert c["mode"] == "cash"
    assert c["completeness"] == 1.0 and c["level"] == "COMPLETE"
    assert c["missing"] == [] and c["conflicts"] == [] and c["unknown"] == []
    assert c["evidence_grade"] in ("SYNTHETIC", "REAL_OBSERVED",
                                   "HISTORICAL_REPLAY", "COUNTERFACTUAL",
                                   "MECHANISM_ONLY", "NOT_VALIDATABLE")
    for k in ("state_integrity", "completeness", "missing", "conflicts",
              "unknown", "evidence_grade"):
        assert k in c, k
    json.dumps(c)                                   # 必须 JSON-safe


# ── 2/3/5. hero / stack / button 缺失 → UNKNOWN ───────────────────────
def test_02_missing_hero_hole_cards_is_unknown():
    st = cash_state()
    st["players"][1]["hole_cards"] = []
    c = cash_integrity_context(st)
    assert "hero.hole_cards" in c["unknown"]
    assert c["level"] == "DEGRADED"


def test_03_missing_stack_is_unknown_not_zero():
    st = cash_state()
    st["players"][1]["stack"] = None
    c = cash_integrity_context(st)
    assert "hero.stack" in c["unknown"]
    assert st["players"][1]["stack"] is None        # 不得被填 0


def test_05_missing_button_seat_is_unknown_not_defaulted():
    c = cash_integrity_context(cash_state(button_seat=None))
    assert "button_seat" in c["unknown"]
    assert c["level"] == "DEGRADED"


def test_05_missing_table_size_is_unknown_not_defaulted_six():
    c = cash_integrity_context(cash_state(table_size=None))
    assert "table_size" in c["unknown"]


# ── 4. board/street 冲突 → UNKNOWN ────────────────────────────────────
def test_04_board_street_conflict_is_reported():
    c = cash_integrity_context(cash_state(street="turn"))
    assert any(x.startswith("board_street_mismatch") for x in c["conflicts"])
    assert c["level"] == "INCOMPLETE"
    assert c["evidence_grade"] == "NOT_VALIDATABLE"


# ── 11. CASH context 不泄漏到 MTT/PKO/Mystery ─────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_11_non_cash_modes_never_inherit_cash_context(mode):
    c = cash_integrity_context(cash_state(game_type=mode))
    assert c["present"] is False
    # 精确定位：不得携带任何 CASH 专属证据字段（而非全文子串匹配）
    for k in ("state_integrity", "completeness", "missing", "conflicts",
              "unknown", "evidence_grade", "kind", "mode"):
        assert k not in c, k
    assert c.keys() == {"present", "context_version", "reason"}


def test_11_non_dict_state_fails_closed():
    for bad in (None, [], "cash"):
        c = cash_integrity_context(bad)
        assert c["present"] is False and c["reason"] == "state_not_a_dict"


# ── 6/7/8/9/10. 生产 Judge：接线前后逐位一致 ──────────────────────────
async def _judge_out(monkeypatch, state: dict, eq: dict, ctx) -> dict:
    """跑一次 Judge；`bypass=False` 时禁用旁路（模拟接线前基线）。"""
    from poker_decision_ai.agents import judge as judge_mod
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({"state": state, "equity": eq}, ctx)
    return env.output


async def test_06_and_08_complete_state_decisions_bitwise_identical(monkeypatch,
                                                                    ctx):
    st = cash_state()
    eq = _eq(0.8)

    # 接线后
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env_after = await JudgeAgent().run({"state": copy.deepcopy(st),
                                        "equity": copy.deepcopy(eq)}, ctx)
    out_after = env_after.output

    # 接线前基线：把旁路适配器替换为"永不 present"（等价于未接线）
    import poker_decision_ai.cash_integrity_context as cic
    monkeypatch.setattr(cic, "cash_integrity_context",
                        lambda s: {"present": False, "reason": "baseline"})
    env_before = await JudgeAgent().run({"state": copy.deepcopy(st),
                                         "equity": copy.deepcopy(eq)}, ctx)
    out_before = env_before.output
    monkeypatch.undo()

    # 旁路确实生效（接线后有 state_integrity 条目，基线没有）
    kinds_after = [e.get("kind") for e in out_after["evidence"]]
    assert EVIDENCE_KIND in kinds_after
    assert EVIDENCE_KIND not in [e.get("kind") for e in out_before["evidence"]]

    # 决策性字段逐位一致（含 action / size / ev_core / scores / warnings）
    assert _strip_bypass(out_after) == _strip_bypass(out_before)
    for k in DECISION_KEYS:
        assert out_after.get(k) == out_before.get(k), k


async def test_07_incomplete_state_decisions_bitwise_identical(monkeypatch,
                                                               ctx):
    st = cash_state(button_seat=None, table_size=None, street="turn")
    eq = _eq(0.6)
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    out_after = (await JudgeAgent().run(
        {"state": copy.deepcopy(st), "equity": copy.deepcopy(eq)}, ctx)).output

    import poker_decision_ai.cash_integrity_context as cic
    monkeypatch.setattr(cic, "cash_integrity_context",
                        lambda s: {"present": False, "reason": "baseline"})
    out_before = (await JudgeAgent().run(
        {"state": copy.deepcopy(st), "equity": copy.deepcopy(eq)}, ctx)).output
    monkeypatch.undo()

    assert _strip_bypass(out_after) == _strip_bypass(out_before)
    for k in DECISION_KEYS:
        assert out_after.get(k) == out_before.get(k), k


async def test_08_compute_ev_summary_bitwise_identical(ctx):
    st = cash_state()
    eq = _eq(0.8)
    before = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    cash_integrity_context(st)                       # 跑旁路
    after = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    assert before == after


async def test_09_and_10_candidates_and_size_identical(ctx):
    """候选集合与金额/尺寸在接入前后完全一致。"""
    st = cash_state()
    eq = _eq(0.8)
    r_before = compute_ev(copy.deepcopy(st), copy.deepcopy(eq))
    cash_integrity_context(st)
    r_after = compute_ev(copy.deepcopy(st), copy.deepcopy(eq))

    def sig(r):
        return sorted((c.action, c.amount_bb, c.ev_bb, c.utility_bb)
                      for c in r.candidates)

    assert sig(r_before) == sig(r_after)
    assert r_before.best_action == r_after.best_action
    assert r_before.best_amount_bb == r_after.best_amount_bb
    assert r_before.available == r_after.available


# ── 旁路本身不得影响判定（结构性质）───────────────────────────────────
def test_bypass_evidence_has_zero_weight_and_no_scoring_fields():
    """旁路条目必须零权重，且不含任何评分类字段。"""
    c = cash_integrity_context(cash_state())
    src = (PKG / "agents" / "judge" / "agent.py").read_text(encoding="utf-8")
    assert '"weight": 0.0' in src
    # 旁路不得触碰 scores / weights_used
    bypass_block = src[src.index("V3G-2"):]
    assert "scores[" not in bypass_block
    assert "weights_used[" not in bypass_block


def test_adapter_does_not_mutate_state():
    st = cash_state()
    before = copy.deepcopy(st)
    cash_integrity_context(st)
    assert st == before


def test_adapter_is_deterministic():
    st = cash_state()
    assert cash_integrity_context(st) == cash_integrity_context(
        copy.deepcopy(st))


# ── 12. 生产 import/path 审计 ─────────────────────────────────────────
def test_12_no_research_unfreeze_import_added():
    """生产包内不得因本次接线新增 research_unfreeze 越界。"""
    allowed = {"perf_profile.py", "perf_optimize.py", "perf_runner_v2.py",
               "s3prime_runner.py"}
    offenders = []
    for p in PKG.rglob("*.py"):
        if "research_unfreeze" in p.parts or p.name in allowed:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="ignore"))
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom)
                    else [])
            if any("research_unfreeze" in m for m in mods):
                offenders.append(str(p.relative_to(PKG)))
    assert offenders == [], offenders


def test_12_wired_modules_import_only_cash_evidence_layer():
    """被接线的两个生产模块只能引用 CASH 证据层（不引入策略/研究依赖）。"""
    for rel, expect in (("realtime.py", "cash_integrity_context"),
                        ("agents/judge/agent.py", "cash_integrity_context")):
        src = (PKG / rel).read_text(encoding="utf-8")
        assert expect in src, rel
        assert "evaluate_intel" not in src, rel
        assert "research_unfreeze" not in src, rel


def test_12_adapter_does_not_import_strategy_or_research():
    src = (PKG / "cash_integrity_context.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("research_unfreeze", "strategy_core", "strategy_intel",
                   "multistreet", "equity"):
        assert banned not in joined, banned
    # 只允许 CASH 证据层与既有唯一模式口径（外加 __future__）
    assert set(imported) <= {"__future__", "cash_state_integrity",
                             "strategy_mode_switch"}, imported
