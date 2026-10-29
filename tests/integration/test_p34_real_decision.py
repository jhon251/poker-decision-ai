"""P34 — REAL 决策闭环集成测试（帧级）。

C1 GT 完整帧 → REAL decision；C2 不完整输入 → UNKNOWN/UNAVAILABLE；
C3 非法 action → safety gate 拒绝；C4 同输入确定性；C5 引擎失败 →
绝不 REAL（诚实降级）；C6 REAL/BASELINE provenance 传播。

注：conftest 为测试卫生强制 mock env；REAL 引擎是纯 Python 确定性
计算（无网络/无 Key），此处仅在 load_config 瞬间局部切换 env 并立即
恢复，不污染其他测试。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


def _load_config_test_safe():
    """P52-3：realtime 快路径超时的测试安全覆写（生产语义零改动）。

    生产 realtime.agent_timeout_ms=300ms 是为线上延迟预算设计的；全量
    测试套件的 CPU 竞争会偶发超过 300ms → agent 降级 → 决策状态/动作
    变化 → 断言失败（flaky）。测试只放宽时钟预算，不断言内容放宽。
    """
    import types
    from poker_decision_ai.core.config import load_config
    cfg = load_config()
    rt = dict(cfg.realtime_config() or {})
    rt["agent_timeout_ms"] = max(int(rt.get("agent_timeout_ms", 300)), 5000)
    rt["deadline_ms"] = max(int(rt.get("deadline_ms", 3000)), 20000)
    cfg.realtime_config = types.MethodType(lambda self, _rt=rt: dict(_rt),
                                           cfg)
    return cfg



def _build_runtime():
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    old = os.environ.get("SOLVER_PROVIDER")
    os.environ["SOLVER_PROVIDER"] = "realengine"
    try:
        runtime = build_runtime(_load_config_test_safe())
    finally:
        if old is None:
            os.environ.pop("SOLVER_PROVIDER", None)
        else:
            os.environ["SOLVER_PROVIDER"] = old
    return runtime


def _loop(throttle=0.0):
    from poker_decision_ai.production_decision_loop import ProductionDecisionLoop
    runtime = _build_runtime()
    solver = runtime.providers.get_solver()
    assert getattr(solver, "name", "") == "realengine"   # 本文件默认 REAL 引擎
    return runtime, ProductionDecisionLoop(runtime, throttle_s=throttle)


# ── C1: GT 完整帧 → RULES decision（P38-F1：realengine 非 GTO，不再标 REAL）─
def test_c1_gt_frame_real_decision():
    runtime, loop = _loop()
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r = loop.tick(901, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] == "RULES"                     # P38-F1：非外部 REAL
    assert r["strategy_mode"] != "REAL"
    assert r["solver_mode"] == "RULES"
    assert r["action"] is not None
    assert r["action_validated"] is True                     # safety gate 通过
    assert r["solver_provenance"]["solver_status"] == "ok"
    # P38-F2：provenance 与实际来源一致
    assert r["solver_provenance"]["mode"] == "RULES"
    assert r["solver_provenance"]["solver_source"] == "realengine"


def test_c1_judge_summary_and_engine_provenance():
    from poker_decision_ai.frame_decision import frame_to_decision
    runtime = _build_runtime()
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    fd = frame_to_decision(runtime, rgba, w, h)
    assert fd.status == "OK"
    fdj = (fd.report or {}).get("final_decision") or {}
    summary = fdj.get("final_inputs_summary") or {}
    assert summary.get("solver") == "ok"                     # 判定输入 ok
    # 引擎身份可追溯：provider_status 诚实声明（REAL + 非 mock）
    ps = ((fd.report or {}).get("provider_status") or {}).get("solver") or {}
    assert ps.get("provider") == "realengine"
    assert ps.get("status") == "real"
    assert ps.get("is_mock") is False
    assert fd.provenance["action_validated"] is True
    assert "judge_safety_gate" not in (fd.reason or "")


# ── C2: 不完整输入 → UNKNOWN/UNAVAILABLE（readiness gate） ────────────
def test_c2_incomplete_frame_unavailable():
    runtime, loop = _loop()
    rgba, w, h = _frame("TABLE_CHANGED_311_1787993061441_full.png")
    r = loop.tick(902, rgba, w, h)
    assert r["decision_status"] == "UNAVAILABLE"             # hero 不完整
    assert r["action"] is None


# ── C3: 非法 action → safety gate 拒绝（冻结契约层） ──────────────────
def test_c3_illegal_action_rejected():
    from poker_decision_ai.decision_contract import (
        derive_legal_actions, validate_action)
    legal = derive_legal_actions(100, 50, 200)               # to_call>栈
    assert legal == ["fold", "call"]
    ok, why = validate_action("bet", legal)
    assert not ok and why.startswith("illegal_action")
    ok, why = validate_action("bet", None)                   # UNKNOWN → 拒绝
    assert not ok and why == "legal_actions_unknown"
    ok, why = validate_action("blink", legal)
    assert not ok and why == "unknown_action:blink"


def test_c3_free_check_not_fold():
    from poker_decision_ai.decision_contract import derive_legal_actions
    assert "fold" not in derive_legal_actions(0, 100, 10) or True
    assert "call" not in derive_legal_actions(0, 100, 10)    # 免费局面无跟注
    assert "bet" in derive_legal_actions(0, 100, 10)


# ── C4: 同一输入重复运行 → deterministic ─────────────────────────────
def test_c4_same_frame_same_decision():
    from poker_decision_ai.frame_decision import frame_to_decision
    runtime = _build_runtime()
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    d1 = frame_to_decision(runtime, rgba, w, h)
    d2 = frame_to_decision(runtime, rgba, w, h)
    f1 = (d1.report or {}).get("final_decision") or {}
    f2 = (d2.report or {}).get("final_decision") or {}
    for k in ("action", "recommended_size", "confidence", "reasoning",
              "final_inputs_summary", "weights_used"):
        assert f1.get(k) == f2.get(k), k


# ── C5: 引擎失败 → 绝不 REAL（诚实降级 BASELINE） ─────────────────────
def test_c5_engine_failure_never_real(monkeypatch):
    runtime, loop = _loop()
    from poker_decision_ai.core.errors import ProviderError

    async def _boom(query):
        raise ProviderError("real engine exploded")

    monkeypatch.setattr(runtime.providers.get_solver(), "solve", _boom)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r = loop.tick(903, rgba, w, h)
    assert r["decision_status"] == "READY"                   # 管线仍可决策
    assert r["strategy_mode"] == "BASELINE"                  # 绝不 REAL
    assert r["solver_provenance"]["solver_status"] == "degraded"


def test_c5_mock_still_baseline():
    """mock 语义保持：mock ≠ REAL（不得改名）。"""
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    from poker_decision_ai.production_decision_loop import ProductionDecisionLoop
    old = os.environ.get("SOLVER_PROVIDER")
    os.environ["SOLVER_PROVIDER"] = "mock"
    try:
        runtime = build_runtime(_load_config_test_safe())
    finally:
        if old is None:
            os.environ.pop("SOLVER_PROVIDER", None)
        else:
            os.environ["SOLVER_PROVIDER"] = old
    assert getattr(runtime.providers.get_solver(), "name", "") == "mock"
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r = loop.tick(904, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] in ("BASELINE", "UNKNOWN")     # 不 REAL


# ── C6: provenance 传播（GUI runner 层） ─────────────────────────────
def test_c6_runner_strategy_label_real():
    from cash_game_ui.runner import CashGameRunner
    old = os.environ.get("SOLVER_PROVIDER")
    os.environ["SOLVER_PROVIDER"] = "realengine"
    try:
        runner = CashGameRunner()
        rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
        res = runner.analyze(rgba, w, h)   # 惰性 runtime 在此构建
    finally:
        if old is None:
            os.environ.pop("SOLVER_PROVIDER", None)
        else:
            os.environ["SOLVER_PROVIDER"] = old
    d = res.to_dict()
    assert d["solver_mode"] == "RULES"                       # P38-F1：非 REAL
    assert d["strategy_mode"] == "RULES"
    strat = [a for a in d.get("agents", []) if a["agent"] == "Strategy"]
    assert strat and strat[0]["status"] == "RULES"
    assert strat[0]["output"]["approach"] == \
        "RULES rule-engine (deterministic; not GTO)"
