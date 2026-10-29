"""V3F-P36 — 连续生产闭环组装层测试。

覆盖：连续多帧正常 · 重复帧/decision_id 幂等 · Hero UNKNOWN · Dealer
UNKNOWN · Position UNKNOWN · solver 未配置（诚实 BASELINE）· solver 异常 ·
非法 action 拒绝 · 安全停止 · GUI 单帧回归 · provenance/无自动下注。
冻结组件零修改：FrameStream/EventStreamConsumer/ProductionDecisionLoop/
bridge/RealtimeRunner 全部按原语义消费。
"""
from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

F_OK = "8bf82f134651db0add626be0304a2ff1.png"          # 5c3c，action=call
F_HERO_UNKNOWN = "BUTTON_CHANGED_167_1787992682749_full.png"
F_POS_UNKNOWN = "TABLE_CHANGED_311_1787993061441_full.png"
F_DEALER_UNKNOWN = "SCENE_CHANGED_37_1787487606667_full.png"


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


def _runner(**kw):
    """测试 runner：throttle_s=0.0 —— 冻结 _state_key 仅采样帧前 4096 字节，
    不同 evidence 帧共享前缀；throttle>0 时 warm 环境下相邻帧会被
    同状态节流窗吞掉（冻结语义，生产默认 1.0 保留）。"""
    kw.setdefault("throttle_s", 0.0)
    from poker_decision_ai.continuous_runner import ContinuousProductionRunner
    from poker_decision_ai.core.config import load_config
    return ContinuousProductionRunner(load_config(), **kw)


# ── 1. 连续多帧正常运行 ─────────────────────────────────────────────
def test_multi_frame_normal_run():
    r = _runner(fps=100.0)
    outs = []
    for fid in (F_OK, F_HERO_UNKNOWN, F_POS_UNKNOWN):
        rgba, w, h = _frame(fid)
        outs.append(r.process_image(rgba, w=w, h=h))
    assert outs[0]["decision_status"] == "READY"
    assert outs[1]["decision_status"] in ("UNAVAILABLE", "READY")
    assert all("decision_id" in o and o["decision_id"] for o in outs)
    assert all(o.get("provenance") is not None for o in outs)
    assert r.counters["frames_processed"] == 3
    assert len(r.loop.decisions) == 3
    s = r.summary()
    assert s["counters"]["frames_in"] == 3
    # 无自动下注：结果只是建议记录，不含任何执行器
    assert all("executor" not in o for o in outs)


# ── 2. 重复帧 / decision_id 幂等 ────────────────────────────────────
def test_duplicate_frame_dedup():
    r = _runner(fps=100.0)
    rgba, w, h = _frame(F_OK)
    r1 = r.process_image(rgba, w=w, h=h)
    r2 = r.process_image(rgba, w=w, h=h)          # 同画面 → FrameStream 去重
    assert r1["decision_status"] == "READY"
    assert r2["status"] == "FRAME_SKIPPED"
    assert r2["reason"] == "duplicate_frame"
    assert r.acquirer.metrics.frames_deduplicated == 1
    assert len(r.loop.decisions) == 1             # 不重复决策


def test_decision_id_idempotent_via_loop():
    from poker_decision_ai.production_decision_loop import ProductionDecisionLoop
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame(F_OK)
    a = loop.tick(700, rgba, w, h)
    b = loop.tick(700, rgba, w, h)
    assert a["decision_status"] == "READY"
    assert b["decision_id"] == a["decision_id"]   # 同 decision_id 幂等
    assert b["decision_status"] == "READY"        # 返回上次结果（不重复 decide）
    assert len(loop.decisions) == 1


# ── 3. Hero UNKNOWN 阻断（无默认手牌/空列表替代） ───────────────────
def test_hero_unknown_blocks_no_default():
    r = _runner(fps=100.0)
    rgba, w, h = _frame(F_HERO_UNKNOWN)
    out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] == "UNAVAILABLE"
    assert out["action"] is None
    # V-1 FM-1：该帧现于 street 层即被诚实拒绝（板区仅识别 2 卡 →
    # NEW_HAND 不可表示）——warning 为 street_new_hand_unconfirmed
    # V-2 更新（test-side only）：召回修复后该帧（目视 GT 4 卡）可计数控 4 / FULL 3 → 拒绝原因 street_board_contradiction（诚实拒绝语义不变）。
    assert ("street_new_hand_unconfirmed" in (out["warning"] or "")
            or "street_board_contradiction" in (out["warning"] or ""))
    # 桥层语义：hole_cards=None（非 [] 非伪造）
    rgba2, w2, h2 = _frame(F_HERO_UNKNOWN)
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
        frame_to_decision_request)
    br = frame_to_decision_request(rgba2, w2, h2)
    # V-1 FM-1：该帧 street 层结构性拒绝（observation=None——hole_cards
    # 不再可及；结构性拒绝严格强于 DECISION_UNAVAILABLE）
    assert br["status"] == "UNAVAILABLE"
    assert br["bridge"]["observation"] is None


# ── 4. Dealer UNKNOWN 阻断 ──────────────────────────────────────────
def test_dealer_unknown_blocks():
    r = _runner(fps=100.0)
    rgba, w, h = _frame(F_DEALER_UNKNOWN)
    out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] == "UNAVAILABLE"
    assert out["action"] is None
    # dealer 证据独立：该帧 dealer 置信 0（不借用 hero/position 反推）
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (
        frame_to_decision_request)
    rgba2, w2, h2 = _frame(F_DEALER_UNKNOWN)
    br = frame_to_decision_request(rgba2, w2, h2)
    assert br["bridge"]["field_confidences"]["dealer"] == 0.0


# ── 5. Position UNKNOWN 阻断 ────────────────────────────────────────
def test_position_unknown_blocks():
    r = _runner(fps=100.0)
    rgba, w, h = _frame(F_POS_UNKNOWN)
    out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] == "UNAVAILABLE"
    assert out["action"] is None
    assert "hero_position" in (out["warning"] or "")


# ── 6. REAL solver 未配置 → 诚实 BASELINE ───────────────────────────
def test_solver_unconfigured_honest_baseline():
    from poker_decision_ai.continuous_runner import ContinuousProductionRunner
    from poker_decision_ai.core.config import load_config, NestedDict
    cfg = load_config()
    # path 未设置 → 未配置（read-only NestedDict → merged 覆盖）
    cfg.root = cfg.root.merged(NestedDict({"solver": {"provider": "postflop"}}))
    r = ContinuousProductionRunner(cfg, fps=100.0)
    rgba, w, h = _frame(F_OK)
    out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] in ("READY", "UNAVAILABLE")
    assert out["solver_mode"] != "REAL"               # 绝不伪装 REAL
    assert out["strategy_mode"] != "REAL"


# ── 7. solver 异常 → 降级不崩溃 ─────────────────────────────────────
def test_solver_exception_degrades():
    from poker_decision_ai.continuous_runner import ContinuousProductionRunner
    from poker_decision_ai.core.config import load_config, NestedDict
    from poker_decision_ai.core.providers.solver import (
        PostflopSolverCLIProvider, ProviderError)
    cfg = load_config()
    cfg.root = cfg.root.merged(NestedDict({
        "solver": {"provider": "postflop",
                   "path": "C:/__nonexistent__/postflopsolver.exe"}}))
    r = ContinuousProductionRunner(cfg, fps=100.0)
    with mock.patch.object(PostflopSolverCLIProvider, "solve",
                           side_effect=ProviderError("boom")):
        rgba, w, h = _frame(F_OK)
        out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] in ("READY", "UNAVAILABLE")
    assert out["solver_mode"] != "REAL"               # 失败不伪装成功
    assert out["strategy_mode"] != "REAL"


# ── 8. 非法 action → Judge gate 拒绝 ────────────────────────────────
def test_illegal_action_rejected_by_judge_gate():
    r = _runner(fps=100.0)

    class FakeReport:
        def to_dict(self):
            return {"final_decision": {"action": "bet_the_moon",
                                       "amount": None},
                    "agents_ran": [], "degraded": [], "skipped": []}

    async def fake_decide(self, request):
        return FakeReport()

    from poker_decision_ai import realtime as rt
    with mock.patch.object(rt.RealtimeRunner, "decide", fake_decide):
        rgba, w, h = _frame(F_OK)
        out = r.process_image(rgba, w=w, h=h)
    assert out["decision_status"] == "UNAVAILABLE"
    assert out["action_validated"] is False           # Judge gate 未放行
    assert "judge_safety_gate" in (out["warning"] or "")


# ── 9. 安全停止 ─────────────────────────────────────────────────────
def test_safe_stop_mid_run():
    r = _runner(fps=100.0)
    seen = []

    def sink(res):
        seen.append(res["decision_id"])
        if len(seen) >= 2:
            r.stop()                                   # 消费端触发安全停止

    r.sink = sink
    frames = [_frame(f) for f in (F_OK, F_POS_UNKNOWN,
                                  F_HERO_UNKNOWN, F_OK)]
    r.run_replay([f[0] for f in frames],               # rgba（同尺寸帧）
                 w=frames[0][1], h=frames[0][2])
    assert len(seen) == 2
    assert r.stopped
    assert r.counters["frames_processed"] == 2
    rgba, w, h = _frame(F_OK)
    out = r.process_image(rgba, w=w, h=h)
    assert out["status"] == "STOPPED"


def test_safe_stop_before_start():
    r = _runner(fps=100.0)
    r.stop()
    rgba, w, h = _frame(F_OK)
    out = r.process_image(rgba, w=w, h=h)
    assert out["status"] == "STOPPED"
    assert r.counters["frames_processed"] == 0


# ── 10. GUI 单帧入口回归（兼容性） ──────────────────────────────────
def test_gui_single_frame_entry_compatible():
    from cash_game_ui.runner import CashGameRunner
    runner = CashGameRunner()
    rgba, w, h = _frame(F_OK)
    res = runner.analyze(rgba, w, h).to_dict()
    assert res["status"] in ("OK", "DECISION_UNAVAILABLE", "UNAVAILABLE")
    if res["status"] == "OK":
        assert res["decision_action"] == "call"
        assert res["vision"].get("hero_cards") == ["5c", "3c"]


# ── 11. 采集后端不可用 → 明确状态（不伪装帧） ────────────────────────
def test_capture_backend_unavailable_honest():
    from poker_decision_ai.continuous_runner import ContinuousProductionRunner
    from poker_decision_ai.core.config import load_config
    r = ContinuousProductionRunner(load_config(), fps=100.0, backend=None)
    out = r.process_capture()
    assert out["status"] == "CAPTURE_BACKEND_UNAVAILABLE"
    assert r.counters["frames_processed"] == 0        # 不伪装新帧


# ── 12. 视觉事实流不产生行为事件（来源边界） ────────────────────────
def test_visual_fact_only_no_behavior():
    r = _runner(fps=100.0)
    for fid in (F_OK, F_POS_UNKNOWN):
        rgba, w, h = _frame(fid)
        r.process_image(rgba, w=w, h=h)
    s = r.event_consumer.summary()
    assert s["behavior_events"] == 0                  # 视觉事实 ≠ 行为归因
    assert s["source"]["status"] != "RECOGNIZED" or s["source"]["source"] != "HAND_HISTORY"
