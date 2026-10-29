"""Phase 10 (V2) Calibration 测试：Brier/LogLoss/校准曲线/可靠性 + Bluff 标签。"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from poker_decision_ai.calibration import (
    CalibrationCollector,
    brier_score,
    calibration_curve,
    label_bluff,
    log_loss,
    reliability,
)
from poker_decision_ai.replay_engine import ReplayEngine

ROOT = Path(__file__).parent.parent.parent
FULL = (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8")
SHOWDOWN = (ROOT / "examples/hands/cash_showdown_hand.txt").read_text(encoding="utf-8")


def test_brier_and_log_loss_known_values():
    # 预测 0.7，实际 1 → brier 0.09；log_loss = -ln(0.7)
    assert brier_score([0.7], [1.0]) == pytest.approx(0.09)
    assert log_loss([0.7], [1.0]) == pytest.approx(-math.log(0.7))


def test_perfect_calibration_zero_reliability():
    # 构造真正校准的样本：桶中心预测 = 桶内实际率（0.1 桶 10% 阳性；0.9 桶 90% 阳性）
    preds = [0.1] * 10 + [0.9] * 10
    actuals = [1.0] + [0.0] * 9 + [1.0] * 9 + [0.0]
    r = reliability(preds, actuals)
    assert r is not None and r < 0.01, f"reliability={r}"
    # Brier = 不可约误差（对 0.1/0.9 构造 = (0.9² + 9*0.1² + 9*0.1² + 0.9²)/20 = 0.09）
    b = brier_score(preds, actuals)
    assert b == pytest.approx(0.09)
    ll = log_loss(preds, actuals)
    assert ll is not None and math.isfinite(ll)


def test_miscalibration_detected():
    # 系统总说 0.9，实际只有 50% → 可靠性偏差大
    preds = [0.9] * 100
    actuals = [1.0] * 50 + [0.0] * 50
    r = reliability(preds, actuals)
    assert r is not None and r > 0.3


def test_calibration_curve_buckets():
    preds = [0.05, 0.15, 0.55, 0.95]
    actuals = [0.0, 0.0, 1.0, 1.0]
    curve = calibration_curve(preds, actuals, buckets=10)
    assert len(curve) == 4
    assert curve[0]["range"] == [0.0, 0.1]
    assert curve[0]["mean_actual"] == 0.0


def test_collector_skips_unlabeled_and_min_sample_note():
    c = CalibrationCollector(kind="bluff", min_samples=30)
    c.add(0.7, None)          # 无标签 → 跳过
    c.add(0.6, True)
    s = c.summary()
    assert s.n == 1 and s.labeled == 1
    assert s.kind == "bluff"
    assert "样本不足" in s.note


def test_label_bluff_deterministic():
    board = ["Qh", "7d", "2c", "3s", "Kd"]
    assert label_bluff(["Qd", "Qs"], board) is False      # set → 价值
    assert label_bluff(["Jh", "9c"], board) is True       # 高牌 → 诈唬
    assert label_bluff(["As", "Kd"], board) is False      # 一对 → 价值
    assert label_bluff([], board) is None
    assert label_bluff(["Xx", "Yy"], board) is None       # 非法牌 → 不参与


async def test_replay_collects_bluff_calibration(runtime):
    # BENCH-1 Task0.3：ReplayEngine 走 RealtimeRunner（默认 3000ms 死线）；
    # 冷启动+并行负载下偶发超时属顺序依赖脆弱失败——测试内提高死线。
    rt_cfg = dict(runtime.config.realtime_config())
    rt_cfg["deadline_ms"] = 60000
    rt_cfg["agent_timeout_ms"] = 10000
    runtime.config.root._data["realtime"] = rt_cfg
    eng = ReplayEngine(runtime)
    result = await eng.replay_hand(SHOWDOWN, hero_id="Hero")
    # Bob 摊牌 QdQs（价值）→ 其下注决策点的 bluff 预测获得标签
    assert result.calibration is not None
    assert result.calibration["kind"] == "bluff"
    assert result.calibration["labeled"] >= 1
    assert result.calibration["brier_score"] is not None
    # 样本不足提示（诚实声明）
    assert "样本不足" in result.calibration["note"]


async def test_replay_fold_hand_no_labels(runtime):
    eng = ReplayEngine(runtime)
    result = await eng.replay_hand(FULL, hero_id="Hero")
    assert result.calibration is None       # 无摊牌 → 无标签 → 无校准摘要
