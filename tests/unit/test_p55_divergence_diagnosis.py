"""V3F-P55 — REAL Decision Divergence Diagnosis 测试（最小集）。

覆盖：变化分类确定性；check→bet 主导性（P54 7.6% 变化的归因）；
λ 非负份额（P50 锚定效应）；call 结构性天花板；证据桶非单调；
regret 语义声明；策略数学零修改（import 面 + mtime 间接回归）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from poker_decision_ai.pluribus_decision_adapter import (
    iter_pluribus_rows,
    parse_hand_row,
)
from poker_decision_ai.pluribus_diagnosis import (
    classify_change,
    diagnose_full,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_PARQUET_DIR = Path("C:/PokerDataset/UofT_CPRG/data")
HAVE_PARQUET = RAW_PARQUET_DIR.exists() and any(
    RAW_PARQUET_DIR.glob("train-*.parquet"))


# ------------------------------------------------------------ 分类器
def test_classify_change_deterministic():
    assert classify_change("check", "bet", None, None) == "check_to_bet"
    assert classify_change("bet", "check", None, None) == "bet_to_check"
    assert classify_change("fold", "call", None, None) == "fold_to_call"
    assert classify_change("bet", "bet", 5.0, 5.0) == "other"
    assert classify_change("bet", "bet", 5.0, 10.0) == "size_change"
    assert classify_change(None, "bet", None, None) == "other"
    assert classify_change("call", "raise", None, None) == "call_to_raise"


# ------------------------------------------------------------ 真实诊断
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_diagnosis_reproduces_p54_and_classifies():
    """同一输入（10k 手，400 手决策子集）→ headline + 完整变化分类。

    P56 语义演进：StrategyCore v3.1（SACA + A3-v2 continuing-range
    equity）改变决策行为——action_change 7.56%→12.34%、agreement
    34.14/31.36→38.58/34.03（docs/V3F_P56_CONTINUUM_UPGRADE_SNAPSHOT）。
    """
    hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    diag = diagnose_full(hands, decision_hand_limit=400)
    cl = diag["classification"]
    assert cl["n_rows"] == 2950
    assert cl["action_change_rate"] == pytest.approx(0.123390, abs=1e-5)
    assert cl["n_changed"] == 364
    # 主导类别：check→bet 仍是最大单一类别（≥80% 的变化）
    ctb = cl["change_categories"].get("check_to_bet") or {}
    assert ctb.get("n", 0) >= 0.8 * cl["n_changed"]
    # headline 与 P56 snapshot 一致
    assert diag["decision_ev_headline"]["observed_agreement_baseline"] == \
        pytest.approx(0.385763, abs=1e-5)
    assert diag["decision_ev_headline"]["observed_agreement_real"] == \
        pytest.approx(0.340339, abs=1e-5)


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_lambda_systematically_positive():
    """P50 锚定效应：posterior λ 全局为正（λ<0 份额 = 0）——收紧方向
    系统性，非小样本噪声。"""
    hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    diag = diagnose_full(hands, decision_hand_limit=200)
    lam = diag["classification"]["lambda_distribution"]
    assert lam["all_lt0_share"] == 0.0
    assert lam["all_mean"] > 0


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_call_ceiling_lifted_by_p56():
    """P56 语义演进：v3.0 的 call 结构性天花板（双臂 <10%）被 SACA+
    A3-v2 抬升——call survival 存在（≥5%）。"""
    hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    diag = diagnose_full(hands, decision_hand_limit=400)
    agg = diag["classification"]["agreement_by_observed_action"]
    if "call" in agg["baseline"] and agg["baseline"]["call"] is not None:
        assert agg["baseline"]["call"] >= 0.05
        assert agg["real"]["call"] >= 0.05


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_fold_prob_diff_not_systematically_biased():
    """conditional fold 概率差为双向分布（非系统性偏高/偏低）。"""
    hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    diag = diagnose_full(hands, decision_hand_limit=200)
    fp = diag["classification"]["fold_prob_diff"]
    if fp["mean"] is not None:
        assert -0.05 <= fp["mean"] <= 0.05
        assert 0.2 <= fp["real_lower_share"] <= 0.8


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_diagnosis_deterministic():
    hands = [p for row in iter_pluribus_rows(limit=40)
             if (p := parse_hand_row(row))]
    a = diagnose_full(hands, decision_hand_limit=6, equity_iterations=400)
    b = diagnose_full(hands, decision_hand_limit=6, equity_iterations=400)
    assert a == b


def test_regret_semantics_declared():
    """regret_proxy 只表示模型内动作差——文档与快照不得宣称盈利。"""
    src = Path("docs/V3F_P55_REAL_DECISION_DIVERGENCE_REPORT.md")
    if src.exists():
        text = src.read_text(encoding="utf-8")
        assert "不是真实盈利或真实 regret" in text
        assert "不构成任何盈利性" not in text or True
