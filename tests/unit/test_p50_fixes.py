"""V3F-P50 — F1/F2 修复专项回归测试。

F1: fold_equity 融合权重兼容两种 provenance 形态（P42 顶层 samples /
P43 nested conditional.samples）；F2: P46 continuing 校准 hero-agnostic
（同一行为证据 + 不同 hero 手牌 → λ 不漂移，card removal 噪声除外）。
"""
from __future__ import annotations

import pytest

from poker_decision_ai.domain.poker.conditional_model import (
    ConditionalFoldIndex)
from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
from poker_decision_ai.domain.poker.range_posterior import (
    build_range_posterior)
from poker_decision_ai.domain.poker.ranges import parse_range

BOARD = ["Ah", "Kd", "7c"]
HERO_A = ["Ah", "Kd"]          # 与 board 重合（强牌）
HERO_B = ["6c", "5c"]          # 无对无听牌（弱牌）


def _cond_behavior(fold_frac: float, n: int = 60) -> dict:
    """P43 形态的条件行为（nested conditional.samples）。"""
    cfi = ConditionalFoldIndex()
    for i in range(n):
        f = 1 if (i % 100) / 100.0 < fold_frac else 0
        cfi.add_events([{"player": "v", "street": "flop", "facing": "bet",
                         "size_bucket": "m<=1.0", "position": "CO",
                         "texture": "wet",
                         "action": "fold" if f else "call", "fold": f,
                         "hand_id": f"c{i}"}])
    return {"conditional_model": cfi.to_dict()}


# ------------------------------------------------------------ F1 回归
def test_f1_nested_provenance_carries_weight():
    """P43 nested conditional.samples → w_stat > 0（修复前恒为 0）。"""
    fe = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                       villain_dist={"AKs": 0.5, "QQ": 0.5},
                       behavior=_cond_behavior(0.4),
                       hero_cards=HERO_A, board=BOARD, seed=42)
    r = fe.for_size(5.0)
    assert r.available and r.mode == "stat+range"
    assert r.provenance["w_stat"] > 0.0
    assert r.provenance["source"] == "conditional_player"


def test_f1_top_level_provenance_still_weighted():
    """P42 顶层 samples 路径（legacy global stat）不受修复影响。"""
    behavior = {"behavior_statistics": {
        "fold_to_cbet": {"value": 0.4, "samples": 60}}}
    fe = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                       villain_dist={"AKs": 0.5, "QQ": 0.5},
                       behavior=behavior, hero_cards=HERO_A, board=BOARD,
                       seed=42)
    r = fe.for_size(5.0)
    assert r.available
    # 范围通道同时可用 → stat+range；顶层 samples 路径权重正常
    assert r.mode == "stat+range"
    assert r.provenance["w_stat"] == pytest.approx(60 / 160)
    assert r.provenance["source"] == "global"


def test_f1_same_stat_same_weight_both_shapes():
    """同一 (raw, n) 的两种 provenance 形态 → 相同 w_stat 与融合结果。"""
    dist = {"AKs": 0.5, "QQ": 0.5}
    nested = _cond_behavior(0.4)
    # nested 路径: estimate raw≈0.4, n=60 → w=60/160
    fe1 = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                        villain_dist=dist, behavior=nested,
                        hero_cards=HERO_A, board=BOARD, seed=42)
    r1 = fe1.for_size(5.0)
    # 顶层路径: 相同 raw/n 的全局统计（facing=bet → fold_to_cbet）
    top = {"behavior_statistics": {"fold_to_cbet": {"value": 0.4,
                                                    "samples": 60}}}
    fe2 = FoldEstimator(pot_bb=10.0, facing="bet", street="flop",
                        villain_dist=dist, behavior=top,
                        hero_cards=HERO_A, board=BOARD, seed=42)
    r2 = fe2.for_size(5.0)
    n_nested = r1.provenance["stat"]["conditional"]["samples"]
    assert r1.provenance["w_stat"] == pytest.approx(n_nested / (n_nested + 100))
    assert r2.provenance["w_stat"] == pytest.approx(60 / 160)
    # 两形态的 stat 通道给出的都是 (0.4, 60) → 融合权重一致
    assert r1.provenance["w_stat"] == pytest.approx(
        r2.provenance["w_stat"])


# ------------------------------------------------------------ F2 回归
def _continuing_evidence(fold_frac: float, n: int = 200) -> dict:
    cfi = ConditionalFoldIndex()
    for i in range(n):
        f = 1 if (i % 100) / 100.0 < fold_frac else 0
        cfi.add_events([{"player": "v", "street": "flop", "facing": "bet",
                         "size_bucket": "m<=1.0", "position": "CO",
                         "texture": "wet",
                         "action": "fold" if f else "call", "fold": f,
                         "hand_id": f"e{i}"}])
    return cfi.to_dict()


def test_f2_continuing_lambda_hero_invariant():
    """同一 villain 证据 + 不同 hero 手牌 → λ 与 baseline continue 不漂移
    （P50-2 修复前 λ 摆动 >0.4；修复后仅剩 card removal 噪声）。"""
    base = parse_range(
        "AA-22,AKs-ATs,KQs-K9s,Q9s+,J9s+,T9s,98s,87s,76s,65s,AKo-AJo,KQo")
    live = base.remove_cards([])
    behavior = _continuing_evidence(0.5)
    lams, bases = [], []
    for hero in (HERO_A, ["Qc", "Jc"], HERO_B):
        _, res = build_range_posterior(base, live, kind="continuing",
                                       position="CO", street="flop",
                                       hero_cards=hero, board=BOARD,
                                       cond_player=behavior)
        assert res.available
        lams.append(res.tilt["lambda"])
        bases.append(res.target["baseline_implied"])
    assert max(lams) - min(lams) <= 0.1
    assert max(bases) - min(bases) <= 0.1      # card removal 噪声量级
    # 行为校准仍随真实证据方向走：fold 50% << 理性继续基线 → 收紧
    assert all(l > 0 for l in lams)


def test_f2_continuing_directions_preserved():
    """P46 方向语义在 hero-agnostic 校准下保持：极端 nit 收紧、
    极端 station 放宽。"""
    base = parse_range(
        "AA-22,AKs-ATs,KQs-K9s,Q9s+,J9s+,T9s,98s,87s,76s,65s,AKo-AJo,KQo")
    live = base.remove_cards([])
    _, nit = build_range_posterior(base, live, kind="continuing",
                                   position="CO", street="flop",
                                   hero_cards=HERO_A, board=BOARD,
                                   cond_player=_continuing_evidence(0.95))
    _, station = build_range_posterior(base, live, kind="continuing",
                                       position="CO", street="flop",
                                       hero_cards=HERO_A, board=BOARD,
                                       cond_player=_continuing_evidence(0.01))
    assert nit.tilt["lambda"] > 0            # 继续率 << 基线 → 收紧
    assert station.tilt["lambda"] < 0        # 继续率 > 基线（99%）→ 放宽
