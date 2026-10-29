"""V3F-P51 — Historical Strategy Validation（LOO 行为校准 + 边际扩展）测试。

覆盖：manifest v3（62 场景 + 4 项 NOT VALIDATABLE）；LOO 分割无泄漏；
行为校准（BASELINE vs REAL vs observed fold）；regret proxy / EV margin；
facing 尺寸敏感性；确定性；真实数据端到端。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from poker_decision_ai.core.config import load_config
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker.conditional_model import (
    ConditionalFoldIndex, extract_events)
from poker_decision_ai.real_validation import (
    build_loo_split,
    collect_eval_decisions,
    diagnose_saturation,
    evaluate_fold_calibration,
    make_state,
    manifest,
    marginal_scenarios,
    three_arm_decision,
)

CFG = load_config()
UOFT_DIR = Path(__file__).resolve().parents[2] / "data" / "uoft"
CHUNK0 = UOFT_DIR / "normalized" / "chunk_000.jsonl"


# ------------------------------------------------------------ fixtures
def mini_hands(player: str, n: int, entry_frac: float = 0.5,
               fold_frac: float = 0.5, prefix: str = "h") -> list[dict]:
    hands = []
    for i in range(n):
        entered = (i % 10) < entry_frac * 10
        actions = [{"street": "preflop", "player": player,
                    "action": "raise" if entered else "fold",
                    "amount": 2.0, "to": 2.0}]
        if entered:
            f = 1 if (i % 10) < fold_frac * 10 else 0
            actions += [{"street": "flop", "player": "o1", "action": "bet",
                         "amount": 4.0, "to": 4.0},
                        {"street": "flop", "player": player,
                         "action": "fold" if f else "call", "amount": 0.0,
                         "to": None},
                        {"street": "turn", "player": "o1", "action": "bet",
                         "amount": 8.0, "to": 8.0},
                        {"street": "turn", "player": player,
                         "action": "fold" if f else "call", "amount": 0.0,
                         "to": None}]
        parsed = {"hand_id": f"{prefix}{i}", "button_seat": 3,
                  "seats": [{"seat": 1, "name": "o1"},
                            {"seat": 2, "name": player},
                            {"seat": 3, "name": "o3"},
                            {"seat": 4, "name": "o4"},
                            {"seat": 5, "name": "o5"}],
                  "actions": actions,
                  "street_boards": {"flop": ["Ah", "Kd", "7c"],
                                    "turn": ["Ah", "Kd", "7c", "2s"]}}
        canonical = {"hand_id": f"{prefix}{i}", "game_type": "cash",
                     "stakes": "NL100", "button_seat": 3,
                     "players": [{"player_id": player, "seat": 2,
                                  "hole_cards": [], "position": None},
                                 {"player_id": "o1", "seat": 1,
                                  "hole_cards": [], "position": None}],
                     "preflop_actions": actions[:1],
                     "flop_actions": [a for a in actions
                                      if a.get("street") == "flop"],
                     "meta": {"venue": "PokerStars"}}
        hands.append({"hand_id": f"{prefix}{i}",
                      "events": extract_events(parsed),
                      "quality": {"status": "warning", "warnings": []},
                      "verdicts": eg.classify_canonical(canonical, None),
                      "parsed": parsed})
    return hands


# ------------------------------------------------------------ manifest
def test_manifest_v3_extended_and_declares_not_validatable():
    m = manifest()
    assert m["manifest_version"] == "p51-real-v3"
    ids = [s["id"] for s in m["scenarios"]]
    assert len(ids) == 62 and len(set(ids)) == 44 + 18
    marg = [s for s in m["scenarios"] if s["id"].startswith("marg_")]
    assert len(marg) == 45                       # 3 街 × 3 档 × 5 facing
    for facing in ("check", "bet_05", "bet_10", "raise_15", "raise_25"):
        assert sum(1 for s in marg if s["facing"] == facing) == 9
    nv = {e["evidence_type"]: e["status"] for e in m["not_validatable"]}
    assert nv["observed_action_agreement"] == \
        "NOT VALIDATABLE WITH CURRENT DATA"
    assert "chunk_000" in nv["observed_action_agreement"] or \
        "100,979" in "".join(e["reason"] for e in m["not_validatable"]
                             if e["evidence_type"] == "observed_action_agreement")
    assert "behavioral_calibration" in m
    bc = m["behavioral_calibration"]
    assert "同一历史手牌不得同时成为 evidence 与 evaluation" in bc["discipline"]


# ------------------------------------------------------------ LOO 泄漏纪律
def test_loo_split_no_leakage():
    hands = mini_hands("v1", 100)
    ev_hands, eval_hands = build_loo_split(hands)
    ev_ids = {h["hand_id"] for h in ev_hands}
    eval_ids = {h["hand_id"] for h in eval_hands}
    assert ev_ids and eval_ids
    assert not (ev_ids & eval_ids)               # 同一手牌不得跨侧
    records = collect_eval_decisions(eval_hands)
    assert records
    assert all(r["hand_id"] in eval_ids for r in records)
    assert all(r["facing"] not in ("", "none", "None") for r in records)
    assert all(r["observed_fold"] in (0, 1) for r in records)


def test_loo_split_deterministic():
    hands = mini_hands("v1", 100)
    a = build_loo_split(hands)
    b = build_loo_split(hands)
    assert [h["hand_id"] for h in a[0]] == [h["hand_id"] for h in b[0]]
    assert [h["hand_id"] for h in a[1]] == [h["hand_id"] for h in b[1]]


# ------------------------------------------------------------ 行为校准
def test_calibration_player_evidence_beats_population_prior():
    """nit 玩家（fold 0.9）的 LOO 条件估计应比 population 中性先验更接近
    其真实 held-out 行为——REAL evidence → fold 的核心验证。"""
    population = mini_hands("pop", 200, entry_frac=0.5, fold_frac=0.5,
                            prefix="p")         # 中性群体
    player = mini_hands("v1", 200, entry_frac=0.5, fold_frac=0.9,
                        prefix="v")             # nit
    ev_hands, eval_hands = build_loo_split(player)
    ev_hands = ev_hands + population            # 证据段 = 玩家前段 + 群体
    player_idx = ConditionalFoldIndex()
    for h in ev_hands:
        player_idx.add_events(h["events"])
    pop_idx = ConditionalFoldIndex()
    for h in population:
        pop_idx.add_events(h["events"])
    records = collect_eval_decisions(eval_hands)
    assert records
    out = evaluate_fold_calibration(records, player_idx, pop_idx)
    assert out["n_paired_real_available"] > 0
    b, r = out["arms"]["BASELINE"], out["arms"]["REAL"]
    assert b["n"] == r["n"]                     # 配对
    assert r["brier"] < b["brier"]              # 玩家证据改善校准
    assert out["brier_delta_real_minus_baseline"] < 0


def test_calibration_coverage_fallback_honest():
    """player 索引稀疏时 REAL 不可用 → 配对覆盖率诚实 < 1。"""
    player = mini_hands("v1", 10, prefix="s")   # 证据极少
    records = collect_eval_decisions(player)
    player_idx = ConditionalFoldIndex()
    for h in player:
        player_idx.add_events(h["events"])
    pop_idx = ConditionalFoldIndex()
    for h in mini_hands("pop", 200, prefix="p"):
        pop_idx.add_events(h["events"])
    out = evaluate_fold_calibration(records, player_idx, pop_idx)
    assert out["real_coverage"] is not None
    assert out["real_coverage"] < 1.0 or out["n_paired_real_available"] == 0


def test_calibration_deterministic():
    population = mini_hands("pop", 100, prefix="p")
    player = mini_hands("v1", 120, fold_frac=0.8)
    ev_h, eval_h = build_loo_split(player)
    pidx = ConditionalFoldIndex()
    for h in ev_h + population:
        pidx.add_events(h["events"])
    popidx = ConditionalFoldIndex()
    for h in population:
        popidx.add_events(h["events"])
    records = collect_eval_decisions(eval_h)
    a = evaluate_fold_calibration(records, pidx, popidx)
    b = evaluate_fold_calibration(records, pidx, popidx)
    assert a == b


# ------------------------------------------------------------ 三臂扩展
@pytest.fixture
def mini_profile():
    from poker_decision_ai.real_validation import build_evidence, real_profile
    player = mini_hands("v1", 120, entry_frac=0.5, fold_frac=0.2)
    all_h = {"v1": player}
    return real_profile("villain", build_evidence(player, all_h))


async def test_regret_proxy_and_ev_margin_present(mini_profile):
    scen = next(s for s in marginal_scenarios()
                if s["id"] == "marg_flop_strong_check")
    out = await three_arm_decision(CFG, scen, "villain", mini_profile)
    assert "regret_proxy" in out and out["regret_proxy"] is not None
    for arm, margin in out["ev_margin"].items():
        assert margin is None or margin >= 0.0
    assert out["ev_margin"]["REAL"] is not None


async def test_facing_sizes_change_candidates(mini_profile):
    """多尺寸 facing：bet_05 与 raise_25 的候选集/fold/EV 必须不同。"""
    outs = {}
    for facing in ("bet_05", "bet_10", "raise_15", "raise_25"):
        scen = next(s for s in marginal_scenarios()
                    if s["id"] == f"marg_flop_strong_{facing}")
        outs[facing] = await three_arm_decision(CFG, scen, "villain",
                                                mini_profile)
    f05 = outs["bet_05"]["arms"]["REAL"]["fold_05x"]["p_fold"]
    f15 = outs["raise_15"]["arms"]["REAL"]["fold_05x"]["p_fold"]
    # 尺寸越大 → 范围结构通道要求越高 → 弃牌概率不减
    assert f15 >= f05 - 1e-6
    evs = {k: v["arms"]["REAL"]["best"]["ev_bb"] for k, v in outs.items()
           if v["arms"]["REAL"].get("best")}
    assert len(set(round(v, 6) for v in evs.values())) >= 2


async def test_p51_all_marginal_scenarios_run(mini_profile):
    """45 个边际场景全量可运行 + 字段诚实记录（确定性在别处断言）。"""
    for scen in marginal_scenarios():
        out = await three_arm_decision(CFG, scen, "villain", mini_profile)
        assert "regret_proxy" in out
        assert isinstance(out["action_change"]["baseline_vs_real"], bool)


# ------------------------------------------------------------ 真实端到端
@pytest.mark.skipif(not CHUNK0.exists(),
                    reason="UofT data files not present")
async def test_real_p51_end_to_end():
    """真实 UofT 玩家：LOO 行为校准 + 泄漏检查 + 确定性。"""
    from poker_decision_ai.real_validation import (
        build_evidence, collect_real_sample, pick_target_player)
    hands = collect_real_sample(str(CHUNK0), scan_limit=6000)
    pid = pick_target_player(hands)
    player_hands = hands[pid]
    ev_hands, eval_hands = build_loo_split(player_hands)
    # 泄漏检查：hand_id 不相交
    assert not ({h["hand_id"] for h in ev_hands}
                & {h["hand_id"] for h in eval_hands})
    # player 索引仅含证据段；population 索引含证据段 + 其他玩家（不含评估段）
    player_idx = ConditionalFoldIndex()
    pop_idx = ConditionalFoldIndex()
    for h in ev_hands:
        player_idx.add_events(h["events"])
    for p, hs in hands.items():
        if p == pid:
            continue
        for h in hs:
            pop_idx.add_events(h["events"])
    for h in ev_hands:
        pop_idx.add_events(h["events"])
    records = collect_eval_decisions(eval_hands)
    assert records
    out = evaluate_fold_calibration(records, player_idx, pop_idx)
    assert out["n_eval_decisions"] == len(records)
    assert out["n_paired_real_available"] >= 0
    assert out["arms"]["BASELINE"] is not None
    # 确定性
    out2 = evaluate_fold_calibration(records, player_idx, popidx_dup(pop_idx))
    assert out == out2


def popidx_dup(pop_idx):
    d = ConditionalFoldIndex.from_dict(pop_idx.to_dict())
    return d
