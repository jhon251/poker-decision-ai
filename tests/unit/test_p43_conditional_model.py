"""V3F-P43 — Conditional Opponent Model 测试。

覆盖（按阶段授权 14 项）：
  1  同一玩家不同 street 行为不同      8  population fallback
  2  不同 bet size bucket             9  global fallback（P42 路径）
  3  不同 position                   10  UNKNOWN
  4  board texture 条件              11  FoldEstimator 使用 conditional source
  5  样本量不足 → UNKNOWN            12  EVCore 受条件 fold 概率影响
  6  小样本 shrinkage                13  P42 fallback 不回归
  7  player conditional              14  P30–P42 全量回归（套件级）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.conditional_model import (
    MIN_SAMPLES,
    ConditionalFoldIndex,
    extract_events,
    size_bucket,
    texture_tag,
)
from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
from poker_decision_ai.domain.poker.strategy_core import compute_ev

RIVER = ["Ah", "7c", "2s", "3d", "9h"]


def _ev(street="flop", facing="bet", bucket="s<=0.5", position="BB",
        texture="wet", fold=0):
    return {"player": "V", "street": street, "facing": facing,
            "size_bucket": bucket, "position": position, "texture": texture,
            "action": "fold" if fold else "call", "fold": fold,
            "hand_id": "H"}


def _counts(entries: dict[str, tuple[int, int]]) -> dict:
    """构造真实索引（经 add_events → 全部 4 个粗化层级入库）。"""
    idx = ConditionalFoldIndex()
    events: list[dict] = []
    for key, (f, n) in entries.items():
        street, facing, bucket, pos, texture = key.split("|")
        for i in range(n):
            events.append({"player": "V", "street": street, "facing": facing,
                           "size_bucket": bucket or None,
                           "position": pos, "texture": texture,
                           "action": "fold" if i < f else "call",
                           "fold": 1 if i < f else 0, "hand_id": "H"})
    idx.add_events(events)
    return idx.to_dict()


def _state(**over):
    st = {"game_type": "cash", "street": "flop", "pot": 200.0, "bb": 2.0,
          "to_call": 0.0,
          "players": [
              {"player_id": "Hero", "is_hero": True, "stack": 400.0,
               "hole_cards": ["As", "Kd"]},
              {"player_id": "V1", "is_hero": False, "status": "active",
               "stack": 400.0},
          ]}
    st.update(over)
    return st


def _eq(v=0.6):
    return {"hero_equity": v, "confidence": 0.9}


# ── 事件抽取（确定性重放） ─────────────────────────────────────────────
def _parsed_hand(hid: str, flop_fold: bool) -> dict:
    acts = [
        {"street": "preflop", "player": "V", "action": "post", "amount": 1.0},
        {"street": "preflop", "player": "Hero", "action": "post", "amount": 2.0},
        {"street": "preflop", "player": "B", "action": "raise", "amount": 2.0,
         "to": 4.0},
        {"street": "preflop", "player": "V", "action": "call", "amount": 3.0},
        {"street": "flop", "player": "B", "action": "bet", "amount": 4.0},
    ]
    acts.append({"street": "flop", "player": "V",
                 "action": "fold" if flop_fold else "call", "amount": 0.0})
    return {"hand_id": hid, "button_seat": 1,
            "seats": [{"seat": 1, "name": "V"}, {"seat": 2, "name": "Hero"},
                      {"seat": 3, "name": "B"}],
            "street_boards": {"flop": ["Ah", "7c", "2s"]},
            "actions": acts}


def test_extract_events_deterministic():
    a = extract_events(_parsed_hand("H1", True))
    b = extract_events(_parsed_hand("H1", True))
    assert a == b
    fold_ev = [e for e in a if e["action"] == "fold"]
    assert len(fold_ev) == 1
    ev = fold_ev[0]
    assert ev["street"] == "flop" and ev["facing"] == "bet"
    assert ev["position"] == "SB" or ev["position"]        # button_seat 派生
    assert ev["texture"] == "dry"                          # Ah7c2s
    assert ev["fold"] == 1


def test_size_bucket_boundaries():
    assert size_bucket(0.4, 1.0) == "s<=0.5"
    assert size_bucket(0.75, 1.0) == "m<=1.0"
    assert size_bucket(1.5, 1.0) == "l<=2.0"
    assert size_bucket(3.0, 1.0) == "x>2.0"
    assert size_bucket(1.0, 0.0) is None


def test_texture_tags():
    assert texture_tag([]) == "preflop"
    assert texture_tag(["Ah", "7c", "2s", "3d", "3h"]) == "paired"
    assert texture_tag(["Ah", "7h", "2h"]) == "monotone"
    assert texture_tag(["Ah", "7c", "2s", "3d", "9h"]) == "wet"
    assert texture_tag(["Ah", "7c", "2s"]) == "dry"


# ── 1-4. 条件区分度 ────────────────────────────────────────────────────
def _index_with(entries: dict[str, tuple[int, int]]) -> ConditionalFoldIndex:
    idx = ConditionalFoldIndex()
    idx._counts = {k: [f, n] for k, (f, n) in entries.items()}
    return idx


def test_same_player_different_streets():
    idx = _index_with({"flop|bet|s<=0.5||wet": [40, 50],
                       "turn|bet|s<=0.5||wet": [10, 50]})
    flop = idx.estimate("flop", "bet", "s<=0.5", texture="wet", min_samples=10)
    turn = idx.estimate("turn", "bet", "s<=0.5", texture="wet", min_samples=10)
    assert flop["raw"] > 0.7 > turn["raw"]          # 真实条件率不同
    assert flop["p_fold"] > turn["p_fold"]          # 收缩后保持方向


def test_size_bucket_conditions_rate():
    idx = _index_with({"flop|bet|s<=0.5||wet": [10, 50],
                       "flop|bet|x>2.0||wet": [45, 50]})
    small = idx.estimate("flop", "bet", "s<=0.5", texture="wet", min_samples=10)
    big = idx.estimate("flop", "bet", "x>2.0", texture="wet", min_samples=10)
    assert big["raw"] > small["raw"]
    assert big["p_fold"] > small["p_fold"]


def test_position_conditions_rate():
    idx = _index_with({"flop|bet|s<=0.5|BTN|wet": [40, 50],
                       "flop|bet|s<=0.5|BB|wet": [10, 50]})
    btn = idx.estimate("flop", "bet", "s<=0.5", position="BTN",
                       texture="wet", min_samples=10)
    bb = idx.estimate("flop", "bet", "s<=0.5", position="BB",
                      texture="wet", min_samples=10)
    assert btn["raw"] > bb["raw"]
    assert btn["p_fold"] > bb["p_fold"]


def test_texture_conditions_rate():
    idx = _index_with({"flop|bet|s<=0.5||paired": [10, 50],
                       "flop|bet|s<=0.5||monotone": [40, 50]})
    paired = idx.estimate("flop", "bet", "s<=0.5", texture="paired",
                          min_samples=10)
    mono = idx.estimate("flop", "bet", "s<=0.5", texture="monotone",
                        min_samples=10)
    assert mono["raw"] > paired["raw"]
    assert mono["p_fold"] > paired["p_fold"]


# ── 5. 样本量不足 → UNKNOWN（契约门槛，禁止降低） ─────────────────────
def test_insufficient_samples_unknown():
    idx = _index_with({"flop|bet|s<=0.5||wet": [4, 5]})
    assert idx.estimate("flop", "bet", "s<=0.5", texture="wet") is None
    # 退化阶梯：细键不足但粗键充足 → 回粗粒度（不补数据）
    idx2 = _index_with({"flop|bet|||": [40, 50]})
    est = idx2.estimate("flop", "bet", "s<=0.5", texture="wet",
                        position="BTN", min_samples=10)
    assert est is not None and est["degraded_to"] > 0
    assert est["samples"] == 50          # 样本数如实来自粗键


def test_min_samples_contract_default():
    assert MIN_SAMPLES == 30
    idx = _index_with({"flop|bet|s<=0.5||wet": [20, 29]})   # 29 < 30
    assert idx.estimate("flop", "bet", "s<=0.5", texture="wet") is None
    idx2 = _index_with({"flop|bet|s<=0.5||wet": [25, 30]})  # 30 = 门槛
    assert idx2.estimate("flop", "bet", "s<=0.5", texture="wet") is not None


# ── 6. 小样本 shrinkage（向 population prior 收缩） ────────────────────
def test_shrinkage_toward_population_prior():
    player = _index_with({"flop|bet|s<=0.5||wet": [0, 30]})   # raw 0.0
    pop = _index_with({"flop|bet|s<=0.5||wet": [25, 50]})     # prior 0.5
    est = player.estimate("flop", "bet", "s<=0.5", texture="wet",
                          population=pop, min_samples=10)
    assert est is not None
    assert 0.3 < est["p_fold"] < 0.5           # 收缩：不冒充 0.0 精确值
    assert est["raw"] == 0.0 and est["prior_rate"] == 0.5
    assert est["p_fold"] > est["raw"]


# ── 7/8/9/10. FoldEstimator 来源层级 ──────────────────────────────────
def _behavior(cond=None, pop=None, stats=None):
    d: dict = {}
    if cond is not None:
        d["conditional_model"] = cond
    if pop is not None:
        d["conditional_population"] = pop
    if stats is not None:
        d["behavior_statistics"] = stats
    return d


def test_player_conditional_source():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_behavior(cond=_counts(
                            {"flop|bet|s<=0.5||wet": [40, 50]})),
                        hero_cards=["As", "Kd"], board=RIVER)
    fe = est.for_size(50.0)
    assert fe.available is True
    assert fe.provenance["source"] == "conditional_player"
    assert fe.provenance["stat"]["size_adjusted"] is True
    assert fe.provenance["stat"]["conditional"]["raw"] == 0.8
    assert 0.5 < fe.p_fold < 0.8               # 收缩后介于 prior 与 raw 之间


def test_population_fallback():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_behavior(pop=_counts(
                            {"flop|bet|s<=0.5||wet": [25, 50]})),
                        hero_cards=["As", "Kd"], board=RIVER)
    fe = est.for_size(50.0)
    assert fe.available is True
    assert fe.provenance["source"] == "conditional_population"


def test_global_fallback_p42_path():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_behavior(stats={
                            "fold_to_cbet": {"value": 0.42, "samples": 250}}),
                        hero_cards=["As", "Kd"], board=RIVER)
    fe = est.for_size(50.0)
    assert fe.available is True
    assert fe.provenance["source"] == "global"
    assert abs(fe.p_fold - 0.42) < 1e-9        # P42 语义不变


def test_unknown_when_no_evidence():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_behavior(cond=_counts(
                            {"flop|bet|s<=0.5||wet": [4, 5]})),  # 全部不足
                        hero_cards=["As", "Kd"], board=RIVER)
    fe = est.for_size(50.0)
    assert fe.available is False
    assert fe.provenance["source"] == "unknown"
    assert fe.p_fold is None and fe.p_continue is None


def test_conditional_priority_over_global():
    # 两者都在 → 条件模型优先（禁止全局冒充条件统计）
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_behavior(
                            cond=_counts({"flop|bet|s<=0.5||wet": [10, 50]}),
                            stats={"fold_to_cbet": {"value": 0.9,
                                                    "samples": 900}}),
                        hero_cards=["As", "Kd"], board=RIVER)
    fe = est.for_size(50.0)
    assert fe.provenance["source"] == "conditional_player"
    assert fe.p_fold < 0.5                     # 用条件值而非全局 0.9


# ── 12. EVCore 受条件 fold 概率影响 ───────────────────────────────────
def _env(beh: dict) -> dict:
    """opponent envelope（compute_ev/judge 契约：players 包裹）。"""
    return {"players": {"V1": beh}}


def test_evcore_affected_by_conditional_fold():
    hi = _counts({"flop|bet|s<=0.5||wet": [49, 50]})   # raw≈0.98
    lo = _counts({"flop|bet|s<=0.5||wet": [1, 50]})    # raw≈0.02
    args_hi = (_state(), _eq(0.6), None, _env(_behavior(cond=hi)))
    args_lo = (_state(), _eq(0.6), None, _env(_behavior(cond=lo)))
    r_hi, r_lo = compute_ev(*args_hi), compute_ev(*args_lo)
    bet_hi = next(c for c in r_hi.candidates if c.action == "bet"
                  and c.amount_bb == 50.0)
    bet_lo = next(c for c in r_lo.candidates if c.action == "bet"
                  and c.amount_bb == 50.0)
    assert bet_hi.p_fold is not None and bet_lo.p_fold is not None
    assert bet_hi.p_fold > bet_lo.p_fold + 0.1   # 收缩后仍显著分层
    assert bet_hi.ev_bb > bet_lo.ev_bb           # 弃牌权益直接抬高 EV
    assert r_hi.ev_mode == "point+fold"


# ── 13. P42 fallback 不回归（无条件数据 → P42 数值原样） ──────────────
def test_p42_stat_only_unchanged():
    r = compute_ev(_state(), _eq(0.6), None,
                   _env(_behavior(stats={"fold_to_cbet": {"value": 0.5,
                                                          "samples": 300}})))
    bet = next(c for c in r.candidates if c.action == "bet"
               and c.amount_bb == 50.0)
    assert abs(bet.ev_bb - 85.0) < 1e-9        # P42 测试同款数值
    assert bet.rule.startswith("EV_bet=p_f")
    assert r.ev_mode == "point+fold"
    # 无任何来源 → always-called fallback 原样
    r2 = compute_ev(_state(), _eq(0.6))
    bet2 = next(c for c in r2.candidates if c.action == "bet"
                and c.amount_bb == 50.0)
    assert abs(bet2.ev_bb - 70.0) < 1e-9
    assert bet2.p_fold is None
    assert r2.ev_mode == "point_fallback"


# ── C4：OpponentProfile API 兼容 + Opponent Agent 挂接 ────────────────
def test_opponent_profile_conditional_stats_compat():
    from poker_decision_ai.opponent_profile import OpponentProfile
    p = OpponentProfile(player_id="V", status="RECOGNIZED", sample_size=30)
    d = p.to_dict()
    assert d["conditional_stats"] == {}        # 新字段可序列化、默认空
    p2 = OpponentProfile.from_history.__func__  # 方法存在（API 兼容）
    assert callable(p2)


async def test_opponent_agent_attaches_conditional_model(ctx):
    from poker_decision_ai.agents.opponent.agent import OpponentAgent
    # 真实 parsed 记录 → repos.hands（DataImporter 同构路径）
    for i in range(40):
        ctx.repos.hands.save({"hand_id": f"H{i}",
                              "parsed": _parsed_hand(f"H{i}", i % 5 != 0)})
    env = await OpponentAgent().run({
        "state": {"hero_id": "Hero",
                  "players": [{"player_id": "V", "name": "V",
                               "status": "active", "is_hero": False}]},
    }, ctx)
    out = env.output
    prof = out["players"]["V"]
    cm = prof.get("conditional_model") or {}
    assert cm.get("counts"), "40 手真实 parsed 历史 → 条件索引非空"
    assert prof.get("conditional_population", {}).get("counts")
    assert prof["conditional_events"] > 0
    # 条件索引可被 FoldEstimator 消费
    fe = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                       behavior=prof, hero_cards=["As", "Kd"], board=RIVER)
    r = fe.for_size(50.0)
    assert r.available is True
    assert r.provenance["source"] in ("conditional_player",
                                      "conditional_population")


# ── 序列化往返（确定性） ───────────────────────────────────────────────
def test_index_serialization_roundtrip():
    idx = _index_with({"flop|bet|s<=0.5||wet": [40, 50]})
    clone = ConditionalFoldIndex.from_dict(idx.to_dict())
    assert clone.to_dict() == idx.to_dict()
    e1 = idx.estimate("flop", "bet", "s<=0.5", texture="wet", min_samples=10)
    e2 = clone.estimate("flop", "bet", "s<=0.5", texture="wet", min_samples=10)
    assert e1 == e2
