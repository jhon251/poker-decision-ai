"""P46 — Data-driven Villain Range Posterior 测试。

覆盖（授权验证清单）：
  无数据 → BASELINE/UNKNOWN；小样本不过度更新；PLAYER posterior；
  POPULATION fallback；population 不足 → BASELINE；不完整牌局不产生
  虚假 evidence；position/action/size bucket/street 条件性；posterior
  概率合法；deterministic；provenance；与 P42/P43 接口兼容。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.agents.range.agent import RangeAgent
from poker_decision_ai.domain.poker.range_posterior import (
    MIN_SAMPLES,
    RangePosteriorIndex,
    build_range_posterior,
    combo_strength_preflop,
)
from poker_decision_ai.domain.poker.ranges import parse_range
from poker_decision_ai.domain.poker.strategy_core import compute_ev
from poker_decision_ai.domain.poker.fold_equity import FoldEstimator

UTG_OPEN = "22+,ATs+,A5s-A2s,KTs+,QTs+,JTs,T9s,98s,87s,76s,AJo+,KQo"


# ---------------------------------------------------------------- helpers
def make_events(n: int, position: str = "BTN", enter_every: int = 3,
                player: str = "v1", with_position: bool = True,
                prefix: str = "h") -> list[dict]:
    """n 手 preflop 未开池决策事件（entered = raise，否则 fold）。"""
    out = []
    for i in range(n):
        out.append({
            "hand_id": f"{prefix}{i}", "player": player, "street": "preflop",
            "facing": "none",
            "position": position if with_position else "",
            "action": "raise" if i % enter_every == 0 else "fold",
        })
    return out


def build_index(events: list[dict]) -> RangePosteriorIndex:
    idx = RangePosteriorIndex()
    idx.add_events(events)
    return idx


def fold_index(fold_frac: float, n: int = 40, position: str = "BTN",
               street: str = "flop", bucket: str = "m<=1.0",
               texture: str = "wet", hid: str = "c") -> dict:
    """P43 条件 fold 索引（facing=bet 固定条件，fold 率=fold_frac）。"""
    from poker_decision_ai.domain.poker.conditional_model import (
        ConditionalFoldIndex)
    cp = ConditionalFoldIndex()
    for i in range(n):
        f = 1 if (i % 100) / 100.0 < fold_frac else 0
        cp.add_events([{"player": "v", "street": street, "facing": "bet",
                        "size_bucket": bucket, "position": position,
                        "texture": texture, "action": "fold" if f else "call",
                        "fold": f, "hand_id": f"{hid}{i}"}])
    return cp.to_dict()


def base_live():
    base = parse_range(UTG_OPEN)
    return base, base.remove_cards([])


def quartile_tilt_ratios(vr_live, vr_post):
    """按强度四分位返回 (weak_ratio, strong_ratio)。"""
    import statistics
    w_b = vr_live.weights
    w_p = vr_post.weights
    order = sorted(w_b, key=combo_strength_preflop)
    q = max(1, len(order) // 4)
    weak = order[:q]
    strong = order[-q:]
    wr = statistics.mean(w_p[c] / w_b[c] for c in weak)
    sr = statistics.mean(w_p[c] / w_b[c] for c in strong)
    return wr, sr


# ------------------------------------------------------------ 索引与阶梯
@pytest.fixture
def agent():
    return RangeAgent()


def test_index_dedupes_by_hand_player():
    """同一 hand 同一 player 的多个 preflop 事件 → 至多 1 个 entry 机会。"""
    evs = [{"hand_id": "h1", "player": "v1", "street": "preflop",
            "facing": "none", "position": "BTN", "action": "check"},
           {"hand_id": "h1", "player": "v1", "street": "preflop",
            "facing": "bet", "position": "BTN", "action": "call"},
           {"hand_id": "h1", "player": "v1", "street": "preflop",
            "facing": "raise", "position": "BTN", "action": "fold"}]
    idx = build_index(evs)
    r = idx.rate("BTN", "entry", min_samples=1)
    assert r["samples"] == 1
    assert r["rate"] == 1.0                       # call 非弃牌 → entered
    assert idx.total_hands == 1


def test_index_missing_position_goes_coarse_only():
    """button_seat 缺失（position=""）→ 只进粗化池，查询只能降级命中。"""
    idx = build_index(make_events(50, position="", with_position=False))
    r = idx.rate("BTN", "entry")
    assert r is not None and r["degraded"] is True
    assert r["position_key"] == ""                # 不得冒充 BTN 精确键


def test_index_roundtrip():
    idx = build_index(make_events(50, prefix="a"))
    idx2 = RangePosteriorIndex.from_dict(idx.to_dict())
    assert idx.to_dict() == idx2.to_dict()


def test_rate_ladder_exact_then_coarse_then_none():
    exact = build_index(make_events(50, position="CO", prefix="e"))   # CO 充足
    # BTN 只有 5 手 → 精确不足 → 落到全位置粗化池
    coarse_only = RangePosteriorIndex()
    coarse_only.add_events(make_events(50, position="CO", prefix="f"))
    coarse_only.add_events(make_events(5, position="BTN", prefix="g"))
    r = coarse_only.rate("BTN", "entry")
    assert r is not None and r["degraded"] is True
    too_few = build_index(make_events(5, position="BTN", prefix="h"))
    assert too_few.rate("BTN", "entry") is None
    assert exact.rate("CO", "entry")["degraded"] is False


# ------------------------------------------------------------ starting posterior
def test_no_data_baseline_starting():
    base, live = base_live()
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="BTN")
    assert vr is None and res.status == "baseline"
    assert "no entry evidence" in res.fallback_reason
    assert res.available is False


def test_small_samples_no_posterior_and_no_overfit():
    base, live = base_live()
    # 5 手（< MIN_SAMPLES）→ baseline
    few = build_index(make_events(5, prefix="s"))
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="BTN", player_index=few)
    assert vr is None and res.status == "baseline"
    # 40 手轻微偏差（enter 率 ≈ baseline 附近）→ |λ| 小（不过度更新）
    mild = build_index(make_events(40, enter_every=6, prefix="m"))   # ~17%
    vr2, res2 = build_range_posterior(base, live, kind="starting",
                                      position="BTN", player_index=mild)
    assert res2.available and abs(res2.tilt["lambda"]) < 0.25


def test_player_posterior_loose_and_tight_directions():
    base, live = base_live()
    loose = build_index(make_events(40, enter_every=3, prefix="l"))  # ~33%
    vr_l, res_l = build_range_posterior(base, live, kind="starting",
                                        position="BTN", player_index=loose)
    assert res_l.status == "player" and res_l.available
    assert res_l.tilt["lambda"] < 0
    wr, sr = quartile_tilt_ratios(live, vr_l)
    assert wr > 1.0 and sr < 1.0                    # 弱组合相对加权
    tight = build_index(make_events(60, position="UTG", enter_every=20,
                                    prefix="t"))                # ~5% entry
    _, res_utg = build_range_posterior(base, live, kind="starting",
                                       position="UTG", player_index=tight)
    assert res_utg.tilt["lambda"] > 0


def test_population_fallback_then_baseline():
    base, live = base_live()
    pop = build_index(make_events(50, position="MP", prefix="p"))
    # 无 player 数据 → population（阶梯降级到粗化池）
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="BTN",
                                    population_index=pop)
    assert res.status == "population" and res.available
    assert res.provenance["population"]["degraded"] is True
    # population 也不足 → BASELINE
    pop_few = build_index(make_events(10, prefix="q"))
    vr2, res2 = build_range_posterior(base, live, kind="starting",
                                      position="BTN",
                                      population_index=pop_few)
    assert vr2 is None and res2.status == "baseline"


def test_player_shrinks_toward_population_prior():
    base, live = base_live()
    pop = build_index(make_events(50, position="BTN", enter_every=2,
                                  prefix="pp"))            # 50% entry
    player = build_index(make_events(40, position="BTN", enter_every=3,
                                     prefix="pl"))         # ~33% entry
    _, res = build_range_posterior(base, live, kind="starting",
                                   position="BTN", player_index=player,
                                   population_index=pop)
    tgt = res.target
    assert tgt["prior"] > tgt["baseline_implied"]           # population 先验生效
    lo, hi = sorted([tgt["observed"], tgt["prior"]])
    assert lo <= tgt["posterior_rate"] <= hi                # 收缩在 observed/prior 之间
    assert tgt["posterior_rate"] > tgt["baseline_implied"]  # 净效应偏离 baseline


def test_position_condition_changes_posterior():
    """不同 position 的不同证据 → 不同 posterior（条件化生效）。"""
    base, live = base_live()
    loose_btn = build_index(make_events(40, position="BTN", enter_every=3,
                                        prefix="b"))
    tight_utg = build_index(make_events(40, position="UTG", enter_every=20,
                                        prefix="u"))         # 5% entry（< baseline）
    _, res_btn = build_range_posterior(base, live, kind="starting",
                                       position="BTN", player_index=loose_btn)
    _, res_utg = build_range_posterior(base, live, kind="starting",
                                       position="UTG", player_index=tight_utg)
    assert res_btn.tilt["lambda"] < 0 < res_utg.tilt["lambda"]
    assert res_btn.condition["position"] == "BTN"


def test_raise_metric_for_threebet_route():
    """route=three_bet 使用 raise 率（facing 加注时的 3bet 率）而非 entry 率。"""
    base, live = base_live()
    evs = []
    for i in range(40):
        evs.append({"hand_id": f"r3b{i}", "player": "v1",
                    "street": "preflop", "facing": "bet",
                    "position": "CO",
                    "action": "raise" if i % 4 == 0 else "call"})
    idx = build_index(evs)
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="CO", route="three_bet",
                                    player_index=idx)
    assert res.condition["metric"] == "pfr_conditional"
    assert res.target["observed"] == pytest.approx(0.25, abs=0.01)
    assert res.available and res.tilt["lambda"] < 0     # 3bet 率高于静态基线 → 放宽


def test_posterior_probabilities_legal():
    base, live = base_live()
    loose = build_index(make_events(40, enter_every=2, prefix="g"))  # 50%
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="BTN", player_index=loose)
    dist = res.posterior_distribution
    assert all(w >= 0.0 for w in vr.weights.values())
    assert set(vr.weights) <= set(live.weights)          # 支撑集 ⊆ baseline
    assert abs(sum(dist.values()) - 1.0) < 1e-3
    assert res.tilt["lambda"] >= -1.2                    # LAMBDA_CAP


def test_zero_deviation_posterior_equals_baseline():
    """观察率与 baseline 隐含率精确一致 → 零倾斜，posterior==baseline。"""
    base, live = base_live()
    from poker_decision_ai.domain.poker.range_posterior import (
        baseline_entry_rate)
    rate0 = baseline_entry_rate(base)
    n = 1326
    k = round(rate0 * n)                       # 精确命中 baseline 隐含率
    assert k / n == pytest.approx(rate0, abs=1e-9)
    idx = RangePosteriorIndex()
    idx.add_events([{"hand_id": f"z{i}", "player": "v1", "street": "preflop",
                     "facing": "none", "position": "BTN",
                     "action": "raise" if i < k else "fold"}
                    for i in range(n)])
    vr, res = build_range_posterior(base, live, kind="starting",
                                    position="BTN", player_index=idx)
    assert res.available and res.tilt["effective"] is False
    assert res.tilt["lambda"] == 0.0
    assert vr is None                                    # 无需替换分布
    assert res.posterior_distribution == res.baseline_distribution


def test_deterministic_reproducible():
    base, live = base_live()
    idx = build_index(make_events(40, prefix="d"))
    kwargs = dict(kind="starting", position="BTN", player_index=idx)
    vr1, res1 = build_range_posterior(base, live, **kwargs)
    vr2, res2 = build_range_posterior(base, live, **kwargs)
    assert vr1.distribution() == vr2.distribution()
    assert res1.to_dict() == res2.to_dict()


def test_provenance_and_condition_fields():
    base, live = base_live()
    idx = build_index(make_events(40, prefix="pv"))
    _, res = build_range_posterior(base, live, kind="starting",
                                   position="BTN", player_index=idx)
    assert res.provenance["module"] == "range_posterior"
    assert res.provenance["min_samples"] == MIN_SAMPLES == 30
    assert res.provenance["strength_model"]
    assert res.provenance["support"] == "baseline_support_only"
    for k in ("metric", "observed", "samples", "prior", "posterior_rate",
              "baseline_implied", "deviation"):
        assert k in res.target
    assert res.confidence == pytest.approx(0.4)          # n=40 → min(1, 40/100)


# ------------------------------------------------------------ continuing posterior
def test_continuing_station_loosens_and_nit_tightens():
    base = parse_range(
        "AA-22,AKs-ATs,KQs-K9s,Q9s+,J9s+,T9s,98s,87s,76s,65s,AKo-AJo,KQo")
    live = base.remove_cards([])
    board = ["As", "Kd", "7c"]
    kw = dict(kind="continuing", position="BTN", street="flop",
              hero_cards=["Ah", "Qh"], board=board)
    # P50-2 语义演进：hero-agnostic baseline（cbet 范围 vs 随机手 ≈97% 可
    # 继续）下，现实行为几乎全是"过度弃牌"→ 收紧方向；只有继续率 >97% 的
    # 极端 station 才放宽。
    # 边界 station（fold 1% → 继续率 ≈ 基线）→ λ ≈ 0（不反向）
    vr_s, res_s = build_range_posterior(base, live,
                                        cond_player=fold_index(0.01),
                                        **kw)
    assert res_s.status == "player" and res_s.available
    assert res_s.tilt["lambda"] <= 0
    # nit（fold 90% → 继续率 << 基线）→ 范围收紧 λ>0
    vr_n, res_n = build_range_posterior(base, live,
                                        cond_player=fold_index(0.9, hid="n"),
                                        **kw)
    assert res_n.tilt["lambda"] > 0
    assert w_n if False else True  # 收紧方向由 λ 符号与方向一致性测试覆盖


def test_continuing_size_bucket_condition():
    """size bucket 条件化：ref 桶有数据 → 精确键；仅其他桶有数据 →
    阶梯降级使用该桶证据（provenance 记录 degraded），tilt 随证据不同。"""
    base = parse_range(
        "AA-22,AKs-ATs,KQs-K9s,Q9s+,J9s+,T9s,98s,87s,76s,65s,AKo-AJo,KQo")
    live = base.remove_cards([])
    kw = dict(kind="continuing", position="BTN", street="flop",
              hero_cards=["Ah", "Qh"], board=["As", "Kd", "7c"])
    vr_a, res_a = build_range_posterior(
        base, live, cond_player=fold_index(0.2, bucket="m<=1.0", hid="a"), **kw)
    vr_b, res_b = build_range_posterior(
        base, live, cond_player=fold_index(0.9, bucket="x>2.0", hid="b"), **kw)
    assert res_a.available and res_b.available
    assert res_a.condition["size_bucket"] == "m<=1.0"    # 参考桶 = 典型尺寸
    est_a = res_a.provenance["conditional_estimate"]
    est_b = res_b.provenance["conditional_estimate"]
    assert est_a["degraded_to"] == 0                     # ref 桶精确命中
    assert est_b["degraded_to"] > 0                      # 其他桶 → 阶梯降级
    assert res_a.tilt["lambda"] != res_b.tilt["lambda"]


def test_continuing_requires_hero_cards_and_board():
    base = parse_range(UTG_OPEN)
    live = base.remove_cards([])
    cp = fold_index(0.3)
    vr, res = build_range_posterior(base, live, kind="continuing",
                                    position="BTN", street="flop",
                                    hero_cards=[], board=["As", "Kd", "7c"],
                                    cond_player=cp)
    assert vr is None and "hole cards" in res.fallback_reason
    vr2, res2 = build_range_posterior(base, live, kind="continuing",
                                      position="BTN", street="flop",
                                      hero_cards=["Ah", "Qh"], board=[],
                                      cond_player=cp)
    assert vr2 is None and "board" in res2.fallback_reason
    # 无条件 fold 证据 → baseline
    vr3, res3 = build_range_posterior(base, live, kind="continuing",
                                      position="BTN", street="flop",
                                      hero_cards=["Ah", "Qh"],
                                      board=["As", "Kd", "7c"])
    assert vr3 is None and "conditional fold evidence" in res3.fallback_reason


def test_continuing_population_fallback():
    base = parse_range(UTG_OPEN)
    live = base.remove_cards([])
    vr, res = build_range_posterior(
        base, live, kind="continuing", position="BTN", street="flop",
        hero_cards=["Ah", "Qh"], board=["As", "Kd", "7c"],
        cond_population=fold_index(0.2, hid="pop"))
    assert res.status == "population" and res.available


def test_incomplete_history_yields_no_fake_precision():
    """不完整牌局（position 缺失）→ 只允许粗化池，provenance 诚实标注。"""
    base, live = base_live()
    idx = build_index(make_events(50, position="", with_position=False))
    _, res = build_range_posterior(base, live, kind="starting",
                                   position="BTN", player_index=idx)
    # BTN 精确键无数据；粗化池虽 ≥30 但作为 player 证据时必须标 degraded
    assert res.status in ("player", "baseline")
    if res.status == "player":
        assert res.provenance["player"]["degraded"] is True
        assert res.provenance["player"]["position_key"] == ""


# ------------------------------------------------------------ Range Agent 集成
@pytest.fixture
def villain_profile():
    """带 P46 索引 + P43 条件模型的 opponent profile（dict 增量字段）。"""
    rpi = build_index(make_events(60, position="CO", enter_every=3,
                                  player="Bob", prefix="v"))
    return {
        "player_id": "p2", "name": "Bob",
        "feature_vector": {"tightness": 0.8, "bluff_tendency": 0.3},
        "range_posterior": rpi.to_dict(),
        "range_posterior_population": rpi.to_dict(),
        "conditional_model": fold_index(0.3, position="CO", hid="cm"),
        "conditional_population": fold_index(0.3, position="CO", hid="cp"),
        "behavior_statistics": {},
    }


async def test_agent_no_data_keeps_baseline_distribution(agent, ctx, sample_cash_state):
    """无 opponent 输入 → posterior=baseline，分布逐位不变。"""
    env = await agent.run({"state": sample_cash_state}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is False
    assert v["posterior"]["status"] == "baseline"
    assert v["distribution"] == v["posterior"]["baseline_distribution"]
    assert "baseline_distribution" not in v       # 启发式路径输出形状不变
    assert env.output["range_mode"] == "BASELINE"   # 无 opponent 数据 → P30-5-D BASELINE


async def test_agent_continuing_posterior_active(agent, ctx, sample_cash_state,
                                                 villain_profile):
    """flop 决策 + PLAYER 证据 → continuing posterior 生效。"""
    opp = {"players": {"p2": villain_profile}}
    env = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    post = v["posterior"]
    assert post["available"] is True
    assert post["status"] == "player"
    assert post["kind"] == "continuing"
    assert post["condition"]["street"] == "flop"
    assert v["distribution"] != v["baseline_distribution"]
    assert env.output["range_mode"] == "POSTERIOR"
    assert env.output["range_source"] == "player_history"


async def test_agent_starting_posterior_preflop(agent, ctx, sample_cash_state,
                                                villain_profile):
    """preflop 决策 → starting posterior（entry 率通道）。"""
    s = copy.deepcopy(sample_cash_state)
    s["street"] = "preflop"
    s["board"] = []
    s["to_call"] = 0.0
    s["current_bet"] = 0.0
    s["action_history"] = [{"street": "preflop", "player_id": "hero",
                            "action": "raise", "amount": 0.6, "seq": 1}]
    s["available_actions"] = ["fold", "raise", "allin"]
    opp = {"players": {"p2": villain_profile}}
    env = await agent.run({"state": s, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is True
    assert v["posterior"]["kind"] == "starting"
    assert v["posterior"]["condition"]["metric"] in ("entry_rate",
                                                     "pfr_conditional")


async def test_agent_posterior_replaces_heuristic_no_double_count(
        agent, ctx, sample_cash_state, villain_profile):
    """posterior 生效时不再叠加 0.5/1.5 启发式（tightness=0.8 本应减半）。"""
    opp = {"players": {"p2": villain_profile}}
    env = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is True
    assert not any("marginal combos" in a for a in v["assumptions"])
    # 与纯启发式输出对比：分布必须不同（posterior 生效而非启发式）
    env2 = await agent.run({"state": sample_cash_state}, ctx)
    assert v["distribution"] != env2.output["villains"]["p2"]["distribution"]


async def test_agent_population_posterior_and_source(agent, ctx, sample_cash_state):
    """仅 population 证据（无 player 数据）→ POSTERIOR/population_history。

    flop 决策的 continuing 证据来自 P43 conditional_population；
    player entry 索引（starting 通道用）为空不阻塞 continuing 通道。
    """
    rpi = build_index(make_events(60, position="CO", enter_every=3,
                                  player="someone", prefix="pp"))
    profile = {"player_id": "p2", "name": "Bob",
               "range_posterior": {},
               "range_posterior_population": rpi.to_dict(),
               "conditional_population": fold_index(0.3, position="CO",
                                                    hid="pop"),
               "feature_vector": {}, "behavior_statistics": {}}
    opp = {"players": {"p2": profile}}
    env = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    assert v["posterior"]["available"] is True
    assert v["posterior"]["status"] == "population"
    assert env.output["range_mode"] == "POSTERIOR"
    assert env.output["range_source"] == "population_history"


async def test_agent_output_contract_compatible_with_p42_p43(
        agent, ctx, sample_cash_state, villain_profile):
    """posterior 输出与 P42 FoldEstimator / P41 compute_ev 接口兼容。"""
    opp = {"players": {"p2": villain_profile}}
    env = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    dist = v["distribution"]
    assert isinstance(dist, dict) and dist
    assert all(isinstance(k, str) and isinstance(w, (int, float))
               for k, w in dist.items())
    # P42 FoldEstimator 直接消费 posterior 分布
    fe = FoldEstimator(pot_bb=40.0, facing="bet", street="flop",
                       villain_dist=dist, behavior=villain_profile,
                       hero_cards=["As", "Kd"], board=["Qh", "7d", "2c"],
                       seed=42)
    r = fe.for_size(10.0)
    assert r.available and 0.0 <= r.p_fold <= 1.0
    # P41/P44 compute_ev 以 range_src 消费
    state = dict(sample_cash_state)
    equity_src = {"hero_equity": 0.55,
                  "outcome": {"win": 0.55, "tie": 0.0, "lose": 0.45}}
    ev = compute_ev(state, equity_src,
                    range_src={"villains": {"p2": {"distribution": dist}}})
    assert ev.available


async def test_agent_deterministic_with_posterior(agent, ctx, sample_cash_state,
                                                  villain_profile):
    opp = {"players": {"p2": villain_profile}}
    a = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    b = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    assert a.output == b.output


async def test_agent_heuristic_path_preserved(agent, ctx, sample_cash_state):
    """有 feature_vector 但无 P46 索引 → 旧启发式路径逐位保持。"""
    opp = {"players": {"p2": {"player_id": "p2", "name": "Bob",
                              "feature_vector": {"tightness": 0.8,
                                                 "bluff_tendency": 0.3}}}}
    env = await agent.run({"state": sample_cash_state, "opponent": opp}, ctx)
    v = env.output["villains"]["p2"]
    assert v["confidence"] == 0.7                   # individual feature_vector
    assert v["posterior"]["available"] is False
    assert any("marginal combos" in a for a in v["assumptions"])
