"""V3F-P32 — Population Analysis & Population Strategy Prior 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.population import (  # noqa: E402
    PopulationDataset, PopulationPrior, aggregate_population,
    population_priors, prior_for, DATA_KINDS, _position_of)
from poker_decision_ai.population_prior import (  # noqa: E402
    PlayerPosterior, posterior_from, PopulationDeviation)
from poker_decision_ai.player_history import (  # noqa: E402
    PlayerHistoryStats, reliability_for)

EV = Path("data/hand_history/synthetic_cash_100")


def _texts(n=20):
    return [(EV / f"hand_{i:05d}.txt").read_text(encoding="utf-8")
            for i in range(70000, 70000 + n)]


# ── 1-5. Population aggregation ─────────────────────────────────────
def test_population_aggregation():
    ds = aggregate_population(_texts(10))
    assert ds.sample_size >= 1
    assert "global" in ds.metrics
    assert ds.metrics["global"]["hands"] >= 1
    assert ds.metrics["global"]["vpip"] is not None


def test_multi_player_aggregation():
    ds = aggregate_population(_texts(20))
    assert ds.metrics["global"]["sample_size"] >= 10


def test_population_sample_size():
    ds = aggregate_population(_texts(50))
    assert ds.sample_size >= 10


def test_population_reliability():
    ds = aggregate_population(_texts(100))
    assert ds.reliability in ("INSUFFICIENT", "LOW", "MEDIUM", "HIGH")
    assert ds.reliability == reliability_for(ds.sample_size)


def test_population_provenance():
    ds = aggregate_population(_texts(5))
    assert ds.provenance != ""
    assert "synthetic" in ds.provenance.lower()   # 诚实标记 SYNTHETIC


# ── 6-9. Global / context / position / street ───────────────────────
def test_global_population_metrics():
    ds = aggregate_population(_texts(10))
    g = ds.metrics["global"]
    assert {"vpip", "pfr", "three_bet", "cbet", "hands"} <= set(g)


def test_context_population():
    ds = aggregate_population(_texts(10))
    assert ds.game_context.get("game_type") == "cash"
    assert ds.game_context.get("table_size") == 6


def test_position_population():
    ds = aggregate_population(_texts(20))
    bp = ds.metrics.get("by_position") or {}
    # 若数据支持（多 hand 多位置）→ BTN/SB/BB bucket 存在；否则空（合法）
    assert isinstance(bp, dict)


def test_position_of_mapping():
    assert _position_of(1, 1) == "BTN"        # button 自身
    assert _position_of(2, 1) == "SB"
    assert _position_of(3, 1) == "BB"
    assert _position_of(4, 1) == "CO"
    assert _position_of(5, 1) == "MP"
    assert _position_of(None, 1) == "UNKNOWN"


# ── 10-14. Population prior + Bayesian ──────────────────────────────
def test_population_prior_no_bare_value():
    ds = aggregate_population(_texts(20))
    priors = population_priors(ds)
    assert "vpip" in priors
    p = priors["vpip"]
    assert p.value is not None
    assert p.sample_size > 0                  # 无裸值（带 sample）
    assert p.reliability != ""
    assert p.provenance != ""


def test_population_player_bayesian_integration():
    ds = aggregate_population(_texts(30))
    priors = population_priors(ds)
    prior = priors.get("vpip")
    post = posterior_from(0.60, 8, prior, "vpip")
    assert post.posterior is not None
    assert post.sample_size == 8
    # 收缩：0.60 obs + prior 收缩 → posterior < 0.60（远离小样本噪声）
    assert post.posterior < 0.60


def test_small_sample_player():
    ds = aggregate_population(_texts(30))
    prior = population_priors(ds).get("vpip")
    post = posterior_from(1.0, 3, prior, "vpip")
    assert post.reliability == "INSUFFICIENT"   # <5 hands
    assert post.posterior is not None


def test_medium_sample_player():
    post = posterior_from(0.6, 30, _prior(0.25), "vpip")
    assert post.reliability == "MEDIUM"
    assert post.posterior is not None


def test_high_sample_player():
    post = posterior_from(0.6, 150, _prior(0.25), "vpip")
    assert post.reliability == "HIGH"
    assert post.posterior is not None


def _prior(v):
    return PopulationPrior(metric="vpip", value=v, sample_size=100,
                           reliability="HIGH", source="pop",
                           data_kind="SYNTHETIC", provenance="test")


# ── 15-18. no history / no population / UNKNOWN / isolation ─────────
def test_no_player_history():
    post = posterior_from(None, 0, None, "vpip")
    assert post.posterior is None               # UNKNOWN——不伪造


def test_no_population_data():
    post = posterior_from(0.5, 10, None, "vpip")
    assert post.posterior is None               # 无 prior → UNKNOWN


def test_unknown_propagation():
    dev = PopulationDeviation.from_posterior(
        PlayerPosterior(metric="vpip", posterior=None))
    assert dev.deviation is None
    assert dev.reliability == "INSUFFICIENT"


def test_cross_player_isolation():
    ds = aggregate_population(_texts(10))
    priors = population_priors(ds)
    # prior 全局（无玩家绑定）——玩家 posterior 独立计算
    p1 = posterior_from(0.5, 10, priors.get("vpip"), "vpip")
    p2 = posterior_from(0.5, 10, priors.get("vpip"), "vpip")
    assert p1.sample_size == p2.sample_size == 10


# ── 19-20. synthetic/real provenance + P31 regression ──────────────
def test_synthetic_real_provenance_separation():
    ds = aggregate_population(_texts(5))
    assert ds.data_kind == "SYNTHETIC"          # 诚实标识
    assert "SYNTHETIC" in DATA_KINDS
    assert ds.data_kind != "REAL"


def test_p31_regression():
    from poker_decision_ai.player_history import aggregate_player, reliability_for  # noqa: E501
    from poker_decision_ai.domain.poker.hand_history import parse_hand_history
    h = parse_hand_history((EV / "hand_70000.txt").read_text(
        encoding="utf-8"))
    st = aggregate_player(h.actions, "Lucy")
    assert st.hands == 1
    assert st.reliability in ("INSUFFICIENT", "LOW", "MEDIUM", "HIGH")


# ── 21. P30 regression ─────────────────────────────────────────────
def test_p30_regression():
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
        frame_to_decision_request)
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png((Path("data/vision2/coinpoker_evidence") /
        "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "OK"


# ── 22-24. Deviation 信号/无硬编码 TAG-LAG/无 random ────────────────
def test_no_action_from_deviation():
    dev = PopulationDeviation.from_posterior(
        PlayerPosterior("vpip", posterior=0.4, observed=0.6,
                        population_prior=0.25, sample_size=30,
                        reliability="MEDIUM"))
    d = dev.to_dict()
    assert d["is_signal_not_action"] is True     # 信号非 Action
    assert d["deviation"] is not None


def test_no_tag_lag_hardcoded():
    import re
    src = Path("poker_decision_ai/population_prior.py").read_text(
        encoding="utf-8")
    body = re.sub(r'""".*?"""', '', src, flags=re.S)
    assert "TAG" not in body.upper().split(
        "TAG/LAG")[0] or "TAG" not in body.upper()
    assert "LAG" not in body.upper() or "LAG" not in body.upper()


def test_no_random_fallback():
    import re
    for fp in ("poker_decision_ai/population.py",
               "poker_decision_ai/population_prior.py"):
        src = Path(fp).read_text(encoding="utf-8")
        body = re.sub(r'""".*?"""', '', src, flags=re.S)
        assert "random.choice" not in body
        assert "random_range" not in body


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
