"""V3F-P33 — Deviation Signal & Exploitability Layer 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.population import (  # noqa: E402
    PopulationPrior, aggregate_population)
from poker_decision_ai.population_prior import (  # noqa: E402
    PlayerPosterior, PopulationDeviation, posterior_from)
from poker_decision_ai.deviation_signal import (  # noqa: E402
    DeviationSignal, signal_from, ExploitabilityScore,
    exploitability_from_signals, deviation_signal_for_player,
    RELIABILITY_CONFIDENCE_CAP, DIRECTIONS, Z_SIGNIFICANT,
    MIN_SAMPLE_SIGNAL)
from poker_decision_ai.player_history import PlayerHistoryStats  # noqa: E402

EV = Path("data/hand_history/synthetic_cash_100")


def _dev(obs, n, pop=0.25, player="p1", metric="vpip"):
    return PopulationDeviation(
        player_id=player, metric=metric, observed=obs, population=pop,
        posterior=(obs + pop) / 2 if obs is not None and pop is not None
        else None,
        deviation=(round(obs - pop, 4) if obs is not None and pop is not None
                   else None),
        sample_size=n, reliability=reliability_for(n),
        provenance="deviation_signal_population_prior")


def reliability_for(n):
    if n < 5:
        return "INSUFFICIENT"
    if n < 25:
        return "LOW"
    if n < 100:
        return "MEDIUM"
    return "HIGH"


def _prior(v):
    return PopulationPrior(metric="vpip", value=v, sample_size=100,
                           reliability="HIGH", source="pop",
                           data_kind="SYNTHETIC", provenance="test")


# ── 1. Contract ─────────────────────────────────────────────────────
def test_deviation_contract():
    s = signal_from(_dev(0.6, 100))
    d = s.to_dict()
    for k in ("player_id", "metric", "observed", "population_prior",
              "posterior", "deviation", "sample_size", "reliability",
              "direction", "confidence", "source", "provenance"):
        assert k in d
    assert d["is_signal_not_action"] is True


# ── 2-5. 各 metric ──────────────────────────────────────────────────
def test_vpip_deviation():
    s = signal_from(_dev(0.55, 100, pop=0.25), metric="vpip")
    assert s.metric == "vpip"
    assert s.observed == 0.55


def test_pfr_deviation():
    s = signal_from(_dev(0.30, 100, pop=0.15), metric="pfr")
    assert s.metric == "pfr"
    assert s.deviation == 0.15


def test_three_bet_deviation():
    s = signal_from(_dev(0.12, 100, pop=0.08), metric="three_bet")
    assert s.metric == "three_bet"


def test_cbet_deviation():
    s = signal_from(_dev(0.70, 100, pop=0.50), metric="cbet")
    assert s.metric == "cbet"


# ── 6-9. HIGHER/LOWER/NEUTRAL/UNKNOWN ───────────────────────────────
def test_direction_higher():
    s = signal_from(_dev(0.55, 100, pop=0.25))
    assert s.direction == "HIGHER_THAN_POPULATION"
    assert s.z_score >= Z_SIGNIFICANT


def test_direction_lower():
    s = signal_from(_dev(0.05, 100, pop=0.25))
    assert s.direction == "LOWER_THAN_POPULATION"


def test_direction_neutral():
    s = signal_from(_dev(0.26, 100, pop=0.25))
    assert s.direction == "WITHIN_EXPECTED_RANGE"


def test_direction_unknown():
    s = signal_from(_dev(None, 100, pop=0.25))
    assert s.direction == "UNKNOWN"


# ── 10-12. small/medium/high sample ────────────────────────────────
def test_small_sample_insufficient():
    s = signal_from(_dev(0.9, 3, pop=0.25))
    assert s.status == "INSUFFICIENT"
    assert s.direction == "UNKNOWN"
    assert s.confidence == 0.0
    assert s.sample_size < MIN_SAMPLE_SIGNAL


def test_medium_sample():
    s = signal_from(_dev(0.55, 30, pop=0.25))
    assert s.status == "RECOGNIZED"
    assert s.confidence > 0


def test_high_sample():
    s = signal_from(_dev(0.55, 150, pop=0.25))
    assert s.status == "RECOGNIZED"
    assert s.sample_size >= 100
    assert s.confidence > 0


# ── 13-14. posterior-based / uncertainty ────────────────────────────
def test_posterior_based_deviation():
    prior = _prior(0.25)
    post = posterior_from(0.60, 8, prior, "vpip")
    dev = PopulationDeviation.from_posterior(post, player_id="p8")
    s = signal_from(dev)
    assert s.posterior is not None
    assert s.posterior != 0.60             # 收缩（不直接用观察值）


def test_uncertainty_considered():
    """同样 deviation、不同样本 → z 不同（统计显著性考虑样本量）。"""
    s_big = signal_from(_dev(0.40, 100, pop=0.25))
    s_small = signal_from(_dev(0.40, 6, pop=0.25))
    assert s_big.z_score is not None and s_small.z_score is not None
    assert abs(s_big.z_score) > abs(s_small.z_score)


# ── 15-16. multi-metric / isolation ─────────────────────────────────
def test_multi_metric_conservative():
    sigs = [signal_from(_dev(0.55, 100, pop=0.25), metric="vpip"),
            signal_from(_dev(0.35, 100, pop=0.15), metric="pfr"),
            signal_from(_dev(0.26, 100, pop=0.25), metric="three_bet")]
    sc = exploitability_from_signals(sigs, player_id="p1")
    assert sc.score is not None
    assert set(sc.supporting_metrics) == {"vpip", "pfr"}   # WITHIN 不计
    assert sc.is_signal_not_action is True


def test_cross_player_isolation():
    s1 = signal_from(_dev(0.55, 100, pop=0.25, player="p1"))
    s2 = signal_from(_dev(0.05, 100, pop=0.25, player="p2"))
    assert s1.player_id == "p1" and s2.player_id == "p2"
    assert s1.direction != s2.direction


# ── 17-19. provenance / synthetic / ≠ Action ────────────────────────
def test_provenance():
    s = signal_from(_dev(0.55, 100, pop=0.25))
    assert s.provenance == "deviation_signal_binomial_se_z"
    assert s.data_kind == "SYNTHETIC"


def test_synthetic_real_separation():
    s = signal_from(_dev(0.55, 100, pop=0.25), data_kind="SYNTHETIC")
    assert s.data_kind != "REAL"


def test_signal_not_action():
    s = signal_from(_dev(0.55, 100, pop=0.25))
    d = s.to_dict()
    assert d["is_signal_not_action"] is True
    # 信号对象无 action 字段（信号 ≠ 动作）
    assert "action" not in d
    assert "action" not in ExploitabilityScore().to_dict()


# ── 20. UNKNOWN propagation ─────────────────────────────────────────
def test_unknown_propagation():
    s = signal_from(_dev(None, 3, pop=0.25))
    assert s.direction == "UNKNOWN"
    assert s.confidence == 0.0
    sc = exploitability_from_signals([s])
    assert sc.score is None                    # 无显著 → 不强造


# ── 21-23. P30/P31/P32 regression ──────────────────────────────────
def test_p30_regression():
    from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
        frame_to_decision_request)
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png((Path("data/vision2/coinpoker_evidence") /
        "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"


def test_p31_regression():
    from poker_decision_ai.player_history import aggregate_player
    from poker_decision_ai.domain.poker.hand_history import parse_hand_history
    h = parse_hand_history((EV / "hand_70000.txt").read_text(
        encoding="utf-8"))
    assert aggregate_player(h.actions, "Lucy").hands == 1


def test_p32_regression():
    ds = aggregate_population(_texts(20))
    assert ds.metrics["global"]["vpip"] is not None
    assert ds.data_kind == "SYNTHETIC"


def _texts(n):
    return [(EV / f"hand_{i:05d}.txt").read_text(encoding="utf-8")
            for i in range(70000, 70000 + n)]


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}


# ══ P33 追加：主链 builder + UNKNOWN 门 + 置信封顶 ═══════════════════
def _hist(value=None, hands=100, metric="vpip"):
    h = PlayerHistoryStats(player_id="p1", hands=hands,
                           source="HAND_HISTORY",
                           provenance="test_history")
    if value is not None:
        setattr(h, metric, value)
    h.reliability = reliability_for(hands)
    return h


# ── A. builder 全链（P31 history + P32 prior/posterior → signal）────
def test_builder_full_chain_recognized():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(0.55, hands=100), prior=_prior(0.25))
    assert s.direction == "HIGHER_THAN_POPULATION"
    assert s.status == "RECOGNIZED"
    assert s.observed == 0.55
    assert s.population_prior == 0.25
    assert s.posterior is not None and s.posterior < 0.55   # 收缩语义
    assert s.deviation == 0.30
    assert s.sample_size == 100
    assert s.reliability == "HIGH"
    assert s.provenance == "deviation_signal_binomial_se_z"
    assert s.is_signal_not_action is True


def test_builder_chain_matches_manual_composition():
    """builder 结果 == 手工组合 posterior_from → from_posterior →
    signal_from（同一语义链——不另起炉灶）。"""
    prior = _prior(0.25)
    hist = _hist(0.60, hands=8)
    s = deviation_signal_for_player(player_id="p8", metric="vpip",
                                    history=hist, prior=prior)
    post = posterior_from(0.60, 8, prior, "vpip")
    manual = signal_from(PopulationDeviation.from_posterior(post,
                                                            player_id="p8"))
    assert s.posterior == manual.posterior == post.posterior
    assert s.direction == manual.direction
    assert post.posterior < 0.60             # P32-E 收缩语义继承


def test_builder_metric_supported_set():
    """仅 P31/P32 已聚合 metric 可产信号——不发明新指标。"""
    s = deviation_signal_for_player(
        player_id="p1", metric="made_up_metric",
        history=_hist(0.55), prior=_prior(0.25))
    assert s.direction == "UNKNOWN" and s.status == "UNKNOWN"
    assert s.provenance == "deviation_signal_metric_unsupported"


# ── B. 身份门（身份冲突 = UNKNOWN）──────────────────────────────────
def test_builder_identity_conflict_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(0.55, hands=100), prior=_prior(0.25),
        identity_status="CONFLICT")
    assert s.direction == "UNKNOWN" and s.status == "UNKNOWN"
    assert s.confidence == 0.0
    assert s.observed is None                 # 冲突身份→观察不可归属
    assert s.sample_size == 0 and s.reliability == "INSUFFICIENT"
    assert s.provenance == "deviation_signal_identity_conflict"


def test_builder_identity_unknown_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(0.55), prior=_prior(0.25),
        identity_status="UNKNOWN")
    assert s.direction == "UNKNOWN"
    assert s.provenance == "deviation_signal_identity_unknown"


# ── C. 数据门（无历史 / 无观察 / 越界 / 无 prior = UNKNOWN）─────────
def test_builder_no_history_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=None, prior=_prior(0.25))
    assert s.direction == "UNKNOWN" and s.status == "UNKNOWN"
    assert s.confidence == 0.0 and s.deviation is None
    assert s.provenance == "deviation_signal_no_history"


def test_builder_zero_hands_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(None, hands=0), prior=_prior(0.25))
    assert s.direction == "UNKNOWN"
    assert s.provenance == "deviation_signal_no_history"


def test_builder_no_observed_metric_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(None, hands=50), prior=_prior(0.25))
    assert s.direction == "UNKNOWN"
    assert s.provenance == "deviation_signal_no_observed_metric"


def test_builder_observed_out_of_range_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(1.5, hands=100), prior=_prior(0.25))
    assert s.direction == "UNKNOWN"
    assert s.provenance == "deviation_signal_observed_out_of_range"


def test_builder_no_prior_unknown():
    s = deviation_signal_for_player(
        player_id="p1", metric="vpip",
        history=_hist(0.55, hands=100), prior=None)
    assert s.direction == "UNKNOWN" and s.status == "UNKNOWN"
    assert s.confidence == 0.0
    assert s.observed == 0.55                  # 观察诚实保留，但不产信号
    assert s.deviation is None and s.posterior is None
    assert s.provenance == "deviation_signal_no_population_prior"


# ── D. 置信封顶（小样本 ≠ 高置信 exploit signal）────────────────────
def test_confidence_capped_by_reliability_tier():
    assert RELIABILITY_CONFIDENCE_CAP["LOW"] == 0.5
    assert RELIABILITY_CONFIDENCE_CAP["MEDIUM"] == 0.75
    s_lo = signal_from(_dev(0.95, 6, pop=0.25))      # z≈4.0 显著，但 LOW
    assert s_lo.direction == "HIGHER_THAN_POPULATION"
    assert s_lo.status == "RECOGNIZED"
    assert s_lo.confidence <= 0.5                     # 封顶：LOW ≠ 高置信
    s_md = signal_from(_dev(0.55, 30, pop=0.25))      # MEDIUM
    assert s_md.confidence <= 0.75
    s_hi = signal_from(_dev(0.55, 150, pop=0.25))     # HIGH
    assert s_hi.confidence <= 1.0
    assert s_lo.confidence < s_md.confidence < s_hi.confidence  # 单调


def test_exploitability_small_sample_capped():
    sc = exploitability_from_signals(
        [signal_from(_dev(0.95, 6, pop=0.25))], player_id="p1")
    assert sc.score is not None
    assert sc.score <= 0.5                    # 小样本 exploit 不给高置信


# ── E. 模块红线扫描（Deviation→Action / random = 0）────────────────
def test_no_action_verbs_in_signal_module():
    import re
    src = Path("poker_decision_ai/deviation_signal.py").read_text(
        encoding="utf-8")
    body = re.sub(r'""".*?"""', '', src, flags=re.S)
    hits = re.findall(r"\b(call|fold|raise|tag|lag|random)\b", body, re.I)
    assert hits == []                         # 信号层零动作语义零随机
    for s in (signal_from(_dev(0.55, 100)),
              deviation_signal_for_player(
                  player_id="p1", metric="vpip",
                  history=_hist(0.55), prior=_prior(0.25))):
        d = s.to_dict()
        assert d["is_signal_not_action"] is True
        assert "action" not in d
