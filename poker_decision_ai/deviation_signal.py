"""P33 — Deviation Signal & Exploitability Layer（Signal ≠ Action）。

DeviationSignal（P33-A）：
  player_id/metric/observed/population_prior/posterior/deviation/
  sample_size/reliability/direction/confidence/source/provenance

direction（P33-C——仅统计方向，**无扑克动作语义**）：
  HIGHER_THAN_POPULATION / LOWER_THAN_POPULATION /
  WITHIN_EXPECTED_RANGE / UNKNOWN

P33-B 统计显著性：不只 observed-population——纳入
  sample_size（二项 SE）、population/posterior 不确定性；
  信息不足 → UNKNOWN/INSUFFICIENT（不强行产生 exploit signal）。

P33-D 多指标：保守组合（去重 metric；小样本不放大；第一版宁简单）。

P33-A 主链 builder：deviation_signal_for_player——
  Population Prior + Player History + Player Posterior → DeviationSignal；
  fail-closed UNKNOWN 门：身份非 RECOGNIZED（含 CONFLICT）/ 无 history /
  无 observed / 无 population prior / observed 越界 → UNKNOWN（不伪造）。

P33-B2 置信封顶（contract 冻结）：confidence 上限由 reliability tier
  决定（INSUFFICIENT=0 / LOW=0.5 / MEDIUM=0.75 / HIGH=1.0）——
  小样本结构性到不了高置信（小样本 ≠ 高置信 exploit signal）。

P33-E ExploitabilityScore：score/confidence/supporting_metrics/
  sample_size/provenance——**SIGNAL**（绝不产生 raise_more/call_more）。

P33-F 边界：DeviationSignal → Strategy INPUT（合法链
  Signal → Opponent Profile → Range/Strategy → Judge → Legal Action）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .population_prior import (PopulationDeviation, PlayerPosterior,
                               posterior_from)
from .population import PopulationPrior
from .player_history import PlayerHistoryStats, reliability_for


#: 方向枚举（仅统计方向——无动作语义）
DIRECTIONS = ("HIGHER_THAN_POPULATION", "LOWER_THAN_POPULATION",
              "WITHIN_EXPECTED_RANGE", "UNKNOWN")

#: 显著性阈值（contract 冻结——z 分数下方向性成立；< 2.0 → WITHIN）
Z_SIGNIFICANT = 2.0
#: Exploitability 最低样本（insufficient → UNKNOWN）
MIN_SAMPLE_SIGNAL = 5
#: 置信封顶（contract 冻结——P33-B2：reliability tier 直接封顶
#: confidence；小样本结构性到不了高置信 exploit signal；未知 tier → 0
#: fail-closed）
RELIABILITY_CONFIDENCE_CAP = {
    "INSUFFICIENT": 0.0,
    "LOW": 0.5,
    "MEDIUM": 0.75,
    "HIGH": 1.0,
}
#: P31 PlayerHistoryStats 上受支持的 metric（仅 P31/P32 已合法聚合的
#: 统计——不发明新指标；其它 metric → UNKNOWN）
HISTORY_METRICS = ("vpip", "pfr", "three_bet", "cbet")


def _binomial_se(p: float, n: int) -> float | None:
    """二项标准误（样本不确定性）。"""
    if n <= 0 or p is None or not (0.0 <= p <= 1.0):
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def direction_of(deviation: float | None, z: float | None) -> str:
    """deviation + z → 方向（统计显著才 HIGHER/LOWER——否则 WITHIN）。"""
    if deviation is None or z is None:
        return "UNKNOWN"
    if z >= Z_SIGNIFICANT:
        return "HIGHER_THAN_POPULATION"
    if z <= -Z_SIGNIFICANT:
        return "LOWER_THAN_POPULATION"
    return "WITHIN_EXPECTED_RANGE"


@dataclass
class DeviationSignal:
    """P33-A Contract（Signal——绝不直接映射动作）。"""
    player_id: str = ""
    metric: str = ""
    observed: float | None = None
    population_prior: float | None = None
    posterior: float | None = None
    deviation: float | None = None
    z_score: float | None = None
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    direction: str = "UNKNOWN"
    confidence: float = 0.0
    status: str = "UNKNOWN"            # RECOGNIZED | INSUFFICIENT | UNKNOWN
    source: str = ""                   # HAND_HISTORY | EVENT_STREAM
    data_kind: str = "SYNTHETIC"
    provenance: str = ""
    is_signal_not_action: bool = True

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "metric": self.metric,
                "observed": self.observed,
                "population_prior": self.population_prior,
                "posterior": self.posterior,
                "deviation": self.deviation, "z_score": self.z_score,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "direction": self.direction,
                "confidence": self.confidence, "status": self.status,
                "source": self.source, "data_kind": self.data_kind,
                "provenance": self.provenance,
                "is_signal_not_action": self.is_signal_not_action}


def signal_from(dev: PopulationDeviation, *,
                metric: str = "vpip",
                source: str = "HAND_HISTORY",
                data_kind: str = "SYNTHETIC") -> DeviationSignal:
    """P32 PopulationDeviation + 统计显著性 → DeviationSignal。

    - z = deviation / SE（SE 由 sample_size + population 先验近似；
      population 不确定性并入 strength 项——保守）；
    - z 不足 / 样本不足 → direction=WITHIN/UNKNOWN + confidence 低；
    - 信息不足 → status=INSUFFICIENT（不强行产生 exploit signal）。
    """
    n = int(dev.sample_size or 0)
    obs, pop, post = dev.observed, dev.population, dev.posterior
    if obs is None or pop is None or n < MIN_SAMPLE_SIGNAL:
        return DeviationSignal(
            player_id=dev.player_id, metric=metric, observed=obs,
            population_prior=pop, posterior=post, deviation=dev.deviation,
            sample_size=n, reliability=dev.reliability,
            direction="UNKNOWN", confidence=0.0, status="INSUFFICIENT",
            source=source, data_kind=data_kind,
            provenance="deviation_signal_insufficient_sample")
    se = _binomial_se(pop, n)
    if se is None or se <= 0:
        return DeviationSignal(
            player_id=dev.player_id, metric=metric, observed=obs,
            population_prior=pop, posterior=post, deviation=dev.deviation,
            sample_size=n, reliability=dev.reliability,
            direction="UNKNOWN", confidence=0.0, status="INSUFFICIENT",
            source=source, data_kind=data_kind,
            provenance="deviation_signal_degenerate_se")
    z = (obs - pop) / se
    direction = direction_of(dev.deviation, z)
    # confidence：z 显著性与样本可靠性合成（0..1——保守），
    # 再按 reliability tier 封顶（P33-B2——小样本 ≠ 高置信）
    z_conf = min(1.0, abs(z) / 4.0) if z is not None else 0.0
    sample_conf = min(1.0, n / 100.0)
    cap = RELIABILITY_CONFIDENCE_CAP.get(dev.reliability, 0.0)
    confidence = round(min(0.6 * z_conf + 0.4 * sample_conf, cap), 4)
    # WITHIN_EXPECTED_RANGE 也是 recognized 结论（"样本充分、无显著偏离"）；
    # 只有数据/样本不足才走上方 INSUFFICIENT 早退分支。
    status = "RECOGNIZED"
    return DeviationSignal(
        player_id=dev.player_id, metric=metric, observed=obs,
        population_prior=pop, posterior=post, deviation=dev.deviation,
        z_score=round(z, 4), sample_size=n, reliability=dev.reliability,
        direction=direction, confidence=confidence, status=status,
        source=source, data_kind=data_kind,
        provenance="deviation_signal_binomial_se_z")


@dataclass
class ExploitabilityScore:
    """P33-E —— SIGNAL（score+confidence+supporting metrics；无动作）。"""
    player_id: str = ""
    score: float | None = None         # 0..1 保守合成
    confidence: float = 0.0
    supporting_metrics: list = field(default_factory=list)
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    provenance: str = ""
    is_signal_not_action: bool = True

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "score": self.score,
                "confidence": self.confidence,
                "supporting_metrics": list(self.supporting_metrics),
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "provenance": self.provenance,
                "is_signal_not_action": self.is_signal_not_action}


def exploitability_from_signals(signals: list[DeviationSignal],
                                player_id: str = "") -> ExploitabilityScore:
    """P33-D 多指标保守组合（去重 metric；小样本不放大）。

    - 仅计入 direction ∈ {HIGHER, LOWER}（WITHIN 不计 exploit）；
    - 每指标权重 = confidence（去重 metric 取最高）；
    - 全部不足 → score=None（UNKNOWN——不强行）。
    """
    best: dict[str, DeviationSignal] = {}
    for s in signals or []:
        if s.direction in ("HIGHER_THAN_POPULATION",
                           "LOWER_THAN_POPULATION"):
            cur = best.get(s.metric)
            if cur is None or s.confidence > cur.confidence:
                best[s.metric] = s
    if not best:
        sample = max((s.sample_size for s in (signals or [])), default=0)
        return ExploitabilityScore(
            player_id=player_id, score=None, confidence=0.0,
            supporting_metrics=[], sample_size=sample,
            reliability=reliability_for(sample),
            provenance="exploitability_no_significant_deviation")
    score = sum(s.confidence for s in best.values()) / len(best)
    sample = min(s.sample_size for s in best.values())
    return ExploitabilityScore(
        player_id=player_id, score=round(score, 4),
        confidence=round(score, 4),
        supporting_metrics=sorted(best.keys()),
        sample_size=sample,
        reliability=reliability_for(sample),
        provenance="exploitability_conservative_mean")


def deviation_signal_for_player(
        *, player_id: str, metric: str,
        history: PlayerHistoryStats | None,
        prior: PopulationPrior | None,
        identity_status: str = "RECOGNIZED",
        source: str = "HAND_HISTORY",
        data_kind: str = "",
        prior_strength: float = 10.0) -> DeviationSignal:
    """P33-A 主链：Population Prior + Player History + Player Posterior
    → DeviationSignal（fail-closed UNKNOWN 门——顺序固定）。

    1. 身份非 RECOGNIZED（UNKNOWN/CONFLICT）→ UNKNOWN（身份冲突绝不产
       信号——观察无法归属到玩家）；
    2. metric 不在 P31/P32 受支持集合 → UNKNOWN（不发明指标）；
    3. 无 history / hands<=0 → UNKNOWN（无历史 ≠ 伪造观察）；
    4. observed 缺失 / 越界（非 0..1）→ UNKNOWN；
    5. 无 population prior → UNKNOWN（posterior 无收缩基线——不伪造）；
    6. 全部通过 → posterior_from → PopulationDeviation → signal_from
       （样本/z 门照常生效——小样本仍 INSUFFICIENT/UNKNOWN）。

    UNKNOWN 信号：direction=UNKNOWN / confidence=0 / status=UNKNOWN——
    绝不 fallback 到 random 或任何动作语义。
    """
    def _unknown(provenance: str, *, observed: float | None = None,
                 pop: float | None = None, n: int = 0,
                 rel: str = "INSUFFICIENT") -> DeviationSignal:
        dk = data_kind or (prior.data_kind if prior is not None
                           else "SYNTHETIC")
        return DeviationSignal(
            player_id=player_id, metric=metric, observed=observed,
            population_prior=pop, posterior=None, deviation=None,
            sample_size=n, reliability=rel, direction="UNKNOWN",
            confidence=0.0, status="UNKNOWN", source=source,
            data_kind=dk, provenance=provenance)

    if identity_status != "RECOGNIZED":
        reason = ("deviation_signal_identity_conflict"
                  if identity_status == "CONFLICT"
                  else "deviation_signal_identity_unknown")
        return _unknown(reason)
    if metric not in HISTORY_METRICS:
        return _unknown("deviation_signal_metric_unsupported")
    hands = int(getattr(history, "hands", 0) or 0) if history else 0
    if history is None or hands <= 0:
        return _unknown("deviation_signal_no_history",
                        pop=(prior.value if prior is not None else None))
    observed = getattr(history, metric, None)
    if observed is None:
        return _unknown("deviation_signal_no_observed_metric",
                        pop=(prior.value if prior is not None else None),
                        n=hands, rel=history.reliability)
    if not isinstance(observed, (int, float)) \
            or not (0.0 <= float(observed) <= 1.0):
        return _unknown("deviation_signal_observed_out_of_range",
                        n=hands, rel=history.reliability)
    if prior is None or prior.value is None:
        return _unknown("deviation_signal_no_population_prior",
                        observed=observed, n=hands,
                        rel=history.reliability)
    post = posterior_from(float(observed), hands, prior, metric,
                          prior_strength=prior_strength)
    dev = PopulationDeviation.from_posterior(post, player_id=player_id)
    dk = data_kind or (prior.data_kind if prior is not None
                       else "SYNTHETIC")
    return signal_from(dev, metric=metric, source=source, data_kind=dk)
