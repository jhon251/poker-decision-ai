"""P32-E/F/H — Hierarchical Bayesian + PopulationDeviation（信号非 Action）。

层级：Population Prior → Player Observation → Player Posterior
（P31 Beta-Binomial 升级为 population-prior 参数化）。

PLAYER_BASED 判定（与 P31 range_mode_for 一致）：
  充分历史 → PLAYER_BASED（posterior 用 population prior 收缩）；
  不足 → POPULATION（用 prior）；
  无 population → UNKNOWN（绝不 random/TAG/LAG）。

PopulationDeviation：信号（observed vs population vs posterior）——
绝不直接映射 action。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .population import PopulationPrior, PopulationDataset
from .player_history import reliability_for


@dataclass
class PlayerPosterior:
    """sample-aware posterior（population prior 收缩——不直接用观察值）。"""
    metric: str
    posterior: float | None = None
    observed: float | None = None
    population_prior: float | None = None
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"metric": self.metric, "posterior": self.posterior,
                "observed": self.observed,
                "population_prior": self.population_prior,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "provenance": self.provenance}


def posterior_from(observed: float | None, sample_size: int,
                   prior: PopulationPrior | None,
                   metric: str, prior_strength: float = 10.0
                   ) -> PlayerPosterior:
    """层级收缩：posterior = (obs*n + prior*strength) / (n + strength)。

    prior=None（无 population）→ posterior=None（UNKNOWN——不伪造）。
    """
    pp = prior_to_value(prior) if prior else None
    if observed is None or pp is None or sample_size <= 0:
        return PlayerPosterior(metric=metric, posterior=None,
                               observed=observed,
                               population_prior=pp,
                               sample_size=sample_size,
                               reliability=reliability_for(sample_size),
                               provenance="hierarchical_bayes_unknown")
    shifted = (observed * sample_size + pp * prior_strength) \
        / (sample_size + prior_strength)
    return PlayerPosterior(
        metric=metric, posterior=round(shifted, 4), observed=observed,
        population_prior=pp, sample_size=sample_size,
        reliability=reliability_for(sample_size),
        provenance="hierarchical_bayes_population_prior")


def prior_to_value(prior: PopulationPrior) -> float | None:
    return prior.value if prior is not None and prior.value is not None \
        else None


@dataclass
class PopulationDeviation:
    """偏离信号（observed vs population vs posterior——信号非 Action）。"""
    player_id: str = ""
    metric: str = ""
    observed: float | None = None
    population: float | None = None
    posterior: float | None = None
    deviation: float | None = None     # observed - population
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    provenance: str = ""

    @classmethod
    def from_posterior(cls, post: PlayerPosterior,
                       player_id: str = "") -> "PopulationDeviation":
        if post.posterior is None or post.observed is None:
            return cls(player_id=player_id, metric=post.metric,
                       sample_size=post.sample_size,
                       reliability=post.reliability,
                       provenance="deviation_unknown")
        pop = post.population_prior
        return cls(
            player_id=player_id, metric=post.metric,
            observed=post.observed, population=pop,
            posterior=post.posterior,
            deviation=(round(post.observed - pop, 4)
                       if pop is not None else None),
            sample_size=post.sample_size,
            reliability=post.reliability,
            provenance="deviation_signal_population_prior")

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "metric": self.metric,
                "observed": self.observed, "population": self.population,
                "posterior": self.posterior,
                "deviation": self.deviation,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "provenance": self.provenance,
                "is_signal_not_action": True}
