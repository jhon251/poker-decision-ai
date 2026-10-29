"""Deterministic opponent frequency prior (MC1-R5).

从手牌库的动作观测统计出的分玩家/分位置/分街动作频率先验——与
`stats.py` 同层的确定性引擎（绝不 LLM 参与数值）。真实数据支撑
MC1-r4 的结论：朴素频率基线（Brier≈0.249）显著优于学习型响应模型
（Brier≈0.342）。

诚实边界（fail-closed / 不编造）：
  - 样本不足 → 回退群体先验并附 warning（source 标注层级）；
  - 完全无数据 → frequencies=None、source=UNAVAILABLE、NO_DATA warning；
    绝不产生任何编造的数字；
  - 时间衰减按观测级 ts（store 路径按桶 ts_max，provenance 声明粒度）；
  - 输出自带 Wilson 区间 + 样本量，供下游按证据强度加权。

输入观测的最小协议（由调用方从合法来源构造）：
    FrequencyObservation(player_id, street, facing_bet, action, position, ts)
context 类（动作合法性由是否面对下注决定）：
    facing_bet=True  → 合法动作 {fold, call, raise}
    facing_bet=False → 合法动作 {check, bet}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from poker_decision_ai.domain.poker.stats import time_decay_weights

#: 输出频率的动作类（任务契约：check/call/fold/bet/raise）
CONTEXT_ACTIONS_FACING_BET: tuple[str, ...] = ("fold", "call", "raise")
CONTEXT_ACTIONS_NO_BET: tuple[str, ...] = ("check", "bet")

#: 回退层级（先个体后群体；首层满足 min_sample_size 即用）
FALLBACK_LEVELS: tuple[tuple[str, ...], ...] = (
    ("player", "position", "street"),
    ("player", "position"),
    ("player", "street"),
    ("player",),
    ("population", "position", "street"),
    ("population", "street"),
    ("population",),
)

DEFAULT_MIN_SAMPLE_SIZE = 20
DEFAULT_HALF_LIFE_DAYS = 90.0
DEFAULT_WILSON_Z = 1.96


@dataclass
class FrequencyPriorConfig:
    """全部数值声明于 configs/config.yaml::frequency_prior（无代码写死）。"""
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    wilson_z: float = DEFAULT_WILSON_Z

    @classmethod
    def from_mapping(cls, cfg: dict | None) -> "FrequencyPriorConfig":
        cfg = dict(cfg or {})
        return cls(
            min_sample_size=int(cfg.get("min_sample_size", DEFAULT_MIN_SAMPLE_SIZE)),
            half_life_days=float(cfg.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)),
            wilson_z=float(cfg.get("wilson_z", DEFAULT_WILSON_Z)),
        )

    def to_dict(self) -> dict:
        return {"min_sample_size": self.min_sample_size,
                "half_life_days": self.half_life_days,
                "wilson_z": self.wilson_z}


@dataclass
class FrequencyObservation:
    """一次决策级动作观测（调用方从合法来源构造；禁止伪造）。"""
    player_id: str
    street: str
    facing_bet: bool
    action: str                    # check|call|fold|bet|raise
    position: str | None = None
    ts: float | None = None        # epoch seconds；None → 不衰减（等权）

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "street": self.street,
                "facing_bet": self.facing_bet, "action": self.action,
                "position": self.position, "ts": self.ts}


@dataclass
class ActionFreq:
    """单动作频率 + Wilson 区间 + 样本量（n=原始整数计数；n_eff=衰减加权有效样本）。"""
    action: str
    rate: float
    wilson_low: float
    wilson_high: float
    n: int
    n_eff: float

    def to_dict(self) -> dict:
        return {"action": self.action, "rate": round(self.rate, 4),
                "wilson_low": round(self.wilson_low, 4),
                "wilson_high": round(self.wilson_high, 4),
                "n": self.n, "n_eff": round(self.n_eff, 3)}


@dataclass
class FrequencyPrior:
    player_id: str
    position: str | None
    street: str | None
    facing_bet: bool | None
    source: str                    # player_* | population_* | UNAVAILABLE
    context_actions: list          # 该 context 的合法动作类
    frequencies: dict | None       # action -> ActionFreq.to_dict()；无数据 → None
    p_continue: float | None       # facing_bet: call+raise；no_bet: bet
    p_fold: float | None           # facing_bet: fold；no_bet: None
    n_total: int
    n_eff_total: float
    warnings: list = field(default_factory=list)
    config_echo: dict = field(default_factory=dict)
    decay_granularity: str = "observation_ts"

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id, "position": self.position,
            "street": self.street, "facing_bet": self.facing_bet,
            "source": self.source, "context_actions": list(self.context_actions),
            "frequencies": self.frequencies,
            "p_continue": (round(self.p_continue, 4)
                           if self.p_continue is not None else None),
            "p_fold": (round(self.p_fold, 4)
                       if self.p_fold is not None else None),
            "n_total": self.n_total,
            "n_eff_total": round(self.n_eff_total, 3),
            "warnings": list(self.warnings),
            "provenance": {"source_type": "frequency_baseline",
                           "config": dict(self.config_echo),
                           "decay_granularity": self.decay_granularity},
        }


def _context_actions(facing_bet: bool | None) -> tuple[str, ...]:
    if facing_bet is True:
        return CONTEXT_ACTIONS_FACING_BET
    if facing_bet is False:
        return CONTEXT_ACTIONS_NO_BET
    return CONTEXT_ACTIONS_FACING_BET + CONTEXT_ACTIONS_NO_BET


def _weighted_rates(obs: list, actions: tuple[str, ...],
                    cfg: FrequencyPriorConfig) -> tuple[dict, int, float]:
    """时间衰减加权频率 + 逐动作加权 Wilson 区间。

    衰减：stats.time_decay_weights（ts=None 的观测等权，声明语义）。
    Wilson：按有效样本量 n_eff=Σw 计算（加权 Wilson——声明语义）。
    """
    ts = [o.ts for o in obs if o.ts is not None]
    if ts and len(ts) == len(obs):
        weights = time_decay_weights(ts, half_life_days=cfg.half_life_days)
    else:
        weights = [1.0] * len(obs)     # 混合缺 ts / 全缺 ts → 等权（诚实降级）
    total_w = sum(weights)
    n_int = len(obs)
    out: dict[str, ActionFreq] = {}
    for a in actions:
        hits = [w for o, w in zip(obs, weights) if o.action == a]
        succ_w = sum(hits)
        rate = (succ_w / total_w) if total_w > 0 else 0.0
        if n_eff := total_w:
            lo, hi = _wilson_float(succ_w, n_eff, cfg.wilson_z)
        else:
            lo, hi = 0.0, 1.0
        out[a] = ActionFreq(action=a, rate=rate, wilson_low=lo,
                            wilson_high=hi,
                            n=sum(1 for o in obs if o.action == a),
                            n_eff=total_w)
    return out, n_int, total_w


def _wilson_float(successes: float, trials: float,
                  z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间（浮点有效样本量版——加权 Wilson，声明语义）。"""
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def action_frequencies(observations: Iterable[FrequencyObservation], *,
                       player_id: str, facing_bet: bool | None,
                       position: str | None = None, street: str | None = None,
                       config: FrequencyPriorConfig | None = None) -> FrequencyPrior:
    """确定性频率先验查询：玩家/位置/街/context → 动作频率 + Wilson + n + 衰减。

    回退链（首层满足 min_sample_size 即用；个体层全不满足 → 群体层；
    群体层样本仍不足 → 用最大群体层并附 INSUFFICIENT_SAMPLE warning；
    完全无数据 → UNAVAILABLE + NO_DATA，frequencies=None）。
    """
    cfg = config or FrequencyPriorConfig()
    actions = _context_actions(facing_bet)
    pool = [o for o in (observations or [])
            if isinstance(o, FrequencyObservation)
            and o.action in actions
            and (facing_bet is None or o.facing_bet == facing_bet)]

    warnings: list[dict] = []
    if facing_bet is None:
        warnings.append({"code": "CONTEXT_UNSPLIT",
                         "message": "query without facing_bet context; "
                                    "frequencies pool both contexts"})

    def _match(level: tuple[str, ...], o: FrequencyObservation) -> bool:
        for dim in level:
            if dim == "player" and o.player_id != player_id:
                return False
            if dim == "position" and position is not None \
                    and o.position != position:
                return False
            if dim == "street" and street is not None and o.street != street:
                return False
        return True

    chosen_level: tuple[str, ...] | None = None
    chosen: list[FrequencyObservation] = []
    for level in FALLBACK_LEVELS:
        subset = [o for o in pool if _match(level, o)]
        if len(subset) >= cfg.min_sample_size:
            chosen_level, chosen = level, subset
            break
        # 记住可用但不达阈值的最大群体层（兜底）
    if chosen_level is None:
        pop_levels = [lv for lv in FALLBACK_LEVELS if "population" in lv]
        best_level, best = None, []
        for level in pop_levels:
            subset = [o for o in pool if _match(level, o)]
            if len(subset) > len(best):
                best_level, best = level, subset
        if not best:
            warnings.append({"code": "NO_DATA",
                             "message": (f"no observations for player={player_id!r} "
                                         f"context(facing_bet={facing_bet}); "
                                         "no frequencies fabricated")})
            return FrequencyPrior(
                player_id=player_id, position=position, street=street,
                facing_bet=facing_bet, source="UNAVAILABLE",
                context_actions=list(actions), frequencies=None,
                p_continue=None, p_fold=None, n_total=0, n_eff_total=0.0,
                warnings=warnings, config_echo=cfg.to_dict())
        chosen_level, chosen = best_level, best
        warnings.append({"code": "INSUFFICIENT_SAMPLE",
                         "message": (f"below min_sample_size={cfg.min_sample_size}; "
                                     f"using largest population level "
                                     f"({'×'.join(chosen_level)}) n={len(chosen)}")})
    elif chosen_level[0] == "population":
        warnings.append({"code": "FALLBACK_POPULATION",
                         "message": (f"player sample below min_sample_size="
                                     f"{cfg.min_sample_size}; fell back to "
                                     f"{'×'.join(chosen_level)}")})

    rates, n_int, n_eff = _weighted_rates(chosen, actions, cfg)
    source = "×".join(chosen_level)
    p_continue: float | None = None
    p_fold: float | None = None
    if n_int > 0:
        if facing_bet is True:
            p_continue = (rates["call"].rate + rates["raise"].rate)
            p_fold = rates["fold"].rate
        elif facing_bet is False:
            p_continue = rates["bet"].rate
    return FrequencyPrior(
        player_id=player_id, position=position, street=street,
        facing_bet=facing_bet, source=source,
        context_actions=list(actions),
        frequencies={a: rates[a].to_dict() for a in actions},
        p_continue=p_continue, p_fold=p_fold,
        n_total=n_int, n_eff_total=n_eff,
        warnings=warnings, config_echo=cfg.to_dict())


# ----------------------------------------------------------------------
# 观测构造器（从合法来源；重放 street 推导 facing_bet——绝不猜测）
# ----------------------------------------------------------------------

_NORMALIZE_ACTION = {"allin": "raise", "all-in": "raise", "cbr": "raise",
                     "cc": "call", "f": "fold", "sm": None, "post": None}


def observations_from_street_actions(
        street_actions: list, *, street: str,
        positions_by_player: dict | None = None,
        ts: float | None = None) -> list[FrequencyObservation]:
    """把一条街的动作行重放为决策级观测（facing_bet 由重放推导）。

    行协议（最小）：{player_id, action, amount?, to?}；action ∈
    post|check|call|fold|bet|raise|allin（大小写不敏感）。
    `bet` 语义 = 本街无人下注时的进攻；`raise` = 已有下注后的进攻；
    `to`（如有）= 进攻后的累计承诺额。post/未知动作跳过（不产生观测）。
    """
    positions_by_player = dict(positions_by_player or {})
    street_max = 0.0
    committed: dict[str, float] = {}
    out: list[FrequencyObservation] = []
    for row in street_actions or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or "")
        act = str(row.get("action") or "").lower()
        if not pid or act in ("post", ""):
            continue
        facing = street_max > 0
        amount = row.get("amount")
        to = row.get("to")
        norm = _NORMALIZE_ACTION.get(act, act)
        if norm not in ("check", "call", "fold", "bet", "raise"):
            continue
        if norm == "bet":
            new_max = float(to) if to is not None else float(amount or 0.0)
        elif norm == "raise":
            new_max = float(to) if to is not None else \
                street_max + float(amount or 0.0)
        elif norm == "call":
            new_max = street_max
        else:
            new_max = None
        out.append(FrequencyObservation(
            player_id=pid, street=street, facing_bet=facing, action=norm,
            position=positions_by_player.get(pid), ts=ts))
        if norm in ("bet", "raise") and new_max is not None:
            street_max = max(street_max, new_max)
            committed[pid] = max(committed.get(pid, 0.0), new_max)
        elif norm == "call":
            committed[pid] = max(committed.get(pid, 0.0), street_max)
    return out
