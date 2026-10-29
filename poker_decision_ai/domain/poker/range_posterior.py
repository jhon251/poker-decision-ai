"""P46 — Data-driven Villain Range Posterior（纯数学；仅真实历史；禁止伪造）。

目标（P46 授权范围）：把 Range Agent 的
    `configs/ranges.yaml 静态 base range × 0.5/1.5 启发式`
升级为可审计的数据驱动 posterior——**不新增 Agent**，只在上游改分布，
Equity / Fold / Strategy Core 通过既有 `villains[pid].distribution`
接口无感使用（strategy_core.compute_ev 的 range_src 契约不变）。

数据来源（唯一合法源）：repos.hands 中携带 "parsed" 键的手牌记录经
`conditional_model.extract_events` 产出的条件事件（与 P43 同源同管）。
无 parsed 历史 / 样本不足 → BASELINE（绝不伪造 posterior）。

两类目标（kind）：
  starting（preflop）: P(enter pot | position) 或 P(raise | position)——
        由 preflop 条件事件按 (hand, player) 去重聚合成 entry/raise 率。
        open/cold_call → entry 率；three_bet/four_bet → raise 率。
  continuing（flop/turn/river）: P(continue | street, facing=bet, ...)——
        复用 P43 ConditionalFoldIndex（player→population 阶梯已内建），
        目标继续率 = 1 − p_fold（条件估计，内部已向 population 收缩）。

后验数学（显式声明，非 GTO、非精确贝叶斯范围反演）：
  1) 率的 Beta-Binomial 收缩（复用 stats.bayes_shrink，先验 =
     population 同条件率，否则 baseline 隐含率——与 P43 阶梯同构）；
  2) 偏差 d = (rate_post − rate_base) / max(rate_base, ε)，clip 到
     ±DEVIATION_CAP；
  3) 强度倾斜 w_c ← w_c · exp(λ·s̃_c)，λ = −SENSITIVITY·d（松玩家
     d>0 → λ<0 → 相对弱组合加权；紧玩家反向），clip ±LAMBDA_CAP；
  4) 支撑集 = baseline 支撑集（v1 不外扩——loosening 只在既有支撑内
     重分配，声明为已知限制）。

强度评分 s_c（确定性、seeded MC，模块级缓存）：
  starting  : 该 combo 类 vs 随机一手牌的 preflop equity；
  continuing: 该 combo 类在当前 board 上 vs 随机一手牌的 equity
              （P50-2 F2 修复：hero-agnostic 校准——board 条件 +
              hero 死牌剔除保留，hero 手牌强度不再作为条件）。
  s̃ = (s − mean)/std（支撑集内标准化）——评分只用于单调倾斜方向。

UNKNOWN / fallback 阶梯（与 P43 同纪律）：
  player 精确 position ≥ MIN_SAMPLES → player 粗化(全位置) ≥ MIN_SAMPLES
  → population（同阶梯）→ BASELINE（fallback_reason 记录原因）。
  continuing 还要求 hero 两张 hole cards + 合法 board，缺失 → BASELINE。

确定性：相同输入 → 相同输出（seeded MC、固定阶梯、无全局随机、无 IO）。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .cards import Card, parse_card
from .conditional_model import (
    MIN_SAMPLES as _COND_MIN_SAMPLES,
    size_bucket as _size_bucket,
    texture_tag,
)
from .equity import evaluate7, remaining_deck
from .ranges import Range, _total_assignments
from .stats import bayes_shrink

# ---- contract 常量（与 P42/P43 同族；禁止为通过率调低门槛） ----
MIN_SAMPLES = _COND_MIN_SAMPLES      # 30：率估计进入 posterior 的样本门槛
PRIOR_STRENGTH = 200.0               # Beta-Binomial 先验伪计数（P43 同值）
#: 相对偏差 clip（|d| 上限——极端率差异不产生极端倾斜）
DEVIATION_CAP = 1.0
#: 倾斜灵敏度（d=1.0 → λ=−1.0 → 弱组合相对权重 ~e^1）；声明式一阶响应
SENSITIVITY = 1.0
#: |λ| 上限（防数值噪声冒充极端松/紧）
LAMBDA_CAP = 1.2
#: 继续率标定的参考下注尺寸（占底池比；"典型下注"声明，非当前候选尺寸）
REF_BET_FRACTION = 0.66
#: starting 强度评分：每 combo 类 vs 随机手的 MC runouts（模块级缓存）
STRENGTH_RUNOUTS = 16
STRENGTH_SEED = 42
#: continuing 强度评分：每 combo 类 vs hero 的 MC runouts（每决策计算）
CONT_RUNOUTS = 12

_P46_ID = "P46.1.0"
_EPS = 1e-6


# ─────────────────────────── 事件索引（starting 证据） ─────────────────────
class RangePosteriorIndex:
    """preflop entry/raise 率的条件计数索引（按 position 阶梯；可序列化）。

    事件来源 = conditional_model.extract_events 的输出；按 (hand_id, player)
    去重聚合（一手牌一玩家至多贡献 1 个 entry 机会 + 1 个 raise 机会），
    与 VPIP/PFR 的"按手计"口径一致。position 缺失（button_seat 缺失等
    不完整牌局）→ 计入 ""（粗化池，仅作阶梯降级用，永不冒充精确位置）。
    """

    def __init__(self) -> None:
        # key -> {"entry": [hands, entered], "raise": [facing_hands, raised]}
        self._counts: dict[str, dict[str, list[int]]] = {}

    def add_events(self, events: list[dict]) -> None:
        """单手牌的事件列表 → 聚合（确定性；不合法输入忽略）。"""
        by_hp: dict[tuple[str, str], list[dict]] = {}
        for ev in events or []:
            if not isinstance(ev, dict) or ev.get("street") != "preflop":
                continue
            hp = (str(ev.get("hand_id") or ""), str(ev.get("player") or ""))
            if not hp[1]:
                continue
            by_hp.setdefault(hp, []).append(ev)
        for (_hid, _pid), evs in by_hp.items():
            pos = str(evs[0].get("position") or "")
            entered = any(e.get("action") in ("call", "raise", "bet") for e in evs)
            facing = [e for e in evs if e.get("facing") not in (None, "", "none")]
            raised = any(e.get("action") in ("raise", "bet") for e in facing)
            for key in (pos, ""):
                cell = self._counts.setdefault(
                    key, {"entry": [0, 0], "raise": [0, 0]})
                cell["entry"][0] += 1
                cell["entry"][1] += 1 if entered else 0
                if facing:
                    cell["raise"][0] += 1
                    cell["raise"][1] += 1 if raised else 0

    def rate(self, position: str, metric: str = "entry",
             min_samples: int = MIN_SAMPLES) -> dict | None:
        """条件率（position 阶梯：精确 → 全位置粗化；样本不足 → None）。

        返回 {"rate", "samples", "position_key", "degraded"}；metric:
        "entry"（进入底池率）| "raise"（面对加注时的加注率）。
        """
        cell = self._counts.get(str(position or ""))
        degraded = False
        if cell is None or cell[metric][0] < min_samples:
            cell = self._counts.get("")
            degraded = True
            if cell is None or cell[metric][0] < min_samples:
                return None
        n, hits = cell[metric]
        return {"rate": hits / n if n else 0.0, "samples": n,
                "position_key": str(position or "") if not degraded else "",
                "degraded": degraded}

    @property
    def total_hands(self) -> int:
        """去重后的 (hand, player) 决策机会总数（粗化池口径）。"""
        cell = self._counts.get("")
        return cell["entry"][0] if cell else 0

    def to_dict(self) -> dict:
        return {"counts": {k: {m: list(v) for m, v in cell.items()}
                           for k, cell in sorted(self._counts.items())}}

    @classmethod
    def from_dict(cls, d: dict | None) -> "RangePosteriorIndex":
        idx = cls()
        for k, cell in ((d or {}).get("counts") or {}).items():
            if not isinstance(cell, dict):
                continue
            ok = True
            conv: dict[str, list[int]] = {}
            for m in ("entry", "raise"):
                v = cell.get(m)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    conv[m] = [int(v[0]), int(v[1])]
                else:
                    ok = False
                    break
            if ok:
                idx._counts[str(k)] = conv
        return idx


# ─────────────────────────── 强度评分（seeded、缓存） ─────────────────────
_STRENGTH_CACHE: dict[tuple[int, int, bool], float] = {}


def combo_strength_preflop(combo: tuple[int, int, bool]) -> float:
    """combo 类 vs 随机一手牌的 preflop equity（seeded MC；模块级缓存）。

    确定性：seed=STRENGTH_SEED；评分仅用于倾斜方向（单调序），粗粒度 MC
    已声明（A-P46-strength）。
    """
    key = (combo[0], combo[1], combo[2])
    hit = _STRENGTH_CACHE.get(key)
    if hit is not None:
        return hit
    from .ranges import combo_to_cards
    hero = combo_to_cards(combo)
    rng = random.Random(STRENGTH_SEED + key[0] * 100 + key[1] * 4
                        + (1 if key[2] else 0))
    base_rem = remaining_deck(hero)
    wins = ties = 0
    for _ in range(STRENGTH_RUNOUTS):
        villain = rng.sample(base_rem, 2)       # 均匀 1326 组合（无类别偏置）
        board = rng.sample([c for c in base_rem if c not in villain], 5)
        hs = evaluate7(hero + board)
        vs = evaluate7(villain + board)
        if hs > vs:
            wins += 1
        elif hs == vs:
            ties += 1
    score = (wins + 0.5 * ties) / STRENGTH_RUNOUTS
    _STRENGTH_CACHE[key] = score
    return score


def _combo_equities_vs_hero(vr: Range, hero_cards: list[Card],
                            board: list[Card], seed: int = STRENGTH_SEED
                            ) -> dict[tuple[int, int, bool], float] | None:
    """每个 live combo 类 vs hero 手牌的 equity（villain 视角；seeded MC）。

    与 fold_equity 范围通道同构（river 精确）；失败（牌冲突等）→ None。
    """
    if len(hero_cards) != 2 or not 3 <= len(board) <= 5:
        return None
    dead = {c.code for c in hero_cards + board}
    needed = 5 - len(board)
    rng = random.Random(seed)
    hs_cache = evaluate7(hero_cards + board) if needed == 0 else None
    from .ranges import sample_concrete_pair
    out: dict[tuple[int, int, bool], float] = {}
    for combo in vr.weights:
        hi, lo, suited = combo
        wins = ties = 0
        used = 0
        for _ in range(CONT_RUNOUTS):
            c1 = sample_concrete_pair(rng, hi, lo, suited, hero_cards + board)
            if c1 is None:
                continue
            rem = remaining_deck(hero_cards + board + c1)
            if needed == 0:
                hs = hs_cache
                vs = evaluate7(c1 + board)
            else:
                full = board + rng.sample(rem, needed)
                hs = evaluate7(hero_cards + full)
                vs = evaluate7(c1 + full)
            if vs > hs:
                wins += 1
            elif vs == hs:
                ties += 1
            used += 1
        if used:
            out[combo] = (wins + 0.5 * ties) / used
    return out or None


def _combo_strength_vs_random(vr: Range, board: list[Card],
                              dead_cards: list[Card],
                              seed: int = STRENGTH_SEED
                              ) -> dict[tuple[int, int, bool], float] | None:
    """P50-2（F2 修复）：每个 live combo 在固定 board 上 vs **随机一手牌**
    的 equity——行为校准的 hero-agnostic 强度评分。

    条件区分（F2 语义）：
      - board 条件：保留（continuing 行为本来就是街/牌面条件化的）；
      - card removal：尊重 hero 死牌（hero 已知牌从对手采样池剔除——
        这是 hero 条件中"合法"的部分）；
      - hero 手牌强度：**不再作为条件**——此前 vs-hero equity 使
        baseline continue 率随 hero 换牌大幅漂移（P49 F2 伪影）。
    """
    if not 3 <= len(board) <= 5:
        return None
    needed = 5 - len(board)
    rng = random.Random(seed)
    from .ranges import sample_concrete_pair
    out: dict[tuple[int, int, bool], float] = {}
    for combo in vr.weights:
        hi, lo, suited = combo
        wins = ties = used = 0
        for _ in range(CONT_RUNOUTS):
            c1 = sample_concrete_pair(rng, hi, lo, suited, dead_cards)
            if c1 is None:
                continue
            rem = remaining_deck(dead_cards + c1)
            if len(rem) < 2 + needed:
                continue
            opp = rng.sample(rem, 2)
            rem2 = [c for c in rem if c.code not in {o.code for o in opp}]
            full = board + rng.sample(rem2, needed)
            hs = evaluate7(c1 + full)
            vs = evaluate7(opp + full)
            if hs > vs:
                wins += 1
            elif hs == vs:
                ties += 1
            used += 1
        if used:
            out[combo] = (wins + 0.5 * ties) / used
    return out or None


def _standardize(scores: dict) -> dict:
    """支撑集内 z-score 标准化（退化 → 全 0）。"""
    vals = list(scores.values())
    if not vals:
        return {k: 0.0 for k in scores}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    if std <= _EPS:
        return {k: 0.0 for k in scores}
    return {k: (v - mean) / std for k, v in scores.items()}


def _tilted(vr: Range, s_tilde: dict, lam: float) -> Range:
    """w_c · exp(λ·s̃_c)（支撑不变；权重 ≤0 的组合保持剔除）。"""
    out = Range(name=vr.name)
    for combo, w in vr.weights.items():
        out.weights[combo] = w * math.exp(lam * s_tilde.get(combo, 0.0))
    return out


# ─────────────────────────── baseline 隐含率 ──────────────────────────────
def baseline_entry_rate(vr_full: Range) -> float:
    """起始范围的隐含进入率 = Σ w_c·mult_c / 1326（pre-removal 口径）。

    ranges.yaml 的组合权重即"打这个组合的频率"——加权组合数占比即 VPIP
    语义的隐含率（A-P46-baseline）。vr_full = parse_range 后、remove_cards
    前的 Range。
    """
    total = sum(w * _total_assignments(hi, lo, su)
                for (hi, lo, su), w in vr_full.weights.items())
    return total / 1326.0


# ─────────────────────────── posterior 主入口 ─────────────────────────────
@dataclass
class RangePosteriorResult:
    available: bool = False
    status: str = "baseline"           # player | population | baseline
    fallback_reason: str = ""
    kind: str = ""                     # starting | continuing
    condition: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    tilt: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    confidence: float = 0.0
    baseline_distribution: dict = field(default_factory=dict)
    posterior_distribution: dict | None = None

    def to_dict(self) -> dict:
        return {"available": self.available, "status": self.status,
                "fallback_reason": self.fallback_reason, "kind": self.kind,
                "condition": self.condition, "target": self.target,
                "tilt": self.tilt, "provenance": self.provenance,
                "confidence": self.confidence,
                "baseline_distribution": self.baseline_distribution,
                "posterior_distribution": self.posterior_distribution}


def build_range_posterior(
    vr_full: Range,
    vr_live: Range,
    *,
    kind: str,
    position: str,
    street: str = "preflop",
    route: str = "open",
    hero_cards: list[str] | None = None,
    board: list[str] | None = None,
    player_index: RangePosteriorIndex | None = None,
    population_index: RangePosteriorIndex | None = None,
    cond_player: dict | None = None,
    cond_population: dict | None = None,
    seed: int = STRENGTH_SEED,
    min_samples: int = MIN_SAMPLES,
) -> tuple[Range | None, RangePosteriorResult]:
    """构建 villain 范围 posterior。

    返回 (posterior_range | None, result)。posterior 不可用时返回
    (None, result)——result.status="baseline" + fallback_reason，
    调用方保持 baseline 分布（绝不伪造）。

    参数:
      vr_full : parse_range 后、remove_cards 前的 baseline Range（隐含率口径）
      vr_live : remove_cards 后的 live Range（tilt 作用对象，支撑集）
      kind    : "starting"（preflop entry/raise 率）| "continuing"（postflop）
      position: villain 位置标签（可为 ""——只走粗化阶梯）
      route   : 路线标签（three_bet/four_bet → raise 率，其余 → entry 率）
      cond_*  : P43 ConditionalFoldIndex.to_dict()（continuing 证据）
    """
    res = RangePosteriorResult(available=False, kind=kind)
    res.baseline_distribution = vr_live.distribution()
    res.condition = {"kind": kind, "position": str(position or ""),
                     "street": str(street)}
    res.provenance = {"module": "range_posterior", "version": _P46_ID,
                      "min_samples": min_samples,
                      "prior_strength": PRIOR_STRENGTH,
                      "sensitivity": SENSITIVITY,
                      "deviation_cap": DEVIATION_CAP,
                      "lambda_cap": LAMBDA_CAP,
                      "support": "baseline_support_only",
                      "strength_model": ""}

    if kind == "starting":
        metric = "raise" if route in ("three_bet", "four_bet") else "entry"
        metric_label = ("pfr_conditional" if metric == "raise" else "entry_rate")
        res.condition["metric"] = metric_label
        res.provenance["strength_model"] = (
            f"preflop equity vs random hand (seeded MC x{STRENGTH_RUNOUTS}, "
            f"seed={STRENGTH_SEED}, cached)")
        rate_base = baseline_entry_rate(vr_full)
        obs = _lookup_rate(player_index, position, metric, min_samples)
        if obs is not None:
            prior = _lookup_rate(population_index, position, metric,
                                 min_samples)
            prior_rate = prior["rate"] if prior else rate_base
            r_post = bayes_shrink(obs["rate"], obs["samples"], prior_rate,
                                  PRIOR_STRENGTH)
            res.status = "player"
            res.provenance["player"] = dict(obs)
            res.provenance["population_prior"] = dict(prior) if prior else None
        else:
            pop = _lookup_rate(population_index, position, metric, min_samples)
            if pop is not None:
                r_post = bayes_shrink(pop["rate"], pop["samples"], rate_base,
                                      PRIOR_STRENGTH)
                res.status = "population"
                res.provenance["population"] = dict(pop)
            else:
                res.fallback_reason = (
                    f"no {metric} evidence >= {min_samples} samples "
                    f"(player and population)")
                res.target = {"metric": metric_label,
                              "baseline_implied": round(rate_base, 6)}
                return None, res
        rate_post = min(max(r_post, 0.0), 1.0)
        denom = max(rate_base, 0.05)
        deviation = max(-DEVIATION_CAP, min(DEVIATION_CAP,
                                            (rate_post - rate_base) / denom))
        lam = max(-LAMBDA_CAP, min(LAMBDA_CAP, -SENSITIVITY * deviation))
        res.target = {"metric": metric_label,
                      "observed": (round(obs["rate"], 6) if obs is not None
                                   else None),
                      "samples": (obs["samples"] if obs is not None
                                  else (pop or {}).get("samples")),
                      "prior": round((prior_rate if obs is not None
                                      else rate_base), 6),
                      "posterior_rate": round(rate_post, 6),
                      "baseline_implied": round(rate_base, 6),
                      "deviation": round(deviation, 6)}
        scores = {c: combo_strength_preflop(c) for c in vr_live.weights}
        n_ref = obs["samples"] if obs is not None else (pop or {}).get("samples", 0)
    elif kind == "continuing":
        res.condition["facing"] = "bet"
        res.condition["ref_bet_fraction"] = REF_BET_FRACTION
        res.provenance["strength_model"] = (
            f"board-conditioned equity vs random hand (hero card-removal "
            f"only; hero-agnostic calibration P50-2, seeded MC x{CONT_RUNOUTS}, "
            f"seed={seed})")
        try:
            hero_c = [parse_card(str(c)) for c in (hero_cards or [])]
            board_c = [parse_card(str(c)) for c in (board or [])]
        except ValueError as e:
            res.fallback_reason = f"invalid card: {e}"
            return None, res
        if len(hero_c) != 2 or not 3 <= len(board_c) <= 5:
            res.fallback_reason = (
                "continuing posterior requires hero hole cards (2) and "
                "board (3-5 cards)")
            return None, res
        res.condition["texture"] = texture_tag(board_c)
        s_ref = REF_BET_FRACTION
        req_ref = s_ref / (1.0 + 2.0 * s_ref)
        res.condition["ref_required_equity"] = round(req_ref, 6)
        res.condition["size_bucket"] = _size_bucket(s_ref, 1.0)
        # P50-2（F2 修复）：行为校准 hero-agnostic——baseline continue 与
        # tilt 评分都改用"board 条件 + hero 死牌剔除"的 vs-random 强度；
        # hero 手牌不再作为强度条件（card removal 除外）。
        strengths = _combo_strength_vs_random(vr_live, board_c,
                                              hero_c + board_c, seed)
        if strengths is None:
            res.fallback_reason = "combo strength evaluation failed (card conflict)"
            return None, res
        total_w = sum(vr_live.weights.values())
        if total_w <= 0:
            res.fallback_reason = "empty live range"
            return None, res
        rate_base = (sum(w for c, w in vr_live.weights.items()
                         if strengths.get(c, 0.0) >= req_ref) / total_w)
        est, est_status = _continue_estimate(cond_player, cond_population,
                                             res.condition, min_samples)
        if est is None:
            res.fallback_reason = (
                f"no conditional fold evidence >= {min_samples} samples "
                f"(player and population)")
            res.target = {"metric": "continue_rate",
                          "baseline_implied": round(rate_base, 6)}
            return None, res
        res.status = est_status
        res.provenance["conditional_estimate"] = est
        # tilt 目标率：以 baseline 隐含继续率为先验重新收缩（A-P46-anchor）。
        # P43 estimate 内部向 population/中性 0.5 收缩——那是 fold 模型语义；
        # tilt 的偏差参照系是 baseline，若直接用 0.5 锚收缩值会把所有玩家
        # 系统性判松。raw 样本率 + baseline 先验 = 无证据 → 零倾斜。
        est_n = int(est.get("samples") or 0)
        continue_raw = 1.0 - float(est.get("raw") or 0.0)
        rate_post = min(max(bayes_shrink(continue_raw, est_n, rate_base,
                                         PRIOR_STRENGTH), 0.0), 1.0)
        denom = max(rate_base, 0.05)
        deviation = max(-DEVIATION_CAP, min(DEVIATION_CAP,
                                            (rate_post - rate_base) / denom))
        lam = max(-LAMBDA_CAP, min(LAMBDA_CAP, -SENSITIVITY * deviation))
        res.target = {"metric": "continue_rate",
                      "observed": round(rate_post, 6),
                      "samples": est.get("samples"),
                      "prior": est.get("prior_rate"),
                      "posterior_rate": round(rate_post, 6),
                      "baseline_implied": round(rate_base, 6),
                      "deviation": round(deviation, 6)}
        scores = {c: strengths.get(c, 0.0) for c in vr_live.weights}
        n_ref = int(est.get("samples") or 0)
    else:
        res.fallback_reason = f"unknown kind {kind!r}"
        return None, res

    if abs(lam) < 1e-9:
        # 偏差被 clip 为 0 → posterior 与 baseline 逐位相同（诚实标注）
        res.available = True
        res.tilt = {"lambda": 0.0, "effective": False}
        res.confidence = _confidence(n_ref)
        res.posterior_distribution = dict(res.baseline_distribution)
        res.provenance["note"] = "deviation ~ 0 → posterior == baseline"
        return None, res

    s_tilde = _standardize(scores)
    vr_post = _tilted(vr_live, s_tilde, lam)
    res.available = True
    res.tilt = {"lambda": round(lam, 6), "effective": True,
                "sensitivity": SENSITIVITY, "lambda_cap": LAMBDA_CAP,
                "standardized_strength": True,
                "support": "baseline_support_only"}
    res.confidence = _confidence(n_ref)
    res.posterior_distribution = vr_post.distribution()
    return vr_post, res


def _lookup_rate(idx: RangePosteriorIndex | None, position: str,
                 metric: str, min_samples: int) -> dict | None:
    if idx is None:
        return None
    return idx.rate(position, metric, min_samples)


def _continue_estimate(cond_player: dict | None, cond_population: dict | None,
                       condition: dict, min_samples: int) -> tuple[dict | None, str]:
    """P43 条件 fold 估计（player→population 优先级，与 fold_equity 一致）。"""
    from .conditional_model import ConditionalFoldIndex
    cplayer = (ConditionalFoldIndex.from_dict(cond_player)
               if isinstance(cond_player, dict) and cond_player.get("counts")
               else None)
    cpop = (ConditionalFoldIndex.from_dict(cond_population)
            if isinstance(cond_population, dict) and cond_population.get("counts")
            else None)
    args = (condition.get("street"), "bet",
            condition.get("size_bucket"), condition.get("position"),
            condition.get("texture"))
    if cplayer is not None:
        est = cplayer.estimate(*args, population=cpop,
                               min_samples=min_samples)
        if est is not None:
            return est, "player"
    if cpop is not None:
        est = cpop.estimate(*args, min_samples=min_samples)
        if est is not None:
            return est, "population"
    return None, "baseline"


def _confidence(n: int) -> float:
    """样本置信（与 P43 confidence 同口径）。"""
    return round(min(1.0, max(0, int(n)) / 100.0), 4)
