"""Opponent Agent (玩家画像，三层模型核心): hero/villain 画像。

Input : {state: dict, statistics: dict|None}
Output: OpponentOutput {players: {pid: profile}, confidence, evidence, warnings}

两层来源（契约 §4 opponent）：
  - 档案存在  : ctx.repos.players.get(pid) -> 直接用档案字段
                （feature_vector / behavior_statistics / recent_behavior /
                  historical_behavior / strategy_regime / sample_size / last_seen）
  - 档案缺失  : 从 ctx.repos.hands 该 player_id 的归一化记录聚合
                compute_all -> behavior_statistics；
                recent = 最近 N=100 手（按 ts 或顺序取尾部，time_decay_weights 90 天半衰期加权）；
                historical = 全部。

feature_vector 映射公式（bayes_shrink(rate, n, prior, strength=200)）：
  tightness        = 1 - bayes_shrink(vpip.value, vpip.samples, prior=0.3)
  aggression       = bayes_shrink(aggression_frequency.value, samples, prior=0.35)
  calling_tendency = 1 - bayes_shrink(fold_rate, prior=0.5)
                     （fold_rate 优先取 fold_to_cbet；否则 calls/(calls+folds)）
  folding_tendency = 1 - calling_tendency
  bluff_tendency   = bayes_shrink(aggression_frequency.value, n, 0.3) * 0.5
                     （无 showdown 数据时保守估计：诈唬频率无法直接从统计观察，故乘 0.5）
  其余 5 个维度（overbet/check_raise/tilt/icm/stage_adaptivity）：
     记录里有对应可选字段就用，否则 0.5 中性 + 样本 0。

confidence = min(1, sample_size/2000) 结合 wilson 宽度（样本 <50 -> 低置信）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker.stats import (
    bayes_shrink,
    compute_all,
    time_decay_weights,
    wilson_interval,
)

_RECENT_N = 100          # recent 窗口手数
_HALF_LIFE_DAYS = 90.0   # 时间衰减半衰期
_MAX_PARSED_HANDS = 5000 # P43：条件事件抽取的手牌上限（性能边界，真实数据）

# feature_vector 的 10 个维度
_FV_KEYS = ["tightness", "aggression", "bluff_tendency", "calling_tendency",
            "folding_tendency", "overbet_tendency", "check_raise_tendency",
            "tilt_risk", "icm_sensitivity", "stage_adaptivity"]

# V1.1 赏金特征（个人模型；经 Population->Type->Individual 三级收缩）
_BOUNTY_FEATURE_KEYS = ("bounty_hunting_tendency", "bounty_call_adjustment",
                        "bounty_aggression", "bounty_sensitivity",
                        "large_bounty_behavior", "small_bounty_behavior")


@dataclass
class OpponentOutput:
    players: dict = field(default_factory=dict)   # pid -> profile dict
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class OpponentAgent(BaseAgent):
    name: ClassVar[str] = "opponent"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "玩家画像 Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state", "statistics"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = OpponentOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state.get("hero_id") and not state.get("players"):
            return False, "state has no players to profile"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> OpponentOutput:
        state = inp.get("state") or {}
        statistics = inp.get("statistics")
        warnings: list[dict] = []

        try:
            all_records = ctx.repos.hands.list(limit=100000)
        except Exception as e:
            all_records = []
            warnings.append({"code": "REPO_ERROR", "message": f"hands repo read failed: {e}"})

        hero_id = str(state.get("hero_id") or "")
        players = state.get("players") or []
        name_map = {str(p.get("player_id")): str(p.get("name") or p.get("player_id"))
                    for p in players if isinstance(p, dict)}

        # 目标 = hero + 每个 active villain
        targets: list[str] = [hero_id] if hero_id else []
        for p in players:
            pid = str(p.get("player_id"))
            status = p.get("status") if isinstance(p, dict) else getattr(p, "status", "active")
            if status in ("active", "allin") and pid != hero_id:
                targets.append(pid)
        targets = list(dict.fromkeys(targets))

        # ---- P43：parsed 手牌 → 条件行为事件（真实历史；仅 action 级数据） ----
        # P47：evidence gate——gate 标记记录须 action_level 允许才参与事件
        # 抽取（legacy 无 gate 记录维持既有语义）。
        parsed_hands = [r.get("parsed") for r in all_records
                        if isinstance(r, dict)
                        and isinstance(r.get("parsed"), dict)
                        and eg.record_allowed(r, "action_level")][:_MAX_PARSED_HANDS]
        events_by_player: dict[str, list[dict]] = {}
        population_index = None
        population_rpi = None
        if parsed_hands:
            from poker_decision_ai.domain.poker.conditional_model import (
                ConditionalFoldIndex, extract_events)
            from poker_decision_ai.domain.poker.range_posterior import (
                RangePosteriorIndex)
            population_index = ConditionalFoldIndex()
            population_rpi = RangePosteriorIndex()
            for ph in parsed_hands:
                try:
                    evs = extract_events(ph)
                except Exception:
                    continue
                for ev in evs:
                    events_by_player.setdefault(ev["player"], []).append(ev)
                population_index.add_events(evs)
                population_rpi.add_events(evs)

        profiles: dict[str, dict] = {}
        for pid in targets:
            profiles[pid] = self._profile_for(pid, name_map, all_records,
                                              statistics, ctx, warnings)
        # ---- P43：条件模型挂接（player 条件索引 + population 先验索引） ----
        # ---- P46：范围 posterior 索引挂接（同源事件；additive 字段） ----
        if population_index is not None:
            pop_ser = population_index.to_dict()
            for pid, prof in profiles.items():
                name = str(prof.get("name") or "")
                pevents = events_by_player.get(name) or []
                pidx = ConditionalFoldIndex()
                pidx.add_events(pevents)
                # 条件数据仅 RECOGNIZED（真实名字来源匹配）且有事件时挂接
                prof["conditional_model"] = (pidx.to_dict()
                                             if pidx.total_events > 0 else {})
                prof["conditional_population"] = pop_ser
                prof["conditional_events"] = pidx.total_events
                # P46：player 范围 posterior 索引 + population 先验索引
                prpi = RangePosteriorIndex()
                prpi.add_events(pevents)
                prof["range_posterior"] = (prpi.to_dict()
                                           if prpi.total_hands > 0 else {})
                prof["range_posterior_population"] = (population_rpi.to_dict()
                                                      if population_rpi
                                                      and population_rpi.total_hands > 0
                                                      else {})

        if hero_id and hero_id in profiles:
            confidence = float(profiles[hero_id].get("confidence") or 0.0)
        else:
            confs = [float(v.get("confidence") or 0.0) for v in profiles.values()]
            confidence = (sum(confs) / len(confs)) if confs else 0.0

        return OpponentOutput(
            players=profiles, confidence=round(confidence, 4),
            evidence=[{"kind": "opponent", "source": "repos.players|repos.hands",
                       "text": f"为 {len(profiles)} 名玩家构建画像", "weight": 1.0}],
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    def _gate_summary(self, records: list) -> dict:
        """P47：该玩家记录的 evidence gate 汇总（无 gate 标记 → {}）。

        返回 {data_kind, dataset, evidence_gate: {etype: {verdict}},
        n_gate_tagged}——verdict 取该类型记录中的最高允许级（usable >
        partially_usable > excluded）；聚合诚实反映混合语料。
        """
        rank = {eg.EXCLUDED: 0, eg.UNKNOWN: 0,
                eg.PARTIALLY_USABLE: 1, eg.USABLE: 2}
        best: dict[str, str] = {}
        kinds: set[str] = set()
        dataset: dict | None = None
        n_tagged = 0
        for r in records or []:
            if not isinstance(r, dict):
                continue
            if r.get("data_kind"):
                kinds.add(str(r["data_kind"]))
                if dataset is None and isinstance(r.get("dataset"), dict):
                    dataset = dict(r["dataset"])
            ev = r.get("evidence_gate")
            if not isinstance(ev, dict):
                continue
            n_tagged += 1
            for t, v in ev.items():
                if not isinstance(v, dict):
                    continue
                vd = str(v.get("verdict") or eg.UNKNOWN)
                cur = best.get(t)
                if cur is None or rank.get(vd, 0) > rank.get(cur, 0):
                    best[t] = vd
        if not n_tagged and not kinds:
            return {}
        data_kind = (kinds.pop() if len(kinds) == 1 else
                     "MIXED" if kinds else None)
        return {"data_kind": data_kind,
                "dataset": dataset,
                "evidence_gate": {t: {"verdict": best[t]}
                                  for t in sorted(best)},
                "n_gate_tagged": n_tagged}

    # ------------------------------------------------------------------ #
    def _profile_for(self, pid: str, name_map: dict, all_records: list,
                     statistics: dict | None, ctx: AgentContext, warnings: list) -> dict:
        name = name_map.get(pid, pid)

        # ---- 1) 档案存在：直接用档案字段 ----
        try:
            existing = ctx.repos.players.get(pid)
        except Exception as e:
            existing = None
            warnings.append({"code": "REPO_ERROR", "message": f"players.get({pid}) failed: {e}"})
        if existing:
            return self._from_profile(existing, pid, name, all_records)

        # ---- 2) 档案缺失：从 hands 聚合 ----
        records = [h for h in all_records if str(h.get("player_id")) == pid]
        if not records:
            warnings.append({"code": "NO_DATA", "message": f"player {pid}: 无档案且无手牌记录，返回中性画像"})
            return self._neutral_profile(pid, name)

        historical = compute_all(records).to_dict()
        recent = self._recent_behavior(records)
        feature_vector = self._build_feature_vector(records, historical)
        sample_size = len(records)
        vpip_m = historical.get("vpip", {"value": 0.0, "samples": 0})
        confidence = _confidence(sample_size, vpip_m)
        last_seen = _ts_to_iso(_last_ts(records))
        regime = _regime_from_vector(feature_vector, confidence)

        return {
            "player_id": pid, "name": name,
            "feature_vector": feature_vector,
            "behavior_statistics": historical,
            "recent_behavior": recent,
            "historical_behavior": historical,
            "strategy_regime": regime,
            "bounty_features": _bounty_features(records, None),
            "sample_size": sample_size,
            "confidence": confidence,
            "last_seen": last_seen,
            **self._gate_summary(records),
        }

    def _from_profile(self, existing: dict, pid: str, name: str, all_records: list) -> dict:
        """档案存在：以档案为基底，并用 hands 推导的统计**充实**档案。

        档案只存了 3 个核心指标时，hands 能补齐 14 项全量统计与 recent_behavior，
        使 Confidence Agent 的 data_quality/recency 评分反映真实数据量。
        """
        fv = dict(existing.get("feature_vector") or {})
        fv = {k: float(fv.get(k, 0.5)) for k in _FV_KEYS}  # 补 0.5 中性
        sample_size = int(existing.get("sample_size") or 0)
        behavior = dict(existing.get("behavior_statistics") or {})
        recent = dict(existing.get("recent_behavior") or {})
        last_seen = str(existing.get("last_seen") or "")

        records = [h for h in all_records if str(h.get("player_id")) == pid]
        if records:
            hands_stats = compute_all(records).to_dict()
            # 合并：hands 推导值优先，除非档案里的样本数更大
            merged = dict(hands_stats)
            for k, v in behavior.items():
                if not isinstance(v, dict):
                    continue
                if k in merged and int(v.get("samples") or 0) > int(merged[k].get("samples") or 0):
                    merged[k] = v
                elif k not in merged:
                    merged[k] = v
            behavior = merged
            recent = self._recent_behavior(records) or recent
            if len(records) > sample_size:
                sample_size = len(records)
            if not last_seen:
                last_seen = _ts_to_iso(_last_ts(records))

        confidence = float(existing.get("confidence") or _confidence(sample_size, behavior.get("vpip")))
        return {
            "player_id": pid, "name": str(existing.get("name") or name),
            "feature_vector": fv,
            "behavior_statistics": behavior,
            "recent_behavior": recent,
            "historical_behavior": existing.get("historical_behavior") or behavior,
            "strategy_regime": existing.get("strategy_regime") or {"label": "", "since": "", "confidence": 0.0},
            "bounty_features": _bounty_features(records, existing),
            "sample_size": sample_size,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "last_seen": last_seen,
            **self._gate_summary(records),
        }

    def _neutral_profile(self, pid: str, name: str) -> dict:
        return {
            "player_id": pid, "name": name,
            "feature_vector": {k: 0.5 for k in _FV_KEYS},
            "behavior_statistics": {}, "recent_behavior": {}, "historical_behavior": {},
            "strategy_regime": {"label": "unknown", "since": "", "confidence": 0.0},
            "bounty_features": {k: {"value": 0.5, "samples": 0} for k in _BOUNTY_FEATURE_KEYS},
            "sample_size": 0, "confidence": 0.0, "last_seen": "",
        }

    # ------------------------------------------------------------------ #
    def _recent_behavior(self, records: list) -> dict:
        """最近 N 手（按 ts 或顺序取尾部），time_decay_weights 加权计算主要比率指标。"""
        recent = _recent_records(records, _RECENT_N)
        if not recent:
            return {}
        bundle = compute_all(recent).to_dict()
        ts_list = [float(h.get("ts")) for h in recent if h.get("ts") is not None]
        if ts_list:
            weights = time_decay_weights(ts_list, half_life_days=_HALF_LIFE_DAYS)
            if weights:
                # 对布尔比率类指标做时间衰减加权（更重视近期表现）
                for field in ("vpip", "pfr", "three_bet", "cbet"):
                    m = _weighted_rate(recent, field, weights)
                    if m is not None:
                        bundle[field] = m
                for opp, act in (("faced_cbet", "folded_to_cbet"), ("cbet_opportunity", "cbet")):
                    m = _weighted_opp_rate(recent, opp, act, weights)
                    if m is not None:
                        bundle[act] = m
        return bundle

    def _build_feature_vector(self, records: list, behavior: dict) -> dict:
        """由行为统计映射 feature_vector（bayes_shrink 向先验收缩）。"""
        vpip_m = behavior.get("vpip", {"value": 0.0, "samples": 0})
        agg_m = behavior.get("aggression_frequency", {"value": 0.0, "samples": 0})
        ftc_m = behavior.get("fold_to_cbet", {"value": 0.0, "samples": 0})

        vpip_v = float(vpip_m.get("value") or 0.0)
        vpip_n = int(vpip_m.get("samples") or 0)
        agg_v = float(agg_m.get("value") or 0.0)
        agg_n = int(agg_m.get("samples") or 0)

        # tightness = 1 - shrink(vpip, prior=0.3)
        tightness = 1.0 - bayes_shrink(vpip_v, vpip_n, prior_rate=0.3)
        # aggression = shrink(aggression_frequency, prior=0.35)
        aggression = bayes_shrink(agg_v, agg_n, prior_rate=0.35)

        # fold_rate：优先 fold_to_cbet；否则 calls/(calls+folds)
        ftc_v = float(ftc_m.get("value") or 0.0)
        ftc_n = int(ftc_m.get("samples") or 0)
        if ftc_n > 0:
            fold_rate = bayes_shrink(ftc_v, ftc_n, prior_rate=0.5)
        else:
            calls = sum(int(h.get("calls") or 0) for h in records)
            folds = sum(int(h.get("folds") or 0) for h in records)
            if calls + folds > 0:
                fold_rate = bayes_shrink(folds / (calls + folds), calls + folds, prior_rate=0.5)
            else:
                fold_rate = 0.5
        calling_tendency = 1.0 - fold_rate
        folding_tendency = fold_rate

        # bluff_tendency = shrink(aggression_frequency, prior=0.3) * 0.5（无 showdown 数据保守）
        bluff_tendency = bayes_shrink(agg_v, agg_n, prior_rate=0.3) * 0.5

        fv = {
            "tightness": _clip01(tightness),
            "aggression": _clip01(aggression),
            "bluff_tendency": _clip01(bluff_tendency),
            "calling_tendency": _clip01(calling_tendency),
            "folding_tendency": _clip01(folding_tendency),
        }
        # 其余 5 维：记录里有可选字段就用（按样本数），否则 0.5 中性 + 样本 0
        for dim, field_name in (
            ("overbet_tendency", "overbets"),
            ("check_raise_tendency", "cr_attempts"),
            ("tilt_risk", "tilt_risk"),
            ("icm_sensitivity", "icm_sensitivity"),
            ("stage_adaptivity", "stage_adaptivity"),
        ):
            fv[dim] = _optional_rate(records, field_name)
        return {k: round(fv[k], 4) for k in _FV_KEYS}


# --------------------------------------------------------------------------- #
# 模块级 helpers
# --------------------------------------------------------------------------- #
def _bounty_features(records: list, existing: dict | None) -> dict:
    """V1.1 个人赏金特征（三层模型：档案 > hands 推导(收缩) > 0.5 中性）。

    档案 bounty_features 优先；其次从 hands 的可选字段推导并 bayes_shrink
    向 0.5 收缩（小样本不置信）；都没有 → 0.5 中性、samples=0。
    """
    profile_bf = dict(existing.get("bounty_features") or {}) if isinstance(existing, dict) else {}

    def rate(field: str):
        vals = [h for h in records if isinstance(h.get(field), bool)]
        if not vals:
            return None
        return sum(1 for h in vals if h[field]) / len(vals), len(vals)

    derived = {
        "bounty_hunting_tendency": rate("chased_bounty"),
        "bounty_call_adjustment": rate("bounty_overcall"),
        "bounty_aggression": rate("bounty_aggressive"),
        "bounty_sensitivity": rate("bounty_sensitive_fold"),
        "large_bounty_behavior": rate("large_bounty_aggressive"),
        "small_bounty_behavior": rate("small_bounty_aggressive"),
    }
    out: dict[str, dict] = {}
    for key in _BOUNTY_FEATURE_KEYS:
        prof = profile_bf.get(key)
        if isinstance(prof, dict):
            out[key] = {"value": float(prof.get("value", 0.5)), "samples": int(prof.get("samples") or 0)}
        elif isinstance(prof, (int, float)):
            out[key] = {"value": float(prof), "samples": int(existing.get("sample_size") or 0)}
        elif derived[key] is not None:
            r, n = derived[key]
            out[key] = {"value": round(bayes_shrink(r, n, 0.5, 200), 4), "samples": n}
        else:
            out[key] = {"value": 0.5, "samples": 0}
    return out


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _confidence(sample_size: int, vpip_m: dict | None) -> float:
    """confidence = min(1, n/2000) 结合 wilson 宽度；样本 <50 -> 低置信。"""
    n = int(sample_size or 0)
    base = min(1.0, n / 2000.0)
    if vpip_m and int(vpip_m.get("samples") or 0) > 0:
        vn = int(vpip_m["samples"])
        v = float(vpip_m.get("value") or 0.0)
        lo, hi = wilson_interval(int(round(v * vn)), vn)
        base *= max(0.0, 1.0 - (hi - lo))   # 区间越窄 -> 置信越高
    if n < 50:
        base = min(base, 0.4)               # 样本 <50 -> 低置信
    return round(_clip01(base), 4)


def _recent_records(records: list, n: int) -> list:
    """按 ts 排序取最近 n 手；无 ts 则取顺序尾部。"""
    if not records:
        return []
    with_ts = [(h, float(h["ts"])) for h in records if h.get("ts") is not None]
    if with_ts:
        ordered = sorted(with_ts, key=lambda x: x[1], reverse=True)[:n]
        return [h for h, _ in ordered]
    return records[-n:]


def _last_ts(records: list) -> float | None:
    ts_list = [float(h["ts"]) for h in records if h.get("ts") is not None]
    return max(ts_list) if ts_list else None


def _ts_to_iso(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OSError, OverflowError):
        return ""


def _weighted_rate(hands: list, field: str, weights: list[float]) -> dict | None:
    """布尔字段的加权比率 Metric。无 bool 样本则返回 None。"""
    idx = [i for i, h in enumerate(hands) if isinstance(h.get(field), bool)]
    if not idx:
        return None
    total_w = sum(weights[i] for i in idx)
    if total_w <= 0:
        return None
    val = sum((1.0 if hands[i][field] else 0.0) * weights[i] for i in idx) / total_w
    return {"value": round(val, 4), "samples": len(idx)}


def _weighted_opp_rate(hands: list, opp_field: str, act_field: str, weights: list[float]) -> dict | None:
    """机会型指标的加权比率（如 fold_to_cbet）。"""
    idx = [i for i, h in enumerate(hands) if h.get(opp_field)]
    if not idx:
        return None
    total_w = sum(weights[i] for i in idx)
    if total_w <= 0:
        return None
    hits = sum(weights[i] for i in idx if hands[i].get(act_field))
    return {"value": round(hits / total_w, 4), "samples": len(idx)}


def _optional_rate(records: list, field: str) -> float:
    """optional 字段的比率；无数据 -> 0.5 中性。"""
    known = [h for h in records if h.get(field) is not None]
    if not known:
        return 0.5
    # 布尔 -> 比率；数值 -> 均值
    vals = [h[field] for h in known]
    if all(isinstance(v, bool) for v in vals):
        return round(sum(1.0 for v in vals if v) / len(vals), 4)
    try:
        return round(sum(float(v) for v in vals) / len(vals), 4)
    except (TypeError, ValueError):
        return 0.5


def _regime_from_vector(fv: dict, confidence: float) -> dict:
    """由 feature_vector 生成简单策略 regime 标签（仅解释用）。"""
    tight = fv.get("tightness", 0.5)
    agg = fv.get("aggression", 0.5)
    if tight >= 0.6 and agg >= 0.55:
        label = "TAG"
    elif tight >= 0.6:
        label = "Nit"
    elif agg >= 0.6:
        label = "LAG"
    elif fv.get("calling_tendency", 0.5) >= 0.6:
        label = "Calling Station"
    else:
        label = "Regular"
    return {"label": label, "since": "", "confidence": round(confidence, 4)}
