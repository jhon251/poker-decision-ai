"""Frequency Prior Agent (MC1-R5): 确定性对手动作频率先验（生产接线）。

Input : {state: dict, statistics?: dict}
Output: FrequencyPriorOutput {available, selection, villains, confidence,
        evidence, warnings}

数据来源（config.yaml::frequency_prior.source_dir，默认 data/uoft/normalized）：
  canonical 语料（含稳定 player_id 与 per-street 动作行）→ 重放每条街推导
  facing_bet → 决策级 FrequencyObservation → 按 (player, position, street,
  facing, action) 聚合为持久索引（缓存文件可复用；桶级衰减用桶内最新 ts，
  粒度 provenance 声明）。数据源缺失/为空 → available=False + warning
  （fail-closed；绝不编造频率）。

影子模式（互不覆盖）：
  本 agent 只产出频率先验；judge 在 shadow_comparison 中与旧对手建模链
  的建议分字段记录（见 agents/judge/agent.py）。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import frequency_prior as fp
from poker_decision_ai.agents.frequency_prior.session_store import (
    VillainSessionStore, get_or_create_session_store,
)

_MODULE_CACHE: dict[tuple, Any] = {}

#: M3（R5）：聚合索引 schema 版本。v2 = 聚合索引键加入**级别维度**
#: （POOL_BOT_PLAN 数据政策第 1 条的前置条件：不同级别/池子的频率不得互相
#: 污染）。schema 变更使旧磁盘缓存自然失效（fingerprint 含 schema）。
INDEX_SCHEMA_VERSION = 2

#: 级别分段缺省段（沿用 population 段位约定；生产值在 config.yaml::
#: frequency_prior.level 声明，此处仅是 config 缺项时的诚实缺省）。
DEFAULT_CORPUS_LEVEL = "NL10:cash:6max"


@dataclass
class FrequencyPriorOutput:
    available: bool = False
    selection: dict = field(default_factory=dict)
    villains: dict = field(default_factory=dict)     # pid -> FrequencyPrior.to_dict()
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class FrequencyPriorAgent(BaseAgent):
    name: ClassVar[str] = "frequency_prior"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "对手频率先验 Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = FrequencyPriorOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        villains = _villain_seats(state)
        if not villains:
            return False, "state has no active non-hero player with an id"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> FrequencyPriorOutput:
        state = inp.get("state") or {}
        cfg_map = _frequency_prior_config(ctx)
        cfg = fp.FrequencyPriorConfig.from_mapping(cfg_map)
        level = str(cfg_map.get("level") or DEFAULT_CORPUS_LEVEL)
        warnings: list[dict] = []

        store = _session_store_for(ctx, cfg_map, warnings)
        index, prov = _load_index(cfg_map, ctx, warnings)
        store_has_data = bool(store is not None and store.players())
        if not index and not store_has_data:
            return FrequencyPriorOutput(
                available=False, warnings=warnings + [{
                    "code": "NO_DATA",
                    "message": "frequency prior index unavailable; "
                               "no frequencies fabricated"}])

        street = str(state.get("street") or "")
        to_call = float(state.get("to_call") or 0.0)
        facing_bet = to_call > 0
        villains = _villain_seats(state)
        selection = _select_villain(state, villains)
        out_villains: dict[str, dict] = {}
        evidence: list[dict] = []
        for pid, meta in villains.items():
            prior, layer, layer_warnings = _two_level_prior(
                index, store, player_id=pid,
                position=meta.get("position"), street=street,
                facing_bet=facing_bet, cfg=cfg, level=level,
                session_min_sample=_session_min_sample(cfg_map, cfg))
            warnings.extend(layer_warnings)
            d = prior.to_dict()
            d["layer"] = layer
            d["level"] = level
            out_villains[pid] = d
        sel_pid = selection.get("player_id") or ""
        sel_prior = out_villains.get(sel_pid) or {}
        evidence.append({
            "kind": "frequency_prior",
            "source": "frequency_baseline",
            "text": (f"villain {sel_pid or '?'} {street}/"
                     f"{selection.get('position') or '?'} "
                     f"facing_bet={facing_bet}: "
                     f"p_continue={sel_prior.get('p_continue')} "
                     f"(source={sel_prior.get('source')}, "
                     f"layer={sel_prior.get('layer')}, "
                     f"n={sel_prior.get('n_total')})"),
            "weight": 1.0,
            "provenance": {"index_provenance": prov,
                           "selection": dict(selection),
                           "layer": sel_prior.get("layer"),
                           "level": level},
        })
        available = bool(sel_prior.get("frequencies"))
        confidence = _confidence_for(sel_prior)
        if sel_prior.get("warnings"):
            warnings.extend(sel_prior["warnings"])
        # P30-9-BP：范围通道补全弃牌权益（默认关闭）
        try:
            _added = _enrich_from_range(out_villains, state=state,
                                         range_in=inp.get("range") or {},
                                         facing_bet=facing_bet,
                                         cfg=ctx.config.frequency_prior_config() or {},
                                         warnings=warnings)
            if _added and sel_prior.get("frequencies"):
                available = True
                confidence = max(confidence, 0.5)
        except Exception:                                      # noqa: BLE001
            pass
        return FrequencyPriorOutput(
            available=available, selection=selection, villains=out_villains,
            confidence=confidence, evidence=evidence, warnings=warnings)


# ----------------------------------------------------------------------
# villain 选择（确定性、声明式）
# ----------------------------------------------------------------------

def _enrich_from_range(out_villains: dict, *, state: dict, range_in: dict,
                       facing_bet: bool | None, cfg: dict,
                       warnings: list) -> int:
    """用 FoldEstimator 的**范围通道**补全弃牌率（仅当配置开启且统计缺失）。

    ⇒ 让冻结 judge 的 fold-equity / call 通道真正拿到数值；
       来源标 `range_only`，绝不冒充个体统计。
    """
    import os as _os
    src = str(_os.environ.get("FP_FOLD_FROM_RANGE")
               or cfg.get("fold_from_range", "false")).lower()
    _blend_on = False
    try:
        _blend_on = float(_os.environ.get("FP_RANGE_BLEND")
                           or cfg.get("range_blend_weight", 0.0) or 0.0) > 0
    except Exception:                                          # noqa: BLE001
        _blend_on = False
    if src in ("false", "0", "", "no", "off") and not _blend_on:
        return 0
    from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
    vmap = (range_in or {}).get("villains") or {}
    players = state.get("players") or []
    hero = next((p for p in players if isinstance(p, dict) and p.get("is_hero")), {})
    hero_cards = [str(c) for c in (hero.get("hole_cards") or [])]
    board = [str(c) for c in (state.get("board") or [])]
    bb = float(state.get("bb") or 0.0) or 1.0
    pot_bb = float(state.get("pot") or 0.0) / bb
    to_call_bb = float(state.get("to_call") or 0.0) / bb
    facing = "raise" if to_call_bb > 0 else "bet"
    added = 0
    # 混合权重（默认 0 = 纯群体统计，行为不变）
    try:
        import os as _os2
        w_range = float(_os2.environ.get("FP_RANGE_BLEND")
                         or cfg.get("range_blend_weight", 0.0) or 0.0)
    except Exception:                                          # noqa: BLE001
        w_range = 0.0
    for pid, entry in (out_villains or {}).items():
        if not isinstance(entry, dict):
            continue
        _has_stat = bool(entry.get("frequencies"))
        if _has_stat and w_range <= 0:
            continue                                   # 现状路径：群体统计不被触碰
        dist = ((vmap.get(pid) or {}).get("distribution")
                if isinstance(vmap.get(pid), dict) else None)
        if not dist:
            continue
        try:
            est = FoldEstimator(pot_bb=pot_bb, facing=facing,
                                 street=str(state.get("street") or "flop"),
                                 villain_dist=dist, hero_cards=hero_cards,
                                 board=board)
            res = est.for_size(max(0.0, float(entry.get("villain_call_bb") or 0.0))
                                or max(0.5 * pot_bb, to_call_bb))
        except Exception:                                      # noqa: BLE001
            continue
        if not getattr(res, "available", False) or res.p_fold is None:
            continue
        p_fold = float(res.p_fold)
        p_cont = float(res.p_continue if res.p_continue is not None
                        else 1.0 - p_fold)
        if _has_stat and w_range > 0:
            # 混合：群体统计 × (1−w) + 范围结构 × w
            try:
                f_stat = float((entry["frequencies"].get("fold") or {}).get("rate") or 0.0)
            except Exception:                                  # noqa: BLE001
                f_stat = 0.0
            p_fold = round((1.0 - w_range) * f_stat + w_range * p_fold, 4)
            p_cont = round(1.0 - p_fold, 4)
            warnings.append({"code": "FOLD_STAT_RANGE_BLEND",
                              "message": (f"villain {pid}: 群体 fold={f_stat:.3f} 与范围结构 "
                                           f"fold={round(float(res.p_fold), 3)} 按 "
                                           f"w_range={w_range} 混合 ⇒ {p_fold}")})
        entry["frequencies"] = {
            "fold": {"rate": round(p_fold, 4)},
            "call": {"rate": round(max(0.0, p_cont) * 0.7, 4)},
            "raise": {"rate": round(max(0.0, p_cont) * 0.3, 4)},
            "bet": {"rate": round(max(0.0, p_cont), 4)},
        }
        entry["p_fold"] = round(p_fold, 4)
        entry["p_continue"] = round(p_cont, 4)
        entry["source"] = ("population×street+range" if _has_stat else "range_only")
        entry["layer"] = "range_structure"
        entry["n_total"] = 0
        entry["range_fold_provenance"] = {
            "mode": getattr(res, "mode", "range"),
            "p_fold": round(p_fold, 4),
            "provenance": getattr(res, "provenance", {}),
            "note": "由对手范围结构估计（非个体行为统计）"}
        warnings.append({"code": "FOLD_FROM_RANGE",
                          "message": (f"villain {pid}: 无行为统计 ⇒ 用范围通道估计 "
                                       f"f_fold={round(p_fold, 4)}（来源 range_only）")})
        added += 1
    return added


def _villain_seats(state: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in state.get("players") or []:
        if not isinstance(p, dict) or p.get("is_hero"):
            continue
        if str(p.get("status") or "") not in ("active", "allin"):
            continue
        pid = str(p.get("player_id") or "")
        if not pid:
            continue
        out[pid] = {"position": p.get("position"), "seat": p.get("seat"),
                    "status": p.get("status")}
    return out


def _select_villain(state: dict, villains: dict[str, dict]) -> dict:
    """主对手 = last_aggressor（如有）否则第一个活跃非 hero（确定性）。"""
    last = str(state.get("last_aggressor") or "")
    if last and last in villains:
        return {"player_id": last, "rule": "last_aggressor",
                "position": villains[last].get("position")}
    for pid, meta in villains.items():
        return {"player_id": pid, "rule": "first_active",
                "position": meta.get("position")}
    return {"player_id": "", "rule": "none"}


def _confidence_for(prior: dict) -> float:
    if not prior.get("frequencies"):
        return 0.0
    n = int(prior.get("n_total") or 0)
    return min(1.0, round(n / 100.0, 3))


# ----------------------------------------------------------------------
# M3：个体 > 群体 两级先验（会话个体层 → 语料层）
# ----------------------------------------------------------------------

def _session_min_sample(cfg_map: dict, cfg: fp.FrequencyPriorConfig) -> int:
    """会话个体层启用阈值（config::frequency_prior.session_min_sample_size）。"""
    raw = cfg_map.get("session_min_sample_size")
    try:
        v = int(raw) if raw is not None else int(cfg.min_sample_size)
    except (TypeError, ValueError):
        v = int(cfg.min_sample_size)
    return max(1, v)


def _session_store_for(ctx: Any, cfg_map: dict,
                       warnings: list) -> VillainSessionStore | None:
    """会话个体层 store：显式注入优先（ctx.extra），否则按 session_id 注册。"""
    if not bool(cfg_map.get("session_enabled", True)):
        return None
    injected = None
    extra = getattr(ctx, "extra", None)
    if isinstance(extra, dict):
        injected = extra.get("villain_session_store")
    if isinstance(injected, VillainSessionStore):
        return injected
    session_id = str(getattr(ctx, "session_id", "") or "realtime")
    return get_or_create_session_store(session_id, cfg_map, warnings)


def _two_level_prior(index: dict, store: VillainSessionStore | None, *,
                     player_id: str, position: Any, street: str,
                     facing_bet: bool, cfg: fp.FrequencyPriorConfig,
                     level: str, session_min_sample: int
                     ) -> tuple[fp.FrequencyPrior, str, list[dict]]:
    """个体 > 群体 两级解析（M3）：

      1. session_individual —— 本会话逐手累积的该 villain 观测（≥
         session_min_sample 才可用；有观测但不达标 → 回退并附 warning）；
      2. 语料层 —— 既有聚合索引（player 级别 → population 级别，回退链
         在 _query_index 内），source/警示语义不变。
    返回 (prior, layer, layer_warnings)；layer ∈ {session_individual,
    corpus_individual, corpus_population, unavailable}。
    """
    layer_warnings: list[dict] = []
    if store is not None:
        sprior = store.prior(player_id, position=position, street=street,
                             facing_bet=facing_bet)
        if sprior.frequencies is not None \
                and sprior.n_total >= session_min_sample:
            return sprior, "session_individual", layer_warnings
        if sprior.n_total > 0:
            layer_warnings.append({
                "code": "SESSION_INSUFFICIENT_SAMPLE",
                "message": (f"session sample n={sprior.n_total} below "
                            f"session_min_sample_size={session_min_sample}; "
                            "falling back to corpus layer")})
    prior = _query_index(index, player_id=player_id, position=position,
                         street=street, facing_bet=facing_bet, cfg=cfg,
                         level=level)
    if prior.source.startswith("player"):
        return prior, "corpus_individual", layer_warnings
    if prior.source.startswith("population"):
        return prior, "corpus_population", layer_warnings
    return prior, "unavailable", layer_warnings


# ----------------------------------------------------------------------
# 数据源：canonical 语料 → 聚合索引（持久缓存；桶级衰减粒度声明）
# ----------------------------------------------------------------------

def _frequency_prior_config(ctx: AgentContext) -> dict:
    cfg = getattr(ctx, "config", None)
    try:
        return dict(cfg.frequency_prior_config() or {})
    except Exception:
        return {}


def _load_portable_index(path: Path, level: str,
                         warnings: list) -> tuple[dict, dict] | None:
    """EXE-1：装载随包分发的便携聚合索引（population 级桶）。

    文件由 scripts/build_portable_freq_index.py 在**构建期**从完整索引生成；
    schema/level 不匹配 → None（上游走 SOURCE_UNAVAILABLE 诚实降级）。
    """
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if int(blob.get("schema") or 0) != INDEX_SCHEMA_VERSION:
            warnings.append({"code": "PORTABLE_INDEX_SCHEMA_MISMATCH",
                             "message": f"portable index schema mismatch: {path}"})
            return None
        lv = str(blob.get("level") or level)
        inner_raw = (blob.get("index") or {}).get(lv) or {}
        inner = {tuple(json.loads(k)): v for k, v in inner_raw.items()}
        index = {lv: inner}
        prov = {"source_dir": "<portable: source corpus not shipped>",
                "max_chunks": 0, "chunks": [],
                "decay_granularity": "bucket_ts_max",
                "level": level, "index_schema": INDEX_SCHEMA_VERSION,
                "cache_path": str(path), "portable": True,
                "scope": "population-level buckets only (individual "
                         "corpus layer not shipped)"}
        return index, prov
    except Exception as exc:                              # noqa: BLE001
        warnings.append({"code": "PORTABLE_INDEX_READ_FAILED",
                         "message": f"portable index unreadable ({path}): "
                                    f"{exc}; honest degradation"})
        return None


def _load_index(cfg: dict, ctx: AgentContext, warnings: list) -> tuple[dict, dict]:
    """加载聚合索引（模块级缓存 + 可选磁盘缓存）；失败 → 空 + warning。

    M3 v2：索引外层按**级别段**（config::frequency_prior.level，缺省
    DEFAULT_CORPUS_LEVEL）分桶 —— {level: {inner_key: bucket}}；查询只命中
    同级别段，不同级别语料互不污染（POOL_BOT_PLAN 数据政策第 1 条）。
    """
    source_dir = str(cfg.get("source_dir") or "data/uoft/normalized")
    max_chunks = int(cfg.get("max_chunks") or 3)
    cache_path = cfg.get("cache_path") or "data/cache/frequency_prior_cache.json"
    level = str(cfg.get("level") or DEFAULT_CORPUS_LEVEL)
    key = (source_dir, max_chunks, level)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    root = Path(source_dir)
    if not root.exists() or not sorted(root.glob("chunk_*.jsonl"))[:max_chunks]:
        # EXE-1 产品便携模式：源语料（GB 级）不随包分发，但**构建期落盘的
        # 聚合索引**（frequency_prior_portable.json，population 级桶）随包
        # 分发——直接装载，provenance 显式声明 portable（诚实降级，不伪造
        # 源可用性）。仓库内源语料在场时永不触发（零行为变化）。
        portable = _load_portable_index(Path(cache_path).with_name(
            "frequency_prior_portable.json"), level, warnings)
        if portable is not None:
            _MODULE_CACHE[key] = portable
            return portable
    if not root.exists():
        warnings.append({"code": "SOURCE_UNAVAILABLE",
                         "message": f"frequency source dir missing: {source_dir}"})
        return {}, {}
    chunks = sorted(root.glob("chunk_*.jsonl"))[:max_chunks]
    if not chunks:
        warnings.append({"code": "SOURCE_UNAVAILABLE",
                         "message": f"no chunk_*.jsonl under {source_dir}"})
        return {}, {}
    cache_file = Path(cache_path)
    fingerprint = (f"schema={INDEX_SCHEMA_VERSION}|level={level}|"
                   f"{source_dir}|{max_chunks}|" + ",".join(
                       f"{c.name}:{int(c.stat().st_mtime)}" for c in chunks))
    index = None
    pkl = cache_file.with_suffix(".pkl")
    if pkl.exists():
        try:
            import pickle
            blob = pickle.loads(pkl.read_bytes())
            if blob.get("fingerprint") == fingerprint \
                    and int(blob.get("schema") or 0) == INDEX_SCHEMA_VERSION:
                index = blob.get("index")
        except Exception as exc:
            index = None
            warnings.append({
                "code": "CACHE_READ_FAILED",
                "message": (f"frequency prior cache unreadable "
                            f"({pkl}): {exc}; rebuilding from source")})
    if index is None and cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if blob.get("fingerprint") == fingerprint \
                    and int(blob.get("schema") or 0) == INDEX_SCHEMA_VERSION:
                # v2 行格式：[level, k0..k5, n, ts_max]（外层级别段 + 内层桶）
                index: dict[tuple, dict] = {}
                for level_i, k0, k1, k2, k3, k4, k5, n, ts_max in blob.get(
                        "rows", []):
                    index.setdefault(str(level_i), {})[
                        (k0, k1, k2, k3, k4, k5)] = {"n": n, "ts_max": ts_max}
        except Exception as exc:
            index = None
            warnings.append({
                "code": "CACHE_READ_FAILED",
                "message": (f"frequency prior cache unreadable "
                            f"({cache_path}): {exc}; rebuilding from source")})
    if index is None:
        index = build_index_from_chunks(chunks, level=level)
        try:
            import pickle
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pkl = cache_file.with_suffix(".pkl")
            tmp = pkl.with_suffix(".pkl.tmp")
            tmp.write_bytes(pickle.dumps({
                "schema": INDEX_SCHEMA_VERSION,
                "fingerprint": fingerprint, "index": index,
                "built_at": time.time()}, protocol=5))
            tmp.replace(pkl)            # 原子替换：多进程并发写不会产生半个缓存
            # JSON 副本仅作可读审计（构建慢，异步无害）
            try:
                rows = [[level, *k, b.get("n", 0), b.get("ts_max")]
                        for lv, inner in index.items() for k, b in inner.items()]
                cache_file.write_text(json.dumps({
                    "schema": INDEX_SCHEMA_VERSION,
                    "fingerprint": fingerprint, "rows": rows,
                    "built_at": time.time()}, ensure_ascii=False),
                    encoding="utf-8")
            except OSError:
                pass                    # JSON 审计副本失败不影响 pickle 主缓存
        except (OSError, TypeError, ValueError) as exc:
            # BENCH-1 Task0.2：写缓存失败绝不静默（历史实现 here 用
            # `except Exception: pass` 吞掉了 tuple-key 序列化 TypeError，
            # 导致缓存从未真正落盘）——显式降级为进程内索引（决策不被
            # 阻塞；下次启动将重建），并把降级事实带给调用方。
            warnings.append({
                "code": "CACHE_WRITE_FAILED",
                "message": (f"frequency prior cache write failed "
                            f"({cache_path}): {exc}; using in-process index "
                            f"(will rebuild next start)")})
    prov = {"source_dir": source_dir, "max_chunks": max_chunks,
            "chunks": [c.name for c in chunks],
            "decay_granularity": "bucket_ts_max",
            "level": level,
            "index_schema": INDEX_SCHEMA_VERSION,
            "cache_path": str(cache_file)}
    _MODULE_CACHE[key] = (index, prov)
    return index, prov


def build_index_from_chunks(chunks: list[Path],
                            level: str = DEFAULT_CORPUS_LEVEL) -> dict:
    """canonical chunk → 聚合索引（M3 v2：外层级别段键）。

    返回 {level: {inner_key: bucket}}；inner 键（与 v1 兼容）：
      ('player', pid, position, street, facing, action) -> {'n': int, 'ts_max': float}
      ('population', position, street, facing, action)  -> {'n': int, 'ts_max': float}
    不同级别段各自独立成桶——查询侧只看同一 level，跨级别不互相污染。
    桶 ts_max = 该桶观测中最新手牌开始时间（桶级衰减的声明粒度）。
    """
    def bump(store: dict, key: tuple, ts: float | None) -> None:
        b = store.get(key)
        if b is None:
            store[key] = {"n": 1, "ts_max": ts}
        else:
            b["n"] += 1
            if ts is not None and (b["ts_max"] is None or ts > b["ts_max"]):
                b["ts_max"] = ts

    inner: dict[tuple, dict] = {}
    for path in chunks:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                canon = rec.get("canonical") or rec
                players = canon.get("players") or []
                pos_by_player = {str(p.get("player_id")): p.get("position")
                                 for p in players if isinstance(p, dict)}
                ts = _parse_ts(canon.get("started_at"))
                for street in ("preflop", "flop", "turn", "river"):
                    rows = canon.get(f"{street}_actions") or []
                    if not rows:
                        continue
                    obs = fp.observations_from_street_actions(
                        rows, street=street, positions_by_player=pos_by_player,
                        ts=ts)
                    for o in obs:
                        k1 = ("player", o.player_id, o.position, o.street,
                              o.facing_bet, o.action)
                        bump(inner, k1, o.ts)
                        k2 = ("population", o.position, o.street,
                              o.facing_bet, o.action)
                        bump(inner, k2, o.ts)
        except Exception:
            continue                     # 单 chunk 损坏不阻塞（诚实跳过）
    return {str(level): inner}


def _resolve_level_index(index: dict, level: str | None) -> tuple[dict, str | None]:
    """外层级别段解析：显式 level 优先；None → 单段索引直接用（v1 兼容）。"""
    if not isinstance(index, dict) or not index:
        return {}, level
    if level is not None:
        return dict(index.get(str(level)) or {}), str(level)
    levels = [k for k in index.keys() if isinstance(k, str)]
    if len(index) == 1:
        only = next(iter(index.items()))
        return dict(only[1] or {}), str(only[0])
    return {}, ",".join(sorted(levels)) or None


def _query_index(index: dict, *, player_id: str, position: Any,
                 street: str, facing_bet: bool,
                 cfg: fp.FrequencyPriorConfig,
                 level: str | None = None) -> fp.FrequencyPrior:
    """在聚合索引上按回退链查询（桶级计数 → 等权频率；ts_max 做桶级衰减）。

    M3 v2：index 为 {level: {inner_key: bucket}}；查询只看 `level` 段——
    段缺失/为空 → 与"无数据"同路径（fail-closed，不跨级别借数）。
    """
    actions = fp._context_actions(facing_bet)
    inner, resolved_level = _resolve_level_index(index, level)
    if level is not None and not inner:
        return fp.FrequencyPrior(
            player_id=player_id, position=position, street=street,
            facing_bet=facing_bet, source="UNAVAILABLE",
            context_actions=list(actions), frequencies=None,
            p_continue=None, p_fold=None, n_total=0, n_eff_total=0.0,
            warnings=[{"code": "LEVEL_NO_DATA",
                       "message": (f"level segment {level!r} empty or absent "
                                   "(cross-level data is never borrowed)")}],
            config_echo=cfg.to_dict(),
            decay_granularity="bucket_ts_max")

    # 单遍扫描索引，同时聚合全部回退层的行（O(|index|) 每查询）
    per_level: dict[tuple, list] = {lv: [] for lv in fp.FALLBACK_LEVELS}
    acts = set(actions)
    for key, b in inner.items():
        kind = key[0]
        if kind == "player":
            _, pid, pos, st, fac, act = key
            if pid != player_id or fac != facing_bet or act not in acts:
                continue
            if pos == position and st == street:
                per_level[("player", "position", "street")].append((act, b))
            if pos == position:
                per_level[("player", "position")].append((act, b))
            if st == street:
                per_level[("player", "street")].append((act, b))
            per_level[("player",)].append((act, b))
        else:
            pos, st, fac, act = key[1], key[2], key[3], key[4]
            if fac != facing_bet or act not in acts:
                continue
            if pos == position and st == street:
                per_level[("population", "position", "street")].append((act, b))
            if st == street:
                per_level[("population", "street")].append((act, b))
            per_level[("population",)].append((act, b))

    def exact_rows(level: tuple[str, ...]) -> list[tuple[str, dict]]:
        return per_level[level]

    warnings: list[dict] = []
    chosen_level = None
    rows: list[tuple[str, dict]] = []
    for level in fp.FALLBACK_LEVELS:
        r = exact_rows(level)
        total = sum(b.get("n", 0) for _, b in r)
        if total >= cfg.min_sample_size:
            chosen_level, rows = level, r
            break
    if chosen_level is None:
        pop_levels = [lv for lv in fp.FALLBACK_LEVELS if "population" in lv]
        best_level, best_rows, best_total = None, [], 0
        for level in pop_levels:
            r = exact_rows(level)
            total = sum(b.get("n", 0) for _, b in r)
            if total > best_total:
                best_level, best_rows, best_total = level, r, total
        if not best_rows:
            return fp.FrequencyPrior(
                player_id=player_id, position=position, street=street,
                facing_bet=facing_bet, source="UNAVAILABLE",
                context_actions=list(actions), frequencies=None,
                p_continue=None, p_fold=None, n_total=0, n_eff_total=0.0,
                warnings=[{"code": "NO_DATA",
                           "message": "index has no observations for context"}],
                config_echo=cfg.to_dict(),
                decay_granularity="bucket_ts_max")
        chosen_level, rows = best_level, best_rows
        warnings.append({"code": "INSUFFICIENT_SAMPLE",
                         "message": (f"below min_sample_size={cfg.min_sample_size}; "
                                     f"largest population level "
                                     f"({'×'.join(chosen_level)}) n={best_total}")})
    elif chosen_level[0] == "population":
        warnings.append({"code": "FALLBACK_POPULATION",
                         "message": f"fell back to {'×'.join(chosen_level)}"})

    # 桶级时间衰减（确定性）：now = 查询结果内最新 ts_max（同输入同输出）
    n_total = sum(b.get("n", 0) for _, b in rows)
    ts_list = [b.get("ts_max") for _, b in rows if b.get("ts_max") is not None]
    now = max(ts_list) if ts_list else None
    lam = math.log(2) / max(cfg.half_life_days, 0.001)

    def eff(b: dict) -> float:
        n = float(b.get("n", 0))
        ts = b.get("ts_max")
        if now is None or ts is None:
            return n                    # 无时间信息 → 等权（诚实降级）
        return n * math.exp(-lam * max(0.0, now - ts) / 86400.0)

    eff_total = sum(eff(b) for _, b in rows)
    freqs: dict[str, dict] = {}
    p_continue = p_fold = None
    if eff_total > 0:
        by_act_eff: dict[str, float] = {}
        by_act_n: dict[str, int] = {}
        for a in actions:
            by_act_eff[a] = sum(eff(b) for act, b in rows if act == a)
            by_act_n[a] = sum(b.get("n", 0) for act, b in rows if act == a)
        for a in actions:
            rate = by_act_eff[a] / eff_total
            lo, hi = fp._wilson_float(by_act_eff[a], eff_total, cfg.wilson_z)
            freqs[a] = {"action": a, "rate": round(rate, 4),
                        "wilson_low": round(lo, 4),
                        "wilson_high": round(hi, 4),
                        "n": by_act_n[a], "n_eff": round(by_act_eff[a], 3)}
        if facing_bet is True:
            p_continue = (by_act_eff["call"] + by_act_eff["raise"]) / eff_total
            p_fold = by_act_eff["fold"] / eff_total
        elif facing_bet is False:
            p_continue = by_act_eff["bet"] / eff_total
    return fp.FrequencyPrior(
        player_id=player_id, position=position, street=street,
        facing_bet=facing_bet, source="×".join(chosen_level),
        context_actions=list(actions), frequencies=freqs if freqs else None,
        p_continue=p_continue, p_fold=p_fold, n_total=n_total,
        n_eff_total=eff_total, warnings=warnings,
        config_echo=cfg.to_dict(), decay_granularity="bucket_ts_max")


def _parse_ts(value: Any) -> float | None:
    """ISO 时间 → epoch seconds（确定性；解析失败 → None=等权）。"""
    if value is None:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)   # naive 视为 UTC（声明语义）
        return dt.timestamp()
    except Exception:
        return None
