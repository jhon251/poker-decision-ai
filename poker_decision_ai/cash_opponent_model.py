"""V3G-5 — CASH 对手建模闭环（只读结构层；不含策略、不改变任何决策）。

定位
----
把项目中**既有**的对手统计能力，组织成 **CASH-only** 的结构化对手模型，并
**明确区分三层**：

  1. **观测事实（observed）** —— 逐手布尔证据（只记录真的被观察到的行动/机会）；
  2. **统计推断（inferred）** —— 由 `stats.compute_all` 计算的比率 + 样本量 +
     置信状态（**小样本不得强行分类**）；
  3. **UNKNOWN** —— 样本为 0、字段未观察、或记录被排除时的**显式未知**。

复用而非重实现（审计结论）
--------------------------
* `domain/poker/stats.py`：`compute_all` / `Metric` / `StatsBundle` —— 全部
  标准指标（vpip / pfr / three_bet / fold_to_3bet / four_bet / cbet /
  fold_to_cbet / wtsd / won_at_showdown / wwsf / aggression_factor /
  aggression_frequency / bb_per_100 / roi）**直接复用**；本模块**不重算**
  任何比率。
* `stats.bayes_shrink` / `stats.wilson_interval` / `HAND_RECORD_FIELDS` —— 复用。
* V3G-2 `cash_integrity_context` —— 模式判定与证据旁路（唯一口径）。

硬约束
------
1. **只读 + 确定性**：不修改输入；无 IO、**无随机**、无 wall-clock；
   同输入 → 同输出。
2. **玩家身份隔离**：模型按 `player_id` 分区；**禁止跨玩家污染**
   （每个玩家的记录只进自己的分区；测试逐项断言）。
3. **不得伪造未观察到的行动**：只有 `True`/`False` 布尔值计入样本；
   `None`/缺失 → 不计入（沿用 `stats._rate` 的既有语义），且**不**补默认为
   否。"本手行动"作为新证据进入模型时，同样只写**真实观察到**的字段。
4. **小样本 = UNKNOWN/低置信**：样本为 0 → `level="unknown"`（值不暴露）；
   低于阈值 → `low`；不强行贴标签（本层**不产出** player type 分类）。
5. **CASH-only，fail closed**：非 CASH → `present=False`；非 CASH 的**手牌
   记录**不并入 CASH 统计，并记入 `excluded`。
6. **不使用视觉推断替代行为证据**：只消费行动/机会布尔证据，不消费视觉字段。
7. **不接 Solver / 不声称 exploit/GTO 强度提升**：本层无 EV、无策略、无建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context
from .domain.poker.stats import (HAND_RECORD_FIELDS, Metric, compute_all)

OPPONENT_MODEL_VERSION = "v3g5-cash-opponent-model-v1"

#: 既有指标全集（**源自** `stats.compute_all` 的键；本层不新增指标）
METRIC_KEYS: tuple[str, ...] = (
    "vpip", "pfr", "three_bet", "fold_to_3bet", "four_bet", "cbet",
    "fold_to_cbet", "wtsd", "won_at_showdown", "wwsf", "aggression_factor",
    "aggression_frequency", "bb_per_100", "roi",
)

#: 置信阈值（**展示性分级**，非策略参数；仅用于表达样本充分度）
MIN_SAMPLES = 30          # < 30 → UNKNOWN（不暴露比率）
ADEQUATE_SAMPLES = 100    # >= 100 → adequate

LEVEL_UNKNOWN = "unknown"
LEVEL_LOW = "low"
LEVEL_ADEQUATE = "adequate"

#: 非 CASH 的 `game` 标记（fail closed：不并入 CASH 统计）
NON_CASH_GAMES = ("mtt", "tournament", "sng", "satellite", "pko", "ko",
                  "mystery", "spin")

#: 行动 → 手牌记录布尔字段（**只写真实观察到的**）。
#: 注意：`fold`/`check`/`call` 只构成 VPIP 证据；**不能**据此推断 PFR
#: （未观察到加注 ≠ 未加注以外的事实）——故不写 pfr。
_ACTION_TO_FLAGS: dict[str, tuple[str, ...]] = {
    "fold": ("vpip",),
    "check": ("vpip",),
    "call": ("vpip",),
    "bet": ("vpip", "pfr"),
    "raise": ("vpip", "pfr"),
    "allin": ("vpip", "pfr"),
}


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _metric_entry(m: Metric | None) -> dict:
    """Metric → 结构化条目（UNKNOWN 语义：samples==0 → **值不暴露**）。"""
    if m is None or m.samples <= 0:
        return {"value": None, "samples": 0, "level": LEVEL_UNKNOWN,
                "evidence": "no_observed_samples"}
    if m.samples >= ADEQUATE_SAMPLES:
        level = LEVEL_ADEQUATE
    elif m.samples < MIN_SAMPLES:
        level = LEVEL_UNKNOWN
    else:
        level = LEVEL_LOW
    return {"value": round(m.value, 4), "samples": m.samples, "level": level,
            "evidence": f"observed_samples={m.samples}"}


def _split_records(records: list) -> tuple[dict, list, list, list]:
    """按 player_id 分区；返回 (per_player, conflicts, excluded, incomplete)。"""
    per_player: dict[str, list] = {}
    conflicts: list[dict] = []
    excluded: list[dict] = []
    incomplete: list[dict] = []
    seen: dict[tuple, dict] = {}
    for i, r in enumerate(records or []):
        if not isinstance(r, dict):
            incomplete.append({"index": i, "reason": "record_not_a_dict"})
            continue
        pid = r.get("player_id")
        if pid is None or (isinstance(pid, str) and not pid.strip()):
            # 无身份 → **不**归入任何玩家（禁止跨玩家污染 / 禁止猜身份）
            incomplete.append({"index": i, "reason": "missing_player_id"})
            continue
        pid_s = str(pid)
        game = r.get("game")
        if isinstance(game, str) and game.strip().lower() in NON_CASH_GAMES:
            excluded.append({"index": i, "player_id": pid_s,
                             "reason": f"non_cash_game:{game}"})
            continue
        hid = r.get("hand_id")
        if hid is not None:
            key = (pid_s, str(hid))
            prev = seen.get(key)
            if prev is not None:
                # 冲突数据：同一玩家同一手出现**不同**记录 → 如实标记，
                # 且**两条都不采信**（不猜哪条正确）
                if _record_signature(prev) != _record_signature(r):
                    conflicts.append({"player_id": pid_s, "hand_id": str(hid),
                                      "reason": "duplicate_hand_id_conflicting"})
                    per_player[pid_s] = [x for x in per_player.get(pid_s, [])
                                         if str(x.get("hand_id")) != str(hid)]
                    continue
            seen[key] = r
        per_player.setdefault(pid_s, []).append(r)
    return per_player, conflicts, excluded, incomplete


def _record_signature(r: dict) -> tuple:
    """记录的可比签名（只含**证据字段**；与键顺序无关）。"""
    return tuple((k, r.get(k)) for k in HAND_RECORD_FIELDS)


def build_opponent_model(state: dict | None, *,
                         records: list | None = None,
                         this_hand_actions: list | None = None,
                         prior: dict | None = None) -> dict:
    """CASH 结构化对手模型（只读；确定性；不含策略）。

    参数:
      state              : 规范化 CASH state（提供模式判定与证据旁路）
      records            : 归一化手牌记录（`stats.HAND_RECORD_FIELDS` 形状）
      this_hand_actions  : 本手已观察到的行动
                           `[{player_id, action, street, amount, sequence}]`
      prior              : 可选先验（仅**原样转述**，不做收缩运算）

    返回:
      {"present", "version", "reason"?, "model": {player_id: {...}},
       "conflicts", "excluded", "incomplete", "evidence"}
    """
    if not isinstance(state, dict):
        return {"present": False, "version": OPPONENT_MODEL_VERSION,
                "reason": "state_not_a_dict"}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": OPPONENT_MODEL_VERSION,
                "reason": ev.get("reason", "not_cash_mode")}

    per_player, conflicts, excluded, incomplete = _split_records(records or [])
    obs, obs_incomplete = _observations_to_records(this_hand_actions or [])
    incomplete.extend(obs_incomplete)
    for pid, rs in obs.items():
        per_player.setdefault(pid, []).extend(rs)

    model: dict[str, dict] = {}
    for pid in sorted(per_player):
        hands = per_player[pid]
        bundle = compute_all(hands)
        metrics = {k: _metric_entry(bundle.metrics.get(k))
                   for k in METRIC_KEYS}
        observed = sum(1 for m in metrics.values() if m["samples"] > 0)
        model[pid] = {
            "player_id": pid,
            "hands": len(hands),
            "hands_with_evidence": sum(
                1 for h in hands
                if any(_is_bool(h.get(f)) for f in
                       ("vpip", "pfr", "three_bet", "cbet", "faced_cbet",
                        "saw_flop", "went_to_showdown"))),
            "observed": _observed_facts(hands),
            "inferred": metrics,
            "confidence": _confidence(hands),
            "prior": (dict(prior) if isinstance(prior, dict) else None),
            "evidence": {
                "source": "normalized_hand_records+this_hand_actions",
                "metric_engine": "domain.poker.stats.compute_all",
                "player_isolation": True,
                "note": ("观测事实与统计推断分区存放；样本不足 → UNKNOWN，"
                         "不强行分类；无视觉推断"),
            },
        }

    return {
        "present": True, "version": OPPONENT_MODEL_VERSION,
        "mode": ev.get("mode"),
        "model": model,
        "conflicts": conflicts, "excluded": excluded,
        "incomplete": incomplete,
        "evidence": {"present": True, "state_integrity": ev.get("state_integrity"),
                     "source": "v3g5_cash_opponent_model"},
        "integrity": dict(ev.get("state_integrity") or {}),
    }


def _observed_facts(hands: list) -> dict:
    """**观测事实**计数（只数布尔证据；None 不计入）。

    键名与 `inferred` 的指标名区分（`observed.<name>` vs
    `observed.<name>_opportunity`），避免同一字典里出现同名不同语义的条目。
    """
    facts: dict[str, dict] = {}
    for name, field_name in (("vpip", "vpip"), ("pfr", "pfr"),
                             ("three_bet", "three_bet"),
                             ("fold_to_3bet", "folded_to_3bet"),
                             ("cbet", "cbet"),
                             ("fold_to_cbet", "folded_to_cbet"),
                             ("wtsd", "went_to_showdown"),
                             ("won_at_showdown", "won_at_showdown"),
                             ("saw_flop", "saw_flop")):
        obs = sum(1 for h in hands if _is_bool(h.get(field_name)))
        true_n = sum(1 for h in hands if h.get(field_name) is True)
        facts[name] = {"kind": "flag", "field": field_name,
                       "observed": obs, "true": true_n,
                       "unknown": len(hands) - obs}
    for name, opp, act in (("cbet", "cbet_opportunity", "cbet"),
                           ("fold_to_cbet", "faced_cbet", "folded_to_cbet"),
                           ("fold_to_3bet", "faced_3bet", "folded_to_3bet")):
        moments = sum(1 for h in hands if h.get(opp) is True)
        taken = sum(1 for h in hands
                    if h.get(opp) is True and h.get(act) is True)
        facts[f"{name}_opportunity"] = {
            "kind": "opportunity", "opportunity_field": opp, "act_field": act,
            "opportunities": moments, "taken": taken,
            "unknown": len(hands) - moments}
    return facts


def _confidence(hands: list) -> dict:
    """置信状态：由**样本量**决定（不强行分类；不产出 player type）。"""
    n = len(hands)
    with_evidence = sum(
        1 for h in hands
        if any(_is_bool(h.get(f)) for f in ("vpip", "pfr", "three_bet",
                                            "cbet", "saw_flop")))
    if with_evidence == 0:
        level = LEVEL_UNKNOWN
    elif with_evidence < MIN_SAMPLES:
        level = LEVEL_UNKNOWN
    elif with_evidence < ADEQUATE_SAMPLES:
        level = LEVEL_LOW
    else:
        level = LEVEL_ADEQUATE
    return {"level": level, "hands": n, "hands_with_evidence": with_evidence,
            "min_samples": MIN_SAMPLES, "adequate_samples": ADEQUATE_SAMPLES,
            "type_label": None,
            "note": "小样本不分类；分类属后续阶段（本层不产出 type）"}


def _observations_to_records(actions: list) -> tuple[dict, list]:
    """本手行动 → 逐玩家手牌记录**增补**（只写真实观察到的字段）。

    规则：
      * 未知行动（`action` 缺失/非法）→ **不写任何布尔字段**（不计入样本），
        并记入 `incomplete`；
      * 只对**该行动者自己**写入（禁止跨玩家污染）；
      * 不推断未观察到的字段（保持缺省）。
    """
    out: dict[str, list] = {}
    incomplete: list[dict] = []
    for i, a in enumerate(actions or []):
        if not isinstance(a, dict):
            incomplete.append({"index": i, "reason": "action_not_a_dict"})
            continue
        pid = a.get("player_id")
        if pid is None or (isinstance(pid, str) and not pid.strip()):
            incomplete.append({"index": i, "reason": "missing_player_id"})
            continue
        pid_s = str(pid)
        act = a.get("action")
        if not isinstance(act, str) or act.strip().lower() not in _ACTION_TO_FLAGS:
            incomplete.append({"index": i, "player_id": pid_s,
                               "reason": f"unknown_action:{act!r}"})
            continue
        flags = _ACTION_TO_FLAGS[act.strip().lower()]
        rec: dict[str, Any] = {
            "player_id": pid_s,
            "hand_id": a.get("hand_id"),
            "game": a.get("game", "cash"),
            "ts": a.get("ts"),
        }
        if "vpip" in flags:
            rec["vpip"] = True
        if "pfr" in flags:
            rec["pfr"] = True
        if "cbet" in flags:
            rec["cbet"] = True
        out.setdefault(pid_s, []).append(rec)
    return out, incomplete


__all__ = ["OPPONENT_MODEL_VERSION", "METRIC_KEYS", "MIN_SAMPLES",
           "ADEQUATE_SAMPLES", "LEVEL_UNKNOWN", "LEVEL_LOW",
           "LEVEL_ADEQUATE", "NON_CASH_GAMES", "build_opponent_model"]
