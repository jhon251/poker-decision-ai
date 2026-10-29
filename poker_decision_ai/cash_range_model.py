"""V3G-6 — CASH Range Modeling / 手内范围更新（只读结构层；不含策略、不改变决策）。

定位
----
建立 **CASH-only** 的结构化 opponent range 表示，并支持按**本手已真实观察到
的 action** 做**确定性**更新。

    preflop 初始 range（configs/ranges.yaml，可追溯）
      → 已知牌移除（hero hole cards + board）
      → 按已观察 action 的确定性重加权（单调、有界、可审计）
      → 归一化 + 组合过滤
      → 结构化 range + evidence/provenance

复用而非重实现（审计结论）
--------------------------
* `domain/poker/ranges.py`：`Range` / `parse_range` / `remove_cards` /
  `distribution` / `stats` —— **全部直接复用**；本模块**不重写**任何组合
  数学、卡牌移除或 169 分桶。
* `core/config.load_config().default_ranges()`：**唯一** preflop 初始 range
  来源（`configs/ranges.yaml`）——真实、可追溯、非硬编码。

三层语义（与 V3G-5 一致）
-------------------------
1. **观察事实（observed）** —— 本手逐条已观察 action（含未观察/未知者被标记）；
2. **range 推断（inferred）** —— 组合权重分布 + 169 分桶 + 归一化统计；
3. **UNKNOWN** —— 证据不足时的显式未知（不猜、不用默认）。

硬约束
------
1. **只读 + 确定性**：不修改输入；**不 import `random`**、不采样 runout、
   **不生成未来牌**、无 IO、无 wall-clock；同输入 → 同输出。
2. **不伪造**：不伪造未观察到的 action、不伪造对手 hole cards（对手手牌始终
   未知——本层只给**组合分布**，远非"具体牌力"）、不伪造 future cards。
3. **不把统计当牌力**：V3G-5 的统计**不得**直接当作具体牌力；只允许影响
   **显式声明的** uniform prior multiplier（且默认 1.0 = 不调整）。
4. **多玩家严格隔离**：每个 `player_id` 各自一条 range。
5. **玩家身份隔离**：无 `player_id` 的 action/记录不归属任何玩家。
6. **CASH-only，fail closed**：非 CASH → `present=False`；位置未知 → 不猜。
7. **未知 action 不更新**；重复/冲突 action 的处理见 §更新规则。
8. **不冒充 GTO/equilibrium**：本层是"卡片组合空间上的证据重加权"，
   **不是**均衡解；不含 EV、不含建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context
from .domain.poker.ranges import Range, parse_range

RANGE_MODEL_VERSION = "v3g6-cash-range-model-v1"

#: 已观察 action → **声明式**权重因子（单调、有界；非策略结论）
#: 语义：**投入筹码越多 → 强手权重越高**（单调原则，不主张具体频率）。
#: 未知 action → **不更新**（不在表中）。
ACTION_WEIGHT: dict[str, float] = {
    "fold": 0.25,
    "check": 0.75,
    "call": 1.0,
    "bet": 1.5,
    "raise": 2.0,
    "allin": 3.0,
}

#: 权重上下界（防退化；确定性钳制）
MIN_WEIGHT = 0.05
MAX_WEIGHT = 5.0

#: 低于此权重的组合直接剔除（归一化前的组合过滤；确定性）
DROP_BELOW = 0.05

#: range 组合数上限（输出体积保护；超出则按权重降序截断——确定性）
MAX_COMBOS_OUT = 1326


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _hero(state: dict) -> dict | None:
    players = state.get("players")
    if not isinstance(players, list):
        return None
    hid = state.get("hero_id")
    for p in players:
        if isinstance(p, dict) and hid is not None and p.get("player_id") == hid:
            return p
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _player(state: dict, pid: str) -> dict | None:
    for p in state.get("players") or []:
        if isinstance(p, dict) and str(p.get("player_id")) == str(pid):
            return p
    return None


def _dead_cards(state: dict) -> tuple[list, list]:
    """已知牌 → (dead_cards, unknown_reasons)。**不猜**：缺 hero 手牌即如实标 UNKNOWN。"""
    from .domain.poker.ranges import parse_single_card
    from .domain.poker.cards import RANK_VALUES
    dead: list = []
    unknown: list[str] = []
    hero = _hero(state)
    holes = hero.get("hole_cards") if hero else None
    if isinstance(holes, list) and len(holes) == 2 and all(
            not _is_blank(c) for c in holes):
        for c in holes:
            try:
                dead.append(parse_single_card(str(c)))
            except Exception:
                unknown.append(f"unparsable_card:{c}")
    else:
        unknown.append("hero.hole_cards")
    board = state.get("board")
    if isinstance(board, list):
        for c in board:
            if _is_blank(c):
                unknown.append("board.card")
                continue
            try:
                dead.append(parse_single_card(str(c)))
            except Exception:
                unknown.append(f"unparsable_card:{c}")
    return dead, unknown


def initial_range(position: str | None, *, ranges_config: dict | None = None,
                  route: str = "open") -> dict:
    """preflop 初始 range（**来源可追溯**：configs/ranges.yaml）。

    * `position` 必须已知；未知/不在配置位置表 → `present=False`（**不猜**，
      不使用任何默认位置）；
    * 返回 range 文本、来源路径标记、组合数。
    """
    if _is_blank(position):
        return {"present": False, "reason": "unknown_position"}
    cfg = ranges_config if isinstance(ranges_config, dict) else {}
    if not cfg:
        try:
            from .core.config import load_config
            cfg = load_config().default_ranges()
        except Exception as exc:
            return {"present": False,
                    "reason": f"ranges_config_unavailable:{type(exc).__name__}"}
    positions = cfg.get("positions")
    known = ([str(p) for p in positions] if isinstance(positions, list) else [])
    if known and position not in known:
        return {"present": False, "reason": f"unknown_position:{position}"}
    if route == "open":
        table = cfg.get("open") or {}
        text = table.get(position) if isinstance(table, dict) else None
        if not text:
            return {"present": False,
                    "reason": f"no_configured_range:{route}:{position}"}
    else:
        section = cfg.get(route)
        text = (section.get("default") if isinstance(section, dict)
                else section if isinstance(section, str) else None)
        if not text:
            return {"present": False,
                    "reason": f"no_configured_range:{route}"}
    return {"present": True, "route": route, "position": position,
            "text": str(text), "source": "configs/ranges.yaml",
            "source_key": f"ranges.{route}"
                          + (f".{position}" if route == "open" else ".default")}


def _split_actions(actions: list, skip_pid: str | None) -> tuple[dict, list, list]:
    """行动历史 → 按玩家分区的**已观察**更新序列。

    返回 (per_player, conflicts, unknown)。规则：
      * 无 `player_id` → 不归属（unknown）；
      * `skip_pid`（通常是 hero）→ 不计入对手 range（英雄行动不更新对手范围）；
      * 未知 action → **不更新**（记入 unknown）；
      * 同 `(street, sequence)` 槽位出现**不同** action → 冲突：该槽位
        **两条都不应用**（不猜哪条正确）；
      * 同槽位相同 action → 视为重复上报，应用一次。
    """
    per_player: dict[str, list] = {}
    conflicts: list[dict] = []
    unknown: list[dict] = []
    slots: dict[tuple, dict] = {}
    for i, a in enumerate(actions or []):
        if not isinstance(a, dict):
            unknown.append({"index": i, "reason": "action_not_a_dict"})
            continue
        pid = a.get("player_id")
        if _is_blank(pid):
            unknown.append({"index": i, "reason": "missing_player_id"})
            continue
        pid_s = str(pid)
        if skip_pid is not None and pid_s == str(skip_pid):
            continue
        act = a.get("action")
        if not isinstance(act, str) or act.strip().lower() not in ACTION_WEIGHT:
            unknown.append({"index": i, "player_id": pid_s,
                            "reason": f"unknown_action:{act!r}"})
            continue
        act_n = act.strip().lower()
        street = a.get("street")
        seq = a.get("sequence")
        key = (pid_s, street, seq)
        prev = slots.get(key)
        if prev is not None and prev["action"] != act_n:
            conflicts.append({"player_id": pid_s, "street": street,
                              "sequence": seq, "actions": sorted(
                                  {prev["action"], act_n}),
                              "reason": "conflicting_action_same_slot"})
            # 冲突槽位：两条都不应用
            per_player[pid_s] = [x for x in per_player.get(pid_s, [])
                                 if x["key"] != key]
            continue
        if prev is None:
            entry = {"action": act_n, "street": street, "sequence": seq,
                     "index": i, "key": key}
            slots[key] = entry
            per_player.setdefault(pid_s, []).append(entry)
    return per_player, conflicts, unknown


def _apply_actions(rng: Range, entries: list) -> tuple[Range, list]:
    """按已观察行动做确定性重加权（单调、有界）+ 组合过滤。"""
    out = Range(dict(rng.weights), name=rng.name)
    applied: list[dict] = []
    for e in sorted(entries, key=lambda x: (str(x["street"]), x["index"])):
        factor = ACTION_WEIGHT[e["action"]]
        for combo, w in list(out.weights.items()):
            nw = min(MAX_WEIGHT, max(MIN_WEIGHT, w * factor))
            if nw < DROP_BELOW:
                out.weights.pop(combo, None)
            else:
                out.weights[combo] = nw
        applied.append({"action": e["action"], "street": e["street"],
                        "sequence": e["sequence"], "factor": factor,
                        "source": "declared_ACTION_WEIGHT"})
    return out, applied


def _normalize(rng: Range) -> dict:
    """确定性归一化（最大权重 → 1.0；空 → 空）。

    键使用**既有** `combo_name()` 的 169 分类名（与 `distribution()` 同口径），
    便于与既有 range 工件对照。
    """
    if not rng.weights:
        return {}
    top = max(rng.weights.values())
    if top <= 0:
        return {}
    from .domain.poker.ranges import combo_name
    return {combo_name(k): round(v / top, 6)
            for k, v in sorted(rng.weights.items())}


def build_range_model(state: dict | None, *,
                      ranges_config: dict | None = None,
                      route: str = "open",
                      opponent_model: dict | None = None,
                      prior_multiplier: float = 1.0,
                      skip_player: str | None = None) -> dict:
    """CASH 结构化 opponent range 模型（只读；确定性；不含策略）。

    * `present=False`：非 CASH / 输入非 dict —— fail closed；
    * 每个对手 → 一条 range 条目；证据不足 → 该玩家 `level="unknown"`；
    * `opponent_model`（V3G-5 输出）：**只**允许影响显式声明的 uniform prior
      multiplier（默认 1.0）；统计**不被**当作牌力。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": RANGE_MODEL_VERSION,
                "reason": "state_not_a_dict"}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": RANGE_MODEL_VERSION,
                "reason": ev.get("reason", "not_cash_mode")}

    hero = _hero(state) or {}
    hero_pid = hero.get("player_id")
    dead, dead_unknown = _dead_cards(state)

    actions = state.get("action_history")
    if not isinstance(actions, list):
        actions = []
        actions_unknown = ["action_history"]
    else:
        actions_unknown = []
    per_player, conflicts, act_unknown = _split_actions(actions, hero_pid)

    priors = _priors_from_opponent_model(opponent_model)
    base_mult = _num(prior_multiplier)
    if base_mult is None or base_mult < 0:
        base_mult = 1.0

    model: dict[str, dict] = {}
    for p in state.get("players") or []:
        if not isinstance(p, dict):
            continue
        pid = p.get("player_id")
        if _is_blank(pid):
            continue
        pid_s = str(pid)
        if hero_pid is not None and pid_s == str(hero_pid):
            continue
        if p.get("status") == "folded":
            continue
        model[pid_s] = _player_range(
            pid_s, p, dead, dead_unknown, per_player.get(pid_s, []),
            ranges_config=ranges_config, route=route,
            prior_mult=base_mult, prior=priors.get(pid_s))

    return {
        "present": True, "version": RANGE_MODEL_VERSION, "mode": ev.get("mode"),
        "route": route,
        "known_cards": {"count": len(dead), "unknown": dead_unknown},
        "model": model,
        "conflicts": conflicts,
        "unknown": actions_unknown + [x for x in act_unknown],
        "evidence": {"present": True, "source": "v3g6_cash_range_model",
                     "engine": "domain.poker.ranges.Range",
                     "initial_range_source": "configs/ranges.yaml",
                     "state_integrity": ev.get("state_integrity")},
        "integrity": dict(ev.get("state_integrity") or {}),
    }


def _priors_from_opponent_model(opponent_model: dict | None) -> dict:
    """V3G-5 输出 → 每玩家 **可追溯** 的 prior（默认不调整）。

    严格要求：只有 `confidence.level == "adequate"` **且** vpip 自身
    `level == "adequate"` 才产生 multiplier；否则不调整。

    返回**带 provenance 的记录**（授权要求："所有由统计推断产生的 range 权重
    必须可追溯到 metric、samples、evidence"）：

        {"multiplier": float,
         "provenance": {"source", "player_id", "metric", "value", "samples",
                        "metric_level", "confidence_level", "evidence",
                        "anchor", "slope", "bounds", "note"}}
    """
    out: dict[str, dict] = {}
    if not isinstance(opponent_model, dict) or not opponent_model.get("present"):
        return out
    for pid, m in (opponent_model.get("model") or {}).items():
        if not isinstance(m, dict):
            continue
        conf = (m.get("confidence") or {}).get("level")
        vpip = ((m.get("inferred") or {}).get("vpip") or {})
        if conf != "adequate" or vpip.get("level") != "adequate":
            continue
        val = _num(vpip.get("value"))
        samples = vpip.get("samples")
        if val is None or not isinstance(samples, int):
            continue
        # 声明式缩放：以 0.30 为中性锚点，斜率 0.5，钳制于 [0.8, 1.25]
        raw = 1.0 + (val - 0.30) * 0.5
        mult = round(min(1.25, max(0.8, raw)), 4)
        out[str(pid)] = {
            "multiplier": mult,
            "provenance": {
                "source": "v3g5_opponent_model",
                "player_id": str(pid),
                "metric": "vpip",
                "value": val,
                "samples": samples,
                "metric_level": vpip.get("level"),
                "confidence_level": conf,
                "evidence": str(vpip.get("evidence") or ""),
                "anchor": 0.30, "slope": 0.5, "bounds": [0.8, 1.25],
                "note": ("显式声明的 uniform prior —— 只缩放、不改形状；"
                         "统计**不是**牌力"),
            },
        }
    return out


def _player_range(pid: str, p: dict, dead: list, dead_unknown: list,
                  entries: list, *, ranges_config: dict | None,
                  route: str, prior_mult: float,
                  prior: float | None) -> dict:
    pos = p.get("position")
    init = initial_range(pos if not _is_blank(pos) else None,
                         ranges_config=ranges_config, route=route)
    entry: dict[str, Any] = {
        "player_id": pid,
        "position": pos if not _is_blank(pos) else None,
        "initial_range": init,
        "observed": {
            "actions": [{"action": e["action"], "street": e["street"],
                         "sequence": e["sequence"]} for e in entries],
            "applied_factors": [],
        },
        "inferred": None,
        "level": None,
        "evidence": {"prior_multiplier": prior_mult,
                     "opponent_prior_multiplier": (
                         prior["multiplier"] if prior else None),
                     "opponent_prior_provenance": (
                         prior["provenance"] if prior else None),
                     "card_removal": {"dead_count": len(dead),
                                      "unknown": list(dead_unknown)}},
    }
    if not init.get("present"):
        entry["level"] = "unknown"
        entry["reason"] = init.get("reason")
        return entry

    try:
        rng = parse_range(init["text"], name=f"{route}:{pos}")
    except Exception as exc:
        entry["level"] = "unknown"
        entry["reason"] = f"unparsable_range:{type(exc).__name__}"
        return entry

    # 1) 已知牌移除（确定性；hero 手牌 + board）
    rng = rng.remove_cards(dead)
    # 2) 声明式 uniform prior multiplier（默认 1.0；V3G-5 仅当充分样本）
    eff = prior_mult * (prior["multiplier"] if prior else 1.0)
    if abs(eff - 1.0) > 1e-9:
        for k, w in list(rng.weights.items()):
            rng.weights[k] = min(MAX_WEIGHT, max(MIN_WEIGHT, w * eff))
    # 3) 已观察行动 → 确定性重加权 + 组合过滤
    rng, applied = _apply_actions(rng, entries)
    entry["observed"]["applied_factors"] = applied
    # 4) 归一化 + 分桶
    norm = _normalize(rng)
    if not norm:
        entry["level"] = "unknown"
        entry["reason"] = "range_empty_after_filtering"
        entry["inferred"] = {"combos": 0, "distribution": {},
                             "stats": None, "normalized": {}}
        return entry

    items = sorted(norm.items(), key=lambda kv: (-kv[1], kv[0]))
    truncated = len(items) > MAX_COMBOS_OUT
    entry["inferred"] = {
        "combos": len(norm),
        "combos_returned": min(len(items), MAX_COMBOS_OUT),
        "truncated": truncated,
        "normalized": dict(items[:MAX_COMBOS_OUT]),
        "distribution": rng.distribution(),
        "stats": rng.stats(),
        "note": ("组合权重分布；对手**具体手牌仍未知**——"
                 "这不是牌力，也不是均衡解"),
    }
    entry["level"] = "inferred"
    return entry


__all__ = ["RANGE_MODEL_VERSION", "ACTION_WEIGHT", "MIN_WEIGHT", "MAX_WEIGHT",
           "DROP_BELOW", "MAX_COMBOS_OUT", "initial_range",
           "build_range_model"]
