"""P65 — Mode Switch（CASH/MTT/PKO 模式隔离；构造性防污染）。

职责：把 P44/P60 的三套独立策略语义正式接入 mode switch：
  resolve_mode(state)          → 语义模式（cash|mtt|pko|ko|mystery|unknown）
  sanitize_state_for_mode      → 按 mode 剥离**禁止**的跨 mode 上下文
                                 （构造性隔离——compute_ev 冻结不动）；
                                 缺失的 mode 内上下文原样保留（诚实
                                 unavailable，不借用其他 mode 补全）
  mode_capability_report       → 每能力的状态（permitted/enabled/disabled/
                                 forbidden/unavailable:missing_X）
  verify_mode_isolation        → 对 intel 结果做隔离审计（违规列表）

能力矩阵（contract）：
  CASH   : icm=forbidden  bounty=forbidden（不得产生 ICM/payout/bounty）
  MTT    : icm=permitted  bounty=forbidden（不得启用 PKO bounty 逻辑）
  PKO/KO/Mystery : icm=permitted  bounty=permitted

M8 隔离补强：MTT 与 CASH 一样剥离全部 bounty 载体
（`bounty_type`、顶层 `bounties`、玩家级 `bounty`）。

确定性：纯函数、深拷贝、无 IO。
"""
from __future__ import annotations

import copy
from typing import Any

SEMANTIC_MODES = ("cash", "mtt", "pko", "ko", "mystery")

#: 能力矩阵（contract；mode → {capability: permitted|forbidden}）
MODE_CAPABILITIES: dict[str, dict[str, str]] = {
    "cash": {"icm": "forbidden", "payouts": "forbidden",
             "bounty": "forbidden"},
    "mtt": {"icm": "permitted", "payouts": "permitted",
            "bounty": "forbidden"},
    "pko": {"icm": "permitted", "payouts": "permitted",
            "bounty": "permitted"},
    "ko": {"icm": "permitted", "payouts": "permitted",
           "bounty": "permitted"},
    "mystery": {"icm": "permitted", "payouts": "permitted",
                "bounty": "permitted"},
}

#: 各 mode 禁止进入 compute_ev 的 state 字段（跨 mode 污染源）
#: M8：MTT 的禁止键必须与 CASH 同样覆盖顶层 `bounties` 表 —— 契约是
#: 「MTT: icm=permitted, bounty=forbidden」，而 `bounties` 是 PKO 语义
#: 载体；此前仅剥离 `bounty_type` / 玩家级 `bounty`，顶层 `bounties`
#: 会残留（PKO 状态串入 MTT）。补齐后 MTT 下不再残留任何 bounty 载体。
_FORBIDDEN_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "cash": ("tournament", "bounty_type", "bounties"),
    "mtt": ("bounty_type", "bounties"),
}
#: 玩家级跨 mode 污染字段
_FORBIDDEN_PLAYER_KEYS: dict[str, tuple[str, ...]] = {
    "cash": ("bounty",),
    "mtt": ("bounty",),
}


def resolve_mode(state: dict) -> tuple[str, str]:
    """语义模式解析（确定性）。返回 (mode, unknown_reason)。

    mode ∈ cash|mtt|pko|ko|mystery|unknown；game_type 缺失/未知 →
    unknown（UNKNOWN-first，不猜测）。
    """
    gt = str((state or {}).get("game_type") or "").strip().lower()
    if not gt:
        return "unknown", "game_type missing"
    if gt in SEMANTIC_MODES:
        return gt, ""
    return "unknown", f"unknown game_type {gt!r}"


def _strip_player_bounty(players: list) -> list:
    out = []
    for p in players:
        if isinstance(p, dict):
            p = {k: v for k, v in p.items() if k != "bounty"}
        out.append(p)
    return out


def sanitize_state_for_mode(state: dict, mode: str) -> tuple[dict, list[str]]:
    """按 mode 剥离禁止的跨 mode 上下文（深拷贝；确定性）。

    只剥离 **forbidden** 字段（cash 的 tournament/bounty、mtt 的
    bounty）；mode 内合法但缺失的上下文原样保留——缺失 → compute_ev
    诚实 unavailable，不借用其他 mode 补全。返回 (sanitized, stripped)。
    """
    st = copy.deepcopy(state or {})
    stripped: list[str] = []
    for key in _FORBIDDEN_STATE_KEYS.get(mode, ()):
        if key in st and st[key] not in (None, [], {}):
            stripped.append(f"state.{key}")
        st.pop(key, None)
    if mode in _FORBIDDEN_PLAYER_KEYS:
        keys = _FORBIDDEN_PLAYER_KEYS[mode]
        players = st.get("players")
        if isinstance(players, list):
            new_players = []
            for p in players:
                if isinstance(p, dict):
                    for k in keys:
                        if k in p and p[k] not in (None, 0, 0.0, [], {}):
                            stripped.append(f"players.{k}")
                    p = {kk: vv for kk, vv in p.items() if kk not in keys}
                new_players.append(p)
            st["players"] = new_players
    return st, stripped


def mode_capability_report(state: dict, mode: str) -> dict:
    """能力报告：每能力 permitted/forbidden + enabled/unavailable 状态。

    enabled = permitted 且上下文真实存在；unavailable:missing_X =
    permitted 但 X 缺失（诚实降级，不借用其他 mode 补全）。
    """
    caps = MODE_CAPABILITIES.get(mode)
    if caps is None:
        return {"mode": mode, "status": "unknown",
                "capabilities": {},
                "unknown_reason": f"unknown mode {mode!r}"}
    has_payouts = bool((state.get("tournament") or {}).get("payouts"))
    has_bounty = any(
        isinstance(p, dict) and p.get("bounty")
        for p in (state.get("players") or [])) \
        or bool(state.get("bounty_type"))
    report: dict[str, dict] = {}
    for cap, perm in caps.items():
        entry: dict[str, str] = {"permission": perm}
        if perm == "forbidden":
            entry["status"] = "forbidden"
        elif cap in ("icm", "payouts"):
            entry["status"] = ("enabled" if has_payouts
                               else "unavailable:missing_payouts")
        elif cap == "bounty":
            entry["status"] = ("enabled" if has_bounty
                               else "unavailable:missing_bounty_context")
        report[cap] = entry
    return {"mode": mode, "status": "resolved",
            "capabilities": report}


def verify_mode_isolation(intel_result: dict, mode: str) -> list[str]:
    """对 intel 结果做隔离审计（确定性；违规列表）。

    cash → objective 必须 chip_ev 且无 icm/bounty fallback 激活；
    mtt → 不得出现 bounty_util；pko/ko/mystery → bounty 允许但须有
    上下文来源。违规即列出（不修改结果）。
    """
    v: list[str] = []
    caps = MODE_CAPABILITIES.get(mode)
    if caps is None:
        return [f"unknown mode {mode!r}"]
    if not intel_result.get("available"):
        return v
    if caps["icm"] == "forbidden":
        if intel_result.get("provenance", {}).get("icm_available"):
            v.append("cash_icm_activated")
        if intel_result.get("objective") != "chip_ev":
            v.append(f"cash_objective:{intel_result.get('objective')}")
    if caps["bounty"] == "forbidden":
        for c in intel_result.get("candidates") or []:
            if c.get("adjustments", {}).get("bounty_util_bb"):
                v.append(f"mtt_bounty_util:{c['action']}")
    return v
