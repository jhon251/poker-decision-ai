"""PI-EV 输入单位契约（修复 `_state_for` 归一化被二次缩放缺陷）。

背景（缺陷实证，见 `docs/V3F_R22PRIME_ACCEPTANCE_REPORT.md` §5）
------------------------------------------------------------
`pluribus_validation._state_for()` 返回的 state **已是 bb 归一化**：

    state["bb"]        = 1.0                    ← 归一化标记（不是筹码 bb！）
    state["pot"]       = pot_before / hand_bb   ← **单位 = bb**
    state["to_call"]   = to_call   / hand_bb    ← **单位 = bb**
    players[i]["stack"] = stacks[i] / hand_bb   ← **单位 = bb**
    hand["bb"]         = 100.0（Pluribus）      ← **单位 = 筹码**，仅供还原/审计

历史缺陷：调用方**再次除以 hand_bb(100)** → pot 6.0 → 0.06、stack 100.0 → 1.0
（100× 偏小），导致 PI-EV 参照被风险项主导 → 仪器失效。

本模块提供**唯一合法**的 PI-EV 输入构造路径，并把单位混淆变成**硬失败**：
  * `pi_ev_inputs(state, raw=None)` —— 只从归一化 state 取值；**禁止**再缩放；
    若 state 显式为**筹码单位**（`state["bb"] > 1.0`）则**恰好转换一次**。
  * `UnitContractError` —— 任何单位/比例异常直接抛错（不得静默继续）。
  * `sanity_check(...)` —— 与原始动作记录（`pot_before`/`to_call`）比对比例，
    检出 100×、0.01× 等异常；PI-EV 计算前必须调用。
  * `AuditLog` —— 逐条记录 raw → normalized → PI-EV 输入（采样门）。

命名/类型防混淆（不做花哨类型系统，靠显式契约 + 运行时断言）：
  * 所有 PI-EV 输入字段名一律带 `_bb` 后缀；
  * 任何以 `_chips` 结尾的字段**不得**直接进入 PI-EV；
  * `assert_bb(...)` 校验量级合理性（pot/stack/to_call 的 bb 上界）。
"""
from __future__ import annotations

from typing import Any

UNIT_CONTRACT_VERSION = "pi-ev-unit-contract-v1"

# 量级上界（用于捕获「未归一化/重复归一化」两类典型错误）
MAX_POT_BB = 10_000.0
MAX_STACK_BB = 100_000.0
MAX_TO_CALL_BB = 100_000.0
# 与原始记录比对时允许的比例带（检出 100× / 0.01× 类错误）
RATIO_TOL = (0.5, 2.0)


class UnitContractError(ValueError):
    """单位契约被违反（禁止降级继续；调用方必须 BLOCKED/拒绝该点）。"""


def assert_bb(value: float, *, name: str, maximum: float) -> float:
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise UnitContractError(f"{name} 非有限值: {value!r}")
    if v < 0:
        raise UnitContractError(f"{name} 为负: {v}")
    if v > maximum:
        raise UnitContractError(
            f"{name}={v} 超 bb 合理上界 {maximum} → 疑似未归一化的筹码值")
    return v


def _validate_against_raw(inputs: dict, raw: dict, hand_bb: float) -> dict:
    """与原始动作记录（筹码单位）比对；比例异常 → 抛错。"""
    hb = float(hand_bb)
    if hb <= 0:
        raise UnitContractError(f"hand_bb 非法: {hand_bb!r}")
    expected_pot = float(raw.get("pot_before") or 0.0) / hb
    expected_to_call = float(raw.get("to_call") or 0.0) / hb
    ratios: dict[str, float | None] = {}
    for name, got, exp in (("pot_bb", float(inputs["pot_bb"]), expected_pot),
                           ("to_call_bb", float(inputs["to_call_bb"]),
                            expected_to_call)):
        if exp > 1e-9:
            ratio = got / exp
            ratios[name] = round(ratio, 9)
            if not (RATIO_TOL[0] <= ratio <= RATIO_TOL[1]):
                raise UnitContractError(
                    f"{name} 与原始记录比例异常 {ratio:.6g}"
                    f"（got={got}, expected={exp}, hand_bb={hb}）"
                    f" → 疑似 100×/0.01× 标度错误")
    return ratios


def pi_ev_inputs(state: dict, *, raw: dict | None = None,
                 hand_bb: float | None = None) -> dict:
    """返回 PI-EV 唯一合法输入：`{pot_bb, to_call_bb, hero_stack_bb}`（单位 bb）。

    规则：
      1. 默认路径：直接取 `state["pot"] / state["to_call"] / hero.stack`——
         **它们已是 bb，禁止再除**（历史缺陷即发生在此处）。
      2. 仅当 `state["bb"]` 显式 > 1.0（= 筹码单位 state）时，才做**唯一一次**
         `/ state["bb"]` 转换。
      3. 若提供 `raw`（原始动作 dict，含 `pot_before`/`to_call`，单位=筹码）与
         `hand_bb`，则做比例校验：`pot_bb × hand_bb ≈ pot_before`。
    """
    hero = next((p for p in state.get("players") or []
                 if isinstance(p, dict) and p.get("is_hero")), None)
    if hero is None:
        raise UnitContractError("state 缺 hero 玩家，无法构造 PI-EV 输入")
    state_bb = float(state.get("bb") or 0.0)
    if state_bb <= 0:
        raise UnitContractError(f"state['bb'] 非法: {state.get('bb')!r}")
    already_normalized = abs(state_bb - 1.0) < 1e-12
    factor = 1.0 if already_normalized else (1.0 / state_bb)
    pot_bb = assert_bb(float(state.get("pot") or 0.0) * factor,
                       name="pot_bb", maximum=MAX_POT_BB)
    to_call_bb = assert_bb(float(state.get("to_call") or 0.0) * factor,
                           name="to_call_bb", maximum=MAX_TO_CALL_BB)
    hero_stack_bb = assert_bb(float(hero.get("stack") or 0.0) * factor,
                              name="hero_stack_bb", maximum=MAX_STACK_BB)
    out = {"pot_bb": round(pot_bb, 9), "to_call_bb": round(to_call_bb, 9),
           "hero_stack_bb": round(hero_stack_bb, 9),
           "normalized_once": bool(already_normalized),
           "conversion_factor": factor}
    if raw is not None and hand_bb:
        out["ratios_vs_raw"] = _validate_against_raw(out, raw, hand_bb)
    return out


def sanity_check(inputs: dict, *, raw: dict | None = None,
                 hand_bb: float | None = None) -> dict:
    """PI-EV 计算**前**的 sanity check（返回证据；异常直接抛错）。"""
    if not all(k in inputs for k in ("pot_bb", "to_call_bb", "hero_stack_bb")):
        raise UnitContractError(f"PI-EV 输入字段缺失: {sorted(inputs)}")
    if inputs["to_call_bb"] > inputs["hero_stack_bb"] + 1e-9:
        # to_call > 有效栈 → allin call 语义，合法但需记录
        pass
    if inputs["pot_bb"] + inputs["to_call_bb"] <= 0:
        raise UnitContractError(f"pot/to_call 同时为 0，输入可疑: {inputs}")
    evidence = {"units": "bb", "contract": UNIT_CONTRACT_VERSION,
                "inputs": dict(inputs)}
    if raw is not None and hand_bb:
        evidence["raw"] = {"pot_before_chips": raw.get("pot_before"),
                           "to_call_chips": raw.get("to_call"),
                           "hand_bb_chips": float(hand_bb)}
        # 仪器门：与原始记录的比例必须落在容忍带内（检出 100×/0.01×）
        evidence["ratios_vs_raw"] = _validate_against_raw(inputs, raw, hand_bb)
    return evidence


class AuditLog:
    """逐条记录 raw → normalized → PI-EV 输入（仪器验证门用）。"""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self.rows: list[dict] = []

    def record(self, *, hand_id: str, street: str, raw: dict,
               state_pot_bb: float, state_to_call_bb: float,
               hero_stack_bb: float, hand_bb: float,
               inputs: dict) -> None:
        if len(self.rows) >= self.capacity:
            return
        self.rows.append({
            "hand_id": hand_id, "street": street,
            "raw": {"pot_before_chips": raw.get("pot_before"),
                    "to_call_chips": raw.get("to_call"),
                    "hand_bb_chips": float(hand_bb)},
            "state": {"pot_bb": state_pot_bb, "to_call_bb": state_to_call_bb,
                      "hero_stack_bb": hero_stack_bb},
            "pi_ev_input": dict(inputs),
            "ratios": {
                "pot_input_over_state": (round(inputs["pot_bb"]
                                               / state_pot_bb, 9)
                                         if state_pot_bb else None),
                "stack_input_over_state": (round(inputs["hero_stack_bb"]
                                                 / hero_stack_bb, 9)
                                           if hero_stack_bb else None),
                "pot_state_over_raw_bb": (round(state_pot_bb
                                                / (float(raw.get("pot_before")
                                                         or 0.0)
                                                   / float(hand_bb)), 9)
                                          if raw.get("pot_before") else None)},
        })
