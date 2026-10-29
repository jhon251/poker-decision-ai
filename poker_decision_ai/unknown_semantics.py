"""V3F-P30-7-E — UNKNOWN 三态语义（UNKNOWN / KNOWN_ZERO / KNOWN_VALUE）。

StateAgent 既有 `float(x or 0.0)` 把 UNKNOWN 与 KNOWN_ZERO 混同。
本模块是**最小语义边界**（不重写 StateAgent）：把数值字段显式区分为
三态——供 V3F 路径上的新 consumer 使用；旧 V2 路径不变（兼容保持）。

契约：
  UNKNOWN      ≠ 0（未观测——不得使用断言）；
  KNOWN_ZERO   = 0（合法观测零——如过牌 to_call=0）；
  KNOWN_VALUE  = v（观测值）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TriState:
    status: str          # UNKNOWN | KNOWN_ZERO | KNOWN_VALUE
    value: Any = None

    def to_dict(self) -> dict:
        return {"status": self.status, "value": self.value}


def classify(value, status: str | None = None) -> TriState:
    """分类数值字段三态（status 优先；value 推断兜底——不制造值）。

    - status="RECOGNIZED" 且 value is not None → KNOWN_VALUE；
    - status="RECOGNIZED" 且 value == 0 → KNOWN_ZERO；
    - status 非 RECOGNIZED / value is None → UNKNOWN。
    """
    if status == "RECOGNIZED":
        if value is None:
            return TriState("UNKNOWN")
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return TriState("UNKNOWN")
        if fv == 0.0:
            return TriState("KNOWN_ZERO", 0.0)
        return TriState("KNOWN_VALUE", fv)
    return TriState("UNKNOWN")


def tri_to_float(t: TriState) -> float | None:
    """三态 → float：KNOWN_ZERO→0.0；KNOWN_VALUE→v；UNKNOWN→None（不转 0）。"""
    if t.status == "KNOWN_ZERO":
        return 0.0
    if t.status == "KNOWN_VALUE":
        return float(t.value)
    return None
