"""实验分组与数据泄漏防护（V2 Phase 4 §十·十一）。

1. Time Split：按 started_at 时间顺序切分 Train / Validation / Test。
   缺失时间戳的手牌 → 不进入任何分组（安全第一，不猜时间）。
2. Player Model 时间泄漏防护：回放/建模某时刻 T 的手牌时，只允许使用
   started_at < T 的样本（point-in-time 过滤，`as_of` 助手）。
3. Experiment Arms 接口（§十）：A=Population only / B=+Individual /
   C=+LLM / D=+Solver / E=Full。注意：本模块只定义接口与切换标志，
   各臂的执行约束由 Replay/Experiment 运行器按标志实施（当前阶段为接口准备）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXPERIMENT_ARMS = {
    "A": {"id": "A", "name": "population_only",
          "flags": {"use_population": True, "use_individual": False,
                    "use_llm": False, "use_solver": False}},
    "B": {"id": "B", "name": "population_plus_individual",
          "flags": {"use_population": True, "use_individual": True,
                    "use_llm": False, "use_solver": False}},
    "C": {"id": "C", "name": "population_individual_llm",
          "flags": {"use_population": True, "use_individual": True,
                    "use_llm": True, "use_solver": False}},
    "D": {"id": "D", "name": "population_individual_solver",
          "flags": {"use_population": True, "use_individual": True,
                    "use_llm": False, "use_solver": True}},
    "E": {"id": "E", "name": "full_agent",
          "flags": {"use_population": True, "use_individual": True,
                    "use_llm": True, "use_solver": True}},
}


@dataclass
class TimeSplit:
    train: list = field(default_factory=list)
    validation: list = field(default_factory=list)
    test: list = field(default_factory=list)
    excluded_no_timestamp: list = field(default_factory=list)

    def counts(self) -> dict:
        return {"train": len(self.train), "validation": len(self.validation),
                "test": len(self.test),
                "excluded_no_timestamp": len(self.excluded_no_timestamp)}


def time_split(hands: list[dict], *, train_frac: float = 0.7,
               val_frac: float = 0.15) -> TimeSplit:
    """按 started_at 升序切分（时间在前 → train；时间在后 → test）。

    缺失/非法 started_at → excluded_no_timestamp（禁止用未来信息，
    没有时间戳就无法安全切分，宁可排除也不混入）。
    """
    ordered: list[tuple[str, dict]] = []
    excluded: list[dict] = []
    for h in hands:
        ts = h.get("started_at") if isinstance(h, dict) else None
        if isinstance(ts, str) and ts:
            ordered.append((ts, h))
        else:
            excluded.append(h)
    ordered.sort(key=lambda x: x[0])
    n = len(ordered)
    train_n = int(n * train_frac)
    val_n = int(n * val_frac)
    return TimeSplit(
        train=[h for _, h in ordered[:train_n]],
        validation=[h for _, h in ordered[train_n:train_n + val_n]],
        test=[h for _, h in ordered[train_n + val_n:]],
        excluded_no_timestamp=excluded,
    )


def as_of(hands: list[dict], timestamp: str) -> list[dict]:
    """Point-in-time 过滤：只返回 started_at < timestamp 的手牌。

    回放时刻 T 的决策时用它取"当时可见"的历史——防止用未来数据训练过去模型
    （Player Model 时间泄漏防护的核心助手）。
    """
    out = []
    for h in hands:
        ts = h.get("started_at") if isinstance(h, dict) else None
        if isinstance(ts, str) and ts and ts < timestamp:
            out.append(h)
    return out


def arm_spec(arm: str) -> dict:
    if arm.upper() not in EXPERIMENT_ARMS:
        raise ValueError(f"unknown arm '{arm}' (expected A|B|C|D|E)")
    return dict(EXPERIMENT_ARMS[arm.upper()])


def arm_request(request: dict, arm: str) -> dict:
    """把臂标志写入请求（实验运行器按标志执行约束）。"""
    spec = arm_spec(arm)
    out = dict(request)
    out["arm"] = spec["id"]
    out["arm_name"] = spec["name"]
    out["arm_flags"] = dict(spec["flags"])
    return out
