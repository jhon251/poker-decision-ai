"""Phase 8B Step 3：两臂策略（PAM 注入）+ 预注册合成 recommendation provider。

- Baseline A = PAMPolicy（PAM 表查询；8A 冻结 BaselineArm 采样逻辑复用）
- Exploit Arm E(k,ε) = 同 PAMPolicy + 冻结乘子（8A ExploitArm 复用）+
  **合成 recommendation provider**（预注册参数；真实 7C 链路禁止接入 Pluribus
  replay——无时间顺序，任何伪造排序/相邻文件推断 history 都硬失败）
- 两臂共享同 seed 双 RNG 流；exploit signal 关闭/中性乘子 → 逐点一致（硬测试）
- provider 使用独立于臂采样流的 per-decision 种子 RNG（不消耗臂流，保对齐）

输出：无（纯函数库）；契约由 tests/test_phase8b_*.py 断言。
"""
from __future__ import annotations

import random
from pathlib import Path

import sys

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase8a_replay import (  # noqa: E402
    ACTION_MAPPINGS, ALLOWED_OBS_KEYS, BaselineArm, ExploitArm)

ACTIONS = ("fold", "check", "call", "bet", "raise")
SIZE_CENTERS = {"s1": 0.25, "s2": 0.5, "s3": 0.8, "s4": 1.25}   # ×pot（预注册）

# ---- 预注册合成 recommendation provider 参数（全部冻结常量）----
SYNTH_REC = {
    "label": "synthetic",
    "mode": "on",
    "steal_spot": ("vpip", "under", +1),        # 无人入池 + hero BTN/LP
    "open_spot": ("fold_to_3bet", "over_or_under", 0),  # villain open + hero facing
    "confidence_lo": 0.5, "confidence_hi": 0.95,
    "deviation_magnitude": 0.1,
}


class SyntheticRecommendationProvider:
    """合成推荐（仅机制验证用）。参数全部预注册；模式 off → 恒 None。

    独立 per-decision 种子 RNG：不消耗臂采样流（两臂流对齐契约不受影响）。
    真实 7C 链禁止接入 Pluribus replay（8B Audit §9-10 硬规则）。
    """

    def __init__(self, hand_id: str, mode: str = "on") -> None:
        assert mode in ("on", "off")
        self.hand_id = hand_id
        self.mode = mode
        self._i = 0

    def _rng(self) -> random.Random:
        self._i += 1
        return random.Random(f"8b-syn-{self.hand_id}-{self._i}")

    def __call__(self, obs: dict) -> dict | None:
        if self.mode != "on":
            return None
        assert set(obs) <= set(ALLOWED_OBS_KEYS), "provider 只接收白名单观测"
        rng = self._rng()
        street = obs.get("street")
        pos = obs.get("hero_position")
        to_call = obs.get("to_call") or 0.0
        history = obs.get("action_history") or []
        if street == "preflop" and to_call <= 1e-9:
            voluntary = any(r.get("action") in ("call", "bet", "raise")
                            for r in history if r.get("street") == "preflop")
            if not voluntary and pos in ("BTN", "LP"):
                feature, direction, sign = SYNTH_REC["steal_spot"]
                mapping = ACTION_MAPPINGS[(feature, direction)]
                return self._rec(mapping, sign)
        if street == "preflop" and to_call > 1e-9:
            # villain open：方向 over/under 对半（fold_to_3bet 映射）
            over = rng.random() < 0.5
            direction = "over" if over else "under"
            feature = "fold_to_3bet"
            mapping = ACTION_MAPPINGS[(feature, direction)]
            sign = mapping["sign"]
            return self._rec(mapping, sign)
        return None

    def _rec(self, mapping: dict, sign: int) -> dict:
        rng = self._rng()
        confidence = rng.uniform(SYNTH_REC["confidence_lo"],
                                 SYNTH_REC["confidence_hi"])
        return {
            "eligible": True,
            "label": SYNTH_REC["label"],
            "mapping": {"action_class": mapping["action_class"], "sign": sign},
            "confidence": round(confidence, 4),
            "deviation_magnitude": SYNTH_REC["deviation_magnitude"],
        }


class PAMPolicy:
    """PAM 表 → 观测 → 合法行动概率（查询时阶梯回退，与拟合阶梯一致）。"""

    def __init__(self, pam: dict) -> None:
        self.pam = pam
        self._n_fallback = 0
        self._n_queries = 0

    def cell(self, obs: dict) -> str:
        street = obs.get("street")
        position = obs.get("hero_position") or "NA"
        to_call = obs.get("to_call") or 0.0
        pot = max(float(obs.get("pot") or 0.0), 1e-9)
        history = obs.get("action_history") or []
        if street == "preflop":
            voluntary = any(r.get("action") in ("call", "bet", "raise")
                            for r in history if r.get("street") == "preflop")
            if not voluntary:
                spot = "unopened"
            else:
                spot = "facing_bet" if to_call > 1e-9 else "checked_to"
        else:
            spot = "facing_bet" if to_call > 1e-9 else "checked_to"
        bucket = "b0"
        if spot == "facing_bet":
            r = to_call / pot
            bucket = "b1" if r <= 0.5 else ("b2" if r <= 1.5 else "b3")
        return f"{street}|{position}|{spot}|{bucket}"

    def probs_for(self, cell: str, legal: list[str]) -> dict[str, float]:
        self._n_queries += 1
        probs = self.pam.get("action_probs", {})
        src = probs.get(cell)
        if src is None:
            street, _, spot, _ = cell.split("|")
            src = probs.get(f"{street}|NA|{spot}|b0") or probs.get(
                f"{street}|NA|{spot}|b1")
            src = src or self._street_probs(street)
            self._n_fallback += 1
        out = {a: src.get(a, 0.0) for a in legal}
        total = sum(out.values())
        if total <= 0:
            out = {a: 1.0 / len(legal) for a in legal}   # 均匀回退（预注册）
        else:
            out = {a: v / total for a, v in out.items()}
        return out

    def _street_probs(self, street: str) -> dict[str, float]:
        probs = self.pam.get("action_probs", {})
        cand = [v for k, v in probs.items() if k.startswith(street + "|")]
        if not cand:
            return {a: 1.0 / len(ACTIONS) for a in ACTIONS}
        agg = {a: sum(c.get(a, 0.0) for c in cand) for a in ACTIONS}
        total = sum(agg.values()) or 1.0
        return {a: v / total for a, v in agg.items()}

    def __call__(self, obs: dict) -> dict[str, float]:
        legal = [a for a in ACTIONS if a in (obs.get("available_actions") or [])]
        if not legal:
            legal = ["fold"]
        return self.probs_for(self.cell(obs), legal)


def make_arms(pam: dict, hand_id: str, seed: int, k: float, eps: float,
              rec_mode: str = "on"):
    """构造两臂：同 seed 双 RNG 流；唯一差异 = exploit 乘子（+合成 rec）。"""
    rng_b = random.Random(seed)
    rng_e = random.Random(seed)
    baseline = BaselineArm(rng_b, policy=PAMPolicy(pam))
    provider = SyntheticRecommendationProvider(hand_id, mode=rec_mode)
    exploit = ExploitArm(rng_e, k=k, eps=eps, recommendation_provider=provider,
                         policy=PAMPolicy(pam))
    return baseline, exploit
