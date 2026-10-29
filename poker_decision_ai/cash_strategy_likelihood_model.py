"""V3H-6 — CASH 声明式 Action-Likelihood 模型（显式参数 + 严格校验）。

**先审计（授权要求）：既有实现到什么程度**
--------------------------------------------
审计发现：`cash_range_model` **已经**在做序贯动作加权，且**已声明**一张扁平表
（`ACTION_WEIGHT = {fold:0.25, check:0.75, call:1.0, bet:1.5, raise:2.0, allin:3.0}`），
按观测动作**逐条**应用并记录 `applied_factors`（实测：`raise`+`bet` → 因子 `[2.0, 1.5]`）。

**因此本层不重复实现"动作加权"本身**，只补齐它**缺的东西**：

| 既有 `ACTION_WEIGHT` | 本层补齐 |
|---|---|
| 扁平：仅按 action | **多维**：action × street × position × size_bucket × profile |
| 无 provenance/source | **每项**带 `source` / `evidence_level` / `provenance` |
| 无 uncertainty | 每项带 `uncertainty`（**不发明**数值，只登记"未量化"或调用方给出的界） |
| 无校验 | **严格校验**：非负、有界、缺失不静默补全、冲突不静默覆盖 |
| 无 prior_only 语义 | 未提供 → `prior_only=true`（显式） |

**更正记录（诚实性）**：V3H-4 的 `cash_strategy_dynamic_belief` 文档曾表述
"已声明权重的应用由 range 模型完成，本层不重复施加"；就"不重复施加"这点是对的，
但 V3H-5 belief-eval 的措辞 "NO action likelihood was applied" 在
**range_model 已按 ACTION_WEIGHT 施加**这一点上**不够精确**。本层在文档与输出中
**显式更正**：prior-only 指的是"**没有额外的、调用方声明的 likelihood 模型**"，
而**不是**"posterior 等于原始先验"。

硬约束
------
1. **只读 + 确定性**：不 import `random`；无 IO；同输入 → 同输出（含错误列表顺序）。
2. **不发明频率**：本层**不内置**任何"真实玩家频率"；未提供参数即
   `prior_only=true` 且 `applied=false`。
3. **严格校验**：
   * `multiplier` / `likelihood` **非负**（负数 → 拒绝并记为 `invalid`）；
   * **有界**（`0 < w <= MAX_MULTIPLIER`；`0` 合法表示"该组合不可能"，但需显式）；
   * **缺失不静默补全**：未命中任何规则 → 该动作 `unmatched`，**不使用默认值**；
   * **冲突不静默覆盖**：同一 `(action, street, size_bucket, profile)` 出现两个
     不同乘数 → 记为 `conflict`，该键**不产出**乘数（由调用方解决）。
4. **CASH-only**：本层不接受 mtt/pko 等模式参数；模式门禁由上层 state 判定。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_LIKELIHOOD_MODEL_VERSION = "v3h6-cash-strategy-likelihood-model-v1"

#: 容许的乘数范围（下界不含 0：0 需显式 `zero_means_impossible=True`）
MIN_MULTIPLIER = 0.0
MAX_MULTIPLIER = 100.0

#: 证据等级枚举（授权要求显式区分）
EVIDENCE_LEVELS: tuple[str, ...] = (
    "DECLARED_MODEL",     # 项目声明的模型参数（**非**真实玩家数据）
    "RESEARCH_ONLY",      # 研究性假设
    "UNAVAILABLE",        # 无法获得
)

#: 合法动作（与既有动作语法一致；本层不新增动作语义）
KNOWN_ACTIONS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise", "all_in")

#: 合法 street
KNOWN_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")

#: size bucket 名（由调用方给出；本层不自动分桶）
KNOWN_SIZE_BUCKETS: tuple[str, ...] = (
    "no_investment", "small", "medium", "large", "allin", "unknown",
)

#: 参数项的必需字段（缺一即视为非法项）
REQUIRED_FIELDS: tuple[str, ...] = (
    "action", "street", "multiplier", "source", "evidence_level",
    "provenance", "uncertainty",
)

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_real_player_frequency": False,
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "wired_to_production": False,
}

#: 输出中必须带 provenance 的字段说明（供审计与测试断言）
SCHEMA_FIELDS: tuple[str, ...] = (
    "action", "street", "size_bucket", "profile", "multiplier", "source",
    "evidence_level", "provenance", "uncertainty", "prior_only", "seed",
)


def _norm_dim(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip():
        return v.strip().lower()
    return None


def validate_parameters(params: Any) -> dict:
    """校验声明式 likelihood 参数（**严格；不静默补全、不静默覆盖**）。

    Parameters
    ----------
    params : list[dict] | None
        每项字段见 `SCHEMA_FIELDS`。``None`` / ``[]`` → `prior_only=True`。

    Returns
    -------
    dict with keys: present, prior_only, valid[], invalid[], conflicts[],
    unmatched_keys[], unknown[], applied
    """
    if params is None:
        return {
            "present": True, "prior_only": True, "valid": [], "invalid": [],
            "conflicts": [], "unknown": ["no_parameters_supplied"], "applied": False,
        }
    if not isinstance(params, list):
        return {
            "present": False, "prior_only": True, "valid": [], "invalid": [],
            "conflicts": [], "unknown": ["parameters_not_a_list"], "applied": False,
            "reason": "parameters_not_a_list",
        }

    valid: list[dict] = []
    invalid: list[dict] = []
    conflicts: list[dict] = []
    seen: dict[tuple, float] = {}

    for i, item in enumerate(params):
        if not isinstance(item, dict):
            invalid.append({"index": i, "reason": "item_not_a_dict"})
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            invalid.append({"index": i, "reason": "missing_required_fields",
                            "missing": missing})
            continue

        action = _norm_dim(item.get("action"))
        street = _norm_dim(item.get("street"))
        if action not in KNOWN_ACTIONS:
            invalid.append({"index": i, "reason": f"unknown_action:{item.get('action')!r}"})
            continue
        if street not in KNOWN_STREETS:
            invalid.append({"index": i, "reason": f"unknown_street:{item.get('street')!r}"})
            continue

        size_bucket = _norm_dim(item.get("size_bucket"))
        if size_bucket is not None and size_bucket not in KNOWN_SIZE_BUCKETS:
            invalid.append({"index": i,
                            "reason": f"unknown_size_bucket:{item.get('size_bucket')!r}"})
            continue
        profile = _norm_dim(item.get("profile"))
        if profile is not None and not isinstance(item.get("profile"), str):
            invalid.append({"index": i, "reason": "profile_not_a_string"})
            continue

        mult = item.get("multiplier")
        if isinstance(mult, bool) or not isinstance(mult, (int, float)):
            invalid.append({"index": i, "reason": "multiplier_not_numeric"})
            continue
        mult = float(mult)
        if mult < MIN_MULTIPLIER:
            invalid.append({"index": i, "reason": "multiplier_negative",
                            "value": mult})
            continue
        if mult > MAX_MULTIPLIER:
            invalid.append({"index": i, "reason": "multiplier_out_of_range",
                            "value": mult, "max": MAX_MULTIPLIER})
            continue
        if mult == 0.0 and item.get("zero_means_impossible") is not True:
            invalid.append({"index": i,
                            "reason": "zero_multiplier_requires_explicit_flag"})
            continue

        level = item.get("evidence_level")
        if level not in EVIDENCE_LEVELS:
            invalid.append({"index": i,
                            "reason": f"unknown_evidence_level:{level!r}"})
            continue
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            invalid.append({"index": i, "reason": "source_empty"})
            continue
        if not isinstance(item.get("provenance"), dict):
            invalid.append({"index": i, "reason": "provenance_not_a_dict"})
            continue
        uncertainty = item.get("uncertainty")
        if not isinstance(uncertainty, dict):
            invalid.append({"index": i, "reason": "uncertainty_not_a_dict"})
            continue

        key = (action, street, size_bucket, profile)
        if key in seen:
            if abs(seen[key] - mult) > 1e-12:
                conflicts.append({
                    "key": list(key), "existing": seen[key], "incoming": mult,
                    "reason": "conflicting_multiplier_for_same_key",
                })
            else:
                conflicts.append({
                    "key": list(key), "existing": seen[key], "incoming": mult,
                    "reason": "duplicate_identical_entry",
                })
            continue
        seen[key] = mult
        valid.append({
            "action": action, "street": street, "size_bucket": size_bucket,
            "profile": profile, "multiplier": round(mult, 6),
            "source": item["source"], "evidence_level": level,
            "provenance": dict(item["provenance"]),
            "uncertainty": dict(uncertainty),
            "prior_only": False,
            "seed": item.get("seed"),
        })

    # 冲突键必须从 valid 中剔除（不静默覆盖、也不静默择一）
    conflict_keys = {tuple(c["key"]) for c in conflicts}
    if conflict_keys:
        valid = [v for v in valid
                 if (v["action"], v["street"], v["size_bucket"],
                     v["profile"]) not in conflict_keys]

    unknown: list[str] = []
    if invalid:
        unknown.append(f"invalid_parameters:{len(invalid)}")
    if conflicts:
        unknown.append(f"conflicting_parameters:{len(conflicts)}")

    applied = bool(valid)
    return {
        "present": True,
        "prior_only": not applied,
        "valid": valid,
        "invalid": invalid,
        "conflicts": conflicts,
        "unknown": unknown,
        "applied": applied,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_conflicts": len(conflicts),
        "bounds": {"min": MIN_MULTIPLIER, "max": MAX_MULTIPLIER},
    }


def resolve_multiplier(valid: list[dict], *, action: str, street: str,
                       size_bucket: str | None = None,
                       profile: str | None = None) -> dict:
    """按**最具体优先**解析单个动作的乘数（无命中 → `unmatched`，**不默认**）。

    优先级：profile+size_bucket → profile → size_bucket → 仅 action+street。
    """
    a, s = _norm_dim(action), _norm_dim(street)
    sb, pf = _norm_dim(size_bucket), _norm_dim(profile)

    def _bucket_ok(r: dict) -> bool:
        """规则的 bucket 必须与观测 bucket **同时**已知且相等。

        观测 bucket 未知（None）时，**不得**命中任何指定了 bucket 的规则
        （否则等于用未知信息做匹配 = 猜）。
        """
        rb = r.get("size_bucket")
        if rb is None:
            return True                      # 规则不限 bucket
        return sb is not None and rb == sb

    def _profile_ok(r: dict, require: bool) -> bool:
        rp = r.get("profile")
        if require:
            return rp is not None and rp == pf
        return rp is None

    tiers = [
        ("profile+size_bucket",
         lambda r: _profile_ok(r, True) and _bucket_ok(r)
         and r.get("size_bucket") is not None),
        ("profile", lambda r: _profile_ok(r, True)),
        ("size_bucket", lambda r: _profile_ok(r, False)
         and r.get("size_bucket") is not None and _bucket_ok(r)),
        ("action+street", lambda r: _profile_ok(r, False)
         and r.get("size_bucket") is None),
    ]
    for tier, pred in tiers:
        for r in valid:
            if r["action"] != a or r["street"] != s:
                continue
            if pred(r):
                return {"matched": True, "tier": tier, "multiplier": r["multiplier"],
                        "source": r["source"], "evidence_level": r["evidence_level"],
                        "provenance": r["provenance"],
                        "uncertainty": r["uncertainty"]}
    return {"matched": False, "tier": None, "multiplier": None,
            "reason": "unmatched_no_default_applied"}


def action_likelihood_model(params: Any = None) -> dict:
    """声明式 action-likelihood 模型入口（可审计、可复算）。"""
    v = validate_parameters(params)
    return {
        "present": v["present"],
        "version": CASH_STRATEGY_LIKELIHOOD_MODEL_VERSION,
        "schema_fields": list(SCHEMA_FIELDS),
        "evidence_levels": list(EVIDENCE_LEVELS),
        "bounds": v.get("bounds"),
        "prior_only": v["prior_only"],
        "applied": v["applied"],
        "parameters": v["valid"],
        "invalid": v["invalid"],
        "conflicts": v["conflicts"],
        "unknown": v["unknown"],
        "evidence_grade": ("MODEL_INTERNAL" if v["applied"] else "NONE"),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("参数是**声明式模型**，不是真实玩家频率；"
                 "未提供参数时 prior_only=true 且不做任何加权"),
    }
