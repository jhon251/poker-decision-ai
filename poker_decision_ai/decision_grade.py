"""P52 — DecisionGradeDataset contract（决策级数据契约；纯判定；无 IO）。

目标：定义"未来能够验证真实 Poker 决策"所必须具备的字段与字段级质量
门控，使新数据源（带 hero hole cards 的 cash / MTT / PKO hand history）
可被客观评估是否达到 decision grade——而不是凭感觉宣称。

与 P47 evidence gate 同风格：
  - verdict 四值：usable / partially_usable / excluded / unknown；
  - 缺失字段不伪造（unknown/excluded 如实标注）；
  - 纯确定性：相同输入 → 相同输出；无随机、无网络、无 LLM。

决策级验证的最小字段集（DECISION_GRADE_FIELDS，13 组）：
  hand_id / player_id（身份）
  hero_hole_cards（观察者视角数据在此字段永远 excluded → 决策级不可用）
  board_by_street / pot_effective_stack / position_button
  legal_actions / action_sequence_sizes（含 observed hero action）
  opponent_identity_actions / game_mode / blind_level
  payout_icm_context（MTT 必须）/ bounty_context（PKO/KO/Mystery 必须）

数据集级裁决（dataset_verdict）：
  decision_grade      —— 全部 required 字段 usable（或仅含声明的部分可用）
  behavior_grade      —— 仅 action-level 证据可用（如 UofT：可做行为
                         校准，不能做 observed-action 决策对照）
  insufficient        —— 连 action-level 都不满足

模式声明（per game mode 的可验证范围）由 `mode_validation_scope` 给出，
UofT 的裁决通过 `classify_uoft()` 显式标记（不升级、不伪造）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---- verdict（与 P47 evidence gate 同一词汇） ----
from .domain.poker.evidence_gate import (
    ALLOWED_VERDICTS,
    EXCLUDED,
    PARTIALLY_USABLE,
    UNKNOWN,
    USABLE,
    VERDICTS,
)

# ---- 决策级字段契约（13 组；required = 决策级验证的硬前提） ----
DECISION_GRADE_FIELDS: tuple[dict, ...] = (
    {"key": "hand_id", "required": True,
     "desc": "手牌唯一标识（跨数据集可追溯）"},
    {"key": "player_id", "required": True,
     "desc": "稳定玩家身份（跨手聚合的前提）"},
    {"key": "hero_hole_cards", "required": True,
     "desc": "hero 真实底牌（observed-action 决策对照的硬前提）"},
    {"key": "board_by_street", "required": True,
     "desc": "逐街公共牌"},
    {"key": "pot_effective_stack", "required": True,
     "desc": "底池与有效栈（EV/尺寸语义）"},
    {"key": "position_button", "required": True,
     "desc": "位置与按钮（positions_for 输入）"},
    {"key": "legal_actions", "required": True,
     "desc": "决策点合法动作集（安全门/合法性行为验证前提）"},
    {"key": "action_sequence_sizes", "required": True,
     "desc": "逐街动作序列与尺寸（范围重建/条件模型输入）"},
    {"key": "observed_hero_action", "required": True,
     "desc": "hero 在决策点的真实动作（observed-action 对照的 ground truth）"},
    {"key": "opponent_identity_actions", "required": True,
     "desc": "对手身份与动作历史（剥削/画像证据）"},
    {"key": "game_mode", "required": True,
     "desc": "cash / mtt / pko / ko / mystery（路由与 objective）"},
    {"key": "blind_level", "required": False,
     "desc": "盲注级别（MTT 涨盲语义；cash stakes 即等价）"},
    {"key": "payout_icm_context", "required": "mtt",
     "desc": "payout/ICM 上下文（MTT 决策级验证必须）"},
    {"key": "bounty_context", "required": "bounty",
     "desc": "bounty 上下文（PKO/KO/Mystery 决策级验证必须）"},
)

#: 数据集级裁决
DECISION_GRADE = "decision_grade"
BEHAVIOR_GRADE = "behavior_grade"
INSUFFICIENT = "insufficient"
DATASET_VERDICTS = (DECISION_GRADE, BEHAVIOR_GRADE, INSUFFICIENT)


def classify_record(record: dict, mode: str = "cash") -> dict:
    """单条候选记录 → 字段级 verdict（确定性；缺失=unknown，不伪造）。

    record: 适配器产出的字段 dict（见 DecisionDatasetAdapter.field_spec）。
    mode: cash | mtt | pko | ko | mystery（决定条件必填字段）。
    """
    out: dict[str, dict] = {}
    for spec in DECISION_GRADE_FIELDS:
        key = spec["key"]
        req = spec["required"]
        required_here = (req is True
                         or (isinstance(req, str) and req == mode)
                         or (req == "bounty" and mode in ("pko", "ko",
                                                          "mystery")))
        v = record.get(key)
        if v is None or v == "" or v == [] or v == {}:
            out[key] = {"verdict": UNKNOWN,
                        "reason": "missing"
                        if required_here else "optional_missing"}
        elif isinstance(v, dict) and v.get("verdict"):
            out[key] = v                    # 适配器已给字段级判定 → 直通
        else:
            out[key] = {"verdict": USABLE, "reason": ""}
    return out


def _required_keys(mode: str) -> tuple[str, ...]:
    """该模式下 decision-critical 的字段集（含条件必填）。"""
    out = []
    for spec in DECISION_GRADE_FIELDS:
        req = spec["required"]
        if (req is True
                or (isinstance(req, str) and req == mode)
                or (req == "bounty" and mode in ("pko", "ko", "mystery"))):
            out.append(spec["key"])
    return tuple(out)


def dataset_verdict(field_verdicts: dict, mode: str = "cash") -> dict:
    """字段级判定 → 数据集级裁决（单条记录口径；mode 感知条件必填）。

    decision_grade：该模式全部 decision-critical 字段 usable；
    behavior_grade：身份/动作/序列可用但决策级字段有缺口（只能做行为
      校准，不能做决策对照）；
    insufficient：action-level 类字段缺失（连行为校准都不支持）。
    """
    def _v(k):
        return (field_verdicts.get(k) or {}).get("verdict")

    critical = [k for k in _required_keys(mode)
                if k not in ("blind_level",)]
    critical_ok = all(_v(k) == USABLE for k in critical)
    if critical_ok:
        return {"verdict": DECISION_GRADE,
                "reason": "all decision-critical fields usable"}
    action_ok = all(_v(k) in ALLOWED_VERDICTS
                    for k in ("hand_id", "player_id", "action_sequence_sizes"))
    if action_ok:
        missing = [k for k in critical if _v(k) not in ALLOWED_VERDICTS]
        return {"verdict": BEHAVIOR_GRADE,
                "reason": "action-level usable; decision-level missing: "
                          + ",".join(missing)}
    return {"verdict": INSUFFICIENT,
            "reason": "action-level fields missing"}


# ---------------------------------------------------------------- adapter
class DecisionDatasetAdapter:
    """未来数据源的最小接入接口（本阶段只定义契约，不实现数据源）。

    子类必须实现：
      dataset_meta() -> dict        数据集身份（name/version/license/source/
                                    known_issues——禁止隐瞒已知缺陷）
      iter_candidates(limit) -> Iterator[dict]
                                    逐条产出"字段候选 dict"（字段名 =
                                    DECISION_GRADE_FIELDS 的 key；值可为
                                    原始值或 {"verdict", "reason"} 判定）；
                                    不伪造：缺什么就不给什么。
      mode_of(record) -> str        cash | mtt | pko | ko | mystery

    评估流程（evaluate_dataset）：iter_candidates → classify_record →
    dataset_verdict → 聚合报告。聚合含各字段 usable 比例与数据集级
    裁决分布——新数据源达到 decision_grade 的比例即其决策级可用性。
    """

    def dataset_meta(self) -> dict:     # pragma: no cover - interface
        raise NotImplementedError

    def iter_candidates(self, limit: int):      # pragma: no cover
        raise NotImplementedError

    def mode_of(self, record: dict) -> str:     # pragma: no cover
        raise NotImplementedError

    def evaluate_dataset(self, limit: int = 1000) -> dict:
        """确定性评估前 limit 条候选 → 字段级/数据集级聚合报告。"""
        field_counts: dict[str, dict[str, int]] = {
            s["key"]: {v: 0 for v in VERDICTS} for s in DECISION_GRADE_FIELDS}
        level_counts = {v: 0 for v in DATASET_VERDICTS}
        n = 0
        modes: dict[str, int] = {}
        for rec in self.iter_candidates(limit):
            n += 1
            mode = self.mode_of(rec)
            modes[mode] = modes.get(mode, 0) + 1
            fv = classify_record(rec, mode)
            for k, cell in fv.items():
                field_counts[k][cell["verdict"]] = \
                    field_counts[k].get(cell["verdict"], 0) + 1
            level_counts[dataset_verdict(fv, mode)["verdict"]] += 1
        ratios = {k: {v: round(c.get(v, 0) / n, 6) if n else 0.0
                      for v in VERDICTS}
                  for k, c in field_counts.items()}
        return {"adapter": type(self).__name__,
                "dataset_meta": self.dataset_meta(),
                "n_records": n,
                "modes": modes,
                "field_usable_ratio": {k: r[USABLE] for k, r in ratios.items()},
                "field_verdict_ratio": ratios,
                "dataset_verdict_counts": level_counts,
                "dataset_verdict_ratio": {
                    v: round(level_counts[v] / n, 6) if n else 0.0
                    for v in DATASET_VERDICTS}}


# ---------------------------------------------------------------- UofT 裁决
#: UofT CPRG 已知限制（P47/P51 实测；显式带入契约——不隐瞒）
_UOFT_KNOWN_ISSUES = (
    "observer view: hero hole cards absent in 100% of sampled hands "
    "(chunk_000 full scan 100,979 hands: 0 with hole cards, P51)",
    "2009-07 Ongame accounting warnings (chip consistency)",
    "coarse position labels (EP/LP) vs positions_for vocabulary",
    "cash-only corpus: no tournament/bounty context",
)


class UofTDecisionGradeProbe(DecisionDatasetAdapter):
    """UofT CPRG 在决策级契约下的裁决（只读探针；不引入数据本身）。

    依据 P47 gate 与 P51 全量实测给出字段级判定：hero_hole_cards 与
    observed_hero_action 永远 excluded → 数据集级 = behavior_grade
    （可做行为校准，不能做 observed-action 决策验证）。
    """

    def dataset_meta(self) -> dict:
        return {"name": "uoft_cprg",
                "version": "uoft-cprg-v2-normalized-5.0",
                "license": "cc-by-4.0",
                "source": "https://zenodo.org/records/13997158",
                "known_issues": list(_UOFT_KNOWN_ISSUES)}

    def iter_candidates(self, limit: int):
        for _ in range(limit):
            yield {
                "hand_id": {"verdict": USABLE, "reason": ""},
                "player_id": {"verdict": USABLE,
                              "reason": "stable anonymous ids"},
                "hero_hole_cards": {"verdict": EXCLUDED,
                                    "reason": "observer_view_no_hole_cards"},
                "board_by_street": {"verdict": USABLE, "reason": ""},
                "pot_effective_stack": {"verdict": PARTIALLY_USABLE,
                                        "reason": "no stack_end per player"},
                "position_button": {"verdict": PARTIALLY_USABLE,
                                    "reason": "coarse EP/LP labels; "
                                              "12.7% missing"},
                "legal_actions": {"verdict": UNKNOWN,
                                  "reason": "not captured by source"},
                "action_sequence_sizes": {"verdict": USABLE, "reason": ""},
                "observed_hero_action": {"verdict": EXCLUDED,
                                         "reason": "no hero view"},
                "opponent_identity_actions": {"verdict": USABLE,
                                              "reason": ""},
                "game_mode": {"verdict": USABLE, "reason": "cash"},
                "blind_level": {"verdict": USABLE, "reason": "stakes"},
                "payout_icm_context": {"verdict": EXCLUDED,
                                       "reason": "cash-only corpus"},
                "bounty_context": {"verdict": EXCLUDED,
                                   "reason": "cash-only corpus"},
            }

    def mode_of(self, record: dict) -> str:
        return "cash"


def mode_validation_scope(mode: str) -> dict:
    """各模式的决策级可验证范围声明（contract；不随数据集变化）。"""
    scopes = {
        "cash": {
            "decision_grade_validation": True,
            "requires": ["hero_hole_cards", "observed_hero_action",
                         "legal_actions", "pot_effective_stack",
                         "position_button"],
            "objective": "chip_ev",
        },
        "mtt": {
            "decision_grade_validation": True,
            "requires": ["hero_hole_cards", "observed_hero_action",
                         "legal_actions", "payout_icm_context",
                         "blind_level"],
            "objective": "chip_ev+icm",
        },
        "pko": {
            "decision_grade_validation": True,
            "requires": ["hero_hole_cards", "observed_hero_action",
                         "bounty_context", "payout_icm_context"],
            "objective": "chip_ev+icm+bounty",
        },
        "ko": {
            "decision_grade_validation": True,
            "requires": ["hero_hole_cards", "observed_hero_action",
                         "bounty_context", "payout_icm_context"],
            "objective": "chip_ev+icm+bounty",
        },
        "mystery": {
            "decision_grade_validation": True,
            "requires": ["hero_hole_cards", "observed_hero_action",
                         "bounty_context", "payout_icm_context"],
            "objective": "chip_ev+icm+bounty",
        },
    }
    return scopes.get(mode, {"decision_grade_validation": False,
                             "requires": [], "objective": "unknown"})
