"""P63 — Strategy Fusion（调整账本融合层；确定性；只读聚合）。

目标：把多个候选调整来源（exploit 流、未来 solver、未来流）统一为
**可审计的 adjustment ledger**，并给出确定性融合规则：
  1. 同方向（同号）→ 代数叠加；
  2. 相反方向 → 按证据强度裁决：净量 = Σ（代数和），并裁决
     |net| 与双方强度——|net| ≥ RESOLVE_MARGIN×max(|a|,|b|) → 可靠裁决
     （取净量）；否则 → 冲突不可裁决 → 该动作不调整 + conflict 标注
     （UNKNOWN，不猜测）；
  3. 每条 entry 携带 source/动作域/confidence/provenance——来源可追溯；
  4. 数值上限：|fused| ≤ CAP_BB（防极端）。

语义：融合结果为**加性候选调整层**（bb），由 strategy_intel 叠加到
utility 上；base EV / SACA / A3-v2 / mode objective 不变。双计审计由
调用方（strategy_intel.double_counting_guard）报告 evidence 路由。
"""
from __future__ import annotations

#: 融合上限（bb；与 P62 MAX_EXPLOIT_BB 同族；contract 常数）
FUSION_CAP_BB = 0.5
#: 冲突不可裁决余量：|net| < RESOLVE_MARGIN × max(|a|,|b|) → 不调整
RESOLVE_MARGIN = 0.35
#: 参与冲突裁决的最小条目强度
STRENGTH_FLOOR = 0.05


def make_entry(source: str, action: str | None, delta_bb: float,
               confidence: float, evidence_streams: list[str] | None = None,
               provenance: str = "") -> dict:
    """构造一条 ledger entry（确定性）。

    action=None → 作用于全部动作；否则仅该动作。
    """
    return {"source": source, "action": action,
            "delta_bb": round(float(delta_bb), 6),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
            "evidence_streams": list(evidence_streams or []),
            "provenance": provenance}


def fuse_entries(entries: list[dict]) -> dict:
    """ledger 条目 → 每动作融合结果（确定性；同向叠加/反向裁决）。

    返回 {"actions": {action: {delta_bb, sources, conflicted,
    resolution}}, "conflicted_actions": [...], "n_entries": n,
    "capped": [action...]}。
    """
    by_action: dict[str, list[dict]] = {}
    for e in entries or []:
        act = e.get("action")            # None = 全动作
        key = act if act else "*"
        by_action.setdefault(key, []).append(e)

    actions: dict[str, dict] = {}
    conflicted: list[str] = []
    capped: list[str] = []

    def _fuse_group(group: list[dict], label: str) -> dict:
        pos = [e for e in group if e["delta_bb"] > 0]
        neg = [e for e in group if e["delta_bb"] < 0]
        net = round(sum(e["delta_bb"] for e in group), 6)
        sources = [e["source"] for e in group]
        conflicted_flag = False
        resolution = "stack"
        if pos and neg:
            # 反向：强度 = |delta|×confidence；取强者方向，净量按代数和
            strengths = [(abs(e["delta_bb"]) * e["confidence"], e)
                         for e in group]
            strengths.sort(key=lambda x: -x[0])
            top = strengths[0][0]
            second = strengths[1][0] if len(strengths) > 1 else 0.0
            if abs(net) < RESOLVE_MARGIN * max(top, second, 1e-9):
                conflicted_flag = True
                resolution = "unresolvable_conflict_no_adjustment"
                net = 0.0
            else:
                resolution = "resolved_by_strength"
        if abs(net) > FUSION_CAP_BB:
            net = round(max(-FUSION_CAP_BB,
                            min(FUSION_CAP_BB, net)), 6)
            capped.append(label)
        return {"delta_bb": net, "sources": sources,
                "conflicted": conflicted_flag, "resolution": resolution,
                "n_entries": len(group)}

    for label, group in sorted(by_action.items()):
        fused = _fuse_group(group, label)
        actions[label] = fused
        if fused["conflicted"]:
            conflicted.append(label)
    # 展开通配（*）到显式动作：显式优先，通配叠加
    if "*" in actions:
        star = actions.pop("*")
        for act, fused in list(actions.items()):
            merged_net = round(fused["delta_bb"] + star["delta_bb"], 6)
            merged_net = max(-FUSION_CAP_BB, min(FUSION_CAP_BB,
                                                 merged_net))
            fused["delta_bb"] = merged_net
            fused["sources"] = fused["sources"] + star["sources"]
            if star["conflicted"] and not fused["conflicted"]:
                fused["conflicted"] = True
                conflicted.append(act)
        actions["*"] = star
    return {"actions": actions, "conflicted_actions": conflicted,
            "n_entries": len(entries or []), "capped_actions": capped}


def fused_delta_for_action(fused: dict, action: str) -> tuple[float, bool]:
    """取某动作的融合调整（显式已含通配合并——fuse 阶段完成叠加）。

    显式组存在 → 其 delta 已并入通配（fuse 内合并），直接取显式；
    无显式 → 通配对该动作生效；两者皆无 → 0.0（无调整）。
    """
    acts = fused.get("actions") or {}
    explicit = acts.get(action)
    star = acts.get("*")
    conflicted = False
    if explicit is not None:
        total = explicit["delta_bb"]
        conflicted = explicit["conflicted"]
    elif star is not None:
        total = star["delta_bb"]
        conflicted = star["conflicted"]
    else:
        total = 0.0
    total = max(-FUSION_CAP_BB, min(FUSION_CAP_BB, total))
    return round(total, 6), conflicted


def exploit_to_entries(opportunity: dict) -> list[dict]:
    """P62 机会 → ledger entries（source=exploit；动作域显式）。"""
    if not opportunity.get("available"):
        return []
    aff = opportunity.get("affected_actions") or {}
    conf = float(opportunity.get("confidence") or 0.0)
    streams = list(opportunity.get("streams_used") or [])
    return [make_entry("exploit", act, bb, conf, streams,
                       provenance="p62 deviation→bounded tilt")
            for act, bb in sorted(aff.items()) if bb]


def future_solver_to_entries(candidate: dict) -> list[dict]:
    """未来真实 solver 候选 → ledger entries（source=solver；
    solver_label 纪律；仅当调用方显式提供，P35 下恒不出现）。"""
    if not candidate:
        return []
    act = candidate.get("action")
    delta = float(candidate.get("utility_bb") or 0.0)
    return [make_entry("solver", act, delta,
                       float(candidate.get("confidence") or 1.0),
                       evidence_streams=["solver"],
                       provenance="solver_label (not human ground truth)")]
