"""P54 — Pluribus 决策级数据适配与验证 harness（只读；确定性）。

数据源（P53 审计确认）：UofT CPRG raw parquet 的 pluribus 子集——
10,000 手 6-max cash-style NLHE（5 人类职业玩家 + 1 Pluribus AI 座位，
50/100 盲注，100bb 起始栈，cc-by-4.0，本地 C:/PokerDataset/UofT_CPRG）。

纪律（P54 授权边界）：
  - 只输出**人类座位**的决策级记录；Pluribus AI 座位动作 = solver_label，
    完全排除出 human-action validation（数量显式报告）；
  - legal_actions 由完整下注状态（栈/盲注/动作）确定性推导动作类型集
    （fold/check/call/bet/raise；allin 为尺寸上限）；无法严格重建的
    决策点 → UNKNOWN，不猜测；
  - 不修改任何策略数学（P41–P50 冻结）；不做 MTT/PKO；不宣称盈利；
  - 确定性：固定排序、固定分割、固定 MC 种子；同输入 → 同输出。
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .domain.poker import evidence_gate as eg

#: 数据集身份（P53 审计带入）
PLURIBUS_DATASET_META = {
    "name": "uoft-cprg::pluribus-human-seats",
    "license": "cc-by-4.0",
    "source": "https://zenodo.org/records/13997158 (raw parquet)",
    "view": "player_view_research_match",
    "ai_seat": "Pluribus",
    "ground_truth_claim": ("human seats = real_observed_action; "
                           "Pluribus seat = solver_label (excluded)"),
    "caveats": ["research match conditions (not natural online play)",
                "single blind level 50/100", "100bb starting stacks",
                "10k hands"],
}

PARQUET_DIR = Path("C:/PokerDataset/UofT_CPRG/data")
AI_SEAT = "Pluribus"

# ---- PHH 动作语法（与 scripts/analyze_uoft_cprg.py 冻结语义一致） ----
# 'd dh pN XXXX'  发牌；'d db XXX' 公共牌（3/1/1 张）
# 'pN f'          fold
# 'pN cc'         call（to_call>0）/ check（to_call==0）
# 'pN cbr X'      本街总投入至 X（call/bet/raise 合一）
# 'pN sm'         show/muck（展示动作——不影响下注状态；跳过）
# 实测全部动作码（前 3000 手）：d / f / cbr / cc / sm


def iter_pluribus_rows(parquet_dir: Path | str = PARQUET_DIR,
                       limit: int = 0) -> Iterator[dict]:
    """只读遍历 raw parquet 的 pluribus 行（确定性文件序；0=全部）。"""
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
    except ImportError as e:                       # 环境缺 pyarrow → 诚实失败
        raise RuntimeError("pyarrow required for pluribus parquet access") from e
    files = sorted(glob.glob(str(Path(parquet_dir) / "train-*.parquet")))
    n = 0
    cols = ["source_file", "hand", "actions", "players",
            "blinds_or_straddles", "starting_stacks", "variant",
            "small_bet", "big_bet", "min_bet"]
    for fp in files:
        t = pq.read_table(fp, columns=cols)
        starts = pc.starts_with(t.column("source_file"), "pluribus")
        sl = starts.to_pylist()
        idxs = [i for i, b in enumerate(sl) if b]
        for i in idxs:
            if limit and n >= limit:
                return
            yield {c: t.column(c)[i].as_py() for c in cols}
            n += 1


def parse_hand_row(row: dict) -> dict | None:
    """pluribus phh 行 → 结构化手牌（确定性；不完整 → None）。

    返回 {hand_id, sb, bb, stacks, players, hole_cards, board_by_street,
    streets: [{street, actions: [{player, action, amount, to, all_in}]}]}。
    """
    players = [str(p) for p in (row.get("players") or [])]
    n = len(players)
    if n != 6 or len(set(players)) != n:
        return None
    stacks = [float(x) for x in (row.get("starting_stacks") or [])]
    blinds = [float(x) for x in (row.get("blinds_or_straddles") or [])]
    if len(stacks) != n or len(blinds) != n or any(s <= 0 for s in stacks):
        return None
    if row.get("variant") != "NT":
        return None                                 # 仅 NLHE

    hole: dict[str, list[str]] = {}
    board_by_street: dict[str, list[str]] = {}
    streets: list[dict] = []
    cur = "preflop"
    cur_acts: list[dict] = []
    committed = [0.0] * n                           # 本街投入（cbr X=本街总投入）
    invested = list(blinds)                         # 跨街累计（=底池口径）
    level = max(blinds) if blinds else 0.0
    aggression_n = 0                                # 本街激进行为数
    sb = min(b for b in blinds if b > 0) if any(blinds) else 0.0
    bb = max(blinds) if blinds else 0.0
    bb = max(bb, 1.0)

    def _flush():
        nonlocal cur_acts
        if cur_acts:
            streets.append({"street": cur, "actions": cur_acts})
            cur_acts = []

    for tok in row.get("actions") or []:
        parts = str(tok).strip().split()
        if not parts:
            continue
        if parts[0] == "d":
            if len(parts) >= 3 and parts[1] == "dh":
                pidx = int(parts[2][1:]) - 1
                cards = [parts[3][j:j + 2] for j in range(0, len(parts[3]), 2)]
                hole[players[pidx]] = cards
            elif len(parts) >= 3 and parts[1] == "db":
                cards = [parts[2][j:j + 2] for j in range(0, len(parts[2]), 2)]
                _flush()
                if len(cards) == 3:
                    cur = "flop"
                    board_by_street[cur] = cards
                elif len(cards) == 1:
                    cur = "turn" if "flop" in board_by_street and \
                        "turn" not in board_by_street else "river"
                    board_by_street[cur] = cards
                aggression_n = 0
                level = 0.0
                committed = [0.0] * n               # 本街投入按街重置
            continue
        if not parts[0].startswith("p"):
            continue
        try:
            pidx = int(parts[0][1:]) - 1
        except ValueError:
            return None
        if not 0 <= pidx < n:
            return None
        pid = players[pidx]
        code = parts[1] if len(parts) > 1 else ""
        to_call = max(0.0, level - committed[pidx])
        remaining = stacks[pidx] - invested[pidx]
        if code == "f":
            cur_acts.append({"player": pid, "action": "fold", "amount": 0.0,
                             "to": None, "all_in": False,
                             "to_call": round(to_call, 4),
                             "pot_before": round(sum(invested), 4),
                             "remaining_stack": round(remaining, 4)})
        elif code == "cc":
            if to_call > 1e-9:
                inc = min(to_call, remaining)
                allin = remaining <= to_call + 1e-9
                cur_acts.append({"player": pid, "action": "call",
                                 "amount": round(inc, 4),
                                 "to": round(committed[pidx] + inc, 4),
                                 "all_in": allin, "to_call": round(to_call, 4),
                                 "pot_before": round(sum(invested), 4),
                                 "remaining_stack": round(remaining, 4)})
                invested[pidx] += inc
                committed[pidx] += inc
                level = max(level, committed[pidx])
            else:
                cur_acts.append({"player": pid, "action": "check",
                                 "amount": 0.0, "to": None, "all_in": False,
                                 "to_call": 0.0,
                                 "pot_before": round(sum(invested), 4),
                                 "remaining_stack": round(remaining, 4)})
        elif code == "cbr" and len(parts) > 2:
            try:
                x = float(parts[2])
            except ValueError:
                return None
            inc = x - committed[pidx]
            if inc <= 1e-9:
                cur_acts.append({"player": pid, "action": "check",
                                 "amount": 0.0, "to": None, "all_in": False,
                                 "to_call": 0.0,
                                 "pot_before": round(sum(invested), 4),
                                 "remaining_stack": round(remaining, 4)})
                continue
            allin = (committed[pidx] + inc) >= stacks[pidx] - 1e-9
            act = "bet" if level <= 1e-9 else "raise"
            cur_acts.append({"player": pid, "action": act, "amount": round(inc, 4),
                             "to": round(x, 4), "all_in": allin,
                             "to_call": round(to_call, 4),
                             "pot_before": round(sum(invested), 4),
                             "remaining_stack": round(remaining, 4)})
            invested[pidx] += inc
            committed[pidx] = x
            level = max(level, x)
            aggression_n += 1
        elif code == "sm":
            continue                                # show/muck：展示，不改状态
        else:
            return None                             # 未知动作码 → 不猜测
    _flush()

    if len(hole) != n or not streets:
        return None
    # 6-max 标准旋转：p1=SB, p2=BB → BTN = p6（seat 6）
    return {"hand_id": f"pluribus|{row.get('source_file')}|{row.get('hand')}",
            "sb": sb, "bb": bb, "stacks": stacks, "blinds": blinds,
            "players": players, "hole_cards": hole,
            "board_by_street": board_by_street, "streets": streets}


# ---- legal actions（动作类型集；完整状态严格推导） ----
def legal_action_types(to_call: float, remaining: float) -> list[str] | None:
    """决策点合法动作**类型**集（确定性；严格来自下注状态）。

    fold 恒合法（弃牌不需筹码）；to_call==0 → check 否则 call（不足则
    allin call 仍属 call）；bet/raise 合法 iff 剩余栈 > to_call（allin
    为尺寸上限，min-raise 尺寸细节不在类型集内——诚实声明）。
    栈耗尽（remaining<=0）→ 无任何合法动作（不应出现）。
    """
    if remaining <= 0:
        return None
    out = ["fold"]
    out.append("check" if to_call <= 1e-9 else "call")
    if remaining > to_call + 1e-9:
        out.extend(["bet"] if to_call <= 1e-9 else ["raise"])
    return out


# ---- 决策点抽取（人类座位 only） ----
def extract_decision_points(parsed: dict) -> list[dict]:
    """结构化手牌 → 人类座位决策点（每点含严格推导的合法动作集）。

    每个决策点：{player, street, to_call_bb, pot_bb, remaining_stack_bb,
    position, board, hole_cards, legal_actions, observed_action, ...}。
    AI 座位不出现在 player 维度（其动作只作为对手历史存在于 state 重建）。
    """
    bb = parsed["bb"] or 1.0
    n = len(parsed["players"])
    btn_seat = n                                    # p1=SB,p2=BB → BTN=pN
    pos_map = _positions_for_seats(n, btn_seat)
    seat_of = {p: i + 1 for i, p in enumerate(parsed["players"])}
    boards = parsed["board_by_street"]
    _board_cache: dict[str, list[str]] = {}

    def _board_for(street: str) -> list[str]:
        if street not in _board_cache:
            order = ("flop", "turn", "river")
            out: list[str] = []
            for s in order[:order.index(street)] if street in order else []:
                out.extend(boards.get(s, []))
            _board_cache[street] = out
        return _board_cache[street]

    out = []
    for st in parsed["streets"]:
        street = st["street"]
        board = [] if street == "preflop" else _board_for(street)
        for a in st["actions"]:
            pid = a["player"]
            if pid == AI_SEAT:
                continue                            # AI 座位：不出决策点
            la = legal_action_types(a["to_call"], a["remaining_stack"])
            if la is None:
                continue
            obs_type = a["action"]
            if obs_type == "call" and a["to_call"] <= 1e-9:
                obs_type = "check"
            out.append({
                "player": pid, "street": street,
                "position": str(pos_map.get(seat_of.get(pid)) or ""),
                "to_call_bb": round(a["to_call"] / bb, 6),
                "pot_bb": round(a["pot_before"] / bb, 6),
                "remaining_stack_bb": round(a["remaining_stack"] / bb, 6),
                "hole_cards": list(parsed["hole_cards"].get(pid) or []),
                "board": [str(x) for x in board],
                "legal_actions": la,
                "observed_action": obs_type,
                "observed_size_to_bb": (round(float(a.get("to") or 0.0) / bb, 6)
                                        if a.get("to") else None),
                "all_in": bool(a.get("all_in")),
                "hand_id": parsed["hand_id"],
                "bb": bb,
            })
    return out


def _positions_for_seats(n: int, button_seat: int) -> dict[int, str]:
    """冻结 positions_for 的薄包装（不修改；seat → 标签）。"""
    from .domain.poker.models import positions_for
    return positions_for([{"seat": i} for i in range(1, n + 1)], button_seat)


# ---- P52 adapter 包装 ----
class PluribusDecisionAdapter:
    """P52 DecisionDatasetAdapter 实现（human seats only；确定性）。

    field 语义（与 P53 审计映射一致）：
      pot_effective_stack: usable（栈完整）；
      legal_actions: usable（动作类型集严格推导；min-raise 尺寸细节声明）；
      observed_hero_action: usable（人类座位真实动作序列）。
    """

    def __init__(self, limit: int = 0, parquet_dir: Path | str = PARQUET_DIR):
        self.limit = limit
        self.parquet_dir = parquet_dir
        self.ai_seat_excluded = 0
        self.hands_parsed = 0
        self.hands_skipped = 0

    def dataset_meta(self) -> dict:
        return dict(PLURIBUS_DATASET_META)

    def mode_of(self, record: dict) -> str:
        return "cash"

    def iter_records(self) -> Iterator[dict]:
        for row in iter_pluribus_rows(self.parquet_dir, self.limit):
            parsed = parse_hand_row(row)
            if parsed is None:
                self.hands_skipped += 1
                continue
            self.hands_parsed += 1
            humans = [p for p in parsed["players"] if p != AI_SEAT]
            self.ai_seat_excluded += 1                 # 每手排除 1 个 AI 座位
            dps = extract_decision_points(parsed)
            for pid in humans:
                # hero_hole_cards / observed_hero_action 均为该人类座位
                hero_dps = [d for d in dps if d["player"] == pid]
                # preflop 即结束的手：board 真实为空——显式表达而非缺失
                board_field = (parsed["board_by_street"]
                               or {"preflop_only": True})
                yield {
                    "hand_id": parsed["hand_id"],
                    "player_id": pid,
                    "hero_hole_cards": parsed["hole_cards"].get(pid) or [],
                    "board_by_street": board_field,
                    "pot_effective_stack": {
                        "starting_stack_bb": round(parsed["stacks"][0]
                                                   / parsed["bb"], 4),
                        "bb": parsed["bb"]},
                    "position_button": {
                        "button_seat": len(parsed["players"]),
                        "position": next((d["position"] for d in hero_dps
                                          if d["position"]), "")},
                    "legal_actions": {"verdict": "usable",
                                      "reason": "action_type_legality_"
                                                "derived_from_complete_state",
                                      "note": "min_raise sizing detail "
                                              "declared"},
                    "action_sequence_sizes": [
                        {"street": d["street"], "player": d["player"],
                         "action": d["observed_action"],
                         "to_bb": d["observed_size_to_bb"]}
                        for d in dps if d["player"] == pid]
                    + [{"street": d["street"], "player": AI_SEAT,
                        "action": "ai_seat_action",
                        "to_bb": None} for d in []],
                    "observed_hero_action": [
                        {"street": d["street"], "action": d["observed_action"],
                         "to_call_bb": d["to_call_bb"],
                         "all_in": d["all_in"]} for d in hero_dps],
                    "opponent_identity_actions": [
                        p for p in parsed["players"] if p != pid],
                    "game_mode": "cash",
                    "blind_level": f"{parsed['sb']:.0f}/{parsed['bb']:.0f}",
                    "pluribus_ai_action": {
                        "verdict": "solver_label",
                        "seat": AI_SEAT,
                        "excluded_from_human_validation": True},
                    "_decision_points": hero_dps,
                }

    def iter_candidates(self, limit: int):
        n = 0
        for rec in self.iter_records():
            if limit and n >= limit:
                return
            dp = rec.pop("_decision_points", [])
            rec["legal_actions"] = rec["legal_actions"]  # verdict dict 直通
            rec["_decision_points"] = dp
            yield rec
            n += 1

    def evaluate_dataset(self, limit: int = 1000) -> dict:
        """P52 evaluate_dataset（human seats only；含 AI 排除计数）。"""
        from .decision_grade import (
            DATASET_VERDICTS, DECISION_GRADE_FIELDS, classify_record,
            dataset_verdict)
        field_counts = {s["key"]: {} for s in DECISION_GRADE_FIELDS}
        level_counts: dict[str, int] = {}
        n = 0
        human_action_points = 0
        for rec in self.iter_candidates(limit):
            n += 1
            mode = self.mode_of(rec)
            dp = rec.pop("_decision_points", [])
            human_action_points += len(dp)
            rec["_decision_points"] = dp
            fv = classify_record(rec, mode)
            for k, cell in fv.items():
                field_counts[k][cell["verdict"]] = \
                    field_counts[k].get(cell["verdict"], 0) + 1
            v = dataset_verdict(fv, mode)["verdict"]
            level_counts[v] = level_counts.get(v, 0) + 1
        ratios = {k: {v: round(c.get(v, 0) / n, 6) if n else 0.0
                      for v in ("usable", "partially_usable", "excluded",
                                "unknown")}
                  for k, c in field_counts.items()}
        return {"adapter": type(self).__name__,
                "dataset_meta": self.dataset_meta(),
                "n_records": n,
                "human_decision_points": human_action_points,
                "ai_seat_excluded_hands": self.ai_seat_excluded,
                "hands_skipped": self.hands_skipped,
                "field_usable_ratio": {k: r["usable"]
                                       for k, r in ratios.items()},
                "field_verdict_ratio": ratios,
                "dataset_verdict_ratio": {
                    v: round(level_counts.get(v, 0) / n, 6) if n else 0.0
                    for v in ("decision_grade", "behavior_grade",
                              "insufficient")}}


# ---- manifest（固定；确定性） ----
def manifest() -> dict:
    return {
        "manifest_version": "p54-decision-v1",
        "data_source": dict(PLURIBUS_DATASET_META),
        "split": {
            "rule": "hand_id 字典序确定性排序；前 70% = evidence，后 30% = evaluation",
            "evidence_frac": 0.7,
            "no_overlap": "hand_id 集合不相交（测试断言）",
            "ai_seat": "Pluribus 动作不进入任何证据/评估（solver_label）",
        },
        "validation": {
            "fold_calibration": "eval 人类 fold 决策点：BASELINE=population "
                                "条件率（human seats 证据）vs REAL=player "
                                "条件估计 vs observed fold（Brier/agreement）",
            "decision_ev": "eval 决策点重建 state → BASELINE（静态范围）vs "
                           "REAL（P46 posterior + P43 条件 fold，证据仅来自 "
                           "train 分割）→ compute_ev；model_recommendation "
                           "vs observed_human_action；regret proxy；"
                           "action-change rate",
            "equity_iterations": 1500,
            "eval_hand_limit_declared": True,
        },
        "label_discipline": {
            "observed_human_action": "人类座位真实动作（唯一 ground truth）",
            "pluribus_ai_action": "solver_label——排除出 human validation",
            "solver_label": "非真实玩家行为",
            "model_recommendation": "compute_ev 输出——非 ground truth",
        },
        "honesty": {
            "mtt": "NOT VALIDATABLE WITH CURRENT DATA",
            "pko": "NOT VALIDATABLE WITH CURRENT DATA",
            "research_play": ("研究对局条件——结论不得外推为自然线上玩家行为"),
            "profitability": "本验证不构成任何盈利性/GTO 证明",
        },
    }
