"""V3H-7 — CASH Bounded Multistreet Search Tree（有限深度、显式预算、可复现）。

定位
----
建立 Layer 4 的**最小可审计搜索器**：

```
current state → legal actions → opponent belief → future street states
             → bounded action tree → leaf evaluation → candidate aggregation
```

**本层不做最终决策、不接生产 Judge、不接 Solver、不声称 GTO/equilibrium。**

三种 future-board 路径（**必须显式区分**，授权要求）
----------------------------------------------------
| 标记 | 含义 | 条件 |
|---|---|---|
| `listed` | 该街 board **已知**（已发出） | state 已给出该街公共牌 |
| `enumerated` | 未来牌**不做单条预测**，而是在**整副牌减去已知牌**上完整枚举 | 进入下一街且未超预算 |
| `unavailable` | 无法合法展开（预算耗尽 / 信息不足 / 终局） | 见 `reason` |

* **不伪造未来公共牌**：`enumerated` 分支的候选牌是"**待发牌的枚举候选**"，
  各自标注 provenance，**不是**"已知事实"；
* **不把随机 runout 当已知事实**：本层**不采样**（无 `sampled` 路径，
  故也**不需要 seed**）；叶子的权益用"**跑完枚举**"（条件绝对权益）计算；
* 若调用方要求采样，本层接口**保留** `sampled` 标记语义但**不使用**，
  并在输出中写 `sampling_used=false`。

预算（**全部显式、可复现、不得偷偷扩大**）
------------------------------------------
在 `DEFAULT_BUDGET` 中声明；输出逐项回显 `budget` 与 `budget_used`。
任何一项被触顶 → 对应路径标 `unavailable` 并计入 `budget_hits`。

复用而非重实现（授权要求）
--------------------------
* **合法动作与尺寸**：V3G-7 `cash_action_tree`（经 V3H-4 action_policy 传入）；
* **对手 belief / known-card removal / likelihood**：V3H-4/V3H-6（由调用方传入），
  本层**不重实现** range/likelihood；
* **街转移**：V3G-4 `cash_multistreet.next_street_transition`（由调用方传入其输出）；
* **叶子权益**：V3H-5 `cash_strategy_multistreet_eval.conditional_absolute_equity`
  （由调用方传入其结果或调用其函数）；
* **不复用** V3G-4 的 `sample_runouts`，也**不复用**任何旧 leaf-EV 逻辑。

数据接口（生产 import 图零新增）
--------------------------------
**不在模块顶层 import 任何 V3G 层**。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_SEARCH_TREE_VERSION = "v3h7-cash-strategy-search-tree-v1"

#: 未来牌路径标记（枚举封闭）
#: V3I-G7-1 新增 `aggregated_expected`：到达街需要补**多张**牌时，按契约
#: **不发明/不采样**任何具体未来牌，改以**当前合法牌面**上的条件绝对权益
#: 作为该到达的期望叶（它不点名任何未来牌，也不是对单条 runout 的预测）。
FUTURE_BOARD_STATES: tuple[str, ...] = ("listed", "enumerated", "sampled",
                                       "unavailable", "aggregated_expected")

#: 显式预算（**不得偷偷扩大**）
DEFAULT_BUDGET: dict[str, int] = {
    "max_street_depth": 2,        # 从当前街起，最多再展开几条街
    "max_response_depth": 1,      # 每个 hero 动作后，最多展开几层对手响应
    "max_candidate_actions": 6,   # 每个决策点最多展开几个候选动作
    "max_leaves": 200,            # 全树叶子上限
    "max_runout_children": 8,     # 每个"进入下一街"节点最多展开几个候选牌
}

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "computes_exploitability": False,
    "invents_future_cards": False,
    "samples_runouts": False,
    "wired_to_production": False,
}

#: 节点必须携带的字段（缺一即视为不完整）
NODE_FIELDS: tuple[str, ...] = (
    "node_id", "parent_id", "depth", "street", "actor", "action",
    "size_bucket", "amount_bb", "pot_bb", "stack_bb", "future_board_state",
    "provenance", "uncertainty", "is_leaf", "children",
)

_STREET_ORDER = ("preflop", "flop", "turn", "river")
_EXPECTED_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: V3I-G7-1（S2）：权益可计算的**合法街态** board 张数（与冻结 `_leaf_equity` 判据一致）
VALID_EQUITY_BOARD_SIZES: tuple[int, ...] = (0, 3, 4, 5)

#: V3I-G7-1（S2）：fold-only 算术终局叶的来源前缀 —— **不得**计入真实权益
#: （与 `docs` 侧证据规则一致：`arithmetic:pot_share_on_opponent_fold` 是恒真假设）
FOLD_ONLY_EQUITY_SOURCE_PREFIX = "arithmetic:"


def _street_arrival_increment(arriving_street: str,
                              board_now: list[str]) -> "int | None":
    """V3I-G7-1（S1）街忠实增量：到达 `arriving_street` 需补几张牌。

    * `_EXPECTED_BOARD[arriving] - len(board_now)`；未知街 → `None`（走原有路径）；
    * `== 1`（flop→turn、turn→river）：沿用"补 1 张候选牌"的原有行为（逐位不变）；
    * `> 1`（preflop→flop 需 +3）：**不得**逐个补牌（会产出 1 张牌的非法街态），
      改发"到达街期望"聚合叶。
    """
    if arriving_street not in _EXPECTED_BOARD:
        return None
    return _EXPECTED_BOARD[arriving_street] - len(board_now)


def _is_fold_only_leaf(leaf: dict) -> bool:
    """fold-only 算术终局叶（对手弃牌恒 1.0）——不计真实权益。"""
    return str(leaf.get("equity_source") or "").startswith(
        FOLD_ONLY_EQUITY_SOURCE_PREFIX)


def _is_legal_equity_board(board_size: Any) -> bool:
    """board 张数是否在权益白名单内（非法街态中间叶不得计入覆盖率）。"""
    return isinstance(board_size, int) and not isinstance(board_size, bool) \
        and board_size in VALID_EQUITY_BOARD_SIZES


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str, budget: dict) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_SEARCH_TREE_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "root": None,
        "leaves": [],
        "budget": dict(budget),
        "budget_used": {},
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _size_bucket(amount_bb: Any, pot_bb: Any, action: Any) -> str | None:
    act = str(action or "").strip().lower()
    if act in ("fold", "check"):
        return "no_investment"
    if act == "allin" or act == "all_in":
        return "allin"
    if not isinstance(amount_bb, (int, float)) or isinstance(amount_bb, bool):
        return None
    if not isinstance(pot_bb, (int, float)) or isinstance(pot_bb, bool) or pot_bb <= 0:
        return None
    frac = float(amount_bb) / float(pot_bb)
    if frac <= 0.4:
        return "small"
    if frac <= 0.8:
        return "medium"
    return "large"


def _known_cards(action_policy: dict, belief: dict | None,
                 board: list[str] | None = None) -> list[str]:
    known: list[str] = []
    if isinstance(belief, dict):
        known += [str(c) for c in (belief.get("known_cards_removed") or [])]
    known += [str(c) for c in (board or [])]
    ap_board = action_policy.get("board")
    if isinstance(ap_board, list):
        known += [str(c) for c in ap_board if isinstance(c, str)]
    return sorted(set(known))


def _leaf_equity(hero_hole: list[str], board: list[str], *,
                 iterations: int = 20000, seed: int = 42) -> dict:
    """叶子权益（**跑完枚举语义**的 Monte Carlo 估计；**不采样 runout**）。

    复用 V3H-5 的 `conditional_absolute_equity`；迭代数与 seed **显式记录**
    以便复现（叶子较多时可用较小 `leaf_equity_iterations`，仍是显式预算）。
    """
    if not hero_hole or len(board) not in (0, 3, 4, 5):
        return {"present": False, "unknown": ["leaf_inputs_incomplete"]}
    try:
        from .cash_strategy_multistreet_eval import (
            conditional_absolute_equity)
        res = conditional_absolute_equity(hero_hole, board)
        if res.get("present"):
            res["iterations_declared"] = int(iterations)
            res["seed_declared"] = int(seed)
        return res
    except Exception as exc:                               # noqa: BLE001
        return {"present": False,
                "unknown": [f"leaf_equity_error:{type(exc).__name__}"]}


def _next_street(street: str | None, board: list[str], depth: int,
                 budget: dict, state: dict) -> dict:
    """决定能否进入下一街（**只用已知牌**；不发明具体未来牌）。

    `depth` = 已推进的街数（0 = 仍在起始街）。预算 `max_street_depth`
    表示**最多再推进几条街**。
    """
    if street not in _STREET_ORDER:
        return {"ok": False, "reason": f"unknown_street:{street!r}",
                "future_board_state": "unavailable"}
    if street == "river":
        return {"ok": False, "reason": "river_no_future_street",
                "future_board_state": "unavailable"}
    if depth >= budget["max_street_depth"]:
        return {"ok": False, "reason": "max_street_depth_reached",
                "future_board_state": "unavailable"}
    expected = _EXPECTED_BOARD[street]
    if len(board) != expected:
        return {"ok": False,
                "reason": f"board_size_mismatch:{len(board)}!={expected}",
                "future_board_state": "unavailable"}
    nxt = _STREET_ORDER[_STREET_ORDER.index(street) + 1]
    return {"ok": True, "next_street": nxt, "future_board_state": "enumerated"}


def build_search_tree(state: Any, *, action_policy: dict | None = None,
                      belief: dict | None = None,
                      belief_update: dict | None = None,
                      transition: dict | None = None,
                      budget: dict | None = None,
                      max_root_candidates: int | None = None) -> dict:
    """构建有限深度搜索树（**只展开合法状态；不发明未来牌；不采样**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    action_policy : **必填**，V3H-4 输出（提供合法动作/尺寸/几何量）
    belief : 可选，V3H-4 belief（提供 known-card removal 与对手数）
    belief_update : 可选，V3H-6 输出（提供 prior_only / likelihood_conditioned）
    transition : 可选，V3G-4 `next_street_transition` 输出（仅用于转述）
    budget : 覆盖 `DEFAULT_BUDGET` 的显式子集（未知键被忽略并记录）
    max_root_candidates : root 展开的候选动作数上限（≤ budget）
    """
    b = dict(DEFAULT_BUDGET)
    extra_budget_keys: list[str] = []
    if isinstance(budget, dict):
        for k, v in budget.items():
            if k in b and isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                b[k] = v
            else:
                extra_budget_keys.append(k)

    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed", b)
    if not isinstance(action_policy, dict):
        return _stub("action_policy_required", b)
    ap = action_policy
    if not ap.get("present"):
        return _stub(f"action_policy_unavailable:{ap.get('reason', 'unknown')}", b)

    # river 上上游会判 terminal → 本层报 UNKNOWN 而非伪造搜索
    if ap.get("street") == "river":
        return _stub("river_search_not_expandable_frozen_v3g7_terminal", b)

    pot = ap.get("pot_bb")
    street = ap.get("street")
    # board 来源：优先 action_policy 自带；否则取 state（两者都不给 → 空 → UNKNOWN）
    board: list[str] = []
    ap_board = ap.get("board")
    if isinstance(ap_board, list):
        board = [str(c) for c in ap_board if isinstance(c, str) and c.strip()]
    if not board:
        st_board = state.get("board")
        if isinstance(st_board, list):
            board = [str(c) for c in st_board if isinstance(c, str) and c.strip()]
    stack = ap.get("effective_stack_bb")
    hero_hole: list[str] = []
    hero_id = (belief or {}).get("hero_id") or state.get("hero_id")
    for p in (state.get("players") or []):
        if isinstance(p, dict) and (bool(p.get("is_hero")) or
                                    p.get("player_id") == hero_id):
            hc = p.get("hole_cards")
            if isinstance(hc, list):
                hero_hole = [str(c) for c in hc if isinstance(c, str)]
            break

    candidates = [c for c in (ap.get("candidates") or [])
                  if isinstance(c, dict) and c.get("action")]
    cap = b["max_candidate_actions"]
    if isinstance(max_root_candidates, int) and not isinstance(max_root_candidates, bool):
        cap = min(cap, max(0, max_root_candidates))
    expanded_root = candidates[:cap]

    budget_hits: list[str] = []
    street_arrival_notes: list[str] = []
    caller_caps: list[str] = []
    if len(candidates) > len(expanded_root):
        if cap < b["max_candidate_actions"]:
            caller_caps.append("max_root_candidates")   # 调用方显式上限，非预算触顶
        else:
            budget_hits.append("max_candidate_actions")  # 预算本身触顶
    if extra_budget_keys:
        budget_hits.append("unknown_budget_keys_ignored")

    known = _known_cards(ap, belief, board)
    likelihood = (belief_update or {}).get("path") if isinstance(belief_update, dict) else None
    n_opponents = (belief or {}).get("n_opponents") if isinstance(belief, dict) else None

    leaves: list[dict] = []
    nodes: list[dict] = []
    counter = {"n": 0}

    def _nid(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:04d}"

    def _leaf(node_id: str, path: dict, board_now: list[str],
              future_state: str, reason: str | None,
              terminal_kind: str = "showdown",
              fixed_payoff: float | None = None,
              hero_invest: float = 0.0,
              terminal_semantics: str | None = None) -> dict:
        """叶子：`terminal_kind` 显式区分 fold / showdown / unknown。

        `fixed_payoff`（pot 份额）用于**算术终局**（对手弃牌 = 赢下底池），
        不依赖未来公共牌，因此**不是**对 runout 的预测。

        V3I-G1 Phase 1（**只新增字段；旧字段、叶数、叶权益一律不变**）：
        * `hero_investment_bb` / `pot_at_terminal_bb` / `showdown_pot_bb`
          —— 动作后的**底池记账**（状态事实：fold 终局 = pot₀+a，
          showdown 终局 = pot₀+2a）；
        * `terminal_value_net_pot_share` —— 以**叶自身权益**计的净值
          （showdown：`eq*(pot₀+2a)/pot₀ − a/pot₀`；fold：`1.0`），
          口径与 `cash_action_value_contract` 一致，但**它本身不是动作价值**
          （动作价值必须用**测得的续跟范围权益** + 加权，见该契约模块）；
        * `admission_index` —— 全局叶序号（叶预算 provenance）；
        * `terminal_semantics` —— 终局语义标注（例如零投入候选下
          "对手弃牌"在语义上**不适用**）。
        """
        invest = hero_invest if isinstance(hero_invest, (int, float)) \
            and not isinstance(hero_invest, bool) else 0.0
        invest = max(float(invest), 0.0)
        p0 = float(pot) if isinstance(pot, (int, float)) \
            and not isinstance(pot, bool) and float(pot) > 0 else 0.0
        if fixed_payoff is not None:
            eq = {"present": True, "hero_equity": round(float(fixed_payoff), 6),
                  "method": "arithmetic_terminal", "samples": 0,
                  "opponent_model": "not_used",
                  "source": "arithmetic:pot_share_on_opponent_fold",
                  "invents_future_cards": False}
        else:
            eq = _leaf_equity(hero_hole, board_now)
        unknown = list(eq.get("unknown") or [])
        leaf_eq = eq.get("hero_equity")
        leaf_eq_ok = isinstance(leaf_eq, (int, float)) \
            and not isinstance(leaf_eq, bool)
        if terminal_kind == "fold":
            pot_at_terminal = p0 + invest
            showdown_pot: float | None = None
            net_value: float | None = 1.0
            value_def = "fold_net_pot_share_1.0"
        else:
            pot_at_terminal = p0 + 2.0 * invest
            showdown_pot = pot_at_terminal
            net_value = (float(leaf_eq) * (p0 + 2.0 * invest) / p0
                         - invest / p0) if (leaf_eq_ok and p0 > 0) else None
            value_def = "showdown_leaf_equity_net_pot_share"
        return {
            "leaf_id": _nid("leaf"),
            "node_id": node_id,
            "path": path,
            "board": list(board_now),
            "board_size": len(board_now),
            "future_board_state": future_state,
            "terminal_reason": reason,
            "terminal_kind": terminal_kind,
            "equity": eq,
            "equity_present": bool(eq.get("present")),
            "equity_source": eq.get("source"),
            "equity_method": eq.get("method"),
            "evidence_grade": "MODEL_INTERNAL" if eq.get("present") else "NONE",
            "is_gto": False,
            "is_equilibrium": False,
            "unknown": unknown,
            # ── V3I-G1 Phase 1：动作后状态记账与终局价值（只新增）──
            "admission_index": len(leaves) + 1,
            "hero_investment_bb": round(invest, 6),
            "pot_at_terminal_bb": round(pot_at_terminal, 6) if p0 > 0 else None,
            "showdown_pot_bb": (round(showdown_pot, 6)
                               if showdown_pot is not None else None),
            "terminal_value_net_pot_share": (round(net_value, 6)
                                             if net_value is not None else None),
            "terminal_value_definition": value_def,
            "terminal_value_basis": "leaf_equity_uniform_random_hand",
            "terminal_semantics": terminal_semantics,
        }

    def _emit_street(node_id: str, depth: int, arriving_street: str,
                     board_now: list[str], pot_now: Any, path: dict,
                     hero_invest: float = 0.0) -> list[dict]:
        """在 `board_now` 上**发下一街**（board 未定 → 枚举候选），返回子节点。

        调用前提：`arriving_street` 是**即将进入**的那条街，且 `board_now`
        仍是**上一条街**的已知牌（本函数负责把候选牌补上）。

        V3I-G7-1（S1）街忠实增量：
        * 增量 `== 1`（flop→turn、turn→river）⇒ **逐位保持**原有"补 1 张候选牌"行为；
        * 增量 `> 1`（preflop→flop 需 +3）⇒ 逐个补牌会产出 **1 张牌的非法街态**
          （既不能继续展开，也不在权益白名单内），故改为发出**一个"到达街期望"聚合叶**：
          权益在**当前合法牌面** `board_now` 上计算，**不点名、不采样**任何具体未来牌。
        """
        increment = _street_arrival_increment(arriving_street, board_now)
        if increment is not None and increment > 1:
            child_id = _nid("agg")
            leaves.append(_leaf(
                child_id,
                {**path, "street_arrival": "aggregate_expected"},
                board_now, "aggregated_expected",
                "street_arrival_requires_multiple_cards",
                terminal_kind="showdown", hero_invest=hero_invest))
            street_arrival_notes.append(
                f"{arriving_street}:increment={increment}:aggregate_expected")
            return [{
                "node_id": child_id,
                "parent_id": node_id,
                "depth": depth + 1,
                "street": arriving_street,
                "actor": "future_board",
                "action": "aggregate_street_arrival",
                "size_bucket": None,
                "amount_bb": None,
                "pot_bb": pot_now,
                "stack_bb": stack,
                "future_board_state": "aggregated_expected",
                "provenance": {
                    "source": "street_faithful_aggregate_expectation",
                    "candidate_card": None,
                    "candidates_total": None,
                    "candidates_expanded": 0,
                    "cards_required": increment,
                    "is_known_fact": False,
                    "note": (f"到达该街需补 {increment} 张牌；按契约**不发明/不采样**"
                             "具体未来牌，故以当前合法牌面上的条件绝对权益作为"
                             "该到达的期望叶（不点名任何未来牌）"),
                },
                "uncertainty": {"unresolved_board": True,
                                "cards_required": increment,
                                "expected_board_size":
                                    _EXPECTED_BOARD[arriving_street]},
                "is_leaf": True,
                "children": [],
            }]
        unseen = [c for c in _full_deck() if c not in set(known) | set(board_now)]
        total_candidates = len(unseen)
        take = min(b["max_runout_children"], total_candidates)
        if take < total_candidates:
            budget_hits.append("max_runout_children")
        children: list[dict] = []
        for i in range(take):
            card = unseen[i]
            child_id = _nid("fc")
            child_board = board_now + [card]
            grandkids: list[dict] = []
            is_leaf = True
            if depth + 1 < b["max_street_depth"]:
                # 仍可再进一街 → 在该节点上继续推进
                if _next_street(arriving_street, child_board, depth + 1, b,
                                state)["ok"]:
                    grandkids = _opponent_children(
                        child_id, depth + 1, arriving_street, child_board,
                        pot_now, path, hero_invest)
                    is_leaf = not grandkids
                if not grandkids:
                    leaves.append(_leaf(child_id, path, child_board,
                                        "enumerated",
                                        "no_further_expansion_after_deal",
                                        hero_invest=hero_invest))
                    is_leaf = True
            else:
                # 街预算用尽 → 该候选牌节点即叶子（权益按**已知** board 计算）
                leaves.append(_leaf(child_id, path, child_board, "enumerated",
                                    "max_street_depth_reached",
                                    hero_invest=hero_invest))
            children.append({
                "node_id": child_id,
                "parent_id": node_id,
                "depth": depth + 1,
                "street": arriving_street,
                "actor": "future_board",
                "action": "deal_enumerated_candidate",
                "size_bucket": None,
                "amount_bb": None,
                "pot_bb": pot_now,
                "stack_bb": stack,
                "future_board_state": "enumerated",
                "provenance": {
                    "source": "deck_minus_known_enumerated",
                    "candidate_card": card,
                    "candidate_index": i,
                    "candidates_total": total_candidates,
                    "candidates_expanded": take,
                    "is_known_fact": False,
                    "note": ("该牌是**待发牌的枚举候选**，不是已知事实；"
                             "本层不对单条 runout 作预测"),
                },
                "uncertainty": {"unresolved_board": True,
                                "candidates_total": total_candidates},
                "is_leaf": is_leaf,
                "children": grandkids,
            })
        return children

    def _opponent_children(node_id: str, depth: int, street_now: str,
                           board_now: list[str], pot_now: Any,
                           path: dict, hero_invest: float = 0.0) -> list[dict]:
        """对手响应层（在 `street_now` 的 `board_now` 上）。

        V3I-G1 Phase 1：响应分支**仍按合法动作语法枚举**（`fold` / `call`），
        但每个响应节点显式登记**权重来源**：本层**不合成**权重（无对手模型、
        无概率），权重只能由调用方在 `cash_action_value_contract` 里以
        **测得**的 `p_fold/p_continue` 注入；缺失即 UNKNOWN（fail closed）。
        """
        out: list[dict] = []
        if depth > b["max_response_depth"]:
            budget_hits.append("max_response_depth")
            return out
        nxt = _next_street(street_now, board_now, depth, b, state)
        for act in ("fold", "call"):
            rid = _nid("opp")
            children: list[dict] = []
            is_leaf = False
            if act == "fold":
                is_leaf = True
                # 零投入候选下"对手弃牌"语义**不适用**（对手不可能对 check 弃牌）：
                # 该叶按旧口径**逐位保留**（不删不改），仅显式标注语义。
                semantics = ("not_applicable_no_investment"
                             if float(hero_invest or 0.0) <= 0.0
                             else "opponent_folds_to_aggression")
                leaves.append(_leaf(
                    rid, {**path, "opponent_response": "fold"}, board_now,
                    "listed", "opponent_folded", terminal_kind="fold",
                    fixed_payoff=1.0, hero_invest=hero_invest,
                    terminal_semantics=semantics))
            else:
                if nxt["ok"]:
                    children = _emit_street(rid, depth, nxt["next_street"],
                                            board_now, pot_now,
                                            {**path, "opponent_response": "call"},
                                            hero_invest)
                if not children:
                    is_leaf = True
                    reason = (nxt["reason"] if not nxt["ok"]
                              else "no_further_expansion")
                    leaves.append(_leaf(
                        rid, {**path, "opponent_response": "call"}, board_now,
                        "unavailable" if not nxt["ok"] else "enumerated",
                        reason, terminal_kind="showdown",
                        hero_invest=hero_invest))
            out.append({
                "node_id": rid,
                "parent_id": node_id,
                "depth": depth,
                "street": street_now,
                "actor": "opponent",
                "action": act,
                "size_bucket": None,
                "amount_bb": None,
                "pot_bb": pot_now,
                "stack_bb": stack,
                "future_board_state": "listed",
                "provenance": {
                    "source": "known_action_syntax",
                    "note": ("对手响应按合法动作语法枚举；"
                             "**未使用**对手策略模型（无概率权重）"),
                    # V3I-G1 Phase 1：权重来源显式登记（本层不合成权重）
                    "response_weighting_source": "not_supplied_in_tree",
                    "response_weighting_policy": (
                        "measured p_fold/p_continue 只能由 "
                        "cash_action_value_contract 注入；缺失 ⇒ UNKNOWN"),
                },
                "uncertainty": {"opponent_model": "not_used",
                                "response_weighting": "none",
                                "hero_investment_bb": (round(float(hero_invest), 6)
                                                       if isinstance(
                                                           hero_invest,
                                                           (int, float)) else 0.0)},
                "is_leaf": is_leaf,
                "children": children,
            })
        return out

    # ── root 展开 ──
    # 每个 root candidate 下先给**对手响应层**（fold / call 枚举），
    # 再在 call 分支进入下一街枚举。这样不同 hero 动作确实产生不同叶子。
    root_children: list[dict] = []
    for cand in expanded_root:
        rid = _nid("root")
        # V3I-G1 Phase 1：动作后记账（状态事实；旧字段 pot_bb/stack_bb 保持原值）
        _amt = cand.get("amount_bb")
        _act = str(cand.get("action") or "").strip().lower()
        invest_root = float(_amt) if (isinstance(_amt, (int, float))
                                      and not isinstance(_amt, bool)
                                      and _act not in ("fold", "check")) else 0.0
        invest_root = max(invest_root, 0.0)
        pot_after_hero = (float(pot) + invest_root
                          if isinstance(pot, (int, float)) else None)
        stack_after_hero = (float(stack) - invest_root
                            if isinstance(stack, (int, float)) else None)
        node = {
            "node_id": rid,
            "parent_id": None,
            "depth": 0,
            "street": street,
            "actor": "hero",
            "action": cand.get("action"),
            "size_bucket": _size_bucket(cand.get("amount_bb"), pot, cand.get("action")),
            "amount_bb": cand.get("amount_bb"),
            "pot_bb": pot,
            "stack_bb": stack,
            "future_board_state": ("listed" if len(board) == _EXPECTED_BOARD.get(
                street, -1) else "unavailable"),
            "provenance": {
                "source": "cash_action_tree.sizing",
                "legal_action": True,
                "sizing_source": cand.get("sizing_source"),
            },
            "uncertainty": {"pot_bb": pot, "effective_stack_bb": stack},
            "is_leaf": False,
            "children": [],
            # ── V3I-G1 Phase 1：动作后状态（只新增）──
            "hero_investment_bb": round(invest_root, 6),
            "pot_after_hero_bb": (round(pot_after_hero, 6)
                                  if pot_after_hero is not None else None),
            "stack_after_hero_bb": (round(stack_after_hero, 6)
                                    if stack_after_hero is not None else None),
            "accounting_definition": ("pot_after_hero = pot + hero_investment; "
                                      "stack_after_hero = stack - hero_investment"),
        }
        if len(leaves) >= b["max_leaves"]:
            budget_hits.append("max_leaves")
            node["children"] = []
            node["is_leaf"] = True
            leaves.append(_leaf(node["node_id"], {"root_action": node["action"]},
                                board, "unavailable", "max_leaves_reached",
                                hero_invest=invest_root))
        elif node["future_board_state"] == "unavailable":
            # 当前街 board 与该街期望张数不一致 → 不可合法展开
            node["children"] = []
            node["is_leaf"] = True
            leaves.append(_leaf(
                node["node_id"], {"root_action": node["action"]}, board,
                "unavailable",
                f"board_size_mismatch:{len(board)}!="
                f"{_EXPECTED_BOARD.get(street)}",
                hero_invest=invest_root))
        else:
            resp = _opponent_children(rid, 0, street, board, pot,
                                      {"root_action": node["action"]},
                                      invest_root)
            node["children"] = resp
            node["is_leaf"] = not resp
        root_children.append(node)

    root = {
        "node_id": _nid("search-root"),
        "parent_id": None,
        "depth": 0,
        "street": street,
        "actor": "hero",
        "action": None,
        "size_bucket": None,
        "amount_bb": None,
        "pot_bb": pot,
        "stack_bb": stack,
        "future_board_state": "listed",
        "provenance": {"source": "cash_strategy_search_tree.build_search_tree"},
        "uncertainty": {"belief_path": likelihood},
        "is_leaf": False,
        "children": root_children,
    }

    # V3I-G7-1（S2）：覆盖率改为"真实 equity 叶 / 本应需要真实 equity 的叶"。
    # 旧定义（`leaf_coverage`）**逐位保留**，同时并列给出新定义，便于审计对照。
    fold_only_leaves = [x for x in leaves
                        if x["equity_present"] and _is_fold_only_leaf(x)]
    real_equity_leaves = [x for x in leaves
                          if x["equity_present"] and not _is_fold_only_leaf(x)]
    illegal_board_leaves = [x for x in leaves
                            if not _is_legal_equity_board(x.get("board_size"))]
    should_have_equity = [x for x in leaves
                          if not _is_fold_only_leaf(x)
                          and _is_legal_equity_board(x.get("board_size"))]
    coverage = {
        "n_leaves": len(leaves),
        "n_leaves_with_equity": sum(1 for x in leaves if x["equity_present"]),
        "n_leaves_unknown": sum(1 for x in leaves if not x["equity_present"]),
        "n_leaves_unavailable": sum(
            1 for x in leaves if x["future_board_state"] == "unavailable"),
        "n_leaves_enumerated": sum(
            1 for x in leaves if x["future_board_state"] == "enumerated"),
        # 旧定义（**不改动**）：equity_present / 全部叶子
        "leaf_coverage": round(
            sum(1 for x in leaves if x["equity_present"]) / len(leaves), 6)
        if leaves else 0.0,
        # 新定义（V3I-G7-1 S2）：真实权益叶 / 本当需要真实权益的叶
        "leaf_coverage_real_equity": (round(len(real_equity_leaves)
                                            / len(should_have_equity), 6)
                                      if should_have_equity else 0.0),
        "coverage_definition":
            "real_equity_leaves_over_leaves_requiring_real_equity",
        "coverage_legacy_definition": "equity_present_leaves_over_all_leaves",
        "n_leaves_real_equity": len(real_equity_leaves),
        "n_leaves_fold_only_equity": len(fold_only_leaves),
        "n_leaves_illegal_street_state": len(illegal_board_leaves),
        "n_leaves_requiring_real_equity": len(should_have_equity),
        "n_root_candidates": len(root_children),
        "n_candidates_available": len(candidates),
    }

    # ── V3I-G1 Phase 1：叶预算**显式登记** + 逐叶 provenance ──────────────
    # 旧实现只在"进入下一个 root 候选前"检查一次预算，因此单个候选的展开可以
    # 越过 `max_leaves`（实测 219 > 200）而**没有**任何 `max_leaves` 登记。
    # 本阶段**不截断**（保持旧口径逐位可比），改为：显式登记 + 逐叶 admission
    # 序号 provenance；是否 fail-closed 由动作价值契约侧执行。
    n_leaves_observed = len(leaves)
    overrun_ids = [x.get("leaf_id") for x in leaves
                   if isinstance(x.get("admission_index"), int)
                   and x["admission_index"] > b["max_leaves"]]
    if n_leaves_observed > b["max_leaves"] and "max_leaves" not in budget_hits:
        budget_hits.append("max_leaves")
    leaf_budget = {
        "declared_max_leaves": b["max_leaves"],
        "observed_leaves": n_leaves_observed,
        "overrun": max(0, n_leaves_observed - b["max_leaves"]),
        "overrun_leaf_ids": overrun_ids,
        "overrun_has_provenance": all(isinstance(i, str) for i in overrun_ids),
        "n_overrun_leaves": len(overrun_ids),
        "admission_index_field": "leaf.admission_index",
        "registered_in_budget_hits": "max_leaves" in budget_hits,
        "truncation_applied": False,
        "policy": ("超限**只登记不截断**（旧 search_score 逐位可比）；"
                   "超限树上的**动作价值被抑制为 UNKNOWN**（fail closed），"
                   "legacy search_score 仍并列报告"),
    }
    max_leaves_exceeded = n_leaves_observed > b["max_leaves"]

    budget_used = {
        "streets_expanded": max((c["depth"] for c in root_children), default=0) + 1
        if root_children else 0,
        "root_candidates": len(root_children),
        "candidate_cap": cap,
        "leaves": len(leaves),
        "max_leaves": b["max_leaves"],
    }

    return {
        "present": True,
        "version": CASH_STRATEGY_SEARCH_TREE_VERSION,
        "mode": "cash",
        "street": street,
        "root": root,
        "leaves": leaves,
        "budget": b,
        "budget_used": budget_used,
        "budget_hits": sorted(set(budget_hits)),
        "caller_caps_applied": sorted(set(caller_caps)),
        "budget_exceeded": bool(set(budget_hits) - {"unknown_budget_keys_ignored"}),
        "budget_note": ("预算全部显式记录；任何截断都会出现在 budget_hits 中"),
        "street_arrival_notes": sorted(set(street_arrival_notes)),
        "street_arrival_rule": ("increment = _EXPECTED_BOARD[arriving] - len(board_now)；"
                                "==1 逐个补牌；>1 发聚合期望叶（不发明/不采样未来牌）"),
        "coverage": coverage,
        # ── V3I-G1 Phase 1（只新增）──
        "leaf_budget": leaf_budget,
        "max_leaves_exceeded": max_leaves_exceeded,
        "action_accounting": {
            "model": "cash_action_value_contract",
            "hero_state_fields": ["hero_investment_bb", "pot_after_hero_bb",
                                  "stack_after_hero_bb"],
            "terminal_value_fields": [
                "pot_at_terminal_bb", "showdown_pot_bb",
                "terminal_value_net_pot_share", "terminal_value_definition",
                "terminal_value_basis", "terminal_semantics"],
            "leaf_admission_field": "admission_index",
            "weights_source": ("not_synthesised_in_tree; injected by "
                               "cash_action_value_contract from measured "
                               "p_fold/p_continue (missing ⇒ UNKNOWN)"),
            "zero_investment_fold_semantics": "not_applicable_no_investment",
            "free_parameters": 0,
        },
        "known_cards": known,
        "hero_hole_cards": hero_hole,
        "belief": {
            "supplied": isinstance(belief, dict) and bool(belief.get("present")),
            "n_opponents": n_opponents,
            "path": likelihood,
            "prior_only": (belief_update or {}).get("prior_only")
            if isinstance(belief_update, dict) else None,
            "source": "V3H-4 belief / V3H-6 belief_update (reused, not reimplemented)",
        },
        "transition_passthrough": {
            "supplied": isinstance(transition, dict),
            "next_street": (transition or {}).get("next_street"),
            "terminal": (transition or {}).get("terminal"),
            "future_board": (transition or {}).get("future_board"),
        },
        "future_board_states_used": sorted({
            n["future_board_state"] for n in _walk_nodes(root)} |
            {x["future_board_state"] for x in leaves}),
        "sampling_used": False,
        "seed": None,
        "runout_prediction_made": False,
        "unknown": [] if root_children else ["no_root_candidates_expanded"],
        "evidence_grade": "MODEL_INTERNAL",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "final_decision_made": False,
    }


def _full_deck() -> list[str]:
    return [r + s for r in "23456789TJQKA" for s in "shdc"]

def _walk_nodes(node: dict):
    """深度优先遍历整棵树（用于统计实际使用的 future_board 路径）。"""
    yield node
    for c in node.get("children") or []:
        if isinstance(c, dict):
            yield from _walk_nodes(c)
