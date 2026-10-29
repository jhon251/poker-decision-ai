"""P30-9-CG：6-max 位置键 spot 查表（与 HU 的 `SpotTable` 并存）。

为什么需要：本项目原 spot 库是 **HU**（桶键含 oop/ip 范围字符串）；
外部 6-max 解是**位置对 + 节点**（`srp-LJ-BTN`、`first_in`/`facing_bet`）
⇒ 需要新的键空间才能被 live 命中。

键：`(street, oop_pos, ip_pos, node, pot档, 纹理类[, 尺寸桶])`
命中 ⇒ 返回与 HU 表同形的结果，并声明
`solver_name="postflop"` + `node` ⇒ 恰好满足 `anchor_from_solver` 契约（翻后 + 真实 solver）。

来源与许可：数据来自 github.com/jensbaagaard/poker-practice（README: MIT；
data/openSourcePokerData 可自由使用；翻后数据为 TexasSolver 离线求解输出）。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

SPOT6_VERSION = "p30-9-cg-spot-table-6max-v1"
DEFAULT_DIR = "data/solver/spots_6max"

#: 6-max 位置翻后行动顺序（用于 OOP/IP 与"谁先行动"判定）
POS_ORDER = {"SB": 0, "BB": 1, "LJ": 2, "UTG": 2, "HJ": 3, "MP": 3, "CO": 4, "BTN": 5}
#: 底池档（相对匹配容差）
POT_TOL_REL = 0.30
#: "底池近似"上限（无严格命中时；必须显式标注 is_approximate=True ⇒ 不锚定）
POT_TOL_REL_WIDE = 0.60
#: SPR (stack/pot) relative tolerance - labels "pot approximate but SPR close"
SPR_TOL_REL = 0.25
#: 尺寸近似的**有界倍数**：live ratio / solved ratio 必须落在 [1/F, F] 内，否则诚实 miss
#: （不同底池赔率 ⇒ 策略不可迁移；F 可用 SOLVER6_SIZE_FACTOR 覆盖，0=关闭该约束）
SIZE_APPROX_MAX_FACTOR = 2.0
#: 尺寸桶容差（bet_ratio 相对）
RATIO_TOL_REL = 0.35
#: P30-9-GD：**精确匹配的相对底池容差**——此前精确匹配只按 pot 档（"big" 跨 16–∞bb），
#: 会把底池差数倍的解标成 exact（误导）⇒ 超出该容差不标 exact（退桶/共识并标近似）。
POT_EXACT_TOL_REL = 0.25


def norm_pos(pos: Any) -> str:
    """位置名归一化：引擎侧 UTG/MP 与外部库 LJ/HJ 视为同一位置（6-max）。"""
    s = str(pos or "").upper()
    return {"UTG": "LJ", "MP": "HJ"}.get(s, s)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


# ── P30-9-HP：**请求纪元**（请求边界门控缓存可见性 ✓）───────────────
_REQUEST_EPOCH = 0


def begin_request() -> int:
    """标记一次**新请求**开始：此后缓存的新写入才会被表看到 ✓。

    动机（第 28 轮实测）：同一请求内后台写入缓存会立刻被重载 ⇒
    同输入可能得到不同输出（真实帧指标 5/7、5/7、4/7 波动 ✗）。
    """
    global _REQUEST_EPOCH
    _REQUEST_EPOCH += 1
    return _REQUEST_EPOCH


def current_epoch() -> int:
    """当前请求纪元（0 = 尚未调用 begin_request ✓）。"""
    return _REQUEST_EPOCH


def _consensus_from_rows(rows: list[dict]) -> dict | None:
    """P30-9-FT：把同纹理桶多个牌面的策略**聚合**成一个先验行。

    * 动作集合取各行**交集**（缺项会让平均有偏）；交集为空则退回首行动作集；
    * 逐手频率按**手牌字符串**对齐后取均值（仅在包含该手牌的行上平均，记录观测数）；
    * 返回值是首行的副本 + 聚合后的 `actions`/`freq_per_hand`/`hands`，
      `spot_id` 标为 `<首行>#consensus(n=k)` 以便审计；`consensus_n` 记录参与行数。
    """
    if not rows:
        return None
    names = None
    for r in rows:
        cur = {str(a.get("name")) for a in (r.get("actions") or [])}
        names = cur if names is None else (names & cur)
    if not names:
        names = {str(a.get("name")) for a in (rows[0].get("actions") or [])}
    if not names:
        return None
    # 手牌 → 动作 → 观测值列表
    acc: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    size_hint: dict[str, Any] = {}
    for r in rows:
        hands = [str(h) for h in (r.get("hands") or [])]
        fph = r.get("freq_per_hand") or {}
        for name in names:
            arr = fph.get(name)
            if not isinstance(arr, list):
                continue
            for i, h in enumerate(hands):
                if i >= len(arr):
                    break
                v = arr[i]
                if isinstance(v, (int, float)):
                    acc[name].setdefault(h, []).append(float(v))
        for a in (r.get("actions") or []):
            nm = str(a.get("name"))
            if nm in names and nm not in size_hint:
                size_hint[nm] = a.get("size_bb")
    if not any(acc.values()):
        return None
    hands_union = sorted({h for m in acc.values() for h in m})
    freq_out: dict[str, list[float]] = {}
    actions_out: list[dict] = []
    for name in sorted(names, key=lambda n: -(len(acc[n]))):
        vals = [ (sum(acc[name][h]) / len(acc[name][h])) if h in acc[name] else 0.0
                  for h in hands_union ]
        freq_out[name] = [round(v, 4) for v in vals]
        mean_freq = round(sum(vals) / len(vals), 4) if vals else 0.0
        actions_out.append({"name": name, "size_bb": size_hint.get(name),
                             "frequency": mean_freq})
    out = dict(rows[0])
    out.update({
        "spot_id": f"{rows[0].get('spot_id')}#consensus(n={len(rows)})",
        "actions": actions_out,
        "hands": hands_union,
        "freq_per_hand": freq_out,
        "consensus_n": len(rows),
        "consensus_from": [r.get("spot_id") for r in rows][:8],
        "pot_approximate": True,          # 共识必然不是该牌面的解 ⇒ 标近似
    })
    return out


def texture_class(board: list) -> str:
    """与 HU 表同源的纹理分类（调用其实现，避免两套口径）。"""
    from .spot_table import _texture_class
    return _texture_class([str(c) for c in (board or [])])


class SpotTable6max:
    """6-max 位置键查表（惰性加载；只读）。"""

    def __init__(self, dir_path: str = DEFAULT_DIR) -> None:
        self.dir = pathlib.Path(dir_path)
        self._loaded = False
        self._rows: list[dict] = []
        # P30-9-GI：现场解缓存目录（可选）
        self._cache_sig: tuple = ()
        # P30-9-HP：**请求纪元门控** —— 缓存写入只在**下一次请求**才可见 ✓
        #   （此前同一请求内后台写入缓存会立刻被重载 ⇒ 同输入不同输出 ✗，
        #     第 28 轮实测真实帧指标因此波动：5/7、5/7、4/7 ✗）
        self._cache_epoch: int = -1
        self._cache_dir = None
        try:
            from .live_cache import cache_dir as _cd
            self._cache_dir = _cd()
        except Exception:                                      # noqa: BLE001
            self._cache_dir = None

    # ---- 加载 ----------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._rows)
        self._loaded = True
        if not self.dir.is_dir():
            return False
        # P30-9-GI：把现场解缓存目录也纳入（P30-9-HP：**仅在请求边界**接受新增 ✓）
        _dirs = [self.dir]
        if self._cache_dir is not None:
            try:
                if self._cache_dir.is_dir():
                    _sig = (len(list(self._cache_dir.glob("*.json"))),
                            int(self._cache_dir.stat().st_mtime))
                    # 请求纪元未推进 ⇒ 本次请求内**忽略**缓存变化 ✓（同输入同输出 ✓）
                    if _sig != self._cache_sig and current_epoch() != self._cache_epoch:
                        self._cache_sig = _sig
                        self._cache_epoch = current_epoch()
                        _dirs.append(self._cache_dir)
            except Exception:                                  # noqa: BLE001
                pass
        for f in [p for d in _dirs for p in sorted(d.glob("*.json"))]:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            if not d.get("solved") or not d.get("actions"):
                continue
            pos = d.get("positions") or {}
            oop, ip = norm_pos(pos.get("oop")), norm_pos(pos.get("ip"))
            if oop not in POS_ORDER or ip not in POS_ORDER:
                continue
            self._rows.append({
                "spot_id": str(d.get("spot_id") or f.stem),
                "street": str(d.get("street") or "flop"),
                "board": [str(c) for c in (d.get("board") or [])],
                "texture": texture_class(d.get("board") or []),
                "pot_bb": _num(d.get("pot_bb")),
                "eff_bb": _num(d.get("effective_stack_bb")),
                # P30-9-CN：SPR = 有效筹码 / 底池（GTO 策略的主决定量）
                "spr": ((_num(d.get("effective_stack_bb")) or 0.0)
                         / (_num(d.get("pot_bb")) or 1.0)),
                "oop": oop, "ip": ip,
                "node": str(d.get("node") or "first_in"),
                # P30-9-EK：行动者（hero 位置）
                "actor": norm_pos(d.get("actor_position")),
                # P30-9-FR：翻前家族（3bp/srp/limp）——防跨族撞键
                "family": str(d.get("spot_id") or "").split("-")[0].lower(),
                # 到达路径（审计：深度 2 的 xb/xx 等）
                "reached_by": str(d.get("reached_by") or ""),
                "to_call_bb": _num(d.get("to_call_bb")),
                "bet_ratio": _num(d.get("bet_ratio")),
                "actions": [dict(a) for a in (d.get("actions") or [])],
                # P30-9-EF：逐手频率（动作→数组）与手牌列表透传
                "hands": list(d.get("private_cards_actor") or []),
                "freq_per_hand": dict(d.get("freq_per_hand") or {}),
                "source": str(d.get("source") or ""),
                "source_note": str(d.get("source_note") or ""),
            })
        return bool(self._rows)

    # ---- 查询 ----------------------------------------------------------
    def lookup(self, query: dict) -> dict | None:
        """query: {street, board, pot_bb, node, positions:{oop,ip}, bet_ratio?}

        匹配：纹理 + 位置对 + 节点 + 底池最近（±30%） + facing 时尺寸最近（±35%）。
        无命中 ⇒ None（诚实 miss，绝不跨位置/跨节点套用）。
        """
        if not self._ensure_loaded():
            return None
        street = str(query.get("street") or "flop")
        node = str(query.get("node") or "first_in")
        board = [str(c) for c in (query.get("board") or [])]
        tex = texture_class(board)
        pot = _num(query.get("pot_bb"))
        pos = query.get("positions") or {}
        oop, ip = norm_pos(pos.get("oop")), norm_pos(pos.get("ip"))
        ratio = _num(query.get("bet_ratio"))
        cands = []
        for r in self._rows:
            if r["street"] != street or r["node"] != node:
                continue
            if r["texture"] != tex:
                continue
            # P30-9-EK：行动者必须一致（区分"OOP 过牌后面向"与"IP 下注后面向"；
            # 缺省不约束以保持向后兼容）
            _want_actor = norm_pos(query.get("actor"))
            if _want_actor and r.get("actor") and r["actor"] != _want_actor:
                continue
            # P30-9-FR：翻前家族必须一致（防深层节点底池涨档后跨族撞键）
            _want_fam = str(query.get("family") or "").lower()
            if _want_fam and r.get("family") and r["family"] != _want_fam:
                continue
            if oop and ip:
                if {r["oop"], r["ip"]} != {oop, ip}:
                    continue
            if pot is not None and r["pot_bb"]:
                _rel = abs(r["pot_bb"] - pot) / max(pot, 1e-9)
                if _rel > POT_TOL_REL:
                    # 无严格命中时允许"底池近似"（仅同类节点；≤60%，显式标注）
                    if _rel > POT_TOL_REL_WIDE:
                        continue
                    r = dict(r)
                    r["pot_approximate"] = True
            # 尺寸桶：严格匹配优先；失败时允许"同类（普通/全下）尺寸近似"
            size_approx = False
            if node == "facing_bet" and ratio is not None and r["bet_ratio"]:
                _strict = abs(r["bet_ratio"] - ratio) / max(ratio, 1e-9) <= RATIO_TOL_REL
                if not _strict:
                    _toc = _num(query.get("to_call_bb"))
                    _stk = _num(query.get("stack_bb"))
                    # 全下判定：跟注额 ≥ 90% 有效筹码（栈缺失时退回下注比 ≥ 1.5 的保守判据）
                    _live_allin = ((_toc is not None and _stk and _stk > 0
                                     and _toc >= 0.85 * _stk)
                                    or (ratio >= 1.5))
                    _row_allin = str(r.get("reached_by") or "").startswith("AI")
                    if _live_allin != _row_allin:
                        continue                     # 绝不跨"普通 ↔ 全下"
                    if _live_allin:
                        continue                     # 全下只做严格匹配
                    # P30-9-DW：有界倍数（默认 2×；超界 ⇒ 诚实 miss）
                    import os as _os
                    _F = float(_os.environ.get("SOLVER6_SIZE_FACTOR") or
                                SIZE_APPROX_MAX_FACTOR)
                    if _F > 0:
                        _f = max(ratio, 1e-9) / max(float(r["bet_ratio"]), 1e-9)
                        if _f > _F or _f < (1.0 / _F):
                            continue
                    size_approx = True               # 普通下注之间允许近似（倍数内）
                    r = dict(r)
                    r["size_approximate"] = True
            score = abs((r["pot_bb"] or 0.0) - (pot or 0.0))
            if node == "facing_bet" and ratio is not None and r["bet_ratio"]:
                score += abs(r["bet_ratio"] - ratio) * 10.0
            # P30-9-CN：SPR 维度（仅用于同档内排序与标注；不改变"锚定须精确"纪律）
            _stk_q = _num(query.get("stack_bb"))
            if _stk_q and pot and r.get("spr"):
                _spr_live = _stk_q / pot
                score += abs(r["spr"] - _spr_live) * 0.5
                r = dict(r)
                r["spr_live"] = round(_spr_live, 3)
                r["spr_matched"] = (abs(r["spr"] - _spr_live)
                                     / max(_spr_live, 1e-9) <= SPR_TOL_REL)
            cands.append((score, r))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        # P30-9-FK：**牌面完全一致**优先（否则同纹理桶会返回别的牌面却标 exact ✗）
        _board_strict = str(__import__("os").environ.get("SPOT6_BOARD_STRICT", "1")).lower() \
            not in ("0", "off", "false", "no")
        _same_board = None
        _pot_gate_approx = False
        if _board_strict and board:
            # P30-9-FZ：**牌面规范化**（排序）——真实客户端牌面顺序任意，
            # 不规范化则 3 张牌只有 1/6 排列能精确命中（实测 0.055% vs 理论 0.217%）
            _want_b = tuple(sorted(str(c) for c in board))
            # P30-9-GD：**相对底池容差**——精确匹配此前只按 pot 档（"big" 跨 16–∞bb），
            # 会把底池差数倍的解标成 exact（误导）。要求相对差 ≤ POT_EXACT_TOL_REL，
            # 否则退到桶/共识路径并如实标近似。
            for _s, _r in cands:
                if tuple(sorted(str(c) for c in (_r.get("board") or []))) != _want_b:
                    continue
                _rp = _num(_r.get("pot_bb"))
                if _rp is None or pot is None or pot <= 0:
                    _same_board = _r
                    break
                if abs(_rp - pot) / max(pot, 1e-9) <= POT_EXACT_TOL_REL:
                    _same_board = _r
                    break
                _pot_gate_approx = True        # 牌面对上但底池差太多 ⇒ 不标 exact
        if _same_board is not None:
            best = _same_board
            _board_approx = False
            _consensus = None
        else:
            # P30-9-FT：**桶内共识**——把同纹理桶多个牌面的策略**聚合**成先验，
            # 而非任取一个代表（留出验证显示任取代表的主选一致仅 ~49%）。
            # 仍标近似（它不是该牌面的解）⇒ 弱先验、不锚定 ✓。
            _use_consensus = str(__import__("os").environ.get(
                "SPOT6_CONSENSUS", "1")).lower() not in ("0", "off", "false", "no")
            _consensus = (_consensus_from_rows([r for _s, r in cands])
                           if (_use_consensus and len(cands) > 1) else None)
            best = _consensus or cands[0][1]
            _board_approx = bool(_board_strict and board)
        _pot_approx = bool(best.get("pot_approximate"))
        _size_approx = bool(best.get("size_approximate"))
        _spr_matched = bool(best.get("spr_matched"))
        _board_approx = bool(_board_approx) or bool(_pot_gate_approx)
        _approx = _size_approx or _pot_approx or _board_approx
        return {
            "solved": True,
            "solver_name": "postflop",          # 锚定契约要求
            "model": (f"spot_6max:{best['spot_id']}"
                       + ("（底池近似）" if _pot_approx else "")
                       + ("（尺寸近似）" if _size_approx else "")
                       + ("（SPR 接近）" if (_pot_approx and _spr_matched) else "")
                       + (f"（桶内共识 n={best['consensus_n']}）"
                          if best.get("consensus_n") else "")),
            "node": best["node"],
            "actions": [dict(a) for a in best["actions"]],
            # P30-9-EF：逐手频率透传（与 hands 对齐）
            "hands": list(best.get("hands") or []),
            "freq_per_hand": dict(best.get("freq_per_hand") or {}),
            "bet_sizings_bb": [],
            # 底池/尺寸近似 ⇒ 标 is_approximate=True ⇒ judge 视作弱先验（**不锚定**）
            "is_approximate": _approx,
            "size_approximate": _size_approx,
            "pot_approximate": _pot_approx,
            "board_approximate": _board_approx,
            "spr_matched": _spr_matched,
            "raw": {"spot_id": best["spot_id"], "node": best["node"],
                     "actor": best.get("actor"),
                     "consensus_n": best.get("consensus_n"),
                     "consensus_from": best.get("consensus_from"),
                     "positions": {"oop": best["oop"], "ip": best["ip"]},
                     "pot_bb": best["pot_bb"], "to_call_bb": best["to_call_bb"],
                     "bet_ratio": best["bet_ratio"],
                     "source": best["source"] or "external_gto_json",
                     "source_note": best["source_note"],
                     "size_approximate": _size_approx,
                     "pot_approximate": _pot_approx,
                     "board_approximate": _board_approx,
                     "spr_matched": _spr_matched,
                     "solved_spr": (round(best["spr"], 3)
                                     if best.get("spr") else None),
                     "live_spr": best.get("spr_live"),
                     "solved_bet_ratio": best.get("bet_ratio"),
                     "solved_pot_bb": best.get("pot_bb"),
                     "table": SPOT6_VERSION},
        }

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._rows)


_CACHE: dict[str, SpotTable6max] = {}


def get_spot_table_6max(dir_path: str = DEFAULT_DIR) -> SpotTable6max:
    if dir_path not in _CACHE:
        _CACHE[dir_path] = SpotTable6max(dir_path)
    return _CACHE[dir_path]
