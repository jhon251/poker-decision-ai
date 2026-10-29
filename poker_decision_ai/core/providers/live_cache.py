"""P30-9-GI：**现场解缓存**（异步求解 flop 并落盘为 6-max 记录，供后续精确命中）。

动机（实测）：
  * flop 现场求解约 **7.5 s**（缩尺寸菜单反而超时；极简菜单虽 0.3s 但策略退化）✗
    ⇒ 无法进实时路径；
  * 但真实使用中**同一局面被反复分析**（日志里 `Qh7d2c` 出现 19 次）✓
    ⇒ "解一次、之后精确命中"收益极大 ✓。

设计：
  * 缓存目录 `data/solver/spots_6max_livecache/`（env `SPOT6_LIVE_CACHE_DIR` 可改）；
  * 6-max 表**同时加载**该目录（按 mtime 变化自动重载 ✓）；
  * agent 在"两人池 + 有声明范围 + flop 近似命中"时**后台**调度一次求解（同进程至多 1 个在飞），
    完成后写入缓存记录（schema 与 6-max 表一致，`source=live_cache` 如实标注）；
  * 决策**不等待**求解结果（立即返回既有近似先验 ✓，绝不阻塞）。
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DIR = "data/solver/spots_6max_livecache"

_TASKS: dict[str, asyncio.Task] = {}


def cache_dir() -> pathlib.Path:
    p = pathlib.Path(os.environ.get("SPOT6_LIVE_CACHE_DIR") or DEFAULT_DIR)
    return p if p.is_absolute() else (ROOT / p)


def enabled() -> bool:
    return str(os.environ.get("SOLVER_LIVE_CACHE", "1")).lower() not in (
        "0", "off", "false", "no")


def resolve_ranges(street: str, pot_bb: float,
                    family: str | None = None) -> tuple[str, str] | None:
    """解析可用范围：先按 (street, pot 档)，再按家族（3bp/srp）回退。

    起因（实测）：40bb 的 3-bet 池 _pot_class 落在 "big"，而 r1/fb 套件的档只有
    srp/3bp ⇒ 直接按 pot 档取会得到 None，导致现场求解/缓存被静默跳过 ✗。
    """
    try:
        from .spot_table import _pot_class, get_spot_table
        st = get_spot_table()
        for pcls in (_pot_class(float(pot_bb)),
                      ("3bp" if str(family or "").lower().startswith("3") else "srp"),
                      "3bp", "srp"):
            dr = st.declared_ranges(str(street), pcls)
            if dr:
                return (dr[0], dr[1])
    except Exception:                                          # noqa: BLE001
        return None
    return None


def spot_id_for(*, family: str, oop: str, ip: str, street: str, board: list[str],
                node: str = "first_in", size_pct: int | None = None) -> str:
    """缓存记录 id —— **必须含位置对**（P30-9-GR ✓）与**节点/尺寸**（P30-9-HF ✓）。

    facing 节点必须带尺寸档：同一牌面、不同下注尺寸是**不同的节点策略** ✗。
    """
    tag = "" if node == "first_in" else f"-{node[:2]}{size_pct or 0}"
    return (f"{family}-livecache-{street}-"
            f"{str(oop).upper()}{str(ip).upper()}{tag}-{''.join(str(c) for c in board)}")


def to_6max_record(result: Any, *, family: str, oop: str, ip: str, street: str,
                    board: list[str], pot_bb: float, stack_bb: float,
                    ranges: dict | None = None) -> dict:
    """把 SolverResult 转成 6-max 表 schema 记录（字段与既有记录一致）。"""
    actions = []
    for a in (getattr(result, "actions", None) or []):
        name = str(getattr(a, "name", "") or "").lower()
        if not name:
            continue
        actions.append({"name": name,
                         "size_bb": (round(float(a.size_bb), 4)
                                      if getattr(a, "size_bb", None) is not None else None),
                         "frequency": round(float(getattr(a, "frequency", 0.0) or 0.0), 4)})
    return {
        "spot_id": spot_id_for(family=family, oop=oop, ip=ip, street=street, board=board),
        "street": street, "board": [str(c) for c in board],
        "pot_bb": round(float(pot_bb), 3), "effective_stack_bb": round(float(stack_bb), 3),
        "positions": {"oop": oop.upper(), "ip": ip.upper()},
        "actor_position": oop.upper(), "node": "first_in",
        "to_call_bb": None, "bet_ratio": None,
        "actions": actions, "private_cards_actor": [], "freq_per_hand": {},
        "solved": True, "source": "live_cache",
        "source_note": ("现场求解缓存（provider live solve；范围=套件声明范围，"
                         "范围假设如实标注）"),
        "agent": "live_cache_v1", "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


async def _solve_and_store(provider: Any, *, family: str, oop: str, ip: str, street: str,
                            board: list[str], pot_bb: float, stack_bb: float,
                            ranges: dict, timeout_s: float, max_iterations: int) -> None:
    key = spot_id_for(family=family, oop=oop, ip=ip, street=street, board=board)
    try:
        q = {"street": street, "board": [str(c) for c in board],
              "ranges": {"oop": ranges.get("oop"), "ip": ranges.get("ip")},
              "pot": float(pot_bb), "effective_stack": float(stack_bb), "bb": 1.0,
              # P30-9-GS：后台解用最低进程优先级 ⇒ 不抢客户端 CPU ✓
              "idle_priority": True,
              "max_iterations": int(max_iterations),
              "target_exploitability_pot_pct": 1.0}
        res = await asyncio.wait_for(provider.solve(q), timeout=timeout_s)
        rec = to_6max_record(res, family=family, oop=oop, ip=ip, street=street,
                              board=board, pot_bb=pot_bb, stack_bb=stack_bb, ranges=ranges)
        if not rec["actions"]:
            return
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        return                                                 # 静默失败：绝不阻塞决策
    finally:
        _TASKS.pop(key, None)


def schedule(*, provider: Any, family: str, oop: str, ip: str, street: str,
             board: list[str], pot_bb: float, stack_bb: float, ranges: dict | None,
             timeout_s: float = 3600.0, max_iterations: int = 300) -> str | None:
    """后台调度一次现场求解（同进程至多一个在飞；已缓存则跳过）。返回 spot_id 或 None。"""
    if not enabled() or not ranges:
        return None
    if street not in ("flop", "turn", "river"):
        return None
    key = spot_id_for(family=family, oop=oop, ip=ip, street=street, board=board)
    if (cache_dir() / f"{key}.json").is_file() or key in _TASKS:
        return None
    if _TASKS:                       # 至多 1 个在飞 ⇒ 不抢占决策 CPU
        return None
    try:
        _TASKS[key] = asyncio.get_running_loop().create_task(
            _solve_and_store(provider, family=family, oop=oop, ip=ip, street=street,
                              board=board, pot_bb=pot_bb, stack_bb=stack_bb,
                              ranges=ranges, timeout_s=timeout_s,
                              max_iterations=max_iterations))
    except RuntimeError:
        return None
    return key


# ── P30-9-HF：**facing 节点**的"遇到即解"（含实际下注尺寸 ✓）───────────────
async def solve_facing_now(*, oop: str, ip: str, street: str, board: list[str],
                            pot_bb: float, stack_bb: float, ranges: dict,
                            bet_ratio: float, timeout_s: float = 2.0,
                            max_iterations: int = 300) -> dict | None:
    """P30-9-HG：**同步**解出 facing 节点（turn/river 实测 ~50ms ⇒ 可进实时 ✓）。

    为什么需要：现场真解此前只返回**街根节点**（OOP 首行动 ✗），而真实 turn/river 里
    hero 常是**面对下注** ✗ ⇒ 该情形拿不到真解。做法：直接调 CLI（**采纳 `bet_sizes`** ✓，
    P30-9-HB 实测）并读 `node_exports`（含 `actor/bet_ratio/to_call_bb/actions` ✓）。
    失败一律返回 None（调用方诚实降级 ✓）。
    """
    if not ranges or street not in ("flop", "turn", "river"):
        return None
    pct = max(5, min(200, int(round(float(bet_ratio) * 100))))
    cli = ROOT / "tools" / "bin" / "postflop-bench-cli.exe"
    if not cli.is_file():
        return None
    q = {"street": street, "board": [str(c) for c in board], "pot_bb": float(pot_bb),
          "effective_stack_bb": float(stack_bb),
          "oop_range": ranges.get("oop"), "ip_range": ranges.get("ip"),
          "bet_sizes": f"{pct}%,a", "bet_sizes_ip": f"{pct}%,a",
          "export_nodes": [[0, 1]], "max_iterations": int(max_iterations),
          "target_exploitability_pot_pct": 1.0}
    t0 = time.perf_counter()
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000040          # IDLE 优先级（不抢客户端 ✓）
        proc = await asyncio.create_subprocess_exec(
            str(cli), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, **kwargs)
        out, _ = await asyncio.wait_for(proc.communicate(json.dumps(q).encode("utf-8")),
                                          timeout=timeout_s)
        rec = json.loads(out.decode("utf-8", "replace"))
        ex = next((e for e in (rec.get("node_exports") or [])
                    if str(e.get("actor") or "").lower() == "oop"), None)
        if ex is None:
            return None
        actions = [{"name": str(a.get("name") or "").lower(), "size_bb": a.get("amount_bb"),
                     "frequency": round(float(a.get("frequency") or 0), 4),
                     "ev": a.get("ev_bb")} for a in (ex.get("actions") or [])]
        if not actions:
            return None
        return {"actions": actions, "bet_ratio": ex.get("bet_ratio"),
                 "to_call_bb": ex.get("to_call_bb"), "pot_bb": ex.get("pot_bb"),
                 "ms": round((time.perf_counter() - t0) * 1000, 1), "size_pct": pct}
    except Exception:                                          # noqa: BLE001
        return None


def _facing_key(*, family: str, oop: str, ip: str, street: str, board: list[str],
                 pct: int) -> str:
    return spot_id_for(family=family, oop=oop, ip=ip, street=street, board=board,
                        node="facing_bet", size_pct=pct)


def schedule_facing(*, family: str, oop: str, ip: str, street: str, board: list[str],
                     pot_bb: float, stack_bb: float, ranges: dict | None,
                     bet_ratio: float, to_call_bb: float | None = None,
                     timeout_s: float = 3600.0, max_iterations: int = 300,
                     solver_threads: int = 2) -> str | None:
    """facing 节点的后台调度（**按该局面真实下注比例**构造菜单 ✓）。

    背景（P30-9-HB 实测）：CLI **采纳 `bet_sizes`** ✓ ⇒ 可解出与实际下注尺寸一致的
    facing 节点（导出 `history=[0,1]`、`actor=oop` ✓），落盘为 6-max facing 记录 ✓。
    决策**不等待** ✓（立即返回既有近似先验 ✓）。
    """
    if not enabled() or not ranges or street not in ("flop", "turn", "river"):
        return None
    pct = max(5, min(200, int(round(float(bet_ratio) * 100))))
    key = _facing_key(family=family, oop=oop, ip=ip, street=street, board=board, pct=pct)
    if (cache_dir() / f"{key}.json").is_file() or key in _TASKS or _TASKS:
        return None
    try:
        _TASKS[key] = asyncio.get_running_loop().create_task(
            _facing_solve_and_store(family=family, oop=oop, ip=ip, street=street,
                                     board=board, pot_bb=pot_bb, stack_bb=stack_bb,
                                     ranges=ranges, pct=pct, to_call_bb=to_call_bb,
                                     timeout_s=timeout_s, max_iterations=max_iterations,
                                     solver_threads=solver_threads))
    except RuntimeError:
        return None
    return key


async def _facing_solve_and_store(*, family: str, oop: str, ip: str, street: str,
                                   board: list[str], pot_bb: float, stack_bb: float,
                                   ranges: dict, pct: int, to_call_bb: float | None,
                                   timeout_s: float, max_iterations: int,
                                   solver_threads: int) -> None:
    """跑 `solve_spots_fb.py`（它向 CLI 传 `export_nodes` ✓）并把导出节点写成记录 ✓。"""
    key = _facing_key(family=family, oop=oop, ip=ip, street=street, board=board, pct=pct)
    tmp = ROOT / ".tmp" / "livecache_facing"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        cfg = tmp / f"{key}.yaml"
        outd = tmp / key
        cfg.write_text(json.dumps({"spot_suite": {
            "name": f"live-facing-{key}", "cli": "tools/bin/postflop-bench-cli.exe",
            "solver": "postflop-solver", "n_spots": 1},
            "spots": [{"spot_id": key, "street": street, "board": [str(c) for c in board],
                        "pot_bb": float(pot_bb), "effective_stack_bb": float(stack_bb),
                        "ranges": {"oop": ranges.get("oop"), "ip": ranges.get("ip")},
                        "bet_sizes": f"{pct}%,a", "bet_sizes_ip": f"{pct}%,a",
                        "export_nodes": [[0, 1]],
                        "exports": [{"export_id": "facing", "path": [0, 1]}],
                        "max_iterations": int(max_iterations),
                        "target_exploitability_pot_pct": 1.0}]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000040          # IDLE_PRIORITY_CLASS ✓ 不抢客户端
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "scripts" / "solve_spots_fb.py"),
            "--config", str(cfg), "--out-dir", str(outd),
            "--workers", "1", "--solver-threads", str(solver_threads),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            cwd=str(ROOT), **kwargs)
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        rec_file = outd / f"{key}.json"
        if not rec_file.is_file():
            return
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        ex = next((e for e in (rec.get("exports") or [])
                    if str(e.get("actor") or "").lower() == "oop"), None)
        if ex is None:
            return
        actions = [{"name": str(a.get("name") or "").lower(), "size_bb": a.get("amount_bb"),
                     "frequency": round(float(a.get("frequency") or 0), 4)}
                    for a in (ex.get("actions") or [])]
        if not actions:
            return
        ratio = ex.get("bet_ratio")
        out = {"spot_id": key, "street": street, "board": [str(c) for c in board],
                "pot_bb": round(float(ex.get("pot_bb") or pot_bb), 3),
                "effective_stack_bb": round(float(stack_bb), 3),
                "positions": {"oop": str(oop).upper(), "ip": str(ip).upper()},
                "actor_position": str(oop).upper(), "node": "facing_bet",
                "to_call_bb": ex.get("to_call_bb"), "bet_ratio": ratio,
                "actions": actions, "private_cards_actor": [], "freq_per_hand": {},
                "solved": True, "source": "live_cache_facing",
                "source_note": (f"现场 facing 解缓存（按实际比例构造菜单 {pct}%；"
                                 "范围=套件声明范围，假设如实标注）"),
                "agent": "live_cache_facing_v1"}
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        return                                                 # 静默失败：绝不阻塞决策
    finally:
        _TASKS.pop(key, None)
