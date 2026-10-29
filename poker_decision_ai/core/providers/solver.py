"""Solver Provider Interface + Mock + real adapter.

Design rule (spec §18): the system must NEVER let an LLM pretend to be a
solver.  GTO baselines come from a solver provider only.

  - MockSolverProvider        : deterministic heuristic baseline, explicitly
                                flagged `is_approximate=True`.  It is NOT GTO
                                and its results carry that label everywhere.
  - PostflopSolverCLIProvider : adapter for the open-source Postflop Solver
                                CLI (https://github.com/b-inary/postflop-solver).
                                Requires `solver.path`; raises
                                ProviderNotConfiguredError otherwise.

A solver query is a normalized dict built by the Solver Agent:
  {board, hole_cards, pot, effective_stack, ranges: {hero: [...], villain: [...]},
   bet_sizings, oop/ip, street, game: "cash"|"tournament"}
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ProviderError, ProviderNotConfiguredError
from .base import Provider


@dataclass
class SolverAction:
    name: str                      # check | bet | call | raise | fold | allin
    size_bb: float | None = None   # bet/raise size in bb (None for checks/calls/folds)
    frequency: float = 0.0         # 0..1 strategy frequency
    ev: float | None = None        # expected value in bb when known

    def to_dict(self) -> dict:
        return {"name": self.name, "size_bb": self.size_bb, "frequency": self.frequency, "ev": self.ev}


@dataclass
class SolverResult:
    solver_name: str
    model: str = ""
    actions: list = field(default_factory=list)          # list[SolverAction]
    bet_sizings_bb: list = field(default_factory=list)   # sizings used by the solve
    notes: str = ""
    is_approximate: bool = True
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "solver_name": self.solver_name, "model": self.model,
            "actions": [a.to_dict() if isinstance(a, SolverAction) else a for a in self.actions],
            "bet_sizings_bb": self.bet_sizings_bb, "notes": self.notes,
            "is_approximate": self.is_approximate, "raw": self.raw,
        }


class SolverProvider(Provider):
    """GTO / solver baseline interface."""

    name = "solver"

    @abstractmethod
    async def solve(self, query: dict) -> SolverResult:
        """query: normalized solver query dict (see module docstring)."""


class MockSolverProvider(SolverProvider):
    """Deterministic heuristic baseline - clearly labelled approximate.

    It produces balanced-looking frequency distributions from equity + stack
    depth so downstream agents (bluff/exploit/judge) have a *baseline*, but
    every result carries is_approximate=True and the Judge treats it as a weak
    prior, never as truth.
    """

    name = "mock"
    is_mock = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    async def solve(self, query: dict) -> SolverResult:
        seed_key = str(sorted(query.get("board") or [])) + str(query.get("hole_cards"))
        rng = random.Random(self.seed ^ hash(seed_key) % (2**32))
        equity = float(query.get("hero_equity", 0.5))
        pot = float(query.get("pot", 1.0))
        eff = float(query.get("effective_stack", 20.0))
        spr = eff / max(pot, 1e-9)
        street = query.get("street", "flop")

        actions: list[SolverAction] = []
        if street == "preflop":
            if query.get("to_call", 0) > 0:
                actions = [SolverAction("call", None, 0.55 + 0.25 * equity), SolverAction("raise", 3.0, 0.35), SolverAction("fold", None, max(0.0, 0.45 - 0.25 * equity))]
            else:
                actions = [SolverAction("raise", 2.5, 0.85), SolverAction("call", None, 0.15)]
        else:
            if query.get("to_call", 0) > 0:
                actions = [
                    SolverAction("call", None, round(min(0.8, max(0.1, equity - 0.1)), 2)),
                    SolverAction("raise", round(0.75 * pot, 1), 0.15),
                    SolverAction("fold", None, round(max(0.0, 0.9 - equity), 2)),
                ]
            else:
                if equity > 0.66 and spr > 3:
                    actions = [SolverAction("bet", round(0.66 * pot, 1), 0.6), SolverAction("bet", round(1.2 * pot, 1), 0.25), SolverAction("check", None, 0.15)]
                elif equity > 0.45:
                    actions = [SolverAction("bet", round(0.33 * pot, 1), 0.35), SolverAction("check", None, 0.65)]
                else:
                    actions = [SolverAction("check", None, 0.9), SolverAction("bet", round(0.33 * pot, 1), 0.1)]
        total = sum(a.frequency for a in actions) or 1.0
        for a in actions:
            a.frequency = round(a.frequency / total, 3)
        return SolverResult(
            solver_name=self.name, model=f"mock-heuristic(seed={self.seed})",
            actions=actions, bet_sizings_bb=[0.33, 0.66, 0.75, 1.2],
            notes="Mock 启发式基线（非真实 GTO 求解，仅作弱先验）", is_approximate=True,
            raw={"equity": equity, "spr": round(spr, 2)},
        )


#: P30-9-AW：单次求解默认超时（秒）。原实现默认 600s 且超时不回收子进程
#: ⇒ 孤儿求解器堆积（实测 7 个 / ~4.5GB）。交互场景取 25s，可用环境变量覆盖。
DEFAULT_SOLVE_TIMEOUT_S = float(os.environ.get("SOLVER_TIMEOUT_S") or 8.0)   # P30-9-BM：见 config 注释


class PostflopSolverCLIProvider(SolverProvider):
    """Adapter for the real Postflop Solver (b-inary/postflop-solver).

    SOLVER-1：可执行文件 = `postflop-bench-cli`（Rust CLI，封装
    b-inary/postflop-solver crate 的 CFR 求解）。协议：
      stdin  = JSON 查询（street/board/oop_range/ip_range/pot_bb/
               effective_stack_bb/bet_sizes/raise_sizes/max_iterations/
               target_exploitability_pot_pct）
      stdout = JSON 结果（solved/actions[{name,amount_bb,frequency,ev_bb}]/
               exploitability_bb/simplifications/...）

    配置 `solver.provider=postflop` + `solver.path` 指向该 CLI。未配置时
    solve() 抛 ProviderNotConfiguredError（管线诚实降级，绝不伪造求解）。

    诚实边界（随每次结果的 raw.simplifications 传播）：
      heads-up；下注尺寸抽象；树从当前街起（无 future-game 残差）；
      迭代/exploitability 停止规则；preflop 不在覆盖范围（用 chart 基线）。
    """

    name = "postflop"
    is_mock = False

    def __init__(self, executable: str = "") -> None:
        self.executable = executable or os.environ.get("POSTFLOP_SOLVER_PATH", "")

    def is_configured(self) -> bool:
        return bool(self.executable) and Path(self.executable).exists()

    def status_reason(self) -> str:
        if self.is_configured():
            return ""
        if not self.executable:
            return "solver executable path not set (solver.path / POSTFLOP_SOLVER_PATH)"
        return f"solver executable not found: {self.executable}"

    def build_cli_query(self, query: dict) -> dict:
        """归一化查询 → bench-cli JSON 查询。

        * pot/effective_stack/bb：query 里是筹码口径 → /bb = bb 口径；
        * ranges：{oop, ip} 字符串（postflop 记法）——显式要求，缺失即报错
          （不伪造范围）；
        * bet_sizes/raise_sizes：BetSizeOptions 记法字符串（默认 60%,e,a / 2.5x）。
        """
        street = str(query.get("street") or "")
        if street == "preflop":
            raise ProviderError(
                "postflop solver does not cover preflop; use the preflop "
                "chart baseline (source=chart) instead")
        if street not in ("flop", "turn", "river"):
            raise ProviderError(f"unsupported street for postflop solve: {street!r}")
        ranges = query.get("ranges") or {}
        oop_range = str(ranges.get("oop") or "")
        ip_range = str(ranges.get("ip") or "")
        if not oop_range or not ip_range:
            raise ProviderError(
                "postflop solve requires ranges {oop, ip} range strings")
        bb = float(query.get("bb") or 0.0) or None
        pot = query.get("pot")
        eff = query.get("effective_stack")
        pot_bb = (float(pot) / bb) if (bb and pot is not None) else float(pot or 0.0)
        eff_bb = (float(eff) / bb) if (bb and eff is not None) else float(eff or 0.0)
        return {
            "street": street,
            "board": [str(c) for c in query.get("board", [])],
            "oop_range": oop_range,
            "ip_range": ip_range,
            "pot_bb": round(pot_bb, 4),
            "effective_stack_bb": round(eff_bb, 4),
            "bet_sizes": str(query.get("bet_sizes_oop") or "60%,e,a"),
            "bet_sizes_ip": str(query.get("bet_sizes_ip") or "60%,e,a"),
            "raise_sizes": str(query.get("raise_sizes") or "2.5x"),
            "max_iterations": int(query.get("max_iterations", 1000)),
            "target_exploitability_pot_pct": float(
                query.get("target_exploitability_pot_pct", 0.5)),
        }

    # postflop Debug 名 → 生产动作名 + 是否带尺寸
    _ACTION_KINDS: dict = {
        "Check": ("check", False), "Bet": ("bet", True),
        "Call": ("call", False), "Fold": ("fold", False),
        "Raise": ("raise", True), "AllIn": ("allin", True),
    }

    async def solve(self, query: dict) -> SolverResult:
        if not self.is_configured():
            raise ProviderNotConfiguredError(
                "Postflop Solver CLI not configured. Install it and set solver.path "
                "(or POSTFLOP_SOLVER_PATH); otherwise use solver.provider=mock.")
        cli_query = self.build_cli_query(query)
        payload = json.dumps(cli_query).encode("utf-8")
        # P30-9-AW：默认超时 600s → 交互可用值（可用 SOLVER_TIMEOUT_S 覆盖）
        timeout_s = float(query.get("timeout_s", DEFAULT_SOLVE_TIMEOUT_S))
        proc = None
        try:
            # P30-9-GS：**后台/空闲求解**用最低进程优先级 —— 不抢客户端 CPU ✓
            #   现场解缓存这类"可以慢、不能卡"的求解：SOLVER_CLI_IDLE_PRIORITY=1，
            #   或调用方在 query 里给 idle_priority=True。
            _idle = bool(query.get("idle_priority")) or (
                str(os.environ.get("SOLVER_CLI_IDLE_PRIORITY", "0")).lower()
                not in ("0", "", "false", "no"))
            _flags = 0x00000040 if (_idle and os.name == "nt") else 0   # IDLE_PRIORITY_CLASS
            proc = await asyncio.create_subprocess_exec(
                self.executable,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **({"creationflags": _flags} if _flags else {}),
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(payload), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                raise ProviderError(
                    f"postflop solver timed out after {timeout_s}s") from e
        except FileNotFoundError as e:
            raise ProviderNotConfiguredError(
                f"solver executable not found: {self.executable}") from e
        finally:
            # P30-9-AW：**任何路径**都要终止并回收子进程，避免孤儿（曾堆积 ~4.5GB）
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:                      # noqa: BLE001
                    pass
                try:
                    await proc.wait()
                except Exception:                      # noqa: BLE001
                    pass
                self.last_killed = True
        if proc.returncode != 0:
            raise ProviderError(
                "postflop solver failed: "
                + err.decode(errors="replace")[:400])
        try:
            data = json.loads(out.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ProviderError(f"malformed solver output: {e}") from e
        if not data.get("solved"):
            raise ProviderError("postflop solver returned solved=false")

        actions: list[SolverAction] = []
        for row in data.get("actions") or []:
            raw_name = str(row.get("name") or "")
            kind = self._ACTION_KINDS.get(raw_name)
            if kind is None:
                raise ProviderError(f"unknown solver action name: {raw_name!r}")
            name, has_amount = kind
            amount_bb = row.get("amount_bb") if has_amount else None
            actions.append(SolverAction(
                name=name,
                size_bb=(float(amount_bb) if amount_bb is not None else None),
                frequency=round(float(row.get("frequency") or 0.0), 6),
                ev=(round(float(row["ev_bb"]), 6)
                    if row.get("ev_bb") is not None else None),
            ))

        simplifications = list(data.get("simplifications") or [])
        return SolverResult(
            solver_name=self.name,
            model="postflop-solver(b-inary) via postflop-bench-cli",
            actions=actions,
            bet_sizings_bb=sorted({a.size_bb for a in actions
                                   if a.size_bb is not None}),
            notes=("真实 postflop CFR 求解（HU；尺寸抽象；从当前街起树；"
                   "无 future-game 残差；迭代/exploitability 停止规则）"),
            is_approximate=False,
            raw={
                "source": "postflop",
                "near_nash_equilibrium": True,           # NFNE 近似均衡（含上述抽象）
                "is_mock": False,
                "exploitability_bb": data.get("exploitability_bb"),
                "exploitability_chips": data.get("exploitability_chips"),
                "iterations": data.get("iterations"),
                "target_exploitability_chips":
                    data.get("target_exploitability_chips"),
                "n_private_hands_actor": data.get("n_private_hands_actor"),
                "ev_avg_bb": data.get("ev_avg_bb"),
                "equity_avg_pct": data.get("equity_avg"),
                "simplifications": simplifications,
            },
        )


# ─────────────────────── P61 — 环境发现 + 输出解析契约 ───────────────────────
def discover_solver_environment(parquet_dir: str | None = None) -> dict:
    """审计当前环境是否存在可真实调用的 Postflop Solver（只读；确定性）。

    检查通道（各自独立报告，绝不伪造）：
      executable_path : POSTFLOP_SOLVER_PATH / solver.path 指向的可执行文件
      path_search     : PATH 上的 postflop-solver(.exe)
      pip_package     : `import postflop_solver`（官方 Python 绑定）
      sample_outputs  : 仓库内的真实 solver 输出样例（当前无）

    返回 {"available": bool, "channels": {...}, "blocked_reason": str,
    "checked_at_contract": "P61"}——available=True 需至少一个通道真实命中。
    """
    channels: dict[str, dict] = {}
    exe = os.environ.get("POSTFLOP_SOLVER_PATH", "") or parquet_dir or ""
    if exe and Path(exe).exists():
        channels["executable_path"] = {"found": True, "path": exe}
    else:
        channels["executable_path"] = {
            "found": False,
            "reason": (f"POSTFLOP_SOLVER_PATH set but missing: {exe}"
                       if exe else "POSTFLOP_SOLVER_PATH not set")}
    found = None
    for name in ("postflop-solver.exe", "postflop-solver",
                 "postflop_solver.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(d) / name
            if cand.exists():
                found = str(cand)
                break
        if found:
            break
    channels["path_search"] = {"found": bool(found), "path": found}
    try:
        import importlib
        importlib.import_module("postflop_solver")
        channels["pip_package"] = {"found": True, "package": "postflop_solver"}
    except ImportError:
        channels["pip_package"] = {"found": False,
                                   "reason": "postflop_solver not installed"}
    channels["sample_outputs"] = {"found": False,
                                  "reason": "no real solver output samples "
                                            "in repository"}
    available = any(c.get("found") for c in channels.values())
    return {"available": available,
            "channels": channels,
            "blocked_reason": ("" if available else
                               "no real solver found on any discovery "
                               "channel (P35 BLOCKED persists; install the "
                               "open-source postflop-solver and set "
                               "POSTFLOP_SOLVER_PATH to integrate)"),
            "contract": "P61"}


def parse_solver_strategy_output(text: str) -> dict:
    """严格解析 solver 输出（P61 契约草案；malformed → ProviderError）。

    契约（草案，未来与真实 solver 样本对齐后冻结）：JSON object，含
    "strategy": {action: frequency(0..1)}，频率和 ≤ 1+1e-6。
    注意：本函数只做结构校验——真实语义对齐必须待真实样本（不得伪造）。
    """
    import json
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ProviderError(f"malformed solver output: not JSON ({e})") from e
    if not isinstance(data, dict):
        raise ProviderError("malformed solver output: not a JSON object")
    strategy = data.get("strategy")
    if not isinstance(strategy, dict) or not strategy:
        raise ProviderError("malformed solver output: missing 'strategy' map")
    total = 0.0
    for action, freq in strategy.items():
        if not isinstance(freq, (int, float)) or not 0.0 <= float(freq) <= 1.0:
            raise ProviderError(
                f"malformed solver output: frequency for {action!r} "
                f"out of range: {freq!r}")
        total += float(freq)
    if total > 1.0 + 1e-6:
        raise ProviderError(
            f"malformed solver output: strategy frequencies sum {total} > 1")
    out = {"strategy": dict(strategy), "contract": "p61-draft"}
    if isinstance(data.get("ev"), dict):
        out["ev"] = data["ev"]
    return out
