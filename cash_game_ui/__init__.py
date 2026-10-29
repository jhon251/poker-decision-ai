"""Cash-Game AI — UI package。

R13-EXE-2：app（Tkinter）不再在包导入期加载——产品 CLI 构建排除
tkinter，`analyze --screenshot` 只需 runner（纯逻辑）。GUI 入口
（CashGameApp/main）经 __getattr__ 惰性转发；行为对既有导入方
`from cash_game_ui import CashGameApp, main` 完全兼容。
"""
from __future__ import annotations

from .runner import (CashGameRunner, AnalysisResult,  # noqa: F401
                     AgentOutput)

__all__ = ["CashGameRunner", "AnalysisResult", "AgentOutput",
           "CashGameApp", "main"]


def __getattr__(name: str):
    if name in ("CashGameApp", "main"):
        from . import app as _app
        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
