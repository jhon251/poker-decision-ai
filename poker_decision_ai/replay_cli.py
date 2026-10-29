"""Replay CLI wrapper: .\\.venv\\Scripts\\python.exe -m poker_decision_ai.replay_cli <dir>"""
from __future__ import annotations

import asyncio
import sys

from .cli import main


def run() -> int:
    sys.argv = [sys.argv[0], "replay"] + sys.argv[1:]
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
