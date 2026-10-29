"""poker_decision_ai.play — EXE-1 产品对打模式（非实时辅助；纯本地）。

组成：
  match    — 6-max 交互对打回路（引擎调用 external_harness，零语义复制）
  history  — play_history JSONL 落盘/加载（训练数据底座）
  console  — 终端渲染 + 动作输入解析（CLI 前端）

红线：无屏幕捕捉/自动点击路径（非实时辅助）；诚实 provenance 随每手落盘。
"""
from __future__ import annotations

PLAY_MODE_VERSION = "exe1-play-v1"
