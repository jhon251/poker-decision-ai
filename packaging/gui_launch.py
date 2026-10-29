"""EXE Preview GUI 启动器（仅打包用）。

职责：
  1. frozen 模式下 chdir 到 EXE 目录（configs/logs/data 相对路径解析）；
  2. 清除可能污染冻结环境的 PYTHONPATH/PYTHONHOME；
  3. stderr 诚实横幅（stdout 不承载 JSON——GUI 模式）；
  4. 可选 --frame <png>：预览测试开关——把「捕获」一步替换为读取指定 PNG
     （仍是每次点击执行一次完整链；默认为真实 BitBlt 屏幕捕获）；
  5. 启动 Tk 常驻窗口 cash_game_ui.preview_gui.PreviewGUI。

红线：不自动下注/点击牌桌/鼠标键盘控制；无定时采集；不伪造 REAL。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _banner() -> None:
    print(
        "[poker-ai EXE Preview GUI]\n"
        "  - Preview ≠ Final Product；最小常驻 GUI 层。\n"
        "  - 一次点击 = 一次 Capture → Vision → Decision；无定时轮询/自动采集。\n"
        "  - 不自动下注/点击牌桌；只输出建议（UNKNOWN-first；BASELINE ≠ REAL）。\n"
        "  - LLM/Vision/Solver 未配置 REAL 时如实显示 mock/unavailable。\n"
        "  - configs/ 与 data/、logs/ 位于本 EXE 旁边，可直接编辑。",
        file=sys.stderr,
    )


def main() -> int:
    if getattr(sys, "frozen", False):
        os.chdir(_app_dir())
        for var in ("PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(var, None)
    _banner()

    ap = argparse.ArgumentParser(description="poker-ai EXE Preview GUI")
    ap.add_argument("--frame", default=None,
                    help="预览测试开关：用指定 PNG 代替实时屏幕捕获"
                         "（每次点击仍执行一次完整链）")
    ap.add_argument("--selftest", action="store_true",
                    help="诊断开关：无窗口执行一次完整链（捕获/读取 → Vision → "
                         "Decision），结果 JSON 打印到 stdout 后退出")
    args = ap.parse_args()

    if args.selftest:
        import json

        from cash_game_ui.runner import CashGameRunner
        from cash_game_ui.preview_gui import capture_frame_smart

        runner = CashGameRunner()
        if args.frame:
            from poker_decision_ai.vision2_platform.png_decode import decode_png
            w, h, rgba = decode_png(open(args.frame, "rb").read())
            d = runner.analyze(rgba, w, h).to_dict()
        else:
            rgba, w, h = capture_frame_smart()
            if rgba is None:
                d = {"status": "UNAVAILABLE",
                     "reason": "capture_failed:no_visible_coinpoker_table",
                     "decision_action": None, "solver_mode": "UNKNOWN"}
            else:
                d = runner.analyze(rgba, w, h).to_dict()
        slim = {k: d.get(k) for k in
                ("status", "decision_action", "confidence", "solver_mode",
                 "strategy_mode", "reason", "blocking_fields")}
        vis = d.get("vision") or {}
        slim["vision"] = {k: vis.get(k) for k in
                          ("hero_cards", "hero_seat", "dealer_seat", "pot",
                           "to_call", "board", "street", "position")}
        slim["agents"] = [a.get("agent") for a in d.get("agents") or []]
        stages = d.get("gui_stages")
        if stages:
            slim["gui_stages"] = stages
        print(json.dumps(slim, ensure_ascii=False, indent=1))
        return 0

    import tkinter as tk

    from cash_game_ui.preview_gui import PreviewGUI

    root = tk.Tk()
    PreviewGUI(root, frame_path=args.frame)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
