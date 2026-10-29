"""EXE Preview GUI 源码级冒烟（真实 Tk 事件循环；窗口 withdraw 不弹前台）。

验证（授权任务验收点）：
  1. 单次点击 → 恰好一次 pipeline → 结果显示；
  2. 连续点击两次 → 分别执行两次（两次独立完整链）；
  3. 无 CoinPoker 窗口时的真实捕获路径 → 诚实 UNAVAILABLE（不猜值）。

运行：& .venv\\Scripts\\python.exe scripts\\p33_gui_smoke.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk

from cash_game_ui.preview_gui import PreviewGUI

FRAME = "data/vision2/coinpoker_evidence/8bf82f134651db0add626be0304a2ff1.png"
LEGAL_ACTIONS = {"fold", "check", "call", "bet", "raise", "allin"}


def pump_until(gui: PreviewGUI, cond, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        gui.root.update()          # 处理 after 事件泵（结果队列）
        if cond():
            return True
        time.sleep(0.02)
    return False


def settle(gui: PreviewGUI, seconds: float = 0.4) -> None:
    """断言前沉降：再泵事件循环，让日志队列刷新到文本框。"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        gui.root.update()
        time.sleep(0.02)


def log_text(gui: PreviewGUI) -> str:
    return gui.log_txt.get("1.0", "end")


def main() -> int:
    root = tk.Tk()
    root.withdraw()                # 冒烟不弹前台
    ok = True

    # ── 场景 1：--frame 模式，单次点击 = 一次完整链 ──────────────────
    gui = PreviewGUI(root, frame_path=FRAME)
    gui.on_collect()
    ok1 = pump_until(gui, lambda: gui._executed >= 1, 120)
    settle(gui)
    txt = log_text(gui)
    ok1 = ok1 and "▶ 点击 #1" in txt and "✔ #1" in txt and "status=OK" in txt
    action_lbl = str(gui.action_lbl.cget("text")).upper()
    # 断言"合法动作"（与项目测试 test_cash_game_ui_runner 同口径——
    # P30 决策链对本帧允许 fold/call 等，不锁定具体动作）
    ok1 = ok1 and action_lbl in {a.upper() for a in LEGAL_ACTIONS}
    print(f"[1] 单次点击→一次执行: {'PASS' if ok1 else 'FAIL'} "
          f"(executed={gui._executed}, action_label={action_lbl})")
    ok = ok and ok1

    # ── 场景 2：连续点击两次 → 分别执行两次 ─────────────────────────
    before = gui._executed
    gui.on_collect()               # 连续两次（中间不等待）
    gui.on_collect()
    ok2 = pump_until(gui, lambda: gui._executed >= before + 2, 120)
    settle(gui)
    txt = log_text(gui)
    ok2 = ok2 and txt.count("✔ #") >= before + 2
    ok2 = ok2 and f"▶ 点击 #{before + 1}" in txt and f"▶ 点击 #{before + 2}" in txt
    print(f"[2] 连续两次点击→两次执行: {'PASS' if ok2 else 'FAIL'} "
          f"(executed {before}→{gui._executed}, 两次入队 #"
          f"{before + 1}/#{before + 2})")
    ok = ok and ok2

    # ── 场景 3：真实捕获路径，无 CoinPoker → 诚实 UNAVAILABLE ────────
    gui3 = PreviewGUI(root, frame_path=None)
    gui3.on_collect()
    ok3 = pump_until(gui3, lambda: gui3._executed >= 1, 120)
    settle(gui3)
    txt3 = log_text(gui3)
    ok3 = ok3 and ("UNAVAILABLE" in txt3 or "ERROR" in txt3)
    print(f"[3] 无牌桌窗口→诚实 UNAVAILABLE: {'PASS' if ok3 else 'FAIL'} "
          f"(log 含 UNAVAILABLE/ERROR)")
    ok = ok and ok3

    # 排队上限护栏（不触发 pipeline，仅日志）
    for _ in range(MAXP := 5):
        gui3.on_collect()
    settle(gui3)
    ok4 = "队列已满" in log_text(gui3)
    print(f"[4] 排队上限护栏: {'PASS' if ok4 else 'FAIL'}")
    ok = ok and ok4

    root.destroy()
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
