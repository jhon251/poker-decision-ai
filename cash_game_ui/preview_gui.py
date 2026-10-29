"""Cash-Game AI — EXE Preview 最小常驻 GUI（Tkinter，零新依赖）。

交互契约（授权任务——EXE Preview 之上的最小 GUI 常驻层）：
  - 启动后常驻（Tk mainloop）；前台窗口展示日志 + 最近一次分析结果；
  - 「采集窗口」按钮 = 用户主动触发：一次点击 → 恰好一次
    Capture → Vision → Decision Pipeline（复用 CashGameRunner 一键链）；
  - 无定时轮询 / 无后台自动采集：主线程 after() 仅作「结果队列事件泵」，
    绝不触发采集；采集只由按钮点击入队；
  - 连续点击可排队（上限 3），单 worker 线程串行执行 —— 1 点击 = 1 次执行；
  - GUI 只启动/触发/展示：不重新实现 Vision/Decision（runner 是唯一链）；
  - 结果原样展示 status / decision_action / solver_mode / strategy_mode /
    reason / blocking_fields（UNKNOWN-first；BASELINE 不冒充 REAL）；
  - 无自动下注 / 点击牌桌 / 鼠标键盘控制。

测试支持：--frame <png>（预览测试开关）把「捕获」一步替换为读取指定 PNG
（与单测同一 decode_png），仍是每次点击执行一次完整链。默认（无 --frame）
为真实 BitBlt 屏幕捕获；找不到 CoinPoker 窗口 → 诚实 UNAVAILABLE。
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

try:
    from .runner import CashGameRunner
except ImportError:  # PyInstaller onefile 兼容
    from cash_game_ui.runner import CashGameRunner

MAX_PENDING = 3          # 排队上限（防堆积；每次点击仍是 1 次执行）
RUN_TIMEOUT_S = 45       # 单次运行看门狗（超时转储线程栈到 logs/，便于定位卡死）
DRAIN_MS = 120           # 结果队列事件泵周期（只搬运结果，绝不采集）


def find_table_window():
    """只读枚举：标题含 CoinPoker 的可见窗口中筛选"像牌桌"的候选。

    背景：CoinPoker 客户端有一个 159×27、位于屏幕外的后台辅助窗口，
    runner.capture_frame 的 locate_coinpoker（取最大匹配）在没有真牌桌
    时会匹配到它 → BitBlt 抓屏外窗口得到空帧 → 观察器 reshape 崩溃。
    本筛选（GUI 触发层）：可见、非最小化、≥600×300、未整体在屏幕外，
    取最大者；找不到 → None（诚实 UNAVAILABLE，不猜）。
    """
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    cands = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not u32.IsWindowVisible(hwnd):
            return True
        if u32.IsIconic(hwnd):
            return True
        ln = u32.GetWindowTextLengthW(hwnd)
        if ln <= 0:
            return True
        b = ctypes.create_unicode_buffer(ln + 1)
        u32.GetWindowTextW(hwnd, b, ctypes.sizeof(b))
        if "coinpoker" not in b.value.lower():
            return True
        r = wintypes.RECT()
        u32.GetWindowRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w < 600 or h < 300:          # 后台辅助小窗
            return True
        if r.right <= 0 or r.bottom <= 0:   # 整体在屏幕外
            return True
        cands.append({"bbox": [r.left, r.top, w, h]})
        return True

    u32.EnumWindows(cb, 0)
    return max(cands, key=lambda w: w["bbox"][2] * w["bbox"][3]) \
        if cands else None


def capture_frame_smart():
    """GUI 触发层捕获：筛选牌桌窗口 → 只读 BitBlt → 裁剪。

    返回 (rgba_bytes, w, h)；无合格窗口/捕获失败 → (None, None, None)。
    与 runner.capture_frame 使用同一 CtypesBitBltBackend 原语（只读）。
    """
    import numpy as np
    from poker_decision_ai.vision2.pixel.realclient.capture import (
        CAPTURE_OK, CtypesBitBltBackend)
    win = find_table_window()
    if win is None:
        return None, None, None
    x0, y0, cw, ch = win["bbox"]
    backend = CtypesBitBltBackend()
    r = backend.capture()
    if r["status"] != CAPTURE_OK:
        return None, None, None
    arr = np.frombuffer(r["image_bytes"], dtype=np.uint8).reshape(
        r["height"], r["width"], 4)
    if x0 + cw > r["width"] or y0 + ch > r["height"]:
        cw = min(cw, r["width"] - x0)
        ch = min(ch, r["height"] - y0)
    sub = arr[y0:y0 + ch, x0:x0 + cw]
    if sub.size == 0:
        return None, None, None
    return sub.tobytes(), cw, ch


def _runlog_dir() -> str:
    r"""运行文件日志目录（EXE 旁 logs\ ——与既有 EventLogger 同目录）。"""
    return "logs"


def _runlog(token: int, status: str, detail: str) -> None:
    """每次执行的文件日志（logs/gui-runs.log，追加）——便于诊断。"""
    try:
        import os
        os.makedirs(_runlog_dir(), exist_ok=True)
        with open(os.path.join(_runlog_dir(), "gui-runs.log"), "a",
                  encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} #{token} "
                     f"status={status} {detail}\n")
    except OSError:
        pass  # 日志失败不阻塞 UI（诚实：UI 日志面板仍有记录）


def build_stamp() -> str:
    """本进程**实际运行**的可执行文件/源码构建时间（用于排除"跑的是旧副本"）。"""
    import datetime
    import os
    import sys
    target = sys.executable if getattr(sys, "frozen", False) else __file__
    try:
        ts = datetime.datetime.fromtimestamp(
            os.path.getmtime(target)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        ts = "unknown"
    return f"build {ts}" + (" · frozen" if getattr(sys, "frozen", False)
                             else " · source")


class PreviewGUI:
    """最小常驻窗口：按钮触发 + 单 worker 串行执行 + 结果安全回传。"""

    def __init__(self, root: tk.Tk, frame_path: str | None = None,
                 runner: CashGameRunner | None = None):
        self.root = root
        self.frame_path = frame_path
        self.runner = runner or CashGameRunner()
        self._req_q: queue.Queue[int] = queue.Queue()
        self._res_q: queue.Queue[tuple] = queue.Queue()
        self._log_q: queue.Queue[str] = queue.Queue()   # 线程安全日志（worker 可用）
        # P30-9-AX：启动清扫——回收"父进程已死"的求解器孤儿（曾堆积 ~4.5GB）
        try:
            from cash_game_ui.orphan_sweep import sweep_orphan_solvers
            _sw = sweep_orphan_solvers()
            if _sw.get("killed"):
                self._log(f"↺ 启动清扫：回收 {_sw['killed']} 个孤儿求解器进程"
                          f"（父进程已退出；PID {_sw.get('orphans')}）")
        except Exception:                          # noqa: BLE001
            pass
        self._pending = 0
        self._next_token = 0
        self._executed = 0
        self._worker: threading.Thread | None = None
        root.title("Poker AI — EXE Preview GUI（只输出建议；不下注/不点击）")
        root.geometry("640x760")
        root.configure(bg="#0f1419")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._log("GUI 就绪（预览构建：诚实能力；UNKNOWN-first；无自动操作）")
        self.root.after(DRAIN_MS, self._drain)   # 结果事件泵（见模块 docstring）

    # ── 布局（只展示，不实现 pipeline） ─────────────────────────────
    def _build(self):
        tk.Label(self.root, text=build_stamp(), font=("Consolas", 8),
                 fg="#5b7285", bg="#0f1419").pack(anchor="center")
        tk.Label(self.root, text="CASH-GAME AI · PREVIEW",
                 font=("Segoe UI", 16, "bold"), fg="#ffffff",
                 bg="#0f1419").pack(pady=(12, 2))
        tk.Label(self.root,
                 text="一次点击 = 一次 Capture → Vision → Decision；"
                      "无轮询/无自动下注；UNKNOWN 不猜值；BASELINE ≠ REAL",
                 font=("Segoe UI", 9), fg="#88a0b8", bg="#0f1419").pack()

        self.btn = tk.Button(self.root,
                             text="📸  识别当前牌局（一次点击=一次完整识别链）",
                             font=("Segoe UI", 13), bg="#3aa0e8", fg="#ffffff",
                             activebackground="#2b8cd0", height=2,
                             relief="flat", command=self.on_collect)
        self.btn.pack(fill="x", padx=28, pady=10)

        # P30-9-K：盲注兜底输入（header 金额读不到时使用；用户声明的级别）
        bl = tk.Frame(self.root, bg="#0f1419")
        bl.pack(fill="x", padx=28)
        tk.Label(bl, text="盲注兜底(可选) SB", font=("Segoe UI", 9),
                 fg="#88a0b8", bg="#0f1419").pack(side="left")
        self._blinds_user_edited = False          # P30-9-P：用户手改过 ⇒ 不再自动覆盖
        self.sb_var = tk.StringVar(value=self._saved_blinds().get("sb", ""))
        self.sb_entry = tk.Entry(bl, textvariable=self.sb_var, width=8,
                                 font=("Consolas", 10), bg="#111820",
                                 fg="#c8d6e5", insertbackground="#c8d6e5",
                                 relief="flat")
        self.sb_entry.pack(side="left", padx=(4, 10))
        tk.Label(bl, text="BB", font=("Segoe UI", 9), fg="#88a0b8",
                 bg="#0f1419").pack(side="left")
        self.bb_var = tk.StringVar(value=self._saved_blinds().get("bb", ""))
        self.bb_entry = tk.Entry(bl, textvariable=self.bb_var, width=8,
                                 font=("Consolas", 10), bg="#111820",
                                 fg="#c8d6e5", insertbackground="#c8d6e5",
                                 relief="flat")
        self.bb_entry.pack(side="left", padx=(4, 10))
        # P30-9-P：实时归一（中文输入法的「。」「，」与全角数字都能输入）
        for e, v in ((self.sb_entry, self.sb_var), (self.bb_entry, self.bb_var)):
            e.bind("<KeyRelease>", lambda _ev, var=v: self._on_blinds_key(var))
            e.bind("<FocusOut>", lambda _ev, var=v: self._normalize_blinds_var(var))
            e.bind("<Control-v>", lambda _ev, var=v: self.root.after(
                10, lambda: self._on_blinds_key(var)))
        tk.Label(bl, text="（header 读不到盲注时使用；BB 必填、SB 可留空；留空=自动；"
                          "支持 0.02 这样的写法）",
                 font=("Segoe UI", 8), fg="#5b7285", bg="#0f1419").pack(side="left")

        # P30-9-AK：多人保守模式（**默认关** = 与原来完全一致）
        # 打开后：用「多人联合胜率」（同时击败所有对手）替换引擎的逐对手平均。
        mw = tk.Frame(self.root, bg="#0f1419")
        mw.pack(fill="x", padx=28, pady=(2, 0))
        self.multiway_var = tk.BooleanVar(value=True)   # P30-9-BF：默认开启
        self.multiway_chk = tk.Checkbutton(
            mw, text="多人联合胜率（范围化；默认开）",
            variable=self.multiway_var, font=("Segoe UI", 9),
            fg="#88a0b8", bg="#0f1419", selectcolor="#111820",
            activebackground="#0f1419", activeforeground="#c8d6e5",
            relief="flat", anchor="w")
        self.multiway_chk.pack(side="left")
        tk.Label(mw, text="默认使用「同时击败所有对手」的范围化联合胜率（用引擎自身范围）；取消勾选则回到引擎的逐对手平均"
                          "",
                 font=("Segoe UI", 8), fg="#5b7285", bg="#0f1419").pack(
            side="left", padx=6)

        js = tk.Frame(self.root, bg="#0f1419")
        js.pack(fill="x", padx=28, pady=(0, 4))
        self.btn_json = tk.Button(
            js, text="复制本次建议 JSON", font=("Segoe UI", 9),
            bg="#1d2b3a", fg="#c8d6e5", activebackground="#2b8cd0",
            relief="flat", command=self.on_copy_json)
        self.btn_json.pack(side="left")
        self.json_lbl = tk.Label(js, text="（尚无结果）", font=("Segoe UI", 8),
                                 fg="#5b7285", bg="#0f1419")
        self.json_lbl.pack(side="left", padx=8)

        self.status_lbl = tk.Label(self.root, text="● 待命",
                                   font=("Segoe UI", 10), fg="#2dc653",
                                   bg="#0f1419")
        self.status_lbl.pack()

        lf = tk.LabelFrame(self.root, text="最近一次结果",
                           font=("Segoe UI", 10, "bold"), fg="#88a0b8",
                           bg="#0f1419", bd=1, relief="solid")
        lf.pack(fill="x", padx=20, pady=6)
        self.action_lbl = tk.Label(lf, text="—", font=("Segoe UI", 30, "bold"),
                                   fg="#ffffff", bg="#0f1419")
        self.action_lbl.pack(pady=6)
        self.meta_lbl = tk.Label(lf, text="尚未分析", font=("Segoe UI", 10),
                                 fg="#88a0b8", bg="#0f1419", justify="left",
                                 wraplength=560)
        self.meta_lbl.pack(pady=(0, 8))

        tk.Label(self.root, text="运行日志", font=("Segoe UI", 10, "bold"),
                 fg="#88a0b8", bg="#0f1419").pack(anchor="w", padx=22)
        self.log_txt = scrolledtext.ScrolledText(
            self.root, height=12, font=("Consolas", 9), bg="#111820",
            fg="#c8d6e5", relief="flat", wrap="word", state="disabled")
        self.log_txt.pack(fill="both", expand=True, padx=20, pady=(2, 12))

    def _prefill_blinds_from_evidence(self) -> dict | None:
        """读一次屏 → 桥 → 盲注建议 → 写入输入框（**仅预填，需人工再点一次**）。"""
        try:
            rgba, w, h = capture_frame_smart()
            if rgba is None:
                return None
            from poker_decision_ai.vision2.pixel.realclient.production_bridge \
                import frame_to_decision_request
            from poker_decision_ai.facing_evidence import suggest_blinds
            obs = (frame_to_decision_request(rgba, w, h).get("bridge") or {}).get(
                "observation") or {}
            ev = suggest_blinds(obs)
            if not ev.get("bb"):
                return None
            if ev.get("sb"):
                self.sb_var.set(f"{float(ev['sb']):g}")
            self.bb_var.set(f"{float(ev['bb']):g}")
            return ev
        except Exception:                          # noqa: BLE001
            return None

    def _normalize_blinds_var(self, var) -> None:
        """把输入框内容归一为 `[0-9.]`（保留光标位置尽量靠后）。"""
        try:
            from poker_decision_ai.facing_evidence import normalize_decimal_text
            cur = var.get()
            fixed = normalize_decimal_text(cur)
            if fixed != cur:
                var.set(fixed)
        except Exception:                          # noqa: BLE001
            pass

    def _on_blinds_key(self, var) -> None:
        """按键后：标记用户已编辑（此后不被自动回写覆盖）+ 归一全角字符。"""
        self._blinds_user_edited = True
        self._normalize_blinds_var(var)

    @staticmethod
    def _blinds_cfg_path() -> str:
        import os
        d = os.path.join(os.getcwd(), "configs")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return os.path.join(os.getcwd(), "gui_blinds.json")
        return os.path.join(d, "gui_blinds.json")

    def _saved_blinds(self) -> dict:
        try:
            import json
            with open(self._blinds_cfg_path(), encoding="utf-8") as fh:
                d = json.load(fh)
            return {"sb": d.get("sb", ""), "bb": d.get("bb", "")}
        except Exception:                          # noqa: BLE001
            return {"sb": "", "bb": ""}

    def _blinds_override(self) -> dict | None:
        """读取输入框并持久化；无效/留空 ⇒ None（不猜）。"""
        from poker_decision_ai.facing_evidence import normalize_decimal_text
        sb_s = normalize_decimal_text(self.sb_var.get().strip())
        bb_s = normalize_decimal_text(self.bb_var.get().strip())
        self.sb_var.set(sb_s)
        self.bb_var.set(bb_s)
        if not bb_s:
            return None          # BB 必填；SB 可留空（下游按未知处理）
        try:
            sb = float(sb_s) if sb_s else None
            bb = float(bb_s)
        except ValueError:
            self._log("✖ 盲注输入非法（需数字），本次按自动路径")
            return None
        if not (bb > 0) or (sb is not None and not (0 < sb < bb)):
            self._log("✖ 盲注输入不自洽（需 0<SB<BB；SB 可留空），本次按自动路径")
            return None
        try:
            import json
            with open(self._blinds_cfg_path(), "w", encoding="utf-8") as fh:
                json.dump({"sb": sb, "bb": bb}, fh, ensure_ascii=False)
        except Exception:                          # noqa: BLE001
            pass
        return {"sb": sb, "bb": bb}

    def on_copy_json(self):
        """把最近一次建议 JSON 复制到剪贴板（无结果 → 明确提示）。"""
        import os
        path = getattr(self, "_last_advice_json", None)
        if not path or not os.path.exists(path):
            self._log("✖ 尚无建议 JSON（先点一次「识别当前牌局」）")
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._log(f"✔ 已复制建议 JSON 到剪贴板（{os.path.basename(path)}；"
                      f"文件：{path}）")
            self.json_lbl.config(text=f"JSON: {os.path.basename(path)}")
        except Exception as e:                          # noqa: BLE001
            self._log(f"✖ 复制失败：{type(e).__name__}: {e}")

    # ── 触发（主线程：按钮 → 入队） ─────────────────────────────────
    def on_collect(self):
        """「采集窗口」命令：一次点击 = 恰好一次 pipeline（排队执行）。"""
        # P30-9-AO：**合并排队**——连点时丢弃尚未开始的旧请求，只跑最新一次
        dropped = 0
        try:
            while True:
                self._req_q.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        if dropped:
            self._pending = max(0, self._pending - dropped)
            self._log(f"↺ 合并排队：丢弃 {dropped} 次尚未开始的请求"
                      f"（只跑最新一次；正在执行的那次不受影响）")
        if self._pending >= MAX_PENDING:
            self._log(f"✖ 队列已满（{MAX_PENDING}），忽略本次点击")
            return
        self._next_token += 1
        self._pending += 1
        token = self._next_token
        self._req_q.put(token)
        self._log(f"▶ 点击 #{token} → 入队（1 点击 = 1 次 pipeline；"
                  f"排队 {self._pending}）")
        self._refresh_status()
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._work_loop, daemon=True, name="pipeline-worker")
            self._worker.start()

    # ── worker（后台线程：绝不触碰 Tk） ─────────────────────────────
    def _work_loop(self):
        while True:
            token = self._req_q.get()
            if token is None:
                break
            try:
                # P30-9-AO：看门狗——超时把**线程栈**转储，便于定位真卡死
                try:
                    import faulthandler
                    import os
                    _d = os.path.join(os.getcwd(), "logs")
                    os.makedirs(_d, exist_ok=True)
                    _hf = open(os.path.join(
                        _d, time.strftime("hang-%Y%m%d-%H%M%S.txt")), "w",
                        encoding="utf-8")
                    faulthandler.dump_traceback_later(RUN_TIMEOUT_S, file=_hf)
                    self._watchdog_file = _hf
                except Exception:                      # noqa: BLE001
                    self._watchdog_file = None
                self._run_started_at = time.time()
                res = self._run_once()
                d = res.to_dict()
                d["gui_stages"] = dict(getattr(self, "_stages", {}))
            except Exception as e:  # noqa: BLE001 — UI 兜底，诚实显示
                import traceback
                d = {"status": "ERROR", "reason": f"{type(e).__name__}: {e}",
                     "decision_action": None}
                _runlog(token, "EXCEPTION",
                        traceback.format_exc(limit=6).replace("\n", " | "))
            finally:
                try:
                    import faulthandler
                    faulthandler.cancel_dump_traceback_later()
                    if getattr(self, "_watchdog_file", None):
                        self._watchdog_file.close()
                        self._watchdog_file = None
                except Exception:                      # noqa: BLE001
                    pass
                self._run_started_at = None
                self._run_elapsed = None
            self._res_q.put((token, d, time.strftime("%H:%M:%S")))

    def _run_once(self):
        """一次完整链：捕获（或 --frame 读取）→ Vision → Decision。"""
        stages = {}
        t0 = time.perf_counter()
        if not self.frame_path:
            self._log("预检: 枚举标题含 CoinPoker 的可见窗口（只读）…")
            rgba, w, h = capture_frame_smart()
            stages["capture_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if rgba is None:
                from cash_game_ui.runner import AnalysisResult
                self._log("预检: 未找到可见的 CoinPoker 牌桌窗口"
                          "（可能只有后台辅助窗口/窗口最小化）")
                return AnalysisResult(
                    status="UNAVAILABLE",
                    reason="capture_failed:no_visible_coinpoker_table",
                    agents=[])
            self._log(f"预检: 牌桌窗口 bbox 已锁定（{w}x{h}，"
                      f"capture {stages['capture_ms']:.0f}ms）→ 执行链")
            t1 = time.perf_counter()
            # P30-9-J：GUI 自行截图后传入帧 ⇒ 显式允许 hero 牌缺失时的只读重捕获
            # P30-9-K：盲注兜底输入（header 读不到时使用）
            res = self.runner.analyze(
                rgba, w, h,
                multiway_conservative=bool(self.multiway_var.get()),
                allow_recapture=True,
                                      blinds_override=self._blinds_override())
            stages["pipeline_ms"] = round(
                (time.perf_counter() - t1) * 1000, 1)
        else:
            from poker_decision_ai.vision2_platform.png_decode import decode_png
            w, h, rgba = decode_png(open(self.frame_path, "rb").read())
            stages["capture_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            stages["capture_mode"] = "frame_png"
            t1 = time.perf_counter()
            # 固定帧（fixture/测试）⇒ 禁止重捕获（避免对同一 PNG 反复读屏）
            res = self.runner.analyze(rgba, w, h, allow_recapture=False)
            stages["pipeline_ms"] = round(
                (time.perf_counter() - t1) * 1000, 1)
        stages["result_ms"] = round((time.perf_counter() - t0) * 1000
                                    - stages["capture_ms"]
                                    - stages["pipeline_ms"], 1)
        self._stages = stages
        return res

    # ── 结果事件泵（主线程：只搬运结果/日志；不采集） ────────────────
    def _drain(self):
        try:
            while True:
                msg = self._log_q.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        while True:
            try:
                token, d, ts = self._res_q.get_nowait()
            except queue.Empty:
                break
            self._pending = max(0, self._pending - 1)
            self._executed += 1
            try:
                self._show(token, d, ts)
            except Exception as _e:                # noqa: BLE001
                # P30-9-AU：显示异常**不得**终止刷新循环（曾导致界面"卡死"）
                import traceback
                self._log(f"✖ 显示异常（已忽略，循环继续）: "
                          f"{type(_e).__name__}: {_e}")
                try:
                    self.action_lbl.config(text="⚠ 显示异常", fg="#e74c3c")
                    self.meta_lbl.config(
                        text=f"#{token} 显示阶段异常：{type(_e).__name__}: {_e}\n"
                             + traceback.format_exc(limit=4),
                        fg="#e74c3c")
                except Exception:                  # noqa: BLE001
                    pass
        started = getattr(self, "_run_started_at", None)
        self._run_elapsed = (time.time() - started) if started else None
        self._refresh_status()
        self.root.after(DRAIN_MS, self._drain)

    def _sync_blinds_box_from_recognition(self, vis: dict) -> None:
        """把**识别到**的盲注回写输入框（自我纠正——防跨桌沿用旧值）。

        仅当观察里确有 blinds 且与输入框不同才写；写入即持久化到
        `configs/gui_blinds.json`（下次 header 读不到时用的就是本桌真值）。
        """
        b = vis.get("blinds")
        if not isinstance(b, dict):
            return
        # P30-9-O：与底池/桌上投入**矛盾**的 header 盲注不得回写（否则误读值会
        # 被当成"本桌真值"长期沿用，实测案例：0.02/0.05 被读成 2/5）
        if vis.get("blinds_consistent") is False:
            self._log("⚠ 识别到的盲注与底池/桌上投入矛盾（疑 header 误读）"
                      "⇒ 不回写兜底框；请手动确认 SB/BB")
            return
        try:
            sb, bb = float(b.get("sb")), float(b.get("bb"))
        except (TypeError, ValueError):
            return
        if not (sb > 0 and bb > 0):
            return
        cur_sb = self.sb_var.get().strip()
        cur_bb = self.bb_var.get().strip()
        new_sb, new_bb = f"{sb:g}", f"{bb:g}"
        from poker_decision_ai.facing_evidence import writeback_decision
        act = writeback_decision(cur_sb, cur_bb, new_sb, new_bb,
                                 self._blinds_user_edited)
        if act == "noop":
            return
        if act == "skip_user_edited":
            # P30-9-P：用户手填过 ⇒ 绝不自动覆盖（只提示识别值供其参考）
            self._log(f"↳ 识别到盲注 SB={new_sb} BB={new_bb}"
                      f"（你已手填 {cur_sb or '空'}/{cur_bb or '空'}，不覆盖）")
            return
        self.sb_var.set(new_sb)
        self.bb_var.set(new_bb)
        try:
            import json
            with open(self._blinds_cfg_path(), "w", encoding="utf-8") as fh:
                json.dump({"sb": sb, "bb": bb}, fh, ensure_ascii=False)
        except Exception:                              # noqa: BLE001
            pass
        self._log(f"↳ 已用识别到的盲注更新兜底框：SB={new_sb} BB={new_bb}"
                  f"（原 {cur_sb or '空'}/{cur_bb or '空'}；本桌真值）")

    def _show(self, token: int, d: dict, ts: str):
        # R13-EXE-2 三态显示：a. OK → 建议（动作+尺寸+置信+provenance）；
        # b. 识别不到 → 原因 + 缺失字段清单；c. 部分识别 → 已识别+缺失
        # （不编造）。capture 失败/几何不符两分支保留（如实归因）。
        status = d.get("status", "UNKNOWN")
        action = d.get("decision_action")
        size = d.get("recommended_size")
        conf = d.get("confidence")
        solver = d.get("solver_mode", "UNKNOWN")
        strategy = d.get("strategy_mode", "UNKNOWN")
        reason = d.get("reason") or ""
        vis = d.get("vision") or {}
        # P30-9-N：记录本次 JSON 路径（供「复制本次建议 JSON」按钮）
        if d.get("advice_json"):
            self._last_advice_json = d.get("advice_json")
            try:
                import os
                self.json_lbl.config(
                    text=f"JSON: {os.path.basename(self._last_advice_json)}")
            except Exception:                          # noqa: BLE001
                pass
        # P30-9-M：识别到盲注 ⇒ 回写兜底框（自我纠正，防跨桌陈旧值）
        try:
            self._sync_blinds_box_from_recognition(vis)
        except Exception:                              # noqa: BLE001
            pass
        _lbl = None                     # P30-9-AU：大字与详情**原子更新**
        blocking = list(d.get("blocking_fields") or [])
        unknown = list(d.get("unknown_fields") or [])
        got = [f"{k}={vis.get(k)}" for k in
               ("hero_cards", "pot", "board", "blinds", "to_call",
                "street", "position") if vis.get(k) is not None]
        if status == "ADVICE_REFERENCE":
            # 态 a-3（R14-EXE-3 方案 Y）：参考打法（未识别动作选项）——
            # 诚实标注：不是针对当前下注的决策、不代表对当前对手最优解；
            # 动作选项识别恢复后自动升级为正常决策。
            adv = (d.get("raw") or {}).get("advice_reference") or {}
            act_txt = (str(action).upper() if action else "—") \
                + (f" {size}bb" if size else "") + " 参考"
            _lbl = (act_txt, "#f5a623")
            meta = (f"#{token}  参考打法（未识别动作选项）——"
                    "不是针对当前下注的决策，也不代表对当前对手的最优解\n"
                    f"来源 {adv.get('source') or 'n/a'} · "
                    f"场景 {adv.get('scenario') or 'n/a'} · "
                    f"手牌 {adv.get('hand_class') or 'n/a'}\n"
                    f"盲注 {vis.get('blinds')} · 底池 {vis.get('pot')} · "
                    f"{vis.get('street') or ''}")
            color = "#f5a623"
            self._log(f"✔ #{token} [{ts}] status=ADVICE_REFERENCE "
                      f"参考打法 action={action} size={size} "
                      f"source={adv.get('source')} "
                      f"scenario={adv.get('scenario')}")
            _runlog(token, status,
                    f"参考打法 action={action} size={size} "
                    f"source={adv.get('source')} "
                    f"scenario={adv.get('scenario')}")
        elif status == "OK" and action:
            # P30-9-N：策略尺寸被撤下时显示**规则下限尺寸**（具体尺度，非策略）
            rule = d.get("rule_size") or {}
            act_txt = str(action).upper() + (f" {size}bb" if size else "")
            if not size and rule.get("bb"):
                chips = rule.get("chips")
                act_txt += (f" ≥{rule['bb']:g}bb" if chips is None
                             else f" ≥{chips:g}")
            _lbl = (act_txt, "#2dc653")
            rule_txt = ""
            if not size and rule.get("bb"):
                chips = rule.get("chips")
                rule_txt = (f"\n规则下限：{'下注' if rule.get('action') == 'bet' else '加注到'} "
                            f"{rule['bb']:g}bb"
                            + (f" = {chips:g} chips" if chips is not None else "")
                            + "（非策略尺寸）")
            meta = (f"#{token}  Status: {status}   Strategy: {strategy}   "
                    f"Confidence: {conf if conf is not None else 'n/a'}\n"
                    f"盲注 {vis.get('blinds')} · 底池 {vis.get('pot')} · "
                    f"{vis.get('street') or ''} · 位置 {vis.get('position')}"
                    + rule_txt)
            color = "#c8d6e5"
            self._log(f"✔ #{token} [{ts}] status={status} "
                      f"action={str(action).upper()} size={size}"
                      + (f" rule>={rule['bb']:g}bb" if (not size and rule.get('bb'))
                         else "") + " "
                      f"strategy={strategy} conf={conf}")
            # P30-9-IC：**几何归一化提示** ✓（窗口尺寸与参考 1876x1400 不同时，
            #   程序会缩放归一以继续识别 ✓；如实告知用户，便于追求满精度 ✓）
            _geo = getattr(getattr(self, "runner", None), "_last_geometry", None) or {}
            if _geo.get("normalized"):
                _f = _geo.get("from") or (None, None)
                self._log(f"⚙ 几何归一化: {_f[0]}x{_f[1]} → "
                          f"{_geo['reference'][0]}x{_geo['reference'][1]}"
                          f"（缩放 {_geo.get('scale')}）✓ 大部分字段可读；"
                          f"若要满精度，请把牌桌窗口设为 "
                          f"{_geo['reference'][0]}x{_geo['reference'][1]} ✓")
            elif _geo.get("reason"):
                self._log(f"⚠ 采集几何超出可归一范围（{_geo.get('reason')}）✗ "
                          f"⇒ 请把牌桌窗口设为 {_geo['reference'][0]}x{_geo['reference'][1]} ✓")
            _runlog(token, status, f"action={action} size={size} "
                    f"strategy={strategy} conf={conf}")
        elif status == "UNAVAILABLE" and reason.startswith("capture_failed"):
            # 诚实 UNKNOWN-first：未找到可见牌桌窗口 ≠ 程序故障（橙色提示）
            _lbl = ("⚠ 未找到牌桌窗口", "#f5a623")
            meta = (f"#{token}  识别不到：未找到可见的 CoinPoker 牌桌窗口"
                    f"（{reason}）。\n"
                    "UNKNOWN-first：找不到牌桌就不猜值。常见原因：\n"
                    "  - 客户端未打开牌桌（后台辅助窗口不算）；\n"
                    "  - 牌桌窗口被最小化；\n"
                    "  - 只登录了大厅/未进入牌桌界面。\n"
                    "请打开并完整显示牌桌后重试；或用测试开关：\n"
                    "  poker-ai-gui-preview.exe --frame sample_frame.png")
            color = "#f5a623"
            self._log(f"✔ #{token} [{ts}] status=UNAVAILABLE "
                      f"reason={reason}（无可见牌桌——按设计不猜值）")
            _runlog(token, status, f"reason={reason}")
        elif status == "ERROR" and "reshape array of size 0" in reason:
            # 捕获帧与预期几何不符（匹配到的窗口不是牌桌界面/最小化/缩放不符）
            _lbl = ("⚠ 捕获帧无效", "#f5a623")
            meta = (f"#{token}  捕获的窗口内容与牌桌几何不符（观察器裁剪区为空）。\n"
                    "常见原因：标题含 CoinPoker 的窗口不是可见牌桌界面、\n"
                    "窗口被最小化/在副屏或屏幕外、显示缩放不匹配。\n"
                    "请把 CoinPoker 牌桌窗口完整显示在主屏后重试；\n"
                    "或用测试开关验证链路：--frame sample_frame.png\n"
                    f"原始错误: {reason}")
            color = "#f5a623"
            self._log(f"✔ #{token} [{ts}] status=ERROR（捕获几何不符） "
                      f"reason={reason}")
            _runlog(token, status, f"reason={reason}")
        elif blocking and not action:
            # 态 b：MANDATORY 字段识别不到 → 原因 + 缺失清单（+已识别）
            _lbl = ("识别不到", "#e74c3c")
            why = CashGameRunner.blocking_reason(blocking)
            fprov = d.get("frame_provenance") or {}
            diag_bits = []
            if vis.get("hero_cards_reason"):
                _or = vis.get("hero_cards_observer_reason")
                diag_bits.append(
                    "hero牌未识别原因: " + str(vis["hero_cards_reason"])
                    + (f"（观察者: {_or}）" if _or else ""))
            if fprov.get("multiway_mode"):
                diag_bits.append(
                    "多人保守模式: 开（多人估计 "
                    f"{fprov.get('multiway_estimate')}；引擎逐对手平均见推理）")
            if getattr(self.runner, "_last_frame_path", None):
                diag_bits.append(
                    f"失败帧已存档: {self.runner._last_frame_path}")
            if vis.get("hero_cards_ok") is False and not vis.get(
                    "hero_cards_reason"):
                diag_bits.append("hero牌未识别原因: 未知")
            if fprov.get("blinds_source"):
                diag_bits.append(f"盲注来源: {fprov['blinds_source']}")
            if vis.get("facing_from_seat_bets"):
                # P30-9-R：标注座位（你 / SB / BB / BTN），None = 未识别或无投入
                diag_bits.append(
                    f"座位下注: {vis.get('seat_bets_text') or vis['facing_from_seat_bets']}")
            if fprov.get("recapture_attempts") is not None:
                diag_bits.append(f"重捕获次数: {fprov.get('recapture_attempts')}")
            meta = (f"#{token}  识别不到：{why}\n"
                    f"缺失字段: {', '.join(blocking)}\n"
                    f"已识别: {'; '.join(got) if got else '（无）'}"
                    + (f"\n部分未知(不阻断): {', '.join(unknown)}"
                       if unknown else "")
                    + ("\n" + " · ".join(diag_bits) if diag_bits else ""))
            color = "#e74c3c"
            # P30-9-L：盲注阻断且兜底框为空 ⇒ 用证据**预填**（供你确认后一键出建议）
            _prefilled = None
            if "blinds" in blocking and not self.sb_var.get().strip() \
                    and not self.bb_var.get().strip():
                try:
                    _prefilled = self._prefill_blinds_from_evidence()
                except Exception as _e:            # noqa: BLE001
                    self._log(f"⚠ 盲注预填失败（不影响显示）: "
                              f"{type(_e).__name__}: {_e}")
                    _prefilled = None
            _rc = fprov.get("recapture_attempts")
            _rc_txt = f" recapture={_rc}" if _rc is not None else ""
            self._log(f"✔ #{token} [{ts}] status={status} "
                      f"识别不到：{reason} blocking={blocking}{_rc_txt}")
            if _prefilled:
                self._log(f"↳ 已按证据预填盲注兜底：SB={_prefilled.get('sb')} "
                          f"BB={_prefilled.get('bb')}（来源 {_prefilled.get('source')}"
                          f"，非自动采用）→ 再点一次即出建议；如不符请直接改输入框")
            _runlog(token, status,
                    f"reason={reason} blocking={blocking}{_rc_txt}")
        elif action:
            # 态 a-2：决策链已产出建议但门未确认（如实标注，不冒充 READY）
            act_txt = str(action).upper() + (f" {size}bb" if size else "")
            _lbl = (act_txt + " ？", "#f5a623")
            meta = (f"#{token}  建议已产出但决策门未确认：{reason}\n"
                    f"置信 {conf if conf is not None else 'n/a'}"
                    f" · 盲注 {vis.get('blinds')} · 底池 {vis.get('pot')}")
            color = "#f5a623"
            self._log(f"✔ #{token} [{ts}] status={status} 建议(未确认) "
                      f"action={action} reason={reason}")
            _runlog(token, status, f"action={action} unconfirmed "
                    f"reason={reason}")
        else:
            # 态 c：部分识别（已识别 + 缺失清单，如实显示）
            _lbl = ("信息不足", "#f5a623")
            meta = (f"#{token}  {reason or '无法形成安全决策（UNKNOWN-first）'}"
                    f"\n已识别: {'; '.join(got) if got else '（无）'}"
                    f"\n缺失: {', '.join(unknown) if unknown else '（无）'}")
            color = "#f5a623"
            self._log(f"✔ #{token} [{ts}] status={status} reason={reason}"
                      f" unknown={unknown}")
            _runlog(token, status, f"reason={reason} unknown={unknown}")
        stages = d.get("gui_stages") or {}
        if stages:
            st = (f"\n耗时: 捕获 {stages.get('capture_ms', 0):.0f}ms → "
                  f"视觉+决策 {stages.get('pipeline_ms', 0):.0f}ms → "
                  f"结果构建 {stages.get('result_ms', 0):.0f}ms")
            meta += st
            _runlog(token, status,
                    "stages capture={}ms pipeline={}ms result={}ms".format(
                        stages.get("capture_ms", 0),
                        stages.get("pipeline_ms", 0),
                        stages.get("result_ms", 0)))
        if _lbl is not None:            # 与详情同一时刻应用（避免错配）
            self.action_lbl.config(text=_lbl[0], fg=_lbl[1])
        self.meta_lbl.config(text=meta, fg=color)

    def _refresh_status(self):
        if self._pending:
            el = getattr(self, "_run_elapsed", None)
            extra = ""
            if el is not None:
                extra = f"（已 {el:.1f}s）"
                if el > RUN_TIMEOUT_S:
                    extra += f" ⚠ 超过 {RUN_TIMEOUT_S}s，见 logs/hang-*.txt"
            self.status_lbl.config(
                text=f"● 运行中{extra}（排队 {self._pending}）", fg="#f5a623")
        else:
            self.status_lbl.config(
                text=f"● 待命（已完成 {self._executed}）", fg="#2dc653")

    def _log(self, msg: str):
        """线程安全：入日志队列（主线程 _drain 刷到文本框）。"""
        self._log_q.put(time.strftime("[%H:%M:%S] ") + msg)

    def _append_log(self, line: str):
        self.log_txt.config(state="normal")
        self.log_txt.insert(tk.END, line + "\n")
        self.log_txt.see(tk.END)
        self.log_txt.config(state="disabled")

    def _on_close(self):
        try:
            self._req_q.put(None)          # 优雅停止 worker
        except Exception:
            pass
        self.root.destroy()
