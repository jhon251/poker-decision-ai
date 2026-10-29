"""Cash-Game AI — 极简 Decision UI（Tkinter——无大框架）。

交互：点击一次「▶ 视觉 Agent开始工作」→ 完整合法分析链 → UI 显示：
  状态（READY/PROCESSING/UNAVAILABLE）
  最终决策（大号 CALL/CHECK/... — 结论优先）
  Confidence / Status / Solver mode
  Agent 输出列表（状态 + 关键输出）
  [展开详细数据]（可折叠 — 不阻塞结论）

红线：UNKNOWN 不猜值；BASELINE 不显示为 REAL；无自动操作。
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

try:
    from .runner import CashGameRunner
except ImportError:  # PyInstaller onefile 兼容
    from cash_game_ui.runner import CashGameRunner


class CashGameApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.runner = CashGameRunner()
        root.title("CASH-GAME AI")
        root.geometry("560x720")
        root.configure(bg="#0f1419")
        self._build()

    def _build(self):
        # 标题
        tk.Label(self.root, text="CASH-GAME AI", font=("Segoe UI", 20, "bold"),
                 fg="#ffffff", bg="#0f1419").pack(pady=(16, 4))

        # 按钮（两个入口：完整决策链 / 一键识别）
        self.btn = tk.Button(
            self.root, text="▶  视觉 Agent开始工作", font=("Segoe UI", 13),
            bg="#2dc653", fg="#0f1419", activebackground="#28a745",
            command=self._on_click, height=2, relief="flat")
        self.btn.pack(fill="x", padx=32, pady=(12, 4))

        # R13-EXE-2：一键"识别当前牌局"（capture → 全链路识别 → 三态显示）
        self.btn_rec = tk.Button(
            self.root, text="📸  识别当前牌局", font=("Segoe UI", 13),
            bg="#3aa0e8", fg="#ffffff", activebackground="#2b8cd0",
            command=self._on_recognize, height=2, relief="flat")
        self.btn_rec.pack(fill="x", padx=32, pady=4)

        # 状态
        self.status_lbl = tk.Label(self.root, text="● READY", font=(
            "Segoe UI", 11), fg="#2dc653", bg="#0f1419")
        self.status_lbl.pack(pady=2)

        # 决策区（结论优先）
        self.decision_frame = tk.LabelFrame(
            self.root, text="最终决策", font=("Segoe UI", 11, "bold"),
            fg="#88a0b8", bg="#0f1419", bd=1, relief="solid")
        self.decision_frame.pack(fill="both", padx=24, pady=10)
        self.action_lbl = tk.Label(
            self.decision_frame, text="—", font=("Segoe UI", 34, "bold"),
            fg="#ffffff", bg="#0f1419")
        self.action_lbl.pack(pady=10)
        self.meta_lbl = tk.Label(
            self.decision_frame, text="点击按钮开始分析", font=("Segoe UI", 11),
            fg="#88a0b8", bg="#0f1419")
        self.meta_lbl.pack(pady=4)

        # Agent 输出（结构化）
        self.agents_frame = tk.LabelFrame(
            self.root, text="Agent 输出", font=("Segoe UI", 11, "bold"),
            fg="#88a0b8", bg="#0f1419", bd=1, relief="solid")
        self.agents_frame.pack(fill="both", padx=24, pady=6)
        self.agents_txt = tk.Text(self.agents_frame, height=10,
                                  font=("Consolas", 9), bg="#111820",
                                  fg="#c8d6e5", relief="flat", wrap="word")
        self.agents_txt.pack(fill="both", padx=8, pady=8, expand=True)

        # 展开详细数据
        self.detail_btn = tk.Button(
            self.root, text="▼ 展开详细数据", font=("Segoe UI", 10),
            bg="#1a2530", fg="#88a0b8", command=self._toggle_detail,
            relief="flat")
        self.detail_btn.pack(fill="x", padx=24, pady=4)
        self.detail_panel = None

    # ── 交互 ─────────────────────────────────────────────────────────
    def _on_click(self):
        self.btn.config(state="disabled", text="⏳ 分析中…")
        self.status_lbl.config(text="● PROCESSING", fg="#f5a623")
        self.action_lbl.config(text="…", fg="#88a0b8")
        self.meta_lbl.config(text="正在执行视觉 Agent → 决策链", fg="#88a0b8")
        self.agents_txt.delete("1.0", tk.END)

        def work():
            try:
                res = self.runner.analyze()
            except Exception as e:  # noqa: BLE001 - UI 兜底
                res = None
                err = str(e)[:200]
            self.root.after(0, lambda: self._show(res if res is not None
                                                  else self._error(err)))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _error(msg: str) -> str:
        return f"ERROR: {msg}"

    # ── R13-EXE-2：一键"识别当前牌局"三态显示 ────────────────────────
    def _on_recognize(self):
        """capture → 全链路识别 → 三态：
        a. OK → AI 建议（动作+尺寸+置信+provenance）；
        b. 识别不到 → 原因（缺手牌/盲注未识别/画面非牌桌）+ 缺失清单；
        c. 部分识别 → 已识别字段 + 缺失清单（不编造）。"""
        self.btn_rec.config(state="disabled", text="⏳ 识别中…")
        self.btn.config(state="disabled")
        self.status_lbl.config(text="● PROCESSING", fg="#f5a623")
        self.action_lbl.config(text="…", fg="#88a0b8")
        self.meta_lbl.config(text="正在截取 CoinPoker 窗口 → 识别链", fg="#88a0b8")
        self.agents_txt.delete("1.0", tk.END)

        def work():
            try:
                res = self.runner.analyze()
                err = None
            except Exception as e:  # noqa: BLE001 - UI 兜底
                res = None
                err = str(e)[:200]
            self.root.after(0, lambda: self._show_recognize(
                res if res is not None else self._error(err or "unknown")))

        threading.Thread(target=work, daemon=True).start()

    def _show_recognize(self, res):
        self.btn_rec.config(state="normal", text="📸  识别当前牌局")
        self.btn.config(state="normal")
        if isinstance(res, str):
            self.status_lbl.config(text="● ERROR", fg="#e74c3c")
            self.action_lbl.config(text="ERROR", fg="#e74c3c", font=(
                "Segoe UI", 24, "bold"))
            self.meta_lbl.config(text=res, fg="#e74c3c")
            return
        d = res.to_dict() if hasattr(res, "to_dict") else res
        status = d.get("status", "UNKNOWN")
        vis = d.get("vision") or {}
        blocking = list(d.get("blocking_fields") or [])
        unknown = list(d.get("unknown_fields") or [])
        if status == "OK":
            # 态 a：AI 建议（动作+尺寸+置信+provenance）
            self.status_lbl.config(text="● RECOGNIZED → AI 建议", fg="#2dc653")
            action = str(d.get("decision_action") or "—").upper()
            size = d.get("recommended_size")
            act_txt = action + (f" {size}bb" if size else "")
            self.action_lbl.config(text=act_txt, fg="#2dc653",
                                   font=("Segoe UI", 34, "bold"))
            meta = (f"置信 {d.get('confidence') if d.get('confidence') is not None else 'n/a'}"
                    f" · strategy={d.get('strategy_mode', 'UNKNOWN')}"
                    f" · 盲注 {vis.get('blinds')} · 底池 {vis.get('pot')}"
                    f" · {d.get('street') or ''}")
            self.meta_lbl.config(text=meta, fg="#c8d6e5")
        elif status == "ADVICE_REFERENCE":
            # 态 a-3（R14-EXE-3 方案 Y）：参考打法（未识别动作选项）——
            # 诚实标注：不是针对当前下注的决策，不代表对当前对手最优解
            self.status_lbl.config(text="● 参考打法（未识别动作选项）",
                                   fg="#f5a623")
            action = str(d.get("decision_action") or "—").upper()
            size = d.get("recommended_size")
            self.action_lbl.config(
                text=action + (f" {size}bb" if size else "") + " 参考",
                fg="#f5a623", font=("Segoe UI", 30, "bold"))
            self.meta_lbl.config(
                text=str(d.get("reasoning")
                         or d.get("reason")
                         or "参考打法：未识别轮到你的动作选项")
                    + f"\n来源 {((d.get('raw') or {}).get('advice_reference') or {}).get('source') or 'n/a'}"
                    f" · 场景 {((d.get('raw') or {}).get('advice_reference') or {}).get('scenario') or 'n/a'}"
                    f" · 盲注 {vis.get('blinds')} · 底池 {vis.get('pot')}",
                fg="#f5a623")
        elif status == "UNAVAILABLE" and str(d.get("reason", "")).startswith(
                "capture_failed"):
            # 态 b-1：画面非牌桌 / 窗口未找到
            self.status_lbl.config(text="● 识别不到", fg="#e74c3c")
            self.action_lbl.config(text="画面非牌桌", fg="#e74c3c",
                                   font=("Segoe UI", 22, "bold"))
            self.meta_lbl.config(
                text="识别不到：未找到 CoinPoker 窗口（或窗口最小化）——"
                     "请把牌桌窗口置于前台后重试", fg="#e74c3c")
        elif blocking or not d.get("decision_action"):
            # 态 b/c：识别不到（MANDATORY 缺失 → 原因）+ 部分识别
            # （已识别字段 + 缺失清单，如实显示不编造）
            why = self._blocking_reason(blocking) if blocking else (
                d.get("reason") or "识别信息不足")
            got = [f"{k}={vis.get(k)}" for k in
                   ("hero_cards", "pot", "board", "blinds", "to_call",
                    "street", "position") if vis.get(k) is not None]
            self.status_lbl.config(
                text="● 识别不到" if blocking else "● 部分识别",
                fg="#e74c3c" if blocking else "#f5a623")
            self.action_lbl.config(
                text="识别不到" if blocking else "信息不足",
                fg="#e74c3c" if blocking else "#f5a623",
                font=("Segoe UI", 22, "bold"))
            self.meta_lbl.config(
                text=f"识别不到：{why}"
                     f"\n缺失字段: {', '.join(blocking) if blocking else '（见 unknown）'}"
                     f"\n已识别: {'; '.join(got) if got else '（无）'}"
                     + (f"\n部分未知(不阻断): {', '.join(unknown)}"
                        if unknown else ""),
                fg="#e74c3c" if blocking else "#f5a623")
        else:
            # 态 a-2：决策链已产出建议，但 judge 门未确认（如实标注，
            # 不冒充 READY——不编造合法性）
            action = str(d.get("decision_action") or "—").upper()
            size = d.get("recommended_size")
            self.status_lbl.config(text="● 建议（未确认）", fg="#f5a623")
            self.action_lbl.config(
                text=action + (f" {size}bb" if size else ""), fg="#f5a623",
                font=("Segoe UI", 30, "bold"))
            self.meta_lbl.config(
                text=f"建议已产出但决策门未确认：{d.get('reason')}\n"
                     f"置信 {d.get('confidence') if d.get('confidence') is not None else 'n/a'}"
                     f" · 盲注 {vis.get('blinds')} · 底池 {vis.get('pot')}",
                fg="#f5a623")
        # Agent 输出（结构化，同 _show）
        self.agents_txt.delete("1.0", tk.END)
        for a in d.get("agents") or []:
            out = a.get("output") or {}
            line = f"[{a.get('agent')}] {a.get('status')}\n"
            for k, v in out.items():
                line += f"    {k}: {v}\n"
            if a.get("provenance"):
                line += f"    provenance: {a['provenance']}\n"
            self.agents_txt.insert(tk.END, line + "\n")
        if self.detail_panel:
            self.detail_panel["text"] = self._fmt_detail(d)

    @staticmethod
    def _blocking_reason(blocking: list) -> str:
        """MANDATORY 缺失字段 → 人话原因（委托 runner 共享映射）。"""
        from .runner import CashGameRunner
        return CashGameRunner.blocking_reason(blocking)

    def _show(self, res):
        self.btn.config(state="normal", text="▶  视觉 Agent开始工作")
        if isinstance(res, str):
            self.status_lbl.config(text="● ERROR", fg="#e74c3c")
            self.action_lbl.config(text="ERROR", fg="#e74c3c")
            self.meta_lbl.config(text=res, fg="#e74c3c")
            return
        d = res.to_dict() if hasattr(res, "to_dict") else res
        status = d.get("status", "UNKNOWN")
        if status == "OK":
            self.status_lbl.config(text="● READY", fg="#2dc653")
            action = str(d.get("decision_action") or "—").upper()
            self.action_lbl.config(text=action, fg="#2dc653",
                                   font=("Segoe UI", 34, "bold"))
            conf = d.get("confidence")
            solver = d.get("solver_mode", "UNKNOWN")
            meta = (f"Confidence: {conf if conf is not None else 'n/a'}"
                    f"    Status: {status}    Solver: {solver}")
            self.meta_lbl.config(text=meta, fg="#c8d6e5")
        else:
            self.status_lbl.config(text=f"● {status}", fg="#e74c3c")
            self.action_lbl.config(text=status, fg="#e74c3c",
                                   font=("Segoe UI", 22, "bold"))
            reason = d.get("reason") or "无法形成安全决策"
            blocking = ", ".join(d.get("blocking_fields") or [])
            meta = reason + (f"\n阻塞字段: {blocking}" if blocking else "")
            self.meta_lbl.config(text=meta, fg="#e74c3c")
        # Agent 输出（结构化）
        self.agents_txt.delete("1.0", tk.END)
        for a in d.get("agents") or []:
            out = a.get("output") or {}
            line = f"[{a.get('agent')}] {a.get('status')}\n"
            for k, v in out.items():
                line += f"    {k}: {v}\n"
            if a.get("provenance"):
                line += f"    provenance: {a['provenance']}\n"
            self.agents_txt.insert(tk.END, line + "\n")
        # 详细数据
        if self.detail_panel:
            self.detail_panel["text"] = self._fmt_detail(d)

    @staticmethod
    def _fmt_detail(d: dict) -> str:
        import json
        try:
            return json.dumps(d, ensure_ascii=False, indent=1, default=str)
        except Exception:
            return str(d)

    def _toggle_detail(self):
        if self.detail_panel is None:
            self.detail_panel = scrolledtext.ScrolledText(
                self.root, height=8, font=("Consolas", 8), bg="#111820",
                fg="#5d7290", relief="flat")
            self.detail_panel.pack(fill="both", padx=24, pady=4,
                                   before=self.detail_btn)
            self.detail_btn.config(text="▲ 收起详细数据")
        else:
            self.detail_panel.destroy()
            self.detail_panel = None
            self.detail_btn.config(text="▼ 展开详细数据")


def main():
    root = tk.Tk()
    CashGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
