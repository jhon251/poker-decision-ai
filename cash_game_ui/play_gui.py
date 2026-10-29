"""CASH-GAME AI — 对打界面 v1（EXE-1 产品；Tkinter，零新依赖）。

产品形态：与池 bot 6-max 对打（非实时辅助红线内——纯本地对打，无屏幕
捕捉/无自动操作路径；本模块不 import cash_game_ui.runner / vision2）。

交互：
  「新会话」→ 选档位/手数/种子/AI参考 → 后台线程跑 run_play_session；
  事件经队列回 UI 线程渲染（牌桌 6 座位/公共牌/底池）；
  轮到玩家 → 动作按钮启用（弃牌/过牌/跟注/加注滑条/全下）；
  每次行动后 AI 参考建议面板显示 provenance + 与你动作的一致性；
  一手结束 → 解析面板（净收益/亮牌/建议一致率）。

诚实边界：AI 建议与用户动作分栏显示、绝不混淆；provenance 原样透传
（solver_hit/preflop_chart/exploit_deviation/solver_miss_fallback）。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from external_harness.engine.nseat_engine import position_labels
from poker_decision_ai.play.match import (
    DIFFICULTY_TIERS, LINEUP_PRESETS, HumanAction, run_play_session)

BG = "#0f1419"
FG = "#c8d6e5"
FG_DIM = "#88a0b8"
GREEN = "#2dc653"
AMBER = "#f5a623"
RED = "#e74c3c"
PANEL = "#111820"
CARD_BG = "#f5f2e9"

_SUITS = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
_STREET_CN = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌", "river": "河牌"}
_ACTION_CN = {"fold": "弃牌", "check": "过牌", "call": "跟注", "bet": "下注",
              "raise": "加注到", "all-in": "全下"}
_PROV_CN = {"solver_hit": "solver 命中（真实 CFR spot 库）",
            "preflop_chart": "翻前图表基线",
            "exploit_deviation": "有界剥削偏离",
            "solver_miss_fallback": "solver 未命中（诚实降级）"}
_TERMINAL_CN = {"fold_out": "全员弃牌", "showdown": "摊牌"}
#: 微额桌玩家画像中文名（原型定义见 pool_bots.FACTION_SPECS）
_ARCHETYPE_CN = {"loose_passive_fish": "松手鱼", "fish": "鱼",
                 "recreational": "娱乐玩家", "semi_reg": "半常规",
                 "reg": "常规(reg)", "nit": "紧手(nit)", "maniac": "疯子"}
_POS = ("BTN", "SB", "BB", "UTG", "MP", "CO")   # 6-max（引擎 position_labels 同表）

#: 事件回放节奏（ms）：bot 行动/翻街拉开间隔，制造对局临场感
_EVENT_DELAY_MS = {"hand_start": 350, "street": 680, "action": 520,
                   "decision_result": 260, "hero_turn": 180,
                   "hand_end": 1000}

# 牌面 PNG 资产（assets/ 由 build_assets.py 从 cards/svg 光栅化生成）
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
#: 缩放档位 → (牌面目录, 对手小牌目录, 毛毡文件)
_ZOOM_ASSETS = {0.8: ("cards_80", "cards_small_80", "table_felt_80.png"),
                1.0: ("cards", "cards_small", "table_felt.png"),
                1.2: ("cards_120", "cards_small_120", "table_felt_120.png")}
_CARD_RANK_FILE = {"T": "10"}
_CARD_SUIT_FILE = {"s": "spade", "h": "heart", "d": "diamond", "c": "club"}


def _card_file(c: str) -> str | None:
    """引擎牌面记号（如 'As'/'Th'）→ assets/cards 下的 PNG 文件名。"""
    c = str(c)
    if len(c) != 2:
        return None
    rank = _CARD_RANK_FILE.get(c[0], c[0])
    suit = _CARD_SUIT_FILE.get(c[1])
    if suit is None:
        return None
    return f"{rank}_of_{suit}.png"


def _card(c: str) -> str:
    c = str(c)
    return c[0] + _SUITS.get(c[1:], c[1:]) if len(c) == 2 else c


class PlayGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        from cash_game_ui.i18n import tr
        root.title(tr("app_title"))
        # 自适应：按屏幕高度选初始缩放档
        screen_h = root.winfo_screenheight()
        z0 = 1.2 if screen_h >= 1040 else (1.0 if screen_h >= 880 else 0.8)
        self.table_size = 6                     # 桌型（session_start 可更新）
        self._apply_zoom(z0)
        root.geometry(f"{self._z(980)}x{self._z(820)}")
        root.minsize(self._z(880), self._z(690))
        self.zoom_var = tk.DoubleVar(value=self.zoom)
        root.configure(bg=BG)
        self.events: queue.Queue = queue.Queue()
        self.actions: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.turn_ctx = None
        self.advisor = None
        self.lineup: dict[int, dict] = {}
        self.button_seat = 0
        self.stake = "nl2"
        self.session_running = False
        self._peeked_this_turn = False
        self._folded_seats: set[int] = set()
        self._winner_seats: set[int] = set()
        self._pending: deque = deque()          # 事件回放缓冲（节奏控制）
        self._dispatching = False
        self._deal_job_ids: list[str] = []      # 发牌动画定时器
        self._board_job_ids: list[str] = []     # 公共牌动画定时器
        self._bet_pills: dict[int, list[int]] = {s: [] for s in range(6)}
        self._snap: dict = {}                   # 桌面快照（缩放重建后重放渲染）
        self.replay_stop = False                # 回放停止标志
        self._pulse_job: str | None = None       # Hero 回合脉冲描边定时器
        self._pulse_on = False
        self.sound_on = tk.BooleanVar(value=True)
        # 会话实时统计
        self._sess_stake_label = ""
        self._sess_n_hands = 0
        self._cur_hand = 0
        self._stat_hands_done = 0
        self._stat_net_bb = 0.0
        self._stat_agree = 0
        self._stat_adv = 0
        self._build()
        self._make_menu()
        ico = os.path.join(_ASSET_DIR, "poker_ai.ico")
        if os.path.exists(ico):
            try:
                root.iconbitmap(ico)
            except Exception:                 # noqa: BLE001
                pass
        self._newbtn_job: str | None = None
        self._newbtn_on = False
        self._pulse_new_btn()
        root.after(120, self._pump_events)

    # ═══════════════════════ UI 构建 ═══════════════════════
    def _build(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top, text="poker-ai 对打模式", font=("Segoe UI", 15, "bold"),
                 fg="#ffffff", bg=BG).pack(side="left")
        from cash_game_ui.i18n import tr
        self.info_lbl = tk.Label(top, text=tr("not_started"), font=("Segoe UI", 10),
                                 fg=FG_DIM, bg=BG)
        self.info_lbl.pack(side="left", padx=16)
        from cash_game_ui.i18n import tr
        self.new_btn = tk.Button(top, text=tr("new_session"), font=("Segoe UI", 11),
                                 bg=GREEN, fg="#0f1419", relief="flat",
                                 command=self._on_new_session)
        self.new_btn.pack(side="right")
        self.replay_btn = tk.Button(top, text=tr("replay"), font=("Segoe UI", 11),
                                    bg="#1a2530", fg=FG, relief="flat",
                                    command=self._open_replay)
        self.replay_btn.pack(side="right", padx=(0, 8))
        self.report_btn = tk.Button(top, text=tr("report"), font=("Segoe UI", 11),
                                    bg="#1a2530", fg=FG, relief="flat",
                                    command=self._open_weakness_report)
        self.report_btn.pack(side="right", padx=(0, 8))
        from cash_game_ui.i18n import tr as _tr
        self.report_ext_btn = tk.Button(top, text=_tr("report_external"),
                                        font=("Segoe UI", 11),
                                        bg="#1a2530", fg=FG, relief="flat",
                                        command=lambda:
                                        self._open_weakness_report("external"))
        self.report_ext_btn.pack(side="right", padx=(0, 8))
        from cash_game_ui.i18n import tr as _tr
        self.import_btn = tk.Button(top, text=_tr("import_analysis"),
                                    font=("Segoe UI", 11),
                                    bg="#1a2530", fg=FG, relief="flat",
                                    command=self._open_import_analysis)
        self.import_btn.pack(side="right", padx=(0, 8))
        tk.Checkbutton(top, text=tr("sound"), variable=self.sound_on,
                       font=("Segoe UI", 9), bg=BG, fg=FG_DIM,
                       selectcolor="#1a2530", activebackground=BG,
                       activeforeground=FG).pack(side="right", padx=(0, 10))

        # 牌桌容器（缩放重建时只替换其中的 canvas）
        self.table_frame = tk.Frame(self.root, bg=BG)
        self.table_frame.pack(padx=self._z(12), pady=self._z(6))
        self._create_canvas()

        # 动作区
        act = tk.LabelFrame(self.root, text=tr("action_bar"), font=("Segoe UI", 10, "bold"),
                            fg=FG_DIM, bg=BG, bd=1, relief="solid")
        act.pack(fill="x", padx=12, pady=4)
        self.btn_advice = tk.Button(act, text=tr("view_ai"), font=("Segoe UI", 10),
                                    bg="#1a2530", fg=FG, relief="flat",
                                    width=11, state="disabled",
                                    command=self._on_peek)
        self.btn_advice.pack(side="left", padx=8, pady=8)
        self.btn_fold = tk.Button(act, text=tr("fold") + " (F)", font=("Segoe UI", 11, "bold"),
                                  bg="#5a2a2a", fg="#ffffff", relief="flat",
                                  width=8, state="disabled",
                                  command=lambda: self._submit("fold"))
        self.btn_fold.pack(side="left", padx=4)
        self.btn_check = tk.Button(act, text=tr("check") + " (C)", font=("Segoe UI", 11, "bold"),
                                   bg="#2a3a5a", fg="#ffffff", relief="flat",
                                   width=8, state="disabled",
                                   command=lambda: self._submit("check"))
        self.btn_check.pack(side="left", padx=4)
        self.btn_call = tk.Button(act, text=tr("call") + " (C)", font=("Segoe UI", 11, "bold"),
                                  bg="#2a5a3a", fg="#ffffff", relief="flat",
                                  width=8, state="disabled",
                                  command=lambda: self._submit("call"))
        self.btn_call.pack(side="left", padx=4)
        # 快捷尺寸：以"加注到"语义填入滑条（最小 / ½池 / ¾池 / 满池）
        self.btn_min = tk.Button(act, text=tr("min"), font=("Segoe UI", 9),
                                 bg="#1a2530", fg=FG, relief="flat", width=4,
                                 state="disabled",
                                 command=lambda: self._quick_size(None))
        self.btn_half = tk.Button(act, text=tr("half_pot"), font=("Segoe UI", 9),
                                  bg="#1a2530", fg=FG, relief="flat", width=4,
                                  state="disabled",
                                  command=lambda: self._quick_size(0.5))
        self.btn_threeq = tk.Button(act, text=tr("threeq_pot"), font=("Segoe UI", 9),
                                    bg="#1a2530", fg=FG, relief="flat", width=4,
                                    state="disabled",
                                    command=lambda: self._quick_size(0.75))
        self.btn_pot = tk.Button(act, text=tr("pot_full"), font=("Segoe UI", 9),
                                 bg="#1a2530", fg=FG, relief="flat", width=4,
                                 state="disabled",
                                 command=lambda: self._quick_size(1.0))
        self.btn_min.pack(side="left", padx=(6, 2))
        self.btn_half.pack(side="left", padx=2)
        self.btn_threeq.pack(side="left", padx=2)
        self.btn_pot.pack(side="left", padx=2)
        self.raise_lbl = tk.Label(act, text=tr("raise_to") + " 3.0bb", font=("Segoe UI", 9),
                                  fg=FG, bg=BG)
        self.raise_lbl.pack(side="left", padx=(16, 2))
        self.scale = tk.Scale(act, from_=1, to=100, orient="horizontal",
                              length=150, bg=BG, fg=FG_DIM,
                              highlightthickness=0, showvalue=0,
                              command=self._on_scale)
        self.scale.pack(side="left", padx=4)
        self.btn_raise = tk.Button(act, text=tr("raise") + " (R)", font=("Segoe UI", 11, "bold"),
                                   bg="#7a5a1a", fg="#ffffff", relief="flat",
                                   width=8, state="disabled",
                                   command=lambda: self._submit("aggressive"))
        self.btn_raise.pack(side="left", padx=4)
        self.btn_allin = tk.Button(act, text=tr("allin") + " (A)", font=("Segoe UI", 11, "bold"),
                                   bg="#7a1a1a", fg="#ffffff", relief="flat",
                                   width=7, state="disabled",
                                   command=lambda: self._submit("all-in"))
        self.btn_allin.pack(side="left", padx=4)

        # 面板：AI 参考 + 逐手解析 + 日志
        panes = tk.Frame(self.root, bg=BG)
        panes.pack(fill="both", expand=True, padx=12, pady=6)
        left = tk.LabelFrame(panes, text=tr("advice_panel"),
                             font=("Segoe UI", 10, "bold"), fg=FG_DIM, bg=BG,
                             bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.advice_txt = scrolledtext.ScrolledText(
            left, height=9, font=("Consolas", 9), bg=PANEL, fg=FG,
            relief="flat", wrap="word", state="disabled")
        self.advice_txt.pack(fill="both", padx=6, pady=6, expand=True)
        right = tk.LabelFrame(panes, text=tr("log_panel"),
                              font=("Segoe UI", 10, "bold"), fg=FG_DIM, bg=BG,
                              bd=1, relief="solid")
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.log_txt = scrolledtext.ScrolledText(
            right, height=9, font=("Consolas", 9), bg=PANEL, fg=FG_DIM,
            relief="flat", wrap="word", state="disabled")
        self.log_txt.pack(fill="both", padx=6, pady=6, expand=True)

        self.status_lbl = tk.Label(
            self.root, text="● " + tr("ready"),
            font=("Segoe UI", 9), fg=FG_DIM, bg=BG, anchor="w")
        self.status_lbl.pack(fill="x", padx=14, pady=(0, 8))

        # 键盘快捷键（输入框聚焦时自动失效，见 _kb_guard）
        for _key, _fn in (("f", self._kb_fold), ("c", self._kb_checkcall),
                          ("r", self._kb_raise), ("a", self._kb_allin)):
            self.root.bind(_key, _fn)
            self.root.bind(_key.upper(), _fn)

    def _open_import_analysis(self):
        from cash_game_ui import license as _lic
        if not _lic.has_feature("import"):
            self._upgrade_prompt("import")
            return
        dlg = ImportAnalysisDialog(self)
        self._import_dialog = dlg

    def _upgrade_prompt(self, feature: str):
        """受限功能 → 升级提示框（含所需档位与联系方式占位）。"""
        from cash_game_ui import license as _lic
        name = _lic.FEATURE_LABELS.get(feature, feature)
        need = next((t for t, feats in _lic.FEATURES_BY_TIER.items()
                     if feature in feats and t != "trial"), "pro")
        messagebox.showinfo(
            "功能未包含",
            f"「{name}」属于{_lic.tier_label(need)}功能。\n"
            f"当前档位：{_lic.tier_label(_lic.current_tier())}\n\n"
            f"升级请联系发行方获取新激活码（机器码见激活窗口）。")

    def _format_import_card(self, fd: dict) -> str:
        """final_decision → 可读建议卡片（大字建议 + 依据 + 不确定性计数）。"""
        if not fd:
            return ""
        act = _ACTION_CN.get(str(fd.get("action") or ""), str(fd.get("action") or ""))
        size = fd.get("recommended_size")
        conf = fd.get("confidence")
        alt = fd.get("alternative_action")
        prov = fd.get("provenance") or {}
        unc = fd.get("uncertainty") or []
        keys = fd.get("key_evidence") or []
        bar_n = int(round((conf or 0) * 10))
        bar = "█" * bar_n + "░" * (10 - bar_n)
        L = []
        L.append("━━ 建议 " + "─" * 14)
        line = f"  {act}"
        if isinstance(size, (int, float)) and size > 0:
            # 动作词已含"到"（加注到）则不再重复
            line += ("" if act.endswith("到") else " 到") + f" {size:.1f}bb"
        if isinstance(conf, (int, float)):
            line += f"   置信 {conf * 100:.0f}% {bar}"
        L.append(line)
        if alt:
            L.append("  备选: " + _ACTION_CN.get(str(alt), str(alt)))
        L.append("━━ 依据 " + "─" * 14)
        for k in keys[:2]:
            L.append(f"  · {k}")
        src = str(prov.get("source") or "—")
        anchored = ("已锚定" if prov.get("anchored")
                    else "未命中 → 诚实降级")
        L.append(f"  solver 来源: {src}（{anchored}）")
        if unc:
            L.append(f"  不确定性 {len(unc)} 条（详见完整报告）")
        return "\n".join(L) + "\n"

    def _finish_import(self, ev: dict):
        """导入分析结果 → 建议面板 + JSON 落盘。"""
        dlg = getattr(self, "_import_dialog", None)
        dlg_alive = dlg is not None and bool(dlg.winfo_exists())
        if dlg_alive:
            dlg.go_btn.config(state="normal")
            dlg._busy = False
        if not ev.get("ok"):
            err = ev.get("error") or "未知错误"
            self._advice(f"[导入分析] 失败：{err}\n")
            if dlg_alive:
                dlg.status.config(text=f"分析失败：{err}", fg=RED)
            return
        data = ev["data"]
        out = ""
        try:
            os.makedirs("logs", exist_ok=True)
            out = os.path.join(
                "logs", "import_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:                       # noqa: BLE001
            out = ""
        fd = data.get("final_decision") or {}
        status = data.get("status")
        self._advice(f"[导入分析] 状态 {status} · "
                     f"延迟 {data.get('latency_ms')}ms\n")
        self._advice(self._format_import_card(fd))
        for note_key, note_label in (("multiway_note", "多人池提示"),
                                     ("to_call_note", "to_call 未知提示")):
            note = data.get(note_key)
            if note:
                self._advice(f"    {note_label}: {note}\n")
        self._advice("    完整报告: " + (out or "（未落盘）") + "\n")
        if dlg_alive:
            dlg.status.config(text=f"分析完成：状态 {status}", fg=GREEN)

    def _open_weakness_report(self, source: str = "play"):
        from cash_game_ui import license as _lic
        if not _lic.has_feature("report"):
            self._upgrade_prompt("report")
            return
        from cash_game_ui.weakness_report import build_report
        win = tk.Toplevel(self.root)
        win.title("外部牌谱分析（CoinPoker 导入）" if source == "external"
                  else "长期弱点报告（AI 对打）")
        win.configure(bg=BG)
        win.geometry(f"{self._z(760)}x{self._z(600)}")
        txt = scrolledtext.ScrolledText(win, font=("Consolas", self._zf(10)),
                                        bg=PANEL, fg=FG, relief="flat",
                                        wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", build_report(os.path.join("data", "play_history"),
                                       source=source))
        txt.config(state="disabled")

    def _open_replay(self):
        ReplayBrowser(self)

    # ═══════════════════════ 牌桌渲染 ═══════════════════════
    #: 座位几何基准（zoom=1.0 时的 canvas 940x384）；实际值由 _apply_zoom 缩放。
    #: 按桌型取用（seat0 恒为 Hero 下方位）；2 人桌对手在上方中央。
    _BASE_SEATS_BY_SIZE = {
        6: [(470, 302), (140, 245), (140, 88), (470, 78), (800, 88), (800, 245)],
        5: [(470, 302), (140, 245), (140, 88), (800, 88), (800, 245)],
        4: [(470, 302), (140, 245), (470, 78), (800, 88)],
        3: [(470, 302), (140, 180), (800, 180)],
        2: [(470, 302), (470, 78)],
    }
    _BASE_TABLE = (940, 384)

    def _z(self, v: float) -> int:
        """基准几何 → 当前缩放下的整数像素。"""
        return int(round(v * self.zoom))

    def _zf(self, v: float) -> int:
        """字号缩放（设下限避免过小不可读）。"""
        return max(7, int(round(v * self.zoom)))

    def _apply_zoom(self, z: float):
        """应用缩放档位：几何、字体基准与对应尺寸的资产目录。"""
        self.zoom = z
        cards_dir, small_dir, felt = _ZOOM_ASSETS[z]
        self._cards_dir = os.path.join(_ASSET_DIR, cards_dir)
        self._cards_small_dir = os.path.join(_ASSET_DIR, small_dir)
        self._felt_file = os.path.join(_ASSET_DIR, felt)
        self._BIG = (self._z(66), self._z(92))       # hero 手牌/公共牌
        self._SMALL = (self._z(44), self._z(62))     # 对手座位卡牌
        self._CHIP_R = self._z(11)
        self._CHIP_STEP = self._z(6)
        self.n_seats = self.table_size          # 渲染循环用的座位数
        self._SEAT_XY = tuple((self._z(x), self._z(y))
                              for x, y in
                              self._BASE_SEATS_BY_SIZE[self.table_size])
        self._BOARD_Y = self._z(170)
        self._BOARD_DX = self._z(72)
        self._table_w, self._table_h = (self._z(v) for v in self._BASE_TABLE)
        self._img_cache: dict[str, tk.PhotoImage] = {}
        self._img_small_cache: dict[str, tk.PhotoImage] = {}

    def _img(self, name: str) -> tk.PhotoImage | None:
        """加载当前档位 assets 下的牌面（hero/公共牌尺寸），带缓存。"""
        img = self._img_cache.get(name)
        if img is None and name:
            path = os.path.join(self._cards_dir, name)
            if os.path.exists(path):
                img = tk.PhotoImage(file=path, master=self.root)
                self._img_cache[name] = img
        return img

    def _img_small(self, name: str) -> tk.PhotoImage | None:
        """加载当前档位 assets 下的对手小牌，带缓存。"""
        img = self._img_small_cache.get(name)
        if img is None and name:
            path = os.path.join(self._cards_small_dir, name)
            if os.path.exists(path):
                img = tk.PhotoImage(file=path, master=self.root)
                self._img_small_cache[name] = img
        return img

    def _pulse_new_btn(self):
        """未开局时让「新会话」按钮呼吸闪烁（首用引导）；开局后停止。"""
        if self.session_running:
            self._newbtn_job = None
            return
        self._newbtn_on = not self._newbtn_on
        self.new_btn.config(bg="#8fe08a" if self._newbtn_on else GREEN)
        self._newbtn_job = self.root.after(550, self._pulse_new_btn)

    def _create_canvas(self):
        self.table = tk.Canvas(self.table_frame, width=self._table_w,
                               height=self._table_h, bg="#0b3d2e",
                               highlightthickness=1,
                               highlightbackground="#12503c")
        self.table.pack()
        self._draw_static_table()

    def _make_menu(self):
        from cash_game_ui.i18n import current_lang, set_lang, tr
        self.lang_var = tk.StringVar(value=current_lang())
        bar = tk.Menu(self.root)
        view = tk.Menu(bar, tearoff=0)
        z_names = {"zh": ("缩放 80%（小屏）", "缩放 100%", "缩放 120%（大屏）"),
                   "en": ("Zoom 80% (small)", "Zoom 100%", "Zoom 120% (large)")}
        zn = z_names.get(current_lang(), z_names["zh"])
        for z, name in ((0.8, zn[0]), (1.0, zn[1]), (1.2, zn[2])):
            view.add_radiobutton(label=name, value=z,
                                 variable=self.zoom_var,
                                 command=self._on_zoom_menu)
        bar.add_cascade(label=tr("view"), menu=view)
        lang_menu = tk.Menu(bar, tearoff=0)
        lang_menu.add_radiobutton(label="中文", value="zh",
                                  variable=self.lang_var, command=self._on_lang)
        lang_menu.add_radiobutton(label="English", value="en",
                                  variable=self.lang_var, command=self._on_lang)
        bar.add_cascade(label=tr("language"), menu=lang_menu)
        self.root.config(menu=bar)

    def _on_lang(self):
        from cash_game_ui.i18n import set_lang
        set_lang(str(self.lang_var.get()))
        messagebox.showinfo("语言 / Language",
                            "语言已切换，重启程序后生效。\n"
                            "Language changed — restart to apply.")

    def _on_zoom_menu(self):
        z = float(self.zoom_var.get())
        if z == self.zoom:
            return
        self._apply_zoom(z)
        root = self.root
        root.geometry(f"{self._z(980)}x{self._z(820)}")
        root.minsize(self._z(880), self._z(690))
        self.table_frame.pack(padx=self._z(12), pady=self._z(6))
        self._rebuild_table()

    def _rebuild_table(self):
        """缩放切换：取消动画定时器 → 重建 canvas → 重放桌面快照。"""
        self._stop_pulse()
        for jid in self._deal_job_ids + self._board_job_ids:
            try:
                self.root.after_cancel(jid)
            except Exception:                     # noqa: BLE001
                pass
        self._deal_job_ids, self._board_job_ids = [], []
        self.table.destroy()
        self._create_canvas()
        self._reapply_state()

    def _reapply_state(self):
        """把快照里的桌面状态渲染到（新）canvas 上。"""
        self._set_seat_names(self.lineup)
        stacks = self._snap.get("stacks")
        if stacks:
            self._render_stacks(stacks)
        self._folded_seats = set(self._snap.get("folded", ()))
        self._winner_seats = set(self._snap.get("winners", ()))
        self._refresh_seat_styles()
        self._board_set(self._snap.get("board", []))
        self._render_bets(self._snap.get("bets") or [])
        self._street_note = self._snap.get("street_note", "")
        self._render_pot(self._snap.get("pot", 0))
        hero = self._snap.get("hero_hole")
        if hero:
            self._seat_cards(0, tuple(_card_file(c) for c in hero))
        for seat in range(1, self.n_seats):
            self._seat_cards(seat, ("back", "back"))
        for r in self._snap.get("revealed", []):
            if r["seat"] != 0:
                self._seat_cards(r["seat"],
                                 tuple(_card_file(c) for c in r["hole"]))
        for seat in self._folded_seats:
            self._dim_folded_cards(seat)

    def _draw_static_table(self):
        if os.path.exists(self._felt_file):
            self._felt = tk.PhotoImage(file=self._felt_file, master=self.root)
            self.table.create_image(0, 0, image=self._felt, anchor="nw")
        else:  # 资产缺失时回退：旧椭圆桌
            self.table.create_oval(300, 110, 640, 235, outline="#12503c", width=2)
        self._pot_items: list[int] = []
        self._street_note = ""
        self._render_pot(0)
        # 公共牌槽（5 个虚线空槽，发牌后以图像覆盖）
        self.board_slots: list[int] = []
        for i in range(5):
            x = self._z(470) + (i - 2) * self._BOARD_DX
            y = self._BOARD_Y
            w2, h2 = self._BIG[0] // 2, self._BIG[1] // 2
            self.board_slots.append(self.table.create_rectangle(
                x - w2, y - h2, x + w2, y + h2,
                fill="#0c2c1c", outline="#2f6b4a", width=1, dash=(4, 3)))
        self._board_cards: list[int] = []
        self.seat_items: dict[int, dict] = {}
        for seat in range(self.n_seats):
            x, y = self._SEAT_XY[seat]
            cw, ch = self._BIG if seat == 0 else self._SMALL
            # 框刚好包住两张牌（牌间距 8 + 内边距 5），名字/筹码在框外上方
            hw = cw + self._z(9)
            hh = ch // 2 + self._z(5)
            box = self.table.create_rectangle(x - hw, y - hh, x + hw, y + hh,
                                              fill="#0d2b1d",
                                              outline="#2f6b4a", width=1)
            name = self.table.create_text(x, y - hh - self._z(23), text="",
                                          font=("Segoe UI", self._zf(9)),
                                          fill="#9fd3b4")
            stack = self.table.create_text(x, y - hh - self._z(9), text="—",
                                           font=("Segoe UI", self._zf(10), "bold"),
                                           fill="#ffffff")
            slots = []
            for k in (-1, 1):
                cx = x + k * (cw // 2 + self._z(4))
                slots.append(self.table.create_rectangle(
                    cx - cw // 2, y - ch // 2, cx + cw // 2, y + ch // 2,
                    fill="#0a2418", outline="#2f6b4a", width=1, dash=(4, 3)))
            self.seat_items[seat] = {"box": box, "name": name, "stack": stack,
                                     "slots": slots, "cards": [None, None],
                                     "overlays": [None, None],
                                     "bubble": [], "bubble_job": None}
        self._set_seat_names({})

    def _set_seat_names(self, lineup: dict[int, dict]):
        """座位只显示编号与位置；bot 画像不在桌面透露（防"读脸"式针对，
        画像数据仍完整记录在训练数据与弱点报告中）。"""
        labels = position_labels(self.n_seats)
        for seat in range(self.n_seats):
            pos = labels[(seat - self.button_seat) % self.n_seats]
            arch = "你（Hero）" if seat == 0 else ""
            self.table.itemconfigure(
                self.seat_items[seat]["name"],
                text=f"seat{seat} {pos}" + (f"  {arch}" if arch else ""))
        for seat in range(self.n_seats, 6):
            if seat in self.seat_items:
                for key in ("box", "name", "stack"):
                    self.table.itemconfigure(self.seat_items[seat][key],
                                             state="hidden")
                for oid in self.seat_items[seat]["cards"]:
                    if oid:
                        self.table.delete(oid)
                self.seat_items[seat]["cards"] = [None, None]
        self._refresh_seat_styles()

    def _refresh_seat_styles(self):
        """座位框配色：Hero 高亮、按钮位描边、弃牌变暗、赢家金框。"""
        for seat, items in self.seat_items.items():
            if seat >= self.n_seats:
                continue
            fill, outline = "#0d2b1d", "#2f6b4a"
            if seat == self.button_seat:
                fill, outline = "#113a26", "#57b678"
            if seat == 0:
                fill, outline = "#14401f", "#63c77f"
            if seat in self._winner_seats:
                fill, outline = "#3d3510", "#d9a821"
            if seat in self._folded_seats:
                fill, outline = "#0a1f16", "#274d38"
            self.table.itemconfigure(items["box"], fill=fill, outline=outline)
            name_fill = "#5c7a68" if seat in self._folded_seats else "#9fd3b4"
            if seat in self._winner_seats:
                name_fill = "#ffd76e"
            elif seat == 0 and seat not in self._folded_seats:
                name_fill = "#d9f2e1"
            self.table.itemconfigure(items["name"], fill=name_fill)

    def _seat_hw(self, seat: int) -> int:
        """座位框半宽（由牌尺寸推算，框 = 刚好装下两张牌）。"""
        return ((self._BIG[0] if seat == 0 else self._SMALL[0])
                + self._z(9))

    def _seat_cards(self, seat: int, names: tuple[str | None, str | None]):
        """设置座位两张牌：None=空槽，'back'=牌背，其余=牌面 PNG 文件名。"""
        items = self.seat_items[seat]
        for oid in items.get("overlays", (None, None)):   # 重发牌时清掉弃牌遮罩
            if oid is not None:
                self.table.delete(oid)
        items["overlays"] = [None, None]
        for idx, card_name in enumerate(names):
            self._place_seat_card(seat, idx, card_name)

    def _place_seat_card(self, seat: int, idx: int, card_name: str | None):
        """放置/清除座位第 idx 张牌（也是发牌动画的调度单元）。"""
        items = self.seat_items[seat]
        x, y = self._SEAT_XY[seat]
        cw = self._BIG[0] if seat == 0 else self._SMALL[0]
        cx = x + (idx * 2 - 1) * (cw // 2 + self._z(4))
        old = items["cards"][idx]
        if old is not None:
            self.table.delete(old)
            items["cards"][idx] = None
        iid = None
        if card_name:
            if card_name == "back":       # 哨兵：牌背
                card_name = "back.png"
            img = (self._img(card_name) if seat == 0
                   else self._img_small(card_name))
            if img is not None:
                iid = self.table.create_image(cx, y, image=img)
        items["cards"][idx] = iid
        self.table.itemconfigure(items["slots"][idx],
                                 state="hidden" if iid else "normal")
        if iid is not None and seat in self._folded_seats:
            self._dim_folded_cards(seat)     # fold 早于发牌动画的时序兜底

    def _show_bubble(self, seat: int, text: str, hold_ms: int = 1600):
        """座位旁弹出动作气泡，短暂停留后消失。"""
        items = self.seat_items[seat]
        for iid in items.get("bubble", []):
            self.table.delete(iid)
        if items.get("bubble_job"):
            try:
                self.root.after_cancel(items["bubble_job"])
            except Exception:                     # noqa: BLE001
                pass
        x, y = self._SEAT_XY[seat]
        hw = self._seat_hw(seat)
        if seat == 0:
            bx, by = x + hw + self._z(54), y
        elif seat == 3:
            bx, by = x + hw + self._z(54), y - 20
        else:
            s = 1 if x < 470 else -1
            bx, by = x + s * (hw + self._z(54)), y - self._z(20)
        w2 = int((13 * len(text) + 18) * self.zoom) // 2
        winner = seat in self._winner_seats
        fill = "#f5d98d" if (seat == 0 or winner) else "#e8f0e9"
        outline = "#caa539" if (seat == 0 or winner) else "#2f6b4a"
        fg = "#4a3208" if (seat == 0 or winner) else "#12331f"
        rect = self.table.create_rectangle(bx - w2, by - self._z(15),
                                           bx + w2, by + self._z(15),
                                           fill=fill, outline=outline, width=1)
        txt = self.table.create_text(bx, by, text=text,
                                     font=("Segoe UI", self._zf(9), "bold"),
                                     fill=fg)
        items["bubble"] = [rect, txt]
        items["bubble_job"] = self.root.after(
            hold_ms, lambda: self._hide_bubble(seat))

    def _hide_bubble(self, seat: int):
        items = self.seat_items.get(seat)
        if not items:
            return
        for iid in items.get("bubble", []):
            self.table.delete(iid)
        items["bubble"] = []
        items["bubble_job"] = None

    #: 筹码面额（bb）与配色：红 0.1 / 蓝 0.5 / 绿 1.0
    _CHIP_DENOMS = ((1.0, "#3f9d5c", "#2b7343"),
                    (0.5, "#4477b0", "#33598a"),
                    (0.1, "#d34545", "#a83333"))
    _CHIP_R = 11            # 筹码半径
    _CHIP_STEP = 6          # 堆叠纵向间距
    _CHIP_COL_MAX = 8       # 单列最多几张，超出换列

    def _draw_chip(self, cx: float, cy: float, d: int, out: list[int],
                   top: bool = True):
        """画一枚筹码。top=True（堆顶/单枚）画描边 + 内环；
        堆内被覆盖的筹码只画色块，避免叠加出弹簧纹。"""
        _, color, inner = self._CHIP_DENOMS[d]
        r = self._CHIP_R
        out.append(self.table.create_oval(
            cx - r, cy - r + 3, cx + r, cy + r + 3,
            fill=inner, outline=inner, width=1))
        out.append(self.table.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline=inner if top else "", width=1))
        if top:
            out.append(self.table.create_oval(
                cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6,
                fill="", outline="#e9f5ec", width=1))

    @staticmethod
    def _split_chips(bet_chips: float) -> list[int]:
        """筹码金额 → 面额索引列表（贪心：绿 1bb → 蓝 0.5bb → 红 0.1bb）。"""
        units = round(bet_chips / 20.0 * 10)
        counts = ((units // 10, 0), (units % 10 // 5, 1), (units % 5, 2))
        return [d for cnt, d in counts for _ in range(cnt)]

    def _render_replay_advice(self, ev: dict):
        """回放：把该决策当时的 AI 参考写进建议面板（含一致性与来源）。"""
        adv = ev.get("replay_advice") or {}
        act = _ACTION_CN.get(ev["action"], ev["action"])
        street = _STREET_CN.get(str(ev.get("street") or ""), ev.get("street"))
        if not adv.get("available"):
            self._advice(f"[回放·{street}] 你: {act}  |  "
                         f"AI 参考不可用（{adv.get('reason')}）\n")
            return
        ai_act = _ACTION_CN.get(adv.get("action") or "", adv.get("action"))
        size = adv.get("size_bb")
        agree = "✓ 一致" if adv.get("action") == ev["action"] else "✗ 不一致"
        prov = adv.get("provenance") or {}
        src = _PROV_CN.get(str(prov.get("source") or ""),
                           str(prov.get("source") or "—"))
        self._advice(
            f"[回放·{street}] 你: {act}  |  AI 参考: {ai_act}"
            + (f" {size:.1f}bb" if isinstance(size, (int, float)) else "")
            + f"  [{agree}]\n    来源: {src}"
            + (f" @ {prov.get('node')}" if prov.get("node") else "")
            + f"  (延迟 {adv.get('latency_ms')}ms)\n")

    def _render_bets(self, bets):
        """每个座位本街已下注：筹码堆建模 + 下方金额（bb）。"""
        bets = list(bets or [])
        for seat in range(self.n_seats):
            for iid in self._bet_pills[seat]:
                self.table.delete(iid)
            self._bet_pills[seat] = []
            bet = bets[seat] if seat < len(bets) else 0
            if bet <= 0:
                continue
            x, y = self._SEAT_XY[seat]
            hw = self._seat_hw(seat)
            if seat == 0:
                px, py = x - hw - self._z(30), y
            elif seat == 3:
                px, py = x + hw + self._z(30), y + self._z(12)
            else:
                s = 1 if x < 470 else -1
                px, py = x + s * (hw + self._z(30)), y + self._z(20)
            chips = self._split_chips(bet)
            cols = [chips[i:i + self._CHIP_COL_MAX]
                    for i in range(0, len(chips), self._CHIP_COL_MAX)]
            for ci, col in enumerate(cols):
                cx = px + ci * (self._CHIP_R * 2 + 2)
                for i, d in enumerate(col):
                    self._draw_chip(cx, py - i * self._CHIP_STEP, d,
                                    self._bet_pills[seat],
                                    top=(i == len(col) - 1))
            self._bet_pills[seat].append(self.table.create_text(
                px + (len(cols) - 1) * (self._CHIP_R + 1),
                py + self._z(16), text=f"{round(bet / 20.0, 1):.1f}bb",
                font=("Segoe UI", self._zf(9), "bold"), fill="#ffd76e"))

    def _render_pot(self, pot_chips: int):
        """底池：金额 + 紧挨着的筹码堆（垂直堆叠，最多 8 枚/列换列，
        最多显示 24 枚，超出以金额文字为准）。"""
        for iid in self._pot_items:
            self.table.delete(iid)
        self._pot_items = []
        label = (f"底池 {round(pot_chips / 20.0, 1):.1f}bb"
                 if pot_chips > 0 else "底池 —")
        self._pot_items.append(self.table.create_text(
            self._z(540), self._z(371), text=label + self._street_note,
            font=("Segoe UI", self._zf(12), "bold"), fill="#f2f7f3",
            anchor="e"))
        chips = self._split_chips(pot_chips)[:24] if pot_chips > 0 else []
        for ncol, start in enumerate(range(0, len(chips), self._CHIP_COL_MAX)):
            cx = self._z(560) + ncol * (self._CHIP_R * 2 + 2)
            col = chips[start:start + self._CHIP_COL_MAX]
            for i, d in enumerate(col):
                self._draw_chip(cx, self._z(366) - i * self._CHIP_STEP, d,
                                self._pot_items,
                                top=(i == len(col) - 1))

    def _board_set(self, board: list[str]):
        """公共牌：增量渲染；一条街内新翻的牌带 150ms 间隔。"""
        for jid in self._board_job_ids:
            try:
                self.root.after_cancel(jid)
            except Exception:                     # noqa: BLE001
                pass
        self._board_job_ids = []
        while len(self._board_cards) > len(board):    # 新一手清桌
            iid = self._board_cards.pop()
            self.table.delete(iid)
        t = 0
        for i in range(5):
            if i < len(self._board_cards):            # 已亮出的牌保持
                continue
            name = _card_file(board[i]) if i < len(board) else None
            img = self._img(name) if name else None
            x = 470 + (i - 2) * self._BOARD_DX
            if img is not None:
                job = self.root.after(
                    t, lambda i=i, x=x, img=img:
                    self._place_board_card(i, x, img))
                self._board_job_ids.append(job)
                t += 150
            else:
                self.table.itemconfigure(self.board_slots[i], state="normal")

    def _place_board_card(self, idx: int, x: int, img):
        self._board_cards.append(
            self.table.create_image(x, self._BOARD_Y, image=img))
        self.table.itemconfigure(self.board_slots[idx], state="hidden")

    def _dim_folded_cards(self, seat: int):
        """在弃牌座位的两张牌上盖暗色半透明遮罩（stipple 抖动模拟透明）。"""
        items = self.seat_items[seat]
        cw, ch = self._BIG if seat == 0 else self._SMALL
        for idx, iid in enumerate(items["cards"]):
            if iid is None:
                continue
            if items["overlays"][idx] is not None:
                self.table.delete(items["overlays"][idx])
            cx, cy = self.table.coords(iid)
            items["overlays"][idx] = self.table.create_rectangle(
                cx - cw // 2 - 1, cy - ch // 2 - 1,
                cx + cw // 2 + 1, cy + ch // 2 + 1,
                fill="#06140d", outline="#06140d", stipple="gray50")

    def _render_stacks(self, stacks, folded=()):
        for seat in range(self.n_seats):
            if seat >= len(stacks):
                continue
            txt = f"{stacks[seat] / 20:.0f}bb"
            if seat in tuple(folded):
                txt += " (弃)"
            self.table.itemconfigure(self.seat_items[seat]["stack"], text=txt)

    # ═══════════════════════ 回合提醒与会话统计 ═══════════════════════
    def _start_pulse(self):
        """轮到 Hero：座位框琥珀色脉冲 + 可选系统提示音。"""
        self._stop_pulse()
        self._pulse_on = True
        if self.sound_on.get():
            self.root.bell()
        self._pulse_tick()

    def _pulse_tick(self):
        self._pulse_on = not self._pulse_on
        self.table.itemconfigure(self.seat_items[0]["box"],
                                 outline="#f5a623" if self._pulse_on
                                 else "#63c77f",
                                 width=2 if self._pulse_on else 1)
        self._pulse_job = self.root.after(400, self._pulse_tick)

    def _stop_pulse(self):
        if self._pulse_job is not None:
            try:
                self.root.after_cancel(self._pulse_job)
            except Exception:                     # noqa: BLE001
                pass
        self._pulse_job = None
        self._pulse_on = False
        if hasattr(self, "seat_items"):
            self._refresh_seat_styles()

    def _stats_text(self, cur_hand: int | None) -> str:
        s = f"● {self._sess_stake_label}  "
        s += (f"第 {cur_hand}/{self._sess_n_hands} 手" if cur_hand
              else f"计划 {self._sess_n_hands} 手")
        if self._stat_hands_done:
            s += f"  累计 {self._stat_net_bb:+.1f}bb"
        if self._stat_adv:
            s += f"  AI一致 {self._stat_agree}/{self._stat_adv}"
        return s

    def _reset_session_stats(self, stake_label: str, n_hands: int):
        self._sess_stake_label = stake_label
        self._sess_n_hands = int(n_hands)
        self._cur_hand = 0
        self._stat_hands_done = 0
        self._stat_net_bb = 0.0
        self._stat_agree = 0
        self._stat_adv = 0

    # ═══════════════════════ 会话控制 ═══════════════════════
    def _on_new_session(self):
        from cash_game_ui.i18n import tr
        if self.session_running:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(tr("dlg_title"))
        dlg.configure(bg=BG)
        dlg.grab_set()
        tk.Label(dlg, text=tr("dlg_stake"), fg=FG, bg=BG).grid(row=0, column=0, padx=8, pady=6, sticky="e")
        from cash_game_ui import license as _lic
        stake_names = {"nl2": "nl2（$1/$2）", "nl5": "nl5（$2/$5）"}
        if _lic.has_feature("stake_high"):
            stake_names.update({"nl10": "nl10（$5/$10）",
                                "nl25": "nl25（$10/$25）"})
        if _lic.has_feature("stake_premium"):
            stake_names["nl50"] = "nl50（$25/$50）"
        stake_cb = ttk.Combobox(dlg, values=list(stake_names),
                                state="readonly", width=16)
        stake_cb.current(0)
        stake_cb.grid(row=0, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_table"), fg=FG, bg=BG).grid(row=1, column=0, padx=8, pady=6, sticky="e")
        size_names = {"6 人桌": 6, "5 人桌": 5, "4 人桌": 4,
                      "3 人桌": 3, "单挑（2 人）": 2}
        size_cb = ttk.Combobox(dlg, values=list(size_names), state="readonly",
                               width=10)
        size_cb.current(0)
        size_cb.grid(row=1, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_stack"), fg=FG, bg=BG).grid(row=2, column=0, padx=8, pady=6, sticky="e")
        stack_cb = ttk.Combobox(dlg, values=["50bb", "100bb", "200bb"],
                                state="readonly", width=10)
        stack_cb.current(1)
        stack_cb.grid(row=2, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_focus"), fg=FG, bg=BG).grid(row=3, column=0, padx=8, pady=6, sticky="e")
        focus_names = {"全位置轮换": "auto", "固定按钮位（BTN）": "btn",
                       "固定小盲（SB）": "sb", "固定大盲（BB）": "bb"}
        focus_cb = ttk.Combobox(dlg, values=list(focus_names), state="readonly",
                                width=18)
        focus_cb.current(0)
        focus_cb.grid(row=3, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_hands"), fg=FG, bg=BG).grid(row=4, column=0, padx=8, pady=6, sticky="e")
        hands_sb = tk.Spinbox(dlg, from_=1, to=200, width=12)
        hands_sb.delete(0, "end"); hands_sb.insert(0, "6")
        hands_sb.grid(row=4, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_seed"), fg=FG, bg=BG).grid(row=5, column=0, padx=8, pady=6, sticky="e")
        seed_en = tk.Entry(dlg, width=14)
        seed_en.grid(row=5, column=1, pady=6)
        tk.Label(dlg, text=tr("dlg_difficulty"), fg=FG, bg=BG).grid(row=6, column=0,
                                                        padx=8, pady=6, sticky="e")
        tier_names = {t: DIFFICULTY_TIERS[t]["label"]
                      for t in ("easy", "normal", "hard", "max")
                      if (t != "hard" or _lic.has_feature("difficulty_hard"))
                      and (t != "max" or _lic.has_feature("difficulty_max"))}
        diff_cb = ttk.Combobox(dlg, values=list(tier_names.values()),
                               state="readonly", width=38)
        diff_cb.current(1)
        diff_cb.grid(row=6, column=1, sticky="w", pady=6)
        tier_key = {"v": "normal"}

        def on_diff(_e=None):
            for k, name in tier_names.items():
                if name == diff_cb.get():
                    tier_key["v"] = k
            diff_desc.config(text=DIFFICULTY_TIERS[tier_key["v"]]["label"])
        diff_cb.bind("<<ComboboxSelected>>", on_diff)
        diff_desc = tk.Label(dlg, text=DIFFICULTY_TIERS["normal"]["label"],
                             font=("Segoe UI", 8), fg=FG_DIM, bg=BG)
        diff_desc.grid(row=7, column=1, sticky="w")
        tk.Label(dlg, text=tr("dlg_advice"), fg=FG, bg=BG).grid(row=8, column=0, padx=8, pady=6, sticky="e")
        advisor_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dlg, variable=advisor_var, bg=BG, fg=FG,
                       text=tr("dlg_advice_on")).grid(
            row=8, column=1, sticky="w", pady=6)

        def start():
            try:
                hands = max(1, int(hands_sb.get()))
            except ValueError:
                hands = 6
            seed_txt = seed_en.get().strip()
            seed = int(seed_txt) if seed_txt.isdigit() else None
            from cash_game_ui import license as _lic2
            stake = {v: k for k, v in stake_names.items()}.get(
                stake_cb.get(), "nl2")
            if (stake in ("nl10", "nl25") and not _lic2.has_feature("stake_high"))                     or (stake == "nl50" and not _lic2.has_feature("stake_premium")):
                messagebox.showwarning("档位未包含",
                                       "该盲注档位属于更高订阅档。", parent=dlg)
                return
            table_size = size_names.get(size_cb.get(), 6)
            stack_bb = int(stack_cb.get().replace("bb", "") or 100)
            focus_position = focus_names.get(focus_cb.get(), "auto")
            dlg.destroy()
            diff = tier_key["v"]
            if diff == "hard" and not _lic2.has_feature("difficulty_hard"):
                diff = "normal"
            if diff == "max" and not _lic2.has_feature("difficulty_max"):
                diff = "normal"
            self._start_worker(stake, hands, seed, advisor_var.get(),
                               difficulty=diff,
                               table_size=table_size, stack_bb=stack_bb,
                               focus_position=focus_position)

        tk.Button(dlg, text=tr("dlg_start"), bg=GREEN, fg="#0f1419", relief="flat",
                  command=start).grid(row=9, column=0, columnspan=2, pady=10)

    def _start_worker(self, stake: str, hands: int, seed: int | None,
                      advisor_on: bool, difficulty: str = "normal",
                      table_size: int = 6, stack_bb: int = 100,
                      focus_position: str = "auto"):
        self.session_running = True
        self.new_btn.config(state="disabled")
        self.stake = stake
        self._clear_panels()
        self._log(f"新会话：{LINEUP_PRESETS[stake]['name']}"
                  f"｜{DIFFICULTY_TIERS[difficulty]['label']}  {hands} 手  "
                  f"{'种子复盘(LCG 确定性)' if seed is not None else '随机实战(强随机洗牌，不可复盘)'}"
                  f"  AI参考={'开' if advisor_on else '关'}")

        def work():
            try:
                judge = None
                if advisor_on:
                    self._status("构建 AI 参考引擎……", AMBER)
                    from external_harness.adapter.judge_adapter import (
                        JudgeHeroController)
                    from external_harness.protocol.rake_model import (
                        RakeModel, load_rake_config)
                    judge = JudgeHeroController(overrides={
                        "realtime": {"deadline_ms": 60000,
                                     "agent_timeout_ms": 30000},
                        "app": {"logs_dir": "logs/play_gui"}})
                    rake = RakeModel(load_rake_config(), stake=stake)
                else:
                    rake = None
                from poker_decision_ai.play.history import PlayHistoryWriter
                import os
                import time as _t
                import uuid as _u
                sid = _t.strftime("%Y%m%d_%H%M%S") + "-gui-" + _u.uuid4().hex[:4]
                seed_base = (seed if seed is not None else
                             int.from_bytes(os.urandom(8), "little")
                             & ((1 << 63) - 1))
                # 双轨洗牌：填种子=复盘（LCG 确定性）；不填=实战（强随机）
                shuffle_algorithm = "lcg" if seed is not None else "python"
                hist = PlayHistoryWriter("data/play_history", sid, stake=stake,
                                         label=LINEUP_PRESETS[stake]["name"],
                                         seed_base=seed_base,
                                         lineup=[], deterministic=seed is not None,
                                         table_size=table_size,
                                         stack_bb=stack_bb,
                                         focus_position=focus_position)
                summary = run_play_session(
                    stake=stake, n_hands=hands, seed_base=seed_base,
                    get_human_action=lambda ctx: self.actions.get(),
                    on_event=self.events.put, judge=judge, rake=rake,
                    history=hist, session_id=sid,
                    shuffle_algorithm=shuffle_algorithm,
                    difficulty=difficulty)
                self.events.put({"type": "_history_path",
                                 "path": str(hist.path)})
                if judge is not None:
                    try:
                        import asyncio
                        asyncio.run(judge.runtime.close())
                    except Exception:                         # noqa: BLE001
                        pass
            except Exception as exc:                          # noqa: BLE001
                self.events.put({"type": "_worker_error", "error": str(exc)[:300]})

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    # ═══════════════════════ 动作提交 ═══════════════════════
    def _submit(self, kind: str):
        c = self.turn_ctx
        if c is None:
            return
        if kind == "aggressive":
            act = "raise" if "raise" in c.legal_actions else "bet"
            amount = int(self.scale.get() * c.bb_chips)
        elif kind == "all-in":
            act, amount = "all-in", None
        else:
            act, amount = kind, None
        self._set_buttons_enabled(False)
        self.turn_ctx = None
        self.actions.put(HumanAction(act, amount,
                                     peeked_advice=self._peeked_this_turn))
        self._peeked_this_turn = False

    def _on_peek(self):
        """行动前查看 AI 参考（轮询 collect，不阻塞 UI）。"""
        self._peeked_this_turn = True
        if self.advisor is None:
            return
        adv = self.advisor.collect(timeout_s=0.05)
        if adv.get("reason") == "advisor_timeout":
            self._status("AI 参考计算中……", AMBER)
            self.root.after(250, self._on_peek)
            return
        self._show_advice(adv, prefix="[主动查看]\n")

    def _on_scale(self, _v=None):
        self.raise_lbl.config(text=f"加注到 {self.scale.get():.1f}bb")

    def _quick_size(self, frac: float | None):
        """快捷尺寸：按'加注到'语义填滑条。frac=None → 最小加注；
        否则 = 跟注后底池的 frac（标准 pot-size 语义）。"""
        c = self.turn_ctx
        if c is None or c.min_raise_to_chips is None:
            return
        lo = c.min_raise_to_chips / c.bb_chips
        hi = c.max_raise_to_chips / c.bb_chips
        if frac is None:
            target = lo
        else:
            pot_after_call = c.pot_chips + c.to_call_chips
            target = (c.to_call_chips + pot_after_call * frac) / c.bb_chips
        self.scale.set(round(max(lo, min(round(target, 1), hi)), 1))
        self._on_scale()

    def _kb_guard(self, event=None) -> bool:
        """输入控件聚焦时屏蔽快捷键，避免打字误触动作。
        优先用事件自带的 widget（比 focus_get 可靠）。"""
        w = getattr(event, "widget", None) or self.root.focus_get()
        return isinstance(w, (tk.Entry, tk.Spinbox, tk.Text,
                              scrolledtext.ScrolledText, ttk.Combobox))

    def _kb_fold(self, e=None):
        if not self._kb_guard(e) and str(self.btn_fold["state"]) == "normal":
            self._submit("fold")

    def _kb_checkcall(self, e=None):
        if self._kb_guard(e):
            return
        if str(self.btn_check["state"]) == "normal":
            self._submit("check")
        elif str(self.btn_call["state"]) == "normal":
            self._submit("call")

    def _kb_raise(self, e=None):
        if not self._kb_guard(e) and str(self.btn_raise["state"]) == "normal":
            self._submit("aggressive")

    def _kb_allin(self, e=None):
        if not self._kb_guard(e) and str(self.btn_allin["state"]) == "normal":
            self._submit("all-in")

    # ═══════════════════════ 事件泵（UI 线程，带回放节奏） ═══════════════════════
    def _pump_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if self.replay_stop and ev.get("src") == "replay":
                    continue          # 回放被停止：丢弃尚未展示的事件
                self._pending.append(ev)
        except queue.Empty:
            pass
        self._schedule_next()
        self.root.after(120, self._pump_events)

    def _event_delay(self, ev: dict) -> int:
        return _EVENT_DELAY_MS.get(ev.get("type"), 60)

    def _schedule_next(self):
        """逐个派发缓冲事件；重节奏事件后停顿，制造发牌/行动节奏。"""
        if self._dispatching or not self._pending:
            return
        ev = self._pending.popleft()
        self._dispatching = True
        self._dispatch(ev)
        self.root.after(self._event_delay(ev), self._dispatch_done)

    def _dispatch_done(self):
        self._dispatching = False
        self._schedule_next()

    def _dispatch(self, ev: dict):
        t = ev.get("type")
        if t == "session_start":
            self.lineup = {s["seat"]: s for s in ev["lineup"]}
            self.button_seat = 0
            self.table_size = int(ev.get("table_size", 6))
            self._apply_zoom(self.zoom)          # 依据新桌型重算座位几何
            self._bet_pills = {s: [] for s in range(self.table_size)}
            self._snap = {"board": [], "bets": [], "pot": 0, "stacks": None,
                          "folded": [], "winners": [], "revealed": [],
                          "hero_hole": None, "street_note": ""}
            self._rebuild_table()                # 重建座位并复位桌面
            if self._newbtn_job is not None:
                try:
                    self.root.after_cancel(self._newbtn_job)
                except Exception:                 # noqa: BLE001
                    pass
                self._newbtn_job = None
                self.new_btn.config(bg=GREEN)
            if ev.get("src") == "replay":
                self._status("手牌回放中……", "#4da3ff")
            else:
                self._reset_session_stats(ev["stake_label"], ev["n_hands"])
                self.info_lbl.config(text=self._stats_text(None), fg=GREEN)
                self._status("会话运行中", GREEN)
        elif t == "hand_start":
            self.button_seat = ev["button"]
            self._set_seat_names(self.lineup)
            self._render_stacks(ev["stacks"])
            self._folded_seats = set()
            self._winner_seats = set()
            self._refresh_seat_styles()
            self._street_note = "（翻前）"
            self._render_pot(ev["pot_chips"])
            self._cur_hand = ev["hand_index"] + 1
            self._snap.update({"button": ev["button"],
                               "hero_hole": list(ev["hero_hole"]),
                               "stacks": list(ev["stacks"]),
                               "pot": ev["pot_chips"],
                               "bets": [0] * 6, "folded": [],
                               "winners": [], "revealed": [], "board": [],
                               "street_note": "（翻前）"})
            if ev.get("src") != "replay":
                self.info_lbl.config(text=self._stats_text(self._cur_hand))
            self._board_set([])
            self._render_bets([0] * self.n_seats)
            # 发牌动画：两圈轮发（对手先、Hero 最后），Hero 亮牌面
            for seat in range(self.n_seats):
                self._seat_cards(seat, (None, None))
            hero_hole = [_card_file(c) for c in ev["hero_hole"]]
            for job in self._deal_job_ids:
                try:
                    self.root.after_cancel(job)
                except Exception:                     # noqa: BLE001
                    pass
            self._deal_job_ids = []
            t = 250
            for rnd in (0, 1):
                for seat in (1, 2, 3, 4, 5, 0):
                    name = hero_hole[rnd] if (seat == 0 and rnd < len(hero_hole)) \
                        else ("back" if seat != 0 else None)
                    if name is None:
                        continue
                    job = self.root.after(
                        t, lambda s=seat, i=rnd, n=name:
                        self._place_seat_card(s, i, n))
                    self._deal_job_ids.append(job)
                    t += 95
            self._log(f"—— 第 {ev['hand_index'] + 1}/{ev['n_hands']} 手 "
                      f"(按钮 seat{ev['button']})  你的手牌: "
                      f"{' '.join(_card(c) for c in ev['hero_hole'])}")
        elif t == "street":
            self._board_set(ev["board"])
            self._render_bets(ev.get("street_bets"))
            self._street_note = ("（翻前）" if ev["street"] == "preflop" else
                                 f"（{_STREET_CN.get(ev['street'], ev['street'])}）")
            self._render_pot(ev["pot_chips"])
            self._render_stacks(ev["stacks"])
            self._snap.update({"board": list(ev["board"]),
                               "bets": list(ev.get("street_bets") or []),
                               "pot": ev["pot_chips"],
                               "stacks": list(ev["stacks"]),
                               "street_note": ("（翻前）" if ev["street"] == "preflop"
                                               else f"（{_STREET_CN.get(ev['street'], ev['street'])}）")})
            self._log(f"[{_STREET_CN.get(ev['street'], ev['street'])}] "
                      f"{' '.join(_card(c) for c in ev['board'])}")
        elif t == "action":
            who = "你" if ev.get("is_hero") else f"seat{ev['seat']}"
            act = _ACTION_CN.get(ev["action"], ev["action"])
            amt = ev.get("amount_chips")
            self._log(f"{who} {act}" + (f" {amt}ch" if amt else ""))
            if ev["action"] == "fold":        # 弃牌：座位与牌背整体变暗
                self._folded_seats.add(ev["seat"])
                self._refresh_seat_styles()
                self._dim_folded_cards(ev["seat"])
            bubble = act if amt is None else f"{act} {amt}ch"
            self._show_bubble(ev["seat"], bubble)
            if ev.get("src") == "replay" and ev.get("is_hero"):
                self._render_replay_advice(ev)
            self._render_bets(ev.get("street_bets"))
            self._render_pot(ev["pot_chips"])
            self._render_stacks(ev["stacks"])
            self._snap.update({"bets": list(ev.get("street_bets") or []),
                               "pot": ev["pot_chips"],
                               "stacks": list(ev["stacks"]),
                               "folded": sorted(self._folded_seats)})
        elif t == "hero_turn":
            self.turn_ctx = ev["context"]
            self.advisor = ev.get("advisor")
            self._peeked_this_turn = False
            self._setup_action_ui(self.turn_ctx)
            self._start_pulse()
        elif t == "decision_result":
            self._render_decision(ev)
            adv = ev.get("advisor") or {}
            if adv.get("available"):
                self._stat_adv += 1
                if ev.get("advisor_agree"):
                    self._stat_agree += 1
                self.info_lbl.config(text=self._stats_text(self._cur_hand))
        elif t == "hand_end":
            self._render_hand_end(ev)
        elif t == "session_end":
            s = ev.get("summary") or {}
            self._log(f"会话结束（{ev.get('reason')}）："
                      f"{s.get('n_completed')}/{s.get('n_hands')} 手  "
                      f"累计 {s.get('hero_total_bb_rakefree')} bb (rake-free)")
            self._status("会话结束——可开新会话", GREEN)
            self.session_running = False
            self.new_btn.config(state="normal")
            if ev.get("reason") == "completed":
                messagebox.showinfo(
                    "会话总结",
                    f"完成 {s.get('n_completed')}/{s.get('n_hands')} 手"
                    f"（中止 {s.get('n_aborted')}）\n"
                    f"累计 {s.get('hero_total_bb_rakefree') or 0.0:+.1f} bb"
                    f"（rake-free）\n"
                    f"bb/100: {s.get('bb_per_100_rakefree')}"
                    f"（含抽水 {s.get('bb_per_100_net')}）\n"
                    f"AI 参考一致 {self._stat_agree}/{self._stat_adv}\n"
                    f"用时 {s.get('wall_sec')}s")
        elif t == "_history_path":
            self._log(f"训练数据已落盘: {ev['path']}")
        elif t == "_import_result":
            self._finish_import(ev)
        elif t == "_worker_error":
            self._status(f"会话异常: {ev['error']}", RED)
            self.session_running = False
            self.new_btn.config(state="normal")
            messagebox.showerror("对打会话异常", ev["error"])
        elif t == "invalid_action":
            self._log(f"!! 非法动作被拒（{ev.get('reason')}）——重新选择")

    # ═══════════════════════ 面板细节 ═══════════════════════
    def _setup_action_ui(self, c):
        legal = list(c.legal_actions)
        to_call_bb = c.to_call_chips / c.bb_chips
        self.btn_fold.config(state="normal" if "fold" in legal else "disabled")
        self.btn_check.config(state="normal" if "check" in legal else "disabled")
        self.btn_call.config(state="normal" if "call" in legal else "disabled")
        agg = "raise" if "raise" in legal else ("bet" if "bet" in legal else None)
        if agg:
            lo_bb = c.min_raise_to_chips / c.bb_chips
            hi_bb = c.max_raise_to_chips / c.bb_chips
            self.scale.config(from_=round(lo_bb, 1), to=round(hi_bb, 1),
                              resolution=0.1)
            self.scale.set(round(max(lo_bb, min(2.5 * max(to_call_bb, 1), hi_bb)), 1))
            self.btn_raise.config(state="normal",
                                  text="加注 (R)" if agg == "raise" else "下注 (R)")
            self._on_scale()
        else:
            self.btn_raise.config(state="disabled")
        for b in (self.btn_min, self.btn_half, self.btn_threeq, self.btn_pot):
            b.config(state="normal" if agg else "disabled")
        self.btn_allin.config(state="normal" if "all-in" in legal else "disabled")
        self.btn_advice.config(
            state="normal" if self.advisor is not None else "disabled")
        need = f"  需跟 {to_call_bb:.1f}bb" if to_call_bb > 0 else "  可过牌"
        self._status(f"轮到你（{c.position_label}，{_STREET_CN.get(c.street, c.street)}）"
                     f"  底池 {c.pot_chips / c.bb_chips:.1f}bb{need}", AMBER)

    def _set_buttons_enabled(self, on: bool):
        if not on:
            self._stop_pulse()
            for b in (self.btn_fold, self.btn_check, self.btn_call,
                      self.btn_raise, self.btn_allin, self.btn_advice,
                      self.btn_min, self.btn_half, self.btn_threeq,
                      self.btn_pot):
                b.config(state="disabled")

    def _render_decision(self, ev: dict):
        adv = ev.get("advisor") or {}
        user = _ACTION_CN.get(ev.get("user_action") or "", ev.get("user_action"))
        if not adv.get("available"):
            self._advice(f"你: {user}  |  AI 参考不可用"
                         f"（{adv.get('reason')}）——如实记录\n")
            return
        act = _ACTION_CN.get(adv.get("action") or "", adv.get("action"))
        size = adv.get("size_bb")
        agree = ev.get("advisor_agree")
        mark = {True: "✓ 一致", False: "✗ 不一致", None: "—"}.get(agree, "—")
        prov = adv.get("provenance") or {}
        src = _PROV_CN.get(str(prov.get("source") or ""),
                           str(prov.get("source") or "—"))
        self._advice(
            f"你: {user}  |  AI 参考: {act}"
            + (f" {size:.1f}bb" if isinstance(size, (int, float)) else "")
            + f"  [{mark}]\n"
            + f"    来源: {src}"
            + (f" @ {prov.get('node')}" if prov.get("node") else "")
            + ("  [HU解锚定+多人简化]" if prov.get("multiway_simplified") else "")
            + f"  (延迟 {adv.get('latency_ms')}ms)\n")

    def _show_advice(self, adv: dict, prefix: str = ""):
        if not adv.get("available"):
            self._advice(prefix + f"AI 参考不可用（{adv.get('reason')}）\n")
            return
        act = _ACTION_CN.get(adv.get("action") or "", adv.get("action"))
        size = adv.get("size_bb")
        prov = adv.get("provenance") or {}
        src = _PROV_CN.get(str(prov.get("source") or ""),
                           str(prov.get("source") or "—"))
        self._advice(prefix + f"AI 参考: {act}"
                     + (f" {size:.1f}bb" if isinstance(size, (int, float)) else "")
                     + f"\n    来源: {src}"
                     + (f" @ {prov.get('node')}" if prov.get("node") else "")
                     + f"\n")

    def _render_hand_end(self, ev: dict):
        self._set_buttons_enabled(False)
        self.turn_ctx = None
        self.advisor = None
        if ev.get("aborted"):
            self._log(f"本手中止: {ev['aborted']}")
            return
        net = ev.get("net_bb") or 0.0
        sym = "+" if net >= 0 else ""
        self._snap.update({"board": list(ev.get("board") or []),
                           "revealed": list(ev.get("revealed") or []),
                           "winners": list(ev.get("winners") or []),
                           "pot": ev.get("pot_chips") or 0,
                           "stacks": list(ev.get("final_stacks") or []),
                           "street_note": ""})
        if (ev.get("net_bb") is not None
                and ev.get("src") != "replay"):
            self._stat_hands_done += 1
            self._stat_net_bb += ev["net_bb"]
            self.info_lbl.config(text=self._stats_text(self._cur_hand))
        self._street_note = ""
        self._render_pot(ev.get("pot_chips") or 0)
        self._board_set(ev.get("board") or [])
        for r in ev.get("revealed") or []:    # 摊牌亮牌：对手牌面翻出
            if r["seat"] != 0:
                self._seat_cards(r["seat"],
                                 tuple(_card_file(c) for c in r["hole"]))
        self._winner_seats = set(ev.get("winners") or [])
        self._refresh_seat_styles()
        term = _TERMINAL_CN.get(str(ev.get("terminal_reason") or ""),
                               str(ev.get("terminal_reason") or "—"))
        for s in self._winner_seats:
            self._show_bubble(s, f"赢 {int(ev.get('pot_chips') or 0)}ch",
                              hold_ms=4000)
        self._log(f"—— 本手结束（{term}）"
                  f"  净 {sym}{net}bb（含抽水 {sym}{ev.get('net_bb_raked')}bb）")
        for r in ev.get("revealed") or []:
            self._log(f"    seat{r['seat']} 亮牌: "
                      f"{' '.join(_card(c) for c in r['hole'])}")
        an = ev.get("analysis") or {}
        if an.get("n_advisor_available"):
            self._log(f"    解析: {an['n_hero_decisions']} 决策 / "
                      f"AI 参考一致 {an['n_agree']}/{an['n_advisor_available']}"
                      f"；来源 {','.join(an.get('advisor_provenance_sources') or [])}")

    def _clear_panels(self):
        for txt in (self.advice_txt, self.log_txt):
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.config(state="disabled")

    def _log(self, msg: str):
        self.log_txt.config(state="normal")
        self.log_txt.insert("end", msg + "\n")
        self.log_txt.see("end")
        self.log_txt.config(state="disabled")

    def _advice(self, msg: str):
        self.advice_txt.config(state="normal")
        self.advice_txt.insert("end", msg)
        self.advice_txt.see("end")
        self.advice_txt.config(state="disabled")

    def _status(self, msg: str, color: str = FG_DIM):
        self.status_lbl.config(text=f"● {msg}", fg=color)


class ImportAnalysisDialog(tk.Toplevel):
    """导入手牌分析：贴手牌历史文本 / 选文件 → RealtimeRunner（与对打同一决策核心）→ 结果回主界面建议面板。用户主动导入、纯只读分析，不碰非实时辅助红线。"""

    def __init__(self, app: "PlayGuiApp"):
        super().__init__(app.root)
        self.app = app
        from cash_game_ui.i18n import tr
        self.title(tr("dlg_import_title"))
        self.configure(bg=BG)
        self.geometry(f"{app._z(640)}x{app._z(430)}")
        self.grab_set()
        tk.Label(self, text=tr("dlg_paste_hint"), font=("Segoe UI", 9), fg=FG_DIM, bg=BG).pack(anchor="w", padx=10, pady=(10, 2))
        self.txt = scrolledtext.ScrolledText(self, font=("Consolas", 9), bg=PANEL, fg=FG,
                             relief="flat", wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10)
        self.status = tk.Label(self, text="", font=("Segoe UI", 9), fg=FG_DIM, bg=BG)
        self.status.pack(anchor="w", padx=10, pady=4)
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", pady=8)
        self.go_btn = tk.Button(row, text=tr("dlg_analyze"), font=("Segoe UI", 10, "bold"),
                                bg=GREEN, fg="#0f1419", relief="flat", width=10,
                                command=self._analyze)
        self.go_btn.pack(side="left", padx=10)
        tk.Button(row, text=tr("dlg_load_file"), font=("Segoe UI", 10),
                  bg="#1a2530", fg=FG, relief="flat", width=12,
                  command=self._load_file).pack(side="left", padx=6)
        tk.Button(row, text=tr("dlg_close"), font=("Segoe UI", 10),
                  bg="#333", fg=FG, relief="flat", width=10,
                  command=self.destroy).pack(side="right", padx=10)
        self._busy = False

    def _load_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("手牌历史 / 文本", "*.txt *.hh *.json"), ("全部文件", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                self.txt.delete("1.0", "end")
                self.txt.insert("1.0", f.read())
        except OSError as exc:
            messagebox.showerror("导入", f"读取失败：{exc}", parent=self)

    def _analyze(self):
        if self._busy:
            return
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("导入分析", "请先粘贴手牌历史文本。", parent=self)
            return
        self._busy = True
        self.go_btn.config(state="disabled")
        self.status.config(text="AI 分析中……（solver 锚定快路径）", fg=AMBER)
        def work():
            try:
                import asyncio
                from poker_decision_ai.core.orchestration.runtime import build_runtime
                from poker_decision_ai.realtime import RealtimeRunner
                runtime = build_runtime()
                try:
                    report = asyncio.run(RealtimeRunner(runtime).decide(
                        {"kind": "cash_hand", "hero_id": "Hero",
                         "hand_history_text": text}))
                    data = report.to_dict()
                finally:
                    try:
                        asyncio.run(runtime.close())
                    except Exception:
                        pass
                self.app.events.put({"type": "_import_result", "ok": True, "data": data})
            except Exception as exc:
                self.app.events.put({"type": "_import_result", "ok": False,
                                     "error": str(exc)[:300]})

        threading.Thread(target=work, daemon=True).start()

class ReplayBrowser(tk.Toplevel):
    """手牌复盘浏览器：选会话/手牌，把重建的事件回放进主桌面
    （复用全部桌面渲染与节奏；需要 data/play_history 的 v1+ 数据，
    旧文件无 timeline 时回放退化为只看结果）。"""

    def __init__(self, app: "PlayGuiApp"):
        super().__init__(app.root)
        self.app = app
        self.title("手牌复盘")
        self.configure(bg=BG)
        self.geometry(f"{app._z(560)}x{app._z(600)}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        tk.Label(self, text="选择会话：", font=("Segoe UI", 10, "bold"),
                 fg=FG_DIM, bg=BG).pack(anchor="w", padx=10, pady=(10, 2))
        self.sess_lb = tk.Listbox(self, height=6, font=("Consolas", 9),
                                  bg=PANEL, fg=FG, relief="flat",
                                  selectbackground="#1d4a70",
                                  exportselection=False)
        self.sess_lb.pack(fill="x", padx=10)
        self.sess_lb.bind("<<ListboxSelect>>", self._on_session_select)
        tk.Label(self, text="选择手牌：", font=("Segoe UI", 10, "bold"),
                 fg=FG_DIM, bg=BG).pack(anchor="w", padx=10, pady=(8, 2))
        self.hand_lb = tk.Listbox(self, height=10, font=("Consolas", 9),
                                  bg=PANEL, fg=FG, relief="flat",
                                  selectbackground="#1d4a70",
                                  exportselection=False)
        self.hand_lb.pack(fill="both", expand=True, padx=10)
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x", pady=8)
        tk.Button(row, text="▶ 回放", font=("Segoe UI", 10, "bold"),
                  bg=GREEN, fg="#0f1419", relief="flat", width=10,
                  command=self._play).pack(side="left", padx=10)
        tk.Button(row, text="■ 停止", font=("Segoe UI", 10),
                  bg="#5a2a2a", fg="#ffffff", relief="flat", width=10,
                  command=self._stop).pack(side="left", padx=6)
        tk.Button(row, text="⬇ 导出会话", font=("Segoe UI", 10),
                  bg="#1a2530", fg=FG, relief="flat", width=12,
                  command=self._export_session).pack(side="left", padx=6)
        self.hint = tk.Label(self, text="提示：回放会占用主桌面，对打进行中不可用。",
                             font=("Segoe UI", 9), fg=FG_DIM, bg=BG)
        self.hint.pack(anchor="w", padx=10, pady=(0, 8))
        self._sessions: list[tuple[str, dict]] = []
        self._hands: list[dict] = []
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self):
        from poker_decision_ai.play.history import load_session
        d = os.path.join("data", "play_history")
        if not os.path.isdir(d):
            self.hint.config(text="暂无历史数据（data/play_history 为空）")
            return
        files = sorted((f for f in os.listdir(d)
                        if f.startswith("play_session_") and f.endswith(".jsonl")),
                       reverse=True)
        for fname in files:
            path = os.path.join(d, fname)
            try:
                data = load_session(path)
            except Exception:                     # noqa: BLE001
                continue
            head = data["head"]
            self._sessions.append((path, data))
            self.sess_lb.insert(
                "end", f"{head.get('created_at', '?')}  "
                       f"{head.get('stake_label', '?')}  "
                       f"{len(data['hands'])}手")

    def _on_session_select(self, _e=None):
        sel = self.sess_lb.curselection()
        self.hand_lb.delete(0, "end")
        self._hands = []
        if not sel:
            return
        for i, h in enumerate(self._sessions[sel[0]][1]["hands"]):
            net = h.get("net_bb")
            net_s = f"{net:+.1f}" if isinstance(net, (int, float)) else "—"
            hero = " ".join(_card(c) for c in h.get("hero_hole") or [])
            term = h.get("terminal_reason") or h.get("aborted") or ""
            self.hand_lb.insert(
                "end", f"第{h['hand_index'] + 1}手  {hero}  净{net_s}bb  {term}")
            self._hands.append(h)

    def _play(self):
        app = self.app
        if app.session_running:
            messagebox.showwarning("复盘", "对打进行中，无法回放。", parent=self)
            return
        sel_s, sel_h = self.sess_lb.curselection(), self.hand_lb.curselection()
        if not sel_s or not sel_h:
            messagebox.showwarning("复盘", "先选择会话和手牌。", parent=self)
            return
        head = self._sessions[sel_s[0]][1]["head"]
        hand = self._hands[sel_h[0]]
        app.replay_stop = False
        for e in self._build_events(head, hand, len(self._hands)):
            app.events.put(e)
        self.hint.config(text=f"回放第 {hand['hand_index'] + 1} 手中……"
                              f"点「停止」中断")

    def _stop(self):
        self.app.replay_stop = True
        self.hint.config(text="已停止。")

    def _on_close(self):
        self._stop()
        self.destroy()

    def _export_session(self):
        sel = self.sess_lb.curselection()
        if not sel:
            messagebox.showwarning("导出", "先选择会话。", parent=self)
            return
        path, data = self._sessions[sel[0]]
        out = str(path)[:-6] + ".txt"             # .jsonl → .txt
        with open(out, "w", encoding="utf-8") as f:
            f.write(self._format_session(data))
        self.hint.config(text=f"已导出：{out}")

    def _format_session(self, data: dict) -> str:
        head = data["head"]
        street_cn = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌",
                     "river": "河牌"}
        n_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        L = [f"poker-ai 对打会话导出  {head.get('created_at', '')}",
             f"档位: {head.get('stake_label')}  "
             f"桌型: {head.get('table_size', 6)} 人  "
             f"码深: {head.get('stack_bb', 100)}bb  "
             f"训练定向: {head.get('focus_position', 'auto')}"]
        for h in data["hands"]:
            L.append("")
            L.append(f"===== 第 {h['hand_index'] + 1} 手  "
                     f"按钮 seat{h['button']} =====")
            hero = " ".join(_card(c) for c in h.get("hero_hole") or [])
            L.append(f"你的手牌: {hero}")
            cur = None
            for e in h.get("timeline") or []:
                if e.get("street") != cur:
                    cur = e.get("street")
                    nb = n_board.get(cur, 0)
                    board = " ".join(_card(c)
                                     for c in (h.get("board") or [])[:nb])
                    L.append(f"[{street_cn.get(cur, cur)}] {board}".rstrip())
                who = "你" if e.get("seat") == 0 else f"seat{e['seat']}"
                act = _ACTION_CN.get(e.get("action"), e.get("action"))
                amt = e.get("amount_chips")
                L.append(f"  {who} {act}" + (f" {amt}ch" if amt else ""))
            for r in h.get("revealed") or []:
                cards = " ".join(_card(c) for c in r["hole"])
                who = "你" if r["seat"] == 0 else f"seat{r['seat']}"
                L.append(f"摊牌 {who}: {cards}")
            net = h.get("net_bb")
            net_s = f"{net:+.1f}" if isinstance(net, (int, float)) else "—"
            term = h.get("terminal_reason") or h.get("aborted") or ""
            L.append(f"结果: 底池 {h.get('pot_chips', 0):.0f}ch  "
                     f"你净 {net_s}bb  终局 {term}")
        return "\n".join(L) + "\n"

    def _build_events(self, head: dict, h: dict, total_hands: int):
        """历史记录 → 主桌面事件流（与实时对局同 schema，打 src=replay 标）。"""

        def tag(e: dict) -> dict:
            e["src"] = "replay"
            return e

        n_seats = int(head.get("table_size", 6))
        lineup = head.get("lineup") or [
            {"seat": s, "archetype": "—", "bot_kind": "pool_bot"}
            for s in range(1, n_seats)]
        yield tag({"type": "session_start",
                   "stake_label": head.get("stake_label", ""),
                   "n_hands": total_hands, "lineup": lineup,
                   "table_size": n_seats,
                   "stack_bb": int(head.get("stack_bb", 100)),
                   "focus_position": head.get("focus_position", "auto")})
        tl = h.get("timeline") or []
        stacks0 = tl[0]["stacks"] if tl else [2000] * 6
        yield tag({"type": "hand_start", "hand_index": h["hand_index"],
                   "n_hands": total_hands, "button": h["button"],
                   "hero_hole": h.get("hero_hole") or [],
                   "stacks": stacks0, "pot_chips": 0})
        n_board = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        hero_decisions = h.get("hero_decisions") or []
        hero_di = 0                     # 第 N 个 hero 决策 ↔ 顺序对应
        cur_street = None
        for e in tl:
            if e["street"] != cur_street:
                cur_street = e["street"]
                nb = n_board.get(cur_street, len(h.get("board") or []))
                yield tag({"type": "street", "street": cur_street,
                           "board": (h.get("board") or [])[:nb],
                           "pot_chips": e["pot_chips"],
                           "stacks": e["stacks"],
                           "street_bets": e["street_bets"]})
            advice = None
            if e["seat"] == 0 and hero_di < len(hero_decisions):
                advice = hero_decisions[hero_di].get("advisor")
                hero_di += 1
            yield tag({"type": "action", "seat": e["seat"],
                       "is_hero": e["seat"] == 0, "action": e["action"],
                       "amount_chips": e["amount_chips"],
                       "pot_chips": e["pot_chips"], "stacks": e["stacks"],
                       "street_bets": e["street_bets"],
                       "street": e["street"],
                       "replay_advice": advice})
        yield tag({"type": "hand_end",
                   "net_bb": h.get("net_bb"),
                   "net_bb_raked": h.get("net_bb_raked"),
                   "pot_chips": h.get("pot_chips") or 0,
                   "terminal_reason": h.get("terminal_reason"),
                   "board": h.get("board") or [],
                   "winners": h.get("winners") or [],
                   "revealed": h.get("revealed") or [],
                   "analysis": h.get("analysis") or {}})


def _activation_dialog(status: dict) -> bool:
    """试用到期/激活码无效时的激活窗口；返回是否激活成功。"""
    from cash_game_ui import license as lic
    root = tk.Tk()
    root.title("激活 poker-ai")
    root.geometry("560x430")
    root.configure(bg=BG)
    head = ("试用期已结束。" if status.get("mode") == "expired"
            else "激活码无效，请重新输入。")
    tk.Label(root, text=head, font=("Segoe UI", 11, "bold"),
             fg=AMBER, bg=BG).pack(anchor="w", padx=16, pady=(14, 4))
    tk.Label(root, text="免费试用 3 天全功能；结束后选择档位激活"
                       "（年付约 7 折，两台机器绑定）：",
             font=("Segoe UI", 9), fg=FG_DIM, bg=BG).pack(anchor="w", padx=16)
    price = tk.Label(root, text=(
        "基础版  ¥25/月   ¥68/季   ¥198/年"
        "   对打 + AI参考 + 回放（nl2/nl5）\n"
        "专业版  ¥55/月   ¥138/季  ¥428/年"
        "   + 导入分析 / 弱点报告 / nl10-25 / 高手桌\n"
        "大师版  ¥98/月   ¥248/季  ¥798/年"
        "   + nl50 / 大师桌AI / CLI批量\n"
        "纯本地离线，不发任何牌局数据 ｜ 联系发行方：（微信/邮箱占位）"),
        font=("Consolas", 9), fg=FG, bg=BG, justify="left", anchor="w")
    price.pack(fill="x", padx=16, pady=(8, 2))
    mid_row = tk.Frame(root, bg=BG)
    mid_row.pack(fill="x", padx=16)
    tk.Label(mid_row, text=f"本机机器码：{lic.machine_id()}",
             font=("Consolas", 10), fg=FG, bg=BG).pack(side="left")
    tk.Button(mid_row, text="复制机器码", font=("Segoe UI", 9),
              bg="#1a2530", fg=FG, relief="flat",
              command=lambda: root.clipboard_clear() or root.clipboard_append(
                  lic.machine_id())).pack(side="left", padx=8)
    tk.Label(root, text="粘贴激活码：", font=("Segoe UI", 10),
             fg=FG, bg=BG).pack(anchor="w", padx=16, pady=(10, 2))
    key_en = tk.Entry(root, font=("Consolas", 10), width=52)
    key_en.pack(padx=16, anchor="w")
    result = {"ok": False}

    def do_activate():
        if lic.activate(key_en.get()):
            result["ok"] = True
            root.destroy()
        else:
            messagebox.showerror("激活失败",
                                 "激活码无效或不匹配本机，请核对后重试。",
                                 parent=root)

    btns = tk.Frame(root, bg=BG)
    btns.pack(anchor="w", padx=16, pady=14)
    tk.Button(btns, text="激活", font=("Segoe UI", 10, "bold"), bg=GREEN,
              fg="#0f1419", relief="flat", width=10,
              command=do_activate).pack(side="left")
    tk.Button(btns, text="退出", font=("Segoe UI", 10), bg="#5a2a2a",
              fg="#ffffff", relief="flat", width=10,
              command=root.destroy).pack(side="left", padx=8)
    root.mainloop()
    return result["ok"]


def main() -> int:
    from cash_game_ui import license as lic
    status = lic.check_license()
    if not status["ok"]:
        if not _activation_dialog(status):
            return 2
        status = lic.check_license()
    root = tk.Tk()
    lic.set_current_tier(str(status.get("tier") or "base"))
    PlayGuiApp(root)
    label = lic.tier_label(str(status.get("tier") or "base"))
    if status["mode"] == "trial":
        root.title(f"poker-ai 对打模式（产品 v1）"
                   f"——试用版（剩余 {status['days_left']} 天，全功能）")
    else:
        root.title(f"poker-ai 对打模式（产品 v1）——{label}")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
