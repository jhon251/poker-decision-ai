"""V-3a 标注工具 — Tkinter 逐帧人工标注（真人金标准采集）。

盲标原则：本工具**永不读取/展示**批清单里的 machine_snapshot 字段——
标注者只看截图，只填亲眼确认的内容（禁止猜）。

控件（快捷键优先）：
  1/2/3/4/5 → street: preflop/flop/turn/river/无法判断
  Enter     → 保存本帧并下一帧
  b / Left  → 上一帧（可回改）
  s         → 中途保存（进度+结果落盘）
  q         → 质量循环：清晰→模糊→遮挡→其它应用画面
  牌格获得焦点时直接按 rank/suit 字符键填入（a→A, s→s 等）

字段：street、公共牌 0-5 张（rank×suit 下拉，"—"=空）、hero 手牌
（默认"不可见"，勾掉后填两张）、座位 1-6 筹码可读性（可读/模糊/
不可见 循环按钮）、质量（清晰/模糊/遮挡）。

进度落盘 docs/vision_v3/annotation_progress.json，可随时关窗续标；
结果落盘 docs/vision_v3/annotations.jsonl（每次保存全量原子重写）。

冒烟：python scripts/v3_annotate.py --smoke
  （程序化驱动 3 帧 demo 批走通 UI→保存→续标全流程，exit 0/1）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

REPO = Path(__file__).resolve().parents[1]

BATCH = REPO / "docs" / "vision_v3" / "annotation_batch_400.jsonl"
import os as _os
BATCH = Path(_os.environ.get("V3_BATCH", str(BATCH)))
PROGRESS = REPO / "docs" / "vision_v3" / "annotation_progress.json"
OUT = REPO / "docs" / "vision_v3" / "annotations.jsonl"
DEMO_BATCH = REPO / "docs" / "vision_v3" / "demo_batch_3.jsonl"
DEMO_PROGRESS = REPO / "docs" / "vision_v3" / "demo_progress.json"
DEMO_OUT = REPO / "docs" / "vision_v3" / "demo_annotations.jsonl"

TOOL_VERSION = "v3_annotate.py v1"

RANKS = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
SUITS = ("s", "h", "d", "c")
EMPTY = "—"
STREETS = (("1", "preflop"), ("2", "flop"), ("3", "turn"), ("4", "river"),
           ("5", "unclear"))
STREET_LABEL = {"preflop": "preflop(翻前)", "flop": "flop(翻牌)",
                "turn": "turn(转牌)", "river": "river(河牌)",
                "unclear": "无法判断"}
QUALITIES = ("清晰", "模糊", "遮挡", "其它应用画面")
SEAT_STATES = ("可读", "模糊", "不可见")
N_SEATS = 6
N_BOARD = 5

#: 放大查看区（显示辅助，非识别语义）：window class → (board x0,y0,w,h,n, hero(x,y,w,h))
CROPS = {
    "1419": {"board": (664, 551, 140, 177, 5), "hero": (130, 725, 330, 215)},
    "1876": {"board": (590, 551, 140, 180, 5), "hero": (120, 890, 380, 350),
             },
}
WINDOW_BY_GEOMETRY = {(1419, 1066): "1419", (1876, 1400): "1876",
                      (1280, 779): "1419"}


# ── 纯逻辑（可单测） ──────────────────────────────────────────────────
def new_record(file: str, event: str | None, ts, path: str | None = None) -> dict:
    return {
        "file": file, "event": event, "ts": ts,
        "path": path,

        "street": None,
        "board_cards": [],            # 槽位序的 "Rs" 列表（0-5 张）
        "hero_status": "not_visible",  # visible | not_visible
        "hero_cards": None,            # visible 时为 2 张 "Rs"
        "seats": {f"S{i}": "可读" for i in range(1, N_SEATS + 1)},
        "quality": "清晰",
        "annotated_at": None,
        "annotator": "owner",
        "tool": TOOL_VERSION,
    }


def record_complete(rec: dict) -> bool:
    """可提交判定：其它应用画面仅要求质量标注；其余 street 已选 + hero 可见时两张牌齐全。"""
    if rec.get("quality") == "其它应用画面":
        return True
    if not rec.get("street"):
        return False
    if rec.get("hero_status") == "visible":
        hc = rec.get("hero_cards") or []
        if len([c for c in hc if c]) != 2:
            return False
    return True


def build_board_cards(slot_values: list[tuple[str, str]]) -> list[str]:
    """5 个槽位的 (rank, suit) → 槽位序非空 "Rs" 列表。"""
    out = []
    for rank, suit in slot_values:
        if rank != EMPTY and suit != EMPTY:
            out.append(f"{rank}{suit}")
    return out


def progress_payload(order: list[str], records: dict[str, dict],
                     batch: str) -> dict:
    return {"batch": str(batch), "order": order, "records": records,
            "updated_at": datetime.now(timezone.utc).isoformat()}


def save_progress(path: Path, order: list[str], records: dict[str, dict],
                  batch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress_payload(order, records, batch),
                              ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"order": [], "records": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_annotations(path: Path, order: list[str],
                     records: dict[str, dict]) -> int:
    """全量原子重写 annotations.jsonl（order 序）。返回条数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in order:
        rec = records.get(f)
        if rec and record_complete(rec):
            rec = dict(rec)
            rec["annotated_at"] = rec.get("annotated_at") or \
                datetime.now(timezone.utc).isoformat()
            lines.append(json.dumps(rec, ensure_ascii=False))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)
    return len(lines)


# ── UI ────────────────────────────────────────────────────────────────
class AnnotateApp:
    def __init__(self, batch_path: Path, progress_path: Path, out_path: Path,
                 smoke: bool = False):
        self.batch_path = batch_path
        self.progress_path = progress_path
        self.out_path = out_path
        self.smoke = smoke
        self.annotator = "smoke" if smoke else "owner"
        self.batch = [json.loads(l) for l in
                      batch_path.read_text(encoding="utf-8").splitlines() if l]
        self.order = [b["file"] for b in self.batch]
        self.records: dict[str, dict] = {}
        self.idx = 0
        self._photo_refs: list = []
        self._build_progress()
        self._build_ui()

    # 进度/续标 ─
    def _build_progress(self) -> None:
        prog = load_progress(self.progress_path)
        saved = prog.get("records", {})
        if saved and prog.get("order") == self.order:
            self.records = saved
            self.idx = 0
            for i, f in enumerate(self.order):
                if not record_complete(self.records.get(f, {})):
                    self.idx = i
                    break
            else:
                self.idx = len(self.order) - 1  # 全部完成：停在末帧供复核

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"V-3a 标注 {TOOL_VERSION} — 盲标（不看任何机器结果）")
        self.root.geometry("1880x1020")
        top = ttk.Frame(self.root); top.pack(fill="x", padx=6, pady=3)
        self.lbl_pos = ttk.Label(top, font=("Consolas", 12))
        self.lbl_pos.pack(side="left")
        self.lbl_hint = ttk.Label(
            top, text="  快捷键：1-5 选街线 | Enter 下一帧 | b/← 上一帧 | "
                      "s 保存 | q 质量 | 牌格内直接按字符键",
            foreground="#555")
        self.lbl_hint.pack(side="left", padx=12)
        body = ttk.Frame(self.root); body.pack(fill="both", expand=True)
        # 左列：整帧
        left = ttk.Frame(body); left.pack(side="left", fill="both", padx=4)
        self.lbl_img = ttk.Label(left); self.lbl_img.pack()
        # 右列：放大区 + 控件
        right = ttk.Frame(body); right.pack(side="left", fill="both",
                                            expand=True, padx=6)
        zoom = ttk.LabelFrame(right, text="放大：hero 手牌 / 公共牌（槽1-3，槽4-5）")
        zoom.pack(fill="x")
        self.lbl_hero_zoom = ttk.Label(zoom); self.lbl_hero_zoom.pack(
            side="left", padx=2, pady=2)
        boardzoom = ttk.Frame(zoom); boardzoom.pack(side="left", padx=2)
        self.lbl_bz_row1 = ttk.Label(boardzoom); self.lbl_bz_row1.pack()
        self.lbl_bz_row2 = ttk.Label(boardzoom); self.lbl_bz_row2.pack()
        # 控件
        ctl = ttk.LabelFrame(right, text="标注（只填亲眼确认的——禁止猜）")
        ctl.pack(fill="both", expand=True, pady=4)
        sf = ttk.Frame(ctl); sf.pack(fill="x", pady=2)
        ttk.Label(sf, text="street:").pack(side="left")
        self.street_var = tk.StringVar(value="")
        for key, s in STREETS:
            ttk.Radiobutton(sf, text=f"{key}={STREET_LABEL[s]}",
                            value=s, variable=self.street_var).pack(
                side="left", padx=3)
        bf = ttk.Frame(ctl); bf.pack(fill="x", pady=2)
        ttk.Label(bf, text="公共牌(最多5张，—=该槽无牌):").pack(
            anchor="w")
        self.board_rank: list[tk.StringVar] = []
        self.board_suit: list[tk.StringVar] = []
        rank_glyph = {"T": "10"}
        for i in range(N_BOARD):
            cell = ttk.Frame(bf); cell.pack(side="left", padx=4)
            ttk.Label(cell, text=f"槽{i+1}").pack()
            rv = tk.StringVar(value=EMPTY)
            sv = tk.StringVar(value=EMPTY)
            cb_r = ttk.Combobox(cell, textvariable=rv, width=4,
                                state="readonly",
                                values=(EMPTY,) + RANKS)
            cb_s = ttk.Combobox(cell, textvariable=sv, width=3,
                                state="readonly",
                                values=(EMPTY,) + SUITS)
            cb_r.pack(); cb_s.pack()
            self._bind_card_keys(cb_r, rv, RANKS, rank_glyph)
            self._bind_card_keys(cb_s, sv, SUITS, {})
            self.board_rank.append(rv); self.board_suit.append(sv)
        hf = ttk.Frame(ctl); hf.pack(fill="x", pady=2)
        self.hero_state = tk.StringVar(value="not_visible")
        ttk.Radiobutton(hf, text="手牌可见", value="visible",
                        variable=self.hero_state, command=self._on_hero_toggle
                        ).pack(side="left")
        ttk.Radiobutton(hf, text="未见手牌(已弃/等待/不可见)",
                        value="not_visible", variable=self.hero_state,
                        command=self._on_hero_toggle).pack(side="left")
        ttk.Label(hf, text="  hero 牌1/牌2:").pack(side="left")
        self.hero_rank: list[tk.StringVar] = []
        self.hero_suit: list[tk.StringVar] = []
        self.hero_cbs = []
        for i in range(2):
            rv = tk.StringVar(value=EMPTY); sv = tk.StringVar(value=EMPTY)
            cb_r = ttk.Combobox(hf, textvariable=rv, width=4,
                                state="disabled", values=(EMPTY,) + RANKS)
            cb_s = ttk.Combobox(hf, textvariable=sv, width=3,
                                state="disabled", values=(EMPTY,) + SUITS)
            cb_r.pack(side="left", padx=2); cb_s.pack(side="left", padx=2)
            self._bind_card_keys(cb_r, rv, RANKS, rank_glyph)
            self._bind_card_keys(cb_s, sv, SUITS, {})
            self.hero_rank.append(rv); self.hero_suit.append(sv)
            self.hero_cbs.extend([cb_r, cb_s])
        qf = ttk.Frame(ctl); qf.pack(fill="x", pady=2)
        ttk.Label(qf, text="质量(q 循环):").pack(side="left")
        self.quality_var = tk.StringVar(value=QUALITIES[0])
        ttk.Button(qf, textvariable=self.quality_var, width=8,
                   command=self._cycle_quality).pack(side="left", padx=4)
        sef = ttk.Frame(ctl); sef.pack(fill="x", pady=2)
        ttk.Label(sef, text="各座位筹码可读性(点击循环):").pack(anchor="w")
        self.seat_vars: dict[str, tk.StringVar] = {}
        for i in range(1, N_SEATS + 1):
            v = tk.StringVar(value=SEAT_STATES[0])
            ttk.Button(sef, text=f"座位{i}", width=5,
                       command=lambda vv=v: self._cycle_seat(vv)).pack(
                side="left", padx=2)
            ttk.Label(sef, textvariable=v, width=4).pack(side="left")
            self.seat_vars[f"S{i}"] = v
        nf = ttk.Frame(ctl); nf.pack(fill="x", pady=6)
        ttk.Button(nf, text="◀ 上一帧(b)", command=self.prev_frame).pack(
            side="left", padx=2)
        ttk.Button(nf, text="保存(s)", command=self.save).pack(
            side="left", padx=2)
        ttk.Button(nf, text="保存并下一帧(Enter) ▶",
                   command=self.next_frame).pack(side="left", padx=2)
        self.lbl_status = ttk.Label(ctl, foreground="#a00")
        self.lbl_status.pack(anchor="w")
        # 键盘
        for key, s in STREETS:
            self.root.bind(key, lambda e, ss=s: self._set_street(ss))
        self.root.bind("<Return>", lambda e: self.next_frame())
        self.root.bind("<b>", lambda e: self.prev_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<s>", lambda e: self.save())
        self.root.bind("<q>", lambda e: self._cycle_quality())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _bind_card_keys(cb: ttk.Combobox, var: tk.StringVar,
                        values: tuple, glyph: dict) -> None:
        """牌格获得焦点时按字符键直接填值（只接受合法值）。"""
        lower = {v.lower(): v for v in values}
        def on_key(e):
            ch = e.char.lower()
            if ch in lower:
                var.set(lower[ch])
                cb.selection_clear()
            return "break" if ch else None
        cb.bind("<Key>", on_key)

    # 状态操作（供键盘/按钮/冒烟共用） ─
    def _set_street(self, s: str) -> None:
        self.street_var.set(s)
        self._flash("")

    def _cycle_quality(self) -> None:
        cur = QUALITIES.index(self.quality_var.get()) \
            if self.quality_var.get() in QUALITIES else 0
        self.quality_var.set(QUALITIES[(cur + 1) % len(QUALITIES)])

    def _cycle_seat(self, var: tk.StringVar) -> None:
        cur = SEAT_STATES.index(var.get()) if var.get() in SEAT_STATES else 0
        var.set(SEAT_STATES[(cur + 1) % len(SEAT_STATES)])

    def _on_hero_toggle(self) -> None:
        state = "readonly" if self.hero_state.get() == "visible" else "disabled"
        for cb in self.hero_cbs:
            cb.configure(state=state)

    def _collect(self) -> dict:
        rec = self.records.setdefault(
            self.order[self.idx],
            new_record(self.order[self.idx],
                       self.batch[self.idx].get("event"),
                       self.batch[self.idx].get("ts"),
                       self.batch[self.idx].get("path")))
        rec["street"] = self.street_var.get() or None
        rec["board_cards"] = build_board_cards(list(zip(
            (v.get() for v in self.board_rank),
            (v.get() for v in self.board_suit))))
        if self.hero_state.get() == "visible":
            rec["hero_status"] = "visible"
            rec["hero_cards"] = build_board_cards(list(zip(
                (v.get() for v in self.hero_rank),
                (v.get() for v in self.hero_suit))))
        else:
            rec["hero_status"] = "not_visible"
            rec["hero_cards"] = None
        rec["quality"] = self.quality_var.get()
        rec["seats"] = {k: v.get() for k, v in self.seat_vars.items()}
        rec["annotator"] = self.annotator
        return rec

    def _load_into_ui(self) -> None:
        f = self.order[self.idx]
        rec = self.records.get(f) or new_record(
            f, self.batch[self.idx].get("event"), self.batch[self.idx].get("ts"),
            self.batch[self.idx].get("path"))
        self.street_var.set(rec.get("street") or "")
        slots = rec.get("board_cards") or []
        for i in range(N_BOARD):
            card = slots[i] if i < len(slots) else ""
            self.board_rank[i].set(card[:-1] if card else EMPTY)
            self.board_suit[i].set(card[-1:] if card else EMPTY)
        visible = rec.get("hero_status") == "visible"
        self.hero_state.set("visible" if visible else "not_visible")
        self._on_hero_toggle()
        hc = rec.get("hero_cards") or []
        for i in range(2):
            card = hc[i] if i < len(hc) else ""
            self.hero_rank[i].set(card[:-1] if card else EMPTY)
            self.hero_suit[i].set(card[-1:] if card else EMPTY)
        self.quality_var.set(rec.get("quality") or QUALITIES[0])
        for k, v in self.seat_vars.items():
            v.set((rec.get("seats") or {}).get(k, SEAT_STATES[0]))
        done = record_complete(rec)
        self.lbl_status.config(
            text="" if done else "（本帧未完成：需选 street；hero 可见时需两张牌）")
        self._show_image(f)

    def _show_image(self, fname: str) -> None:
        rec_path = next((r.get("path") for r in self.batch
                         if r.get("file") == fname), None)
        p = Path(rec_path) if rec_path and Path(rec_path).exists()             else REPO / "data" / "vision2" / "coinpoker_evidence" / fname
        self._photo_refs = []
        raw = tk.PhotoImage(master=self.root, data=p.read_bytes())
        img = raw
        f = max(1, -(-raw.width() // 940))       # ceil 缩到 ≤940 宽
        if f > 1:
            img = raw.subsample(f, f)
        self.lbl_img.configure(image=img)
        self._photo_refs.append(img)
        # 放大区（几何已知时）：hero 区 + 公共牌槽 1-3 / 4-5 两行，zoom(2)
        cls = WINDOW_BY_GEOMETRY.get((raw.width(), raw.height()))
        crops = CROPS.get(cls)
        if crops:
            bx, by, bw, bh, _n = crops["board"]
            hx, hy, hw, hh = crops["hero"]
            hz = raw.copy(from_coords=(hx, hy, hx + hw, hy + hh), zoom=2)
            row1 = raw.copy(from_coords=(bx, by, bx + 3 * bw, by + bh),
                            zoom=2)
            row2 = raw.copy(from_coords=(bx + 3 * bw, by, bx + 5 * bw,
                                         by + bh), zoom=2)
            self.lbl_hero_zoom.configure(image=hz)
            self.lbl_bz_row1.configure(image=row1)
            self.lbl_bz_row2.configure(image=row2)
            self._photo_refs += [raw, hz, row1, row2]
        else:
            self.lbl_hero_zoom.configure(image="",
                                         text="（未知几何，无放大区）")
        self.lbl_pos.config(text=f"帧 {self.idx + 1}/{len(self.order)}  "
                                 f"{fname}  [已存 {len(self.records)}]")

    # 翻帧/保存 ─
    def _flash(self, msg: str) -> None:
        self.lbl_status.config(text=msg)

    def next_frame(self) -> None:
        rec = self._collect()
        if not record_complete(rec):
            self._flash("未完成：1-5 选 street；hero 可见时需两张牌齐全")
            return
        rec["annotated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        if self.idx < len(self.order) - 1:
            self.idx += 1
            self._load_into_ui()
        else:
            self._flash("已是最后一帧——全部完成请关闭窗口（可 s 再存一次）")

    def prev_frame(self) -> None:
        self._collect()                      # 收集当前（可能不完整，不落盘）
        if self.idx > 0:
            self.idx -= 1
            self._load_into_ui()

    def save(self) -> None:
        self._collect()
        save_progress(self.progress_path, self.order, self.records,
                      str(self.batch_path))
        n = save_annotations(self.out_path, self.order, self.records)
        self._flash(f"已保存：进度 {len(self.records)}/{len(self.order)}，"
                    f"完成 {n} 条 → {self.out_path.name}")

    def _on_close(self) -> None:
        self._collect()
        save_progress(self.progress_path, self.order, self.records,
                      str(self.batch_path))
        save_annotations(self.out_path, self.order, self.records)
        self.root.destroy()

    def run(self) -> None:
        self._load_into_ui()
        self.root.mainloop()


# ── 冒烟（程序化驱动，无人工介入） ────────────────────────────────────
def smoke() -> int:
    """3 帧 demo：填→存→重开续标→断言→清场。exit 0=通过。"""
    app = AnnotateApp(DEMO_BATCH, DEMO_PROGRESS, DEMO_OUT, smoke=True)
    app.root.withdraw()
    # 幂等起点：清掉历史 demo 进度
    app.records.clear(); app.idx = 0
    fills = [
        {"street": "preflop", "hero": ("6", "c", "3", "h")},
        {"street": "flop", "board": ("A", "h", "T", "d", "2", "s")},
        {"street": "river", "board": ("A", "h", "T", "d", "2", "s", "9", "c",
                                      "K", "d"), "quality": "模糊"},
    ]
    for i, spec in enumerate(fills):
        app.idx = i
        app._load_into_ui()
        app._set_street(spec["street"])
        board = spec.get("board") or ()
        for j in range(N_BOARD):
            if 2 * j < len(board):
                app.board_rank[j].set(board[2 * j])
                app.board_suit[j].set(board[2 * j + 1])
            else:
                app.board_rank[j].set(EMPTY); app.board_suit[j].set(EMPTY)
        if "hero" in spec:
            app.hero_state.set("visible"); app._on_hero_toggle()
            app.hero_rank[0].set(spec["hero"][0])
            app.hero_suit[0].set(spec["hero"][1])
            app.hero_rank[1].set(spec["hero"][2])
            app.hero_suit[1].set(spec["hero"][3])
        if "quality" in spec:
            app.quality_var.set(spec["quality"])
        app.next_frame()                     # 收集+校验+保存+翻帧
    # 断言：3 帧完成、annotations 3 行、street/board 落盘正确
    prog = load_progress(DEMO_PROGRESS)
    assert len(prog["records"]) == 3, f"progress records={len(prog['records'])}"
    lines = [json.loads(l) for l in
             DEMO_OUT.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 3, f"annotations lines={len(lines)}"
    assert lines[0]["street"] == "preflop" and \
        lines[0]["hero_cards"] == ["6c", "3h"]
    assert lines[1]["street"] == "flop" and \
        lines[1]["board_cards"] == ["Ah", "Td", "2s"]
    assert lines[2]["street"] == "river" and \
        len(lines[2]["board_cards"]) == 5 and lines[2]["quality"] == "模糊"
    # 续标：新实例应恢复 3 条记录且停在末帧（先销毁前实例——单 Tk root）
    app.root.destroy()
    app2 = AnnotateApp(DEMO_BATCH, DEMO_PROGRESS, DEMO_OUT, smoke=True)
    app2.root.withdraw()
    assert len(app2.records) == 3, f"resume records={len(app2.records)}"
    # 回改第 1 帧再存（prev/回改路径）
    app2.idx = 0; app2._load_into_ui()
    app2._set_street("flop")
    app2.next_frame()
    lines2 = [json.loads(l) for l in
              DEMO_OUT.read_text(encoding="utf-8").splitlines() if l]
    assert lines2[0]["street"] == "flop", "resume-edit failed"
    print("SMOKE OK: 3 demo frames annotated/saved/resumed/re-edited")
    app2.root.destroy()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=BATCH)
    ap.add_argument("--progress", type=Path, default=PROGRESS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--start", type=int, default=None,
                    help="从第 N 帧（1 基）开始")
    ap.add_argument("--smoke", action="store_true",
                    help="程序化冒烟：demo 批 3 帧走通全流程后退出")
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    app = AnnotateApp(args.batch, args.progress, args.out)
    if args.start:
        app.idx = max(0, min(args.start - 1, len(app.order) - 1))
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
