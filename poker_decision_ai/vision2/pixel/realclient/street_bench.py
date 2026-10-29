"""V-1 — Street 机制帧级实测 harness（分块落盘 + 断点续跑）。

用 data/vision2/coinpoker_evidence/ 的 17,243 张真实全帧（文件名
EVENT_<sid>_<ms>_full.png 含毫秒时间戳）对 V-1 修复后的双模式 street
运行时做帧级统计：

- **stream 模式**（全量 17,243 帧）：按全局时间戳排序、间隙阈值分桶成
  runs（120s——依据：语料间隙直方图在 60s-5min 区间仅 4 个样本而 >5min
  有 17 个采集重启样本；≤2min 停顿视为同会话），每个 run 喂入一个持久
  StreetSession（流模式语义——2 帧确认门完整）。
- **single 模式**（种子抽样 ≥3,000 帧）：每帧 fresh StreetObserver →
  单帧豁免决议（修复后）+ 修复前映射仿真（NEW_HAND/PREFLOP→preflop 的
  旧桥接语义）+ 独立 BoardObserver 计数交叉核对。

长跑纪律（ENVIRONMENT.md）：单块 ≤2,000 帧；manifest 记录已完成单元，
重启自动跳过；单帧落盘 JSONL（docs/vision_v1/）。纯 stdlib 依赖
（png_decode / StreetObserver 与生产同路径——零语义旁路）。

用法：
  python -m poker_decision_ai.vision2.pixel.realclient.street_bench \
      --mode index|stream|single|bridge-post [--chunk-frames 2000] \
      [--sample 3400] [--seed 20261004]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import struct
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EV = REPO / "data" / "vision2" / "coinpoker_evidence"
# V-3 覆盖点：环境变量可指向新语料（如 coinpoker_v3），不传保持默认。
import os as _os
EV = Path(_os.environ.get("STREET_BENCH_CORPUS", str(EV)))
OUT = REPO / "docs" / "vision_v1"
# V-3 覆盖点：新语料的索引/统计落盘到 vision_v3（避免覆写 V-1 基线）。
OUT = Path(_os.environ.get("STREET_BENCH_OUT", str(OUT)))

#: 全局时间戳分桶阈值（ms）。依据 corpus 实测：帧间隙呈 <10s（17,074 个）
#: 与 >5min（17 个采集重启）双峰，60s-5min 区间仅 4 个样本 → 120s 干净
#: 分离采集重启且不切断任何手内序列（本语料手内节奏 p99 <10s）。
GAP_THRESHOLD_MS = 120_000

_NAME_RE = re.compile(r"^([A-Z_]+)_(\d+)_(\d+)_full\.png$")

# 修复前桥接 street 语义（production_bridge 旧 §1，用于 before 仿真）：
# NEW_HAND（无 0 卡确认）/ PREFLOP → preflop；FLOP/TURN/RIVER 直映；
# None/UNKNOWN/... → 不可表示。
_LEGACY_DIRECT = {"PREFLOP": "preflop", "FLOP": "flop", "TURN": "turn",
                  "RIVER": "river"}


def png_geometry(path: Path) -> tuple[int, int] | None:
    """PNG IHDR → (w, h)（只读 33 字节头——不解码像素）。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
        if len(head) < 24 or not head.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        return struct.unpack(">II", head[16:24])
    except OSError:
        return None


def build_index() -> dict:
    """扫描 evidence 目录 → 帧索引 + 全局时间戳 runs（含几何）。"""
    rows = []
    for p in sorted(EV.glob("*_full.png")):
        m = _NAME_RE.match(p.name)
        if not m:
            continue
        rows.append({"event": m.group(1), "sid": m.group(2),
                     "ts": int(m.group(3)), "file": p.name})
    rows.sort(key=lambda r: (r["ts"], r["file"]))
    runs = []
    cur: list[dict] = []
    for r in rows:
        if cur and (r["ts"] - cur[-1]["ts"]) > GAP_THRESHOLD_MS:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    for i, run in enumerate(runs):
        geom = png_geometry(EV / run[0]["file"])
        for r in run:
            r["run_id"] = i
            r["w"], r["h"] = geom if geom else (None, None)
        run_meta = {
            "run_id": i, "frames": len(run),
            "ts_start": run[0]["ts"], "ts_end": run[-1]["ts"],
            "span_s": round((run[-1]["ts"] - run[0]["ts"]) / 1000, 1),
            "geometry": f"{geom[0]}x{geom[1]}" if geom else None,
            "events": sorted({r["event"] for r in run}),
        }
        runs[i] = {"meta": run_meta, "rows": run}
    return {"gap_threshold_ms": GAP_THRESHOLD_MS, "total_frames": len(rows),
            "runs": runs}


def ensure_index() -> dict:
    """全量帧索引（含 runs/rows；缓存 frame_index_full.json——
    frame_index.json 仅 metas 供人读）。"""
    full_path = OUT / "frame_index_full.json"
    if full_path.exists():
        return json.loads(full_path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    full_path.write_text(json.dumps(idx, ensure_ascii=False),
                         encoding="utf-8")
    (OUT / "frame_index.json").write_text(json.dumps(
        {"gap_threshold_ms": idx["gap_threshold_ms"],
         "total_frames": idx["total_frames"],
         "runs": [r["meta"] for r in idx["runs"]]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return idx


class _Manifest:
    """断点续跑 manifest：记录已完成单元与单元内偏移（intra-run resume）。

    data = {"units": {unit_id: {"rows_done": int, "complete": bool}},
            "rows": 累计落盘行数, "updated": iso}
    """

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("units", {})
        else:
            self.data = {"units": {}, "rows": 0, "updated": None}

    def rows_done(self, unit: str) -> int:
        u = self.data["units"].get(unit)
        return u["rows_done"] if u else 0

    def complete(self, unit: str) -> bool:
        u = self.data["units"].get(unit)
        return bool(u and u.get("complete"))

    def mark(self, unit: str, rows_done: int, complete: bool):
        self.data["units"][unit] = {"rows_done": rows_done,
                                    "complete": complete}
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.path.write_text(json.dumps(self.data, ensure_ascii=False),
                             encoding="utf-8")


# ── stream 模式：全量 17,243 帧 × 持久 StreetSession ─────────────────
def run_stream(chunk_frames: int, tag: str = "") -> None:
    from .street_session import StreetSession
    from poker_decision_ai.vision2_platform.png_decode import decode_png

    idx = ensure_index()
    frames_path = OUT / f"street_stream_frames{tag}.jsonl"
    runs_path = OUT / f"street_stream_runs{tag}.jsonl"
    man = _Manifest(OUT / f"street_stream_manifest{tag}.json")
    OUT.mkdir(parents=True, exist_ok=True)

    processed = 0
    for run in idx["runs"]:
        meta, rows = run["meta"], run["rows"]
        unit = f"run:{meta['run_id']}"
        if man.complete(unit):
            continue
        offset = man.rows_done(unit)          # intra-run 断点续跑
        t0 = time.time()
        sess = StreetSession(f"bench_run_{meta['run_id']}")
        # 续跑：重放已处理前缀以重建机器状态（确定性——同帧同证据），
        # 保证确认门/泄漏门的时序语义与不断跑完全一致
        for r in rows[:offset]:
            w, h, rgba = decode_png((EV / r["file"]).read_bytes())
            sess.observe(rgba, w, h)
        run_rows: list[str] = []
        for r in rows[offset:]:
            fp = EV / r["file"]
            try:
                w, h, rgba = decode_png(fp.read_bytes())
                info = sess.observe(rgba, w, h)
                row = {"run_id": meta["run_id"], "file": r["file"],
                       "event": r["event"], "ts": r["ts"],
                       "machine_state": info["machine_state"],
                       "street": info["last_street"],
                       "provenance": info["provenance"],
                       "observed_cardinality": info["observed_cardinality"],
                       "confidence": info["confidence"],
                       "reason": info["reason"], "err": None}
            except Exception as e:                      # 单帧失败不中断
                row = {"run_id": meta["run_id"], "file": r["file"],
                       "event": r["event"], "ts": r["ts"],
                       "machine_state": None, "street": None,
                       "provenance": None, "observed_cardinality": None,
                       "confidence": None, "reason": None,
                       "err": str(e)[:120]}
            run_rows.append(json.dumps(row, ensure_ascii=False))
            processed += 1
        done_rows = offset + len(run_rows)
        complete = done_rows >= len(rows)
        if run_rows:
            with open(frames_path, "a", encoding="utf-8") as f:
                f.write("\n".join(run_rows) + "\n")
        if complete:
            # per-run 汇总（机器 evidence chain 聚合——仅在 run 完成时写）
            recs = sess.machine_records()
            trans = {}
            illegal = leak = hand_end = 0
            for rec in recs:
                reason = rec["transition_reason"] or ""
                if reason == "settled_confirmed":
                    key = f"{rec['previous_state']}->{rec['state']}"
                    trans[key] = trans.get(key, 0) + 1
                elif reason.startswith("illegal_transition"):
                    illegal += 1
                elif reason.startswith("confirmed_card_leak"):
                    leak += 1
                elif reason == "board_cleared_hand_ended":
                    hand_end += 1
            reached = sorted({rec["state"] for rec in recs})
            run_summary = {**meta, "machine_states_reached": reached,
                           "confirmed_transitions": trans,
                           "illegal_transitions": illegal,
                           "card_leaks": leak, "hand_ends": hand_end,
                           "dt_s": round(time.time() - t0, 1)}
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")
        man.mark(unit, done_rows, complete)
        print(f"run {meta['run_id']}[{offset}:{done_rows}/{len(rows)}]: "
              f"{round(time.time() - t0, 1)}s — processed+replayed "
              f"{processed}")
        if processed >= chunk_frames:
            print(f"chunk budget {chunk_frames} reached — resumable stop")
            return


# ── single 模式：种子抽样 × 单帧豁免决议（+修复前仿真 +独立交叉核对）──
def run_single(sample: int, seed: int, chunk_frames: int,
               tag: str = "") -> None:
    from .seat_position import BoardObserver, StreetObserver
    from .street_session import resolve_single_frame_street
    from poker_decision_ai.vision2_platform.png_decode import decode_png

    idx = ensure_index()
    all_rows = [r for run in idx["runs"] for r in run["rows"]]
    rng = random.Random(seed)
    sample_rows = sorted(rng.sample(all_rows, min(sample, len(all_rows))),
                         key=lambda r: r["file"])
    frames_path = OUT / f"street_single_frames{tag}.jsonl"
    man = _Manifest(OUT / f"street_single_manifest{tag}.json")
    OUT.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(sample_rows) + chunk_frames - 1) // chunk_frames
    for c in range(n_chunks):
        if man.complete(f"chunk:{c}"):
            continue
        part = sample_rows[c * chunk_frames:(c + 1) * chunk_frames]
        t0 = time.time()
        out_rows = []
        for r in part:
            fp = EV / r["file"]
            try:
                w, h, rgba = decode_png(fp.read_bytes())
                obs = StreetObserver(session_id="bench_single")
                obs.observe(rgba, w, h)
                res = resolve_single_frame_street(obs)
                # 修复前桥接语义仿真（旧 §1：NEW_HAND 静默→preflop）
                ms = res["machine_state"]
                if ms in _LEGACY_DIRECT:
                    pre = _LEGACY_DIRECT[ms]
                elif ms == "NEW_HAND":
                    pre = "preflop"            # 旧桥接静默映射（FM-1）
                else:
                    pre = None                 # unrepresentable
                # 独立 BoardObserver 实例计数（跨实例交叉核对）
                # V-2：口径 = 可计数控（FULL+RANK_ONLY，与 street 决议同口径）
                bo = BoardObserver()
                bo.observe(rgba, w, h)
                bsum = bo.summary()
                bcount = bsum.get("last_countable_count",
                                  bsum.get("last_count"))
                out_rows.append(json.dumps({
                    "file": r["file"], "event": r["event"], "ts": r["ts"],
                    "machine_state": ms,
                    "observed_cardinality": res["observed_cardinality"],
                    "pre_street": pre,
                    "post_street": res["last_street"],
                    "post_provenance": res["provenance"],
                    "board_observer_count": bcount,
                    "err": None}, ensure_ascii=False))
            except Exception as e:
                out_rows.append(json.dumps({
                    "file": r["file"], "event": r["event"], "ts": r["ts"],
                    "machine_state": None, "observed_cardinality": None,
                    "pre_street": None, "post_street": None,
                    "post_provenance": None, "board_observer_count": None,
                    "err": str(e)[:120]}, ensure_ascii=False))
        with open(frames_path, "a", encoding="utf-8") as f:
            f.write("\n".join(out_rows) + "\n")
        man.mark(f"chunk:{c}", len(out_rows), True)
        print(f"single chunk {c}/{n_chunks - 1}: {len(out_rows)} frames "
              f"({round(time.time() - t0, 1)}s) — cumulative "
              f"{man.data['rows']}")


# ── bridge-post 模式：与 prefix 基线同帧集的修复后对照 ────────────────
def run_bridge_post(tag: str = "") -> None:
    from .production_bridge import frame_to_decision_request
    from poker_decision_ai.vision2_platform.png_decode import decode_png

    idx = ensure_index()
    all_files = {r["file"] for run in idx["runs"] for r in run["rows"]}
    pre_path = OUT / "prefix_bridge_baseline.json"
    if pre_path.exists():
        files = [r["file"] for r in json.loads(
            pre_path.read_text(encoding="utf-8"))]
    else:
        files = sorted(f for f in all_files
                       if f.startswith("SETTLED_BOARD_CHECK"))[::6][:120]
    out_path = OUT / f"postfix_bridge_baseline{tag}.json"
    done = set()
    if out_path.exists():
        done = {r["file"] for r in json.loads(
            out_path.read_text(encoding="utf-8"))}
    rows = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else []
    for name in files:
        if name in done or name not in all_files:
            continue
        w, h, rgba = decode_png((EV / name).read_bytes())
        res = frame_to_decision_request(rgba, w, h)
        br = res.get("bridge") or {}
        obs = br.get("observation") or {}
        sr = res.get("street_resolution") or {}
        rows.append({"file": name, "status": res["status"],
                     "reason": res.get("reason"),
                     "street": obs.get("street"), "board": obs.get("board"),
                     "sr_street": sr.get("street"),
                     "sr_provenance": sr.get("provenance"),
                     "sr_machine_state": sr.get("machine_state"),
                     "sr_observed_cardinality": sr.get(
                         "observed_cardinality")})
        if len(rows) % 25 == 0:
            out_path.write_text(json.dumps(rows, ensure_ascii=False,
                                           indent=1), encoding="utf-8")
            print(f"bridge-post: {len(rows)} done")
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"bridge-post complete: {len(rows)} frames")
    return out_path


# ── summarize 模式：帧级统计表（分模式）────────────────────────────────
_NEED = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_CARD_STATE = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def _dist(rows, key):
    d: dict = {}
    for r in rows:
        k = r.get(key)
        d[k] = d.get(k, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: str(kv[0])))


def summarize(tag: str = "") -> dict:
    frames_path = OUT / f"street_stream_frames{tag}.jsonl"
    runs_path = OUT / f"street_stream_runs{tag}.jsonl"
    single_path = OUT / f"street_single_frames{tag}.jsonl"
    stats: dict = {}

    def read_jsonl(p: Path) -> list:
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    # ── 流模式 ──────────────────────────────────────────────────────
    if frames_path.exists():
        rows = read_jsonl(frames_path)
        errs = sum(1 for r in rows if r.get("err"))
        st = {"frames": len(rows), "errors": errs,
              "machine_state_dist": _dist(rows, "machine_state"),
              "resolved_street_dist": _dist(rows, "street"),
              "provenance_dist": _dist(rows, "provenance"),
              "observed_cardinality_dist": _dist(rows,
                                                 "observed_cardinality")}
        # FM-1 pending：机器停留 NEW_HAND 而当前帧已见 3-5 卡
        pend = sum(1 for r in rows if r.get("machine_state") == "NEW_HAND"
                   and (r.get("observed_cardinality") or 0) in (3, 4, 5))
        st["pending_confirm_frames_3_5_cards"] = pend
        # 已确认态帧的 street↔识别卡数一致性（确认门内滞后计入 mismatch）
        conf_rows = [r for r in rows if r.get("machine_state") in
                     ("PREFLOP", "FLOP", "TURN", "RIVER")]
        mis = sum(1 for r in conf_rows
                  if _NEED[r["machine_state"].lower()]
                  != (r.get("observed_cardinality")
                      if r.get("observed_cardinality") is not None else -1))
        st["confirmed_state_frames"] = len(conf_rows)
        st["confirmed_state_card_mismatch"] = mis
        st["confirmed_state_card_mismatch_rate"] = round(
            mis / len(conf_rows), 4) if conf_rows else None
        # 稳定板窗口代理准确率：本帧与前/后帧 observed_cardinality 相同
        # （板面稳定——无转换滞后），机器 street 必须等于卡数对应 street
        by_run: dict = {}
        for i, r in enumerate(rows):
            by_run.setdefault(r["run_id"], []).append((i, r))
        stable = correct = 0
        for _, lst in by_run.items():
            for k in range(1, len(lst) - 1):
                _, prev_r = lst[k - 1]
                _, cur_r = lst[k]
                _, next_r = lst[k + 1]
                c = cur_r.get("observed_cardinality")
                if c in (0, 3, 4, 5) and c is not None \
                        and prev_r.get("observed_cardinality") == c \
                        and next_r.get("observed_cardinality") == c:
                    stable += 1
                    st_cur = (cur_r.get("street") or "").lower()
                    if st_cur == _CARD_STATE[c] or (
                            c == 0 and st_cur == ""
                            and cur_r.get("machine_state")
                            in ("NEW_HAND", "PREFLOP")):
                        correct += 1
        st["stable_board_frames"] = stable
        st["stable_board_proxy_accuracy"] = round(correct / stable, 4) \
            if stable else None
        st["stable_board_proxy_correct"] = correct
        if runs_path.exists():
            runs = read_jsonl(runs_path)
            trans: dict = {}
            illegal = leak = hand_end = 0
            reach_flop = reach_turn = reach_river = full_chain = 0
            for rs in runs:
                for k, v in (rs.get("confirmed_transitions") or {}).items():
                    trans[k] = trans.get(k, 0) + 1
                illegal += rs.get("illegal_transitions", 0)
                leak += rs.get("card_leaks", 0)
                hand_end += rs.get("hand_ends", 0)
                reached = set(rs.get("machine_states_reached") or [])
                reach_flop += 1 if "FLOP" in reached else 0
                reach_turn += 1 if "TURN" in reached else 0
                reach_river += 1 if "RIVER" in reached else 0
                t = rs.get("confirmed_transitions") or {}
                if all(k in t for k in ("PREFLOP->FLOP", "FLOP->TURN",
                                        "TURN->RIVER")):
                    full_chain += 1
            st["runs"] = len(runs)
            st["confirmed_transitions"] = trans
            st["illegal_transitions_rejected"] = illegal
            st["confirmed_card_leaks_blocked"] = leak
            st["hand_end_resets"] = hand_end
            st["runs_reaching"] = {"FLOP": reach_flop, "TURN": reach_turn,
                                   "RIVER": reach_river,
                                   "full_chain_F_T_R": full_chain}
        stats["stream"] = st

    # ── 单帧模式 ────────────────────────────────────────────────────
    if single_path.exists():
        rows = read_jsonl(single_path)
        st = {"sample": len(rows),
              "errors": sum(1 for r in rows if r.get("err")),
              "pre_street_dist": _dist(rows, "pre_street"),
              "post_street_dist": _dist(rows, "post_street"),
              "post_provenance_dist": _dist(rows, "post_provenance"),
              "observed_cardinality_dist": _dist(rows,
                                                 "observed_cardinality")}
        fm1_fixed = sum(1 for r in rows if r.get("pre_street") == "preflop"
                        and (r.get("post_street") or "").lower()
                        in ("flop", "turn", "river"))
        both_preflop = sum(1 for r in rows
                           if r.get("pre_street") == "preflop"
                           and (r.get("post_street") or "").lower()
                           == "preflop")
        rejected = sum(1 for r in rows if r.get("post_street") is None)
        st["fm1_preflop_to_postflop_fixed"] = fm1_fixed
        st["preflop_both_agree"] = both_preflop
        st["post_rejected_unrepresentable"] = rejected
        # 修复后 street ↔ 独立 BoardObserver 计数一致性
        chk = [r for r in rows
               if (r.get("post_street") or "").lower() in _NEED
               and r.get("board_observer_count") is not None]
        xok = sum(1 for r in chk
                  if _NEED[r["post_street"].lower()]
                  == r["board_observer_count"])
        st["crosscheck_frames"] = len(chk)
        st["crosscheck_consistent"] = xok
        st["crosscheck_rate"] = round(xok / len(chk), 4) if chk else None
        stats["single"] = st

    # ── bridge 修复前/后对照 ─────────────────────────────────────────
    pre_p = OUT / "prefix_bridge_baseline.json"
    post_p = OUT / f"postfix_bridge_baseline{tag}.json"
    if not post_p.exists() and tag:
        post_p = OUT / "postfix_bridge_baseline.json"
    if pre_p.exists() and post_p.exists():
        pre = {r["file"]: r for r in json.loads(
            pre_p.read_text(encoding="utf-8"))}
        post = {r["file"]: r for r in json.loads(
            post_p.read_text(encoding="utf-8"))}
        common = sorted(set(pre) & set(post))
        status_matrix: dict = {}
        street_visible_pre = street_visible_post = 0
        postflop_surfaced = 0
        for f in common:
            key = f"{pre[f]['status']} -> {post[f]['status']}"
            status_matrix[key] = status_matrix.get(key, 0) + 1
            if pre[f].get("street"):
                street_visible_pre += 1
            if post[f].get("street"):
                street_visible_post += 1
            if (post[f].get("street") or "") in ("flop", "turn", "river"):
                postflop_surfaced += 1
            sr = post[f].get("sr_street") or ""
            if sr in ("flop", "turn", "river"):
                # 决议层（含 UNAVAILABLE 帧——street_resolution 恒携带）
                postflop_surfaced += 0  # 观测口径：observation 层单独统计
        sr_dist: dict = {}
        for f in common:
            k = (post[f].get("sr_street") or "None",
                 post[f].get("sr_provenance") or "None")
            sr_dist[k] = sr_dist.get(k, 0) + 1
        stats["bridge_compare"] = {
            "frames": len(common),
            "status_matrix": status_matrix,
            "observation_street_visible_pre": street_visible_pre,
            "observation_street_visible_post": street_visible_post,
            "street_resolution_postflop_dist": _dist(
                [post[f] for f in common], "sr_street"),
            "street_resolution_provenance_pairs": dict(
                sorted(((f"{k[0]}|{k[1]}", v) for k, v in sr_dist.items()),
                       key=lambda kv: -kv[1])),
        }
    out_path = OUT / f"street_stats{tag}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["index", "stream", "single", "bridge-post",
                             "summarize"])
    ap.add_argument("--chunk-frames", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=3400)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--tag", type=str, default="",
                    help="output filename suffix (e.g. '_v2')——不覆盖基线")
    args = ap.parse_args()
    if args.mode == "index":
        idx = ensure_index()
        print(f"index: {idx['total_frames']} frames, "
              f"{len(idx['runs'])} runs")
    elif args.mode == "stream":
        run_stream(args.chunk_frames, tag=args.tag)
    elif args.mode == "single":
        run_single(args.sample, args.seed, args.chunk_frames, tag=args.tag)
    elif args.mode == "bridge-post":
        run_bridge_post(tag=args.tag)
    elif args.mode == "summarize":
        summarize(tag=args.tag)


if __name__ == "__main__":
    main()
