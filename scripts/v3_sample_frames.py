"""V-3a 抽样器 — 从 CoinPoker 真实截图语料抽 400 帧给真人标注。

分歧优先（~250）+ 随机层抽（~150）。分歧信号来自 V-1 机器识别清单
（docs/vision_v1/street_stream_frames.jsonl，覆盖全部 *_full.png 帧），
仅用于选帧；机器结果写入批清单的 machine_snapshot 供比对溯源，
**标注工具永不展示**（盲标原则）。

纯逻辑函数（信号判定 / 分层抽样 / 去重）与 IO 分离，供单测
（tests/test_vision_v3_annotation.py）。

用法：
    python scripts/v3_sample_frames.py            # 产出 400 帧正式批
    python scripts/v3_sample_frames.py --demo 3   # 3 帧冒烟批
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STREAM_JSONL = REPO / "docs" / "vision_v1" / "street_stream_frames.jsonl"
CORPUS_DIR = REPO / "data" / "vision2" / "coinpoker_evidence"
OUT_DIR = REPO / "docs" / "vision_v3"
OUT_BATCH = OUT_DIR / "annotation_batch_400.jsonl"
OUT_DEMO = OUT_DIR / "demo_batch_3.jsonl"

N_DISAGREEMENT = 250
N_RANDOM = 150
DEFAULT_SEED = 20261005

#: street → 该街线应有的公共牌数（与冻结 _STREET_NEED 同义，本地副本）
STREET_NEED = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: 分歧信号桶上限（防大桶淹没稀有信号；round-robin 在各桶上限内轮转）
SIGNAL_BUCKET_CAPS = {
    "illegal_transition": 10,
    "overfull_board": 20,
    "street_card_mismatch": 40,
    "partial_cards": 70,
    "ghost_cards": 70,
    "street_unknown": 70,
    "low_confidence": 20,
    "evidence_limited": 40,
    "frame_error": 20,
}


# ── 纯逻辑：行归一化与分歧信号 ────────────────────────────────────────
def normalize_row(row: dict) -> dict:
    """V-1 stream 行 → 统一记录（street 小写、字段齐备）。"""
    street = row.get("street")
    street_l = str(street).lower() if street else None
    return {
        "file": row.get("file"),
        "event": row.get("event"),
        "ts": row.get("ts"),
        "street": street_l,
        "observed_cardinality": row.get("observed_cardinality"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "provenance": row.get("provenance"),
        "err": row.get("err"),
    }


def disagreement_signals(rec: dict) -> list[str]:
    """帧的机器识别分歧信号标签（按严重度排列；空表=无信号）。

    - illegal_transition：冻结状态机判定非法转换（0→4 / 0→5）；
    - overfull_board：观测卡数 >5；
    - street_card_mismatch：street 给了 flop/turn/river 但卡数对不上；
    - partial_cards：观测卡数 1-2（噪声/发牌动画/漏读——单帧模式拒绝）；
    - ghost_cards：preflop 却观测到 >0 张（鬼卡嫌疑）；
    - street_unknown：street 无决议（None）；
    - low_confidence：confidence < 0.55；
    - evidence_limited：证据受限帧。
    """
    out: list[str] = []
    street = rec.get("street")
    obs = rec.get("observed_cardinality")
    reason = rec.get("reason") or ""
    if rec.get("err"):
        out.append("frame_error")
        return out
    if str(reason).startswith("illegal_transition"):
        out.append("illegal_transition")
    if isinstance(obs, int) and obs > 5:
        out.append("overfull_board")
    if street in ("flop", "turn", "river"):
        if not isinstance(obs, int) or obs != STREET_NEED[street]:
            out.append("street_card_mismatch")
    if isinstance(obs, int) and obs in (1, 2):
        out.append("partial_cards")
    if street == "preflop" and isinstance(obs, int) and obs > 0:
        out.append("ghost_cards")
    if street is None:
        out.append("street_unknown")
    conf = rec.get("confidence")
    if isinstance(conf, (int, float)) and conf < 0.55:
        out.append("low_confidence")
    if reason == "evidence_limited":
        out.append("evidence_limited")
    return out


#: 严重度优先序（round-robin 轮转时的桶顺序）
SIGNAL_PRIORITY = ("illegal_transition", "overfull_board",
                   "street_card_mismatch", "partial_cards", "ghost_cards",
                   "street_unknown", "low_confidence", "evidence_limited",
                   "frame_error")


# ── 纯逻辑：抽样 ──────────────────────────────────────────────────────
def build_signal_buckets(recs: list[dict]) -> dict[str, list[dict]]:
    """记录 → {信号: 记录列表}（一帧可入多桶；桶内按文件名排序稳定）。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for sig in disagreement_signals(r):
            buckets[sig].append(r)
    for k in buckets:
        buckets[k].sort(key=lambda r: (r.get("file") or ""))
    return dict(buckets)


def pick_disagreement(recs: list[dict], n: int) -> list[dict]:
    """分歧优先抽样：跨信号桶 round-robin（稀有信号优先入列），
    每桶限幅 SIGNAL_BUCKET_CAPS，去重，返回 ≤n 条（稳定序）。"""
    buckets = build_signal_buckets(recs)
    capped = {}
    for sig, items in buckets.items():
        cap = SIGNAL_BUCKET_CAPS.get(sig, len(items))
        capped[sig] = items[:cap]
    chosen: dict[str, dict] = {}
    order = [s for s in SIGNAL_PRIORITY if s in capped]
    # round-robin：每轮从每个桶取一条，直到满 n 或桶尽
    idx = 0
    while len(chosen) < n and any(idx < len(capped[s]) for s in order):
        for sig in order:
            if len(chosen) >= n:
                break
            items = capped[sig]
            if idx < len(items):
                rec = items[idx]
                chosen.setdefault(rec["file"], rec)
        idx += 1
    if len(chosen) < n:
        # 回填：桶上限+桶间重叠导致不足时，从全部带信号帧（文件名序）
        # 补满——仍是分歧优先语义，只是超出单桶配额的帧。
        all_sig = sorted((r for r in recs if disagreement_signals(r)),
                         key=lambda r: (r.get("file") or ""))
        for rec in all_sig:
            if len(chosen) >= n:
                break
            chosen.setdefault(rec["file"], rec)
    return [chosen[k] for k in sorted(chosen)]


def discover_events(recs: list[dict]) -> list[str]:
    """从数据动态收集事件类型（避免按前缀猜名字漏层）。"""
    return sorted({r.get("event") for r in recs if r.get("event")})


def stratified_random(recs: list[dict], n: int,
                      exclude: set[str], seed: int) -> list[dict]:
    """按事件类型分层随机抽样（比例分配 + 每事件至少 2 帧），
    从 exclude（已入选文件）之外的候选中抽。"""
    rng = random.Random(seed)
    events = discover_events(recs)
    pool = [r for r in recs if (r.get("file") or "") not in exclude
            and r.get("event") in events]
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_event[r["event"]].append(r)
    for ev in by_event:
        by_event[ev].sort(key=lambda r: (r.get("file") or ""))
    if not pool:
        return []
    # 比例配额（大数份额给大事件层），再保证每层 ≥2（若层足够大）
    quota: dict[str, int] = {}
    total = len(pool)
    for ev, items in by_event.items():
        quota[ev] = max(2, round(n * len(items) / total))
    # 收缩/扩张到恰好 n
    while sum(quota.values()) > n:
        biggest = max(quota, key=lambda e: quota[e])
        if quota[biggest] <= 2:
            # 全部层都已到下限还超——只能砍某一层到 1
            over = [e for e in quota if quota[e] > 1]
            if not over:
                break
            biggest = max(over, key=lambda e: quota[e])
        quota[biggest] -= 1
    while sum(quota.values()) < n:
        growable = [e for e in quota if quota[e] < len(by_event[e])]
        if not growable:
            break
        pick = min(growable, key=lambda e: quota[e] / max(len(by_event[e]), 1))
        quota[pick] += 1
    picked: list[dict] = []
    for ev in sorted(quota):
        k = min(quota[ev], len(by_event[ev]))
        picked.extend(rng.sample(by_event[ev], k))
    return picked


def select_frames(recs: list[dict], n_disagreement: int, n_random: int,
                  seed: int) -> tuple[list[dict], list[dict], dict]:
    """主入口：返回（分歧组, 随机组, 统计）。两组文件互斥。"""
    disagreement = pick_disagreement(recs, n_disagreement)
    exclude = {r["file"] for r in disagreement}
    rand = stratified_random(recs, n_random, exclude, seed)
    sig_count = Counter()
    for r in disagreement:
        sigs = [s for s in disagreement_signals(r)]
        sig_count[sigs[0] if sigs else "(none)"] += 1
    stats = {
        "corpus_rows": len(recs),
        "disagreement_selected": len(disagreement),
        "random_selected": len(rand),
        "random_by_event": dict(Counter(r["event"] for r in rand)),
        "disagreement_primary_signal": dict(sig_count),
        "seed": seed,
    }
    return disagreement, rand, stats


def pick_demo(recs: list[dict], n: int, seed: int) -> list[dict]:
    """冒烟批：n 帧覆盖尽量多的事件类型（确定性）。"""
    rng = random.Random(seed)
    by_event: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("event") in discover_events(recs):
            by_event[r["event"]].append(r)
    for ev in by_event:
        by_event[ev].sort(key=lambda r: (r.get("file") or ""))
    picked: list[dict] = []
    events = sorted(by_event)
    i = 0
    while len(picked) < n and events:
        ev = events[i % len(events)]
        items = by_event[ev]
        if items:
            picked.append(items[rng.randrange(len(items))])
            by_event[ev] = [x for x in items
                            if x["file"] != picked[-1]["file"]]
            if not by_event[ev]:
                events.remove(ev)
                i = 0
                continue
        i += 1
    return picked[:n]


# ── IO ────────────────────────────────────────────────────────────────
def load_rows(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(normalize_row(json.loads(line)))
    return recs


def batch_line(rec: dict, group: str, machine_snapshot: dict,
               corpus_dir_effective=CORPUS_DIR) -> dict:
    return {
        "file": rec["file"],
        "path": str(corpus_dir_effective / rec["file"]),
        "event": rec["event"],
        "ts": rec["ts"],
        "group": group,
        "machine_snapshot": machine_snapshot,   # 仅供比对——标注工具不展示
    }


def main() -> int:
    global CORPUS_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream-jsonl", type=Path, default=STREAM_JSONL)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="输出批清单（默认 annotation_batch_400.jsonl；"
                         "--demo 时为 demo_batch_3.jsonl）")
    ap.add_argument("--n-disagreement", type=int, default=N_DISAGREEMENT)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--demo", type=int, default=0,
                    help="生成 N 帧冒烟批（覆盖多事件类型）而非正式批")
    args = ap.parse_args()

    recs = load_rows(args.stream_jsonl)
    # 只保留语料里真实存在的 *_full.png
    CORPUS_DIR = args.corpus_dir
    existing = {p.name for p in args.corpus_dir.glob("*_full.png")}
    recs = [r for r in recs if r["file"] in existing]
    print(f"machine rows matched to corpus: {len(recs)}")

    if args.demo:
        picked = pick_demo(recs, args.demo, args.seed)
        out = args.out or OUT_DEMO
        lines = [batch_line(r, "demo", _snapshot(r)) for r in picked]
    else:
        disagreement, rand, stats = select_frames(
            recs, args.n_disagreement, args.n_random, args.seed)
        picked = disagreement + rand
        out = args.out or OUT_BATCH
        lines = [batch_line(r, "disagreement", _snapshot(r),
                            corpus_dir_effective=CORPUS_DIR)
                 for r in disagreement]
        lines += [batch_line(r, "random", _snapshot(r),
                             corpus_dir_effective=CORPUS_DIR) for r in rand]
        print(json.dumps(stats, ensure_ascii=False, indent=1))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} frames -> {out}")
    return 0


def _snapshot(rec: dict) -> dict:
    """机器识别快照（V-1 stream 口径，溯源用；比对以 fresh 单帧重跑为准）。"""
    return {k: rec.get(k) for k in
            ("street", "observed_cardinality", "confidence", "reason",
             "provenance")}


if __name__ == "__main__":
    sys.exit(main())
