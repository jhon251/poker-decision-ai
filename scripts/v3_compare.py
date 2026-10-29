"""V-3a 比对报告 — 人机一致率（真人标注 vs 机器识别）。

机器口径：**production 单帧模式对同批帧重跑**——StreetObserver +
resolve_single_frame_street（与 production_bridge 单帧路径同函数）+
BoardObserver / HeroCardsObserver（bridge 同款观察器，冻结识别器零改动）。

产出 docs/vision_v3/AGREEMENT_REPORT.md：
- 分项一致率：street / board 张数 / board rank 多重集 / board FULL suit /
  三层全过 / hero 牌；
- 结论行：`board 三层全过一致率 XX%（n=X）→ 达标/未达标`（95% 线，视觉
  自定）；
- 分歧清单（截图并排 + 机器结果 + 人工结果 + 复核判定列）；
- 系统性错误检测（多重集差口径的 rank 混淆、FULL 假阳性整牌）。

分项语义（V-2b Round 12 口径修正，诚实口径）：
- street：human=无法判断 的帧剔除；machine street=None 记 machine_unresolved，
  计为不一致（机器未能决议也是真实失败模式）；
- board L1 张数：human 卡数 vs machine countable 数（FULL+RANK_ONLY，
  street 判定的输入口径）；human unclear 剔除；
- board L2 rank 多重集：human rank 多重集 vs machine countable rank 多重集
  （FULL+RANK_ONLY 的 rank——RANK_ONLY 是设计输出，rank 是真证据）；
- board L3 FULL suit：machine 声明的 FULL 整牌是否全部出现于 human 牌面
  （子多重集）；RANK_ONLY 不进本层（单列统计，不计 suit 错误——旧 6.39%
  把设计性 RANK_ONLY 当错误属口径错误，审计确认后废弃）；
- 三层全过：L1∧L2∧(L3 非 False)；
- hero：human 可见时逐一比较（另有顺序全等口径）。

用法：
  python scripts/v3_compare.py                    # 正式批（400 帧）
  python scripts/v3_compare.py --demo             # 3 帧冒烟
  python scripts/v3_compare.py --skip-machine     # 只用已缓存机器结果
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CORPUS = REPO / "data" / "vision2" / "coinpoker_evidence"
import os as _os
CORPUS = Path(_os.environ.get("V3_CORPUS", str(CORPUS)))
PATH_MAP: dict = {}
OUT_DIR = REPO / "docs" / "vision_v3"
BATCH = OUT_DIR / "annotation_batch_400.jsonl"
ANNOT = OUT_DIR / "annotations.jsonl"
MACHINE_CACHE = OUT_DIR / "machine_results.jsonl"
ADJUDICATIONS = OUT_DIR / "adjudications.jsonl"
REPORT = OUT_DIR / "AGREEMENT_REPORT.md"
DEMO_BATCH = OUT_DIR / "demo_batch_3.jsonl"
DEMO_ANNOT = OUT_DIR / "demo_annotations.jsonl"
DEMO_CACHE = OUT_DIR / "demo_machine_results.jsonl"
DEMO_REPORT = OUT_DIR / "AGREEMENT_REPORT_demo.md"

PASS_LINE = 0.95        # 视觉自定达标线（TARGET_SPEC 无此项）

# ── 机器读取（production 单帧口径） ───────────────────────────────────
def machine_read_frame(path: Path) -> dict:
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        BoardObserver, HeroCardsObserver, StreetObserver)
    from poker_decision_ai.vision2.pixel.realclient.street_session import (
        resolve_single_frame_street)
    from poker_decision_ai.vision2_platform.png_decode import decode_png

    w, h, rgba = decode_png(path.read_bytes())
    so = StreetObserver()
    so.observe(rgba, w, h)
    street = resolve_single_frame_street(so)
    board = BoardObserver().observe(rgba, w, h)
    hero = HeroCardsObserver().observe(rgba, w, h)
    st = street.get("last_street")
    return {
        "file": path.name,
        "street": str(st).lower() if st else None,
        "observed_cardinality": street.get("observed_cardinality"),
        "street_confidence": street.get("confidence"),
        "street_provenance": street.get("provenance"),
        "street_reason": street.get("reason"),
        "board_cards": board.get("cards") or [],
        "board_status": board.get("status"),
        "board_countable": board.get("countable_count") or 0,
        "rank_only_cards": board.get("rank_only_cards") or [],
        "hero_cards": hero.get("cards"),
        "hero_status": hero.get("status"),
    }


def load_machine(cache: Path, files: list[str], skip: bool) -> dict[str, dict]:
    """读缓存 + 对缺失帧重跑（追加落盘，可断点续跑）。"""
    done: dict[str, dict] = {}
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["file"]] = r
    todo = [f for f in files if f not in done]
    if skip or not todo:
        print(f"machine cache hit {len(done)}/{len(files)}; todo={len(todo)}")
        return done
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "a", encoding="utf-8") as fout:
        for i, f in enumerate(todo):
            try:
                _cand = PATH_MAP.get(f) or (CORPUS / f)
                rec = machine_read_frame(_cand)
            except Exception as e:                       # 单帧失败不中断
                rec = {"file": f, "error": str(e)[:200]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[f] = rec
            if (i + 1) % 25 == 0:
                print(f"  machine rerun {i + 1}/{len(todo)}")
    return done


# ── 纯逻辑：逐帧比较与聚合 ────────────────────────────────────────────
def _cards_multiset(cards) -> list:
    return sorted(c for c in (cards or []) if c)


def _m_countable_ranks(machine: dict) -> list:
    """机器 countable rank 多重集（V-2 分层口径）：FULL 牌 rank + RANK_ONLY rank。"""
    full = machine.get("board_cards") or []
    ro = machine.get("rank_only_cards") or []
    return sorted([c[0] for c in full] + [c for c in ro if c])


def compare_frame(human: dict, machine: dict) -> dict:
    """单帧人机比较（纯逻辑）。各项 True/False；None=剔除/不可比。

    V-2b（Round 12）口径修正——board 逐张按 V-2 分层语义拆三层分列：
    - L1 张数（count）：human 卡数 vs machine countable 数（FULL+RANK_ONLY，
      street 判定输入口径）；
    - L2 rank 多重集（rank）：human rank 多重集 vs machine countable rank
      多重集（FULL+RANK_ONLY 的 rank；RANK_ONLY 是设计输出，其 rank 是
      真证据）；
    - L3 FULL suit（suit）：machine 声明的 FULL 牌（rank+suit 整牌）是否
      全部逐张出现于 human 牌面（子多重集）。RANK_ONLY 不进本层（单列
      统计，不计为 suit 错误——审计确认旧 6.39% 把设计性 RANK_ONLY 当
      错误属口径错误）。machine 无 FULL 声明 → 本层 None（无声明可比）。
    - 三层全过（board3）：L1∧L2∧(L3 非 False)——suit 层无声明不否决。
    """
    h_street = human.get("street")
    h_cards = human.get("board_cards") or []
    h_hero = human.get("hero_cards")
    m_street = machine.get("street")
    m_countable = machine.get("board_countable") or 0
    m_cards = machine.get("board_cards") or []
    m_hero = machine.get("hero_cards")

    street = None
    if h_street and h_street != "unclear":
        street = (h_street == m_street) if m_street else False
        if m_street is None:
            street = False          # machine_unresolved 记不一致
    count = None
    if h_street and h_street != "unclear":
        count = (len(h_cards) == m_countable)
    rank = None
    suit = None
    if h_cards:
        hr = sorted(c[0] for c in h_cards)
        rank = (hr == _m_countable_ranks(machine))
        if m_cards:
            mh = Counter(m_cards)
            hh = Counter(h_cards)
            suit = all(hh[c] >= n for c, n in mh.items())
    board3 = None
    if count is not None and rank is not None:
        board3 = bool(count and rank and (suit is not False))
    hero = None
    hero_order = None
    if human.get("hero_status") == "visible" and h_hero and \
            len([c for c in h_hero if c]) == 2:
        hero = (_cards_multiset(h_hero) == _cards_multiset(m_hero or []))
        hero_order = (list(h_hero) == list(m_hero or []))
    return {"file": human.get("file"), "street": street, "count": count,
            "rank": rank, "suit": suit, "board3": board3,
            "cards": board3,   # 兼容旧键（语义=三层全过）
            "hero": hero, "hero_order": hero_order,
            "human": {"street": h_street, "board_cards": h_cards,
                      "hero_cards": h_hero},
            "machine": {"street": m_street,
                        "observed_cardinality":
                            machine.get("observed_cardinality"),
                        "board_cards": m_cards,
                        "board_countable": m_countable,
                        "rank_only_cards": machine.get("rank_only_cards"),
                        "hero_cards": m_hero,
                        "hero_status": machine.get("hero_status"),
                        "street_provenance":
                            machine.get("street_provenance")}}


def _rate(verdicts) -> tuple[int, int, float]:
    v = [x for x in verdicts if x is not None]
    n = len(v)
    k = sum(1 for x in v if x)
    return k, n, (k / n if n else 0.0)


def aggregate(comparisons: list[dict]) -> dict:
    agg = {
        "street": _rate([c["street"] for c in comparisons]),
        "count": _rate([c["count"] for c in comparisons]),
        "rank": _rate([c["rank"] for c in comparisons]),
        "suit": _rate([c["suit"] for c in comparisons]),
        "board3": _rate([c["board3"] for c in comparisons]),
        "hero": _rate([c["hero"] for c in comparisons]),
        "hero_order": _rate([c["hero_order"] for c in comparisons]),
    }
    agg["machine_unresolved_street"] = sum(
        1 for c in comparisons
        if c["street"] is False and c["machine"]["street"] is None)
    # RANK_ONLY 单列统计（设计输出，不计 suit 错误）
    agg["frames_with_rank_only"] = sum(
        1 for c in comparisons if c["machine"].get("rank_only_cards"))
    return agg


def systematic_errors(comparisons: list[dict]) -> dict:
    """逐张分歧的系统性检测：rank/suit 混淆（多重集差口径）。

    V-2b 口径修正：旧版按槽位逐位对齐（FULL 列表与 human 槽位错位时
    产生伪混淆）。现按多重集差：machine 多出的 rank（FULL+RANK_ONLY）
    对 human 多出的 rank 配对（如 machine 多 Q、human 多 T → T→Q）。
    suit 混淆只看 FULL 声明中未出现于 human 的整牌。
    """
    rank_extra_m: Counter = Counter()   # machine 多出的 rank
    rank_extra_h: Counter = Counter()   # human 多出的 rank
    pair_conf: Counter = Counter()      # (machine_extra -> human_extra) 配对
    suit_fp: Counter = Counter()        # FULL 声明中不在 human 牌面的整牌
    for c in comparisons:
        if c.get("rank") is not False and c.get("suit") is not False:
            continue
        h = c["human"]["board_cards"]
        m_full = c["machine"]["board_cards"]
        m_ranks = _m_countable_ranks(c["machine"])
        hr = sorted(x[0] for x in h)
        em = Counter(m_ranks) - Counter(hr)
        eh = Counter(hr) - Counter(m_ranks)
        rank_extra_m += em
        rank_extra_h += eh
        for r, n in em.items():
            for h2, n2 in eh.items():
                pair_conf[f"{r}->{h2}"] += min(n, n2)
        mh = Counter(h)
        for card, n in (Counter(m_full) - mh).items():
            if n > 0:
                suit_fp[card] += n
    return {"rank_extra_machine": dict(rank_extra_m),
            "rank_extra_human": dict(rank_extra_h),
            "rank_confusion": pair_conf,
            "suit_false_positive": suit_fp}


# ── 报告渲染 ──────────────────────────────────────────────────────────
def _fmt_rate(t: tuple[int, int, float]) -> str:
    k, n, r = t
    pct = f"{r * 100:.2f}%"
    return f"{k}/{n} = {pct}" if n else "n=0（无样本）"


def render_report(agg: dict, comparisons: list[dict], meta: dict,
                  adjudications: dict[str, str] | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    k, n, r = agg["board3"]
    verdict = "达标" if (n > 0 and r >= PASS_LINE) else "未达标"
    L: list[str] = []
    L.append("# V-3a 人机一致率报告（AGREEMENT_REPORT）")
    L.append("")
    if meta.get("demo"):
        L.append("> **⚠ 冒烟数据（SMOKE）**：本报告由程序化冒烟标注驱动"
                 "（v3_annotate.py --smoke 的合成值），**不是真人标注**，"
                 "一致率数字无意义——仅证明比对管线端到端可运行。正式报告"
                 "等 400 帧真人标注完成后生成。")
        L.append("")
    L.append(f"- 生成时间：{now}；标注者：项目所有者（真人盲标）；"
             f"机器口径：production 单帧重跑（冻结识别器，零改动）")
    L.append(f"- 批清单：`{meta.get('batch', 'annotation_batch_400.jsonl')}`"
             f"（seed={meta.get('seed', 'n/a')}）；"
             f"已标注 {meta.get('n_annotated', 0)}/"
             f"{meta.get('n_batch', 0)} 帧，本报告比较 "
             f"{meta.get('n_compared', 0)} 帧")
    L.append(f"- 盲标声明：标注工具不展示机器结果；机器结果为标注完成后"
             f"对同批帧的独立单帧重跑（与 V-1 流式口径不同属正常）")
    L.append("- 口径版本：V-2b（Round 12）——board 逐张按 V-2 分层语义"
             "拆三层分列（张数 / rank 多重集 / FULL suit），RANK_ONLY 为"
             "设计输出单列统计、不计入 suit 一致；旧版 6.39% 单一数字"
             "把设计性 RANK_ONLY 当错误，属口径错误（审计确认），已废弃")
    L.append("")
    L.append("## 结论")
    L.append("")
    L.append(f"**board 三层全过一致率 {r * 100:.2f}%（n={n}）→ {verdict}**"
             f"（达标线 {PASS_LINE * 100:.0f}%，视觉自定标准；三层=张数×"
             f"rank 多重集×FULL suit）")
    L.append("")
    L.append("## 分项一致率（board 三层分列）")
    L.append("")
    L.append("| 分项 | 一致/样本 | 一致率 | 口径 |")
    L.append("|---|---|---|---|")
    L.append(f"| street | {_fmt_rate(agg['street'])} | — |"
             f" human≠unclear 帧；machine 未决议计不一致"
             f"（{agg['machine_unresolved_street']} 帧 machine_unresolved）|")
    L.append(f"| board L1 张数 | {_fmt_rate(agg['count'])} | — |"
             f" human 卡数 vs machine countable 数（FULL+RANK_ONLY）|")
    L.append(f"| board L2 rank 多重集 | {_fmt_rate(agg['rank'])} | — |"
             f" human rank 多重集 vs machine countable rank 多重集"
             f"（含 RANK_ONLY 的 rank）|")
    L.append(f"| board L3 FULL suit | {_fmt_rate(agg['suit'])} | — |"
             f" machine FULL 整牌 ⊆ human 牌面（子多重集）；无 FULL 声明"
             f"帧剔除；RANK_ONLY 单列（{agg['frames_with_rank_only']} 帧"
             f"含 RANK_ONLY，不计本层）|")
    L.append(f"| board 三层全过 | {_fmt_rate(agg['board3'])} | — |"
             f" L1∧L2∧(L3 非 False) |")
    L.append(f"| hero 牌（不计序） | {_fmt_rate(agg['hero'])} | — |"
             f" human 可见帧 |")
    L.append(f"| hero 牌（含顺序） | {_fmt_rate(agg['hero_order'])} | — |"
             f" 同上 |")
    L.append("")
    L.append("## 系统性错误检测（多重集差口径）")
    L.append("")
    se = meta.get("systematic") or {}
    rc = se.get("rank_confusion") or Counter()
    sfp = se.get("suit_false_positive") or Counter()
    L.append(f"- rank 多出配对（machine 多→human 多）："
             f"{dict(rc.most_common(10)) if rc else '无'}")
    L.append(f"- machine 多出 rank：{se.get('rank_extra_machine') or '无'}；"
             f"human 多出 rank：{se.get('rank_extra_human') or '无'}")
    L.append(f"- FULL 假阳性整牌（声明但不在人标牌面）："
             f"{dict(sfp.most_common(10)) if sfp else '无'}"
             f"（0 误报纪律口径）")
    L.append("")
    L.append("## 分歧清单（每帧：截图 vs 机器 vs 人工；待人工复核判定）")
    L.append("")
    L.append("复核判定列取值：机器错 / 人错 / 都错 / 存疑。"
             "填 `docs/vision_v3/adjudications.jsonl`"
             "（每行 {\"file\":..., \"verdict\":...}）后重跑本脚本自动带入。")
    L.append("")
    dis = [c for c in comparisons
           if any(c[k2] is False for k2 in ("street", "count", "cards",
                                            "hero"))]
    if not dis:
        L.append("（无分歧帧）")
    for i, c in enumerate(dis, 1):
        f = c["file"]
        img = PATH_MAP.get(f) or (CORPUS / f)
        out_base = Path(meta.get("out_dir") or OUT_DIR)
        img_rel = _os.path.relpath(img.resolve(),
                                  out_base.resolve())
        verdict_col = (adjudications or {}).get(f, "（待复核）")
        L.append(f"### 分歧 {i}: `{f}` — 判定：{verdict_col}")
        L.append("")
        L.append(f"![{f}]({img_rel})")
        L.append("")
        L.append("| 项 | 机器 | 人工 |")
        L.append("|---|---|---|")
        L.append(f"| street | {c['machine']['street'] or '(未决议)'}"
                 f"（{c['machine'].get('street_provenance') or '-'}）"
                 f" | {c['human']['street']} |")
        layers = []
        for key, name in (("count", "张数"), ("rank", "rank多重集"),
                          ("suit", "FULLsuit")):
            v = c[key]
            layers.append(f"{name}={'一致' if v else ('剔除' if v is None else '不一致')}")
        L.append(f"| board 三层 | {' '.join(layers)}（三层全过="
                 f"{'是' if c['board3'] else '否'}） |"
                 f" human={c['human']['board_cards']} |")
        L.append(f"| board 明细 | {c['machine']['board_cards']}"
                 f" countable={c['machine']['board_countable']}"
                 f" rank_only={c['machine']['rank_only_cards']}"
                 f" | {c['human']['board_cards']} |")
        L.append(f"| hero | {c['machine']['hero_cards']} "
                 f"({c['machine']['hero_status']}) "
                 f"| {c['human']['hero_cards']} "
                 f"({c['human'].get('hero_status') or '-'}) |")
        L.append("")
    L.append("## 不是什么")
    L.append("")
    L.append("- 本报告不是真实玩家池盈利或决策质量声明——只测**屏幕识别**"
             "层面的人机一致率；")
    L.append("- 机器口径为单帧重跑（无时序上下文），与 V-1/V-2 的流式"
             "口径、99.86% 召回结论不互斥也不互证；")
    L.append("- 人工标注是项目所有者的单标注者结果，非多人交叉验证；")
    L.append(f"- 达标线 {PASS_LINE * 100:.0f}% 为视觉模块自定标准"
             f"（TARGET_SPEC 无此项）。")
    L.append("")
    return "\n".join(L)


# ── IO 驱动 ───────────────────────────────────────────────────────────
def load_annotations(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["file"]] = r
    return out


def load_adjudications(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["file"]] = r.get("verdict", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=BATCH)
    ap.add_argument("--annotations", type=Path, default=ANNOT)
    ap.add_argument("--machine-cache", type=Path, default=MACHINE_CACHE)
    ap.add_argument("--adjudications", type=Path, default=ADJUDICATIONS)
    ap.add_argument("--out", type=Path, default=REPORT)
    ap.add_argument("--limit", type=int, default=0,
                    help="只比较前 N 帧（冒烟）")
    ap.add_argument("--skip-machine", action="store_true",
                    help="不重跑机器识别，仅用缓存")
    ap.add_argument("--demo", action="store_true",
                    help="demo 批（demo_batch_3.jsonl / demo_annotations）")
    args = ap.parse_args()
    if args.demo:
        args.batch, args.annotations = DEMO_BATCH, DEMO_ANNOT
        args.machine_cache, args.out = DEMO_CACHE, DEMO_REPORT

    batch = [json.loads(l) for l in
             args.batch.read_text(encoding="utf-8").splitlines() if l]
    files = [b["file"] for b in batch]
    if args.limit:
        files = files[:args.limit]
    humans = load_annotations(args.annotations)
    ann_files = [f for f in files if f in humans]
    print(f"batch={len(files)} annotated={len(ann_files)}")
    if not ann_files:
        print("nothing annotated yet — nothing to compare")
        return 1
    global PATH_MAP
    PATH_MAP = {r["file"]: Path(r["path"]) for r in humans.values()
                if r.get("path")}
    # 补充: 标注记录缺 path 时从批清单取
    try:
        for ln in open(args.batch, encoding="utf-8"):
            if ln.strip():
                b = json.loads(ln)
                PATH_MAP.setdefault(b["file"], Path(b["path"]))
    except Exception:
        pass
    # 坏缓存清零: 旧缓存全部指向已删除的旧语料, 全部重跑
    #（--skip-machine 例外：显式要求只用缓存，不清零——V-2b 修正，
    #  否则该开关永远空转）
    if args.machine_cache.exists() and not args.skip_machine:
        args.machine_cache.unlink()
        print("stale machine cache (old corpus) cleared")
    machine = load_machine(args.machine_cache, ann_files,
                           skip=args.skip_machine)
    comparisons = []
    for f in ann_files:
        m = machine.get(f)
        if m is None:
            print(f"  WARN no machine result for {f} (skipped)")
            continue
        if m.get("error"):
            m = {"file": f, "street": None, "observed_cardinality": None,
                 "board_cards": [], "board_countable": 0,
                 "rank_only_cards": [], "hero_cards": None,
                 "hero_status": "ERROR", "street_provenance": "error"}
        comparisons.append(compare_frame(humans[f], m))
    agg = aggregate(comparisons)
    meta = {
        "batch": args.batch.name, "seed": "n/a",
        "out_dir": str(args.out.parent),
        "n_annotated": len(humans), "n_batch": len(files),
        "n_compared": len(comparisons), "systematic":
            systematic_errors(comparisons), "demo": bool(args.demo),
    }
    adj = load_adjudications(args.adjudications)
    report = render_report(agg, comparisons, meta,
                           adjudications=adj or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"report -> {args.out}")
    k, n, r = agg["board3"]
    print(f"board 三层全过一致率 {r * 100:.2f}% (n={n}) | "
          f"张数 {_fmt_rate(agg['count'])} | "
          f"rank多重集 {_fmt_rate(agg['rank'])} | "
          f"FULLsuit {_fmt_rate(agg['suit'])} | "
          f"street {_fmt_rate(agg['street'])} | "
          f"hero {_fmt_rate(agg['hero'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
