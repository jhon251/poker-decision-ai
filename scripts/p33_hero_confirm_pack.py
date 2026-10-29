"""Hero GT 人工确认材料打包（只读——不改 detector/模板/阈值/覆盖面）。

产出 gt_confirm/ 目录：
  pages/page_NNN__<frame>.png   每帧一页：上手牌 pair-ROI 全分辨率 +
                                下方全桌 1/2 缩略（供核对 hero 座位）
  GT_CONFIRM_TEMPLATE.csv       盲认清单（不含机器识别结果，避免锚定）：
                                page, frame, hero_seat_is_S6, card1, card2,
                                confirm_level, notes —— 后 5 列由用户填
  all_unique_candidates.csv     全部去重候选（含机器现状，仅技术参考）
  README.md                     确认说明 + 缺失 label 清单 + 后续流程

红线：本脚本不做任何身份推断；"已识别"帧用现有 detector 只读判定并排除
（它们无需人工确认）；identity 只能来自用户看图确认。
"""
from __future__ import annotations

import csv
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EV = Path("data/vision2/coinpoker_evidence")
CANDS = Path("logs/hero_presence_candidates.json")
OUT = Path("gt_confirm")
PAIR = (795, 1005, 1070, 1195)

# 供参考的目标缺口（README 用）：现有模板 rank={3,5,6,9} suit={c,d,h}
MISSING_RANKS = ["A", "K", "Q", "J", "T", "8", "7", "4", "2"]
MISSING_SUITS = ["s"]


def _process(path_str: str):
    """解码 → ROI 裁剪 + 哈希 + 现有 detector 只读判定。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 \
        import HeroCardsRecognizer1876
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    p = Path(path_str)
    try:
        w, h, rgba = decode_png(p.read_bytes())
    except Exception as e:  # noqa: BLE001
        return {"frame": p.name, "error": str(e)[:80]}
    if (w, h) != (1876, 1400):
        return {"frame": p.name, "skip": "geometry"}
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, y0, x1, y1 = PAIR
    roi = arr[y0:y1, x0:x1]
    roi_hash = hashlib.blake2b(roi.tobytes(), digest_size=12).hexdigest()
    rec = _RECOGNIZER.recognize(arr)
    return {"frame": p.name, "roi_hash": roi_hash,
            "rec_status": rec.get("status"),
            "rec_value": rec.get("value"),
            "rec_reason": rec.get("reason"),
            "roi_png": None}   # ROI 原图由主进程按需编码


_RECOGNIZER = None


def _init_worker():
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 \
        import HeroCardsRecognizer1876
    global _RECOGNIZER
    _RECOGNIZER = HeroCardsRecognizer1876()


def _block_mean2(a: np.ndarray) -> np.ndarray:
    """2×2 块均值缩略（纯 numpy）。"""
    h, w = a.shape[:2]
    h2, w2 = h // 2 * 2, w // 2 * 2
    b = a[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, -1)
    return b.mean(axis=(1, 3)).astype(np.uint8)


def main() -> int:
    import json
    cands = [c["frame"] for c in json.loads(
        CANDS.read_text(encoding="utf-8"))]
    paths = sorted(str(EV / f) for f in cands)
    print(f"candidates: {len(paths)}", flush=True)

    results = []
    with Pool(12, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_process, paths,
                                                  chunksize=8)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(paths)}]", flush=True)

    # 去重（同 ROI 内容哈希 = 同一副牌面；保留 white_px 最高者）
    by_hash = {}
    for r in results:
        if "roi_hash" not in r:
            continue
        cur = by_hash.get(r["roi_hash"])
        if cur is None or r["frame"] < cur["frame"]:
            by_hash[r["roi_hash"]] = r
    unique = sorted(by_hash.values(), key=lambda r: r["frame"])
    need = [r for r in unique if r.get("rec_status") != "OK"]
    already = [r for r in unique if r.get("rec_status") == "OK"]
    print(f"unique(ROI-hash): {len(unique)}  已被现有模板识别: {len(already)}"
          f"  需人工确认: {len(need)}", flush=True)
    from collections import Counter
    dist = Counter(r.get("rec_reason") for r in need)
    print("need 分诊分布:", dict(dist), flush=True)

    (OUT / "pages").mkdir(parents=True, exist_ok=True)

    # 全部去重候选（技术参考，含机器现状）
    with open(OUT / "all_unique_candidates.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "roi_hash", "rec_status", "rec_value",
                     "rec_reason"])
        for r in unique:
            wr.writerow([r["frame"], r.get("roi_hash", ""),
                         r.get("rec_status", ""), r.get("rec_value", ""),
                         r.get("rec_reason", "")])

    # 盲认页 + 模板 CSV（不含机器识别结果）
    # 分诊排序（用现有 detector 失败原因，不做身份推断）：
    #   0 glyph_gate_failed（字形在，只是不在模板集——新 label 概率最高）
    #   1 card_recognition_incomplete（一卡过一卡败）
    #   2 其他（no_card_pair_present 等）
    #   3 corner_ink_insufficient（暗牌/卡背概率大，最末）
    def _group(r):
        return {"glyph_gate_failed": 0,
                "card_recognition_incomplete": 1,
                "corner_ink_insufficient": 3}.get(r.get("rec_reason"), 2)
    need.sort(key=lambda r: (_group(r), r["frame"]))
    rows = []
    for i, r in enumerate(need, 1):
        w, h, rgba = None, None, None
        from poker_decision_ai.vision2_platform.png_decode import decode_png
        w, h, rgba = decode_png((EV / r["frame"]).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        x0, y0, x1, y1 = PAIR
        roi = arr[y0:y1, x0:x1]
        thumb = _block_mean2(arr)
        gap = 8
        page = np.zeros((roi.shape[0] + gap + thumb.shape[0],
                         max(roi.shape[1], thumb.shape[1]), 4), dtype=np.uint8)
        page[..., :3] = 24
        page[..., 3] = 255
        page[:roi.shape[0], :roi.shape[1]] = roi
        page[roi.shape[0] + gap:roi.shape[0] + gap + thumb.shape[0],
             :thumb.shape[1]] = thumb
        from poker_decision_ai.vision2_platform.png_encode import (
            encode_rgba_png)
        name = f"page_{i:03d}__{r['frame'][:-4]}"   # 帧名已含 .png
        (OUT / "pages" / f"{name}.png").write_bytes(
            encode_rgba_png(page.shape[1], page.shape[0], page.tobytes()))
        rows.append([f"pages/{name}.png", r["frame"], "", "", "", "", ""])
        if i % 20 == 0:
            print(f"pages {i}…", flush=True)

    with open(OUT / "GT_CONFIRM_TEMPLATE.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["page_file", "frame", "hero_seat_is_S6(Y/N/unsure)",
                     "card1(如 As)", "card2", "confirm_level(certain/unsure)",
                     "notes"])
        wr.writerows(rows)

    (OUT / "README.md").write_text(
        "# Hero GT 人工确认包\n\n"
        "## 每页需要确认的信息（只填你亲眼确认的）\n"
        "1. **hero_seat_is_S6**：牌面朝上的两张牌是否位于 S6 座位\n"
        "   （下方缩略图中核对座位标签；看不清填 unsure）。\n"
        "2. **card1 / card2**：两张手牌身份，格式如 `As`、`Td`\n"
        "   （card1=左侧/后卡，card2=右侧/前卡——以确认页上手牌区为准）。\n"
        "3. **confirm_level**：certain（完全确定）/ unsure（存疑）。\n"
        "   unsure 或卡背/遮挡 → 该帧不进 GT。\n\n"
        "## 红线\n- 只确认你亲眼看到的；禁止按规则/位置/历史猜牌。\n"
        "- 现有模板覆盖 rank={3,5,6,9} suit={c,d,h}——这些 label 的帧\n"
        "  无需再确认（已被现有 detector 识别并排除在清单外）。\n\n"
        "## 目标缺口（优先找这些）\n"
        f"- rank: {' '.join(MISSING_RANKS)}（最好 card1 与 card2 各一例）\n"
        f"- suit: {' '.join(MISSING_SUITS)}（card1 与 card2 各一例）\n"
        "- 两张牌都是缺口 label 的帧价值最高。\n\n"
        "## 确认后流程（由我执行）\n"
        "1. 你回填 GT_CONFIRM_TEMPLATE.csv → 我转录进\n"
        "   `scripts/p33_hero_gt_extend.py` 的 GT_EXT；\n"
        "2. 跑提取（固定角标窗，阈值/公式零改动，仅增 GT 资产 + manifest 溯源）；\n"
        "3. round-trip（识别器输出必须与你的标注一致）；\n"
        "4. 42 帧回归 diff（已识别帧结果不得改变）；\n"
        "5. 全量测试；6. 重新暂存分发。\n",
        encoding="utf-8")
    print(f"pages: {len(rows)} -> {OUT/'pages'}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
