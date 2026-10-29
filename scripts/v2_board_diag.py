"""V-2 diagnosis — instrumented board slot analysis (no production change).

Phase A: all SETTLED_BOARD_CHECK frames whose V-1 observed_cardinality was 1-2
(the "miss" set) -> per-slot cause distribution (white gate / rank gate /
suit evidence / family filter / segmentation).
Phase B: ghost sample (PREFLOP confirmed + 1-2 cards observed) vs true-preflop
0-card frames -> per-slot white/rank/suit feature distributions for the ghost
gate design.

Uses the production CoinPokerBoardRecognizer internals (same math), only
exposing intermediate values. Pure diagnostics; outputs under
docs/vision_v2_board/.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from poker_decision_ai.vision2.realtime.image import ImageData, to_gray
from poker_decision_ai.vision2.pixel.realclient.coinpoker_board import (
    CoinPokerBoardRecognizer, EMPTY_SLOT_MIN_WHITE, _pad_tight, _slide_prepped)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_hero import (
    _tight_glyph)
from poker_decision_ai.vision2.pixel.realclient.render_class import (
    window_for_frame)
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = REPO / "data" / "vision2" / "coinpoker_evidence"
OUT = REPO / "docs" / "vision_v2_board"


def slot_diag(rec: CoinPokerBoardRecognizer, frame: np.ndarray, slot: int,
              w: int) -> dict:
    """Replicates recognize_slot with intermediates exposed (same math)."""
    x0 = rec.slot_x0 + slot * rec.slot_w
    if x0 + rec.slot_w > w:
        return {"slot": slot, "reason": "slot_oob"}
    slot_card = frame[rec.slot_y0:rec.slot_y0 + rec.slot_h,
                      x0:x0 + rec.slot_w]
    luma = (slot_card[..., 0].astype(np.int16) * 0.299
            + slot_card[..., 1].astype(np.int16) * 0.587
            + slot_card[..., 2].astype(np.int16) * 0.114)
    white = float((luma > 200).mean())
    d = {"slot": slot, "white_frac": round(white, 4),
         "white_gt_150": round(float((luma > 150).mean()), 4),
         "white_gt_180": round(float((luma > 180).mean()), 4),
         "luma_mean": round(float(luma.mean()), 1)}
    if white < EMPTY_SLOT_MIN_WHITE:
        d["reason"] = "empty_slot_white_gate"
        # still measure rank evidence if any bright content exists at all
        if float((luma > 120).mean()) < 0.02:
            d["rank_best"] = None
            return d
    corner = frame[rec.slot_y0 + 6:rec.slot_y0 + 6 + 57,
                   x0 + 6:x0 + 6 + 49]
    suit_zone = frame[rec.slot_y0 + 62:rec.slot_y0 + 62 + 28,
                      x0 + 6:x0 + 6 + 55]
    corner_img = _pad_tight(_tight_glyph(ImageData.from_rgba_bytes(
        corner.shape[1], corner.shape[0], corner.tobytes())))
    suit_img = _pad_tight(_tight_glyph(ImageData.from_rgba_bytes(
        suit_zone.shape[1], suit_zone.shape[0], suit_zone.tobytes())))
    red_dom = rec._red_dominant(corner_img)
    d["red_dom"] = bool(red_dom)
    g = np.asarray(to_gray(corner_img).data, dtype=float)
    gs = np.asarray(to_gray(suit_img).data, dtype=float)
    # filtered (production path)
    ranks_f = rec._fam_prepped(rec._prep_ranks, red_dom)
    suits_f = rec._fam_prepped(rec._prep_suits, red_dom)
    rn, rs, _, rm = _slide_prepped(g, ranks_f)
    cn, cs, _, cm = _slide_prepped(gs, suits_f)
    cen = rec._center_evidence(frame, slot, red_dom)
    # unfiltered (family-off diagnostic)
    rn_a, rs_a, _, rm_a = _slide_prepped(g, rec._prep_ranks)
    cn_a, cs_a, _, cm_a = _slide_prepped(gs, rec._prep_suits)
    d.update({
        "rank": rn.split("@")[0] if rn else None,
        "rank_score": round(rs, 4) if rs is not None else None,
        "rank_margin": round(rm, 4) if rm is not None else None,
        "rank_all": rn_a.split("@")[0] if rn_a else None,
        "rank_all_score": round(rs_a, 4) if rs_a is not None else None,
        "rank_all_margin": round(rm_a, 4) if rm_a is not None else None,
        "corner_suit": cn.split("@")[0] if cn else None,
        "corner_suit_score": round(cs, 4) if cs is not None else None,
        "corner_suit_margin": round(cm, 4) if cm is not None else None,
        "corner_suit_all": cn_a.split("@")[0] if cn_a else None,
        "corner_suit_all_score": round(cs_a, 4) if cs_a is not None else None,
        "center_suit": cen.get("suit"),
        "center_suit_score": cen.get("score"),
    })
    # production gate evaluation (white gate short-circuits first, as prod does)
    rank_ok = rs is not None and rs >= rec.min_score and rm >= rec.min_margin
    corner_ok = cs is not None and cs >= rec.min_score and cm >= rec.min_margin
    center_ok = cen.get("suit") is not None
    if white < EMPTY_SLOT_MIN_WHITE:
        pass                      # keep empty_slot_white_gate (prod returns here)
    elif not rank_ok:
        d["reason"] = "rank_gate"
    elif corner_ok and center_ok and cn != cen.get("suit"):
        d["reason"] = "suit_conflict"
    elif center_ok:
        d["reason"] = "OK_center" if not corner_ok else "OK"
    elif corner_ok:
        d["reason"] = "corner_only_blocked"
    else:
        d["reason"] = "suit_gate_both_fail"
    return d


def analyze(files: list[str], label: str, out_name: str,
            max_frames: int | None = None) -> list[dict]:
    idx_path = REPO / "docs" / "vision_v1" / "frame_index_full.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    geom = {}
    for run in idx["runs"]:
        for r in run["rows"]:
            geom[r["file"]] = (r["w"], r["h"])
    recs_cache: dict[str, CoinPokerBoardRecognizer] = {}
    rows_out = []
    t0 = time.time()
    use = files[:max_frames] if max_frames else files
    for i, f in enumerate(use):
        w, h = geom.get(f, (None, None))
        if not w:
            continue
        cls = window_for_frame(w, h)
        if cls not in ("1419", "1876"):
            continue
        if cls not in recs_cache:
            recs_cache[cls] = CoinPokerBoardRecognizer(window=cls)
        rec = recs_cache[cls]
        W, H, rgba = decode_png((EV / f).read_bytes())
        frame = np.frombuffer(rgba, dtype=np.uint8).reshape(H, W, 4)
        slots = [slot_diag(rec, frame, s, W) for s in range(5)]
        rows_out.append({"file": f, "window": cls, "slots": slots})
        if (i + 1) % 50 == 0:
            print(f"{label}: {i + 1}/{len(use)} "
                  f"({round(time.time() - t0, 1)}s)", flush=True)
    outp = OUT / out_name
    outp.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in rows_out) + "\n", encoding="utf-8")
    print(f"{label}: wrote {len(rows_out)} frames -> {outp} "
          f"({round(time.time() - t0, 1)}s)", flush=True)
    return rows_out


def summarize_settled(rows: list[dict]) -> dict:
    """Miss-cause distribution over non-OK slots in SETTLED miss frames."""
    cause = Counter()
    per_frame_miss_reason = Counter()
    white_hist = {"pass_gate": [], "fail_gate": []}
    rank_gate_detail = Counter()
    for r in rows:
        reasons = [s.get("reason") for s in r["slots"]]
        non_ok = [x for x in reasons if x not in ("OK", "OK_center")]
        for x in non_ok:
            cause[x] += 1
        if non_ok:
            per_frame_miss_reason["|".join(sorted(set(non_ok)))] += 1
        for s in r["slots"]:
            if s.get("white_frac") is None:
                continue
            key = "fail_gate" if s["white_frac"] < EMPTY_SLOT_MIN_WHITE \
                else "pass_gate"
            if len(white_hist[key]) < 4000:
                white_hist[key].append(s["white_frac"])
            if s.get("reason") == "rank_gate":
                sc = s.get("rank_score") or 0
                rank_gate_detail["score_lt_0.5" if sc < 0.5 else
                                 "score_0.5_0.7" if sc < 0.7 else
                                 "margin_lt_0.08"] += 1
    # family-filter mismatch: filtered rank fails but unfiltered passes strongly
    fam_hits = []
    for r in rows:
        for s in r["slots"]:
            if s.get("reason") == "rank_gate" and s.get("rank_all_score") \
                    and s["rank_all_score"] >= 0.7 and \
                    (s.get("rank_all_margin") or 0) >= 0.08:
                fam_hits.append({"file": r["file"], "slot": s["slot"],
                                 "filtered": s.get("rank"),
                                 "unfiltered": s.get("rank_all"),
                                 "score": s["rank_all_score"]})
    return {"slot_cause_dist": dict(cause.most_common()),
            "frame_reason_combo_top": dict(
                per_frame_miss_reason.most_common(12)),
            "rank_gate_detail": dict(rank_gate_detail),
            "family_filter_suspects": fam_hits[:40],
            "white_frac_pass_mean": round(float(np.mean(
                white_hist["pass_gate"])), 4) if white_hist["pass_gate"] else None,
            "white_frac_fail_mean": round(float(np.mean(
                white_hist["fail_gate"])), 4) if white_hist["fail_gate"] else None}


def summarize_ghost(rows: list[dict]) -> dict:
    """Slot-level feature distributions for ghost frames."""
    agg = defaultdict(lambda: {"n": 0, "white": [], "rank_sc": [],
                               "suit_sc": []})
    for r in rows:
        for s in r["slots"]:
            if s.get("white_frac") is None:
                continue
            k = "card_like(white>=0.12)" \
                if s["white_frac"] >= EMPTY_SLOT_MIN_WHITE \
                else "dark(white<0.12)"
            a = agg[k]
            a["n"] += 1
            a["white"].append(s["white_frac"])
            if s.get("rank_score") is not None:
                a["rank_sc"].append(s["rank_score"])
            if s.get("center_suit_score") is not None:
                a["suit_sc"].append(s["center_suit_score"])
    out = {}
    for k, a in agg.items():
        out[k] = {"n": a["n"],
                  "white_p50": round(float(np.percentile(a["white"], 50)), 4),
                  "white_p95": round(float(np.percentile(a["white"], 95)), 4),
                  "rank_sc_p50": round(float(np.percentile(a["rank_sc"], 50)), 4)
                  if a["rank_sc"] else None,
                  "rank_sc_p95": round(float(np.percentile(a["rank_sc"], 95)), 4)
                  if a["rank_sc"] else None,
                  "rank_sc_max": round(float(np.max(a["rank_sc"])), 4)
                  if a["rank_sc"] else None}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets = json.loads((OUT / "v2_diag_sets.json").read_text(
        encoding="utf-8"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("settled", "all"):
        rows = analyze(sets["settled_miss"], "settled-miss",
                       "diag_settled_miss.jsonl")
        summ = summarize_settled(rows)
        (OUT / "diag_settled_summary.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(summ, ensure_ascii=False, indent=1)[:3000])
    if mode in ("ghost", "all"):
        # ghost sample + true-preflop 0-card negative sample
        rows_stream = []
        for line in (REPO / "docs" / "vision_v1" /
                     "street_stream_frames.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                rows_stream.append(json.loads(line))
        ghosts = [r["file"] for r in rows_stream
                  if r["machine_state"] == "PREFLOP"
                  and r["observed_cardinality"] in (1, 2)]
        true0 = [r["file"] for r in rows_stream
                 if r["machine_state"] == "PREFLOP"
                 and r["observed_cardinality"] == 0]
        # deterministic spread sampling (every k-th) across whole session
        gsample = ghosts[::len(ghosts) // 120 + 1][:120]
        zsample = true0[::len(true0) // 120 + 1][:120]
        rows_g = analyze(gsample, "ghost", "diag_ghost.jsonl")
        rows_z = analyze(zsample, "true0", "diag_true0.jsonl")
        (OUT / "diag_ghost_summary.json").write_text(json.dumps(
            {"ghost": summarize_ghost(rows_g),
             "true0": summarize_ghost(rows_z),
             "ghost_files": gsample, "true0_files": zsample},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("ghost summary:",
              json.dumps(summarize_ghost(rows_g), indent=1)[:1500])
        print("true0 summary:",
              json.dumps(summarize_ghost(rows_z), indent=1)[:1500])


if __name__ == "__main__":
    main()
