"""Phase V2M — 10 帧 Replay + Runtime 基准（真实 1848×1388 截图）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    RealClientLayoutProfile)
from poker_decision_ai.vision2.pixel.realclient.runtime import (  # noqa: E402
    FrameStream, RealtimePokerVisionState, TemporalVisionState,
    VisualEventBus, board_transition, roi_digest)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import (  # noqa: E402
    GeometryContext, Rect)
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _load_library():
    manifest = Path("data/vision2/realclient_assets_v6/manifest.json")
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        r = refs.get(tid)
        if not r or not Path(r).exists():
            return None
        w_, h_, rgba_ = decode_png(Path(r).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)
    return RealClientTemplateLibrary.from_json(str(manifest), loader)


def main() -> int:
    sys.argv = sys.argv[:1]
    lib = _load_library()
    digit = RealClientDigitOCRBackend(lib)
    card = RealClientCardRecognitionBackend(lib)
    prof = RealClientLayoutProfile.measured_v2()
    stack_rois = [prof.entries[f"player_{i}_stack_roi"].bbox
                  for i in range(6)]
    files = sorted(Path("picture data").glob("*.png"))

    stream = FrameStream(fps=2.0, stale_gap_s=5.0)
    tvs = TemporalVisionState()
    bus = VisualEventBus()
    per_frame = []
    timings = {"capture": [], "roi": [], "digit": [], "board": [],
               "temporal": [], "total": []}
    prev_board_count = None

    for fi, p in enumerate(files):
        t0 = time.perf_counter()
        # capture（读文件模拟）
        tc = time.perf_counter()
        w, h, rgba = decode_png(p.read_bytes())
        img = ImageData.from_rgba_bytes(w, h, rgba)
        timings["capture"].append((time.perf_counter() - tc) * 1000)
        acc = stream.accept(fi + 1, float(fi), img)
        if not acc["accepted"]:
            per_frame.append({"frame_id": fi + 1, "skipped": acc["reason"]})
            continue
        if acc.get("stale"):
            bus.emit("FRAME_STALE", fi + 1, {"gap_s": acc.get("gap_s")})
        # ROI + 逐座 stack（30% 边距）
        tr = time.perf_counter()
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        profb = bootstrap_profile("real", geo,
                                  {"table": (0.0, 0.0, 1.0, 1.0)})
        mgrb = ROIManager(profb, default_roi_specs())
        roisb, _ = mgrb.resolve()
        cropsb, _ = mgrb.extract_all(img, roisb)
        timings["roi"].append((time.perf_counter() - tr) * 1000)
        td = time.perf_counter()
        stacks = {}
        changed = []
        for i, rect in enumerate(stack_rois):
            if rect is None:
                continue
            px, py = rect.width * 0.3, rect.height * 0.3
            exp = Rect(max(0, rect.x - px), max(0, rect.y - py),
                       min(w, rect.x + rect.width + px) - max(0, rect.x - px),
                       min(h, rect.y + rect.height + py) - max(0, rect.y - py))
            sub = img.crop(exp)
            rd = roi_digest(sub)
            res = digit.recognize(sub, "stacks", "number")
            name = f"player_{i}.stack"
            if res.status.value == "OK":
                stacks[i] = res
                tvs.observe(name, res.value, res.confidence, fi + 1, rd)
                changed.append(name)
            else:
                tvs.uncertain(name, fi + 1)
        timings["digit"].append((time.perf_counter() - td) * 1000)
        # board（默认 board ROI + 牌数转换检查）
        tb = time.perf_counter()
        bres = card.recognize(cropsb.get("board"), "board")
        board_vals = []
        if bres.status.value == "OK":
            board_vals = [f"{c.rank}{c.suit}" for c in bres.cards
                          if c.rank and c.suit]
        trans = board_transition(prev_board_count, len(board_vals))
        if trans["conflict"]:
            bus.emit("STATE_CONFLICT", fi + 1, trans)
        else:
            if len(board_vals) > (prev_board_count or 0):
                bus.emit("CARD_APPEARED", fi + 1,
                         {"count": len(board_vals)})
            prev_board_count = len(board_vals)
        timings["board"].append((time.perf_counter() - tb) * 1000)
        # temporal / 快照
        tt = time.perf_counter()
        tvs.stale_check(fi + 1)
        snapshot = RealtimePokerVisionState(
            timestamp=float(fi), frame_id=fi + 1,
            state_version=tvs.state_version,
            vision_status="PARTIAL_VALID_STATE",
            table={"seats": 6},
            street={"value": {0: "preflop", 3: "flop", 4: "turn",
                              5: "river"}.get(len(board_vals))},
            board={"value": board_vals or None},
            hero_cards={"value": None},
            pot={"value": None},
            players=[{
                "seat": i,
                "stack": {"value": (stacks[i].value if i in stacks
                                    else None),
                          "raw_text": (stacks[i].raw_text
                                       if i in stacks else None),
                          "confidence": (stacks[i].confidence
                                         if i in stacks else None),
                          "source": "observed",
                          "evidence_frame_ids": [fi + 1]
                          if i in stacks else [],
                          "temporal_status": tvs.fields.get(
                              f"player_{i}.stack").temporal_status
                          if f"player_{i}.stack" in tvs.fields else
                          "UNCERTAIN"}} for i in range(6)])
        timings["temporal"].append((time.perf_counter() - tt) * 1000)
        timings["total"].append((time.perf_counter() - t0) * 1000)
        per_frame.append({"frame_id": fi + 1, "snapshot": snapshot.to_dict(),
                          "changed_fields": changed,
                          "events": [e.to_dict() for e in bus.drain(fi)]})

    def pct(xs):
        xs = sorted(xs)
        n = len(xs)
        if not xs:
            return None
        return {"p50": round(xs[n // 2], 1),
                "p95": round(xs[int(n * 0.95)] if n > 1 else xs[-1], 1),
                "p99": round(xs[int(n * 0.99)] if n > 1 else xs[-1], 1)}

    bench = {k: pct(v) for k, v in timings.items()}
    out = Path("data/vision2/pixel_v2m_replay.json")
    out.write_text(json.dumps({"bench_ms": bench, "frames": per_frame},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== V2M Replay ===")
    print("bench(ms):", bench)
    for f in per_frame:
        if "skipped" in f:
            print(f"  f{f['frame_id']} SKIPPED {f['skipped']}")
            continue
        stv = [(p["seat"], p["stack"]["raw_text"]) for p in
               f["snapshot"]["players"] if p["stack"]["value"] is not None]
        evs = [e["kind"] for e in f["events"]]
        print(f"  f{f['frame_id']} board={f['snapshot']['board']['value']} "
              f"stacks={stv} changed={f['changed_fields']} events={evs}")
    print("out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
