"""Phase V2K — 组装 + 完整回测（10 帧真实截图）。

逐帧：逐座 stack/bet（金色路径+30% 边距）、board（真实模板）、
button（D 检测）、hero（严格 UNCERTAIN）→ RawVisionState → assemble →
NormalizedVisionState + provenance + 统计。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplateLibrary)
from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    RealClientLayoutProfile)
from poker_decision_ai.vision2.pixel.realclient.state import (  # noqa: E402
    assemble_visual_state)
from poker_decision_ai.vision2.realtime.image import (  # noqa: E402
    ImageData, to_gray)
from poker_decision_ai.vision2.realtime.coords import Rect  # noqa: E402
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def main() -> int:
    sys.argv = sys.argv[:1]
    manifest = Path("data/vision2/realclient_assets_v6/manifest.json")
    mdata = json.loads(manifest.read_text(encoding="utf-8"))
    refs = {a["template_id"]: a.get("png_ref") for a in mdata["assets"]}

    def loader(tid):
        r = refs.get(tid)
        if not r or not Path(r).exists():
            return None
        w_, h_, rgba_ = decode_png(Path(r).read_bytes())
        return ImageData.from_rgba_bytes(w_, h_, rgba_)

    lib = RealClientTemplateLibrary.from_json(str(manifest), loader)
    digit = RealClientDigitOCRBackend(lib)
    card = RealClientCardRecognitionBackend(lib)
    prof = RealClientLayoutProfile.measured_v2()
    stacks = [prof.entries[f"player_{i}_stack_roi"].bbox for i in range(6)]
    bets = [prof.entries[f"player_{i}_bet_roi"].bbox for i in range(6)]

    files = sorted(Path("picture data").glob("*.png"))
    per_frame = []
    stats = {"street": 0, "board_ok": 0, "button_present": 0,
             "stack_ok": 0, "stack_uncertain": 0, "bet_ok": 0,
             "validator_rejections": 0, "conflicts": 0}
    for fi, p in enumerate(files):
        w, h, rgba = decode_png(p.read_bytes())
        img = ImageData.from_rgba_bytes(w, h, rgba)
        # 逐座 stack/bet（标准 30% 边距）
        st, bt = {}, {}
        for i in range(6):
            for dst, rect in ((st, stacks[i]), (bt, bets[i])):
                if rect is None:
                    continue
                px, py = rect.width * 0.3, rect.height * 0.3
                exp = Rect(max(0, rect.x - px), max(0, rect.y - py),
                           min(w, rect.x + rect.width + px) -
                           max(0, rect.x - px),
                           min(h, rect.y + rect.height + py) -
                           max(0, rect.y - py))
                sub = img.crop(exp)
                res = digit.recognize(sub, "stacks", "number")
                if res.status.value == "OK":
                    dst[i] = res
                    if dst is st:
                        stats["stack_ok"] += 1
                    else:
                        stats["bet_ok"] += 1
        # board（复用实测可用的默认 board ROI——V2F 已验证）
        from poker_decision_ai.vision2.realtime.anchors import (
            bootstrap_profile)
        from poker_decision_ai.vision2.realtime.roi import (
            ROIManager, default_roi_specs)
        from poker_decision_ai.vision2.realtime.coords import GeometryContext
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        profb = bootstrap_profile("real", geo,
                                  {"table": (0.0, 0.0, 1.0, 1.0)})
        mgrb = ROIManager(profb, default_roi_specs())
        roisb, _ = mgrb.resolve()
        cropsb, _ = mgrb.extract_all(img, roisb)
        board_val = None
        bcrop = cropsb.get("board")
        bres = card.recognize(bcrop, "board")
        if bres.status.value == "OK":
            board_val = [f"{c.rank}{c.suit}" for c in bres.cards
                         if c.rank and c.suit]
            stats["board_ok"] += 1
        # button：紧凑红六边形（面积/圆度过滤，避免红牌面/红心误报）
        btn_present = False
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        r_, g_, b_ = arr[..., 0].astype(int), arr[..., 1].astype(int), \
            arr[..., 2].astype(int)
        red = (r_ > 150) & (g_ < 90) & (b_ < 90)
        ys, xs = np.where(red)
        if len(xs) > 0:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            area = len(xs)
            circ = area / (np.pi * (max(bw, bh) / 2) ** 2 + 1e-9)
            if 500 <= area <= 4000 and 0.55 <= circ <= 1.0:
                btn_present = True
        if btn_present:
            stats["button_present"] += 1
        # 组装
        state = assemble_visual_state(
            fi + 1, f"frame_{fi+1}", stacks=st, bets=bt,
            folded={2: None}, all_in={}, board_cards=board_val,
            button_present=btn_present, button_bbox=None, hero_cards=None)
        stats["street"] += 1 if state["normalized"].game.get("street") else 0
        if not state["validation"]["valid"]:
            stats["validator_rejections"] += 1
        per_frame.append({
            "file": p.name, "frame_id": fi + 1,
            "street": state["normalized"].game.get("street"),
            "board": board_val,
            "hero_cards": None,
            "pot": None,
            "button_present": btn_present,
            "players": [{"seat": i, "stack": (st[i].value if i in st
                                              else None),
                         "raw_text": (st[i].raw_text if i in st else None),
                         "unit": ((st[i].extra or {}).get("unit")
                                  if i in st else None),
                         "bet": (bt[i].value if i in bt else None)}
                        for i in range(6)],
            "validation": state["validation"],
            "provenance": state["provenance"]})
    out = Path("data/vision2/pixel_v2k_state.json")
    out.write_text(json.dumps({"stats": stats, "frames": per_frame},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== V2K 组装回测 ===")
    print("stats:", stats)
    for f in per_frame:
        stv = [f"{p['seat']}:{p['raw_text']}" for p in f["players"]
               if p["stack"] is not None]
        print(f"  {f['file'][:9]} street={f['street']} board={f['board']} "
              f"D={f['button_present']} stacks={stv} valid={f['validation']['valid']}")
    print("out:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
