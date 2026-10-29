"""V3F-P30-2B — 1876 Betting Header Recognizer（SB / BB / ante）。

严格基于 P30-2A 冻结视觉事实：
- Header ROI (9,7,176,14)（4 帧实测 bbox 完全一致）；stake 子区 x∈[95,200],
  y∈[0,26]（「NLH <hand_id> - 1K/2K (320)」之后的金额部分）；
- 文本格式（P30-2A 目视确认）：`1K/2K (320)`；K=千后缀，括号=ante；
- 语料实测：stake 子区跨帧 NCC 0.968-1.0（像素级一致——同一 stake 会话）；
- SB/BB 蓝徽章 = **角色**，绝不作为金额来源（空间隔离：徽章在座锚区）。

检测策略（UNKNOWN-first）：
1. header 存在性：ROI 亮文本量（luma>120 近中性）≥ 阈值 → 无 → UNKNOWN；
2. stake 子区掩码 → 整区模板 NCC vs GT 参考（1K/2K(320) 像素位图）；
   - NCC ≥ 0.90 → 结构固定已知 → {SB:1K, BB:2K, ante:320}；
   - NCC < 0.90 → **锚定滑动解析**（R13-EXE-2 新增，见下）；
3. （R13-EXE-2）锚定滑动解析：客户端 header 字体渲染存在两档字重/间距
   （1876 GT 模板源帧 vs coinpoker_v3 语料——整词 NCC 0.24，且 v3 布局
   stake 文本右移至 ABS x∈[139,216)，原 STAKE_ROI x1=200 截掉 `0)`）。
   锚定模式在扩展 ROI（x∈[90,224)）上运行：
   a. 高置信锚字符（slash/K/paren_open/paren_close）在 mask 全宽滑动
      NCC 定位；K 取 slash 左/右窗口（±20px 漂移容忍）内各自最优峰；
   b. 按锚点相对偏移界定 digit 区间（SB=K1 左邻接段、BB=slash..K2、
      ante=双括号之间），列分割后逐段与该套模板 NCC 匹配（≥0.70）；
   c. 粗/细两套模板（v1876 / v3b）分别打分，取高分者
      （provenance.font_variant 记录）；0.90 整词门不动；锚定失败 →
      落回 legacy char parse → UNKNOWN（诚实降级零变化）。
3b. （V-4 Round15）微级别 ₮ 计量格式锚定解析：header `₮0.05/₮0.1`
   （₮=Tether 货币符号、小数点、尾零省略、无 K/ante 括号）。模板集
   v3c_micro（tether/dot/slash/digit{0,1,5,6,7,8}）；锚 = slash +
   tether（左右各一），SB/BB token 区间列分割逐段匹配（dot 为几何
   token：h≤3 & w≤5 & 带内下半部——点模板无方差 NCC 不可用）；
   `_to_num` 裸小数 → 货币单位 float（'0.05'→0.05，与 bet/pot 观测
   同单位）；sb>=bb 结构非法 → 拒绝。与 K 格式路径按分取高
   （reason=header_stake_anchored_parse_micro, currency=USDT）。
   NOT_COVERED → UNKNOWN：₮ 格式带 ante、digit{2,3,4,9}（微语料
   未出现）、小数 K/M、其它货币符号（$ 等）。
4. K 后缀解析：`XK` → X*1000（X=整数 digit{0-9} 模板覆盖——未覆盖 → UNKNOWN）；
   逗号格式 `169,760` → `169760`（几何 token）；M 格式 `1.28M` → `1280000`
   （仅当字符解析路径识别出 M——header 语料未出现 M → NOT_COVERED 记录）；
5. 未覆盖 digit {…} → UNKNOWN；任何歧义 → UNKNOWN。

红线：无 OCR / 无视觉 LLM runtime / 无 poker-rule 推导 / 无 previous-cache /
无跨字段（stack/bet/pot）推导；SB/BB 蓝徽章非金额。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Header ROI（P30-2A 实测 4 帧一致）
HEADER_ROI = (9, 7, 176, 14)
#: stake 子区（header 内右侧金额文本）
STAKE_ROI = (90, 0, 200, 26)
#: 锚定模式扩展 stake ROI（R13-EXE-2：v3 布局 stake 文本止于 ABS x≈215，
#: 原 x1=200 截掉 `0)`；右侧至 224 为空背景，扩展无害）
STAKE_ROI_EXT = (90, 0, 224, 26)
#: header 亮文本存在阈值
HEADER_TEXT_MIN = 100
#: 整区模板 NCC 门（实测 0.968-1.0；0.90 为分布下沿，非调低）
STAKE_NCC_MIN = 0.90
#: 字符模板门（全链冻结）
CHAR_SCORE_MIN = 0.70
CHAR_MARGIN_MIN = 0.08
#: 模板目录（粗体/1876 GT 源——零改动）
HEADER_ASSET_DIR = "data/vision2/coinpoker_assets_v1876betting_header"
#: 第二模板目录（R13-EXE-2：coinpoker_v3 渲染字重，取证裁制）
HEADER_ASSET_DIR_V3B = "data/vision2/coinpoker_assets_v3b_header"
#: 第三模板目录（V-4 Round15：微级别 ₮ 计量 header，`₮0.05/₮0.1` 格式；
#: 446 帧固定窗 NCC≥0.98，字形四实例跨检查位图一致，proof_sheet 目视验证）
HEADER_ASSET_DIR_V3C_MICRO = "data/vision2/coinpoker_assets_v3c_micro"
#: 锚定相对偏移窗口（对 slash/K 锚的 ±20px 漂移容忍；取证实测
#: K1≈slash-7、K2≈slash+15、paren_open≈K2+14、paren_close≈po+27）
ANCHOR_K1_WIN = (-27, 0)      # K1 起点 ∈ [slash_x-27, slash_x)
ANCHOR_K2_WIN = (2, 35)       # K2 起点 ∈ [slash_x+2, slash_x+35)
ANCHOR_PO_WIN = (4, 34)       # po 起点 ∈ [K2_x+4, K2_x+34)
ANCHOR_PC_WIN = (8, 48)       # pc 起点 ∈ [po_x+8, po_x+48)
#: SB 数字与 K 锚的最大间隔（px；dash/hand-id 尾部被此门挡住）
SB_GAP_MAX = 4
SB_WALK_MAX = 26              # SB 数字向左最多回看窗口
#: V-4 微级别（₮ 计量）几何点规则：小数点 run 紧裁高 ≤3 且宽 ≤5，
#: 且 run 底行位于带内下半部（dot 底 y=17；dash "-" 底 y=13 被排除）
MICRO_DOT_MAX_H = 3
MICRO_DOT_MAX_W = 5
MICRO_DOT_BOTTOM_MIN = 14     # 带高 26 的下半界（row index）

_GT_TEXT = "1K/2K (320)"
_GT_VALUES = {"sb": 1000, "bb": 2000, "ante": 320}


def _luma(frame: np.ndarray) -> np.ndarray:
    return (frame[..., 0].astype(np.int16) * 0.299
            + frame[..., 1].astype(np.int16) * 0.587
            + frame[..., 2].astype(np.int16) * 0.114)


def _text_mask(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """亮文本掩码（header 白字；与 ref mask 同法 luma>120——对齐 NCC）。"""
    x0, y0, x1, y1 = roi
    if x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    sub = frame[y0:y1, x0:x1]
    l = _luma(sub)
    return (l > 120).astype(np.float64)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a - a.mean(), b - b.mean()
    an = float(np.sqrt((a0 * a0).sum()))
    bn = float(np.sqrt((b0 * b0).sum()))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float((a0 * b0).sum() / (an * bn))


def _load_ref_mask(asset_dir: str = HEADER_ASSET_DIR) -> np.ndarray | None:
    """载入 GT 参考 stake 掩码（f01 提取，1K/2K(320)）。"""
    p = Path(asset_dir) / "stake_mask_f01.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float64)


def _to_num(text: str) -> int | float | None:
    """金额 token 解析：K 后缀 / 逗号 / 纯整数 / M 后缀 / 裸小数（V-4）。

    只接受 GT 覆盖格式；未覆盖 → None（UNKNOWN）。
    V-4 单位语义选择（写明）：裸小数按**货币单位浮点**返回（'0.05'→0.05），
    非「美分整数换算」——理由：桌面 bet capsule / pot / to_call 观测文本同
    为货币单位（'0.05'），production_bridge._strict_num 统一 float 化；
    header 同单位可直取消费，美分换算会使 sb=5 与 bet=0.05 差 100 倍。
    既有整数格式仍返回 int（零回退）；小数 K（'1.28K'）仍不覆盖（None）。
    """
    t = text.strip()
    if not t:
        return None
    # K 后缀（1K → 1000；1.28K? 语料未见小数 K → 不支持）
    if t.endswith("K") or t.endswith("k"):
        num = t[:-1]
        if num.isdigit() and num:
            return int(num) * 1000
        return None
    # M 后缀（1.28M → 1280000；语料 header 未出现 → NOT_COVERED 仍解析但
    # 仅字符路径可触发）
    if t.endswith("M") or t.endswith("m"):
        num = t[:-1]
        try:
            val = float(num)
            return int(round(val * 1_000_000))
        except ValueError:
            return None
    # 裸小数（V-4 微级别 ₮ 计量：'0.05' → 0.05 float；'0.1' → 0.1）。
    # 仅接受 `<int>.<int>` 单点格式；尾零省略（'0.1' 非 '0.10'）天然支持。
    if "." in t:
        if t.count(".") != 1:
            return None
        ip, _, fp = t.partition(".")
        if not fp or (ip and not ip.isdigit()):
            return None
        return float(t)
    # 逗号（169,760 → 169760）
    if "," in t:
        try:
            return int(t.replace(",", ""))
        except ValueError:
            return None
    # 纯整数
    if t.isdigit() and t:
        return int(t)
    return None


class BettingHeaderRecognizer1876:
    """1876 header → {sb, bb, ante, status, reason, confidence,
    font_variant}。"""

    name = "coinpoker_betting_header_v3f_p30_2b_1876"

    def __init__(self, asset_dir: str = HEADER_ASSET_DIR,
                 variant_dirs: dict | None = None):
        # 变体模板集并存（R13-EXE-2 + V-4）：v1876（GT 源，零改动）+
        # v3b（coinpoker_v3 K 格式渲染）+ v3c_micro（₮ 微级别格式）。
        # 识别时分别打分取高分（font_variant）。
        self.ref_mask = _load_ref_mask(asset_dir)
        self.assets = self._load_assets(asset_dir)
        self._variants: dict[str, dict] = {
            "v1876": {"assets": dict(self.assets)}}
        if variant_dirs is None:
            variant_dirs = {"v3b": HEADER_ASSET_DIR_V3B,
                            "v3c_micro": HEADER_ASSET_DIR_V3C_MICRO}
        for vname, vdir in variant_dirs.items():
            vassets = self._load_assets(vdir)
            if vassets:
                self._variants[vname] = {"assets": vassets}

    @staticmethod
    def _load_assets(asset_dir: str) -> dict:
        mp = Path(asset_dir) / "manifest.json"
        if not mp.exists():
            return {}
        m = json.loads(mp.read_text(encoding="utf-8"))
        from poker_decision_ai.vision2_platform.png_decode import decode_png
        prep = {}
        for a in m.get("assets", []):
            # V-5 kind 隔离：v3c_micro 为混载目录（header + stack/pot/bet
            # 微字形共存）。V-5 digit 资产是 RGB 紧裁（alpha 全 1）——进入
            # header 模板集会退化成全 1 位图毒化 legacy 字符路径，必须排除；
            # header 自身资产（kind=betting_header_v3c_micro / 无 kind 的
            # 既有目录）不受影响。
            if a.get("kind") in ("stack_glyph_micro", "pot_digit_micro",
                                 "bet_digit_micro"):
                continue
            p = Path(asset_dir) / Path(a["crop_ref"]).name
            if not p.exists():
                continue
            w, h, rgba = decode_png(p.read_bytes())
            arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
            g = (arr[..., 3] > 128).astype(np.float64)
            ys, xs = np.where(g > 0)
            if len(xs) < 3:
                continue
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            prep[a["label"]] = g          # 同名 label 后者覆盖（last-wins）
        return prep

    def _char_match(self, seg: np.ndarray) -> tuple:
        """字符段 → 模板 NCC → (label, score)；0/兜底 → (None, 0)。"""
        best, bsc = None, 0.0
        for lab, tpl in self.assets.items():
            g0 = (seg > 0).astype(np.float64)
            # 对齐：缩放到模板尺寸（小图缩放匹配）
            from numpy import interp
            ss = np.array(seg.shape, dtype=float)
            ts = np.array(tpl.shape, dtype=float)
            fy, fx = ss / ts
            yi = np.clip((np.arange(tpl.shape[0]) * fy).astype(int), 0,
                         seg.shape[0] - 1)
            xi = np.clip((np.arange(tpl.shape[1]) * fx).astype(int), 0,
                         seg.shape[1] - 1)
            samp = g0[np.ix_(yi, xi)]
            sc = _ncc(samp, tpl)
            if sc > bsc:
                best, bsc = lab, sc
        return best, bsc

    def recognize(self, frame: np.ndarray) -> dict:
        # 1. header 存在
        hm = _text_mask(frame, HEADER_ROI)
        if hm is None:
            return {"sb": None, "bb": None, "ante": None, "status":
                    "UNKNOWN", "reason": "roi_out_of_frame",
                    "confidence": 0.0}
        if int(hm.sum()) < HEADER_TEXT_MIN:
            return {"sb": None, "bb": None, "ante": None, "status":
                    "UNKNOWN", "reason": "no_header_text",
                    "confidence": 0.0}
        # 2. stake 子区
        sm = _text_mask(frame, STAKE_ROI)
        if sm is None or sm.sum() < 20:
            return {"sb": None, "bb": None, "ante": None, "status":
                    "UNKNOWN", "reason": "no_stake_text",
                    "confidence": 0.0}
        # 整区模板 NCC → 已知固定格式（0.90 门不动；R13-EXE-2 保持）
        if self.ref_mask is not None:
            sc = _ncc(sm, self.ref_mask)
            if sc >= STAKE_NCC_MIN:
                return {"sb": _GT_VALUES["sb"], "bb": _GT_VALUES["bb"],
                        "ante": _GT_VALUES["ante"], "status": "RECOGNIZED",
                        "reason": "header_stake_template_match",
                        "confidence": round(sc, 4),
                        "font_variant": "v1876_whole"}
        # 3. 锚定滑动解析（R13-EXE-2 + V-4：双格式路径按变体分别打分取
        #    高分；容忍 ±20px 漂移与字重差；任一变体/格式失败不阻断其它）
        ext = _text_mask(frame, STAKE_ROI_EXT)
        if ext is not None and ext.sum() >= 20:
            best = None
            for vname, v in self._variants.items():
                parsed = self._anchored_parse(ext, v["assets"])
                if parsed is not None:
                    sb, bb, ante, conf = parsed
                    if best is None or conf > best[3]:
                        best = (sb, bb, ante, conf, vname,
                                "header_stake_anchored_parse")
                parsed_m = self._anchored_parse_micro(ext, v["assets"])
                if parsed_m is not None:
                    sb, bb, conf = parsed_m
                    if best is None or conf > best[3]:
                        best = (sb, bb, None, conf, vname,
                                "header_stake_anchored_parse_micro")
            if best is not None:
                sb, bb, ante, conf, vname, reason = best
                out = {"sb": sb, "bb": bb, "ante": ante,
                       "status": "RECOGNIZED", "reason": reason,
                       "confidence": round(conf, 4),
                       "font_variant": vname}
                if reason == "header_stake_anchored_parse_micro":
                    out["currency"] = "USDT"   # ₮ 计量（V-4 微级别）
                return out
        # 4. legacy 字符解析路径（未匹配模板 → 尽力解析；冲突/失败 → UNKNOWN）
        parsed = self._parse_chars(sm)
        if parsed is None:
            return {"sb": None, "bb": None, "ante": None, "status":
                    "UNKNOWN", "reason": "char_parse_failed",
                    "confidence": 0.0}
        sb, bb, ante = parsed
        return {"sb": sb, "bb": bb, "ante": ante, "status": "RECOGNIZED",
                "reason": "header_stake_char_parse",
                "confidence": 0.8, "font_variant": "v1876_char"}

    def _parse_chars(self, mask: np.ndarray) -> tuple | None:
        """字符解析：列分割 → 标签序列 → 结构解析（K/斜杠/括号）。

        仅处理 GT 结构 `XK/YK (Z)`：digit+K + slash + digit+K +
        paren_open + digit.. + paren_close。
        """
        cold = (mask > 0).sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cold):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                runs.append(run); run = None
        if run:
            runs.append(run)
        merged = []
        for rr in runs:
            if merged and rr[0] - merged[-1][1] <= 1:
                merged[-1][1] = rr[1]
            else:
                merged.append(rr)
        labels = []
        for a, b in merged:
            seg = mask[:, a:b + 1]
            lab, sc = self._char_match(seg)
            if lab is None or sc < CHAR_SCORE_MIN:
                return None
            labels.append((lab, sc))
        # 结构校验：必须含 K 两个、斜杠、括号
        text = "".join(l for l, _ in labels)
        # 简化结构：允许模板标签带数字后缀（K1/K2, digit_2b）
        norm = [l.split("_")[0] if l.startswith("digit") else l
                for l, _ in labels]
        if len(norm) < 7:
            return None
        # 找 "K ... / ... K ... (" 模式
        try:
            k1 = norm.index("K")
            slash = norm.index("slash", k1 + 1)
            k2 = norm.index("K", slash + 1)
            po = norm.index("paren_open", k2 + 1)
            pc = norm.index("paren_close", po + 1)
        except ValueError:
            return None
        sb_digit = "".join(d for d in norm[:k1] if d.startswith("digit"))
        bb_digit = "".join(d for d in norm[slash + 1:k2]
                           if d.startswith("digit"))
        ante_digits = "".join(d.replace("digit_", "")
                              for d in norm[po + 1:pc]
                              if d.startswith("digit"))
        sb = _to_num(sb_digit + "K") if sb_digit else None
        bb = _to_num(bb_digit + "K") if bb_digit else None
        ante = _to_num(ante_digits) if ante_digits else None
        if sb is None or bb is None or ante is None:
            return None
        return sb, bb, ante

    # ── R13-EXE-2：锚定滑动解析（两档字重模板并存，按分自动选择）──────
    @staticmethod
    def _column_runs(mask: np.ndarray) -> list:
        """列分割（gap≤1 合并——与 legacy _parse_chars 同法）。"""
        cold = (mask > 0).sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cold):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                runs.append(run); run = None
        if run:
            runs.append(run)
        merged = []
        for rr in runs:
            if merged and rr[0] - merged[-1][1] <= 1:
                merged[-1][1] = rr[1]
            else:
                merged.append(rr)
        return merged

    def _slide(self, mask: np.ndarray, tpl: np.ndarray,
               x_win: tuple | None = None) -> tuple:
        """模板在 mask 全宽（或 x_win 窗口）滑动 NCC → (score, x, y)。

        y 全带搜索（容忍纵向漂移）；窗口 = 模板紧裁尺寸。
        """
        th, tw = tpl.shape
        H, W = mask.shape
        if th > H or tw > W:
            return 0.0, -1, -1
        x_lo, x_hi = (0, W - tw) if x_win is None else (
            max(0, x_win[0]), min(W - tw, x_win[1]))
        best = (0.0, -1, -1)
        for y in range(0, H - th + 1):
            for x in range(x_lo, x_hi + 1):
                win = mask[y:y + th, x:x + tw]
                sc = _ncc(win, tpl)
                if sc > best[0]:
                    best = (sc, x, y)
        return best

    def _match_digit_seg(self, seg: np.ndarray,
                         assets: dict) -> tuple:
        """段 → digit 模板匹配 → (digit_char, score)；非 digit/低分 →
        (None, best_score)。段先紧裁到 ink bbox（模板为紧裁位图），
        缩放对齐与 legacy _char_match 同法。"""
        from numpy import interp
        ys, xs = np.where(seg > 0)
        if len(xs) < 3:
            return None, 0.0
        seg = seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        g0 = (seg > 0).astype(np.float64)
        best_lab, bsc = None, 0.0
        for lab, tpl in assets.items():
            if not lab.startswith("digit"):
                continue
            ss = np.array(seg.shape, dtype=float)
            ts = np.array(tpl.shape, dtype=float)
            fy, fx = ss / ts
            yi = np.clip((np.arange(tpl.shape[0]) * fy).astype(int),
                         0, seg.shape[0] - 1)
            xi = np.clip((np.arange(tpl.shape[1]) * fx).astype(int),
                         0, seg.shape[1] - 1)
            samp = g0[np.ix_(yi, xi)]
            sc = _ncc(samp, tpl)
            if sc > bsc:
                best_lab, bsc = lab, sc
        if best_lab is None or bsc < CHAR_SCORE_MIN:
            return None, bsc
        return best_lab.split("_", 1)[1], bsc

    def _anchored_parse(self, mask: np.ndarray,
                        assets: dict) -> tuple | None:
        """锚定滑动解析（单模板集）→ (sb, bb, ante, mean_conf) | None。

        先以高置信锚字符（slash/K/括号）全宽滑动定位，再按锚点相对
        偏移窗口（±20px 漂移容忍）匹配其余槽位；digit 区间列分割逐段
        匹配。任一锚/段失败 → None（诚实降级，不猜）。
        """
        need = ("slash", "K", "paren_open", "paren_close")
        if any(n not in assets for n in need):
            return None
        # 1. 锚定位（全宽滑动）
        s_sc, sx, _ = self._slide(mask, assets["slash"])
        if s_sc < CHAR_SCORE_MIN:
            return None
        sw = assets["slash"].shape[1]
        # K1/K2：slash 左/右窗口内各自最优峰（相对偏移 ±20px 容忍）
        k1_sc, k1x, _ = self._slide(
            mask, assets["K"], (sx + ANCHOR_K1_WIN[0], sx + ANCHOR_K1_WIN[1]))
        k2_sc, k2x, _ = self._slide(
            mask, assets["K"], (sx + ANCHOR_K2_WIN[0], sx + ANCHOR_K2_WIN[1]))
        if k1_sc < CHAR_SCORE_MIN or k2_sc < CHAR_SCORE_MIN:
            return None
        kw = assets["K"].shape[1]
        if not (k1x + kw <= sx + 1 and k2x >= sx + sw):
            return None
        # 括号锚：K2 右侧 / po 右侧窗口内最优峰
        po_sc, pox, _ = self._slide(
            mask, assets["paren_open"],
            (k2x + ANCHOR_PO_WIN[0], k2x + ANCHOR_PO_WIN[1]))
        if po_sc < CHAR_SCORE_MIN:
            return None
        pow_ = assets["paren_open"].shape[1]
        pc_sc, pcx, _ = self._slide(
            mask, assets["paren_close"],
            (pox + ANCHOR_PC_WIN[0], pox + ANCHOR_PC_WIN[1]))
        if pc_sc < CHAR_SCORE_MIN:
            return None
        pcw = assets["paren_close"].shape[1]
        # 2. digit 区间（锚点相对偏移界定）
        runs = self._column_runs(mask)
        scores = [s_sc, k1_sc, k2_sc, po_sc, pc_sc]

        def seg_digits(a: int, b: int, tail_walk: bool):
            """区间 [a,b] 内 digit 段匹配。run 与区间取**重叠+裁剪**
            （实测括号与相邻数字共用一个 ink run——`(3`/`0)` 零间隙），
            裁剪后宽 <2 的残段（括号边缘）丢弃。tail_walk=True 时从右
            向左走、段间 gap>SB_GAP_MAX 即停（SB 区间——dash/hand-id
            尾部在窗口内时如实截断）；False 时任一段失败即整体失败。"""
            segs = []
            for r in runs:
                c0, c1 = max(r[0], a), min(r[1], b)
                if c1 - c0 + 1 < 2:
                    continue
                segs.append((c0, c1))
            segs.sort(key=lambda r: r[0])
            if tail_walk:
                if not segs or k1x - segs[-1][1] > SB_GAP_MAX:
                    return ""
                walk, cur = [], segs[-1]
                d, sc = self._match_digit_seg(
                    mask[:, cur[0]:cur[1] + 1], assets)
                if d is None:
                    return ""
                walk.append(d)
                scores.append(sc)
                for r in reversed(segs[:-1]):
                    if cur[0] - r[1] > SB_GAP_MAX:
                        break
                    d, sc = self._match_digit_seg(mask[:, r[0]:r[1] + 1],
                                                  assets)
                    if d is None:
                        break
                    walk.append(d)
                    scores.append(sc)
                    cur = r
                return "".join(reversed(walk))
            out = []
            for c0, c1 in segs:
                d, sc = self._match_digit_seg(mask[:, c0:c1 + 1], assets)
                if d is None:
                    return None
                out.append(d)
                scores.append(sc)
            return "".join(out)

        sb_d = seg_digits(max(0, k1x - SB_WALK_MAX), k1x - 1, tail_walk=True)
        bb_d = seg_digits(sx + sw, k2x - 1, tail_walk=False)
        ante_d = seg_digits(pox + pow_, pcx - 1, tail_walk=False)
        if not sb_d or not bb_d or not ante_d:
            return None
        sb = _to_num(sb_d + "K")
        bb = _to_num(bb_d + "K")
        ante = _to_num(ante_d)
        if sb is None or bb is None or ante is None:
            return None
        return sb, bb, ante, float(sum(scores) / len(scores))

    # ── V-4（Round15）：微级别 ₮ 计量锚定解析 ─────────────────────────
    def _match_micro_seg(self, seg: np.ndarray, assets: dict) -> tuple:
        """段 → token（'0'..'9' / '.'）→ (token, score)。

        先几何点判定（紧裁 h≤3 & w≤5 & 底行≥MICRO_DOT_BOTTOM_MIN → '.'；
        几何而非 NCC——点模板紧裁后全 1 无方差，NCC 无定义），否则 digit
        模板匹配（与 _match_digit_seg 同法缩放对齐，门 CHAR_SCORE_MIN）。
        非 digit/低分 → (None, best_score)——诚实失败。
        """
        ys, xs = np.where(seg > 0)
        if len(xs) < 2:
            return None, 0.0
        seg = seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = seg.shape
        if (h <= MICRO_DOT_MAX_H and w <= MICRO_DOT_MAX_W
                and ys.max() >= MICRO_DOT_BOTTOM_MIN):
            return ".", 1.0
        from numpy import interp
        g0 = (seg > 0).astype(np.float64)
        best_lab, bsc = None, 0.0
        for lab, tpl in assets.items():
            if not lab.startswith("digit"):
                continue
            ss = np.array(seg.shape, dtype=float)
            ts = np.array(tpl.shape, dtype=float)
            fy, fx = ss / ts
            yi = np.clip((np.arange(tpl.shape[0]) * fy).astype(int),
                         0, seg.shape[0] - 1)
            xi = np.clip((np.arange(tpl.shape[1]) * fx).astype(int),
                         0, seg.shape[1] - 1)
            samp = g0[np.ix_(yi, xi)]
            sc = _ncc(samp, tpl)
            if sc > bsc:
                best_lab, bsc = lab, sc
        if best_lab is None or bsc < CHAR_SCORE_MIN:
            return None, bsc
        return best_lab.split("_", 1)[1], bsc

    def _anchored_parse_micro(self, mask: np.ndarray,
                              assets: dict) -> tuple | None:
        """微级别 ₮ 格式锚定解析 → (sb, bb, mean_conf) | None。

        结构：`₮<dec>/<dec>`（尾零省略；无 K、无 ante 括号）。
        锚 = slash + tether（各模板全宽滑动 NCC ≥CHAR_SCORE_MIN）：
        tether_sb 取 slash 左侧 argmax、tether_bb 取右侧 argmax；
        SB token = [tether_sb 右缘, slash) 区间列分割逐段匹配；
        BB token = [tether_bb 右缘, 带尾) 同法。段先按全局 run 重叠裁剪
        （₮0 / 5/₮0 共 run 实测存在）。任一锚/段失败 → None（诚实降级）。
        NOT_COVERED（→None）：₮ 格式带 ante 括号、digit{2,3,4,9}、小数
        K/M——模板未覆盖即失败，不猜。
        """
        need = ("tether", "slash")
        if any(n not in assets for n in need):
            return None
        s_sc, sx, _ = self._slide(mask, assets["slash"])
        if s_sc < CHAR_SCORE_MIN:
            return None
        sw = assets["slash"].shape[1]
        tw = assets["tether"].shape[1]
        # tether_sb：slash 左侧窗口 argmax；tether_bb：slash 右侧窗口 argmax
        t1_sc, t1x, _ = self._slide(mask, assets["tether"], (0, sx - tw))
        t2_sc, t2x, _ = self._slide(
            mask, assets["tether"], (sx + sw, mask.shape[1] - tw))
        if t1_sc < CHAR_SCORE_MIN or t2_sc < CHAR_SCORE_MIN:
            return None
        if not (t1x + tw <= sx and t2x >= sx + sw):
            return None
        runs = self._column_runs(mask)
        scores = [s_sc, t1_sc, t2_sc]

        def span_token(a: int, b: int) -> str | None:
            """[a,b) 区间内 token 序列（digit/dot 几何）；任一段失败 → None。"""
            segs = []
            for r in runs:
                c0, c1 = max(r[0], a), min(r[1], b - 1)
                if c1 - c0 + 1 < 2:
                    continue
                segs.append((c0, c1))
            segs.sort(key=lambda r: r[0])
            out = []
            for c0, c1 in segs:
                tok, sc = self._match_micro_seg(mask[:, c0:c1 + 1], assets)
                if tok is None:
                    return None
                out.append(tok)
                scores.append(sc)
            return "".join(out) if out else None

        sb_t = span_token(t1x + tw, sx)
        bb_t = span_token(t2x + tw, mask.shape[1])
        if not sb_t or not bb_t:
            return None
        sb = _to_num(sb_t)
        bb = _to_num(bb_t)
        if sb is None or bb is None or sb >= bb:
            # sb >= bb 结构性非法（盲注序）→ 诚实拒绝
            return None
        return sb, bb, float(sum(scores) / len(scores))
