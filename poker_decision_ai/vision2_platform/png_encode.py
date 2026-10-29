"""纯标准库 PNG 编码（zlib + struct）：RGBA raw bytes → PNG bytes。

用途：把截图/ROI 裁剪传给远程多模态接口（data URL）。无第三方图像依赖；
解码（真实展示）留待 PIL/浏览器，测试只验证 PNG 结构与 IDAT 可解压。
"""
from __future__ import annotations

import struct
import zlib


def encode_rgba_png(width: int, height: int, rgba: bytes) -> bytes:
    """RGBA（row-major, 每像素 4 字节）→ PNG 字节流（8bit/RGBA, filter 0）。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"非法尺寸 {width}x{height}")
    if len(rgba) != width * height * 4:
        raise ValueError(f"RGBA 字节数 {len(rgba)} != 期望 "
                         f"{width * height * 4}")
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride]
                   for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
