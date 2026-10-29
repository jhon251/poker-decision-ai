"""纯标准库 PNG 解码器（zlib + struct + 反滤波）。

与 png_encode.py 同一哲学：零第三方图像依赖。支持 8-bit、
非隔行的 color type 0/2/3/4/6（灰度/RGB/调色板/灰度+alpha/RGBA）。
不支持的格式（16-bit、Adam7 隔行、非法结构）→ 明确 ValueError，绝不猜测。

用途：真实截图文件 → 像素（ImageData），供 realtime 流水线消费。
本解码器只还原像素，不参与任何识别语义。
"""
from __future__ import annotations

import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int, bpp: int) -> list:
    """逐行反滤波（filter 0-4）。返回 list[bytes]（每行原始采样字节）。"""
    rowbytes = width * bpp
    pos = 0
    prev = bytearray(rowbytes)
    rows = []
    for _ in range(height):
        if pos >= len(raw):
            raise ValueError("PNG scanline data truncated")
        ft = raw[pos]
        pos += 1
        if ft > 4:
            raise ValueError(f"非法 filter 类型 {ft}")
        row = bytearray(raw[pos:pos + rowbytes])
        if len(row) != rowbytes:
            raise ValueError("PNG scanline data truncated")
        pos += rowbytes
        if ft == 0:
            pass
        elif ft == 1:                       # Sub
            for x in range(bpp, rowbytes):
                row[x] = (row[x] + row[x - bpp]) & 0xFF
        elif ft == 2:                       # Up
            for x in range(rowbytes):
                row[x] = (row[x] + prev[x]) & 0xFF
        elif ft == 3:                       # Average
            for x in range(rowbytes):
                a = row[x - bpp] if x >= bpp else 0
                row[x] = (row[x] + ((a + prev[x]) // 2)) & 0xFF
        else:                               # Paeth
            for x in range(rowbytes):
                a = row[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                row[x] = (row[x] + _paeth(a, b, c)) & 0xFF
        rows.append(bytes(row))
        prev = row
    return rows


def decode_png(data: bytes) -> tuple:
    """PNG 字节 → (width, height, rgba_bytes)。

    支持：bit depth 8；color type 0/2/3/4/6；非隔行。其余 → ValueError。
    """
    if not data.startswith(_SIGNATURE):
        raise ValueError("不是合法 PNG（签名不符）")
    pos = 8
    idat = bytearray()
    width = height = bit_depth = color_type = None
    palette: list | None = None
    palette_alpha: list | None = None
    end = False
    while pos < len(data):
        if end:
            raise ValueError("IEND 之后仍有数据")
        if pos + 8 > len(data):
            raise ValueError("PNG chunk 头截断")
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        pos += 8
        if pos + length > len(data):
            raise ValueError("PNG chunk 数据截断")
        chunk = data[pos:pos + length]
        pos += length + 4                        # 跳过 CRC
        if tag == b"IHDR":
            if len(chunk) != 13:
                raise ValueError("IHDR 长度非法")
            width, height, bit_depth, color_type = struct.unpack(
                ">IIBB", chunk[:10])
            comp, filt, interlace = chunk[10], chunk[11], chunk[12]
            if comp != 0 or filt != 0:
                raise ValueError("不支持的压缩/滤波方法")
            if bit_depth != 8:
                raise ValueError(f"仅支持 8-bit PNG，实际 {bit_depth}")
            if color_type not in _CHANNELS:
                raise ValueError(f"不支持的 color type {color_type}")
            if interlace != 0:
                raise ValueError("Adam7 隔行 PNG 暂不支持")
        elif tag == b"PLTE":
            if len(chunk) % 3 != 0:
                raise ValueError("PLTE 长度非法")
            palette = [tuple(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]
        elif tag == b"tRNS":
            palette_alpha = list(chunk)
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            end = True
        # 其余 ancillary chunk 跳过
    if width is None or height is None:
        raise ValueError("缺少 IHDR")
    channels = _CHANNELS[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as e:
        raise ValueError(f"IDAT 解压失败: {e}") from e
    rows = _unfilter(raw, width, height, channels)

    out = bytearray(width * height * 4)
    stride = width * 4
    for y, row in enumerate(rows):
        base = y * stride
        if color_type == 6:                      # RGBA
            out[base:base + width * 4] = row
        elif color_type == 2:                    # RGB
            for x in range(width):
                i, o = x * 3, base + x * 4
                out[o:o + 4] = row[i:i + 3] + b"\xff"
        elif color_type == 0:                    # gray
            for x in range(width):
                v = row[x]
                out[base + x * 4:base + x * 4 + 4] = bytes((v, v, v, 255))
        elif color_type == 4:                    # gray + alpha
            for x in range(width):
                v, a = row[x * 2], row[x * 2 + 1]
                out[base + x * 4:base + x * 4 + 4] = bytes((v, v, v, a))
        elif color_type == 3:                    # palette
            if palette is None:
                raise ValueError("调色板 PNG 缺少 PLTE")
            for x in range(width):
                idx = row[x]
                if idx >= len(palette):
                    raise ValueError(f"调色板索引越界 {idx}")
                r, g, b = palette[idx]
                a = (palette_alpha[idx] if palette_alpha is not None
                     and idx < len(palette_alpha) else 255)
                out[base + x * 4:base + x * 4 + 4] = bytes((r, g, b, a))
    return width, height, bytes(out)
