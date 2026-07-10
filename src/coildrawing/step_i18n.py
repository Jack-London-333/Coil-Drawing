"""STEP 文件中文名称修复（ISO 10303-21 \\X2\\ 转义）。

背景：build123d 写入部件名时把 Python 传入的 UTF-8 字节按 Latin-1
逐字符塞进 TCollection_ExtendedString（少传 multibyte 标志），OCCT 再把
这些错误码位以 UTF-8 写进 STEP，于是文件里的中文名成了双重乱码；
文件头 FILE_NAME 走 TCollection_HAsciiString，非 ASCII 字节直接变 '?'。

本模块在导出完成后对 STEP 文件做确定性后处理：

1. 扫描所有字符串字面量，将乱码还原为原始中文；
2. 按 ISO 10303-21 标准改写为 \\X2\\XXXX\\X0\\（UTF-16BE 十六进制）转义，
   使全文件回到纯 ASCII —— SolidWorks 等符合标准的软件可正确解码显示，
   名称仍是普通产品名属性，导入后可自由改名、进 BOM；
3. FILE_NAME 中被写成 '?' 串的模型名用给定名称重写。
"""

from __future__ import annotations

import re
from pathlib import Path

# STEP 字符串字面量：单引号包裹，内部单引号写作 ''（可跨行）
_STRING_RE = re.compile(r"'(?:''|[^'])*'")
_FILE_NAME_RE = re.compile(r"(FILE_NAME\(\s*')((?:''|[^'])*)(')")


def x2_escape(text: str) -> str:
    """把任意字符串编码为 STEP 纯 ASCII 形式（\\X2\\/\\X4\\ 转义）。

    可打印 ASCII 原样保留（' 和 \\ 按标准转义）；
    BMP 内非 ASCII 字符成段编码为 \\X2\\ + 4 位十六进制；
    BMP 外字符编码为 \\X4\\ + 8 位十六进制。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        o = ord(text[i])
        if 0x20 <= o <= 0x7E:
            ch = text[i]
            if ch == "'":
                out.append("''")
            elif ch == "\\":
                out.append("\\\\")
            else:
                out.append(ch)
            i += 1
        elif o <= 0xFFFF:
            j = i
            while j < n and not (0x20 <= ord(text[j]) <= 0x7E) and ord(text[j]) <= 0xFFFF:
                j += 1
            hexes = "".join(f"{ord(c):04X}" for c in text[i:j])
            out.append(f"\\X2\\{hexes}\\X0\\")
            i = j
        else:
            j = i
            while j < n and ord(text[j]) > 0xFFFF:
                j += 1
            hexes = "".join(f"{ord(c):08X}" for c in text[i:j])
            out.append(f"\\X4\\{hexes}\\X0\\")
            i = j
    return "".join(out)


def _recover_original(literal_body: str) -> str | None:
    """从字符串字面量内容还原原始文本。

    返回 None 表示无需处理（纯 ASCII）。
    兼容两种损坏形态：
      * UTF-8 字节被按 Latin-1 展开成 U+0080..U+00FF 字符（当前 build123d 行为）；
      * 已是正确 Unicode（未来 build123d 修复后仍安全）。
    """
    if all(0x20 <= ord(c) <= 0x7E for c in literal_body):
        return None
    text = literal_body.replace("''", "'")
    if all(ord(c) <= 0xFF for c in text):
        raw = text.encode("latin-1")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            decoded = text  # 本来就是 Latin-1 文本
        return decoded
    return text  # 已是正确 Unicode（含 CJK 等）


def fix_step_names(filepath: str | Path, header_name: str | None = None) -> int:
    """后处理 STEP 文件：所有非 ASCII 名称改写为 \\X2\\ 转义。

    返回被改写的字符串数量。header_name 提供时，用它重写 FILE_NAME 中
    被 OCCT 破坏成 '?' 的模型名。
    """
    path = Path(filepath)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        body = m.group(0)[1:-1]
        original = _recover_original(body)
        if original is None:
            return m.group(0)
        count += 1
        return f"'{x2_escape(original)}'"

    text = _STRING_RE.sub(_sub, text)

    if header_name:
        def _sub_header(m: re.Match) -> str:
            nonlocal count
            if "?" not in m.group(2):
                return m.group(0)
            count += 1
            return f"{m.group(1)}{x2_escape(header_name)}{m.group(3)}"

        text = _FILE_NAME_RE.sub(_sub_header, text, count=1)

    try:
        data = text.encode("ascii")
    except UnicodeEncodeError:
        data = text.encode("utf-8")  # 理论不会发生，兜底不破坏文件
    path.write_bytes(data)
    return count
