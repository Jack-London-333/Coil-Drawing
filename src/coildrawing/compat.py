"""build123d 导入兼容层。

build123d 在导入时会扫描系统字体目录并用 fontTools 解析每个字体文件，
遇到损坏/非标准字体（如 bad sfntVersion）会直接抛异常导致导入失败。
这里在导入 build123d 期间临时替换 glob.glob，把无法解析的字体文件
从扫描结果中剔除，导入完成后恢复原样，不修改 site-packages。
"""

from __future__ import annotations

import glob as _glob_module
import logging

logging.getLogger("fontTools").setLevel(logging.CRITICAL)

_FONT_SUFFIXES = ("ttf", "otf", "ttc")


def _make_safe_glob(orig_glob):
    from fontTools.ttLib import TTFont, ttCollection

    def _probe(path: str) -> None:
        # 与 build123d 内部同样的解析方式：构造 + 读取 name 表
        if str(path).lower().endswith(".ttc"):
            fonts = ttCollection.TTCollection(path)
        else:
            fonts = [TTFont(path)]
        for f in fonts:
            _ = f["name"].names
            if "fvar" in f:
                _ = f["fvar"].instances

    def safe_glob(pattern, *args, **kwargs):
        results = orig_glob(pattern, *args, **kwargs)
        if not str(pattern).lower().endswith(_FONT_SUFFIXES):
            return results
        good = []
        for path in results:
            try:
                _probe(path)
            except UnicodeDecodeError:
                pass  # build123d 自己能处理这种情况
            except Exception:
                continue
            good.append(path)
        return good

    return safe_glob


def import_build123d():
    """导入并返回 build123d，容忍系统中存在损坏的字体文件。"""
    import sys

    if "build123d" in sys.modules:
        return sys.modules["build123d"]

    orig_glob = _glob_module.glob
    _glob_module.glob = _make_safe_glob(orig_glob)
    try:
        import build123d
    finally:
        _glob_module.glob = orig_glob
    return build123d
