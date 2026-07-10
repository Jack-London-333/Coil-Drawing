"""STEP 中文名称 \\X2\\ 转义修复的单元与回读测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.step_i18n import fix_step_names, x2_escape  # noqa: E402


def test_x2_escape_pure_ascii():
    assert x2_escape("Copper-1") == "Copper-1"
    assert x2_escape("a'b") == "a''b"
    assert x2_escape("a\\b") == "a\\\\b"


def test_x2_escape_cjk():
    assert x2_escape("成型线圈") == "\\X2\\6210578B7EBF5708\\X0\\"
    assert x2_escape("绝缘层1-对地云母带 1") == (
        "\\X2\\7EDD7F185C42\\X0\\1-\\X2\\5BF957304E916BCD5E26\\X0\\ 1")


def test_fix_step_names_roundtrip(tmp_path):
    """导出带中文名的小装配 → 修复 → 文件纯 ASCII 且 OCCT 能回读中文。"""
    from coildrawing.compat import import_build123d

    b3d = import_build123d()
    box = b3d.Solid.make_box(2, 2, 2)
    box.label = "铜导体束"
    box.color = b3d.Color(0.8, 0.5, 0.2)
    asm = b3d.Compound(label="成型线圈", children=[box])
    step = tmp_path / "unit_cjk.step"
    b3d.export_step(asm, str(step))

    n = fix_step_names(step, header_name=asm.label)
    assert n > 0

    raw = step.read_bytes()
    raw.decode("ascii")  # 全文件必须纯 ASCII
    text = raw.decode("ascii")
    assert "\\X2\\6210578B7EBF5708\\X0\\" in text      # 成型线圈
    assert "\\X2\\94DC5BFC4F53675F\\X0\\" in text      # 铜导体束
    assert "FILE_NAME('\\X2\\6210578B7EBF5708\\X0\\'" in text
    assert "?" not in text.split("FILE_NAME")[1][:60]

    # OCCT 回读验证名称还原
    from build123d import import_step

    back = import_step(str(step))

    def walk(node):
        yield node.label
        for c in getattr(node, "children", []):
            yield from walk(c)

    assert "铜导体束" in set(walk(back))


def test_fix_step_names_idempotent(tmp_path):
    """重复处理不应再改动文件。"""
    from coildrawing.compat import import_build123d

    b3d = import_build123d()
    box = b3d.Solid.make_box(1, 1, 1)
    box.label = "绝缘层1-对地云母带 1"
    asm = b3d.Compound(label="线圈", children=[box])
    step = tmp_path / "unit_idem.step"
    b3d.export_step(asm, str(step))

    fix_step_names(step, header_name=asm.label)
    first = step.read_bytes()
    n2 = fix_step_names(step, header_name=asm.label)
    assert n2 == 0
    assert step.read_bytes() == first
