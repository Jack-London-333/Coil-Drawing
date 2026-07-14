"""STEP 单实体门禁和 XCAF 写后回读测试。"""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.compat import import_build123d  # noqa: E402
from coildrawing.step_export import (  # noqa: E402
    StepValidationError,
    export_step_guarded,
    validate_solid_shape,
    verify_step_file,
)
from coildrawing.step_i18n import fix_step_names  # noqa: E402


@pytest.fixture(scope="module")
def b3d():
    return import_build123d()


def _part(name, solid):
    return SimpleNamespace(name=name, solid=solid)


def _build123d_writer(b3d):
    def writer(parts, filepath, asm_name):
        children = []
        for part in parts:
            part.solid.label = part.name
            children.append(part.solid)
        assembly = b3d.Compound(label=asm_name, children=children)
        b3d.export_step(assembly, filepath)

    return writer


def test_preflight_requires_top_level_solid(b3d):
    box = b3d.Solid.make_box(2, 3, 4)
    assert validate_solid_shape(box, "铜导线", "测试") == \
        pytest.approx(24.0)

    compound = b3d.Compound(children=[box])
    with pytest.raises(StepValidationError, match="顶层类型.*SOLID"):
        validate_solid_shape(compound, "伪单实体复合体", "测试")

    with pytest.raises(StepValidationError, match="顶层类型.*SOLID"):
        validate_solid_shape(box.faces()[0], "开放曲面", "测试")


def test_guarded_export_roundtrips_leaf_solids_and_names(tmp_path, b3d):
    parts = [
        _part("铜导线", b3d.Solid.make_box(2, 3, 4)),
        _part("匝间云母带", b3d.Solid.make_box(1, 2, 3).moved(
            b3d.Location((10, 0, 0)))),
    ]
    step = tmp_path / "valid.step"
    leaves = export_step_guarded(
        parts, step, "成型线圈", _build123d_writer(b3d), fix_step_names)

    assert step.is_file()
    # build123d 的通用 writer 只保留数字产品序号；项目自建 XCAF writer
    # 会保留中文名，门禁会在名称可靠时严格比较。
    assert len(leaves) == 2
    assert sorted(leaf.volume for leaf in leaves) == pytest.approx([6.0, 24.0])
    assert "OPEN_SHELL" not in step.read_text(encoding="ascii")


def test_preflight_failure_does_not_write_or_delete_existing(tmp_path, b3d):
    target = tmp_path / "preserve.step"
    target.write_text("previous-valid-output", encoding="ascii")
    called = False

    def writer(*_args):
        nonlocal called
        called = True

    bad = _part("曲面", b3d.Solid.make_box(1, 1, 1).faces()[0])
    with pytest.raises(StepValidationError, match="顶层类型.*SOLID"):
        export_step_guarded([bad], target, "装配", writer, fix_step_names)
    assert not called
    assert target.read_text(encoding="ascii") == "previous-valid-output"


@pytest.mark.parametrize("entity", [
    "OPEN_SHELL",
    "SHELL_BASED_SURFACE_MODEL",
    "GEOMETRIC_SET",
])
def test_forbidden_surface_entities_remove_temporary_output(
        tmp_path, b3d, entity):
    target = tmp_path / f"bad_{entity}.step"
    part = _part("实体", b3d.Solid.make_box(1, 1, 1))
    base_writer = _build123d_writer(b3d)

    def writer(parts, filepath, asm_name):
        base_writer(parts, filepath, asm_name)
        with open(filepath, "a", encoding="ascii") as stream:
            stream.write(f"\n/* {entity} */\n")

    with pytest.raises(StepValidationError, match=entity):
        export_step_guarded([part], target, "装配", writer, fix_step_names)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp-*.step"))


def test_postread_rejects_leaf_count_mismatch(tmp_path, b3d):
    parts = [
        _part("零件A", b3d.Solid.make_box(1, 1, 1)),
        _part("零件B", b3d.Solid.make_box(2, 1, 1).moved(
            b3d.Location((5, 0, 0)))),
    ]
    target = tmp_path / "missing_leaf.step"
    base_writer = _build123d_writer(b3d)

    def writer(_parts, filepath, asm_name):
        base_writer(parts[:1], filepath, asm_name)

    with pytest.raises(StepValidationError, match="叶子零件数不一致"):
        export_step_guarded(parts, target, "装配", writer, fix_step_names)
    assert not target.exists()


def test_verify_rejects_step_whose_leaf_is_a_surface(tmp_path, b3d):
    face = b3d.Solid.make_box(2, 2, 2).faces()[0]
    face.label = "曲面零件"
    step = tmp_path / "surface.step"
    b3d.export_step(face, str(step))
    fix_step_names(step, header_name="曲面装配")

    with pytest.raises(StepValidationError,
                       match="OPEN_SHELL|必须恰为一个 SOLID"):
        verify_step_file(step, ["曲面零件"])
