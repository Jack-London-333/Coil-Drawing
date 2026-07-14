"""STEP 导出的实体门禁与写后回读校验。

这个模块故意不依赖 ``model3d``：三维构造只需把最终零件和实际的
XCAF writer 传入 :func:`export_step_guarded`。输出先写入同目录临时文件，
只有通过文本及 XCAF 回读校验后才原子替换目标文件。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable, Iterable
from uuid import uuid4


class StepValidationError(RuntimeError):
    """STEP 不是预期的、可供 CAD 使用的实心实体装配。"""


@dataclass(frozen=True)
class StepLeaf:
    """XCAF 回读得到的一个叶子零件。"""

    name: str
    volume: float


_FORBIDDEN_STEP_ENTITIES = (
    "OPEN_SHELL",
    "SHELL_BASED_SURFACE_MODEL",
    "GEOMETRIC_SET",
)


def _shape_of(obj):
    """接受 build123d 对象或原生 TopoDS_Shape。"""
    return getattr(obj, "wrapped", obj)


def _shape_volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def validate_solid_shape(obj, name: str, stage: str) -> float:
    """严格验证一个叶子恰为单一、闭合、有效且有正体积的 SOLID。

    不接受“只包含一个 solid 的 Compound”。顶层类型本身必须是 SOLID，
    以免 STEP 中形成额外装配层或被 CAD 翻译成曲面体。
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer

    label = name or "<未命名零件>"
    shape = _shape_of(obj)
    if shape is None or shape.IsNull():
        raise StepValidationError(f"{stage}：{label} 是空形状")
    if shape.ShapeType() != TopAbs_ShapeEnum.TopAbs_SOLID:
        kind = str(shape.ShapeType()).rsplit(".", 1)[-1]
        raise StepValidationError(
            f"{stage}：{label} 顶层类型为 {kind}，必须恰为一个 SOLID")
    if not BRepCheck_Analyzer(shape, True).IsValid():
        raise StepValidationError(f"{stage}：{label} 不是有效闭合 B-Rep 实体")

    shells = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SHELL)
    shell_count = 0
    while shells.More():
        shell_count += 1
        shell = shells.Current()
        if not BRep_Tool.IsClosed_s(shell):
            raise StepValidationError(f"{stage}：{label} 含开放 Shell")
        shells.Next()
    if shell_count == 0:
        raise StepValidationError(f"{stage}：{label} 没有闭合 Shell")

    volume = _shape_volume(shape)
    if not volume > 1e-9:
        raise StepValidationError(
            f"{stage}：{label} 体积不是正值（{volume:.9g} mm³）")
    return volume


def _label_name(label) -> str:
    """读取 XCAF 名称，并修复 OCP Python 包装层的 UTF-8/Latin-1 展开。"""
    from OCP.TDataStd import TDataStd_Name

    attr = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return ""
    text = attr.Get().ToExtString()
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def read_step_leaves(filepath: str | Path) -> list[StepLeaf]:
    """用 STEPCAFControl 回读装配，并严格验证每一个叶子零件。"""
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

    path = Path(filepath)
    doc = TDocStd_Document(TCollection_ExtendedString("XCAF"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise StepValidationError(f"STEP 写后回读失败：无法读取 {path.name}")
    if not reader.Transfer(doc):
        raise StepValidationError(f"STEP 写后回读失败：无法转换 {path.name}")

    leaves: list[StepLeaf] = []

    def visit(parent=None) -> None:
        labels = TDF_LabelSequence()
        if parent is None:
            shape_tool.GetFreeShapes(labels)
        else:
            XCAFDoc_ShapeTool.GetComponents_s(parent, labels)

        for index in range(1, labels.Length() + 1):
            instance = labels.Value(index)
            referred = TDF_Label()
            if XCAFDoc_ShapeTool.IsReference_s(instance):
                if (not XCAFDoc_ShapeTool.GetReferredShape_s(instance, referred)
                        or referred.IsNull()):
                    raise StepValidationError(
                        "STEP 写后回读失败：装配实例没有有效的引用零件")
            else:
                referred = instance

            if XCAFDoc_ShapeTool.IsAssembly_s(referred):
                visit(referred)
                continue

            shape = XCAFDoc_ShapeTool.GetShape_s(referred)
            name = _label_name(instance) or _label_name(referred)
            volume = validate_solid_shape(
                shape, name or f"叶子{len(leaves) + 1}", "STEP 写后回读")
            leaves.append(StepLeaf(name=name, volume=volume))

    visit()
    if not leaves:
        raise StepValidationError("STEP 写后回读失败：文件中没有叶子零件")
    return leaves


def _validate_step_text(filepath: str | Path) -> None:
    text = Path(filepath).read_text(encoding="ascii", errors="ignore")
    for entity in _FORBIDDEN_STEP_ENTITIES:
        if re.search(rf"\b{re.escape(entity)}\b", text, flags=re.IGNORECASE):
            raise StepValidationError(
                f"STEP 写后校验失败：文件含禁止的曲面实体 {entity}")


def verify_step_file(filepath: str | Path,
                     expected_names: Iterable[str]) -> list[StepLeaf]:
    """校验 STEP 文本实体类型，并通过 XCAF 逐叶回读。"""
    names = list(expected_names)
    _validate_step_text(filepath)
    leaves = read_step_leaves(filepath)
    if len(leaves) != len(names):
        raise StepValidationError(
            "STEP 写后校验失败：叶子零件数不一致，"
            f"预期 {len(names)}，实际 {len(leaves)}")

    actual_names = [leaf.name for leaf in leaves]
    # 某些 STEP 翻译器只生成 ``1``、``2`` 这样的产品序号，它们不是
    # 可比较的零件名；只有全部名称都不是这种自动占位符时才严格比较。
    def is_reliable(name: str) -> bool:
        stripped = name.strip()
        return bool(stripped) and not (
            re.fullmatch(r"\d+", stripped)
            or re.fullmatch(r"[?_]+", stripped)
            or stripped.startswith("Open CASCADE STEP translator")
        )

    names_reliable = all(is_reliable(name) for name in actual_names)
    if names and all(names) and names_reliable:
        if Counter(actual_names) != Counter(names):
            raise StepValidationError(
                "STEP 写后校验失败：零件名称不一致，"
                f"预期 {names!r}，实际 {actual_names!r}")
    return leaves


def export_step_guarded(
    parts: Iterable,
    filepath: str | Path,
    asm_name: str,
    writer: Callable[[list, str, str], None],
    name_fixer: Callable[..., int],
) -> list[StepLeaf]:
    """经严格门禁写 STEP；失败只删除临时不合格文件。

    ``writer`` 的签名与 ``model3d._export_step_xcaf`` 一致。写入同目录
    临时文件可保证失败时不会留下半成品，也不会破坏已有的合格输出。
    """
    part_list = list(parts)
    if not part_list:
        raise StepValidationError("STEP 导出前：没有可导出的零件")
    expected_names: list[str] = []
    for index, part in enumerate(part_list, start=1):
        name = str(getattr(part, "name", "") or f"零件{index}")
        solid = getattr(part, "solid", None)
        validate_solid_shape(solid, name, "STEP 导出前")
        expected_names.append(name)

    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.stem}.tmp-{uuid4().hex}{target.suffix or '.step'}")
    try:
        writer(part_list, str(temp), asm_name)
        if not temp.is_file() or temp.stat().st_size == 0:
            raise StepValidationError("STEP 写出失败：未生成有效文件")
        name_fixer(temp, header_name=asm_name)
        leaves = verify_step_file(temp, expected_names)
        os.replace(temp, target)
        return leaves
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
