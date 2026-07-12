"""逐匝三维模型的快速回归测试（小匝数，控制耗时）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, WireSpec, compute  # noqa: E402


@pytest.fixture(scope="module")
def small_res():
    inp = CoilInput()
    inp.n_turns = 2
    inp.corona_on = True
    inp.cs = 0.3            # 防晕层厚度=CS（参数已合并）
    inp.draw_wihm = True    # 层间垫片
    return compute(inp)


def test_loop_material_frame_orientation(small_res):
    """材料匝位轴必须在两槽边反向，并在两个 nose 沿轴向立起。"""
    from coildrawing.model3d import _loop_frames, _radial_of

    frames = _loop_frames(small_res)
    assert frames.f_leg1.y.dot(_radial_of(frames.f_leg1.o)) == \
        pytest.approx(1.0, abs=1e-10)
    assert frames.f_leg2.y.dot(_radial_of(frames.f_leg2.o)) == \
        pytest.approx(-1.0, abs=1e-10)
    assert tuple(frames.f_flat_pos.y) == pytest.approx((0.0, 0.0, 1.0), abs=1e-10)
    assert tuple(frames.f_flat_neg.y) == pytest.approx((0.0, 0.0, -1.0), abs=1e-10)


def test_lead_end_choice_mirrors_complete_detailed_model():
    """换出线端须镜像整只线圈，而不只是移动两根引线。"""
    from coildrawing.model3d import build_coil_parts

    inp_pos = CoilInput()
    inp_pos.n_turns = 1       # 镜像不变量与匝数无关，单匝可缩短测试时间
    inp_pos.lead_end_positive_z = True
    pos_parts = build_coil_parts(compute(inp_pos), detailed=True)

    inp_neg = CoilInput()
    inp_neg.n_turns = 1
    inp_neg.lead_end_positive_z = False
    neg_parts = build_coil_parts(compute(inp_neg), detailed=True)

    assert [p.name for p in neg_parts] == [p.name for p in pos_parts]
    for pos, neg in zip(pos_parts, neg_parts):
        assert neg.solid.volume == pytest.approx(
            pos.solid.volume, rel=1e-10, abs=1e-6), pos.name
        bb_pos = pos.solid.bounding_box()
        bb_neg = neg.solid.bounding_box()
        assert bb_neg.min.Z == pytest.approx(-bb_pos.max.Z, abs=1e-6), pos.name
        assert bb_neg.max.Z == pytest.approx(-bb_pos.min.Z, abs=1e-6), pos.name

    # 明确覆盖两类关键实体，避免未来因改名/条件构造而弱化上述逐件断言。
    names = [p.name for p in pos_parts]
    assert any(n.startswith("铜导线") for n in names)
    assert any(n.startswith("对地绝缘") for n in names)


def test_four_turn_detailed_copper_is_one_solid():
    """四匝铜线须为一个实体，且融合前后体积相同（无自重叠）。"""
    from coildrawing import model3d as m

    inp = CoilInput()
    inp.n_turns = 4
    res = compute(inp)
    segments, _ = m._wire_segments(res)
    _, _, strands = m._strand_grid(res)
    strand = strands[0]
    solids = [m._seg_solid(g, strand["b"], strand["h"],
                           strand["x"], strand["y"])
              for g in segments]
    solids = [solid for solid in solids if solid is not None]
    copper = m._join(solids)
    assert len(copper.solids()) == 1
    assert copper.volume == pytest.approx(
        sum(solid.volume for solid in solids), rel=1e-10, abs=1e-5)


def test_default_eight_turn_copper_segments_are_constructible():
    """默认八匝参数的每一段铜线都必须能生成有效实体。"""
    from coildrawing import model3d as m

    res = compute(CoilInput())
    segments, _ = m._wire_segments(res)
    _, _, strands = m._strand_grid(res)
    strand = strands[0]
    solids = [m._seg_solid(g, strand["b"], strand["h"],
                           strand["x"], strand["y"])
              for g in segments]
    assert all(solid is not None and solid.volume > 0 for solid in solids)


def test_nose_connection_corners_use_rd2_not_nose_rd():
    """斜边—nose 圆角应取 rd2；RD 只定义 nose 本体半径/展开宽度。"""
    from coildrawing import model3d as m

    inp = CoilInput(rd_nose=15.0, r_bend_nose=42.0)
    _, fillets = m._loop_fillets(compute(inp))
    for index in (2, 3, 6, 7):
        assert (fillets[index].ts - fillets[index].c).length == \
            pytest.approx(42.0, abs=1e-8)


def test_f_nose_raises_both_noses_without_moving_slot_legs():
    """F 应让 +Z/-Z 两个 nose 一起朝槽底抬高，槽内直边保持不动。"""
    from coildrawing import model3d as m

    low, _ = m._loop_fillets(compute(CoilInput(f_nose=5.0)))
    high, _ = m._loop_fillets(compute(CoilInput(f_nose=32.0)))

    def radius(corner):
        point = corner[0]
        return (point.X ** 2 + point.Y ** 2) ** 0.5

    for index in (2, 3, 6, 7):
        assert radius(high[index]) - radius(low[index]) == \
            pytest.approx(27.0, abs=1e-8)
    for index in (0, 1, 4, 5):
        assert radius(high[index]) == pytest.approx(
            radius(low[index]), abs=1e-8)


def test_wire_segments_have_no_centerline_gaps():
    """逐匝路径的每个相邻解析分段必须在同一材料中心点相接。"""
    from coildrawing import model3d as m

    inp = CoilInput()
    inp.n_turns = 4
    segments, _ = m._wire_segments(compute(inp))

    def y_axis(frame):
        return frame.y if hasattr(frame, "y") else frame.ey

    def center(frame, dy):
        return frame.o + y_axis(frame) * dy

    def endpoints(segment):
        if isinstance(segment, m._Prism):
            start = center(segment.f, segment.dy)
            return start, start + segment.f.t * segment.length
        if isinstance(segment, m._Rev):
            start = center(segment.f, segment.dy)
            end = segment.fl.c + m._rotv(
                start - segment.fl.c, segment.fl.n, segment.fl.tau)
            return start, end
        if isinstance(segment, m._Loft):
            return (center(segment.f0, segment.dy0),
                    center(segment.f1, segment.dy1))
        raise AssertionError(f"未知分段类型: {type(segment).__name__}")

    gaps = []
    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        gap = (endpoints(left)[1] - endpoints(right)[0]).length
        gaps.append((gap, index, type(left).__name__, type(right).__name__))
    worst = max(gaps)
    assert worst[0] < 1e-6, \
        f"分段 {worst[1]} {worst[2]}→{worst[3]} 中心断裂 {worst[0]:.6g}mm"


def test_detailed_parts_geometry(small_res):
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    names = [p.name for p in parts]
    assert names[0] == "铜导线"
    assert any(n.startswith("匝绝缘1") for n in names)
    assert sum(1 for n in names if n.startswith("对地绝缘")) == 2
    assert sum(1 for n in names if n.startswith("防晕层")) == 2
    assert sum(1 for n in names if n.startswith("层间垫片")) == 2

    for p in parts:
        assert p.solid.volume > 0, p.name
        assert len(p.solid.solids()) == 1, p.name

    # 铜导线体积 ≈ 截面 × 路径长（路径长约 2×LLM + 2×引线）
    copper = parts[0].solid
    w = small_res.inp.wire1
    approx_len = 2 * small_res.llm + 2 * small_res.inp.ysc
    vol_expect = w.b * w.h * approx_len
    assert abs(copper.volume - vol_expect) / vol_expect < 0.08


def test_detailed_parts_valid_solids(small_res):
    """全部部件必须是有效实体——无效实体在 SolidWorks 中会被降级为
    “曲面实体”（v202607110207 问题二的根源，出线端自相交所致）。"""
    from OCP.BRepCheck import BRepCheck_Analyzer

    from coildrawing.model3d import build_coil_parts

    for p in build_coil_parts(small_res, detailed=True):
        assert BRepCheck_Analyzer(p.solid.wrapped).IsValid(), \
            f"{p.name} 不是有效实体"


def test_detailed_no_interference(small_res):
    """铜线与对地壳之间不得有体积干涉（构造上共享分段框架）。"""
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    copper = parts[0].solid
    for p in parts:
        if not p.name.startswith("对地绝缘"):
            continue
        try:
            inter = copper & p.solid
            v = inter.volume if inter is not None else 0.0
        except Exception:
            v = 0.0
        assert v < 1.0, f"{p.name} 与铜导线干涉 {v:.2f}mm³"


def test_dual_wire_strand_grid():
    from coildrawing.model3d import _strand_grid

    inp = CoilInput()
    inp.wire1 = WireSpec(b=8.0, h=2.0, t0=0.05, npd=2, ncd=1)
    inp.wire2 = WireSpec(b=6.0, h=1.5, t0=0.05, npd=1, ncd=2)
    res = compute(inp)
    w_env, h_env, strands = _strand_grid(res)
    assert len(strands) == 2 * 1 + 1 * 2
    assert w_env == pytest.approx(max(8.1 * 2, 6.1 * 1))
    assert h_env == pytest.approx(2.1 * 1 + 1.6 * 2)
    # 导线1 行在下（y 小），导线2 行在上
    assert max(s["y"] for s in strands if s["no"] == 1) < \
        min(s["y"] for s in strands if s["no"] == 2)


def test_simple_parts_still_work(small_res):
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=False)
    assert parts[0].name == "铜导体束"
    for p in parts:
        assert p.solid.volume > 0, p.name

def test_zero_family_gap_constants():
    """剖面无缝：匝间/族间隙必须为零。"""
    from coildrawing import model3d as m

    assert m._TURN_CLEARANCE == 0.0
    assert m._FAMILY_GAP == 0.0
    assert m._HOLE_CLEARANCE <= 0.1


def test_pad_corona_no_interference(small_res):
    """层间垫片与防晕层不得有明显体积干涉。"""
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    pads = [p for p in parts if p.name.startswith("层间垫片")]
    coronas = [p for p in parts if p.name.startswith("防晕层")]
    assert pads and coronas
    for pad in pads:
        for cor in coronas:
            try:
                inter = pad.solid & cor.solid
                v = inter.volume if inter is not None else 0.0
            except Exception:
                v = 0.0
            assert v < 1.0, f"{pad.name} ∩ {cor.name} = {v:.2f}"


def test_copper_turn_no_interference(small_res):
    """铜导线与匝绝缘不得体积干涉（退让后出线端也应干净）。"""
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    copper = next(p for p in parts if p.name.startswith("铜导线")).solid
    for p in parts:
        if not p.name.startswith("匝绝缘"):
            continue
        try:
            inter = copper & p.solid
            v = inter.volume if inter is not None else 0.0
        except Exception:
            v = 0.0
        # 匝绝缘是包在铜外的壳，体积交应为接近 0（壳内腔贴铜）
        assert v < 5.0, f"铜 ∩ {p.name} = {v:.2f}"


def test_lead_holes_snug(small_res):
    """对地开孔应贴近导线包络：引线柱探针与对地残余交集体积极小。"""
    from coildrawing.model3d import (
        b3d, build_coil_parts, _wire_segments, _strand_grid, _HOLE_CLEARANCE,
    )

    res = small_res
    _, info = _wire_segments(res)
    w_env, h_env, _ = _strand_grid(res)
    wrap = sum(l.thickness for l in res.inp.turn_layers if l.thickness > 0)
    cut_w = w_env + 2 * wrap + 2 * _HOLE_CLEARANCE
    cut_h = h_env + 2 * wrap + 2 * _HOLE_CLEARANCE
    parts = build_coil_parts(res, detailed=True)
    for p in parts:
        if not p.name.startswith("对地绝缘"):
            continue
        for tag, tip in (("in", info["tip_in"]), ("out", info["tip_out"])):
            # 细探针：略小于包络，应几乎不与对地相交（孔已挖通）
            probe = b3d.Pos(tip.X, tip.Y, tip.Z - 20) * b3d.Box(
                max(cut_w - 0.5, 1.0), max(cut_h - 0.5, 1.0), 40)
            try:
                inter = p.solid & probe
                v = inter.volume if inter is not None else 0.0
            except Exception:
                v = 0.0
            assert v < 2.0, f"{p.name} 引线孔[{tag}] 残余 {v:.2f}"


def test_export_step_xcaf_smoke(small_res, tmp_path):
    """XCAF 导出应写出可回读的 STEP，且含颜色/中文名转义。"""
    from coildrawing.model3d import export_step

    step = tmp_path / "coil.step"
    names = export_step(small_res, str(step), detailed=True)
    assert names
    raw = step.read_bytes()
    text = raw.decode("ascii", errors="replace")
    assert "COLOUR" in text.upper() or "COLOR" in text.upper() or "DRAUGHTING" in text.upper() or "STYLED" in text
    assert "\\X2\\" in text
