"""逐匝三维模型的快速回归测试（小匝数，控制耗时）。"""

import math
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
    """槽边匝序反向；两个 nose 的匝位轴沿圆环法向（鼻端中心线）。"""
    import numpy as np

    from coildrawing.model3d import _loop_frames, _radial_of

    frames = _loop_frames(small_res)
    assert frames.f_leg1.y.dot(_radial_of(frames.f_leg1.o)) == \
        pytest.approx(-1.0, abs=1e-10)
    assert frames.f_leg2.y.dot(_radial_of(frames.f_leg2.o)) == \
        pytest.approx(1.0, abs=1e-10)
    # 卷环入口/出口框架 y = 环平面内指向环外的半径方向（同心嵌套
    # 匝位方向），与环法向垂直。
    for frame, arc, point in (
            (frames.f_flat_pos, frames.nose_pos, frames.nose_pos.ts),
            (frames.f_flat_pos_out, frames.nose_pos, frames.nose_pos.te),
            (frames.f_flat_neg, frames.nose_neg, frames.nose_neg.ts),
            (frames.f_flat_neg_out, frames.nose_neg, frames.nose_neg.te)):
        y_expect = np.asarray(tuple((point - arc.c).normalized()))
        y = np.asarray(tuple(frame.y))
        np.testing.assert_allclose(y, y_expect, atol=1e-10)
        assert abs(frame.y.dot(arc.n)) == pytest.approx(0.0, abs=1e-10)


def test_nose_knuckle_has_zero_twist(small_res):
    """rd2 弯角与卷环零扭转——截面过渡全部位于端部斜边贴近鼻部
    的扭转段内，卷环姿态由 rd2 刚性传递。"""
    from coildrawing import model3d as m

    frames = m._loop_frames(small_res)
    fl = frames.fl
    # rd2 弯角出口/入口姿态与卷环端姿态完全一致：鼻部无拧转。
    pairs = (
        (frames.f_q2.at(fl[2].te), frames.f_flat_pos.at(frames.nose_pos.ts)),
        (frames.f_flat_pos_out.at(frames.nose_pos.te),
         frames.f_q3.at(fl[3].ts)),
        (frames.f_q6.at(fl[6].te), frames.f_flat_neg.at(frames.nose_neg.ts)),
        (frames.f_flat_neg_out.at(frames.nose_neg.te),
         frames.f_q7.at(fl[7].ts)),
    )
    for f_a, f_b in pairs:
        assert (f_a.o - f_b.o).length == pytest.approx(0.0, abs=1e-9)
        assert tuple(f_a.y) == pytest.approx(tuple(f_b.y), abs=1e-10)

    # 斜边靠槽口的大部分保持槽内姿态；扭转压缩在贴鼻部的
    # _SLANT_TWIST_ZONE 区段内。
    flat_limit = 1.0 - m._SLANT_TWIST_ZONE
    for key, nose_at_end in (((1, 2), True), ((3, 4), False),
                             ((5, 6), True), ((7, 0), False)):
        f0, f1 = frames.slant_frames[key]
        law, _slope = frames.slant_laws[key]
        assert abs(m._twist_angle(f0, f1)) > math.radians(30.0)
        flat = ([lam / 10 for lam in range(0, int(flat_limit * 10) + 1)]
                if nose_at_end else
                [1.0 - lam / 10 for lam in range(0, int(flat_limit * 10) + 1)])
        f_slot = f0 if nose_at_end else f1
        for lam in flat:
            frame = m._twist_frame_at(f0, f1, lam, law)
            assert tuple(frame.y) == pytest.approx(
                tuple(f_slot.y), abs=1e-9), (key, lam)


def test_end_arm_sections_respect_slot_pitch_arc_limit():
    """任意端面同心圆与线圈束截面的交弧必须小于槽距 2πr/NS
    （斜边全程——含扭转段——都要满足）。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(n_turns=8, seita3=math.radians(80.0)))
    lf = m._loop_frames(res)
    inp = res.inp
    bundle_h = (inp.n_turns - 1) * res.hbd + res.hc / inp.n_turns

    frames = []
    for key in ((1, 2), (3, 4), (5, 6), (7, 0)):
        f0, f1 = lf.slant_frames[key]
        law, _slope = lf.slant_laws[key]
        frames += [m._twist_frame_at(f0, f1, lam / 20, law)
                   for lam in range(21)]

    for frame in frames:
        # 圆周向占宽（束宽沿 x + 束高沿 y 的圆周向投影）小于槽距。
        radius = math.hypot(frame.o.X, frame.o.Y)
        circ = m.b3d.Vector(frame.o.Y, -frame.o.X, 0).normalized()
        footprint = (abs(frame.x.dot(circ)) * res.wc +
                     abs(frame.y.dot(circ)) * bundle_h)
        assert footprint < 2 * math.pi * radius / inp.ns, \
            (footprint, 2 * math.pi * radius / inp.ns)


def test_four_turn_material_mapping_matches_formed_coil():
    """上层槽口→槽底=4..1，下层槽口→槽底=1..4。"""
    from coildrawing.model3d import _loop_frames

    inp = CoilInput(n_turns=4)
    res = compute(inp)
    frames = _loop_frames(res)
    dys = [res.had * (i - 1.5) for i in range(4)]

    def radial_position(frame, dy):
        point = frame.o + frame.y * dy
        return (point.X ** 2 + point.Y ** 2) ** 0.5

    upper = sorted(range(4), key=lambda i: radial_position(frames.f_leg1, dys[i]))
    lower = sorted(range(4), key=lambda i: radial_position(frames.f_leg2, dys[i]))
    assert [i + 1 for i in upper] == [4, 3, 2, 1]
    assert [i + 1 for i in lower] == [1, 2, 3, 4]


def test_nose_is_crossed_curl_with_patent_centerline_radius():
    """鼻端是交叉卷环：中心线半径 Rc=RD+WA/2，扫角 180°+2β，
    环顶方向（鼻端中心线）与径向直径成 seita3、朝轴向端外。"""
    from coildrawing import model3d as m

    inp = CoilInput(rd_nose=18.0, seita3=math.radians(80.0))
    res = compute(inp)
    layout = m._nose_layout(res)
    radial = m.b3d.Vector(0, 1, 0)
    center_radius = inp.rd_nose + res.wa_turn / 2
    sweep = math.pi + 2.0 * m._NOSE_CROSS_BETA
    for arc, z_sign in ((layout.pos, 1.0), (layout.neg, -1.0)):
        assert (arc.ts - arc.c).length == pytest.approx(center_radius, abs=1e-9)
        assert (arc.ma - arc.c).length == pytest.approx(center_radius, abs=1e-9)
        assert (arc.te - arc.c).length == pytest.approx(center_radius, abs=1e-9)
        assert center_radius - res.wa_turn / 2 == pytest.approx(
            inp.rd_nose, abs=1e-10)
        # 扫角超过 180°（交叉卷回，观感接近闭合的圆环）。
        assert arc.tau == pytest.approx(sweep, abs=1e-12)
        assert arc.tau > math.pi
        # 环顶方向 = 鼻端中心线：与径向直径成 seita3、朝轴向端外。
        axis = (arc.ma - arc.c).normalized()
        assert axis.X == pytest.approx(0.0, abs=1e-10)
        angle = math.acos(max(-1.0, min(1.0, axis.dot(radial))))
        assert angle == pytest.approx(inp.seita3, abs=1e-10)
        assert z_sign * axis.Z > 0
        # 环平面法向与鼻端中心线垂直（环贴着圆柱面、含偏航倾斜）。
        assert arc.n.dot(axis) == pytest.approx(0.0, abs=1e-10)
        # 弧端点/中点自洽。
        end_check = arc.c + m._rotv(arc.ts - arc.c, arc.n, arc.tau)
        mid_check = arc.c + m._rotv(arc.ts - arc.c, arc.n, arc.tau / 2)
        assert (end_check - arc.te).length < 1e-9
        assert (mid_check - arc.ma).length < 1e-9


@pytest.mark.parametrize("degrees", [70.0, 80.0, 90.0])
def test_seita3_rotates_nose_centerline(degrees):
    """70–90° 必须真实改变鼻端中心线方位，而不是只改输入框。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(seita3=math.radians(degrees)))
    nose = m._nose_layout(res).pos
    radial = m.b3d.Vector(0, 1, 0)
    axis = (nose.ma - nose.c).normalized()
    angle = math.degrees(math.acos(max(-1.0, min(1.0, axis.dot(radial)))))
    assert angle == pytest.approx(degrees, abs=1e-9)
    assert axis.X == pytest.approx(0.0, abs=1e-10)


def test_nose_turns_are_concentric_with_hbd_radial_pitch():
    """多匝在环平面内同心嵌套：环心共享、半径相差 HBD。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(n_turns=8, rd_nose=15.0, t3=0.30,
                            seita3=math.radians(80.0)))
    assert res.hbd != pytest.approx(res.had)
    frames = m._loop_frames(res)
    rc = res.inp.rd_nose + res.wa_turn / 2
    dys = [res.hbd * (i - 3.5) for i in range(8)]
    cases = (
        (frames.nose_pos, frames.f_flat_pos),
        (frames.nose_neg, frames.f_flat_neg),
    )
    for arc, entry in cases:
        # 匝位轴 = 环平面内指向环外的半径方向。
        assert tuple(entry.y) == pytest.approx(
            tuple((arc.ts - arc.c).normalized()), abs=1e-10)
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            y_dir = m._rotv((arc.ts - arc.c).normalized(), arc.n,
                            arc.tau * lam)
            for d_left, d_right in zip(dys, dys[1:]):
                left = arc.c + y_dir * (rc + d_left)
                right = arc.c + y_dir * (rc + d_right)
                offset = right - left
                # 同心圆环：同弧位相邻匝严格沿环半径方向相差 HBD。
                assert offset.length == pytest.approx(res.hbd, abs=1e-10)
                assert (left - arc.c).length == pytest.approx(
                    rc + d_left, abs=1e-10)


def test_nose_transition_is_concentric_spiral_g1_at_both_ends():
    """卷环换匝是同心螺旋：截面沿旋转环半径方向相差 HBD，
    两端与 rd2 圆角 G1 对接。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(n_turns=2, rd_nose=15.0, t3=0.30,
                            seita3=math.radians(80.0)))
    frames = m._loop_frames(res)
    fl = frames.fl
    arc = frames.nose_pos
    q2, q3 = fl[2].te, fl[3].ts
    d0, d1 = -res.hbd / 2, res.hbd / 2

    transition = m._nose_transition(arc, d0, d1)
    shifted = m._nose_transition(arc, d0 + res.hbd, d1 + res.hbd)
    first, last = transition.sections[0], transition.sections[-1]
    t0 = arc.n.cross(arc.ts - arc.c).normalized()
    t1 = arc.n.cross(arc.te - arc.c).normalized()

    # 两端与 rd2 圆角端面逐点对接：位置沿环半径方向偏移，切向等于
    # 解析切向。
    assert tuple(first.o) == pytest.approx(
        tuple(q2 + frames.f_q2.y * d0), abs=1e-9)
    assert tuple(last.o) == pytest.approx(
        tuple(q3 + frames.f_q3.y * d1), abs=1e-9)
    assert tuple(first.t) == pytest.approx(tuple(t0), abs=1e-10)
    assert tuple(last.t) == pytest.approx(tuple(t1), abs=1e-10)
    assert tuple(first.ey) == pytest.approx(tuple(frames.f_q2.y), abs=1e-10)
    assert tuple(last.ey) == pytest.approx(tuple(frames.f_q3.y), abs=1e-10)

    plane_normal = arc.n.normalized()
    for section, parallel in zip(transition.sections, shifted.sections):
        # 相邻换匝在每个截面都严格相差一个沿环半径方向的 HBD。
        offset = parallel.o - section.o
        assert offset.length == pytest.approx(res.hbd, abs=1e-9)
        assert tuple(offset.normalized()) == pytest.approx(
            tuple(section.ey), abs=1e-9)
        assert tuple(parallel.ey) == pytest.approx(tuple(section.ey), abs=1e-10)
        # 截面材料 y 轴 = 环半径方向（在环平面内），ex = ±环平面法向。
        assert section.ey.dot(plane_normal) == pytest.approx(0.0, abs=1e-10)
        assert abs(section.ex.dot(plane_normal)) == pytest.approx(
            1.0, abs=1e-10)
        # 换匝曲线是从 Rc+d0 到 Rc+d1 的同心螺旋。
        r_here = (section.o - arc.c).length
        assert min(d0, d1) - 1e-9 <= \
            r_here - (arc.ts - arc.c).length <= max(d0, d1) + 1e-9


def test_crossed_curl_centerline_is_exactly_llm():
    """rd2 恰在环端 G1 相切；环位由 LLM 反解，闭合中心线严格等于
    专利平均匝长。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(seita3=math.radians(80.0)))
    layout = m._nose_layout(res)
    _corners, fillets = m._loop_fillets(res)
    # rd2 圆角恰在环的入口/出口切点结束/开始（无额外直臂）。
    assert tuple(fillets[2].te) == pytest.approx(tuple(layout.q2), abs=1e-9)
    assert tuple(fillets[3].ts) == pytest.approx(tuple(layout.q3), abs=1e-9)
    assert tuple(fillets[6].te) == pytest.approx(tuple(layout.q6), abs=1e-9)
    assert tuple(fillets[7].ts) == pytest.approx(tuple(layout.q7), abs=1e-9)
    assert tuple(layout.q2) == pytest.approx(tuple(layout.pos.ts), abs=1e-12)
    assert tuple(layout.q3) == pytest.approx(tuple(layout.pos.te), abs=1e-12)

    # rd2 与环 G1 相切：圆角出口切向 == 环入口切向。
    tangent_in = layout.pos.n.cross(
        layout.pos.ts - layout.pos.c).normalized()
    corner_out = fillets[2].n.cross(layout.q2 - fillets[2].c).normalized()
    assert tuple(corner_out) == pytest.approx(tuple(tangent_in), abs=1e-9)

    wire, *_rest = m.build_centerline(res)
    assert wire.length == pytest.approx(res.llm, abs=1e-6)


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
        sum(solid.volume for solid in solids), rel=2e-8, abs=1e-3)


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


def test_reported_n8_lead_doglegs_have_no_adjacent_turn_interference():
    """v202607131115 的两根引线不得再穿入 T7/T2 铜线或匝间云母。"""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from coildrawing import model3d as m

    inp = CoilInput(n_turns=8, f_nose=40.0,
                    seita3=math.radians(80.0),
                    rd_nose=15.0, r_bend_nose=30.0,
                    lead_bend_r=15.0, ysc=45.0)
    res = compute(inp)
    segments, info = m._wire_segments(res)
    ranges = info["turn_ranges"]
    w, h, _ = m._strand_grid(res)
    wrap = sum(layer.thickness for layer in inp.turn_layers
               if layer.thickness > 0)
    outer_w = w + 2 * wrap
    outer_h = min(h + 2 * wrap, res.had)

    def overlaps(left, right):
        a, b = left.bounding_box(), right.bounding_box()
        return (a.min.X < b.max.X and b.min.X < a.max.X and
                a.min.Y < b.max.Y and b.min.Y < a.max.Y and
                a.min.Z < b.max.Z and b.min.Z < a.max.Z)

    def common_volume(left, right):
        op = BRepAlgoAPI_Common(left.wrapped, right.wrapped)
        op.SetRunParallel(True)
        op.Build()
        assert op.IsDone(), "OCCT Common 未完成，不能把布尔异常当作零干涉"
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(op.Shape(), props)
        return props.Mass()

    lead_count = len(info["lead_in"][0])
    cases = [
        (range(ranges[0][0], ranges[0][0] + lead_count), ranges[1]),
        (range(ranges[-1][1] - lead_count, ranges[-1][1]), ranges[-2]),
    ]
    for lead_ids, (adj_start, adj_end, _material_i) in cases:
        for lead_i in lead_ids:
            lead_copper = m._seg_solid(segments[lead_i], w, h)
            lead_mica = m._seg_ring(
                segments[lead_i], w, h, outer_w, outer_h)
            for adjacent_i in range(adj_start, adj_end):
                adjacent_copper = m._seg_solid(
                    segments[adjacent_i], w, h)
                adjacent_mica = m._seg_ring(
                    segments[adjacent_i], w, h, outer_w, outer_h)
                for left, right in (
                    (lead_copper, adjacent_copper),
                    (lead_mica, adjacent_copper),
                    (lead_mica, adjacent_mica),
                ):
                    if left is None or right is None or not overlaps(left, right):
                        continue
                    assert common_volume(left, right) < 1e-6

    # 每次接线侧 nose 换匝都由一个 MultiLoft 承担；相邻换匝段的
    # 中心在每个截面都恰差一个沿环半径方向的 HBD（同心螺旋，
    # 真实间距不缩短）。
    transition_ids = [start for start, _end, _material in ranges[1:]]
    assert all(isinstance(segments[index], m._MultiLoft)
               for index in transition_ids)
    for left_i, right_i in zip(transition_ids, transition_ids[1:]):
        left_transition = segments[left_i]
        right_transition = segments[right_i]
        for left_section, right_section in zip(
                left_transition.sections, right_transition.sections):
            offset = right_section.o - left_section.o
            assert offset.length == pytest.approx(res.hbd, abs=1e-9)
            assert abs(offset.normalized().dot(left_section.ey)) == \
                pytest.approx(1.0, abs=1e-9)
            assert tuple(right_section.ey) == pytest.approx(
                tuple(left_section.ey), abs=1e-10)
        left_copper = m._seg_solid(segments[left_i], w, h)
        right_copper = m._seg_solid(segments[right_i], w, h)
        left_mica = m._seg_ring(
            segments[left_i], w, h, outer_w, outer_h)
        right_mica = m._seg_ring(
            segments[right_i], w, h, outer_w, outer_h)
        assert common_volume(left_copper, right_copper) < 1e-6
        assert common_volume(left_mica, right_mica) < 1e-6


def test_lead_bend_radius_is_rejected_instead_of_silently_enlarged():
    """引线圆弧半径必须忠实采用输入；过小值应报错而不是暗改为2mm。"""
    from coildrawing import model3d as m

    inp = CoilInput(lead_bend_r=1.0)
    with pytest.raises(ValueError, match="引线错位圆弧几何不可行"):
        m._wire_segments(compute(inp))


def test_nose_connection_corners_use_rd2_not_nose_rd():
    """斜边—nose 圆角应取 rd2；RD 只定义 nose 本体半径/展开宽度。"""
    from coildrawing import model3d as m

    inp = CoilInput(rd_nose=15.0, r_bend_nose=30.0)
    _, fillets = m._loop_fillets(compute(inp))
    for index in (2, 3, 6, 7):
        assert (fillets[index].ts - fillets[index].c).length == \
            pytest.approx(30.0, abs=1e-8)


@pytest.mark.parametrize("degrees", [70.0, 80.0, 90.0])
def test_common_large_seita3_nose_is_compact_and_exactly_tangent(degrees):
    """70–90° 常用区的 rd2→Q 肩部必须紧凑，Q→P 再形成直鼻臂。"""
    from coildrawing import model3d as m

    inp = CoilInput(f_nose=40.0, rd_nose=15.0,
                    r_bend_nose=30.0,
                    seita3=math.radians(degrees))
    res = compute(inp)
    layout = m._nose_layout(res)
    corners, fillets = m._loop_fillets(res)

    # rd2 不得被空间限幅偷偷缩小；圆角切点必须恰好落在环端 Q
    # （与卷环 G1 相切）。
    for index in (2, 3, 6, 7):
        assert (fillets[index].ts - fillets[index].c).length == \
            pytest.approx(inp.r_bend_nose, abs=1e-8)
    assert (fillets[2].te - layout.q2).length < 1e-8
    assert (fillets[3].ts - layout.q3).length < 1e-8
    assert (fillets[6].te - layout.q6).length < 1e-8
    assert (fillets[7].ts - layout.q7).length < 1e-8

    paths = [
        (corners[1][0], layout.p2, layout.q2),
        (layout.q3, layout.p3, corners[4][0]),
        (corners[5][0], layout.p6, layout.q6),
        (layout.q7, layout.p7, corners[0][0]),
    ]
    for start, virtual, end in paths:
        chord = end - start
        # 虚拟角点在弦方向上必须严格落在两端之间；两段
        # 投影均向前，这是“无回头”的直接几何门禁。
        assert (virtual - start).dot(chord) > 0
        assert (end - virtual).dot(chord) > 0
        ratio = ((virtual - start).length + (end - virtual).length) / \
            (end - start).length
        assert ratio <= 1.5


def test_infeasible_nose_parameters_fail_instead_of_growing_hairpin():
    """无解参数须明确拒绝，不得恢复米级虚拟切线的旧行为。"""
    from coildrawing import model3d as m

    inp = CoilInput(seita3=math.radians(80.0), r_bend_nose=100.0)
    with pytest.raises(ValueError,
                       match="参数组合几何不可行|平均匝长|净距需求"):
        m._nose_layout(compute(inp))


def test_nose_rejects_locally_compact_but_backward_virtual_corner():
    """仅凭转角/路径比仍可能轻微回头；运行时也必须检查弦向单调性。"""
    from coildrawing import model3d as m

    inp = CoilInput(f_nose=200.0, rd_nose=15.0,
                    r_bend_nose=30.0,
                    seita3=math.radians(120.0))
    with pytest.raises(ValueError,
                       match="参数组合几何不可行|平均匝长|净距需求"):
        m._nose_layout(compute(inp))


def test_f_nose_raises_both_noses_without_moving_slot_legs():
    """F 应让 +Z/-Z 两个 nose 一起朝槽底抬高，槽内直边保持不动。"""
    from coildrawing import model3d as m

    low_res = compute(CoilInput(f_nose=5.0))
    high_res = compute(CoilInput(f_nose=32.0))
    low, _ = m._loop_fillets(low_res)
    high, _ = m._loop_fillets(high_res)
    low_nose = m._nose_layout(low_res)
    high_nose = m._nose_layout(high_res)

    def radius(corner):
        point = corner[0]
        return (point.X ** 2 + point.Y ** 2) ** 0.5

    # F 固定地把环的基准径向位置 rn 朝槽底抬高 27mm（环位 shift
    # 沿鼻端中心线另由 LLM 反解，需先剔除其径向分量）；槽内直边
    # 保持不动。
    low_shift = m._nose_pose(low_res)[0]
    high_shift = m._nose_pose(high_res)[0]
    cos3 = math.cos(low_res.inp.seita3)
    for low_arc, high_arc in ((low_nose.pos, high_nose.pos),
                              (low_nose.neg, high_nose.neg)):
        low_rn = low_arc.c.Y + low_shift * cos3
        high_rn = high_arc.c.Y + high_shift * cos3
        assert high_rn - low_rn == pytest.approx(27.0, abs=1e-8)
        assert high_arc.c.X == pytest.approx(low_arc.c.X, abs=1e-8)
    for index in (0, 1, 4, 5):
        assert radius(high[index]) == pytest.approx(
            radius(low[index]), abs=1e-8)


def test_wire_segments_have_no_centerline_gaps():
    """HAD→HBD 变距后的每个相邻解析分段仍须在同一点相接。"""
    from coildrawing import model3d as m

    inp = CoilInput()
    inp.n_turns = 4
    inp.t3 = 0.30
    res = compute(inp)
    assert res.hbd != pytest.approx(res.had)
    segments, _ = m._wire_segments(res)

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
        if isinstance(segment, m._MultiLoft):
            return segment.sections[0].o, segment.sections[-1].o
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
    turn_names = [n for n in names if n.startswith("匝绝缘1")]
    # 第一材料匝没有换匝段，仍是一件；其余每匝均拆成
    # 主体 + 左/右/上/下四条带。
    assert len(turn_names) == 1 + 5 * (small_res.inp.n_turns - 1)
    assert any(n.endswith("-主体") for n in turn_names)
    assert any(n.endswith("-换匝段-左侧条带") for n in turn_names)
    assert sum(1 for n in names if n.startswith("对地绝缘")) == 2
    assert sum(1 for n in names if n.startswith("防晕层")) == 2
    assert sum(1 for n in names if n.startswith("层间垫片")) == 2

    for p in parts:
        assert p.solid.volume > 0, p.name
        assert len(p.solid.solids()) == 1, p.name

    # 铜导线体积 ≈ 截面 × 解析平均匝长。LLM 不展开立面 nose 的
    # 同心半径及光顺换匝细节，因此这里只作量级校核。
    copper = parts[0].solid
    w = small_res.inp.wire1
    approx_len = 2 * small_res.llm + 2 * small_res.inp.ysc
    vol_expect = w.b * w.h * approx_len
    assert abs(copper.volume - vol_expect) / vol_expect < 0.10


def test_detailed_parts_valid_solids(small_res):
    """全部部件必须是有效实体——无效实体在 SolidWorks 中会被降级为
    “曲面实体”（v202607110207 问题二的根源，出线端自相交所致）。"""
    from OCP.BRepCheck import BRepCheck_Analyzer

    from coildrawing.model3d import build_coil_parts

    for p in build_coil_parts(small_res, detailed=True):
        assert BRepCheck_Analyzer(p.solid.wrapped, True).IsValid(), \
            f"{p.name} 不是有效实体"


def test_turn_mica_split_leaves_are_strict_low_tolerance_solids():
    """每个 split-5 叶节点都必须是严格有效的低公差 SOLID。"""
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from coildrawing import model3d as m

    inp = CoilInput(
        lc=600.0, n_turns=4, f_nose=40.0,
        seita3=math.radians(80.0),
        rd_nose=18.0, r_bend_slot=50.0, r_bend_nose=50.0,
        lead_bend_r=15.0,
    )
    res = compute(inp)
    segs, info = m._wire_segments(res)
    w_env, h_env, _ = m._strand_grid(res)
    layer = inp.turn_layers[0]
    w2 = w_env + 2 * layer.thickness
    h2 = min(h_env + 2 * layer.thickness, res.had)

    assert len(info["turn_ranges"]) == inp.n_turns
    for start, end, material_i in info["turn_ranges"]:
        leaves = m._turn_mica_shell_parts(
            segs[start:end], w_env, h_env, w2, h2)
        assert len(leaves) == (1 if material_i == inp.n_turns - 1 else 5)
        for suffix, shell in leaves:
            shape = shell.wrapped
            label = f"第{material_i + 1}匝-{suffix or '完整匝'}"
            assert shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID, label
            assert len(shell.solids()) == 1, label
            assert shell.volume > 0, label
            assert BRepCheck_Analyzer(shape, True).IsValid(), label

            edges = TopExp_Explorer(shape, TopAbs_EDGE)
            while edges.More():
                edge = TopoDS.Edge_s(edges.Current())
                assert BRep_Tool.Tolerance_s(edge) < 1e-4, label
                edges.Next()


def test_strand_self_insulation_multiloft_is_split_into_solids():
    """导线自身绝缘也须拆开光顺换匝环，不能在 SW 中退化为曲面体。"""
    from itertools import combinations

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_ShapeEnum

    from coildrawing import model3d as m

    inp = CoilInput(
        n_turns=2,
        wire1=WireSpec(b=8.2, h=3.35, t0=0.05, npd=1, ncd=1),
    )
    res = compute(inp)
    segs, _info = m._wire_segments(res)
    _w_env, _h_env, strands = m._strand_grid(res)
    strand = strands[0]
    leaves = m._continuous_ring_shell_parts(
        segs, strand["b"], strand["h"],
        strand["bi"], strand["hi"], strand["x"], strand["y"])

    labels = [label for label, _shape in leaves]
    assert labels == [
        "主体段1",
        "换匝1-左侧条带",
        "换匝1-右侧条带",
        "换匝1-上侧条带",
        "换匝1-下侧条带",
        "主体段2",
    ]
    assert len(labels) == len(set(labels))
    for label, shape in leaves:
        assert shape.wrapped.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID, label
        assert len(shape.solids()) == 1, label
        assert shape.volume > 0, label
        assert BRepCheck_Analyzer(shape.wrapped, True).IsValid(), label

    # 拆分前后总体积守恒；拆出的六件以及铜芯之间仅共享边界。
    reference = m._continuous_path_shell(
        segs, strand["b"], strand["h"],
        strand["bi"], strand["hi"], strand["x"], strand["y"])
    assert sum(shape.volume for _label, shape in leaves) == pytest.approx(
        reference.volume, abs=1e-4)

    def common_volume(left, right):
        common = BRepAlgoAPI_Common(left.wrapped, right.wrapped)
        common.Build()
        assert common.IsDone()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(common.Shape(), props)
        return props.Mass()

    for (left_label, left), (right_label, right) in combinations(leaves, 2):
        assert common_volume(left, right) < 1e-6, \
            f"{left_label} ∩ {right_label}"

    copper = m._join([
        m._seg_solid(segment, strand["b"], strand["h"],
                     strand["x"], strand["y"])
        for segment in segs
    ])
    for label, shape in leaves:
        assert common_volume(copper, shape) < 1e-6, f"铜 ∩ {label}"


def test_reported_t7_mica_split5_preserves_volume_and_has_no_interference():
    """精确问题配置的 T7 云母必须以五个实体保持原光顺总体积。

    五件之间、与本匝铜以及相邻 T8/T6 云母只能边界贴合，不得有任何
    正体积 Common。
    """
    from itertools import combinations

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_ShapeEnum

    from coildrawing import model3d as m

    inp = CoilInput(
        lc=1250.0, n_turns=8, f_nose=40.0,
        seita3=math.radians(80.0), rd_nose=15.0,
        r_bend_slot=30.0, r_bend_nose=30.0,
        lead_bend_r=15.0, ysc=45.0,
    )
    res = compute(inp)
    segs, info = m._wire_segments(res)
    ranges = {material_i: (start, end)
              for start, end, material_i in info["turn_ranges"]}
    start, end = ranges[6]  # T7
    w1, h1, _ = m._strand_grid(res)
    layer = inp.turn_layers[0]
    w2 = w1 + 2 * layer.thickness
    h2 = min(h1 + 2 * layer.thickness, res.had)

    leaves = m._turn_mica_shell_parts(segs[start:end], w1, h1, w2, h2)
    assert [suffix for suffix, _shape in leaves] == [
        "主体",
        "换匝段-左侧条带",
        "换匝段-右侧条带",
        "换匝段-上侧条带",
        "换匝段-下侧条带",
    ]
    for suffix, shape in leaves:
        assert shape.wrapped.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID, suffix
        assert len(shape.solids()) == 1, suffix
        assert shape.volume > 0, suffix
        assert BRepCheck_Analyzer(shape.wrapped, True).IsValid(), suffix

    reference = m._continuous_path_shell(segs[start:end], w1, h1, w2, h2)
    split_volume = sum(shape.volume for _suffix, shape in leaves)
    assert split_volume == pytest.approx(reference.volume, abs=1e-4)

    def common_volume(left, right):
        op = BRepAlgoAPI_Common(left.wrapped, right.wrapped)
        op.SetRunParallel(True)
        op.Build()
        assert op.IsDone(), "OCCT Common 未完成"
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(op.Shape(), props)
        return props.Mass()

    # 五件之间只允许共享边界。
    for (_left_name, left), (_right_name, right) in combinations(leaves, 2):
        assert common_volume(left, right) < 1e-6

    combined = m.b3d.Compound(children=[shape for _suffix, shape in leaves])
    assert BRepCheck_Analyzer(combined.wrapped, True).IsValid()

    # split-5 合体不得侵入同一路径铜芯或相邻材料匝。
    copper = m._join([m._seg_solid(seg, w1, h1)
                      for seg in segs[start:end]])
    assert common_volume(combined, copper) < 1e-6
    for adjacent_i in (7, 5):  # T8、T6
        adj_start, adj_end = ranges[adjacent_i]
        adjacent = m._continuous_path_shell(
            segs[adj_start:adj_end], w1, h1, w2, h2)
        assert common_volume(combined, adjacent) < 1e-6


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
        except Exception as exc:
            pytest.fail(f"{p.name} 与铜导线布尔交失败: {exc}")
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
            except Exception as exc:
                pytest.fail(f"{pad.name} 与 {cor.name} 布尔交失败: {exc}")
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
        except Exception as exc:
            pytest.fail(f"铜导线与 {p.name} 布尔交失败: {exc}")
        # 匝绝缘是包在铜外的壳，体积交应为接近 0（壳内腔贴铜）
        assert v < 1e-6, f"铜 ∩ {p.name} = {v:.6g}"


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
            except Exception as exc:
                pytest.fail(f"{p.name} 引线孔[{tag}] 布尔交失败: {exc}")
            assert v < 2.0, f"{p.name} 引线孔[{tag}] 残余 {v:.2f}"


@pytest.mark.parametrize(
    ("overrides", "detailed", "message"),
    [
        ({"t3": 0.25}, True, "T1=0.15.*T3=0.25"),
        ({"t4": 1.35}, False, "T2=1.1.*T4=1.35"),
    ],
)
def test_3d_rejects_unsupported_variable_insulation_sections(
        overrides, detailed, message):
    """恒截面绝缘不能冒充端部变厚/变薄实体，精细与简化模型都应拒绝。"""
    from coildrawing import model3d as m

    res = compute(CoilInput(n_turns=2, **overrides))
    with pytest.raises(ValueError, match=message):
        m.build_coil_parts(res, detailed=detailed)


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
