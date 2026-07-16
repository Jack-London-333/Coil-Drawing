"""二维 nose 几何、轴向匝位与出线端映射。"""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from coildrawing.drawing2d import (  # noqa: E402
    C_COPPER,
    _nose_developed_profiles,
    _project_fillet_end,
    _project_nose_u,
    draw_axial_view,
)
from coildrawing.engine import CoilInput, compute  # noqa: E402


def _draw_axial(lead_end_positive_z: bool):
    inp = CoilInput(
        lead_end_positive_z=lead_end_positive_z,
        rd1_conn=12.0,
        rd2_nonconn=21.0,
        ysc=47.0,
    )
    res = compute(inp)
    fig, ax = plt.subplots()
    draw_axial_view(ax, res)
    return fig, ax, res


def _text(ax, prefix: str):
    return next(item for item in ax.texts if item.get_text().startswith(prefix))


def _lines_with_gid(ax, gid: str):
    return [
        (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
        for line in ax.lines
        if line.get_gid() == gid
    ]


def test_end_projection_shows_crossed_curl():
    """交叉卷环贴圆柱面：rd2 恰在环端相切，端面投影是扁的卷环弧
    （“人”字顶端的小圆环）。"""
    from coildrawing import model3d as m

    inp = CoilInput(rd_nose=18.0, seita3=math.radians(80.0))
    res = compute(inp)
    layout = m._nose_layout(res)
    _, fillets = m._loop_fillets(res)
    rc = inp.rd_nose + res.wa_turn / 2.0
    complete_u = _project_nose_u(
        layout.pos, fillets[2].te, fillets[3].ts, 65)

    # 环平面内半径恒为 Rc；浅螺旋错距只出现在盘面法向上。
    for point in (layout.pos.ts, layout.pos.te):
        v = point - layout.pos.c
        in_plane = v - layout.pos.n * v.dot(layout.pos.n)
        assert in_plane.length == pytest.approx(rc)
    assert layout.pos.radius == pytest.approx(rc)
    assert (fillets[2].te - layout.q2).length == pytest.approx(0.0, abs=1e-9)
    assert (fillets[3].ts - layout.q3).length == pytest.approx(0.0, abs=1e-9)
    # 扫角超过 180°：交叉卷回。
    assert layout.pos.tau > math.pi
    # 环端即 rd2 切点：投影首尾与 q2/q3 重合。
    np.testing.assert_allclose(
        complete_u[0], (layout.q2.X, layout.q2.Y), atol=1e-10)
    np.testing.assert_allclose(
        complete_u[1], (layout.pos.ts.X, layout.pos.ts.Y), atol=1e-10)
    np.testing.assert_allclose(
        complete_u[-2], (layout.pos.te.X, layout.pos.te.Y), atol=1e-10)
    np.testing.assert_allclose(
        complete_u[-1], (layout.q3.X, layout.q3.Y), atol=1e-10)

    # 三维卷环上的采样点到环轴的面内距离恒为 Rc（浅螺旋卷环，
    # 投影为扁椭圆弧）。
    crown = _project_fillet_end(layout.pos, 65)
    assert crown.shape[0] == 65
    points3d = []
    for lam in np.linspace(0.0, 1.0, 33):
        point = layout.pos.point(layout.pos.tau * float(lam))
        v = point - layout.pos.c
        in_plane = v - layout.pos.n * v.dot(layout.pos.n)
        points3d.append(in_plane.length)
    np.testing.assert_allclose(points3d, rc, atol=1e-9)


def test_axial_profiles_use_hbd_and_concentric_spiral_transition():
    """非接线侧同心嵌套；接线侧沿卷环以同心螺旋接相邻 HBD 匝位。"""
    from coildrawing import model3d as m

    inp = CoilInput(n_turns=4, t1=0.15, t3=0.85)
    res = compute(inp)
    assert res.hbd != pytest.approx(res.had)
    constant = _nose_developed_profiles(res, transition=False)
    transitions = _nose_developed_profiles(res, transition=True, count=64)
    layout = m._nose_layout(res)
    rc = inp.rd_nose + res.wa_turn / 2.0
    total = rc * layout.pos.tau

    assert len(constant) == inp.n_turns
    assert len(transitions) == inp.n_turns - 1
    constant_offsets = np.asarray([profile[0, 0] for profile in constant])
    np.testing.assert_allclose(np.diff(constant_offsets), -res.hbd)
    for profile in constant:
        np.testing.assert_allclose(profile[:, 0], profile[0, 0])
        assert profile[0, 1] == pytest.approx(0.0)
        assert profile[-1, 1] == pytest.approx(total)

    for index, profile in enumerate(transitions):
        assert profile[0, 0] == pytest.approx(constant_offsets[index])
        assert profile[-1, 0] == pytest.approx(constant_offsets[index + 1])
        assert profile[-1, 0] - profile[0, 0] == pytest.approx(-res.hbd)
        assert profile[0, 1] == pytest.approx(0.0)
        assert profile[-1, 1] == pytest.approx(total)
        assert np.all(np.diff(profile[:, 1]) > 0)
        # 换位沿整只卷环连续分布（同心螺旋），smoothstep 两端
        # 斜率为零。
        assert abs(profile[1, 0] - profile[0, 0]) < res.hbd / 100
        assert abs(profile[-1, 0] - profile[-2, 0]) < res.hbd / 100
        mid = profile.shape[0] // 2
        assert profile[0, 0] > profile[mid, 0] > profile[-1, 0]


def test_axial_leads_and_annotation_mirror_to_selected_z_end():
    fig_pos, ax_pos, res_pos = _draw_axial(True)
    fig_neg, ax_neg, res_neg = _draw_axial(False)
    try:
        leads_pos = [line for line in ax_pos.lines
                     if line.get_color() == C_COPPER]
        leads_neg = [line for line in ax_neg.lines
                     if line.get_color() == C_COPPER]
        assert len(leads_pos) == len(leads_neg) == 2
        for lead_pos, lead_neg in zip(leads_pos, leads_neg, strict=True):
            np.testing.assert_allclose(
                lead_neg.get_xdata(), -lead_pos.get_xdata())
            np.testing.assert_allclose(
                lead_neg.get_ydata(), lead_pos.get_ydata())
            assert lead_pos.get_xdata()[0] == pytest.approx(
                res_pos.l2 / 2 + res_pos.cc)
            assert lead_neg.get_xdata()[0] == pytest.approx(
                -(res_neg.l2 / 2 + res_neg.cc))
            # 两根引线均在槽内直线部附近，不能被误画到 nose 顶部。
            assert max(lead_pos.get_ydata()) < res_pos.aa1 * 0.1

        pos_label = _text(ax_pos, "槽底侧引线×2")
        neg_label = _text(ax_neg, "槽底侧引线×2")
        assert neg_label.get_position()[0] == pytest.approx(-pos_label.get_position()[0])
        assert neg_label.get_position()[1] == pytest.approx(pos_label.get_position()[1])

        pos_xlim = ax_pos.get_xlim()
        neg_xlim = ax_neg.get_xlim()
        assert neg_xlim[0] == pytest.approx(-pos_xlim[1])
        assert neg_xlim[1] == pytest.approx(-pos_xlim[0])
    finally:
        plt.close(fig_pos)
        plt.close(fig_neg)


def test_lead_end_choice_mirrors_nose_topology_to_selected_end():
    fig_pos, ax_pos, res_pos = _draw_axial(True)
    fig_neg, ax_neg, res_neg = _draw_axial(False)
    try:
        assert res_pos.inp.n_turns == res_neg.inp.n_turns
        n = res_pos.inp.n_turns
        # 选择出线端时，完整换匝 U 组和同位 U 组一起作 Z 镜像；
        # 匝数拓扑不变，也不会把接线/非接线含义留在原轴端。
        for gid, expected in (("nose-connection", n - 1),
                              ("nose-nonconnection", n)):
            paths_pos = _lines_with_gid(ax_pos, gid)
            paths_neg = _lines_with_gid(ax_neg, gid)
            assert len(paths_pos) == len(paths_neg) == expected
            for (xp, yp), (xn, yn) in zip(
                    paths_pos, paths_neg, strict=True):
                np.testing.assert_allclose(xn, -xp)
                np.testing.assert_allclose(yn, yp)

        assert "HBD=" in _text(ax_pos, "接线侧：").get_text()
        assert "卷环同心嵌套" in _text(ax_pos, "非接线侧：").get_text()
    finally:
        plt.close(fig_pos)
        plt.close(fig_neg)
