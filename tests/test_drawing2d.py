"""二维端部侧视图的出线端映射。"""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from coildrawing.drawing2d import (  # noqa: E402
    C_COPPER,
    LW_BOLD,
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


def _main_coil_paths(ax):
    return [
        (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
        for line in ax.lines
        if line.get_color() == "k"
        and math.isclose(line.get_linewidth(), LW_BOLD * 1.6)
    ]


def test_axial_leads_and_annotation_mirror_to_selected_z_end():
    fig_pos, ax_pos, res_pos = _draw_axial(True)
    fig_neg, ax_neg, res_neg = _draw_axial(False)
    try:
        lead_pos = next(line for line in ax_pos.lines if line.get_color() == C_COPPER)
        lead_neg = next(line for line in ax_neg.lines if line.get_color() == C_COPPER)

        np.testing.assert_allclose(lead_neg.get_xdata(), -lead_pos.get_xdata())
        np.testing.assert_allclose(lead_neg.get_ydata(), lead_pos.get_ydata())
        assert lead_pos.get_xdata()[0] == pytest.approx(res_pos.l2 / 2 + res_pos.cc)
        assert lead_neg.get_xdata()[0] == pytest.approx(-(res_neg.l2 / 2 + res_neg.cc))

        pos_label = _text(ax_pos, "引线×2")
        neg_label = _text(ax_neg, "引线×2")
        assert neg_label.get_position()[0] == pytest.approx(-pos_label.get_position()[0])
        assert neg_label.get_position()[1] == pytest.approx(pos_label.get_position()[1])

        pos_xlim = ax_pos.get_xlim()
        neg_xlim = ax_neg.get_xlim()
        assert neg_xlim[0] == pytest.approx(-pos_xlim[1])
        assert neg_xlim[1] == pytest.approx(-pos_xlim[0])
    finally:
        plt.close(fig_pos)
        plt.close(fig_neg)


def test_lead_end_choice_does_not_swap_layer_geometry():
    fig_pos, ax_pos, _ = _draw_axial(True)
    fig_neg, ax_neg, _ = _draw_axial(False)
    try:
        # 出线端选择只改变轴向端别，不镜像或交换上/下层几何。
        paths_pos = _main_coil_paths(ax_pos)
        paths_neg = _main_coil_paths(ax_neg)
        assert len(paths_pos) == len(paths_neg) == 5
        for (xp, yp), (xn, yn) in zip(paths_pos, paths_neg, strict=True):
            np.testing.assert_allclose(xn, xp)
            np.testing.assert_allclose(yn, yp)

        s1_pos = _text(ax_pos, "S1=")
        s1_neg = _text(ax_neg, "S1=")
        assert s1_neg.get_position() == pytest.approx(s1_pos.get_position())
    finally:
        plt.close(fig_pos)
        plt.close(fig_neg)
