"""线圈大样图（2D）绘制：端面投影图 + 梯形梭形大样。

只依赖 matplotlib，供 UI 预览和 PNG 导出。
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

from .engine import CoilResult

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _dim(ax, p1, p2, text, offset=(0, 0), color="0.25", fs=8):
    """简易尺寸标注：双箭头 + 文本。"""
    arrow = FancyArrowPatch(p1, p2, arrowstyle="<->", mutation_scale=10,
                            color=color, lw=0.8, shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)
    mx, my = (p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def draw_end_view(ax, res: CoilResult) -> None:
    """端面投影图：定子内圆、上/下层边截面、端部斜边 ek、鼻端。"""
    inp = res.inp
    r_bore = inp.d2 / 2

    span = res.fai1 * 1.9 + 0.25
    th = np.linspace(-span, span, 200)

    # 定子内圆与上下层边所在圆
    circles = [
        (r_bore, dict(ls="--", lw=1.0, color="0.55"), f"定子内圆 D2/2={r_bore:.1f}"),
        (res.rr1, dict(ls=":", lw=0.9, color="tab:blue"), f"RR1={res.rr1:.1f}"),
        (res.rr2, dict(ls=":", lw=0.9, color="tab:red"), f"RR2={res.rr2:.1f}"),
    ]
    for i, (r, style, label) in enumerate(circles):
        ax.plot(np.sin(th) * r, np.cos(th) * r, **style)
        ax.text(0.02, 0.30 - i * 0.055, label, fontsize=7.5, color=style["color"],
                ha="left", va="top", transform=ax.transAxes)

    def rect_at(radius_bottom, theta, w, h, color, label):
        """在角度 theta、底半径 radius_bottom 处画径向放置的矩形截面。"""
        er = np.array([math.sin(theta), math.cos(theta)])   # 径向单位矢量
        et = np.array([math.cos(theta), -math.sin(theta)])  # 周向单位矢量
        c0 = er * radius_bottom
        pts = [c0 - et * w / 2, c0 + et * w / 2,
               c0 + et * w / 2 + er * h, c0 - et * w / 2 + er * h, c0 - et * w / 2]
        xs, ys = zip(*pts)
        ax.fill(xs, ys, color=color, alpha=0.85, ec="k", lw=0.6, zorder=5)
        if label:
            tip = er * (radius_bottom + h)
            ax.annotate(label, xy=tip, xytext=(tip[0], tip[1] + 28),
                        fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

    # 同槽上下层（节距两端）：上层边在 -fai1，下层边在 +fai2
    rect_at(res.rr1, -res.fai1, res.w_slot, res.h_slot, "#e0a050", "上层边(本线圈)")
    rect_at(res.rr2, +res.fai2, res.w_slot, res.h_slot, "#b06030", "下层边(本线圈)")

    # 相邻线圈示意（淡色）
    slot_ang = 2 * math.pi / inp.ns
    for k in (1, 2):
        rect_at(res.rr1, -res.fai1 + k * slot_ang, res.w_slot, res.h_slot, "#f0d0a8", "")
        rect_at(res.rr2, +res.fai2 - k * slot_ang, res.w_slot, res.h_slot, "#d8b090", "")

    # 端部斜边投影 ek（上层）与鼻端
    e = np.array([res.xe, res.ye])
    k_ = np.array([res.xk, res.yk])
    ax.plot([e[0], k_[0]], [e[1], k_[1]], color="tab:blue", lw=1.6,
            label="上层端部斜边投影 e-k")
    e2 = np.array([res.rr2 * math.sin(res.fai2), res.rr2 * math.cos(res.fai2)])
    rk2 = res.rr2 + inp.f_nose + res.hc * math.sin(inp.seita3)
    k2 = np.array([-res.xk, math.sqrt(max(rk2 ** 2 - res.xk ** 2, 0.0))])
    ax.plot([e2[0], k2[0]], [e2[1], k2[1]], color="tab:red", lw=1.6,
            label="下层端部斜边投影")
    # 鼻端圆
    nose_c = (k_ + k2) / 2
    nose_r = inp.rd_nose + res.wd / 2
    tt = np.linspace(0, 2 * math.pi, 60)
    ax.plot(nose_c[0] + nose_r * np.cos(tt), nose_c[1] + nose_r * np.sin(tt),
            color="0.3", lw=0.9, ls="-")
    ax.text(nose_c[0], nose_c[1] + nose_r + 6, f"鼻端 RD={inp.rd_nose:.0f}",
            fontsize=7, ha="center")

    # 张角标注
    r_ann = res.rr1 * 0.55
    arc = Arc((0, 0), 2 * r_ann, 2 * r_ann, angle=90,
              theta1=-math.degrees(res.fai2), theta2=math.degrees(res.fai1),
              color="0.3", lw=0.9)
    ax.add_patch(arc)
    ax.text(0, r_ann - 22, f"张角 fai={res.fai:.4f} rad", fontsize=8, ha="center")

    ax.set_title("端面投影图（沿轴向看）", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(loc="lower center", fontsize=7)
    ax.set_xlim(-res.rr2 * math.sin(span) * 1.15, res.rr2 * math.sin(span) * 1.15)
    lo = min(res.ye, res.yk, e2[1]) - 60
    hi = res.rr2 + inp.f_nose + 70
    ax.set_ylim(lo, hi)
    ax.axis("off")


def draw_lozenge(ax, res: CoilResult) -> None:
    """梯形梭形大样（绕线模/涨型前的平面形状，示意 + 尺寸）。"""
    l4, l5, h, r = res.l4, res.l5, res.h_lozenge, res.rd1_lozenge
    if h <= 0:
        h = max(res.cc, 1.0)

    # 以下底中心为原点：下底 L5，上底 L4，高 h，四角小圆角 r
    x_bot, x_top = l5 / 2, l4 / 2
    pts = [(-x_bot, 0), (x_bot, 0), (x_top, h), (-x_top, h)]
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.fill(xs, ys, color="#f7e3b0", alpha=0.6, ec="none")
    ax.plot(xs, ys, color="#a06020", lw=2.0)

    # 尺寸标注
    _dim(ax, (-x_bot, -h * 0.35), (x_bot, -h * 0.35), f"L5 = {l5:.1f}")
    _dim(ax, (-x_top, h * 1.35), (x_top, h * 1.35), f"L4 = {l4:.1f}")
    _dim(ax, (x_bot + l5 * 0.03, 0), (x_bot + l5 * 0.03, h), f"h_ = {res.h_lozenge:.1f}",
         offset=(l5 * 0.05, 0))
    mx, my = (x_bot + x_top) / 2, h / 2
    ang = math.degrees(math.atan2(h, x_top - x_bot))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    ax.text(mx - l5 * 0.02, my + h * 0.12, f"斜边 XX1 = {res.xx1:.1f}",
            fontsize=8, rotation=ang, ha="center", va="bottom", color="0.2")
    ax.annotate(f"角部弯弧 RD1 = {r:.0f}", xy=(x_bot, 0), xytext=(x_bot * 0.7, -h * 0.75),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
    ax.text(0, h * 0.45, f"梭长 Lm1 = {res.lm1:.1f}    平均匝长 LLM = {res.llm:.1f}",
            fontsize=9, ha="center", color="0.15",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))

    ax.set_title("梯形梭形大样（涨型前，单位 mm）", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-l5 * 0.62, l5 * 0.62)
    ax.set_ylim(-h * 1.1, h * 1.8)
    ax.axis("off")


def draw_axial_view(ax, res: CoilResult) -> None:
    """轴向侧视示意：直线部、端部斜边爬升 CC、鼻端（单边）。"""
    inp = res.inp
    zl = res.l2 / 2
    z_end = zl + res.cc
    # 展开周向坐标：上层边 s=0，鼻端 s=AA1
    up = [(-zl, 0), (zl, 0)]
    ax.plot(*zip(*up), color="tab:blue", lw=2.4, label="直线部(槽内+伸出)")
    ax.plot([zl, z_end], [0, res.aa1], color="tab:orange", lw=2.4, label="端部斜边(展开)")
    ax.plot([-zl, -z_end], [0, res.aa1], color="tab:orange", lw=2.4)
    ax.plot([z_end, z_end], [res.aa1, res.aa1 + 2 * inp.rd_nose],
            color="0.3", lw=2.4, label="鼻端")
    ax.plot([-z_end, -z_end], [res.aa1, res.aa1 + 2 * inp.rd_nose], color="0.3", lw=2.4)

    # 铁芯区域
    ax.axvspan(-inp.lc / 2, inp.lc / 2, color="0.92", zorder=0)
    ax.text(0, -res.aa1 * 0.18, f"铁芯 LC={inp.lc:.0f}", fontsize=8, ha="center", color="0.4")

    _dim(ax, (zl, -res.aa1 * 0.1), (z_end, -res.aa1 * 0.1), f"CC = {res.cc:.1f}")
    _dim(ax, (-zl, res.aa1 * 0.55), (-z_end, res.aa1 * 0.55), f"S1 = {res.s1:.1f}")
    ax.text(0, res.aa1 * 0.75,
            f"seita1 = {math.degrees(res.seita1):.1f}°   seita2 = {math.degrees(res.seita2):.1f}°",
            fontsize=8, color="0.2", ha="center")
    ax.set_xlim(-z_end * 1.12, z_end * 1.12)

    ax.set_title("端部侧视示意（斜边按展开长度画出）", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_aspect("auto")
    ax.axis("off")


def make_figure(res: CoilResult, dpi: int = 100) -> "plt.Figure":
    """组合大样图：端面投影 + 梭形大样 + 侧视示意。"""
    fig = plt.figure(figsize=(11.5, 7.6), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    draw_end_view(ax1, res)
    draw_axial_view(ax3, res)
    draw_lozenge(ax2, res)
    fig.suptitle("定子成型线圈大样图（依据 CN104965948B 公式体系计算）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save_png(res: CoilResult, filepath: str, dpi: int = 150) -> None:
    fig = make_figure(res, dpi=dpi)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
