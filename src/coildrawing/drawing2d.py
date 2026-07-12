"""线圈大样图（2D，工程制图风格）。

四个视图：端面投影图、端部侧视图、槽内截面图（逐匝真实截面）、
梯形梭形大样（绕线模）。制图约定：

    轮廓        粗实线（黑）
    中心线      细点划线（灰）
    尺寸        细实线 + 实心箭头 + 尺寸界线
    剖面        铁芯打剖面线，材料按 3D 模型配色

只依赖 matplotlib，供 UI 预览与 PNG/PDF/SVG 导出。
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

from .engine import CoilResult, strand_grid

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 与 3D 模型一致的配色
C_COPPER = "#C77833"
C_STRAND = "#993F26"
C_TURN = "#EDCC59"
C_GROUND = ["#D4B038", "#8C4512", "#BFBFBF", "#3373B3", "#59994D"]
C_CORONA = "#1A1A1A"
C_IRON = "#DDDDDD"
C_PAD = "#9C9C9C"
C_WEDGE = "#707070"

# 线型
LW_BOLD = 1.5      # 轮廓
LW_THIN = 0.6      # 尺寸/剖面线
C_DIM = "0.15"
C_CL = "0.45"      # 中心线
FS_DIM = 7.5
FS_NOTE = 8.0
FS_TITLE = 10.5


# ======================================================================
# 标注辅助
# ======================================================================
def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="<|-|>", mutation_scale=8,
        color=C_DIM, lw=LW_THIN, shrinkA=0, shrinkB=0, zorder=20,
        joinstyle="miter", capstyle="butt"))


def _dim_h(ax, x1, x2, y_feat, y_dim, text, text_above=True):
    """水平尺寸：尺寸界线 + 双箭头 + 文字。"""
    for x in (x1, x2):
        ax.plot([x, x], [y_feat, y_dim], color=C_DIM, lw=LW_THIN, zorder=19)
    _arrow(ax, (x1, y_dim), (x2, y_dim))
    dy = 1.0 if text_above else -1.0
    ax.text((x1 + x2) / 2, y_dim, text, ha="center",
            va="bottom" if text_above else "top",
            fontsize=FS_DIM, color=C_DIM, zorder=21)


def _dim_v(ax, y1, y2, x_feat, x_dim, text, text_right=True):
    """竖直尺寸。"""
    for y in (y1, y2):
        ax.plot([x_feat, x_dim], [y, y], color=C_DIM, lw=LW_THIN, zorder=19)
    _arrow(ax, (x_dim, y1), (x_dim, y2))
    ax.text(x_dim, (y1 + y2) / 2, text, ha="left" if text_right else "right",
            va="center", fontsize=FS_DIM, color=C_DIM, rotation=90, zorder=21)


def _dim_ab(ax, p1, p2, text, offset=(0, 0)):
    """两点间斜向尺寸（无界线）。"""
    _arrow(ax, p1, p2)
    mx, my = (p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=FS_DIM,
            color=C_DIM, zorder=21,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _leader(ax, target, txt_xy, text, ha="left"):
    """引出线标注：斜线 + 短横线 + 文字。"""
    elbow = 6 if ha == "left" else -6
    ax.plot([target[0], txt_xy[0], txt_xy[0] + elbow],
            [target[1], txt_xy[1], txt_xy[1]],
            color=C_DIM, lw=LW_THIN, zorder=20, clip_on=False)
    ax.plot([target[0]], [target[1]], marker=".", ms=2.5, color=C_DIM,
            zorder=20, clip_on=False)
    ax.text(txt_xy[0] + elbow * 1.3, txt_xy[1], text, ha=ha, va="center",
            fontsize=FS_DIM, color="0.1", zorder=21)


def _centerline(ax, p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_CL, lw=LW_THIN,
            ls=(0, (12, 3, 2, 3)), zorder=3)


def _view_title(ax, text):
    ax.set_title(text, fontsize=FS_TITLE - 1, pad=6, color="0.1")


# ======================================================================
# 视图 1：端面投影图
# ======================================================================
def draw_end_view(ax, res: CoilResult) -> None:
    inp = res.inp
    r_bore = inp.d2 / 2

    span = res.fai1 * 1.55 + 0.16
    th = np.linspace(-span, span, 240)
    xmax = res.rr2 * math.sin(span) * 1.08
    hi_r = res.rr2 + inp.f_nose + 40
    lo = min(res.ye, res.yk,
             res.rr2 * math.cos(res.fai2)) - 46   # 视图下缘

    # 中心线（线圈中心平面）与三条基准圆弧
    _centerline(ax, (0, lo + 6), (0, hi_r))
    for r in (r_bore, res.rr1, res.rr2):
        ax.plot(np.sin(th) * r, np.cos(th) * r, color=C_CL, lw=LW_THIN,
                ls=(0, (12, 3, 2, 3)), zorder=3)
    ax.text(-xmax * 0.99, lo + 2,
            f"基准圆(点划线,由内至外)\n定子内圆 D2/2={r_bore:.1f}\n"
            f"RR1={res.rr1:.1f}   RR2={res.rr2:.1f}",
            fontsize=FS_DIM, color="0.30", ha="left", va="bottom",
            linespacing=1.5, zorder=15,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none",
                      alpha=0.9))

    def rect_at(radius_bottom, theta, w, h, fc, main):
        er = np.array([math.sin(theta), math.cos(theta)])
        et = np.array([math.cos(theta), -math.sin(theta)])
        c0 = er * radius_bottom
        pts = [c0 - et * w / 2, c0 + et * w / 2,
               c0 + et * w / 2 + er * h, c0 - et * w / 2 + er * h]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.fill(xs, ys, color=fc, ec="k",
                lw=LW_BOLD if main else LW_THIN * 0.8,
                alpha=1.0 if main else 0.45, zorder=6 if main else 4)
        return c0 + er * h  # 顶部中点

    # 本线圈上/下层边 + 相邻线圈
    slot_ang = 2 * math.pi / inp.ns
    for k in (1, 2):
        rect_at(res.rr1, -res.fai1 + k * slot_ang, res.w_slot, res.h_slot,
                C_COPPER, False)
        rect_at(res.rr2, +res.fai2 - k * slot_ang, res.w_slot, res.h_slot,
                C_COPPER, False)
    top1 = rect_at(res.rr1, -res.fai1, res.w_slot, res.h_slot, C_COPPER, True)
    top2 = rect_at(res.rr2, +res.fai2, res.w_slot, res.h_slot, C_COPPER, True)

    # 端部斜边投影
    e = np.array([res.xe, res.ye])
    k_ = np.array([res.xk, res.yk])
    ax.plot([e[0], k_[0]], [e[1], k_[1]], color="k", lw=LW_BOLD, zorder=7)
    e2 = np.array([res.rr2 * math.sin(res.fai2), res.rr2 * math.cos(res.fai2)])
    rk2 = res.rr2 + inp.f_nose + res.hc * math.sin(inp.seita3)
    k2 = np.array([-res.xk, math.sqrt(max(rk2 ** 2 - res.xk ** 2, 0.0))])
    ax.plot([e2[0], k2[0]], [e2[1], k2[1]], color="k", lw=LW_BOLD, zorder=7)

    # 鼻端圆
    nose_c = (k_ + k2) / 2
    nose_r = inp.rd_nose + res.wd / 2
    tt = np.linspace(0, 2 * math.pi, 90)
    ax.plot(nose_c[0] + nose_r * np.cos(tt), nose_c[1] + nose_r * np.sin(tt),
            color="k", lw=LW_THIN * 1.2, zorder=6)

    # 引出标注：上带（视图上方，错层）与下带（视图下方）分列
    y_top = hi_r + 14
    _leader(ax, top1, (-xmax * 0.26, y_top + 15), "上层边(本线圈)", ha="right")
    _leader(ax, (nose_c[0], nose_c[1] + nose_r), (xmax * 0.05, y_top),
            f"鼻端 RD={inp.rd_nose:.0f}", ha="left")
    _leader(ax, top2, (xmax * 0.38, y_top + 15), "下层边(本线圈)", ha="left")
    y_bot = lo - 24
    mid1 = e * 0.42 + k_ * 0.58
    mid2 = e2 * 0.42 + k2 * 0.58
    _leader(ax, mid1, (-xmax * 0.10, y_bot), "上层端部斜边投影 e-k", ha="right")
    _leader(ax, mid2, (xmax * 0.48, y_bot), "下层端部斜边投影", ha="left")

    # 张角（弧放在视图下缘内侧）
    r_ann = lo + 24
    ax.add_patch(Arc((0, 0), 2 * r_ann, 2 * r_ann, angle=90,
                     theta1=-math.degrees(res.fai2),
                     theta2=math.degrees(res.fai1),
                     color=C_DIM, lw=LW_THIN))
    for sgn, fai in ((-1, res.fai1), (1, res.fai2)):
        p1 = (math.sin(sgn * fai) * (r_ann - 7), math.cos(sgn * fai) * (r_ann - 7))
        p2 = (math.sin(sgn * fai) * (r_ann + 7), math.cos(sgn * fai) * (r_ann + 7))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_DIM, lw=LW_THIN, zorder=4)
    ax.text(xmax * 0.12, r_ann - 14,
            f"张角 fai={res.fai:.4f} rad = {math.degrees(res.fai):.2f}°",
            fontsize=FS_DIM, ha="left", color=C_DIM,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                      alpha=0.85))

    ax.text(0, y_top + 34, "端面投影图（沿轴向看）", fontsize=FS_TITLE - 1,
            ha="center", va="bottom", color="0.1")
    ax.set_aspect("equal")
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(y_bot - 12, y_top + 52)
    ax.axis("off")


# ======================================================================
# 视图 2：端部侧视图
# ======================================================================
def draw_axial_view(ax, res: CoilResult) -> None:
    inp = res.inp
    zl = res.l2 / 2
    z_end = zl + res.cc
    nose_h = 2 * inp.rd_nose
    lead_sign = 1.0 if inp.lead_end_positive_z else -1.0

    # 铁芯区域（剖面线）
    ax.add_patch(Rectangle((-inp.lc / 2, -res.aa1 * 0.16), inp.lc, res.aa1 * 0.10,
                           fc=C_IRON, ec="0.4", lw=LW_THIN, hatch="////", zorder=1))
    ax.text(0, -res.aa1 * 0.11, f"铁芯 LC={inp.lc:.0f}", fontsize=FS_DIM,
            ha="center", va="center", color="0.25",
            bbox=dict(boxstyle="square,pad=0.15", fc=C_IRON, ec="none"))

    # 直线部 / 端部斜边 / 鼻端（双线轮廓按线宽示意为粗实线）
    ax.plot([-zl, zl], [0, 0], color="k", lw=LW_BOLD * 1.6, zorder=5)
    for sgn in (+1, -1):
        ax.plot([sgn * zl, sgn * z_end], [0, res.aa1], color="k",
                lw=LW_BOLD * 1.6, zorder=5)
        ax.plot([sgn * z_end, sgn * z_end], [res.aa1, res.aa1 + nose_h],
                color="k", lw=LW_BOLD * 1.6, zorder=5)
    # 引线（接线侧，轴向伸出）。只做 Z 向镜像，不变更上/下层。
    lead_y = res.aa1 + nose_h * 0.4
    lead_base_x = lead_sign * z_end
    lead_tip_x = lead_sign * (z_end + inp.ysc)
    ax.plot([lead_base_x, lead_tip_x], [lead_y] * 2,
            color=C_COPPER, lw=LW_BOLD * 1.5, zorder=6)
    lead_ha = "right" if lead_sign > 0 else "left"
    _leader(ax, (lead_sign * (z_end + inp.ysc * 0.75), lead_y),
            (lead_sign * (z_end - 46), res.aa1 + nose_h * 1.06),
            f"引线×2 ysc={inp.ysc:g}", ha=lead_ha)

    _leader(ax, (0.4 * zl, 0), (0.30 * zl, res.aa1 * 0.30), "直线部(槽内+伸出)")
    mid_s = ((zl + z_end) / 2, res.aa1 / 2)
    _leader(ax, (-mid_s[0], mid_s[1]), (-mid_s[0] + 55, res.aa1 * 0.85),
            "端部斜边(按展开长度)", ha="left")

    # 尺寸
    _dim_h(ax, zl, z_end, 0, -res.aa1 * 0.30, f"CC={res.cc:.1f}", text_above=False)
    _dim_ab(ax, (-zl, res.aa1 * 0.06), (-z_end, res.aa1 * 1.06),
            f"S1={res.s1:.1f}", offset=(26, -6))
    _dim_h(ax, -zl, zl, 0, res.aa1 * 1.30, f"L2={res.l2:.0f}")
    ax.text(0, res.aa1 * 0.62,
            f"seita1={math.degrees(res.seita1):.1f}°   "
            f"seita2={math.degrees(res.seita2):.1f}°",
            fontsize=FS_DIM, color=C_DIM, ha="center")

    _view_title(ax, "端部侧视图（斜边按展开长度画出）")
    ax.set_aspect("auto")
    if lead_sign > 0:
        ax.set_xlim(-z_end * 1.10, z_end + inp.ysc + 90)
    else:
        ax.set_xlim(-z_end - inp.ysc - 90, z_end * 1.10)
    ax.set_ylim(-res.aa1 * 0.55, res.aa1 * 1.65)
    ax.axis("off")


# ======================================================================
# 视图 3：槽内截面图（逐匝真实截面）
# ======================================================================
def draw_slot_section(ax, res: CoilResult) -> None:
    inp = res.inp
    ws, hs = inp.ws, inp.hs
    x0 = -ws / 2

    try:
        w_env, h_env, strands = strand_grid(inp)
    except ValueError:
        w_env = h_env = 0.0
        strands = []

    # 铁芯剖面（槽周围三边）
    bw = ws * 0.55
    for (x, y, w, h) in ((x0 - bw, 0, bw, hs + bw), (ws / 2, 0, bw, hs + bw),
                         (x0 - bw, hs, ws + 2 * bw, bw)):
        ax.add_patch(Rectangle((x, y), w, h, fc=C_IRON, ec="none",
                               hatch="////", zorder=1))
    ax.plot([x0, x0, ws / 2, ws / 2], [0, hs, hs, 0], color="k",
            lw=LW_BOLD, zorder=8)

    wrap_total = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
    ground_list = [l for l in inp.layers if l.thickness > 0]

    def coil_block(y_bot: float, tag: str):
        """一个线圈边（含防晕/对地/逐匝）自 y_bot 向上绘制，返回顶 y。"""
        y_top = y_bot + res.h_slot
        cx = 0.0
        # 防晕层
        if inp.cs > 0 or inp.corona_on:
            ax.add_patch(Rectangle((cx - res.w_slot / 2, y_bot), res.w_slot,
                                   res.h_slot, fc=C_CORONA, ec="k",
                                   lw=LW_THIN, zorder=4))
        # 对地绝缘逐层
        w_in = res.wa_turn
        h_in = res.had * inp.n_turns
        g_total = sum(l.thickness for l in ground_list)
        for i in range(len(ground_list) - 1, -1, -1):
            t_out = sum(l.thickness for l in ground_list[:i + 1])
            ax.add_patch(Rectangle(
                (cx - w_in / 2 - t_out, y_bot + inp.cs + (g_total - t_out)),
                w_in + 2 * t_out, h_in + 2 * t_out,
                fc=C_GROUND[i % len(C_GROUND)], ec="k", lw=LW_THIN * 0.7,
                zorder=5 + (len(ground_list) - i)))
        # 逐匝
        y_turn = y_bot + inp.cs + g_total
        for k in range(inp.n_turns):
            yc = y_turn + k * res.had + res.had / 2
            # 匝绝缘（外廓 had，内空 h_env）
            ax.add_patch(Rectangle((cx - w_env / 2 - wrap_total, yc - res.had / 2),
                                   w_env + 2 * wrap_total, res.had,
                                   fc=C_TURN, ec="k", lw=LW_THIN * 0.6, zorder=9))
            for s in strands:
                sx, sy = cx + s["x"], yc + s["y"]
                if s["t0"] > 0:
                    ax.add_patch(Rectangle((sx - s["bi"] / 2, sy - s["hi"] / 2),
                                           s["bi"], s["hi"], fc=C_STRAND,
                                           ec="none", zorder=10))
                ax.add_patch(Rectangle((sx - s["b"] / 2, sy - s["h"] / 2),
                                       s["b"], s["h"], fc=C_COPPER, ec="k",
                                       lw=LW_THIN * 0.5, zorder=11))
        return y_top

    # 自槽口(y=0)向槽底堆叠
    y = 0.0
    ax.add_patch(Rectangle((x0, y), ws, inp.hsd, fc=C_WEDGE, ec="k",
                           lw=LW_THIN, zorder=4))
    y_wedge = y + inp.hsd / 2
    y += inp.hsd
    ax.add_patch(Rectangle((x0, y), ws, inp.wihu, fc=C_PAD, ec="k",
                           lw=LW_THIN * 0.6, zorder=4))
    y += inp.wihu
    y_coil1 = y
    y = coil_block(y, "上层边")
    ax.add_patch(Rectangle((x0, y), ws, inp.wihm, fc=C_PAD, ec="k",
                           lw=LW_THIN * 0.6, zorder=4))
    y_mid_pad = y + inp.wihm / 2
    y += inp.wihm
    y_coil2 = y
    y = coil_block(y, "下层边")
    ax.add_patch(Rectangle((x0, y), ws, inp.wihb, fc=C_PAD, ec="k",
                           lw=LW_THIN * 0.6, zorder=4))
    y += inp.wihb

    # 标注：尺寸在左侧，引出标注在右侧，互不交叉
    xr = ws / 2
    x_txt = xr + bw + 1.5
    _dim_v(ax, 0, hs, x0, x0 - bw - 7, f"槽深 HS={hs:g}", text_right=False)
    _dim_v(ax, y_coil1, y_coil1 + res.h_slot, x0, x0 - bw - 2.2,
           f"H={res.h_slot:.2f}", text_right=False)
    _dim_h(ax, x0, ws / 2, 0, -bw * 0.55, f"槽宽 WS={ws:g}", text_above=False)
    if res.ha_margin > 0.3:
        _leader(ax, (xr * 0.35, y + res.ha_margin * 0.5),
                (x_txt - 3, hs + bw * 0.35), f"余量 Ha={res.ha_margin:.2f}")
    _leader(ax, (x0 * 0.5, y_wedge), (x_txt, y_wedge),
            f"槽楔 HSD={inp.hsd:g}")
    _leader(ax, (xr * 0.55, y_mid_pad), (x_txt, y_mid_pad),
            f"层间垫片 WIHM={inp.wihm:g}")
    g_total = sum(l.thickness for l in ground_list)
    y_g = y_coil2 + inp.cs + g_total / 2
    _leader(ax, (res.wa_turn / 2 + g_total * 0.5, y_g + 0.6),
            (x_txt, y_g + hs * 0.05), f"对地绝缘 T2={inp.t2:g}")
    y_t1 = y_coil2 + inp.cs + g_total + res.had * 1.0
    _leader(ax, (w_env / 2 + wrap_total * 0.5, y_t1),
            (x_txt, y_t1 + hs * 0.045), f"匝绝缘 T1={inp.t1:g}")
    y_cu = y_coil2 + inp.cs + g_total + res.had * 2.5
    _leader(ax, (0, y_cu), (x_txt, y_cu + hs * 0.06),
            f"铜线 {inp.wire1.b:g}×{inp.wire1.h:g}"
            + (f"×{inp.wire1.npd}并" if inp.wire1.npd > 1 else ""))
    if inp.cs > 0 or inp.corona_on:
        _leader(ax, (res.w_slot / 2 - 0.2, y_coil1 + res.h_slot * 0.75),
                (x_txt, y_coil1 + res.h_slot * 0.75 + hs * 0.03), "防晕层")

    _view_title(ax, "槽内截面图（真实比例）")
    ax.set_aspect("equal")
    ax.set_xlim(x0 - bw - 15, ws / 2 + bw + 30)
    ax.set_ylim(-bw - 7, hs + bw + 7)
    ax.axis("off")


# ======================================================================
# 视图 4：梯形梭形大样
# ======================================================================
def draw_lozenge(ax, res: CoilResult) -> None:
    l4, l5, h, r = res.l4, res.l5, res.h_lozenge, res.rd1_lozenge
    if h <= 0:
        h = max(res.cc, 1.0)

    x_bot, x_top = l5 / 2, l4 / 2

    # 圆角梯形轮廓
    corners = [(-x_bot, 0), (x_bot, 0), (x_top, h), (-x_top, h)]
    n_c = len(corners)
    verts_x: list[float] = []
    verts_y: list[float] = []
    for i in range(n_c):
        q = np.array(corners[(i - 1) % n_c])
        p = np.array(corners[i])
        nx = np.array(corners[(i + 1) % n_c])
        u = (p - q) / np.linalg.norm(p - q)
        v = (nx - p) / np.linalg.norm(nx - p)
        cosv = float(np.clip(np.dot(u, v), -1, 1))
        tau = math.acos(cosv)
        t = min(r * math.tan(tau / 2), 0.4 * min(np.linalg.norm(p - q),
                                                 np.linalg.norm(nx - p)))
        r_eff = t / math.tan(tau / 2) if tau > 1e-9 else 0.0
        ts, te = p - u * t, p + v * t
        cdir = (v - u)
        cdir = cdir / np.linalg.norm(cdir)
        c = p + cdir * (r_eff / math.cos(tau / 2))
        a1 = math.atan2(ts[1] - c[1], ts[0] - c[0])
        a2 = math.atan2(te[1] - c[1], te[0] - c[0])
        while a2 - a1 > math.pi:
            a2 -= 2 * math.pi
        while a1 - a2 > math.pi:
            a2 += 2 * math.pi
        aa = np.linspace(a1, a2, 12)
        verts_x.extend((c[0] + r_eff * np.cos(aa)).tolist())
        verts_y.extend((c[1] + r_eff * np.sin(aa)).tolist())
    verts_x.append(verts_x[0])
    verts_y.append(verts_y[0])
    ax.fill(verts_x, verts_y, color="#F7E9C8", ec="none", zorder=2)
    ax.plot(verts_x, verts_y, color="k", lw=LW_BOLD, zorder=5)

    _centerline(ax, (0, -h * 0.28), (0, h * 1.28))
    _centerline(ax, (-x_bot * 1.06, h / 2), (x_bot * 1.06, h / 2))

    # 尺寸
    _dim_h(ax, -x_bot, x_bot, 0, -h * 0.42, f"L5={l5:.1f}", text_above=False)
    _dim_h(ax, -x_top, x_top, h, h * 1.42, f"L4={l4:.1f}")
    _dim_v(ax, 0, h, x_bot, x_bot + l5 * 0.035, f"h_={res.h_lozenge:.1f}")
    ang = math.degrees(math.atan2(h, x_top - x_bot))
    if ang > 90:
        ang -= 180
    ax.text((x_bot + x_top) / 2 - l5 * 0.045, h * 0.62, f"斜边 XX1={res.xx1:.1f}",
            fontsize=FS_DIM, rotation=ang, ha="center", va="bottom", color=C_DIM)
    _leader(ax, (x_bot - r * 0.35, r * 0.28), (x_bot * 0.72, -h * 0.72),
            f"角部弯弧 RD1={r:.0f}", ha="left")
    ax.text(-x_bot * 0.62, -h * 0.72,
            f"梭长 Lm1={res.lm1:.1f}      平均匝长 LLM={res.llm:.1f}",
            fontsize=FS_NOTE, ha="center", va="center", color="0.1")

    _view_title(ax, "梯形梭形大样（涨型前）")
    ax.set_aspect("equal")
    ax.set_xlim(-l5 * 0.60, l5 * 0.60)
    ax.set_ylim(-h * 1.15, h * 1.85)
    ax.axis("off")


# ======================================================================
# 组图 + 图框
# ======================================================================
def make_figure(res: CoilResult, dpi: int = 100) -> "plt.Figure":
    inp = res.inp
    fig = plt.figure(figsize=(13.4, 8.3), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.45, 0.85],
                          height_ratios=[1.30, 1.0],
                          left=0.015, right=0.985, top=0.905, bottom=0.085,
                          wspace=0.06, hspace=0.12)
    draw_end_view(fig.add_subplot(gs[0, 0]), res)
    draw_axial_view(fig.add_subplot(gs[0, 1]), res)
    draw_slot_section(fig.add_subplot(gs[:, 2]), res)
    draw_lozenge(fig.add_subplot(gs[1, 0:2]), res)

    # 图框 + 标题栏
    fig.add_artist(plt.Rectangle((0.006, 0.008), 0.988, 0.984, fill=False,
                                 ec="0.1", lw=1.6,
                                 transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.006, 0.994], [0.052, 0.052], color="0.1",
                              lw=0.9, transform=fig.transFigure))
    fig.text(0.020, 0.030, "定子成型线圈大样图", fontsize=11, fontweight="bold",
             va="center")
    wire_txt = f"{inp.wire1.b:g}×{inp.wire1.h:g}"
    if inp.wire1.npd > 1 or inp.wire1.ncd > 1:
        wire_txt += f" ({inp.wire1.npd}并{inp.wire1.ncd}层)"
    if inp.wire2.npd > 0 and inp.wire2.ncd > 0 and inp.wire2.b > 0:
        wire_txt += f" + {inp.wire2.b:g}×{inp.wire2.h:g}"
    info = (f"公式体系 CN104965948B    匝数 N={inp.n_turns}    线规 {wire_txt}    "
            f"节距 {inp.taw}槽    平均匝长 LLM={res.llm:.1f}    单位 mm")
    fig.text(0.980, 0.030, info, fontsize=8.5, va="center", ha="right",
             color="0.15")
    fig.text(0.500, 0.958, "定 子 成 型 线 圈 大 样 图", fontsize=13,
             fontweight="bold", ha="center", va="center")
    return fig


def save_figure(res: CoilResult, filepath: str, dpi: int = 150) -> None:
    """按扩展名导出 PNG/PDF/SVG。"""
    fig = make_figure(res, dpi=dpi)
    fig.savefig(filepath, dpi=dpi, facecolor="white")
    plt.close(fig)


def save_png(res: CoilResult, filepath: str, dpi: int = 150) -> None:
    save_figure(res, filepath, dpi=dpi)
