"""成型线圈三维实体建模与 STEP 导出。

根据计算结果构造线圈中心线（闭合三维路径：上层直线边 → 端部斜边 →
鼻端 → 下层直线边 → 非接线侧端部），沿路径扫掠矩形截面得到：

    * 铜导体束（截面 WC×HC，含匝间绝缘的裸组等效截面）
    * 若干层对地绝缘（云母带等），每层为包在内层外的壳体，
      厚度由用户逐层设定

导出为 STEP（AP214，无损 B-Rep）。Parasolid(.x_t) 为西门子私有格式，
开源工具链无法直接生成；在 SolidWorks / NX / Solid Edge 中打开 STEP
后可直接另存为 .x_t（这些软件即 Parasolid 内核，转换零损失）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .compat import import_build123d
from .engine import CoilResult

b3d = import_build123d()


# 颜色（RGB 0-1）：铜 + 一组绝缘层备选色
COPPER_COLOR = (0.78, 0.47, 0.20)
LAYER_COLORS = [
    (0.83, 0.69, 0.22),   # 云母金黄
    (0.55, 0.27, 0.07),   # 深棕
    (0.75, 0.75, 0.75),   # 银灰
    (0.20, 0.45, 0.70),   # 蓝
    (0.35, 0.60, 0.30),   # 绿
]


@dataclass
class CoilPart:
    name: str
    solid: object          # build123d Part/Solid
    color: tuple[float, float, float]


def _cyl(r: float, theta: float, z: float) -> "b3d.Vector":
    """柱坐标 → 直角坐标。theta=0 为线圈中心平面，y 轴为其径向。"""
    return b3d.Vector(r * math.sin(theta), r * math.cos(theta), z)


def _fillet_corner(q: "b3d.Vector", p: "b3d.Vector", r_: "b3d.Vector", radius: float):
    """计算三维角点 p 处的圆角。

    返回 (ts, ma, te)：圆弧起点、弧上中点、圆弧终点。
    q、r_ 分别为前一个、后一个角点。
    """
    u = (p - q).normalized()
    v = (r_ - p).normalized()
    cos_tau = max(-1.0, min(1.0, u.dot(v)))
    tau = math.acos(cos_tau)
    if tau < 1e-9:
        return None  # 共线，无需圆角
    t = radius * math.tan(tau / 2)
    ts = p - u * t
    te = p + v * t
    c = p + (v - u).normalized() * (radius / math.cos(tau / 2))
    mid_chord = (ts + te) * 0.5
    ma = c + (mid_chord - c).normalized() * radius
    return ts, ma, te


def build_centerline(res: CoilResult) -> tuple["b3d.Wire", "b3d.Vector", "b3d.Vector", "b3d.Vector"]:
    """构造线圈闭合中心线。

    返回 (wire, start_point, start_tangent, start_xdir)，
    start_* 用于放置扫掠截面（位于上层直线边上）。
    """
    inp = res.inp
    r1c = res.rr1 + res.hc / 2          # 上层边中心半径（=gf）
    r2c = res.rr2 + res.hc / 2          # 下层边中心半径（=gi）
    rn = (r1c + r2c) / 2 + inp.f_nose   # 鼻端中心半径（含抬高 F）
    th1 = -res.fai1                     # 上层边角位置
    th2 = +res.fai2                     # 下层边角位置
    thn = (inp.rd_nose + res.wd / 2) / rn  # 鼻端平直段半张角
    zl = res.l2 / 2                     # 直线边半长
    zn = zl + res.cc                    # 鼻端轴向位置

    corners = [
        (_cyl(r1c, th1, -zl), inp.r_bend_slot),   # 0 上层边·非接线侧槽口
        (_cyl(r1c, th1, +zl), inp.r_bend_slot),   # 1 上层边·接线侧槽口
        (_cyl(rn, -thn, +zn), inp.rd_nose),       # 2 接线侧鼻端角 1
        (_cyl(rn, +thn, +zn), inp.rd_nose),       # 3 接线侧鼻端角 2
        (_cyl(r2c, th2, +zl), inp.r_bend_slot),   # 4 下层边·接线侧槽口
        (_cyl(r2c, th2, -zl), inp.r_bend_slot),   # 5 下层边·非接线侧槽口
        (_cyl(rn, +thn, -zn), inp.rd_nose),       # 6 非接线侧鼻端角 1
        (_cyl(rn, -thn, -zn), inp.rd_nose),       # 7 非接线侧鼻端角 2
    ]

    n = len(corners)
    fillets = []
    for i in range(n):
        q = corners[(i - 1) % n][0]
        p, rad = corners[i]
        r_ = corners[(i + 1) % n][0]
        # 圆角切线长不能超过相邻段长度的一半，必要时自动缩小
        seg1 = (p - q).length
        seg2 = (r_ - p).length
        u = (p - q).normalized()
        v = (r_ - p).normalized()
        tau = math.acos(max(-1.0, min(1.0, u.dot(v))))
        max_t = 0.45 * min(seg1, seg2)
        rad_eff = rad
        if tau > 1e-9:
            t_need = rad * math.tan(tau / 2)
            if t_need > max_t:
                rad_eff = max_t / math.tan(tau / 2)
        fillets.append(_fillet_corner(q, p, r_, rad_eff))

    edges = []
    for i in range(n):
        f_now = fillets[i]
        f_next = fillets[(i + 1) % n]
        ts, ma, te = f_now
        edges.append(b3d.ThreePointArc(ts, ma, te))
        nts = f_next[0]
        if (nts - te).length > 1e-6:
            edges.append(b3d.Line(te, nts))

    wire = b3d.Wire(edges)

    # 扫掠起始截面放在上层直线边（角点0→角点1 的直线段）中点
    p0 = corners[0][0]
    p1 = corners[1][0]
    start_pt = fillets[0][2]  # 上层直线段起点（角点0圆角出口）
    tangent = (p1 - p0).normalized()          # +Z 方向
    xdir = b3d.Vector(math.cos(th1), -math.sin(th1), 0.0)  # 周向（槽宽方向）
    return wire, start_pt, tangent, xdir


def _sweep_rect(res: CoilResult, path: "b3d.Wire", start_pt, tangent, xdir,
                width: float, height: float):
    """沿路径扫掠 width×height 矩形截面，返回实体。"""
    plane = b3d.Plane(origin=start_pt, x_dir=xdir, z_dir=tangent)
    section = plane * b3d.Rectangle(width, height)
    return b3d.sweep(
        section,
        path=path,
        transition=b3d.Transition.TRANSFORMED,
        is_frenet=False,
    )


def build_coil_parts(res: CoilResult) -> list[CoilPart]:
    """构造铜导体束 + 逐层绝缘壳。"""
    path, start_pt, tangent, xdir = build_centerline(res)

    parts: list[CoilPart] = []
    copper = _sweep_rect(res, path, start_pt, tangent, xdir, res.wc, res.hc)
    parts.append(CoilPart("铜导体束", copper, COPPER_COLOR))

    inner = copper
    grow = 0.0
    for i, layer in enumerate(res.inp.layers):
        if layer.thickness <= 0:
            continue
        grow += layer.thickness
        outer = _sweep_rect(res, path, start_pt, tangent, xdir,
                            res.wc + 2 * grow, res.hc + 2 * grow)
        shell = outer - inner
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        parts.append(CoilPart(f"绝缘层{i + 1}-{layer.name}", shell, color))
        inner = outer
    return parts


def export_step(res: CoilResult, filepath: str) -> list[str]:
    """构造三维模型并导出 STEP。返回部件名列表。"""
    parts = build_coil_parts(res)
    children = []
    for p in parts:
        solid = p.solid
        solid.label = p.name
        solid.color = b3d.Color(*p.color)
        children.append(solid)
    asm = b3d.Compound(label="成型线圈", children=children)
    b3d.export_step(asm, filepath)
    return [p.name for p in parts]
