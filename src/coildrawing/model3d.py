"""成型线圈三维实体建模与 STEP 导出。

几何构造采用“分段解析”方案，而不是整条路径扫掠：

    直线段 → 棱柱（截面姿态精确锚定：槽内沿槽轴、y 指向径向）
    弯角   → 截面绕圆角轴线的旋转体（精确圆环段）
    端部斜边 → 两端截面间的直纹放样（承担扭转——真实线圈张型时
               斜边段导体确实被扭转，使两直线边截面都正对槽底）

这样每一段的截面姿态都是解析确定的，避免了长路径扫掠固有的
截面姿态漂移（实测可达 10°，会导致下层边导体歪斜、部件互相侵入），
也几乎不需要大型布尔运算（仅斜边放样壳与引线开孔用局部布尔）。

两种模型：

* 简化束模型：铜导体等效整束（WC×HC）+ 对地绝缘逐层壳（+ 防晕层）。
* 逐匝精细模型：连续扁铜线（可并绕多股/双线规）绕 N 匝，匝间换位
  爬升位于接线侧鼻端平直段；每股外包自身绝缘，每匝外包“匝绝缘
  分层”，整束外包对地绝缘方壳（引线穿出处开孔）；两端引线折弯点
  位于接线侧鼻端两侧斜边端点（引入线落在最内匝斜边起点、引出线从
  最外匝斜边末端翘起，与平直段上的爬升坡道完全错开，互不干涉），
  竖直伸出（长度=引线长 ysc，折弯半径可调），端头留可调长度裸铜；
  可选槽部防晕层（厚度=CS，可沿导线越过槽口弯角向端臂延伸）；
  可选槽楔/槽楔下垫片/层间垫片/槽底垫片。

导出为 STEP（AP214，无损 B-Rep）。Parasolid(.x_t) 为西门子私有格式，
开源工具链无法直接生成；在 SolidWorks / NX / Solid Edge 中打开 STEP
后可直接另存为 .x_t（这些软件即 Parasolid 内核，转换零损失）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .compat import import_build123d
from .engine import CoilResult
from .step_i18n import fix_step_names

b3d = import_build123d()


# 颜色（RGB 0-1）
COPPER_COLOR = (0.78, 0.47, 0.20)
STRAND_INS_COLOR = (0.60, 0.25, 0.15)   # 导线自身绝缘（漆膜暗红棕）
TURN_LAYER_COLORS = [                    # 匝绝缘（云母带黄系）
    (0.93, 0.80, 0.35),
    (0.82, 0.65, 0.25),
    (0.70, 0.52, 0.18),
]
LAYER_COLORS = [                         # 对地绝缘备选色
    (0.83, 0.69, 0.22),   # 云母金黄
    (0.55, 0.27, 0.07),   # 深棕
    (0.75, 0.75, 0.75),   # 银灰
    (0.20, 0.45, 0.70),   # 蓝
    (0.35, 0.60, 0.30),   # 绿
]
CORONA_COLOR = (0.10, 0.10, 0.10)       # 防晕层黑
PAD_COLOR = (0.87, 0.86, 0.80)          # 垫片灰白（层压绝缘板）
WEDGE_COLOR = (0.58, 0.49, 0.29)        # 槽楔棕黄（环氧层压板）

# 相邻匝的匝绝缘外表面名义上相切（匝距=匝高），叠放方向每边收缩此值
# 留出工艺间隙，避免融合/显示时的共面歧义
_TURN_CLEARANCE = 0.02

# 匝束与对地方壳间的名义装配间隙。匝束与方壳共享同一套束中心线
# 分段几何（匝仅作截面偏移），嵌套由构造保证，此间隙只作为名义
# 包扎工艺余量存在
_FAMILY_GAP = 0.1


@dataclass
class CoilPart:
    name: str
    solid: object          # build123d Part/Solid/Compound
    color: tuple[float, float, float]


# ======================================================================
# 基础几何
# ======================================================================
def _cyl(r: float, theta: float, z: float) -> "b3d.Vector":
    """柱坐标 → 直角坐标。theta=0 为线圈中心平面，y 轴为其径向。"""
    return b3d.Vector(r * math.sin(theta), r * math.cos(theta), z)


def _rotv(v: "b3d.Vector", n: "b3d.Vector", a: float) -> "b3d.Vector":
    """Rodrigues 旋转：v 绕单位轴 n 转角 a。"""
    c, s = math.cos(a), math.sin(a)
    return v * c + n.cross(v) * s + n * (n.dot(v) * (1 - c))


@dataclass
class _Fillet:
    """一个角点的圆角信息。"""

    ts: "b3d.Vector"    # 圆弧起点（切点，位于来向段上）
    ma: "b3d.Vector"    # 弧上中点
    te: "b3d.Vector"    # 圆弧终点（切点，位于去向段上）
    c: "b3d.Vector"     # 圆心
    n: "b3d.Vector"     # 弯曲平面单位法向（u×v 方向）
    tau: float          # 转角 rad


def _fillet_corner(q, p, r_, radius: float) -> _Fillet:
    """三维角点 p 处的圆角（q、r_ 为前后角点）。

    圆角切线长自动限制在相邻段长度的 45% 内。
    """
    u = (p - q).normalized()
    v = (r_ - p).normalized()
    tau = math.acos(max(-1.0, min(1.0, u.dot(v))))
    rad_eff = radius
    if tau > 1e-9:
        max_t = 0.45 * min((p - q).length, (r_ - p).length)
        t_need = radius * math.tan(tau / 2)
        if t_need > max_t:
            rad_eff = max_t / math.tan(tau / 2)
    t = rad_eff * math.tan(tau / 2)
    ts = p - u * t
    te = p + v * t
    c = p + (v - u).normalized() * (rad_eff / math.cos(tau / 2))
    mid_chord = (ts + te) * 0.5
    ma = c + (mid_chord - c).normalized() * rad_eff
    n = u.cross(v).normalized()
    return _Fillet(ts, ma, te, c, n, tau)


def _loop_fillets(res: CoilResult, dr: float = 0.0):
    """径向偏移 dr 处线圈环路的 8 个角点及圆角。

    角点顺序：0/1 上层边非接线/接线侧槽口，2/3 接线侧鼻端角，
    4/5 下层边接线/非接线侧槽口，6/7 非接线侧鼻端角。
    """
    inp = res.inp
    r1c = res.rr1 + res.hc / 2 + dr     # 上层边中心半径
    r2c = res.rr2 + res.hc / 2 + dr     # 下层边中心半径
    rn0 = (res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose  # 束中心鼻端半径
    rn = rn0 + dr
    th1 = -res.fai1
    th2 = +res.fai2
    thn = (inp.rd_nose + res.wd / 2) / rn0  # 半张角统一取束值，各级角点对齐
    zl = res.l2 / 2
    zn = zl + res.cc

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
        fillets.append(_fillet_corner(q, p, r_, rad))
    return corners, fillets


def build_centerline(res: CoilResult) -> tuple["b3d.Wire", "b3d.Vector", "b3d.Vector", "b3d.Vector"]:
    """线圈束中心闭合中心线（供参考/兼容旧接口）。"""
    corners, fillets = _loop_fillets(res, 0.0)
    n = len(corners)
    edges = []
    for i in range(n):
        f = fillets[i]
        edges.append(b3d.ThreePointArc(f.ts, f.ma, f.te))
        nts = fillets[(i + 1) % n].ts
        if (nts - f.te).length > 1e-6:
            edges.append(b3d.Line(f.te, nts))
    wire = b3d.Wire(edges)

    start_pt = fillets[0].te
    tangent = b3d.Vector(0, 0, 1)
    th1 = -res.fai1
    xdir = b3d.Vector(math.cos(th1), -math.sin(th1), 0.0)
    return wire, start_pt, tangent, xdir


# ======================================================================
# 截面框架与分段
# ======================================================================
@dataclass
class _Frame:
    """路径上一点的截面框架：t 行进方向，y 截面纵轴（径向外），x=y×t。"""

    o: "b3d.Vector"
    x: "b3d.Vector"
    y: "b3d.Vector"
    t: "b3d.Vector"

    def plane(self) -> "b3d.Plane":
        # build123d Plane: y_dir = z_dir × x_dir = t × x = y ✓
        return b3d.Plane(origin=self.o, x_dir=self.x, z_dir=self.t)

    def at(self, o: "b3d.Vector") -> "_Frame":
        return _Frame(o, self.x, self.y, self.t)


def _anchor(o, t, y_hint) -> _Frame:
    """理想锚定框架：t 归一，y 为 y_hint 去除 t 分量后归一，x=y×t。"""
    t = t.normalized()
    y = (y_hint - t * y_hint.dot(t)).normalized()
    return _Frame(o, y.cross(t), y, t)


def _transport(f: _Frame, fl: _Fillet) -> _Frame:
    """框架经过圆角（绕 fl.n 转 fl.tau），落位于圆角出口。"""
    return _Frame(fl.te, _rotv(f.x, fl.n, fl.tau),
                  _rotv(f.y, fl.n, fl.tau), _rotv(f.t, fl.n, fl.tau))


def _pre_corner(f_out: _Frame, fl: _Fillet) -> _Frame:
    """反算：欲使经过圆角后的框架为 f_out，圆角入口处应有的框架。"""
    return _Frame(fl.ts, _rotv(f_out.x, fl.n, -fl.tau),
                  _rotv(f_out.y, fl.n, -fl.tau), _rotv(f_out.t, fl.n, -fl.tau))


# 分段描述。dy/dy0/dy1 为截面沿 y 的附加偏移（匝级径向偏移），
# 所有匝共享同一束中心线分段几何，仅截面偏移不同 —— 弯角处各匝
# 绕同一轴线旋转，几何上保证与方壳精确嵌套。
@dataclass
class _Prism:
    f: _Frame
    length: float
    dy: float = 0.0
    bare0: float = 0.0   # 起点侧裸铜长度（绝缘环让开）
    bare1: float = 0.0   # 终点侧裸铜长度


@dataclass
class _Rev:
    f: _Frame            # 位于圆角入口 ts
    fl: _Fillet
    dy: float = 0.0


@dataclass
class _Loft:
    f0: _Frame
    f1: _Frame
    dy0: float = 0.0
    dy1: float = 0.0


def _radial_of(p: "b3d.Vector") -> "b3d.Vector":
    return b3d.Vector(p.X, p.Y, 0).normalized()


def _lerp_frame(f0: _Frame, f1: _Frame, lam: float) -> _Frame:
    """两框架间按比例 lam 插值：原点线性，姿态按轴角插值。

    用于在斜边放样中途取截面框架（如防晕层延伸的自由端）。
    """
    o = f0.o * (1 - lam) + f1.o * lam
    a = ((f0.x.X, f0.y.X, f0.t.X),
         (f0.x.Y, f0.y.Y, f0.t.Y),
         (f0.x.Z, f0.y.Z, f0.t.Z))
    b = ((f1.x.X, f1.y.X, f1.t.X),
         (f1.x.Y, f1.y.Y, f1.t.Y),
         (f1.x.Z, f1.y.Z, f1.t.Z))
    # R = B·Aᵀ，把 f0 姿态映到 f1 姿态
    r = [[sum(b[i][k] * a[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    tr = r[0][0] + r[1][1] + r[2][2]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return _Frame(o, f0.x, f0.y, f0.t)
    s = 2.0 * math.sin(ang)
    axis = b3d.Vector((r[2][1] - r[1][2]) / s,
                      (r[0][2] - r[2][0]) / s,
                      (r[1][0] - r[0][1]) / s).normalized()
    da = ang * lam
    return _Frame(o, _rotv(f0.x, axis, da), _rotv(f0.y, axis, da),
                  _rotv(f0.t, axis, da))


@dataclass
class _LoopFrames:
    """束中心闭合环路的全部分段框架（匝与方壳共用）。"""

    fl: list                     # 8 个 _Fillet
    f_leg1: _Frame               # 上层边锚定框架
    f_leg2: _Frame               # 下层边锚定框架
    f_flat_conn: _Frame          # 接线侧鼻端平直段锚定框架
    f_flat_non: _Frame           # 非接线侧鼻端平直段锚定框架
    slant_frames: dict           # {(i_from,i_to): (f_start, f_end)} 斜边两端框架


def _loop_frames(res: CoilResult) -> _LoopFrames:
    corners, fl = _loop_fillets(res, 0.0)
    c = [x[0] for x in corners]
    yhat = b3d.Vector(0, 1, 0)

    f_leg1 = _anchor(fl[0].te, c[1] - c[0], _radial_of((c[0] + c[1]) * 0.5))
    f_leg2 = _anchor(fl[4].te, c[5] - c[4], _radial_of((c[4] + c[5]) * 0.5))
    f_flat_conn = _anchor(fl[2].te, c[3] - c[2], yhat)
    f_flat_non = _anchor(fl[6].te, c[7] - c[6], yhat)

    anchors = {0: f_leg1, 1: f_leg1, 2: f_flat_conn, 3: f_flat_conn,
               4: f_leg2, 5: f_leg2, 6: f_flat_non, 7: f_flat_non}
    slants = {}
    for i_from, i_to in ((1, 2), (3, 4), (5, 6), (7, 0)):
        f_start = _transport(anchors[i_from].at(fl[i_from].ts), fl[i_from])
        f_end = _pre_corner(anchors[i_to].at(fl[i_to].te), fl[i_to])
        slants[(i_from, i_to)] = (f_start, f_end)
    return _LoopFrames(fl, f_leg1, f_leg2, f_flat_conn, f_flat_non, slants)


def _loop_segments(res: CoilResult) -> list:
    """一圈闭合环路的分段序列（方壳/简化束用）。

    直线边与鼻端平直段为理想锚定棱柱，圆角为旋转体，斜边为扭转放样。
    """
    lf = _loop_frames(res)
    fl = lf.fl
    segs: list = []

    def leg(f_anchor, i_from, i_to):
        segs.append(_Prism(f_anchor.at(fl[i_from].te),
                           (fl[i_to].ts - fl[i_from].te).length))
        segs.append(_Rev(f_anchor.at(fl[i_to].ts), fl[i_to]))

    def slant(i_from, i_to):
        f_start, f_end = lf.slant_frames[(i_from, i_to)]
        segs.append(_Loft(f_start, f_end))
        segs.append(_Rev(f_end, fl[i_to]))

    leg(lf.f_leg1, 0, 1)        # 上层边 0→1，角1
    slant(1, 2)                 # 斜边 1→2，角2
    leg(lf.f_flat_conn, 2, 3)   # 鼻端平直 2→3，角3
    slant(3, 4)                 # 斜边 3→4，角4
    leg(lf.f_leg2, 4, 5)        # 下层边 4→5，角5
    slant(5, 6)                 # 斜边 5→6，角6
    leg(lf.f_flat_non, 6, 7)    # 非接线平直 6→7，角7
    slant(7, 0)                 # 斜边 7→0，角0（闭合）
    return segs


def _wire_segments(res: CoilResult):
    """N 匝连续导线的分段序列（含换位爬升与两端竖直引线）。

    所有匝复用 _loop_frames 的束中心分段框架，仅按匝号作截面径向
    偏移 dy —— 弯角处各匝绕同一轴线旋转、斜边共用同一对端面框架，
    从构造上保证匝与匝、匝与方壳精确嵌套。

    接线侧鼻端平直段只承载 N-1 条换位爬升坡道（相邻坡道整整错开
    一个匝距，互不接触）；两端引线的折弯点位于平直段两侧的斜边
    端点上，与坡道完全错开，从构造上杜绝出线端头的导线重叠：

      * 引入线在第 1 匝（最内层）斜边 3→4 的起点：竖直(-Z)下来，
        折弯后正切切入斜边方向（折弯取代该匝的角3圆角；dy(0)
        高度上该区域没有其他导线经过）；
      * 引出线在第 N 匝（最外层）斜边 1→2 的末端：沿斜边走完后
        折弯转竖直(+Z)伸出（折弯取代该匝的角2圆角）；
      * 两引线均沿轴向伸出，分居鼻端平直段两侧，一内一外，
        与实物线圈的出头位置一致。

    返回 (segs, info)。info 含 tip_in/tip_out/bare_len/lead_in/lead_out。
    """
    inp = res.inp
    n = inp.n_turns
    zhat = b3d.Vector(0, 0, 1)

    lf = _loop_frames(res)
    fl = lf.fl
    ffc = lf.f_flat_conn                     # 接线侧平直段锚定框架

    def dy(k: int) -> float:
        return res.had * (k - (n - 1) / 2)

    rb0 = max(inp.lead_bend_r, 2.0)
    lead_len = max(inp.ysc, 1.5 * rb0)

    f34s, f34e = lf.slant_frames[(3, 4)]
    f56s, f56e = lf.slant_frames[(5, 6)]
    f70s, f70e = lf.slant_frames[(7, 0)]
    f12s, f12e = lf.slant_frames[(1, 2)]

    def clamp_cos(c: float) -> float:
        return max(-1.0, min(1.0, c))

    segs: list = []

    # ---- 引入线：竖直(-Z) → 折弯 → 斜边 3→4 起点（第 1 匝）----
    d0 = dy(0)
    t34 = f34s.t
    start_in = f34s.o + f34s.y * d0               # 第1匝斜边起点（截面中心）
    tau_in = math.acos(clamp_cos((-zhat).dot(t34)))
    t_tan_in = rb0 * math.tan(tau_in / 2)
    p_corner_in = start_in - t34 * t_tan_in       # 竖直线与斜边线的交点
    tip_in = p_corner_in + zhat * (t_tan_in + lead_len)
    fin = _fillet_corner(tip_in, p_corner_in,
                         start_in + t34 * max(2.5 * t_tan_in, 10.0), rb0)
    straight_in = (tip_in - fin.ts).length
    bare = max(0.0, min(inp.lead_bare, straight_in - 1.0))
    f_after_in = _Frame(fin.te, f34s.x, f34s.y, f34s.t)  # 折弯出口=斜边姿态
    f_lead_in = _pre_corner(f_after_in, fin)             # 竖直段姿态（t=-Z）
    segs.append(_Prism(f_lead_in.at(tip_in), straight_in, bare0=bare))
    if fin.tau > 1e-6:
        segs.append(_Rev(f_lead_in.at(fin.ts), fin))

    tip_out = None
    info_out = None
    for k in range(n):
        d = dy(k)
        # 斜边 3→4 + 角4
        segs.append(_Loft(f34s, f34e, dy0=d, dy1=d))
        segs.append(_Rev(f34e, fl[4], dy=d))
        # 下层边 4→5 + 角5
        segs.append(_Prism(lf.f_leg2.at(fl[4].te),
                           (fl[5].ts - fl[4].te).length, dy=d))
        segs.append(_Rev(lf.f_leg2.at(fl[5].ts), fl[5], dy=d))
        # 斜边 5→6 + 角6
        segs.append(_Loft(f56s, f56e, dy0=d, dy1=d))
        segs.append(_Rev(f56e, fl[6], dy=d))
        # 非接线平直 6→7 + 角7
        segs.append(_Prism(lf.f_flat_non.at(fl[6].te),
                           (fl[7].ts - fl[6].te).length, dy=d))
        segs.append(_Rev(lf.f_flat_non.at(fl[7].ts), fl[7], dy=d))
        # 斜边 7→0 + 角0
        segs.append(_Loft(f70s, f70e, dy0=d, dy1=d))
        segs.append(_Rev(f70e, fl[0], dy=d))
        # 上层边 0→1 + 角1
        segs.append(_Prism(lf.f_leg1.at(fl[0].te),
                           (fl[1].ts - fl[0].te).length, dy=d))
        segs.append(_Rev(lf.f_leg1.at(fl[1].ts), fl[1], dy=d))
        # 斜边 1→2（角2 视去向：爬升走圆角，出线走折弯）
        segs.append(_Loft(f12s, f12e, dy0=d, dy1=d))

        if k < n - 1:
            # 角2 + 换位爬升（平直段 dy_k → dy_{k+1}）+ 角3
            segs.append(_Rev(f12e, fl[2], dy=d))
            segs.append(_Loft(ffc.at(fl[2].te), ffc.at(fl[3].ts),
                              dy0=d, dy1=dy(k + 1)))
            segs.append(_Rev(ffc.at(fl[3].ts), fl[3], dy=dy(k + 1)))
        else:
            # ---- 引出线：斜边 1→2 末端 → 折弯 → 竖直(+Z) ----
            t12 = f12e.t
            end_out = f12e.o + f12e.y * d          # 第 N 匝斜边末端
            tau_out = math.acos(clamp_cos(t12.dot(zhat)))
            t_tan_out = rb0 * math.tan(tau_out / 2)
            p_corner_out = end_out + t12 * t_tan_out
            tip_out = p_corner_out + zhat * (t_tan_out + lead_len)
            fout = _fillet_corner(
                end_out - t12 * max(2.5 * t_tan_out, 10.0),
                p_corner_out, tip_out, rb0)
            if fout.tau > 1e-6:
                segs.append(_Rev(f12e, fout, dy=d))
            f_here = _Frame(fout.ts, f12e.x, f12e.y, f12e.t)
            f_lead_out = _transport(f_here, fout)
            straight_out = (tip_out - fout.te).length
            segs.append(_Prism(f_lead_out, straight_out,
                               bare1=max(0.0, min(bare, straight_out - 1.0))))
            info_out = (f_lead_out, fout)

    info = dict(tip_in=tip_in, tip_out=tip_out, bare_len=bare,
                lead_in=(f_lead_in, fin), lead_out=info_out)
    return segs, info


# ======================================================================
# 分段实体化
# ======================================================================
def _face_at(f: _Frame, w: float, h: float, xo: float, yo: float):
    return (f.plane() * b3d.Pos(xo, yo, 0) * b3d.Rectangle(w, h)).face()


def _ring_at(f: _Frame, w1, h1, w2, h2, xo, yo):
    sk = b3d.Rectangle(w2, h2) - b3d.Rectangle(w1, h1)
    return (f.plane() * b3d.Pos(xo, yo, 0) * sk).face()


def _cut(a, b, fuzz: float = 1e-3):
    """局部布尔差集（带模糊容差，用于放样环与开孔）。"""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopTools import TopTools_ListOfShape

    args = TopTools_ListOfShape()
    args.Append(a.wrapped)
    tools = TopTools_ListOfShape()
    tools.Append(b.wrapped)
    op = BRepAlgoAPI_Cut()
    op.SetArguments(args)
    op.SetTools(tools)
    op.SetFuzzyValue(fuzz)
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        return a - b
    shape = op.Shape()
    if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
        comp = b3d.Compound(shape)
        sols = comp.solids()
        return sols[0] if len(sols) == 1 else comp
    return b3d.Solid(shape)


def _seg_solid(seg, w: float, h: float, xo: float = 0.0, yo: float = 0.0):
    """实心截面沿分段的实体（截面再叠加分段自身的匝级偏移 dy）。"""
    if isinstance(seg, _Prism):
        if seg.length <= 1e-6:
            return None
        return b3d.extrude(_face_at(seg.f, w, h, xo, yo + seg.dy),
                           amount=seg.length)
    if isinstance(seg, _Rev):
        axis = b3d.Axis(origin=tuple(seg.fl.c), direction=tuple(seg.fl.n))
        return b3d.revolve(_face_at(seg.f, w, h, xo, yo + seg.dy), axis,
                           revolution_arc=math.degrees(seg.fl.tau))
    if isinstance(seg, _Loft):
        return b3d.loft([_face_at(seg.f0, w, h, xo, yo + seg.dy0),
                         _face_at(seg.f1, w, h, xo, yo + seg.dy1)], ruled=True)
    raise TypeError(seg)


def _seg_ring(seg, w1, h1, w2, h2, xo: float = 0.0, yo: float = 0.0):
    """空心（环形）截面沿分段的实体。w1×h1 内腔，w2×h2 外廓。"""
    if isinstance(seg, _Prism):
        f = seg.f
        length = seg.length - seg.bare0 - seg.bare1
        if length <= 1e-6:
            return None
        if seg.bare0 > 0:
            f = f.at(f.o + f.t * seg.bare0)
        return b3d.extrude(_ring_at(f, w1, h1, w2, h2, xo, yo + seg.dy),
                           amount=length)
    if isinstance(seg, _Rev):
        axis = b3d.Axis(origin=tuple(seg.fl.c), direction=tuple(seg.fl.n))
        return b3d.revolve(_ring_at(seg.f, w1, h1, w2, h2, xo, yo + seg.dy),
                           axis, revolution_arc=math.degrees(seg.fl.tau))
    if isinstance(seg, _Loft):
        outer = b3d.loft([_face_at(seg.f0, w2, h2, xo, yo + seg.dy0),
                          _face_at(seg.f1, w2, h2, xo, yo + seg.dy1)],
                         ruled=True)
        inner = b3d.loft([_face_at(seg.f0, w1, h1, xo, yo + seg.dy0),
                          _face_at(seg.f1, w1, h1, xo, yo + seg.dy1)],
                         ruled=True)
        return _cut(outer, inner)
    raise TypeError(seg)


def _join(solids: list):
    """把一组端面对接的分段实体融合为单一实体（失败则退化为复合体）。

    分段实体之间只在端面（平面）上重合，用 glue 模式融合：只处理
    共享面、不做一般求交，速度快且不产生自相交伪影。
    """
    solids = [s for s in solids if s is not None]
    flat = []
    for s in solids:
        flat.extend(s.solids())
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]
    try:
        fused = flat[0].fuse(*flat[1:], glue=True)
        fused = fused.clean()
        sols = fused.solids()
        return sols[0] if len(sols) == 1 else fused
    except Exception:
        return b3d.Compound(children=flat)


# ======================================================================
# 部件构造
# ======================================================================
def _strand_grid(res: CoilResult):
    """每匝内的股线排布（见 engine.strand_grid）。"""
    from .engine import strand_grid

    return strand_grid(res.inp)


_CORONA_SLANT_CAP = 0.85   # 防晕层沿斜边最多延伸至斜边长度的该比例


def _corona_parts(res: CoilResult, inner_w: float, inner_h: float) -> list[CoilPart]:
    """槽部防晕层（黑色半导电层，厚度=CS）：上下层直线边各一段套管。

    每端伸出长度 corona_overhang 沿导线路径计量：先占直线段，超出
    部分越过槽口弯角（部分/整段旋转环），再沿端部斜边向鼻端延伸
    （直纹放样环，自由端截面按插值框架截取），最多至斜边长度的
    85%——这正是实物线圈低阻防晕层弯向鼻端搭接的形态。
    """
    inp = res.inp
    t_cor = inp.cs
    if not inp.corona_on or t_cor <= 0:
        return []
    lf = _loop_frames(res)
    fl = lf.fl
    w2, h2 = inner_w + 2 * t_cor, inner_h + 2 * t_cor
    ov = max(0.0, inp.corona_overhang)

    def rev_frame(f: _Frame, o: "b3d.Vector") -> _Frame:
        """行进方向取反的框架（y 保持，x 相应翻转）。"""
        return _Frame(o, f.y.cross(f.t * -1.0), f.y, f.t * -1.0)

    def ext_pieces(f_enter: _Frame, fillet: _Fillet, f_sl0: _Frame | None,
                   f_sl1: _Frame | None, remain: float) -> list:
        """一端的延伸段：弯角（部分/整段）+ 斜边（部分）。"""
        out = []
        if remain <= 1e-9:
            return out
        arc_r = (fillet.ts - fillet.c).length
        arc_len = max(arc_r * fillet.tau, 1e-9)
        lam = min(1.0, remain / arc_len)
        fl_part = _Fillet(fillet.ts, fillet.ma, fillet.te, fillet.c,
                          fillet.n, fillet.tau * lam)
        seg = _Rev(f_enter, fl_part)
        ring = _seg_ring(seg, inner_w, inner_h, w2, h2)
        if ring is not None:
            out.append(ring)
        remain -= arc_len
        if remain <= 1e-9 or f_sl0 is None or f_sl1 is None:
            return out
        slant_len = (f_sl1.o - f_sl0.o).length
        lam2 = min(remain / max(slant_len, 1e-9), _CORONA_SLANT_CAP)
        if lam2 > 1e-6:
            seg = _Loft(f_sl0, _lerp_frame(f_sl0, f_sl1, lam2))
            ring = _seg_ring(seg, inner_w, inner_h, w2, h2)
            if ring is not None:
                out.append(ring)
        return out

    parts = []
    for tag, f_leg, i_ent, i_exi, key_bw, key_fw in (
            ("上层边", lf.f_leg1, 0, 1, (7, 0), (1, 2)),
            ("下层边", lf.f_leg2, 4, 5, (3, 4), (5, 6))):
        leg_a, leg_b = fl[i_ent].te, fl[i_exi].ts
        leg_len = (leg_b - leg_a).length
        margin = max(0.0, (leg_len - inp.lc) / 2)   # 每端弯角前的直线余量

        # 直线段套管（铁芯段 + 两端直线内的伸出）
        take = min(ov, margin)
        s0 = margin - take
        length = inp.lc + 2 * take
        seg = _Prism(f_leg.at(leg_a + f_leg.t * s0), length)
        pieces = [_seg_ring(seg, inner_w, inner_h, w2, h2)]

        remain = ov - margin
        if remain > 1e-9:
            # 出口端（沿环路正向：leg → 角 i_exi → 斜边 key_fw）
            fs, fe = lf.slant_frames[key_fw]
            pieces += ext_pieces(f_leg.at(fl[i_exi].ts), fl[i_exi],
                                 fs, fe, remain)
            # 入口端（逆行：leg → 角 i_ent 反向 → 斜边 key_bw 反向）
            fil = fl[i_ent]
            fil_rev = _Fillet(fil.te, fil.ma, fil.ts, fil.c,
                              fil.n * -1.0, fil.tau)
            bs, be = lf.slant_frames[key_bw]
            pieces += ext_pieces(rev_frame(f_leg, fil.te), fil_rev,
                                 rev_frame(be, be.o), rev_frame(bs, bs.o),
                                 remain)
        parts.append(CoilPart(f"防晕层-{tag}", _join(pieces), CORONA_COLOR))
    return parts


def _slot_hardware_parts(res: CoilResult) -> list[CoilPart]:
    """槽内固定件：槽楔 / 槽楔下垫片 / 层间垫片 / 槽底垫片（可选）。

    径向位置按专利截面堆叠链（HH1/HH2）精确计算；宽度取槽宽 WS、
    长度取铁芯长 LC，在线圈上、下层边所在的两个槽内各生成一件。
    """
    inp = res.inp
    r_bore = inp.d2 / 2
    g1 = inp.t1 + inp.t2 + inp.cs        # 铜排外缘到绝缘线圈外缘（单边）
    items = []                           # (开关, 名称, 内半径, 厚度, 颜色)
    if inp.draw_wedge and inp.hsd > 0:
        items.append(("槽楔", r_bore, inp.hsd, WEDGE_COLOR))
    if inp.draw_wihu and inp.wihu > 0:
        items.append(("槽楔下垫片", r_bore + inp.hsd, inp.wihu, PAD_COLOR))
    if inp.draw_wihm and inp.wihm > 0:
        items.append(("层间垫片", res.rr1 + res.hc + g1, inp.wihm, PAD_COLOR))
    if inp.draw_wihb and inp.wihb > 0:
        items.append(("槽底垫片", res.rr2 + res.hc + g1, inp.wihb, PAD_COLOR))
    if not items:
        return []

    zhat = b3d.Vector(0, 0, 1)
    parts = []
    for tag, th in (("上层边槽", -res.fai1), ("下层边槽", +res.fai2)):
        for name, r_in, t, color in items:
            rc = r_in + t / 2
            o = _cyl(rc, th, -inp.lc / 2)
            f = _anchor(o, zhat, _radial_of(o))
            box = b3d.extrude(_face_at(f, inp.ws, t, 0.0, 0.0), amount=inp.lc)
            parts.append(CoilPart(f"{name}-{tag}", box, color))
    return parts


def _ground_parts(res: CoilResult, w_in: float, h_in: float,
                  cutters: list | None = None) -> tuple[list[CoilPart], float]:
    """对地绝缘方壳逐层（沿闭合环分段构造）。返回 (parts, 总厚)。"""
    segs = _loop_segments(res)
    parts: list[CoilPart] = []
    grow = 0.0
    for i, layer in enumerate(res.inp.layers):
        if layer.thickness <= 0:
            continue
        t0, grow = grow, grow + layer.thickness
        pieces = []
        for seg in segs:
            ring = _seg_ring(seg, w_in + 2 * t0, h_in + 2 * t0,
                             w_in + 2 * grow, h_in + 2 * grow)
            if ring is None:
                continue
            for cutter in cutters or []:
                bb1, bb2 = ring.bounding_box(), cutter.bounding_box()
                if (bb1.min.X < bb2.max.X and bb2.min.X < bb1.max.X and
                        bb1.min.Y < bb2.max.Y and bb2.min.Y < bb1.max.Y and
                        bb1.min.Z < bb2.max.Z and bb2.min.Z < bb1.max.Z):
                    ring = _cut(ring, cutter)
            pieces.append(ring)
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        parts.append(CoilPart(f"对地绝缘{i + 1}-{layer.name}",
                              _join(pieces), color))
    return parts, grow


# ----------------------------------------------------------------------
def _build_simple_parts(res: CoilResult) -> list[CoilPart]:
    """简化束模型：铜导体等效整束 + 逐层对地绝缘壳（+ 防晕层/固定件）。"""
    segs = _loop_segments(res)
    copper = _join([_seg_solid(s, res.wc, res.hc) for s in segs])
    parts = [CoilPart("铜导体束", copper, COPPER_COLOR)]

    gparts, grow = _ground_parts(res, res.wc, res.hc)
    parts += gparts
    parts += _corona_parts(res, res.wc + 2 * grow + 2 * _FAMILY_GAP,
                           res.hc + 2 * grow + 2 * _FAMILY_GAP)
    parts += _slot_hardware_parts(res)
    return parts


def _lead_hole_cutters(res: CoilResult, info: dict,
                       cut_w: float, cut_h: float) -> list:
    """引线穿出方壳处的开孔切割体。

    采用竖直方箱（纯平面棱柱）：覆盖引线折弯的水平摆动范围，自弯角
    平面下方一点直通引线顶端上方。相比沿引线路径的扫掠体，方箱与
    方壳弯角处旋转环的布尔差远为健壮（扫掠体×环面布尔在 OCCT 中
    可能不收敛），孔形也更整洁。

    径向半宽取 cut_h/2（≈匝束半高+半个族间隙）：足以让包着匝绝缘的
    导线通过，又不切穿方壳的径向侧壁。
    """
    zhat = b3d.Vector(0, 0, 1)
    cutters = []
    for key, tip in (("lead_in", info["tip_in"]), ("lead_out", info["tip_out"])):
        _f_lead, fl = info[key]
        # 关键点：折弯两切点 + 引线顶端
        pts = [fl.ts, fl.te, tip]
        land = fl.te if key == "lead_in" else fl.ts   # 斜边上的落点
        # a1: 折弯水平行进方向；a2: 水平化的径向（与 a1 正交）
        chord = fl.te - fl.ts
        a1 = b3d.Vector(chord.X, chord.Y, 0)
        a1 = (a1 if a1.length > 1e-9 else b3d.Vector(1, 0, 0)).normalized()
        r_hat = _radial_of(land)
        a2 = (r_hat - a1 * r_hat.dot(a1)).normalized()
        u = [p.dot(a1) for p in pts]
        u_lo, u_hi = min(u) - (cut_w / 2 + 2.0), max(u) + (cut_w / 2 + 2.0)
        v_c = land.dot(a2)
        z_lo = land.Z - 2.0
        z_hi = tip.Z + 5.0
        origin = a1 * (u_lo + u_hi) / 2 + a2 * v_c + b3d.Vector(0, 0, z_lo)
        face = (b3d.Plane(origin=origin, x_dir=tuple(a1), z_dir=(0, 0, 1))
                * b3d.Rectangle(u_hi - u_lo, cut_h)).face()
        cutters.append(b3d.extrude(face, amount=z_hi - z_lo))
    return cutters


def _build_detailed_parts(res: CoilResult) -> list[CoilPart]:
    """逐匝精细模型。"""
    inp = res.inp
    n = inp.n_turns
    w_env, h_env, strands = _strand_grid(res)
    wrap_total = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
    n_strand = len(strands)

    segs, info = _wire_segments(res)
    parts: list[CoilPart] = []

    # ---- 每股铜芯 + 自身绝缘 ----
    for s in strands:
        copper = _join([_seg_solid(g, s["b"], s["h"], s["x"], s["y"])
                        for g in segs])
        if n_strand == 1:
            name, ins_name = "铜导线", "导线自身绝缘"
        else:
            pos = f"(第{s['row'] + 1}层第{s['col'] + 1}根)"
            name = f"铜导线{s['no']}{pos}"
            ins_name = f"导线自身绝缘{s['no']}{pos}"
        parts.append(CoilPart(name, copper, COPPER_COLOR))
        if s["t0"] > 0:
            shell = _join([_seg_ring(g, s["b"], s["h"], s["bi"], s["hi"],
                                     s["x"], s["y"]) for g in segs])
            parts.append(CoilPart(ins_name, shell, STRAND_INS_COLOR))

    # ---- 匝绝缘分层（包每匝导线束）----
    h_cap = res.had - 2 * _TURN_CLEARANCE if n > 1 else float("inf")
    grow = 0.0
    prev_w, prev_h = w_env, h_env
    for i, layer in enumerate(inp.turn_layers):
        if layer.thickness <= 0:
            continue
        grow += layer.thickness
        w2 = w_env + 2 * grow
        h2 = min(h_env + 2 * grow, h_cap)
        shell = _join([_seg_ring(g, prev_w, prev_h, w2, h2) for g in segs])
        color = TURN_LAYER_COLORS[i % len(TURN_LAYER_COLORS)]
        parts.append(CoilPart(f"匝绝缘{i + 1}-{layer.name}", shell, color))
        prev_w, prev_h = w2, h2

    # ---- 对地绝缘方壳（含族间隙，引线穿出处开孔）----
    gap = _FAMILY_GAP
    w_in = w_env + 2 * wrap_total + 2 * gap
    h_in = (n - 1) * res.had + h_env + 2 * wrap_total + 2 * gap
    # 切割体截面须严格介于 匝束外廓 与 方壳内腔 之间：若与内腔尺寸
    # 相同，布尔差将遭遇大面积面-面重合，OCCT 可能失败甚至卡死
    cutters = _lead_hole_cutters(res, info,
                                 w_env + 2 * wrap_total + gap,
                                 h_env + 2 * wrap_total + gap)
    gparts, ggrow = _ground_parts(res, w_in, h_in, cutters)
    parts += gparts

    # ---- 防晕层 / 槽内固定件 ----
    parts += _corona_parts(res, w_in + 2 * ggrow + 2 * gap,
                           h_in + 2 * ggrow + 2 * gap)
    parts += _slot_hardware_parts(res)
    return parts


def build_coil_parts(res: CoilResult, detailed: bool | None = None) -> list[CoilPart]:
    """构造线圈部件列表。detailed=None 时按输入 detail_3d 选择。"""
    if detailed is None:
        detailed = res.inp.detail_3d
    return _build_detailed_parts(res) if detailed else _build_simple_parts(res)


def _ensure_valid(part: CoilPart) -> CoilPart:
    """导出前校验部件实体有效性；无效则尝试 ShapeFix 修复。

    无效实体写入 STEP 后，SolidWorks 等在导入修复失败时会把部件
    降级为“曲面实体”（空心壳），因此这里作为最后一道保险。
    """
    from OCP.BRepCheck import BRepCheck_Analyzer

    shape = part.solid.wrapped
    if BRepCheck_Analyzer(shape).IsValid():
        return part
    try:
        from OCP.ShapeFix import ShapeFix_Shape

        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if BRepCheck_Analyzer(fixed).IsValid():
            return CoilPart(part.name, b3d.Compound(fixed), part.color)
    except Exception:
        pass
    return part


def export_step(res: CoilResult, filepath: str,
                detailed: bool | None = None) -> list[str]:
    """构造三维模型并导出 STEP。返回部件名列表。"""
    parts = [_ensure_valid(p) for p in build_coil_parts(res, detailed)]
    children = []
    for p in parts:
        solid = p.solid
        solid.label = p.name
        solid.color = b3d.Color(*p.color)
        children.append(solid)
    asm = b3d.Compound(label="成型线圈", children=children)
    b3d.export_step(asm, filepath)
    fix_step_names(filepath, header_name=asm.label)
    return [p.name for p in parts]
