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
  位于接线侧鼻端两侧斜边上并沿斜边退让（引入线落在最内匝、引出线
  从最外匝翘起，与邻匝角部及爬升坡道保持间隙），竖直伸出（长度=
  引线长 ysc，折弯半径可调），端头留可调长度裸铜；对地绝缘在穿出处
  按铜∪匝绝缘包络贴身开孔；
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

# 各层之间零间隙精确贴合（真实云母带就是紧包的，剖面无缝）：
# 匝绝缘外高恰=匝距、方壳内腔恰=匝束外廓、防晕层内腔恰=方壳外廓。
# 嵌套由共享同一套束中心线分段几何从构造上保证，不会穿体。
_TURN_CLEARANCE = 0.0
_FAMILY_GAP = 0.0

# 对地方壳引线开孔相对导线包络（含匝绝缘）的单边余量（贴身，避免大方孔）
_HOLE_CLEARANCE = 0.05

# 引线竖直段/折弯与相邻匝角2/角3圆角环之间的最小设计间隙。
# 沿斜边的退让距离在 _wire_segments 内按两者截面外接圆半径与
# 斜边方向水平分量自适应求解（竖直引线爬升时会跨过圆角环所在
# 的倾斜半空间，必须在水平投影上整体让开）
_LEAD_CLEARANCE = 1.0


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


@dataclass
class _Basis:
    """放样中间截面的仿射基（弦向插值，与直纹放样曲面精确一致）。

    直纹放样在参数 lam 处的截面 = 两端截面对应点的线性插值：
    点(u,v) = o + ex·u + ey·v，其中 ex/ey 为两端框架 x/y 轴的线性
    插值（**未归一**，长度略小于 1 是扭转弦缩的精确体现）。
    截面为平面四边形；由它构造的子放样/旋转/棱柱与整条放样的
    曲面严格共面——零间隙贴合所必需。
    """

    o: "b3d.Vector"
    ex: "b3d.Vector"
    ey: "b3d.Vector"
    t: "b3d.Vector"      # 该截面处纤维方向（归一），供棱柱拉伸/折弯用

    def at(self, o: "b3d.Vector") -> "_Basis":
        return _Basis(o, self.ex, self.ey, self.t)

    def rotated(self, n: "b3d.Vector", a: float, o: "b3d.Vector") -> "_Basis":
        return _Basis(o, _rotv(self.ex, n, a), _rotv(self.ey, n, a),
                      _rotv(self.t, n, a))


def _chord_basis(f0: _Frame, f1: _Frame, lam: float, dy: float = 0.0) -> _Basis:
    """整条放样 f0→f1 在参数 lam、径向偏移 dy 处的截面基与纤维方向。"""
    o = f0.o * (1 - lam) + f1.o * lam
    ex = f0.x * (1 - lam) + f1.x * lam
    ey = f0.y * (1 - lam) + f1.y * lam
    t = ((f1.o - f0.o) + (f1.y - f0.y) * dy).normalized()
    return _Basis(o, ex, ey, t)


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
    一个匝距，互不接触）；两端引线的折弯区位于平直段两侧的斜边
    段内，且落点沿斜边向外偏移（离开角2/角3圆角的扫掠区，与相邻
    匝几何保持 ≥ _LEAD_CLEARANCE 的间隙），从构造上杜绝出线端头
    的导线重叠：

      * 引入线在第 1 匝（最内层）斜边 3→4 上：竖直(-Z)下来，
        折弯后正切切入斜边方向，落点沿斜边退让以离开角3圆角环；
      * 引出线在第 N 匝（最外层）斜边 1→2 的**末端**折弯转竖直(+Z)
        伸出（不退让：若沿斜边内移，竖直段会穿进同斜边上的内层匝）；
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

    # 竖直引线与相邻匝角2/角3圆角环所需的水平净距：两者截面外接
    # 半径之和 + 设计间隙（竖直上升会跨过圆角环所在的倾斜半空间，
    # 只有水平投影上让开才真正安全）
    w_env, h_env, _strands_unused = _strand_grid(res)
    wrap_total = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
    sect_diag = math.hypot(w_env + 2 * wrap_total,
                           min(h_env + 2 * wrap_total, res.had))
    h_clear = sect_diag + _LEAD_CLEARANCE

    def retreat(t_dir: "b3d.Vector", t_tan: float, slant_len: float) -> float:
        """沿斜边的退让距离：水平净距 ÷ 斜边方向的水平分量。"""
        horiz = max(math.hypot(t_dir.X, t_dir.Y), 0.15)
        return min(t_tan + h_clear / horiz, 0.35 * slant_len)

    segs: list = []

    # ---- 引入线：竖直(-Z) → 折弯 → 斜边 3→4（第 1 匝，落点内移）----
    d0 = dy(0)
    len34 = (f34e.o - f34s.o).length
    b_probe = _chord_basis(f34s, f34e, 0.0, d0)
    tau_in = math.acos(clamp_cos((-zhat).dot(b_probe.t)))
    t_tan_in = rb0 * math.tan(tau_in / 2)
    # 落点沿斜边下移：折弯与竖直段在水平投影上整体离开角3圆角环
    s_in = retreat(b_probe.t, t_tan_in, len34)
    b_land = _chord_basis(f34s, f34e, s_in / len34, d0)
    land_in = b_land.o + b_land.ey * d0           # 第1匝落点（截面中心）
    p_corner_in = land_in - b_land.t * t_tan_in   # 竖直线与斜边线交点
    tip_in = p_corner_in + zhat * (t_tan_in + lead_len)
    fin = _fillet_corner(tip_in, p_corner_in,
                         land_in + b_land.t * max(2.5 * t_tan_in, 10.0), rb0)
    straight_in = (tip_in - fin.ts).length
    bare = max(0.0, min(inp.lead_bare, straight_in - 1.0))
    # 竖直段基（t=-Z）：落点基绕折弯轴反转（原点已含匝偏移，dy=0）
    f_lead_in = b_land.rotated(fin.n, -fin.tau, fin.ts)
    segs.append(_Prism(f_lead_in.at(tip_in), straight_in, bare0=bare))
    if fin.tau > 1e-6:
        segs.append(_Rev(f_lead_in.at(fin.ts), fin))

    tip_out = None
    info_out = None
    for k in range(n):
        d = dy(k)
        # 斜边 3→4 + 角4（第 1 匝从引入线落点起）
        segs.append(_Loft(b_land if k == 0 else f34s, f34e, dy0=d, dy1=d))
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
        if k < n - 1:
            # 斜边 1→2 + 角2 + 换位爬升（平直段 dy_k → dy_{k+1}）+ 角3
            segs.append(_Loft(f12s, f12e, dy0=d, dy1=d))
            segs.append(_Rev(f12e, fl[2], dy=d))
            segs.append(_Loft(ffc.at(fl[2].te), ffc.at(fl[3].ts),
                              dy0=d, dy1=dy(k + 1)))
            segs.append(_Rev(ffc.at(fl[3].ts), fl[3], dy=dy(k + 1)))
        else:
            # ---- 引出线：斜边 1→2 末端 → 折弯 → 竖直(+Z) ----
            # 注意：最外匝不能像引入线那样沿斜边退让——退让会把竖直段
            # 弯进同斜边上内层匝的实体（v202607121128 上方端口回归）。
            # 角2 圆角对本匝由折弯取代；与内层匝的水平间隙靠端点位置保证。
            b_end = _chord_basis(f12s, f12e, 1.0, d)
            segs.append(_Loft(f12s, f12e, dy0=d, dy1=d))
            end_out = b_end.o + b_end.ey * d       # 第 N 匝斜边末端（截面中心）
            tau_out = math.acos(clamp_cos(b_end.t.dot(zhat)))
            t_tan_out = rb0 * math.tan(tau_out / 2)
            p_corner_out = end_out + b_end.t * t_tan_out
            tip_out = p_corner_out + zhat * (t_tan_out + lead_len)
            fout = _fillet_corner(
                end_out - b_end.t * max(2.5 * t_tan_out, 10.0),
                p_corner_out, tip_out, rb0)
            # 折弯/竖直段框架原点落在偏移路径上（与引入线一致，dy=0）
            f_at_end = _Basis(end_out, b_end.ex, b_end.ey, b_end.t)
            if fout.tau > 1e-6:
                segs.append(_Rev(_Basis(fout.ts, b_end.ex, b_end.ey, b_end.t),
                                 fout))
            f_lead_out = f_at_end.rotated(fout.n, fout.tau, fout.te)
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
def _quad_wire(b: _Basis, w: float, h: float, xo: float, yo: float) -> "b3d.Wire":
    """基 b 上的截面四边形（可为平行四边形）边界线。"""
    pts = []
    for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append(b.o + b.ex * (xo + u) + b.ey * (yo + v))
    return b3d.Wire([b3d.Line(pts[i], pts[(i + 1) % 4]) for i in range(4)])


def _face_at(f, w: float, h: float, xo: float, yo: float):
    if isinstance(f, _Frame):
        return (f.plane() * b3d.Pos(xo, yo, 0) * b3d.Rectangle(w, h)).face()
    return b3d.Face(_quad_wire(f, w, h, xo, yo))


def _ring_at(f, w1, h1, w2, h2, xo, yo):
    if isinstance(f, _Frame):
        sk = b3d.Rectangle(w2, h2) - b3d.Rectangle(w1, h1)
        return (f.plane() * b3d.Pos(xo, yo, 0) * sk).face()
    return b3d.Face(_quad_wire(f, w2, h2, xo, yo),
                    [_quad_wire(f, w1, h1, xo, yo)])


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


def _as_basis(f) -> _Basis:
    return f if isinstance(f, _Basis) else _Basis(f.o, f.x, f.y, f.t)


def _loft_faces(fa, fb, w, h, xo, yo_a, yo_b):
    """放样两端面。任一端为 _Basis 时两端都走显式四边形（角点次序
    一致，保证直纹与整条放样精确共面）；否则维持原矩形路径。"""
    if isinstance(fa, _Basis) or isinstance(fb, _Basis):
        return (b3d.Face(_quad_wire(_as_basis(fa), w, h, xo, yo_a)),
                b3d.Face(_quad_wire(_as_basis(fb), w, h, xo, yo_b)))
    return (_face_at(fa, w, h, xo, yo_a), _face_at(fb, w, h, xo, yo_b))


def _seg_solid(seg, w: float, h: float, xo: float = 0.0, yo: float = 0.0):
    """实心截面沿分段的实体（截面再叠加分段自身的匝级偏移 dy）。"""
    if isinstance(seg, _Prism):
        if seg.length <= 1e-6:
            return None
        face = _face_at(seg.f, w, h, xo, yo + seg.dy)
        if isinstance(seg.f, _Basis):
            return b3d.extrude(face, amount=seg.length, dir=tuple(seg.f.t))
        return b3d.extrude(face, amount=seg.length)
    if isinstance(seg, _Rev):
        axis = b3d.Axis(origin=tuple(seg.fl.c), direction=tuple(seg.fl.n))
        return b3d.revolve(_face_at(seg.f, w, h, xo, yo + seg.dy), axis,
                           revolution_arc=math.degrees(seg.fl.tau))
    if isinstance(seg, _Loft):
        fa, fb = _loft_faces(seg.f0, seg.f1, w, h, xo,
                             yo + seg.dy0, yo + seg.dy1)
        return b3d.loft([fa, fb], ruled=True)
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
        face = _ring_at(f, w1, h1, w2, h2, xo, yo + seg.dy)
        if isinstance(f, _Basis):
            return b3d.extrude(face, amount=length, dir=tuple(f.t))
        return b3d.extrude(face, amount=length)
    if isinstance(seg, _Rev):
        axis = b3d.Axis(origin=tuple(seg.fl.c), direction=tuple(seg.fl.n))
        return b3d.revolve(_ring_at(seg.f, w1, h1, w2, h2, xo, yo + seg.dy),
                           axis, revolution_arc=math.degrees(seg.fl.tau))
    if isinstance(seg, _Loft):
        oa, ob = _loft_faces(seg.f0, seg.f1, w2, h2, xo,
                             yo + seg.dy0, yo + seg.dy1)
        ia, ib = _loft_faces(seg.f0, seg.f1, w1, h1, xo,
                             yo + seg.dy0, yo + seg.dy1)
        outer = b3d.loft([oa, ob], ruled=True)
        inner = b3d.loft([ia, ib], ruled=True)
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
            seg = _Loft(f_sl0, _chord_basis(f_sl0, f_sl1, lam2))
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


def _slot_hardware_parts(res: CoilResult, coil_half_h: float) -> list[CoilPart]:
    """槽内固定件：槽楔 / 槽楔下垫片 / 层间垫片 / 槽底垫片（可选）。

    与本线圈相邻的垫片面**贴着模型实际外表面**定位（coil_half_h =
    束中心到防晕层外表面的径向半高，含 CS 占位），即使绝缘分层总厚
    与 T2 不一致也不会与线圈穿体/留缝；只与相邻（未绘制的）线圈接触
    的面按专利截面堆叠链定位。宽度取槽宽 WS、长度取铁芯长 LC，
    在线圈上、下层边所在的两个槽内各生成一件。
    """
    inp = res.inp
    r_bore = inp.d2 / 2
    g1 = inp.t1 + inp.t2 + inp.cs        # 计算链：铜排外缘→绝缘线圈外缘
    rc1 = res.rr1 + res.hc / 2           # 上层边束中心半径
    rc2 = res.rr2 + res.hc / 2           # 下层边束中心半径
    up, dn = "上层边槽", "下层边槽"
    items = []                           # (名称, {槽: 内半径}, 厚度, 颜色)
    if inp.draw_wedge and inp.hsd > 0:
        items.append(("槽楔", {up: r_bore, dn: r_bore}, inp.hsd, WEDGE_COLOR))
    if inp.draw_wihu and inp.wihu > 0:
        items.append(("槽楔下垫片",
                      {up: rc1 - coil_half_h - inp.wihu,   # 贴本线圈内表面
                       dn: r_bore + inp.hsd},              # 邻线圈侧按计算链
                      inp.wihu, PAD_COLOR))
    if inp.draw_wihm and inp.wihm > 0:
        items.append(("层间垫片",
                      {up: rc1 + coil_half_h,              # 贴本线圈外表面
                       dn: rc2 - coil_half_h - inp.wihm},  # 贴本线圈内表面
                      inp.wihm, PAD_COLOR))
    if inp.draw_wihb and inp.wihb > 0:
        items.append(("槽底垫片",
                      {up: res.rr2 + res.hc + g1,          # 邻线圈侧按计算链
                       dn: rc2 + coil_half_h},             # 贴本线圈外表面
                      inp.wihb, PAD_COLOR))
    if not items:
        return []

    zhat = b3d.Vector(0, 0, 1)
    parts = []
    for tag, th in ((up, -res.fai1), (dn, +res.fai2)):
        for name, rmap, t, color in items:
            rc = rmap[tag] + t / 2
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
    parts += _slot_hardware_parts(
        res, res.hc / 2 + grow + _FAMILY_GAP + res.inp.cs)
    return parts


def _lead_path_cutters(info: dict, cut_w: float, cut_h: float) -> list:
    """沿引线竖直段 + 折弯构造贴身切割体。

    截面 = 铜导线∪匝绝缘包络（cut_w×cut_h）+ 单边 _HOLE_CLEARANCE，
    几何上等价于“布尔减去导线包络与方壳的相交体积”，但只用引线
    局部实体做刀具，避免整圈共面布尔卡死。落点再沿斜边伸出一小段
    stub，保证切穿对地壁厚。
    """
    cl = _HOLE_CLEARANCE
    w, h = cut_w + 2 * cl, cut_h + 2 * cl
    stub = 6.0  # mm，沿斜边伸入方壳，略大于典型对地总厚
    cutters = []
    for key, tip_key in (("lead_in", "tip_in"), ("lead_out", "tip_out")):
        f_lead, fl = info[key]
        tip = info[tip_key]
        f_lead = _as_basis(f_lead)
        pieces = []
        if key == "lead_in":
            straight = (tip - fl.ts).length
            if straight > 1e-6:
                pieces.append(_seg_solid(_Prism(f_lead.at(tip), straight), w, h))
            if fl.tau > 1e-6:
                pieces.append(_seg_solid(_Rev(f_lead.at(fl.ts), fl), w, h))
            f_land = f_lead.rotated(fl.n, fl.tau, fl.te)
            pieces.append(_seg_solid(_Prism(f_land, stub), w, h))
        else:
            f_at_ts = f_lead.rotated(fl.n, -fl.tau, fl.ts)
            pieces.append(_seg_solid(
                _Prism(f_at_ts.at(fl.ts - f_at_ts.t * stub), stub), w, h))
            if fl.tau > 1e-6:
                pieces.append(_seg_solid(_Rev(f_at_ts, fl), w, h))
            straight = (tip - fl.te).length
            if straight > 1e-6:
                pieces.append(_seg_solid(_Prism(f_lead, straight), w, h))
        body = _join(pieces)
        if body is not None:
            cutters.append(body)
    return cutters


def _bundle_lead_cutters(bundle, info: dict, pad: float = 12.0) -> list:
    """用铜∪匝绝缘实体在引线区的裁剪块作刀具（首选，最贴身）。

    失败（布尔不收敛/空结果）时由调用方回退到 `_lead_path_cutters`。
    """
    cutters = []
    for key, tip_key in (("lead_in", "tip_in"), ("lead_out", "tip_out")):
        _f, fl = info[key]
        tip = info[tip_key]
        pts = [tip, fl.ts, fl.te]
        xs = [p.X for p in pts]
        ys = [p.Y for p in pts]
        zs = [p.Z for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        bw = max(xs) - min(xs) + 2 * pad
        bh = max(ys) - min(ys) + 2 * pad
        bd = max(zs) - min(zs) + 2 * pad
        try:
            box = b3d.Pos(cx, cy, cz) * b3d.Box(bw, bh, bd)
            clipped = bundle & box
            if clipped is None:
                continue
            sols = clipped.solids()
            if not sols:
                continue
            vol = sum(s.volume for s in sols)
            if vol < 1e-3:
                continue
            cutters.append(sols[0] if len(sols) == 1 else clipped)
        except Exception:
            continue
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

    # ---- 对地绝缘方壳（零间隙贴匝束外廓；引线处减去铜∪匝绝缘相交体积）----
    gap = _FAMILY_GAP
    w_in = w_env + 2 * wrap_total + 2 * gap
    h_in = (n - 1) * res.had + h_env + 2 * wrap_total + 2 * gap
    env_w = w_env + 2 * wrap_total
    env_h = (min(h_env + 2 * wrap_total, h_cap)
             if n > 1 else h_env + 2 * wrap_total)

    # 引线开孔刀具：截面 = 铜∪匝绝缘包络 + 贴身余量（几何上等价于
    # 减去导线包络与方壳的相交体积）。整圈铜∪匝绝缘布尔裁剪在 OCCT
    # 上可能不收敛，故用引线局部路径实体；_bundle_lead_cutters 保留备用。
    cutters = _lead_path_cutters(info, env_w, env_h)

    gparts, ggrow = _ground_parts(res, w_in, h_in, cutters)
    parts += gparts

    # ---- 防晕层 / 槽内固定件 ----
    parts += _corona_parts(res, w_in + 2 * ggrow + 2 * gap,
                           h_in + 2 * ggrow + 2 * gap)
    parts += _slot_hardware_parts(
        res, h_in / 2 + ggrow + gap + inp.cs)
    return parts



def build_coil_parts(res: CoilResult, detailed: bool | None = None) -> list[CoilPart]:
    """构造线圈部件列表。detailed=None 时按输入 detail_3d 选择。"""
    if detailed is None:
        detailed = res.inp.detail_3d
    return _build_detailed_parts(res) if detailed else _build_simple_parts(res)


def _finish_part(part: CoilPart) -> CoilPart:
    """导出前的部件整理：

    1. 单实体复合体解包为实体（STEP 中保持一层零件、不嵌套子装配）；
    2. ShapeUpgrade_UnifySameDomain 合并同域相切面——分段构造产生的
       大量接缝碎面会让 SolidWorks 经典翻译器报“重建模型错误”，
       合并后面数大减、文件更小；
    3. 有效性校验（无效实体在 SolidWorks 中会降级为“曲面实体”），
       无效则 ShapeFix 修复，仍无效退回原形状。
    """
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_ShapeEnum

    def unwrap(obj):
        if (obj.wrapped.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND
                and len(obj.solids()) == 1):
            return obj.solids()[0]
        return obj

    solid = unwrap(part.solid)
    shape = solid.wrapped
    try:
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        uni = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        uni.Build()
        unified = uni.Shape()
        if BRepCheck_Analyzer(unified).IsValid():
            shape = unified
    except Exception:
        pass
    if not BRepCheck_Analyzer(shape).IsValid():
        try:
            from OCP.ShapeFix import ShapeFix_Shape

            fixer = ShapeFix_Shape(shape)
            fixer.Perform()
            fixed = fixer.Shape()
            if BRepCheck_Analyzer(fixed).IsValid():
                shape = fixed
        except Exception:
            pass
    if shape is not solid.wrapped:
        wrapped = (b3d.Solid(shape)
                   if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SOLID
                   else b3d.Compound(shape))
        return CoilPart(part.name, unwrap(wrapped), part.color)
    return CoilPart(part.name, solid, part.color)


def _export_step_xcaf(parts: list[CoilPart], filepath: str,
                      asm_name: str) -> None:
    """自建 XCAF 装配文档并写 STEP。

    与 build123d 默认导出的关键差异：**颜色只登记在零件原型上**
    （Gen+Surf 两种色型），不在装配实例上重复登记——后者会产出
    OVER_RIDING_STYLED_ITEM（装配级覆盖样式），SolidWorks 的
    3D Interconnect 翻译器读不懂覆盖样式，零件会显示为白色。
    """
    from OCP.APIHeaderSection import APIHeaderSection_MakeHeader
    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.IGESControl import IGESControl_Controller
    from OCP.Interface import Interface_Static
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import (
        STEPCAFControl_Controller, STEPCAFControl_Writer)
    from OCP.STEPControl import STEPControl_Controller, STEPControl_StepModelType
    from OCP.TCollection import (
        TCollection_ExtendedString, TCollection_HAsciiString)
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopoDS import TopoDS_Compound
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import (
        XCAFDoc_ColorType, XCAFDoc_DocumentTool, XCAFDoc_ShapeTool)
    from OCP.XSControl import XSControl_WorkSession

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    app = XCAFApp_Application.GetApplication_s()
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    app.InitDocument(doc)
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)   # mm
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    shape_tool.SetAutoNaming_s(False)

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for p in parts:
        builder.Add(comp, p.solid.wrapped)
    asm_label = shape_tool.AddShape(comp, True)
    TDataStd_Name.Set_s(asm_label, TCollection_ExtendedString(asm_name))

    seq = TDF_LabelSequence()
    XCAFDoc_ShapeTool.GetComponents_s(asm_label, seq)
    for i in range(1, seq.Length() + 1):
        inst = seq.Value(i)
        ref = TDF_Label()
        if not XCAFDoc_ShapeTool.GetReferredShape_s(inst, ref) or ref.IsNull():
            continue
        shape = XCAFDoc_ShapeTool.GetShape_s(ref)
        part = next((p for p in parts if p.solid.wrapped.IsSame(shape)), None)
        if part is None:
            continue
        name = TCollection_ExtendedString(part.name)
        TDataStd_Name.Set_s(inst, name)
        TDataStd_Name.Set_s(ref, name)
        color = Quantity_Color(*part.color, Quantity_TOC_RGB)
        color_tool.SetColor(ref, color, XCAFDoc_ColorType.XCAFDoc_ColorGen)
        color_tool.SetColor(ref, color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
    shape_tool.UpdateAssemblies()

    session = XSControl_WorkSession()
    writer = STEPCAFControl_Writer(session, False)
    writer.SetColorMode(True)
    writer.SetLayerMode(True)
    writer.SetNameMode(True)

    header = APIHeaderSection_MakeHeader(writer.Writer().Model())
    if not header.IsDone():
        header = APIHeaderSection_MakeHeader(0)
        header.Apply(writer.Writer().Model())
    header.SetName(TCollection_HAsciiString(asm_name))
    header.SetOriginatingSystem(TCollection_HAsciiString("CoilDrawing"))

    STEPCAFControl_Controller.Init_s()
    STEPControl_Controller.Init_s()
    IGESControl_Controller.Init_s()
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 1)
    writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError("STEP 写出失败")


def export_step(res: CoilResult, filepath: str,
                detailed: bool | None = None) -> list[str]:
    """构造三维模型并导出 STEP。返回部件名列表。"""
    parts = [_finish_part(p) for p in build_coil_parts(res, detailed)]
    _export_step_xcaf(parts, filepath, asm_name="成型线圈")
    fix_step_names(filepath, header_name="成型线圈")
    return [p.name for p in parts]
