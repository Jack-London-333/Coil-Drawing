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
  分层”，整束外包对地绝缘方壳（引线穿出处开孔）；两端引线沿轴向
  伸出（长度=引线长 ysc，折弯半径可调），端头留可调长度裸铜；
  可选槽部防晕层。

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
    """N 匝连续导线的分段序列（含换位爬升与两端轴向引线）。

    所有匝复用 _loop_frames 的束中心分段框架，仅按匝号作截面径向
    偏移 dy —— 弯角处各匝绕同一轴线旋转、斜边共用同一对端面框架，
    从构造上保证匝与匝、匝与方壳精确嵌套。

    引线布置在接线侧鼻端平直段上（真实线圈出头位置）：
      * 引入线在靠角3端、第 1 匝（最内层）：竖直(-Z)下来折弯后沿
        平直段短暂前行即进角3；换位爬升在该端处于上一层，径向让开；
      * 引出线在靠角2端、第 N 匝（最外层）：过角2后沿平直段短暂
        前行即折弯转竖直(+Z)伸出；其余匝在该端径向让开；
      * 两引线轴向平行伸出，径向相隔 (N-1) 匝高。

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
    flat_len = (fl[3].ts - fl[2].te).length
    stub = min(1.15 * rb0, 0.45 * flat_len)  # 折弯圆角在平直段内的落脚长度

    segs: list = []

    # ---- 引入线：竖直(-Z) → 折弯 → 平直段(第1匝) → 角3 ----
    d0 = dy(0)
    q_in = fl[3].ts - ffc.t * stub                    # 束中心线上的折弯角点
    p_q_in = q_in + ffc.y * d0                        # 第1匝上的折弯角点
    tip_in = p_q_in + zhat * lead_len
    fin = _fillet_corner(tip_in, p_q_in, fl[3].ts + ffc.y * d0, rb0)
    straight_in = (tip_in - fin.ts).length
    bare = max(0.0, min(inp.lead_bare, straight_in - 1.0))
    f_after_in = _Frame(fin.te, ffc.x, ffc.y, ffc.t)  # 折弯出口=平直段姿态
    f_lead_in = _pre_corner(f_after_in, fin)          # 竖直段姿态（t=-Z）
    segs.append(_Prism(f_lead_in.at(tip_in), straight_in, bare0=bare))
    segs.append(_Rev(f_lead_in.at(fin.ts), fin))
    segs.append(_Prism(ffc.at(fin.te - ffc.y * d0),
                       (fl[3].ts - (fin.te - ffc.y * d0)).length, dy=d0))
    segs.append(_Rev(ffc.at(fl[3].ts), fl[3], dy=d0))

    tip_out = None
    info_out = None
    f34s, f34e = lf.slant_frames[(3, 4)]
    f56s, f56e = lf.slant_frames[(5, 6)]
    f70s, f70e = lf.slant_frames[(7, 0)]
    f12s, f12e = lf.slant_frames[(1, 2)]
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
        # 斜边 1→2 + 角2
        segs.append(_Loft(f12s, f12e, dy0=d, dy1=d))
        segs.append(_Rev(f12e, fl[2], dy=d))

        if k < n - 1:
            # 换位爬升：平直段 dy_k → dy_{k+1}，接角3
            segs.append(_Loft(ffc.at(fl[2].te), ffc.at(fl[3].ts),
                              dy0=d, dy1=dy(k + 1)))
            segs.append(_Rev(ffc.at(fl[3].ts), fl[3], dy=dy(k + 1)))
        else:
            # 引出线：平直段短行 → 折弯 → 竖直(+Z)
            q_out = fl[2].te + ffc.t * stub
            p_q_out = q_out + ffc.y * d
            tip_out = p_q_out + zhat * lead_len
            fout = _fillet_corner(fl[2].te + ffc.y * d, p_q_out, tip_out, rb0)
            segs.append(_Prism(ffc.at(fl[2].te),
                               (fout.ts - (fl[2].te + ffc.y * d)).length, dy=d))
            segs.append(_Rev(ffc.at(fout.ts - ffc.y * d), fout, dy=d))
            f_here = _Frame(fout.ts, ffc.x, ffc.y, ffc.t)
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


def _corona_parts(res: CoilResult, inner_w: float, inner_h: float) -> list[CoilPart]:
    """槽部防晕层：上下层直线边各一段矩形套管（棱柱环，无布尔）。

    长度不超过直线边的真实平直范围（槽口弯角圆弧起点以内），
    否则会包到弯角上与导体/方壳干涉。
    """
    inp = res.inp
    if not inp.corona_on or inp.corona_t <= 0:
        return []
    _, fl = _loop_fillets(res, 0.0)
    z_straight = abs(fl[1].ts.Z)             # 直线边平直段半长
    length = min(inp.lc + 2 * inp.corona_overhang, 2 * z_straight - 0.5)
    parts = []
    zhat = b3d.Vector(0, 0, 1)
    for tag, th, rc in (("上层边", -res.fai1, res.rr1 + res.hc / 2),
                        ("下层边", +res.fai2, res.rr2 + res.hc / 2)):
        o = _cyl(rc, th, -length / 2)
        f = _anchor(o, zhat, _radial_of(o))
        ring = b3d.extrude(
            _ring_at(f, inner_w, inner_h,
                     inner_w + 2 * inp.corona_t, inner_h + 2 * inp.corona_t,
                     0.0, 0.0),
            amount=length)
        parts.append(CoilPart(f"防晕层-{tag}", ring, CORONA_COLOR))
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
    """简化束模型：铜导体等效整束 + 逐层对地绝缘壳（+ 防晕层）。"""
    segs = _loop_segments(res)
    copper = _join([_seg_solid(s, res.wc, res.hc) for s in segs])
    parts = [CoilPart("铜导体束", copper, COPPER_COLOR)]

    gparts, grow = _ground_parts(res, res.wc, res.hc)
    parts += gparts
    parts += _corona_parts(res, res.wc + 2 * grow + 2 * _FAMILY_GAP,
                           res.hc + 2 * grow + 2 * _FAMILY_GAP)
    return parts


def _lead_hole_cutters(res: CoilResult, info: dict,
                       cut_w: float, cut_h: float) -> list:
    """引线穿出方壳处的开孔切割体：沿引线真实走向（竖直段+折弯+斜边入段）。"""
    zhat = b3d.Vector(0, 0, 1)
    cutters = []
    for key, tip in (("lead_in", info["tip_in"]), ("lead_out", info["tip_out"])):
        f_lead, fl = info[key]
        top = tip + zhat * 5.0
        if key == "lead_in":
            # 与导线同向：竖直向下 → 折弯 → 斜边入段
            f_down = f_lead.at(top)
            v_end = fl.ts
            fl_use = fl
        else:
            # 导线行进为 斜边→竖直；切割体反向走：竖直向下 → 反向折弯 → 斜边
            f_down = _anchor(top, -zhat, f_lead.y)
            v_end = fl.te
            fl_use = _Fillet(fl.te, fl.ma, fl.ts, fl.c, fl.n * -1.0, fl.tau)
        segs = [_Prism(f_down, (top - v_end).length),
                _Rev(f_down.at(v_end), fl_use)]
        f2 = _transport(f_down.at(v_end), fl_use)
        segs.append(_Prism(f2, cut_w * 1.2))
        cutters.append(_join([_seg_solid(s, cut_w, cut_h) for s in segs]))
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
    cutters = _lead_hole_cutters(res, info,
                                 w_env + 2 * wrap_total + 2 * gap,
                                 h_env + 2 * wrap_total + 2 * gap)
    gparts, ggrow = _ground_parts(res, w_in, h_in, cutters)
    parts += gparts

    # ---- 防晕层 ----
    parts += _corona_parts(res, w_in + 2 * ggrow + 2 * gap,
                           h_in + 2 * ggrow + 2 * gap)
    return parts


def build_coil_parts(res: CoilResult, detailed: bool | None = None) -> list[CoilPart]:
    """构造线圈部件列表。detailed=None 时按输入 detail_3d 选择。"""
    if detailed is None:
        detailed = res.inp.detail_3d
    return _build_detailed_parts(res) if detailed else _build_simple_parts(res)


def export_step(res: CoilResult, filepath: str,
                detailed: bool | None = None) -> list[str]:
    """构造三维模型并导出 STEP。返回部件名列表。"""
    parts = build_coil_parts(res, detailed)
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
