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
* 逐匝精细模型：将涨形前由一根连续扁铜线绕成的梭形线圈保持材料
  顺序地映射为成型线圈。上、下层槽边匝序相反，两个 nose 的匝位
  沿圆环法向（鼻端中心线，与径向直径成 seita3 角）按 HBD 嵌套，
  如同各匝套在同一鼻端芯轴上——轴向看两条端部斜边收拢成“人”字，
  切向看每个 nose 是一个圆环；接线端的完整 U 形端部光顺接续相邻
  材料匝位，不存在额外接头或独立“爬升坡道”。每股外包自身绝缘，
  每匝外包匝绝缘分层，整束外包对地绝缘方壳（引线穿出处贴身开孔）；
  两根引线位于同一 nose，折弯半径、长度和端头裸铜长可调；
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

# S 形引线在槽底极限匝之外再错开一个匝距，并留此设计净余量。
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


# 浅螺旋高度积分的 Gauss-Legendre 节点/权重（32 点，[-1,1]）。
# 被积函数解析光滑，32 点误差远小于 1e-12 mm，供 LLM 反解使用。
def _gauss_legendre_32() -> tuple[list[float], list[float]]:
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(32)
    return list(nodes), list(weights)


_GL_NODES, _GL_WEIGHTS = _gauss_legendre_32()


@dataclass
class _NoseCurl:
    """鼻端交叉卷环：环平面内的圆 + 沿环法向的浅螺旋错距。

    环平面 = 鼻端处两臂带面（法向 ``n``，0 偏航——盘面与带面贴合，
    不再像独立的硬币斜骑在交叉点上）。交叉处两臂所需的错距
    ``drop`` 沿整个扫角以 smoothstep 剖面消化（锁紧垫圈式浅螺旋，
    升角仅几度，两端斜率为零——与 rd2 圆角所在的平面 G1 对接）：
    入口环端在 ``+drop/2``、出口环端在 ``-drop/2``，环中点（鼻尖）
    严格落在环平面内、位于 seita3 定义的鼻端中心线上。

    ``theta`` 为从入口起沿环的转角；多匝同心嵌套由 ``dy``（环平面
    内的半径增量，节距 HBD）表达，与螺旋高度方向正交。
    """

    c: "b3d.Vector"      # 环平面圆心
    n: "b3d.Vector"      # 环平面单位法向（绕行角动量方向）
    r0: "b3d.Vector"     # 入口处单位半径方向（环平面内）
    radius: float        # 束中心环半径 Rc = RD + WA/2
    tau: float           # 扫角 180° + 2β
    drop: float          # 入口→出口沿 n 的总下降量（有符号）

    def ydir(self, theta: float) -> "b3d.Vector":
        return _rotv(self.r0, self.n, theta)

    def height(self, theta: float) -> float:
        return self.drop * (0.5 - _smoothstep(theta / self.tau))

    def dheight(self, theta: float) -> float:
        """dh/dθ。smoothstep 导数在两端为零。"""
        lam = max(0.0, min(1.0, theta / self.tau))
        return -self.drop * 6.0 * lam * (1.0 - lam) / self.tau

    def point(self, theta: float, dy: float = 0.0) -> "b3d.Vector":
        return (self.c + self.ydir(theta) * (self.radius + dy)
                + self.n * self.height(theta))

    def tangent(self, theta: float, dy: float = 0.0) -> "b3d.Vector":
        y = self.ydir(theta)
        return (self.n.cross(y) * (self.radius + dy)
                + self.n * self.dheight(theta)).normalized()

    def frame(self, theta: float, dy: float = 0.0) -> "_Frame":
        """材料框架：y=环平面内半径方向（同心嵌套匝位方向），
        t=螺旋切向（y⊥t 严格成立），x=y×t。零扭转。"""
        t = self.tangent(theta, dy)
        y = self.ydir(theta)
        return _Frame(self.point(theta, dy), y.cross(t).normalized(), y, t)

    def length(self, dy: float = 0.0) -> float:
        """弧长 ∫√((R+dy)² + (dh/dθ)²) dθ（Gauss-Legendre）。"""
        r = self.radius + dy
        if abs(self.drop) <= 1e-12:
            return r * self.tau
        half = self.tau / 2.0
        total = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            theta = half * (node + 1.0)
            total += weight * math.hypot(r, self.dheight(theta))
        return total * half

    @property
    def ts(self) -> "b3d.Vector":
        return self.point(0.0)

    @property
    def ma(self) -> "b3d.Vector":
        return self.point(self.tau / 2.0)

    @property
    def te(self) -> "b3d.Vector":
        return self.point(self.tau)


@dataclass
class _NoseLayout:
    """两个交叉卷环 nose 的螺旋弧与 rd2 虚拟角点。

    ``pos/neg`` 是 ±Z 两端的鼻端卷环（扫角 ``180°+2β``，两臂
    先交叉、环从交叉点外侧卷回）。``q2/q3/q6/q7`` 是环的入口/出口
    切点（rd2 圆角恰在此结束/开始）；``p2/p3/p6/p7`` 是构造 rd2
    圆角所需的虚拟角点。多匝在环平面内同心嵌套：材料匝 ``i`` 的
    环半径 = 束中心环半径 + HBD 匝位偏移。
    """

    pos: _NoseCurl
    neg: _NoseCurl
    p2: "b3d.Vector"
    p3: "b3d.Vector"
    p6: "b3d.Vector"
    p7: "b3d.Vector"
    q2: "b3d.Vector"
    q3: "b3d.Vector"
    q6: "b3d.Vector"
    q7: "b3d.Vector"


def _fillet_corner(q, p, r_, radius: float,
                   max_fraction: float = 0.45) -> _Fillet:
    """三维角点 p 处的圆角（q、r_ 为前后角点）。

    圆角切线长默认限制在相邻段长度的 45% 内。nose 的 rd2 与
    固定肩基 Q 直接相切时可传入 ``max_fraction=1``，使圆角恰好
    在 Q 结束，不留额外的反向直段。
    """
    u = (p - q).normalized()
    v = (r_ - p).normalized()
    tau = math.acos(max(-1.0, min(1.0, u.dot(v))))
    rad_eff = radius
    if tau > 1e-9:
        max_t = max_fraction * min((p - q).length, (r_ - p).length)
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


def _nose_layout(res: CoilResult, dr: float = 0.0,
                 pose: tuple[float, float] | None = None) -> _NoseLayout:
    """按 ``RD/F/seita3`` 构造 ±Z 两端的交叉卷环 nose。

    每个 nose 是涨形前平面梭形线圈鼻端的原样保留：两条端部斜边
    收拢、**交叉**，然后在尖端外侧卷回一个环，环扫角为
    ``180°+2β``（β 是固定交叉半角 ``_NOSE_CROSS_BETA``），所以
    看上去接近闭合的圆环。

    环平面就是鼻端处两臂的带面（**0 偏航**，盘面与带面贴合）：
    由切向（直径方向）和鼻端中心线张成。鼻端中心线（环顶点方向，
    朝轴向端外）与径向直径严格成 ``seita3`` 角——即专利对 seita3
    的定义，用户改 seita3 时鼻端姿态直接随之旋转；环平面近似贴着
    圆柱面，并随 ``F`` 向槽底抬高。交叉处两臂所需的错距 ``drop``
    （由净距需求反解的最小值）沿整个扫角**均匀**消化——环是升角
    仅几度的浅螺旋（锁紧垫圈式），入口环端在带面上方 ``+drop/2``、
    出口环端在下方 ``-drop/2``，环中点（鼻尖）严格落在带面内的
    鼻端中心线上。环沿鼻端中心线的位置由 LLM 反解
    （``nose_axial_shift``，基准位环心轴向在 ``zn = L2/2 + CC``）：
    总长严格等于专利 LLM——LLM 差值只由环位/直臂长度吸收，姿态
    不参与凑长。

    多匝在环平面内**同心嵌套**：材料匝 ``i`` 的环半径 =
    ``Rc + HBD*(i-(N-1)/2)``——内匝环小、外匝环大，相邻差一个
    鼻端匝间节距 HBD。束中心环半径 ``Rc=RD+WA/2`` 与专利平均匝长
    公式一致。同心卷回天然完成上、下层边"原始外匝分别落向槽口/
    槽底"的匝序翻面，导线全程无拧转扭结。

    ``rd2`` 通过有界方程精确求解，使圆角的一个切点正好是环的
    入口/出口切点 Q（与环 G1 相切）。若在转角不超过 120°、
    路径/弦长不超过 1.5 的紧凑分支内无解，则该参数组合几何
    不可行；不再通过无界延长切线来伪造 ``rd2``。
    """
    inp = res.inp
    xhat = b3d.Vector(1, 0, 0)
    rn = ((res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose + dr)
    zn = res.l2 / 2 + res.cc
    radius = inp.rd_nose + res.wa_turn / 2
    sin3, cos3 = math.sin(inp.seita3), math.cos(inp.seita3)
    if pose is None:
        pose = _nose_pose(res)
    shift, drop = pose
    beta = _NOSE_CROSS_BETA
    sweep = math.pi + 2.0 * beta

    def arc_at(positive: bool) -> _NoseCurl:
        z_sign = 1.0 if positive else -1.0
        # 交叉接入：环端扫过竖直线 β 后其切向越过中线，+Z 端上层
        # 斜边（来自 -X）自然接到 +X 环端、下层斜边接 -X 环端——
        # 两臂在环下方交叉成“人”字；-Z 端（遍历为下层→上层）镜像。
        sx = 1.0 if positive else -1.0
        # 鼻端中心线（环顶点方向）：与径向直径成 seita3，朝轴向端外。
        axis = b3d.Vector(0, cos3, z_sign * sin3)
        d_in = xhat * sx
        # 基准位（shift=0）环心在 zn；按 LLM 反解的 shift 沿鼻端
        # 中心线整体内收（shift>0）或外探（shift<0）。
        center = b3d.Vector(0, rn, z_sign * zn) - axis * shift
        # 法向（环绕行角动量）使入口切向朝轴向端外。
        normal = d_in.cross(axis)
        r0 = d_in * math.cos(beta) - axis * math.sin(beta)
        # -Z 端是 +Z 端的轴向镜像且遍历方向相反（下层→上层），
        # 错距随之翻号，保证两端交叉处都是"来自较深一侧的臂压过
        # 另一臂"，整圈保持镜像对称。
        return _NoseCurl(center, normal, r0, radius, sweep, sx * drop)

    base_pos = arc_at(True)
    base_neg = arc_at(False)
    q2, q3 = base_pos.ts, base_pos.te
    q6, q7 = base_neg.ts, base_neg.te
    th1, th2 = -res.fai1, +res.fai2
    slot1 = _cyl(res.rr1 + res.hc / 2 + dr, th1, +res.l2 / 2)
    slot4 = _cyl(res.rr2 + res.hc / 2 + dr, th2, +res.l2 / 2)
    slot5 = _cyl(res.rr2 + res.hc / 2 + dr, th2, -res.l2 / 2)
    slot0 = _cyl(res.rr1 + res.hc / 2 + dr, th1, -res.l2 / 2)

    def tangent(arc: _NoseCurl, at_start: bool) -> "b3d.Vector":
        return arc.tangent(0.0 if at_start else arc.tau)

    def compact_virtual(slot, endpoint, toward_endpoint, radius2: float,
                        label: str) -> "b3d.Vector":
        """从 slot 反向追踪到 nose 端点的 rd2 虚拟角点。

        令虚拟角点 ``V=Q-T*e``，圆角转角为 ``tau(e)``。
        ``e=rd2*tan(tau/2)`` 时，圆角的切线长恰好等于 VQ，
        因此圆角直接在固定肩基 Q 处与直鼻臂 G1 相切。
        """
        if radius2 <= 1e-9:
            return endpoint

        def state(ext: float):
            virtual = endpoint - toward_endpoint * ext
            incoming = virtual - slot
            if incoming.length <= 1e-9:
                return None
            u = incoming.normalized()
            dot = max(-1.0, min(1.0, u.dot(toward_endpoint)))
            tau = math.acos(dot)
            value = ext - radius2 * math.tan(tau / 2)
            return value, tau, virtual, incoming.length

        lo = 0.0
        # 120° 时 tan(tau/2)=sqrt(3)；这是紧凑端臂允许的最大
        # 有界切线长。70–90° 工作区下实际转角约 56–103°。
        hi = radius2 * math.sqrt(3.0)
        low = state(lo)
        high = state(hi)
        if low is None or high is None or high[0] < -1e-10:
            raise ValueError(
                f"nose 参数组合几何不可行（{label}）："
                f"seita3={math.degrees(inp.seita3):.3f}°, "
                f"RD={inp.rd_nose:.3f}mm, rd2={radius2:.3f}mm。"
                "无法在不折返的紧凑路径内保持相切，请调整参数。")

        for _ in range(64):
            mid = (lo + hi) * 0.5
            current = state(mid)
            if current is None:
                lo = mid
            elif current[0] >= 0:
                hi = mid
            else:
                lo = mid
        solved = state(hi)
        if solved is None:
            raise ValueError(f"nose 参数组合几何不可行（{label}）")
        direct = (endpoint - slot).length
        path_ratio = ((solved[3] + hi) / direct
                      if direct > 1e-9 else float("inf"))
        chord = endpoint - slot
        virtual = solved[2]
        projection_tol = 1e-10 * max(chord.length ** 2, 1.0)
        monotonic = (
            (virtual - slot).dot(chord) > projection_tol and
            (endpoint - virtual).dot(chord) > projection_tol
        )
        if (solved[1] > math.radians(120.0) + 1e-9 or
                path_ratio > 1.5 or not monotonic):
            raise ValueError(
                f"nose 参数组合几何不可行（{label}）："
                f"转角={math.degrees(solved[1]):.2f}°, "
                f"路径/弦长={path_ratio:.3f}。"
                "路径会折返或过度绕行，请调整 seita3、F、RD 或 rd2。")
        return solved[2]

    # rd2 的求解严格以环的入口/出口切点 Q 为目标（G1 相切）。
    p2 = compact_virtual(slot1, q2, tangent(base_pos, True),
                         inp.r_bend_nose, "+Z 上层→nose")
    # 出口段按 slot→nose 反向求解，所以端点方向取负。
    p3 = compact_virtual(slot4, q3, tangent(base_pos, False) * -1.0,
                         inp.r_bend_nose, "+Z 下层→nose")
    p6 = compact_virtual(slot5, q6, tangent(base_neg, True),
                         inp.r_bend_nose, "-Z 下层→nose")
    p7 = compact_virtual(slot0, q7, tangent(base_neg, False) * -1.0,
                         inp.r_bend_nose, "-Z 上层→nose")

    return _NoseLayout(base_pos, base_neg, p2, p3, p6, p7, q2, q3, q6, q7)


def _loop_fillets(res: CoilResult, dr: float = 0.0,
                  pose: tuple[float, float] | None = None):
    """径向偏移 dr 处线圈环路的 8 个角点及圆角。

    角点顺序：0/1 上层边非接线/接线侧槽口，2/3 接线侧鼻端角，
    4/5 下层边接线/非接线侧槽口，6/7 非接线侧鼻端角。
    """
    inp = res.inp
    r1c = res.rr1 + res.hc / 2 + dr     # 上层边中心半径
    r2c = res.rr2 + res.hc / 2 + dr     # 下层边中心半径
    th1 = -res.fai1
    th2 = +res.fai2
    zl = res.l2 / 2
    nose = _nose_layout(res, dr, pose)

    corners = [
        (_cyl(r1c, th1, -zl), inp.r_bend_slot),   # 0 上层边·非接线侧槽口
        (_cyl(r1c, th1, +zl), inp.r_bend_slot),   # 1 上层边·接线侧槽口
        (nose.p2, inp.r_bend_nose),                # 2 接线侧斜边-鼻端弯角 rd2
        (nose.p3, inp.r_bend_nose),                # 3 接线侧鼻端-斜边弯角 rd2
        (_cyl(r2c, th2, +zl), inp.r_bend_slot),   # 4 下层边·接线侧槽口
        (_cyl(r2c, th2, -zl), inp.r_bend_slot),   # 5 下层边·非接线侧槽口
        (nose.p6, inp.r_bend_nose),                # 6 非接线侧斜边-鼻端弯角 rd2
        (nose.p7, inp.r_bend_nose),                # 7 非接线侧鼻端-斜边弯角 rd2
    ]

    p = [corner[0] for corner in corners]
    # 四个 rd2 圆角必须恰在环的入口/出口切点 Q 结束（G1 相切）。
    neighbors = [
        (p[7], p[0], p[1]),
        (p[0], p[1], p[2]),
        (p[1], p[2], nose.q2),
        (nose.q3, p[3], p[4]),
        (p[3], p[4], p[5]),
        (p[4], p[5], p[6]),
        (p[5], p[6], nose.q6),
        (nose.q7, p[7], p[0]),
    ]
    fillets = [_fillet_corner(q, point, nxt, corners[i][1],
                              1.0 if i in (2, 3, 6, 7) else 0.45)
               for i, (q, point, nxt) in enumerate(neighbors)]
    return corners, fillets


def _nose_curl_edge(curl: _NoseCurl) -> "b3d.Edge":
    """卷环束中心线的精确 OCC 边。

    浅螺旋高度剖面 ``h(λ)`` 是三次多项式（smoothstep），因此
    (u=τλ, v=h(λ)) 是柱面上的一条**精确三次 Bézier** p-curve——
    几何零误差；3D 表示按 1e-9 容差重建，供 Wire 长度校核使用。
    drop=0 时退化为精确圆弧。
    """
    if abs(curl.drop) <= 1e-12:
        return b3d.ThreePointArc(curl.ts, curl.ma, curl.te)

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepLib import BRepLib
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_BezierCurve
    from OCP.GeomAbs import GeomAbs_Shape
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt, gp_Pnt2d
    from OCP.TColgp import TColgp_Array1OfPnt2d

    axis = gp_Ax3(gp_Pnt(*curl.c), gp_Dir(*curl.n), gp_Dir(*curl.r0))
    surface = Geom_CylindricalSurface(axis, curl.radius)
    half = curl.drop / 2.0
    poles = TColgp_Array1OfPnt2d(1, 4)
    poles.SetValue(1, gp_Pnt2d(0.0, half))
    poles.SetValue(2, gp_Pnt2d(curl.tau / 3.0, half))
    poles.SetValue(3, gp_Pnt2d(2.0 * curl.tau / 3.0, -half))
    poles.SetValue(4, gp_Pnt2d(curl.tau, -half))
    pcurve = Geom2d_BezierCurve(poles)
    edge = BRepBuilderAPI_MakeEdge(pcurve, surface).Edge()
    BRepLib.BuildCurves3d_s(edge, 1e-9, GeomAbs_Shape.GeomAbs_C2, 14, 200)
    return b3d.Edge(edge)


def _centerline_edges(res: CoilResult,
                      pose: tuple[float, float] | None = None):
    """构造束中心闭合中心线的解析边及圆角。

    ``pose=(shift, drop)`` 为卷环姿态；``None`` 使用自动反解值。
    """
    _corners, fillets = _loop_fillets(res, 0.0, pose)
    nose = _nose_layout(res, 0.0, pose)
    edges = []

    def add_fillet(i: int, target: "b3d.Vector") -> None:
        fil = fillets[i]
        edges.append(b3d.ThreePointArc(fil.ts, fil.ma, fil.te))
        if (target - fil.te).length > 1e-6:
            edges.append(b3d.Line(fil.te, target))

    add_fillet(0, fillets[1].ts)
    add_fillet(1, fillets[2].ts)
    add_fillet(2, nose.pos.ts)
    edges.append(_nose_curl_edge(nose.pos))
    if (fillets[3].ts - nose.pos.te).length > 1e-6:
        edges.append(b3d.Line(nose.pos.te, fillets[3].ts))
    add_fillet(3, fillets[4].ts)
    add_fillet(4, fillets[5].ts)
    add_fillet(5, fillets[6].ts)
    add_fillet(6, nose.neg.ts)
    edges.append(_nose_curl_edge(nose.neg))
    if (fillets[7].ts - nose.neg.te).length > 1e-6:
        edges.append(b3d.Line(nose.neg.te, fillets[7].ts))
    add_fillet(7, fillets[0].ts)
    return edges, fillets


# 鼻端卷环的固定交叉半角 β：环扫角 = 180° + 2β。取值使两臂在环
# 下方明显交叉、环观感接近闭合，与真实涨形线圈照片一致。
_NOSE_CROSS_BETA = math.radians(25.0)

# 浅螺旋错距 drop 的反解上限 mm（超出说明参数组合无法在贴合带面的
# 前提下满足交叉净距）。
_NOSE_DROP_MAX = 60.0

# 环位反解的搜索范围（沿鼻端中心线，正值向铁芯内收）。
_NOSE_SHIFT_MIN = -80.0
_NOSE_SHIFT_MAX = 400.0

# 交叉处两臂中心线净距的额外裕量 mm（截面斜置与采样离散的保守量）。
_NOSE_CROSS_MARGIN = 1.5


def _segment_distance(a0, a1, b0, b1) -> float:
    """两条三维线段的最小距离。"""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = u.dot(u)
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)
    denom = a * c - b * b
    if denom > 1e-12:
        s = max(0.0, min(1.0, (b * e - c * d) / denom))
    else:
        s = 0.0
    t = (b * s + e) / c if c > 1e-12 else 0.0
    t = max(0.0, min(1.0, t))
    # 再夹取一次 s（t 被截断后最优 s 会变化）
    if c > 1e-12:
        s = max(0.0, min(1.0, (b * t - d) / a)) if a > 1e-12 else 0.0
    return (a0 + u * s - b0 - v * t).length


def _nose_cross_fiber_slack(res: CoilResult,
                            pose: tuple[float, float]) -> float:
    """+Z 鼻端交叉区**逐匝纤维**净距裕量（最小值，≥0 即无干涉）。

    入口臂（斜边 1→2 + rd2 弯角 2）与出口臂（rd2 弯角 3 + 斜边
    3→4）各含 N 匝纤维（匝位偏移沿材料 y 轴从槽部 HAD 连续过渡到
    鼻端 HBD）。对两臂纤维的采样点对，要求中心距 ≥ 双方截面沿
    连线方向的支撑半宽（含匝绝缘与对地壳厚）之和 + 裕量——束
    中心线级的净距无法覆盖极限匝纤维在交叉区互相斜穿的情形
    （v202607142334 已知问题 1 的根源）。-Z 鼻端与 +Z 严格镜像，
    不再重复计算。
    """
    import numpy as np

    from .engine import strand_grid

    inp = res.inp
    n = inp.n_turns
    w_env, h_env, _strands = strand_grid(inp)
    wrap = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
    ground = sum(l.thickness for l in inp.layers if l.thickness > 0)
    half_x = (w_env + 2 * wrap) / 2 + ground
    h_cap = res.had if n > 1 else h_env + 2 * wrap
    half_y = min(h_env + 2 * wrap, h_cap) / 2 + ground

    lf = _loop_frames(res, pose)
    fl = lf.fl

    def dy_slot(i: int) -> float:
        return res.had * (i - (n - 1) / 2)

    def dy_nose(i: int) -> float:
        return res.hbd * (i - (n - 1) / 2)

    def slant_samples(key, dy0: float, dy1: float, count: int = 32):
        f0, f1 = lf.slant_frames[key]
        law, _slope = lf.slant_laws[key]
        out = []
        for k in range(count + 1):
            lam = k / count
            f = _twist_frame_at(f0, f1, lam, law)
            dy = dy0 + (dy1 - dy0) * lam
            out.append((f.o + f.y * dy, f.x, f.y))
        return out

    def corner_samples(f: _Frame, fil: _Fillet, dy: float,
                       count: int = 24):
        p0 = f.o + f.y * dy
        out = []
        for k in range(count + 1):
            a = fil.tau * k / count
            out.append((fil.c + _rotv(p0 - fil.c, fil.n, a),
                        _rotv(f.x, fil.n, a), _rotv(f.y, fil.n, a)))
        return out

    f12e = lf.slant_frames[(1, 2)][1]        # 位于 fl[2].ts
    entry, exit_ = [], []
    for i in range(n):
        ds, dn = dy_slot(i), dy_nose(i)
        entry += slant_samples((1, 2), ds, dn)
        entry += corner_samples(f12e, fl[2], dn)
        exit_ += corner_samples(lf.f_q3, fl[3], dn)
        exit_ += slant_samples((3, 4), dn, ds)

    def pack(samples):
        pts = np.asarray([tuple(p) for p, _x, _y in samples])
        xs = np.asarray([tuple(x) for _p, x, _y in samples])
        ys = np.asarray([tuple(y) for _p, _x, y in samples])
        return pts, xs, ys

    pa, xa, ya = pack(entry)
    pb, xb, yb = pack(exit_)
    diff = pa[:, None, :] - pb[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.maximum(dist, 1e-12)
    dirs = diff / dist[..., None]
    ra = (np.abs(np.einsum("abk,ak->ab", dirs, xa)) * half_x
          + np.abs(np.einsum("abk,ak->ab", dirs, ya)) * half_y)
    rb = (np.abs(np.einsum("abk,bk->ab", dirs, xb)) * half_x
          + np.abs(np.einsum("abk,bk->ab", dirs, yb)) * half_y)
    slack = dist - ra - rb - 2.0 * _NOSE_CROSS_MARGIN
    return float(slack.min())


def _nose_pose(res: CoilResult) -> tuple[float, float]:
    """求解卷环姿态 ``(shift, drop)`` 并缓存在结果对象上。

    ``shift``：环沿鼻端中心线的位置，由闭合中心线总长 == LLM
    反解；``drop``：浅螺旋错距（入口环端 ``+drop/2``、出口环端
    ``-drop/2`` 沿环法向），使交叉处入口/出口两臂中心线的净距达到
    束宽加包覆绝缘的需求。环平面本身始终与带面贴合（0 偏航），
    seita3 因而直接控制鼻端姿态。二者耦合，外层对 drop 迭代、
    内层对 shift 二分。
    """
    cached = getattr(res, "_nose_pose3d", None)
    if cached is not None:
        return cached

    def slack_at(drop: float) -> float:
        shift = nose_axial_shift(res, drop)
        return _nose_cross_fiber_slack(res, (shift, drop))

    # 交叉裕量随 drop 单调增大；对 drop 二分求"恰好无干涉"的最小
    # 错距（螺旋升角尽可能小、盘面观感最平）。
    drop_lo, drop_hi = 0.0, _NOSE_DROP_MAX
    if slack_at(drop_lo) >= 0.0:
        drop = drop_lo
    else:
        if slack_at(drop_hi) < 0.0:
            raise ValueError(
                "nose 交叉处两臂间距无法满足束宽加绝缘的净距需求，"
                "请调整 RD、rd2、F 或主体尺寸。")
        for _ in range(14):
            mid = (drop_lo + drop_hi) * 0.5
            if slack_at(mid) >= 0.0:
                drop_hi = mid
            else:
                drop_lo = mid
        drop = drop_hi
    shift = nose_axial_shift(res, drop)

    pose = (shift, drop)
    try:
        res._nose_pose3d = pose
    except Exception:
        pass
    return pose


def _centerline_length(res: CoilResult, pose: tuple[float, float]) -> float:
    """闭合束中心线的解析总长（与 ``_centerline_edges`` 等价，
    但不构造 OCC 边，供环位反解的二分迭代高速调用）。"""
    _corners, fillets = _loop_fillets(res, 0.0, pose)
    nose = _nose_layout(res, 0.0, pose)

    def arc_len(seg) -> float:
        if isinstance(seg, _NoseCurl):
            return seg.length()
        return (seg.ts - seg.c).length * seg.tau

    order = [fillets[0], fillets[1], fillets[2], nose.pos,
             fillets[3], fillets[4], fillets[5], fillets[6], nose.neg,
             fillets[7]]
    total = 0.0
    for left, right in zip(order, order[1:] + order[:1]):
        total += arc_len(left)
        total += (right.ts - left.te).length
    return total


def nose_axial_shift(res: CoilResult, drop: float | None = None) -> float:
    """由专利平均匝长反解卷环沿鼻端中心线的位置偏移。

    偏移增大（环向铁芯内收）时两侧斜边缩短、闭合中心线单调变短；
    用二分求得总长严格等于 LLM 的偏移。基准位（偏移=0）环心轴向
    在 ``zn = L2/2 + CC``。
    """
    if drop is None:
        return _nose_pose(res)[0]

    def length_at(shift: float) -> float:
        return _centerline_length(res, (shift, drop))

    def bracket(lo: float, hi: float, samples: int = 24):
        """在可行区间内寻找 length-LLM 变号的一段。"""
        prev = None
        for k in range(samples + 1):
            s = lo + (hi - lo) * k / samples
            try:
                diff = length_at(s) - res.llm
            except ValueError:
                if prev is not None:
                    break
                continue
            if diff == 0.0:
                return s, s
            if prev is not None and prev[1] * diff < 0:
                return prev[0], s
            prev = (s, diff)
        return None

    span = bracket(_NOSE_SHIFT_MIN, _NOSE_SHIFT_MAX)
    if span is None:
        raise ValueError(
            "nose 卷环位置无法匹配专利平均匝长："
            f"LLM={res.llm:.6f}mm 超出可行几何范围。"
            "请调整 RD、rd2、F 或主体尺寸。")
    lo, hi = span
    if lo == hi:
        return lo
    lo_diff = length_at(lo) - res.llm
    for _ in range(64):
        mid = (lo + hi) * 0.5
        diff = length_at(mid) - res.llm
        if diff == 0.0:
            return mid
        if lo_diff * diff < 0:
            hi = mid
        else:
            lo, lo_diff = mid, diff
        if hi - lo <= 1e-11 * max(1.0, abs(hi)):
            break
    return (lo + hi) * 0.5


def build_centerline(res: CoilResult) -> tuple["b3d.Wire", "b3d.Vector", "b3d.Vector", "b3d.Vector"]:
    """线圈束中心闭合中心线；总长严格等于专利计算值 LLM。"""
    edges, fillets = _centerline_edges(res)
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


# 分段描述。dy/dy0/dy1 为截面沿 y 的附加匝位偏移：槽部节距为 HAD，
# nose 立面节距为 HBD，二者在端部斜边上连续插值。
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
    """任意截面的仿射基。

    ``ex``/``ey`` 可表示正交材料截面，也可表示直纹放样中的仿射
    截面；保留二者的实际长度，避免强制归一破坏端面对接。
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


@dataclass
class _MultiLoft:
    """沿一组绝对材料截面生成的单个多截面放样。

    ``breaks`` 把截面序列分成若干组（共享边界截面后分别放样）；
    ``ruled`` 逐组指定直纹（True，与 ``_twisted_lofts`` 的弦式直纹
    墙逐点一致）或光顺 B 样条（False，圆冠等固定姿态段）。空缺时
    默认全部光顺。
    """

    sections: list[_Basis]
    breaks: tuple[int, ...] = ()  # G1 曲率分段点（共享该截面后分别放样）
    ruled: tuple[bool, ...] = ()  # 每组是否直纹放样（与分组一一对应）


@dataclass
class _LoopFrames:
    """束中心闭合环路的材料框架（匝与方壳共用）。

    ``y`` 不只是截面纵轴，也表示涨形前梭形线圈由内到外的匝位方向：

    * 上层直边：径向内（原始外匝落向槽口）；
    * 下层直边：径向外（原始外匝落向槽底）；
    * 端部斜边：靠槽口的大部分保持槽内姿态（窄边朝向轴线，任意
      端面同心圆与线圈束截面的交弧远小于槽距）；槽侧姿态到卷环
      姿态的截面过渡压缩在斜边临近鼻部的 ``_SLANT_TWIST_ZONE``
      区段内完成——这就是真实线圈的“扭转段”；
    * rd2 弯角、卷环：**零扭转**。卷环截面的材料 y 轴 = 环平面内
      指向环外的半径方向，多匝在环平面内按 HBD **同心嵌套**
      （dy>0 = 环半径更大 = 原始外匝）；同心卷回天然完成上、下层
      边的匝序翻面，鼻部没有任何拧转。
    """

    fl: list                     # 8 个 _Fillet
    f_leg1: _Frame               # 上层边锚定框架
    f_leg2: _Frame               # 下层边锚定框架
    f_flat_pos: _Frame           # +Z 卷环入口框架（y=环内半径向外）
    f_flat_pos_out: _Frame       # +Z 卷环出口框架
    f_flat_neg: _Frame           # -Z 卷环入口框架
    f_flat_neg_out: _Frame       # -Z 卷环出口框架
    nose_pos: _NoseCurl          # +Z 交叉卷环（扫角 180°+2β，浅螺旋）
    nose_neg: _NoseCurl          # -Z 交叉卷环
    slant_frames: dict           # {(i_from,i_to): (f_start, f_end)} 斜边两端框架
    slant_laws: dict             # {(i_from,i_to): (law, slope)} 斜边扭转分布律
    f_q2: _Frame                 # +Z 入口 rd2 弯角出口框架（Q2，卷环姿态）
    f_q3: _Frame                 # +Z 出口 rd2 弯角入口框架（Q3，卷环姿态）
    f_q6: _Frame                 # -Z 入口 rd2 弯角出口框架（Q6，卷环姿态）
    f_q7: _Frame                 # -Z 出口 rd2 弯角入口框架（Q7，卷环姿态）


# 斜边扭转段占斜边长度的比例（贴着鼻部一侧）。其余部分保持槽内
# 姿态，与真实线圈"临近鼻部集中扭转"的形态一致。
_SLANT_TWIST_ZONE = 0.35


def _smoothstep(lam: float) -> float:
    lam = max(0.0, min(1.0, lam))
    return lam * lam * (3.0 - 2.0 * lam)


def _slant_twist_law(nose_at_end: bool,
                     zone: float = _SLANT_TWIST_ZONE):
    """返回斜边扭转分布律 ``(law, slope)``。

    ``law(lam)`` 单调 0→1，扭转集中在贴近鼻部的 ``zone`` 区段内
    （smoothstep，两端速率为零，与相邻刚性段 G1 对接）；其余部分
    保持起始姿态。``slope`` 是该律的最大斜率，用于估算放样段数。
    """
    zone = min(1.0, max(1e-6, zone))

    if nose_at_end:
        def law(lam: float) -> float:
            return _smoothstep((lam - (1.0 - zone)) / zone)
    else:
        def law(lam: float) -> float:
            return _smoothstep(lam / zone)
    return law, 1.5 / zone


def _mirrored_law(law):
    """反向遍历同一段时的扭转分布律。"""
    def mirrored(lam: float) -> float:
        return 1.0 - law(1.0 - lam)
    return mirrored


def _loop_frames(res: CoilResult,
                 pose: tuple[float, float] | None = None) -> _LoopFrames:
    corners, fl = _loop_fillets(res, 0.0, pose)
    c = [x[0] for x in corners]
    nose = _nose_layout(res, 0.0, pose)

    f_leg1 = _anchor(fl[0].te, c[1] - c[0],
                     _radial_of((c[0] + c[1]) * 0.5) * -1.0)
    f_leg2 = _anchor(fl[4].te, c[5] - c[4],
                     _radial_of((c[4] + c[5]) * 0.5))

    # 斜边槽侧端：槽口弯角刚性传递的槽内姿态。
    f12s = _transport(f_leg1.at(fl[1].ts), fl[1])
    f56s = _transport(f_leg2.at(fl[5].ts), fl[5])
    f34e = _pre_corner(f_leg2.at(fl[4].te), fl[4])
    f70e = _pre_corner(f_leg1.at(fl[0].te), fl[0])

    # 卷环姿态：材料 y 轴 = 环平面内指向环外的半径方向（同心嵌套
    # 的匝位方向，dy>0 = 环更大 = 原始外匝）。rd2 弯角零扭转：斜边
    # 鼻侧端框架由卷环姿态经弯角刚性反推/传递。浅螺旋高度剖面在
    # 环两端斜率为零，故环端切向就在环平面内，框架与平面 rd2 圆角
    # 严格 G1 对接。
    def crown_frames(arc: _NoseCurl, entry_corner: int):
        f_in = arc.frame(0.0)
        f_end = _pre_corner(f_in.at(fl[entry_corner].te),
                            fl[entry_corner]).at(fl[entry_corner].ts)
        f_out = arc.frame(arc.tau)
        return f_in, f_out, f_end

    f_flat_pos, f_flat_pos_out, f12e = crown_frames(nose.pos, 2)
    f_flat_neg, f_flat_neg_out, f56e = crown_frames(nose.neg, 6)

    # 出口侧：卷环出口姿态经 rd2 刚性传入斜边鼻侧端。
    f_q2 = f_flat_pos.at(fl[2].te)
    f_q3 = f_flat_pos_out.at(fl[3].ts)
    f_q6 = f_flat_neg.at(fl[6].te)
    f_q7 = f_flat_neg_out.at(fl[7].ts)
    f34s = _transport(f_q3, fl[3])
    f70s = _transport(f_q7, fl[7])

    slants = {(1, 2): (f12s, f12e), (3, 4): (f34s, f34e),
              (5, 6): (f56s, f56e), (7, 0): (f70s, f70e)}
    laws = {(1, 2): _slant_twist_law(nose_at_end=True),
            (3, 4): _slant_twist_law(nose_at_end=False),
            (5, 6): _slant_twist_law(nose_at_end=True),
            (7, 0): _slant_twist_law(nose_at_end=False)}
    return _LoopFrames(fl, f_leg1, f_leg2,
                       f_flat_pos, f_flat_pos_out,
                       f_flat_neg, f_flat_neg_out,
                       nose.pos, nose.neg, slants, laws,
                       f_q2, f_q3, f_q6, f_q7)


def _loop_segments(res: CoilResult) -> list:
    """一圈闭合环路的分段序列（方壳/简化束用）。

    直线边为锚定棱柱，圆角为旋转体，nose 为浅螺旋密站位直纹放样。
    槽侧姿态到圆冠姿态的截面扭转压缩在斜边贴近鼻部的扭转段内；
    rd2 弯角、直鼻臂和卷环零扭转。
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
        law, slope = lf.slant_laws[(i_from, i_to)]
        segs.extend(_twisted_lofts(f_start, f_end, law=law, slope=slope))
        segs.append(_Rev(f_end, fl[i_to]))

    def nose(f_q_in: _Frame, f_q_out: _Frame,
             f_crown_in: _Frame, f_crown_out: _Frame, arc: _NoseCurl,
             i_enter: int, i_exit: int):
        pre_len = (arc.ts - fl[i_enter].te).length
        if pre_len > 1e-6:
            segs.extend(_twisted_lofts(f_q_in.at(fl[i_enter].te),
                                       f_crown_in.at(arc.ts)))
        segs.append(_nose_transition(arc, 0.0, 0.0))
        post_len = (fl[i_exit].ts - arc.te).length
        if post_len > 1e-6:
            segs.extend(_twisted_lofts(f_crown_out.at(arc.te),
                                       f_q_out.at(fl[i_exit].ts)))
        segs.append(_Rev(f_q_out.at(fl[i_exit].ts), fl[i_exit]))

    leg(lf.f_leg1, 0, 1)        # 上层边 0→1，角1
    slant(1, 2)                 # 斜边 1→2，角2
    nose(lf.f_q2, lf.f_q3, lf.f_flat_pos, lf.f_flat_pos_out,
         lf.nose_pos, 2, 3)
    slant(3, 4)                 # 斜边 3→4，角4
    leg(lf.f_leg2, 4, 5)        # 下层边 4→5，角5
    slant(5, 6)                 # 斜边 5→6，角6
    nose(lf.f_q6, lf.f_q7, lf.f_flat_neg, lf.f_flat_neg_out,
         lf.nose_neg, 6, 7)
    slant(7, 0)                 # 斜边 7→0，角0（闭合）
    return segs


def _twist_angle(f0: _Frame, f1: _Frame) -> float:
    """f0→f1 绕弦线的有符号材料扭转角。"""
    chord = f1.o - f0.o
    if chord.length <= 1e-12:
        return 0.0
    t = chord.normalized()
    return math.atan2(t.dot(f0.y.cross(f1.y)), f0.y.dot(f1.y))


def _twist_frame_at(f0: _Frame, f1: _Frame, lam: float,
                    law=None) -> _Frame:
    """f0→f1 直线段上参数 ``lam`` 处的正交材料框架。

    扭转角按 ``law``（默认全段 smoothstep）分布：分布律两端速率为
    零，因此扭转段与前后刚性解析段（弯角旋转体、圆冠）的截面姿态
    在接口处 G1 对接。
    """
    lam = max(0.0, min(1.0, lam))
    if lam <= 1e-12:
        return f0
    if lam >= 1.0 - 1e-12:
        return f1
    chord = f1.o - f0.o
    t = chord.normalized()
    angle = _twist_angle(f0, f1)
    blend = _smoothstep(lam) if law is None else law(lam)
    o = f0.o * (1.0 - lam) + f1.o * lam
    y = _rotv(f0.y, t, angle * blend).normalized()
    return _Frame(o, y.cross(t).normalized(), y, t)


def _offset_slant_basis_at(f0: _Frame, f1: _Frame,
                           lam: float, dy: float) -> _Basis:
    """斜边偏移纤维在 ``lam`` 处的截面基及数值切向。"""
    f = _twist_frame_at(f0, f1, lam)
    eps = min(1e-4, max(lam, 1.0 - lam, 1e-4))
    la = max(0.0, lam - eps)
    lb = min(1.0, lam + eps)
    fa = _twist_frame_at(f0, f1, la)
    fb = _twist_frame_at(f0, f1, lb)
    pa = fa.o + fa.y * dy
    pb = fb.o + fb.y * dy
    t = (pb - pa).normalized()
    return _Basis(f.o, f.x, f.y, t)


def _twisted_lofts(f0: _Frame, f1: _Frame,
                    dy0: float = 0.0, dy1: float = 0.0,
                    max_twist: float = math.radians(30.0),
                    law=None, slope: float = 1.5) -> list[_Loft]:
    """用若干正交中间截面构造一条扭转斜边。

    单个两截面直纹放样在约 90° 扭转时会把中间截面压缩成菱形，甚至
    让相邻匝互相侵入。这里沿直线段切成若干段，材料框架绕路径切向按
    ``law``（默认全段 smoothstep）旋转，分布律两端速率为零；每一
    小段的扭转不超过 ``max_twist``（按分布律最大斜率 ``slope``
    估算段数）。``dy`` 也沿整段连续插值，供接线端 nose 将一根连续
    导线光顺排到相邻匝位。
    """
    chord = f1.o - f0.o
    if chord.length <= 1e-9:
        return []
    angle = _twist_angle(f0, f1)
    count = max(1, math.ceil(abs(angle) * max(slope, 1.0)
                             / max(max_twist, 1e-6)))

    frames: list[_Frame] = []
    dys: list[float] = []
    for j in range(count + 1):
        lam = j / count
        frames.append(_twist_frame_at(f0, f1, lam, law))
        dys.append(dy0 * (1.0 - lam) + dy1 * lam)
    return [_Loft(frames[j], frames[j + 1], dys[j], dys[j + 1])
            for j in range(count)]


def _nose_transition(arc: _NoseCurl, dy0: float, dy1: float,
                     count: int = 32) -> _MultiLoft:
    """沿交叉卷环把匝位从 ``dy0`` 光顺换到 ``dy1``（同心螺旋入环）。

    匝位偏移 dy 沿环平面内的旋转半径方向施加：换匝曲线是从半径
    ``Rc+dy0`` 平滑过渡到 ``Rc+dy1`` 的同心螺旋，与涨形前梭形
    线圈相邻两匝在鼻端的自然接续一致；``dy0==dy1`` 时即匝位不变
    的整环。卷环自身的浅螺旋错距（``_NoseCurl.height``）与匝位
    无关地叠加在环法向上。smoothstep 的一阶导数在两端为零，因此
    与前后的 rd2 圆角 G1 对接；截面 ex 恒为环平面法向（与 dy 和
    错距无关），相邻匝截面严格共面嵌套、匝间只共享边界。

    放样按**密站位直纹**：相邻截面间线性插值，SolidWorks 兼容的
    split-5 条带才能与整环放样严格互补（体积守恒、只共享边界）；
    站位角步距 ≤4°，弦高误差远小于制造公差。
    """
    radius = arc.radius
    if radius <= 1e-9 or arc.tau <= 1e-9:
        raise ValueError("nose 卷环弧长必须大于 0")
    count = max(int(count), math.ceil(arc.tau / math.radians(4.0)))
    total = radius * arc.tau
    delta = dy1 - dy0

    sections: list[_Basis] = []
    for j in range(count + 1):
        lam = j / count
        theta = arc.tau * lam
        y = arc.ydir(theta)
        planar_t = arc.n.cross(y)
        blend = _smoothstep(lam)
        dy = dy0 + delta * blend
        dy_rate = delta * 6.0 * lam * (1.0 - lam) / total
        tangent = (planar_t * ((radius + dy) / radius)
                   + y * dy_rate
                   + arc.n * (arc.dheight(theta) / radius)).normalized()
        center = arc.point(theta, dy)
        x = y.cross(planar_t).normalized()   # = ±环平面法向
        sections.append(_Basis(center, x, y, tangent))
    return _MultiLoft(sections, (), (True,))


def _wire_segments(res: CoilResult):
    """一根连续导线从一根引线到另一根引线的完整分段序列。

    匝号 ``i`` 按涨形前梭形线圈由内到外编号。材料框架使同一个
    槽部 ``dy_slot(i)`` 在上层落向槽底、在下层落向槽口；两个 nose
    则用 ``dy_nose(i)`` 沿环平面内的半径方向按 HBD **同心嵌套**
    （内匝环小、外匝环大）。两种节距之差在端部斜边上连续过渡。

    为得到只有两个自由端的一条开曲线，这里从“下层槽底匝”反向
    追踪到“上层槽底匝”：非接线端连接同一材料匝位；接线端沿交叉
    卷环把上层 ``i`` 以同心螺旋光顺接到下层 ``i-1``，不存在额外
    接头、台阶或独立爬升坡道。

    几何先按接线端位于 +Z 构造；若用户选择 -Z，最终所有部件关于
    XY 中面整体镜像。返回 ``(segs, info)``，其中 info 供对地绝缘
    的两个贴身引线孔复用。
    """
    inp = res.inp
    n = inp.n_turns
    zhat = b3d.Vector(0, 0, 1)

    lf = _loop_frames(res)
    fl = lf.fl

    def dy_slot(i: int) -> float:
        """槽部按 HAD 排列的材料匝位。"""
        return res.had * (i - (n - 1) / 2)

    def dy_nose(i: int) -> float:
        """鼻端环平面内按 HBD 同心嵌套的材料匝位（环半径增量）。"""
        return res.hbd * (i - (n - 1) / 2)

    # ysc 表示折弯后的自由直段长度；折弯本身另由 lead_bend_r 控制。
    # 两根引线均从槽底极限匝向更深槽底方向作一对相反的等角圆弧，
    # 再恢复为 +Z 直出。若在槽内直边上直接轴向延长，路径会与相邻匝
    # 的槽口圆角同切向并随后穿体；这个 S 形错位从起点即向相反方向
    # 分离，同时保持起、终切向都为轴向。
    min_bend_r = res.had / 2.0
    if inp.lead_bend_r <= min_bend_r + 1e-9:
        raise ValueError(
            "引线错位圆弧几何不可行："
            f"折弯半径={inp.lead_bend_r:.3f}mm，必须大于匝外高一半"
            f" {min_bend_r:.3f}mm，否则圆弧内侧会自交。")
    rb0 = inp.lead_bend_r
    lead_len = max(inp.ysc, 1.0)

    f34s, f34e = lf.slant_frames[(3, 4)]      # +Z nose → 下层
    f56s, f56e = lf.slant_frames[(5, 6)]
    f70s, f70e = lf.slant_frames[(7, 0)]
    f12s, f12e = lf.slant_frames[(1, 2)]      # 上层 → +Z nose

    def slant_lofts(key, f_start, f_end, dy0, dy1):
        law, slope = lf.slant_laws[key]
        return _twisted_lofts(f_start, f_end, dy0, dy1,
                              law=law, slope=slope)

    def add_constant_nose(f_q_in: _Frame, f_q_out: _Frame,
                          f_crown_in: _Frame, f_crown_out: _Frame,
                          arc: _NoseCurl,
                          i_enter: int, i_exit: int, d: float) -> None:
        pre_len = (arc.ts - fl[i_enter].te).length
        if pre_len > 1e-6:
            segs.extend(_twisted_lofts(f_q_in.at(fl[i_enter].te),
                                       f_crown_in.at(arc.ts), d, d))
        segs.append(_nose_transition(arc, d, d))
        post_len = (fl[i_exit].ts - arc.te).length
        if post_len > 1e-6:
            segs.extend(_twisted_lofts(f_crown_out.at(arc.te),
                                       f_q_out.at(fl[i_exit].ts), d, d))
        segs.append(_Rev(f_q_out.at(fl[i_exit].ts), fl[i_exit], dy=d))

    def add_transition_nose(d0: float, d1: float) -> None:
        # 换匝沿交叉卷环按弧长连续展开（同心螺旋入环）；rd2 仍
        # 分别保持入口/出口的完整 HBD 匝位。
        segs.append(_nose_transition(lf.nose_pos, d0, d1))
        segs.append(_Rev(lf.f_q3.at(fl[3].ts), fl[3], dy=d1))

    def circular_arc(start: "b3d.Vector", center: "b3d.Vector",
                     normal: "b3d.Vector", angle: float) -> _Fillet:
        """由起点、圆心、正向法线和转角构造精确圆弧。"""
        radius = start - center
        return _Fillet(
            start,
            center + _rotv(radius, normal, angle / 2),
            center + _rotv(radius, normal, angle),
            center,
            normal,
            angle,
        )

    def reverse_frame(frame: _Frame, origin: "b3d.Vector" | None = None) -> _Frame:
        """保持截面材料 y 轴，将路径行进方向及 x 轴反转。"""
        return _Frame(origin if origin is not None else frame.o,
                      frame.x * -1.0, frame.y,
                      frame.t * -1.0)

    def reverse_arc(arc: _Fillet) -> _Fillet:
        return _Fillet(arc.te, arc.ma, arc.ts, arc.c,
                       arc.n * -1.0, arc.tau)

    def lead_dogleg(join: "b3d.Vector", slot_frame: _Frame,
                    away: "b3d.Vector") -> tuple[list, list, "b3d.Vector", _Frame]:
        """构造槽底方向 S 形错位引线。

        返回 ``(正向路径, 反向路径, 自由端, 槽内方向框架)``。正向为
        槽内直边 ``join`` → +Z 自由端；反向用于第一根引线从自由端
        追踪回线圈。两段圆弧半径均严格等于 lead_bend_r；转角按“一个
        匝距 + 设计余量”的最小横移量求得，避免把引线抬离槽底过远。
        当目标横移大于 ``2R`` 时，两段 90° 圆弧之间补一段槽底方向
        直线，不会暗中改变用户给定的折弯半径。
        """
        away = away.normalized()
        bend_axis = zhat.cross(away).normalized()
        target_shift = res.had + _LEAD_CLEARANCE
        if target_shift <= 2.0 * rb0:
            angle = math.acos(1.0 - target_shift / (2.0 * rb0))
            bridge = 0.0
        else:
            angle = math.pi / 2
            bridge = target_shift - 2.0 * rb0
        # 出槽方向为 +Z；保留槽内截面的材料 y 轴。
        f0 = _Frame(join, slot_frame.y.cross(zhat).normalized(),
                    slot_frame.y, zhat)
        arc1 = circular_arc(join, join + away * rb0, bend_axis, angle)
        f1 = _transport(f0, arc1)
        forward = [_Rev(f0, arc1)]
        arc2_start = arc1.te
        f_arc2 = f1
        if bridge > 1e-9:
            forward.append(_Prism(f1, bridge))
            arc2_start = arc1.te + f1.t * bridge
            f_arc2 = f1.at(arc2_start)
        axis2 = bend_axis * -1.0
        # 圆心由“起点半径 = tangent × axis × R”反算；该式同时
        # 适用于小于 90° 的最小错位圆弧和带中段的 90° 圆弧。
        radius2 = f_arc2.t.cross(axis2) * rb0
        arc2 = circular_arc(arc2_start, arc2_start - radius2,
                            axis2, angle)
        f2 = _transport(f_arc2, arc2)
        straight = _Prism(f2, lead_len)
        tip = arc2.te + zhat * lead_len
        forward.extend([_Rev(f_arc2, arc2), straight])

        reverse = []
        for segment in reversed(forward):
            if isinstance(segment, _Prism):
                end = segment.f.o + segment.f.t * segment.length
                reverse.append(_Prism(
                    reverse_frame(segment.f, end), segment.length,
                    bare0=segment.bare1, bare1=segment.bare0))
            else:
                end_frame = _transport(segment.f, segment.fl)
                reverse.append(_Rev(reverse_frame(end_frame, segment.fl.te),
                                    reverse_arc(segment.fl)))
        # 对地绝缘开孔需再向线圈内部延伸一小段；这里给出 -Z 框架。
        inside = reverse_frame(f0, join)
        return forward, reverse, tip, inside

    segs: list = []
    turn_ranges: list[tuple[int, int, int]] = []  # (start, end, material_i)
    chunk_start = 0

    # 反向追踪材料路径：下层槽底匝N → ... → 上层槽底匝1。
    order = list(range(n - 1, -1, -1))

    # ---- 第一根引线：+Z 端 → 下层槽底匝N直边 ----
    d0 = dy_slot(order[0])
    join_in = fl[4].te + lf.f_leg2.y * d0
    bare = max(0.0, min(inp.lead_bare, lead_len - 1.0))
    _forward_in, lead_in_path, tip_in, inside_in = lead_dogleg(
        join_in, lf.f_leg2, lf.f_leg2.y)
    lead_in_path[0].bare0 = bare
    segs.extend(lead_in_path)

    tip_out = None
    info_out = None
    for pos, i in enumerate(order):
        d_slot = dy_slot(i)
        d_nose = dy_nose(i)
        # 下层边 4→5 + 角5
        segs.append(_Prism(lf.f_leg2.at(fl[4].te),
                           (fl[5].ts - fl[4].te).length, dy=d_slot))
        segs.append(_Rev(lf.f_leg2.at(fl[5].ts), fl[5], dy=d_slot))
        # -Z 非接线端：同一材料匝位完整连接下层与上层
        segs.extend(slant_lofts((5, 6), f56s, f56e, d_slot, d_nose))
        segs.append(_Rev(f56e, fl[6], dy=d_nose))
        add_constant_nose(lf.f_q6, lf.f_q7,
                          lf.f_flat_neg, lf.f_flat_neg_out,
                          lf.nose_neg, 6, 7, d_nose)
        segs.extend(slant_lofts((7, 0), f70s, f70e, d_nose, d_slot))
        segs.append(_Rev(f70e, fl[0], dy=d_slot))
        # 上层边 0→1 + 角1
        segs.append(_Prism(lf.f_leg1.at(fl[0].te),
                           (fl[1].ts - fl[0].te).length, dy=d_slot))
        if pos < n - 1:
            segs.append(_Rev(lf.f_leg1.at(fl[1].ts), fl[1], dy=d_slot))
            # +Z 接线端的完整 U 形连接：上层 i → 下层 i-1。
            # 匝位变化按“两条直鼻臂 + 圆冠”的完整 U 弧长连续分布。
            # 这样没有独立接头/台阶，也不会让
            # 自由引线所在斜边与相邻 U 形接续只剩 0.75 匝距而穿体。
            next_i = order[pos + 1]
            dn_slot = dy_slot(next_i)
            dn_nose = dy_nose(next_i)
            segs.extend(slant_lofts((1, 2), f12s, f12e, d_slot, d_nose))
            segs.append(_Rev(f12e, fl[2], dy=d_nose))
            # 匝绝缘在接线 nose 换位前分件。下一件从换位 nose 开始，
            # 随后进入下一材料匝；这样同一零件内部不会同时包含接线侧
            # 相邻两匝的整段端臂，避免零间隙贴合在布尔融合后成为
            # 自相交/非流形实体。铜导线仍按完整连续路径一次融合。
            turn_ranges.append((chunk_start, len(segs), i))
            chunk_start = len(segs)
            add_transition_nose(d_nose, dn_nose)
            segs.extend(slant_lofts((3, 4), f34s, f34e, dn_nose, dn_slot))
            segs.append(_Rev(f34e, fl[4], dy=dn_slot))
        else:
            # ---- 第二根引线：上层槽底匝1直边 → +Z 端 ----
            join_out = fl[1].ts + lf.f_leg1.y * d_slot
            lead_out_path, _reverse_out, tip_out, inside_out = lead_dogleg(
                join_out, lf.f_leg1, lf.f_leg1.y * -1.0)
            lead_out_path[-1].bare1 = bare
            segs.extend(lead_out_path)
            info_out = (lead_out_path, join_out, inside_out)
            turn_ranges.append((chunk_start, len(segs), i))
            chunk_start = len(segs)

    info = dict(tip_in=tip_in, tip_out=tip_out, bare_len=bare,
                lead_in=(lead_in_path, join_in, inside_in), lead_out=info_out,
                turn_ranges=turn_ranges)
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
    """局部布尔差集（先精确后模糊容差，用于放样环与开孔）。

    模糊容差在近相切的旋转面/扫掠面组合上可能吞掉错误的一侧、
    留下与刀具体积相交的残料；因此先做精确布尔并校验，失败时才
    回退到带容差的布尔。
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopTools import TopTools_ListOfShape

    def run(fuzzy: float):
        args = TopTools_ListOfShape()
        args.Append(a.wrapped)
        tools = TopTools_ListOfShape()
        tools.Append(b.wrapped)
        op = BRepAlgoAPI_Cut()
        op.SetArguments(args)
        op.SetTools(tools)
        if fuzzy > 0.0:
            op.SetFuzzyValue(fuzzy)
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            return None
        shape = op.Shape()
        if shape.IsNull():
            return None
        if not BRepCheck_Analyzer(shape).IsValid():
            return None
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            comp = b3d.Compound(shape)
            sols = comp.solids()
            return sols[0] if len(sols) == 1 else comp
        return b3d.Solid(shape)

    try:
        exact = run(0.0)
    except Exception:
        exact = None
    if exact is not None:
        return exact
    try:
        fuzzy_cut = run(fuzz)
    except Exception:
        fuzzy_cut = None
    if fuzzy_cut is not None:
        return fuzzy_cut
    return a - b


def _as_basis(f) -> _Basis:
    return f if isinstance(f, _Basis) else _Basis(f.o, f.x, f.y, f.t)


def _loft_faces(fa, fb, w, h, xo, yo_a, yo_b):
    """放样两端面。任一端为 _Basis 时两端都走显式四边形（角点次序
    一致，保证直纹与整条放样精确共面）；否则维持原矩形路径。"""
    if isinstance(fa, _Basis) or isinstance(fb, _Basis):
        return (b3d.Face(_quad_wire(_as_basis(fa), w, h, xo, yo_a)),
                b3d.Face(_quad_wire(_as_basis(fb), w, h, xo, yo_b)))
    return (_face_at(fa, w, h, xo, yo_a), _face_at(fb, w, h, xo, yo_b))


def _multiloft_section_groups(
        seg: _MultiLoft) -> list[tuple[list[_Basis], bool]]:
    """按直鼻臂/圆冠的 G1 曲率边界拆分放样截面组及其直纹标志。

    若把整只 U 一次拟合为单张高阶 B-spline，OCCT 会在 P 点产生可见
    过冲，且端盖与相邻解析圆角只有数值重合、无法 glue。各段共享完全
    相同的 P 截面分别放样后，形态忠实于“直线 + 圆弧 + 直线”，同时
    仍作为同一个连续换匝分段对外使用。直鼻臂组按直纹放样，与外围
    ``_twisted_lofts`` 扭转墙逐点一致。
    """
    last = len(seg.sections) - 1
    cuts = [0, *(i for i in seg.breaks if 0 < i < last), last]
    groups: list[tuple[list[_Basis], bool]] = []
    kept = 0
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        ruled = seg.ruled[kept] if kept < len(seg.ruled) else False
        groups.append((seg.sections[a:b + 1], ruled))
        kept += 1
    return groups


def _seg_solid(seg, w: float, h: float, xo: float = 0.0, yo: float = 0.0,
               trim_bare: bool = False):
    """实心截面沿分段的实体（截面再叠加分段自身的匝级偏移 dy）。

    ``trim_bare`` 仅供连续绝缘包络使用：按引线直段的 ``bare0/1``
    同时截短内、外包络，随后一次整体相减得到干净的材料实体。
    """
    if isinstance(seg, _Prism):
        f = seg.f
        length = seg.length
        if trim_bare:
            length -= seg.bare0 + seg.bare1
            if seg.bare0 > 0:
                f = f.at(f.o + f.t * seg.bare0)
        if length <= 1e-6:
            return None
        face = _face_at(f, w, h, xo, yo + seg.dy)
        if isinstance(f, _Basis):
            return b3d.extrude(face, amount=length, dir=tuple(f.t))
        return b3d.extrude(face, amount=length)
    if isinstance(seg, _Rev):
        axis = b3d.Axis(origin=tuple(seg.fl.c), direction=tuple(seg.fl.n))
        return b3d.revolve(_face_at(seg.f, w, h, xo, yo + seg.dy), axis,
                           revolution_arc=math.degrees(seg.fl.tau))
    if isinstance(seg, _Loft):
        fa, fb = _loft_faces(seg.f0, seg.f1, w, h, xo,
                             yo + seg.dy0, yo + seg.dy1)
        return b3d.loft([fa, fb], ruled=True)
    if isinstance(seg, _MultiLoft):
        pieces = []
        for sections, ruled in _multiloft_section_groups(seg):
            faces = [_face_at(section, w, h, xo, yo)
                     for section in sections]
            pieces.extend(_loft_group(faces, ruled))
        return _join(pieces)
    raise TypeError(seg)


def _loft_group(faces: list, ruled: bool) -> list:
    """按组放样：直纹组逐对放样（与扭转墙一致、可精确拆分互补），
    光顺组一次 B 样条放样。"""
    if not ruled:
        return [b3d.loft(faces, ruled=False)]
    return [b3d.loft([fa, fb], ruled=True)
            for fa, fb in zip(faces, faces[1:])]


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
        return _cut(outer, inner, fuzz=1e-7)
    if isinstance(seg, _MultiLoft):
        # 直接放样“带孔截面”，一次生成有内腔的材料实体；若先分别
        # 放样内外实体再布尔相减，OCCT 会在半圆换匝段产生毫米级边
        # 公差，SolidWorks 随后会把云母带降级为曲面实体。
        pieces = []
        for sections, ruled in _multiloft_section_groups(seg):
            rings = [_ring_at(section, w1, h1, w2, h2, xo, yo)
                     for section in sections]
            pieces.extend(_loft_group(rings, ruled))
        return _join(pieces)
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


def _continuous_path_shell(segs: list, w1: float, h1: float,
                           w2: float, h2: float,
                           xo: float = 0.0, yo: float = 0.0):
    """沿同一解析路径生成一件连续的绝缘材料实体。

    OCCT 对八匝强扭转路径的“完整外实体−完整内实体”全局布尔不稳定；
    因此保留共享端面的解析材料环并一次 glue。关键的半圆换匝段直接
    放样带孔截面，不再做内外布尔差，从源头消除旧版毫米级公差边。
    """
    return _join([_seg_ring(g, w1, h1, w2, h2, xo, yo)
                  for g in segs])


def _multiloft_ring_strips(seg: _MultiLoft,
                           w1: float, h1: float,
                           w2: float, h2: float,
                           xo: float = 0.0,
                           yo: float = 0.0) -> list[tuple[str, object]]:
    """把换匝段的矩形环截面拆成四条无内孔实心放样。

    SolidWorks 2023 会把带内孔的光顺 ``_MultiLoft`` 环降级为曲面体，
    即使 OCCT 将它判为有效 ``SOLID``。这里在每个材料截面上把矩形环
    精确分割成左、右、上、下四个互不重叠的矩形，再分别沿完全相同的
    光顺截面序列放样。四条带只共享边界，合计体积与原环一致；不 fuse
    可确保每个 STEP 叶节点自身都是无内孔的顶层实体。
    """
    side_w = (w2 - w1) / 2.0
    cap_h = (h2 - h1) / 2.0
    if side_w <= 1e-9 or cap_h <= 1e-9:
        raise ValueError(
            "换匝段匝绝缘无法拆成四条实心条带："
            f"内截面={w1:.6g}×{h1:.6g}mm，"
            f"外截面={w2:.6g}×{h2:.6g}mm")

    specs = (
        ("卷环段-左侧条带", side_w, h2,
         xo - (w1 + w2) / 4.0, yo),
        ("卷环段-右侧条带", side_w, h2,
         xo + (w1 + w2) / 4.0, yo),
        ("卷环段-上侧条带", w1, cap_h,
         xo, yo + (h1 + h2) / 4.0),
        ("卷环段-下侧条带", w1, cap_h,
         xo, yo - (h1 + h2) / 4.0),
    )
    result = []
    for label, width, height, x_offset, y_offset in specs:
        pieces = []
        for sections, ruled in _multiloft_section_groups(seg):
            faces = [_face_at(section, width, height, x_offset, y_offset)
                     for section in sections]
            pieces.extend(_loft_group(faces, ruled))
        loft = _join(pieces)
        solids = loft.solids()
        if len(solids) != 1:
            raise RuntimeError(f"{label} 未生成单一实体")
        # b3d.loft 通常已返回 Solid；显式取唯一子实体可防止不同
        # OCCT/build123d 版本在单一实体外再包一层 Compound。极薄的
        # 上/下条带经过强曲率换匝时，OCCT 偶尔会返回拓扑有效但壳朝向
        # 反转的 Solid（有符号体积为负）；只反转拓扑朝向，不改几何。
        solid = solids[0]
        if solid.volume < 0:
            solid = b3d.Solid(solid.wrapped.Reversed())
        result.append((label, solid))
    return result


def _continuous_ring_shell_parts(segs: list,
                                 w1: float, h1: float,
                                 w2: float, h2: float,
                                 xo: float = 0.0,
                                 yo: float = 0.0) -> list[tuple[str, object]]:
    """把一整条连续路径的材料环输出为 SolidWorks 兼容的实体叶节点。

    不含 ``_MultiLoft`` 时仍输出原来的一件完整实体。每遇到一个光顺
    换匝放样，就在该段把带内孔的环拆为四条无孔实心条带；相邻换匝
    段之间的普通解析路径各自融合为一件主体。这样任意匝数均只有
    顶层单一 ``SOLID`` 叶节点，同时各叶节点只共享边界、不重叠体积。
    """
    transition_count = sum(isinstance(segment, _MultiLoft)
                           for segment in segs)
    if transition_count == 0:
        shell = _continuous_path_shell(segs, w1, h1, w2, h2, xo, yo)
        if shell is None or len(shell.solids()) != 1:
            raise RuntimeError("连续绝缘材料环未生成单一实体")
        return [("", shell.solids()[0])]

    result: list[tuple[str, object]] = []
    ordinary: list = []
    body_no = 0
    transition_no = 0

    def flush_ordinary() -> None:
        nonlocal body_no
        if not ordinary:
            return
        shell = _continuous_path_shell(
            ordinary, w1, h1, w2, h2, xo, yo)
        if shell is None or len(shell.solids()) != 1:
            raise RuntimeError(
                f"连续绝缘主体段{body_no + 1}未生成单一实体")
        body_no += 1
        result.append((f"主体段{body_no}", shell.solids()[0]))
        ordinary.clear()

    for segment in segs:
        if not isinstance(segment, _MultiLoft):
            ordinary.append(segment)
            continue
        flush_ordinary()
        transition_no += 1
        for label, strip in _multiloft_ring_strips(
                segment, w1, h1, w2, h2, xo, yo):
            side = label.removeprefix("卷环段-")
            result.append((f"卷环{transition_no}-{side}", strip))
    flush_ordinary()

    if transition_no != transition_count:
        raise RuntimeError("连续绝缘换匝段计数异常")
    return result


def _turn_mica_shell_parts(segs: list,
                           w1: float, h1: float,
                           w2: float, h2: float) -> list[tuple[str, object]]:
    """构造一个材料匝的云母实体叶节点。

    每个卷环/换匝 ``_MultiLoft``（+Z 换匝螺旋、±Z 整环）拆成
    左/右/上/下四条无孔实心条带，其间的普通解析路径各自融合为
    一件主体——与 ``_continuous_ring_shell_parts`` 同一策略。
    """
    return _continuous_ring_shell_parts(segs, w1, h1, w2, h2)


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
                   f_sl1: _Frame | None, remain: float,
                   law=None) -> list:
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
            # 防晕层必须沿与导线相同的扭转分布律采样材料框架，
            # 否则套管会缩扁并切入铜线。按整条斜边的分布律在
            # [0, lam2] 上直接采样，覆盖到扭转段时逐点一致。
            count = max(2, math.ceil(lam2 * 12))
            frames = [_twist_frame_at(f_sl0, f_sl1, lam2 * j / count, law)
                      for j in range(count + 1)]
            for fa, fb in zip(frames, frames[1:]):
                ring = _seg_ring(_Loft(fa, fb), inner_w, inner_h, w2, h2)
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
            law_fw, _slope_fw = lf.slant_laws[key_fw]
            pieces += ext_pieces(f_leg.at(fl[i_exi].ts), fl[i_exi],
                                 fs, fe, remain, law_fw)
            # 入口端（逆行：leg → 角 i_ent 反向 → 斜边 key_bw 反向）
            fil = fl[i_ent]
            fil_rev = _Fillet(fil.te, fil.ma, fil.ts, fil.c,
                              fil.n * -1.0, fil.tau)
            bs, be = lf.slant_frames[key_bw]
            law_bw, _slope_bw = lf.slant_laws[key_bw]
            pieces += ext_pieces(rev_frame(f_leg, fil.te), fil_rev,
                                 rev_frame(be, be.o), rev_frame(bs, bs.o),
                                 remain, _mirrored_law(law_bw))
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
    """沿两根 S 形错位引线构造贴身切割体。

    截面 = 铜导线∪匝绝缘包络（cut_w×cut_h）+ 单边 _HOLE_CLEARANCE，
    几何上等价于“布尔减去导线包络与方壳的相交体积”，但只用引线
    局部实体做刀具，避免整圈共面布尔卡死。落点再沿槽内直边伸出
    一小段 stub，保证切穿对地壁厚。
    """
    cl = _HOLE_CLEARANCE
    w, h = cut_w + 2 * cl, cut_h + 2 * cl
    stub = 6.0  # mm，沿斜边伸入方壳，略大于典型对地总厚
    cutters = []
    for key in ("lead_in", "lead_out"):
        path, join, inside = info[key]
        pieces = [_seg_solid(seg, w, h) for seg in path]
        pieces.append(_seg_solid(_Prism(inside.at(join), stub), w, h))
        body = _join(pieces)
        if body is not None:
            cutters.append(body)
    return cutters


def _bundle_lead_cutters(bundle, info: dict, pad: float = 12.0) -> list:
    """用铜∪匝绝缘实体在引线区的裁剪块作刀具（首选，最贴身）。

    失败（布尔不收敛/空结果）时由调用方回退到 `_lead_path_cutters`。
    """
    cutters = []
    for key in ("lead_in", "lead_out"):
        path, _join_point, _inside = info[key]
        try:
            centerline_box = _join([_seg_solid(seg, 1.0, 1.0)
                                    for seg in path])
            bb0 = centerline_box.bounding_box()
            cx = (bb0.min.X + bb0.max.X) / 2
            cy = (bb0.min.Y + bb0.max.Y) / 2
            cz = (bb0.min.Z + bb0.max.Z) / 2
            bw = bb0.size.X + 2 * pad
            bh = bb0.size.Y + 2 * pad
            bd = bb0.size.Z + 2 * pad
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
            shell_parts = _continuous_ring_shell_parts(
                segs, s["b"], s["h"], s["bi"], s["hi"],
                s["x"], s["y"])
            for suffix, shell in shell_parts:
                part_name = f"{ins_name}-{suffix}" if suffix else ins_name
                parts.append(CoilPart(part_name, shell, STRAND_INS_COLOR))

    # ---- 匝绝缘分层（包每匝导线束）----
    # 每个材料匝单独输出。相邻匝的外高恰好等于匝距，若把全部匝先
    # glue 成一个零件，它们会整面贴合并形成分叉拓扑。含换匝
    # _MultiLoft 的材料匝再拆成“主体 + 四条无孔条带”：这五个叶节点
    # 保持原光顺几何和零间隙贴合，但各自都是 SolidWorks 可直接识别
    # 的单一闭合实体。
    h_cap = res.had - 2 * _TURN_CLEARANCE if n > 1 else float("inf")
    grow = 0.0
    prev_w, prev_h = w_env, h_env
    for i, layer in enumerate(inp.turn_layers):
        if layer.thickness <= 0:
            continue
        grow += layer.thickness
        w2 = w_env + 2 * grow
        h2 = min(h_env + 2 * grow, h_cap)
        color = TURN_LAYER_COLORS[i % len(TURN_LAYER_COLORS)]
        for start, end, material_i in info["turn_ranges"]:
            base_name = f"匝绝缘{i + 1}-{layer.name}-第{material_i + 1}匝"
            shell_parts = _turn_mica_shell_parts(
                segs[start:end], prev_w, prev_h, w2, h2)
            for suffix, shell in shell_parts:
                name = f"{base_name}-{suffix}" if suffix else base_name
                parts.append(CoilPart(name, shell, color))
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


def _validate_3d_insulation_sections(res: CoilResult) -> None:
    """拒绝尚不能由恒截面实体正确表达的端部绝缘组合。

    中心线已经在槽部按 HAD、鼻端按 HBD 排列；但当前匝绝缘与对地
    绝缘仍沿整圈使用一套矩形截面。若 T1/T3 或 T2/T4 不同，继续
    建模会让鼻端云母互穿、对地壳包络不足，或静默生成错误厚度。
    参数计算、二维图和中心线路径不受此限制；这里只阻止错误的三维
    实体输出，待可变截面绝缘实现后再解除。
    """
    inp = res.inp
    mismatches = []
    if not math.isclose(inp.t1, inp.t3, rel_tol=0.0, abs_tol=1e-6):
        mismatches.append(f"T1={inp.t1:g} 与 T3={inp.t3:g}")
    if not math.isclose(inp.t2, inp.t4, rel_tol=0.0, abs_tol=1e-6):
        mismatches.append(f"T2={inp.t2:g} 与 T4={inp.t4:g}")
    if mismatches:
        raise ValueError(
            "当前三维实体要求槽内/端部绝缘厚度一致（T1=T3 且 "
            "T2=T4）；" + "，".join(mismatches) + "。"
            "参数计算和二维图仍可使用，但三维可变截面绝缘尚未实现，"
            "为避免鼻端绝缘互穿或包络不足，已停止生成三维模型。")

def build_coil_parts(res: CoilResult, detailed: bool | None = None) -> list[CoilPart]:
    """构造线圈部件列表。detailed=None 时按输入 detail_3d 选择。"""
    _validate_3d_insulation_sections(res)
    if detailed is None:
        detailed = res.inp.detail_3d
    parts = _build_detailed_parts(res) if detailed else _build_simple_parts(res)
    if not res.inp.lead_end_positive_z:
        # 出线端换侧是整只成型线圈关于轴向中面的镜像，而不是只平移
        # 两根引线。这样 nose 手性、裸铜端、绝缘出口及所有固定件保持
        # 同一材料关系；F 的径向槽底方向不受轴向镜像影响。
        parts = [CoilPart(p.name, p.solid.mirror(b3d.Plane.XY), p.color)
                 for p in parts]
    return parts


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
    # glue/loft 在完全共面的云母带端面上偶尔留下毫米级“记账公差”，
    # 实际端面偏差仅约 1e-13 mm。SolidWorks 会据此放弃缝合并降级为
    # 曲面实体。仅在复制体收紧到 1e-4 mm 后仍通过 B-Rep 检查时采用。
    try:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
        from OCP.ShapeFix import ShapeFix_ShapeTolerance

        copier = BRepBuilderAPI_Copy(shape)
        tight = copier.Shape()
        limiter = ShapeFix_ShapeTolerance()
        limiter.LimitTolerance(tight, 0.0, 1e-4)
        if BRepCheck_Analyzer(tight, True).IsValid():
            shape = tight
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
    from .step_export import export_step_guarded

    parts = [_finish_part(p) for p in build_coil_parts(res, detailed)]
    export_step_guarded(parts, filepath, asm_name="成型线圈",
                        writer=_export_step_xcaf, name_fixer=fix_step_names)
    return [p.name for p in parts]
