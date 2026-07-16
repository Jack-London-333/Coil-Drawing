"""决策示意图:现状 vs 方案A(浅螺旋) vs 方案C(平盘+入口扇区下潜)。

只画束中心线族(红=束心环,蓝/绿=内/外极限匝,橙=入口臂,紫=出口臂),
视角以 +Z 鼻端环心为中心。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

b3d = m.b3d
BETA = m._NOSE_CROSS_BETA
SWEEP = math.pi + 2 * BETA


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(80.0)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)
need = m._nose_cross_clearance_need(res)
D = need
delta = math.radians(115.0)
r_out = (inp.n_turns - 1) / 2 * res.hbd


class Scheme:
    def __init__(self, res, profile):
        self.res, self.inp = res, res.inp
        self.profile = profile
        self.rn = (res.rr1 + res.rr2) / 2 + res.hc / 2 + res.inp.f_nose
        self.zn = res.l2 / 2 + res.cc
        self.Rc = res.inp.rd_nose + res.wa_turn / 2
        s3, c3 = math.sin(res.inp.seita3), math.cos(res.inp.seita3)
        self.axis = b3d.Vector(0, c3, s3)
        self.d_in = b3d.Vector(1, 0, 0)
        self.n = self.d_in.cross(self.axis)
        self.slot1 = m._cyl(res.rr1 + res.hc / 2, -res.fai1, +res.l2 / 2)
        self.slot4 = m._cyl(res.rr2 + res.hc / 2, +res.fai2, +res.l2 / 2)
        self.r0 = self.d_in * math.cos(BETA) - self.axis * math.sin(BETA)

    def ring_point(self, shift, th, dy=0.0):
        c = b3d.Vector(0, self.rn, self.zn) - self.axis * shift
        return (c + m._rotv(self.r0, self.n, th) * (self.Rc + dy)
                + self.n * self.profile(th))

    def ring_tangent(self, shift, th):
        e = 1e-5
        return (self.ring_point(shift, th + e)
                - self.ring_point(shift, th - e)).normalized()

    def _compact(self, slot, endpoint, toward, r2):
        def st(ext):
            v = endpoint - toward * ext
            u = (v - slot).normalized()
            tau = math.acos(max(-1, min(1, u.dot(toward))))
            return ext - r2 * math.tan(tau / 2), v
        lo, hi = 0.0, r2 * math.sqrt(3.0)
        if st(hi)[0] < -1e-10:
            raise ValueError("rd2 hi<0")
        for _ in range(64):
            mid = (lo + hi) / 2
            if st(mid)[0] >= 0:
                hi = mid
            else:
                lo = mid
        return st(hi)[1]

    def paths(self, shift, count=96):
        Q2 = self.ring_point(shift, 0.0)
        Q3 = self.ring_point(shift, SWEEP)
        t2 = self.ring_tangent(shift, 0.0)
        t3 = self.ring_tangent(shift, SWEEP)
        p2 = self._compact(self.slot1, Q2, t2, self.inp.r_bend_nose)
        p3 = self._compact(self.slot4, Q3, t3 * -1.0, self.inp.r_bend_nose)
        f2 = m._fillet_corner(self.slot1, p2, Q2, self.inp.r_bend_nose, 1.0)
        f3 = m._fillet_corner(Q3, p3, self.slot4, self.inp.r_bend_nose, 1.0)

        def arcp(fl):
            return [fl.c + m._rotv(fl.ts - fl.c, fl.n, fl.tau * k / count)
                    for k in range(count + 1)]
        entry = [self.slot1 + (f2.ts - self.slot1) * (k / count)
                 for k in range(count + 1)] + arcp(f2)
        exit_ = arcp(f3) + [f3.te + (self.slot4 - f3.te) * (k / count)
                            for k in range(count + 1)]
        return entry, exit_

    def total_len(self, shift, count=720):
        entry, exit_ = self.paths(shift)
        ring = [self.ring_point(shift, SWEEP * k / count)
                for k in range(count + 1)]

        def pl(p):
            return sum((b - a).length for a, b in zip(p, p[1:]))
        return pl(entry) + pl(ring) + pl(exit_)

    def solve_shift(self, target):
        prev, span = None, None
        for k in range(97):
            s = -80 + 480 * k / 96
            try:
                d = self.total_len(s) - target
            except ValueError:
                if prev is not None:
                    break
                continue
            if prev is not None and prev[1] * d < 0:
                span = (prev[0], s)
                break
            prev = (s, d)
        lo, hi = span
        flo = self.total_len(lo) - target
        for _ in range(50):
            mid = (lo + hi) / 2
            f = self.total_len(mid) - target
            if flo * f < 0:
                hi = mid
            else:
                lo, flo = mid, f
        return (lo + hi) / 2


def target_len(res):
    pose = m._nose_pose(res)
    _c, fl = m._loop_fillets(res, 0.0, pose)
    nose = m._nose_layout(res, 0.0, pose)

    def al(f):
        return (f.ts - f.c).length * f.tau
    return ((fl[2].ts - _c[1][0]).length + al(fl[2]) + al(nose.pos)
            + al(fl[3]) + (_c[4][0] - fl[3].te).length)


target = target_len(res)

# 现状路径
pose = m._nose_pose(res)
_c, fl_c = m._loop_fillets(res, 0.0, pose)
nose_c = m._nose_layout(res, 0.0, pose)


def cur_paths(count=96):
    def arcp(fl, cnt=count):
        return [fl.c + m._rotv(fl.ts - fl.c, fl.n, fl.tau * k / cnt)
                for k in range(cnt + 1)]
    entry = ([_c[1][0] + (fl_c[2].ts - _c[1][0]) * (k / count)
              for k in range(count + 1)] + arcp(fl_c[2]))
    exit_ = (arcp(fl_c[3]) + [fl_c[3].te + (_c[4][0] - fl_c[3].te) * (k / count)
                              for k in range(count + 1)])
    ring = arcp(nose_c.pos, 384)
    return entry, ring, exit_


rows = []
e0, r0_, x0 = cur_paths()
rows.append(("现状(被否定): 盘面绕鼻端中心线偏航30°, 硬币斜骑, seita3失效",
             None, None, e0, r0_, x0))

schA = Scheme(res, lambda th: D / 2 - D * th / SWEEP)
sA = schA.solve_shift(target)
eA, xA = schA.paths(sA)
rows.append((f"方案A 浅螺旋: 盘面=带面(0偏航), 交叉错距D={D:.1f}mm 均匀分布在整个环上(锁紧垫圈式)",
             schA, sA, eA, None, xA))

schC = Scheme(res, lambda th: D * (1 - smoothstep(th / delta))
              if th < delta else 0.0)
sC = schC.solve_shift(target)
eC, xC = schC.paths(sC)
rows.append((f"方案C 平盘+扇区下潜: 环后2/3严格平面共带面, 入口臂抬D={D:.1f}mm 压过交叉后在首115°顺势潜回盘面",
             schC, sC, eC, None, xC))

fig = plt.figure(figsize=(16, 14))
views = [(0, -90, "径向看(从电机外侧)"),
         (90, -90, "沿盘面法向看"),
         (0, 0, "切向看(侧面)——关键对比")]
for irow, (title, sch, s, ent, rng, ext) in enumerate(rows):
    for icol, (elev, azim, vt) in enumerate(views):
        ax = fig.add_subplot(3, 3, irow * 3 + icol + 1, projection="3d")
        segs, colors, widths = [], [], []
        if sch is not None:
            for dy, col in [(-r_out, "#1f77b4"), (0.0, "#d62728"),
                            (r_out, "#2ca02c")]:
                ring = [tuple(sch.ring_point(s, SWEEP * k / 256, dy))
                        for k in range(257)]
                segs += list(zip(ring, ring[1:]))
                colors += [col] * 256
            center = sch.ring_point(s, 0.0) - (
                sch.ring_point(s, 0.0) - sch.ring_point(s, SWEEP)) * 0  # dummy
            cx0 = b3d.Vector(0, sch.rn, sch.zn) - sch.axis * s
        else:
            ring = [tuple(p) for p in rng]
            segs += list(zip(ring, ring[1:]))
            colors += ["#d62728"] * (len(ring) - 1)
            cx0 = nose_c.pos.c
        ent_t = [tuple(p) for p in ent]
        ext_t = [tuple(p) for p in ext]
        segs += list(zip(ent_t, ent_t[1:])) + list(zip(ext_t, ext_t[1:]))
        colors += (["#ff7f0e"] * (len(ent_t) - 1)
                   + ["#9467bd"] * (len(ext_t) - 1))
        lc = Line3DCollection(segs, colors=colors, linewidths=1.6)
        ax.add_collection3d(lc)
        r = 78
        ax.set_xlim(cx0.X - r, cx0.X + r)
        ax.set_ylim(cx0.Y - r, cx0.Y + r)
        ax.set_zlim(cx0.Z - r, cx0.Z + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        if icol == 1:
            ax.set_title(f"{title}\n{vt}", fontsize=9.5)
        else:
            ax.set_title(vt, fontsize=9.5)
fig.suptitle("鼻端方案对比·束中心线族(N=8, seita3=80°)  "
             "红=束心环  蓝/绿=内/外极限匝环  橙=入口臂(上层斜边→环)  紫=出口臂(环→下层斜边)",
             fontsize=11.5)
out = Path(__file__).resolve().parents[1] / "tmp" / "nose_decision.png"
fig.savefig(out, dpi=105, bbox_inches="tight")
print("saved", out)
