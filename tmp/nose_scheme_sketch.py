"""方案对比示意图(只画束中心线族,不动 src):

  行1 现状     : 环平面偏航 yaw=30°(硬币斜骑)——用户否定;
  行2 方案A 浅螺旋: 盘面=带面;环从 Q2(+D/2) 匀速降到 Q3(-D/2),
                  全环均布 ~9° 螺旋升角(像锁紧垫圈);
  行3 方案C 扇区下潜: 盘 2/3 严格共面;入口臂抬高 D 压过交叉,
                  环首扇区 delta=115° 内 smoothstep 下潜回平面。

同时数值验证 A/C 在 seita3=70/80/90° 的:
  rd2 紧凑相切可行性、交叉净距、入/出臂与外匝环纤维净距。
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


def make_res(deg):
    inp = CoilInput()
    inp.n_turns = 8
    inp.f_nose = 20.0
    inp.seita3 = math.radians(deg)
    inp.rd_nose = 15.0
    inp.r_bend_slot = 30.0
    inp.r_bend_nose = 30.0
    return compute(inp)


class Scheme:
    """yaw=0 基准系 + 高度剖面 h(theta), 入口臂端点抬高 h(0)。"""

    def __init__(self, res, profile):
        self.res = res
        inp = res.inp
        self.inp = inp
        self.profile = profile      # h(theta) -> 沿环法向抬高
        self.rn = (res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose
        self.zn = res.l2 / 2 + res.cc
        self.Rc = inp.rd_nose + res.wa_turn / 2
        sin3, cos3 = math.sin(inp.seita3), math.cos(inp.seita3)
        self.axis = b3d.Vector(0, cos3, sin3)
        self.d_in = b3d.Vector(1, 0, 0)
        self.n = self.d_in.cross(self.axis)
        th1, th2 = -res.fai1, +res.fai2
        self.slot1 = m._cyl(res.rr1 + res.hc / 2, th1, +res.l2 / 2)
        self.slot4 = m._cyl(res.rr2 + res.hc / 2, th2, +res.l2 / 2)
        self.r0 = self.d_in * math.cos(BETA) - self.axis * math.sin(BETA)

    def ring_point(self, shift, theta, dy=0.0):
        center = b3d.Vector(0, self.rn, self.zn) - self.axis * shift
        rdir = m._rotv(self.r0, self.n, theta)
        return center + rdir * (self.Rc + dy) + self.n * self.profile(theta)

    def ring_tangent(self, shift, theta):
        eps = 1e-5
        a = self.ring_point(shift, theta - eps)
        b = self.ring_point(shift, theta + eps)
        return (b - a).normalized()

    def _compact(self, slot, endpoint, toward, radius2):
        def state(ext):
            virtual = endpoint - toward * ext
            u = (virtual - slot).normalized()
            tau = math.acos(max(-1, min(1, u.dot(toward))))
            return ext - radius2 * math.tan(tau / 2), tau, virtual
        lo, hi = 0.0, radius2 * math.sqrt(3.0)
        if state(hi)[0] < -1e-10:
            raise ValueError("rd2 hi<0")
        for _ in range(64):
            mid = (lo + hi) / 2
            if state(mid)[0] >= 0:
                hi = mid
            else:
                lo = mid
        _v, tau, virtual = state(hi)
        if tau > math.radians(120) + 1e-9:
            raise ValueError(f"rd2 tau={math.degrees(tau):.1f}°")
        ratio = ((virtual - slot).length + hi) / (endpoint - slot).length
        if ratio > 1.5:
            raise ValueError(f"rd2 ratio={ratio:.2f}")
        chord = endpoint - slot
        if (virtual - slot).dot(chord) <= 0 or (endpoint - virtual).dot(chord) <= 0:
            raise ValueError("rd2 回头")
        return virtual

    def paths(self, shift, count=96):
        """返回 entry(slot1→Q2), ring(束心), exit(Q3→slot4) 采样。"""
        Q2 = self.ring_point(shift, 0.0)
        Q3 = self.ring_point(shift, SWEEP)
        t2 = self.ring_tangent(shift, 0.0)
        t3 = self.ring_tangent(shift, SWEEP)
        p2 = self._compact(self.slot1, Q2, t2, self.inp.r_bend_nose)
        p3 = self._compact(self.slot4, Q3, t3 * -1.0, self.inp.r_bend_nose)
        fl2 = m._fillet_corner(self.slot1, p2, Q2, self.inp.r_bend_nose, 1.0)
        fl3 = m._fillet_corner(Q3, p3, self.slot4, self.inp.r_bend_nose, 1.0)

        def arc_pts(fl):
            return [fl.c + m._rotv(fl.ts - fl.c, fl.n, fl.tau * k / count)
                    for k in range(count + 1)]

        entry = [self.slot1 + (fl2.ts - self.slot1) * (k / count)
                 for k in range(count + 1)] + arc_pts(fl2)
        exit_ = arc_pts(fl3) + [fl3.te + (self.slot4 - fl3.te) * (k / count)
                                for k in range(count + 1)]
        ring = [self.ring_point(shift, SWEEP * k / (count * 4))
                for k in range(count * 4 + 1)]
        return entry, ring, exit_, fl2, fl3

    def total_len(self, shift):
        entry, ring, exit_, fl2, fl3 = self.paths(shift)

        def pl(pts):
            return sum((b - a).length for a, b in zip(pts, pts[1:]))
        return pl(entry) + pl(ring) + pl(exit_)

    def solve_shift(self, target):
        lo_all, hi_all = m._NOSE_SHIFT_MIN, m._NOSE_SHIFT_MAX
        prev, span = None, None
        for k in range(97):
            s = lo_all + (hi_all - lo_all) * k / 96
            try:
                diff = self.total_len(s) - target
            except ValueError:
                if prev is not None:
                    break
                continue
            if prev is not None and prev[1] * diff < 0:
                span = (prev[0], s)
                break
            prev = (s, diff)
        if span is None:
            raise ValueError("shift 无变号区间")
        lo, hi = span
        flo = self.total_len(lo) - target
        for _ in range(60):
            mid = (lo + hi) / 2
            f = self.total_len(mid) - target
            if flo * f < 0:
                hi = mid
            else:
                lo, flo = mid, f
        return (lo + hi) / 2


def min_dist(pa, pb):
    return min((p - q).length for p in pa for q in pb)


def current_paths(res):
    """现状(yaw 反解)的 entry/ring/exit 采样。"""
    pose = m._nose_pose(res)
    _c, fillets = m._loop_fillets(res, 0.0, pose)
    nose = m._nose_layout(res, 0.0, pose)

    def arc_pts(fl, count=96):
        return [fl.c + m._rotv(fl.ts - fl.c, fl.n, fl.tau * k / count)
                for k in range(count + 1)]

    entry = ([fillets[1].te + (fillets[2].ts - fillets[1].te) * (k / 96)
              for k in range(97)] + arc_pts(fillets[2]))
    exit_ = (arc_pts(fillets[3]) +
             [fillets[3].te + (fillets[4].ts - fillets[3].te) * (k / 96)
              for k in range(97)])
    ring = arc_pts(nose.pos, 384)
    return entry, ring, exit_, nose.pos


# ================== 数值验证 ==================
res80 = make_res(80.0)
need = m._nose_cross_clearance_need(res80)
D = need  # 抬高量按净距需求取(略保守,最终版本按净距反解)
r_out_dy = (res80.inp.n_turns - 1) / 2 * res80.hbd
print(f"need={need:.2f} D={D:.2f} Rc={19.25} 外匝dy={r_out_dy:.2f}")

# 参照长度:现状 slot角→slot角 的鼻端路径长(保证同一 LLM 语境下可比)
cur = Scheme(res80, lambda th: 0.0)


def report(tag, profile_factory):
    for deg in (70.0, 80.0, 90.0):
        res = make_res(deg)
        sch = Scheme(res, profile_factory())
        # 目标长度: 现状同 seita3 的 slot1→slot4 鼻端路径长
        entry0, ring0, exit0, _ = current_paths(res)
        # 现状 entry 从 fillets[1].te 起,不含 slot 角;统一改用两端 slot 虚拟角:
        # 直接用平面方案(profile=0)在与现状相同 LLM 下的长度做目标没有意义;
        # 这里以"现状总长(闭环=LLM)"为不变量 → 鼻端段目标 = LLM - 其余段。
        # 其余段与 shift 无关部分:slot直边+槽口角;斜边长随 shift 变,已含在
        # sch.total_len 内(slot 虚拟角→slot 虚拟角)。现状的可比长度:
        pose = m._nose_pose(res)
        _c, fillets = m._loop_fillets(res, 0.0, pose)
        nose = m._nose_layout(res, 0.0, pose)

        def arclen(f):
            return (f.ts - f.c).length * f.tau
        target = ((fillets[2].ts - _c[1][0]).length + arclen(fillets[2])
                  + arclen(nose.pos) + arclen(fillets[3])
                  + (_c[4][0] - fillets[3].te).length)
        try:
            s = sch.solve_shift(target)
            entry, ring, exit_, fl2, fl3 = sch.paths(s)
            d_cross = min_dist(entry, exit_)
            ring_out = [sch.ring_point(s, SWEEP * k / 256, r_out_dy)
                        for k in range(257)]
            d_er = min_dist(entry, ring_out)
            d_xr = min_dist(exit_, ring_out)
            print(f"  {tag} seita3={deg:4.0f}°: shift={s:8.2f} "
                  f"臂-臂={d_cross:6.2f}(需{need:.1f}) "
                  f"入臂-外环={d_er:6.2f} 出臂-外环={d_xr:6.2f}")
        except ValueError as e:
            print(f"  {tag} seita3={deg:4.0f}°: 不可行: {e}")


print("方案A 浅螺旋(全环均匀, Q2=+D/2 → Q3=-D/2):")
report("A", lambda: (lambda th: D / 2 - D * th / SWEEP))
delta = math.radians(115.0)
print("方案C 扇区下潜(Q2=+D, 首115°内smoothstep→0, 其余严格共面):")
report("C", lambda: (lambda th: D * (1 - smoothstep(th / delta))
                     if th < delta else 0.0))

# ================== 示意图 ==================
def target_len(res):
    pose = m._nose_pose(res)
    _c, fillets = m._loop_fillets(res, 0.0, pose)
    nose = m._nose_layout(res, 0.0, pose)

    def arclen(f):
        return (f.ts - f.c).length * f.tau
    return ((fillets[2].ts - _c[1][0]).length + arclen(fillets[2])
            + arclen(nose.pos) + arclen(fillets[3])
            + (_c[4][0] - fillets[3].te).length)


target80 = target_len(res80)
rows = []
entry, ring, exit_, _arc = current_paths(res80)
rows.append(("现状: 盘面绕鼻端中心线偏航30°(硬币斜骑), 用户否定", 
             entry, ring, exit_, None, None))

schA = Scheme(res80, lambda th: D / 2 - D * th / SWEEP)
sA = schA.solve_shift(target80)
eA, rA, xA, _f2, _f3 = schA.paths(sA)
rows.append((f"方案A 浅螺旋: 盘面=带面, 全环均匀升角(锁紧垫圈式), D={D:.1f}mm",
             eA, rA, xA, schA, sA))

schC = Scheme(res80, lambda th: D * (1 - smoothstep(th / delta))
              if th < delta else 0.0)
sC = schC.solve_shift(target80)
eC, rC, xC, _f2, _f3 = schC.paths(sC)
rows.append((f"方案C 扇区下潜: 盘后2/3严格共面, 入口臂抬D={D:.1f}mm压过交叉, "
             f"首115°下潜回平面", eC, rC, xC, schC, sC))

fig = plt.figure(figsize=(17, 15))
views = [(0, -90, "径向看(从电机外侧向轴心)"),
         (90, -90, "沿盘面法向看(垂直环平面)"),
         (0, 0, "切向看(侧面, 看鼻端厚度)")]
for irow, (title, ent, rng, ext, sch, s) in enumerate(rows):
    for icol, (elev, azim, vtitle) in enumerate(views):
        ax = fig.add_subplot(3, 3, irow * 3 + icol + 1, projection="3d")
        segs, colors = [], []
        # 束心环(红) + 内/外极限匝(蓝/绿)
        if sch is not None:
            for dy, col in [(-r_out_dy, "#1f77b4"), (0.0, "#d62728"),
                            (r_out_dy, "#2ca02c")]:
                ring_i = [tuple(sch.ring_point(s, SWEEP * k / 256, dy))
                          for k in range(257)]
                segs += list(zip(ring_i, ring_i[1:]))
                colors += [col] * 256
        else:
            ring_t = [tuple(p) for p in rng]
            segs += list(zip(ring_t, ring_t[1:]))
            colors += ["#d62728"] * (len(ring_t) - 1)
        ent_t = [tuple(p) for p in ent]
        ext_t = [tuple(p) for p in ext]
        segs += list(zip(ent_t, ent_t[1:])) + list(zip(ext_t, ext_t[1:]))
        colors += (["#ff7f0e"] * (len(ent_t) - 1)
                   + ["#9467bd"] * (len(ext_t) - 1))
        lc = Line3DCollection(segs, colors=colors, linewidths=1.3)
        ax.add_collection3d(lc)
        allp = ent_t + ext_t
        cx = sum(p[0] for p in allp) / len(allp)
        cy = sum(p[1] for p in allp) / len(allp)
        cz = sum(p[2] for p in allp) / len(allp)
        r = 100
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        if icol == 0:
            ax.set_title(f"{title}\n{vtitle}", fontsize=9)
        else:
            ax.set_title(vtitle, fontsize=9)
        ax.set_axis_off()
fig.suptitle("鼻端束中心线族对比 (红=束心环, 蓝/绿=内/外极限匝, "
             "橙=入口臂(上层→鼻), 紫=出口臂(鼻→下层))", fontsize=11)
out = Path(__file__).resolve().parents[1] / "tmp" / "nose_scheme_compare.png"
fig.savefig(out, dpi=100, bbox_inches="tight")
print("saved", out)
