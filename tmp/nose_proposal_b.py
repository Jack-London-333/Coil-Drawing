"""方案 b' 可行性验算(不改 src):

盘面贴带面(yaw=0)+ 入口臂整体抬高 D 从交叉点上方压过 +
环的起始扇区 delta 内平滑下潜 D→0,其余环保持严格平面。

验证:
- rd2 紧凑相切在 seita3=70/80/90 仍有解;
- LLM 反解 shift 有解;
- 交叉处两臂中心距 >= need;
- 入口/出口臂与平面环外匝纤维的净距;
- 输出三维中心线示意图。
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


def build_nose(res, shift, D, delta, positive=True):
    """返回 (center, axis, n, d_in, Q2, t2, Q3, t3, ring_pts(束心))."""
    inp = res.inp
    xhat = b3d.Vector(1, 0, 0)
    rn = (res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose
    zn = res.l2 / 2 + res.cc
    Rc = inp.rd_nose + res.wa_turn / 2
    sin3, cos3 = math.sin(inp.seita3), math.cos(inp.seita3)
    z = 1.0 if positive else -1.0
    axis = b3d.Vector(0, cos3, z * sin3)
    d_in = xhat * z
    n = d_in.cross(axis)   # 环法向
    center = b3d.Vector(0, rn, z * zn) - axis * shift
    r0 = (d_in * math.cos(BETA) - axis * math.sin(BETA))  # Q2 单位半径
    Q2 = center + r0 * Rc + n * D
    Q3 = center - (d_in * math.cos(BETA) + axis * math.sin(BETA)) * Rc
    t2 = n.cross(r0).normalized()          # θ=0 平面切向(下潜零斜率)
    r3 = (Q3 - center).normalized()
    t3 = n.cross(r3).normalized()

    def ring_point(theta, dy=0.0):
        rdir = m._rotv(r0, n, theta)
        h = D * (1 - smoothstep(theta / delta)) if theta < delta else 0.0
        return center + rdir * (Rc + dy) + n * h

    return center, axis, n, d_in, Q2, t2, Q3, t3, ring_point


def ring_length(res, shift, D, delta, count=720):
    _c, _a, _n, _d, _q2, _t2, _q3, _t3, rp = build_nose(res, shift, D, delta)
    pts = [rp(SWEEP * k / count) for k in range(count + 1)]
    return sum((b - a).length for a, b in zip(pts, pts[1:]))


def layout(res, shift, D, delta):
    """入口/出口 rd2 + 斜边端点(束心线)。返回采样折线与长度。"""
    inp = res.inp
    center, axis, n, d_in, Q2, t2, Q3, t3, rp = build_nose(
        res, shift, D, delta)
    th1, th2 = -res.fai1, +res.fai2
    slot1 = m._cyl(res.rr1 + res.hc / 2, th1, +res.l2 / 2)
    slot4 = m._cyl(res.rr2 + res.hc / 2, th2, +res.l2 / 2)

    # 复用 src 的紧凑 rd2 求解(通过 _nose_layout 的私有函数不可直接调,
    # 这里重写同逻辑)
    def compact_virtual(slot, endpoint, toward, radius2):
        def state(ext):
            virtual = endpoint - toward * ext
            incoming = virtual - slot
            u = incoming.normalized()
            tau = math.acos(max(-1, min(1, u.dot(toward))))
            return ext - radius2 * math.tan(tau / 2), tau, virtual
        lo, hi = 0.0, radius2 * math.sqrt(3.0)
        if state(hi)[0] < -1e-10:
            raise ValueError("rd2 不可行(hi<0)")
        for _ in range(64):
            mid = (lo + hi) / 2
            if state(mid)[0] >= 0:
                hi = mid
            else:
                lo = mid
        val, tau, virtual = state(hi)
        if tau > math.radians(120) + 1e-9:
            raise ValueError(f"rd2 转角 {math.degrees(tau):.1f}°>120°")
        direct = (endpoint - slot).length
        ratio = ((virtual - slot).length + hi) / direct
        if ratio > 1.5:
            raise ValueError(f"rd2 路径比 {ratio:.2f}>1.5")
        return virtual, tau

    p2, tau2 = compact_virtual(slot1, Q2, t2, inp.r_bend_nose)
    p3, tau3 = compact_virtual(slot4, Q3, t3 * -1.0, inp.r_bend_nose)

    def fillet(q, p, r_, radius):
        return m._fillet_corner(q, p, r_, radius, 1.0)

    # 圆角2: 来向 slot1→p2,去向 p2→Q2;圆角3: Q3→p3→slot4
    fl2 = fillet(slot1, p2, Q2, inp.r_bend_nose)
    fl3 = fillet(Q3, p3, slot4, inp.r_bend_nose)

    def arc_pts(fl, count=64):
        return [fl.c + m._rotv(fl.ts - fl.c, fl.n, fl.tau * k / count)
                for k in range(count + 1)]

    entry_line = [slot1 + (fl2.ts - slot1) * (k / 64) for k in range(65)]
    entry = entry_line + arc_pts(fl2)
    exit_line = [fl3.te + (slot4 - fl3.te) * (k / 64) for k in range(65)]
    exit_ = arc_pts(fl3) + exit_line

    def polylen(pts):
        return sum((b - a).length for a, b in zip(pts, pts[1:]))

    ring_pts = [rp(SWEEP * k / 720) for k in range(721)]
    # 半环长: slot1→Q2→(ring)→Q3→slot4 的鼻端部分(不含槽内/槽口段)
    nose_len = polylen(entry) + polylen(ring_pts) + polylen(exit_)
    info = dict(center=center, axis=axis, n=n, Q2=Q2, Q3=Q3,
                entry=entry, exit=exit_, ring=ring_pts, rp=rp,
                fl2=fl2, fl3=fl3, tau2=tau2, tau3=tau3,
                slot1=slot1, slot4=slot4)
    return nose_len, info


def solve_shift(res, D, delta, target_half):
    """二分 shift 使鼻端段长 = target_half(等效 LLM 守恒验证)。

    先在可行区间内扫描变号段(不可行的 shift 直接跳过,与 src
    的 bracket 行为一致)。
    """
    lo_all, hi_all = -80.0, 400.0
    prev = None
    span = None
    for k in range(49):
        s = lo_all + (hi_all - lo_all) * k / 48
        try:
            diff = layout(res, s, D, delta)[0] - target_half
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
    flo = layout(res, lo, D, delta)[0] - target_half
    for _ in range(60):
        mid = (lo + hi) / 2
        f = layout(res, mid, D, delta)[0] - target_half
        if flo * f < 0:
            hi = mid
        else:
            lo, flo = mid, f
    return (lo + hi) / 2


def min_dist(pa, pb):
    best, arg = 1e18, None
    for p in pa:
        for q in pb:
            d = (p - q).length
            if d < best:
                best, arg = d, (p, q)
    return best, arg


inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(80.0)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)
need = m._nose_cross_clearance_need(res)
Rc = inp.rd_nose + res.wa_turn / 2
n_t = inp.n_turns
r_out_dy = (n_t - 1) / 2 * res.hbd

# 目标: 与现 yaw 方案同一 LLM。取现方案 yaw=0 的鼻端段长做对照没意义,
# 直接检查 shift 可行域内长度单调覆盖(用现方案的 shift 值近似目标)。
# 简化: 用现 auto pose 的鼻端段长为目标(总 LLM 相同,槽段不变)。
pose_now = m._nose_pose(res)
_c, fl_now = m._loop_fillets(res, 0.0, pose_now)
nose_now = m._nose_layout(res, 0.0, pose_now)


def now_nose_len():
    def arclen(f):
        return (f.ts - f.c).length * f.tau
    total = (fl_now[1].te - _c[1][0]).length * 0  # 忽略
    pts = 0.0
    pts += (fl_now[2].ts - fl_now[1].te).length   # 斜边1→2
    pts += arclen(fl_now[2])
    pts += arclen(nose_now.pos)
    pts += arclen(fl_now[3])
    pts += (fl_now[4].ts - fl_now[3].te).length   # 斜边3→4
    return pts


target = now_nose_len()
print(f"参照(现方案)鼻端段长={target:.3f} mm, need={need:.3f}")

D = need + 0.5           # 入口环端抬高
for delta_deg in (90.0, 115.0):
    delta = math.radians(delta_deg)
    for deg in (70.0, 80.0, 90.0):
        inp2 = CoilInput()
        inp2.n_turns = 8
        inp2.f_nose = 20.0
        inp2.seita3 = math.radians(deg)
        inp2.rd_nose = 15.0
        inp2.r_bend_slot = 30.0
        inp2.r_bend_nose = 30.0
        r2 = compute(inp2)
        try:
            s = solve_shift(r2, D, delta, target)
            L, info = layout(r2, s, D, delta)
            d_cross, arg = min_dist(info["entry"], info["exit"])
            # 外匝环纤维(平面段 + 下潜段, dy=+r_out_dy)
            ring_out = [info["rp"](SWEEP * k / 360, r_out_dy)
                        for k in range(361)]
            d_e_ring, _ = min_dist(info["entry"], ring_out)
            d_x_ring, _ = min_dist(info["exit"], ring_out)
            # 下潜段最大坡度
            rp = info["rp"]
            slope = 0.0
            for k in range(1, 200):
                a = rp(delta * (k - 1) / 200)
                b = rp(delta * (k + 1) / 200)
                nv = info["n"]
                dh = abs((b - a).dot(nv))
                dl = ((b - a) - nv * (b - a).dot(nv)).length
                slope = max(slope, dh / max(dl, 1e-9))
            print(f"delta={delta_deg:5.0f}° seita3={deg:4.0f}°: "
                  f"shift={s:8.3f} 臂-臂={d_cross:7.2f}(需{need:.1f}) "
                  f"入臂-外环={d_e_ring:6.2f} 出臂-外环={d_x_ring:6.2f} "
                  f"下潜最大坡度={math.degrees(math.atan(slope)):5.1f}°")
        except ValueError as e:
            print(f"delta={delta_deg}° seita3={deg}°: 不可行: {e}")

# ---- 示意图(seita3=80°, delta=115°) ----
delta = math.radians(115.0)
s = solve_shift(res, D, delta, target)
_L, info = layout(res, s, D, delta)

fig = plt.figure(figsize=(16, 7))
for iax, (elev, azim, title) in enumerate(
        (( 8, -90, "径向看(从电机外侧):盘贴带面"),
         (80, -90, "沿环法向看:交叉+卷回"),
         ( 8,   0, "切向看(侧面):厚度=带宽,入口臂压过交叉")), 1):
    ax = fig.add_subplot(1, 3, iax, projection="3d")
    segs, colors = [], []
    for dy, col in [(-r_out_dy, "#1f77b4"), (0.0, "#d62728"),
                    (r_out_dy, "#2ca02c")]:
        ring = [tuple(info["rp"](SWEEP * k / 360, dy)) for k in range(361)]
        segs += list(zip(ring, ring[1:]))
        colors += [col] * 360
    ent = [tuple(p) for p in info["entry"]]
    ext = [tuple(p) for p in info["exit"]]
    segs += list(zip(ent, ent[1:])) + list(zip(ext, ext[1:]))
    colors += ["#ff7f0e"] * (len(ent) - 1) + ["#9467bd"] * (len(ext) - 1)
    lc = Line3DCollection(segs, colors=colors, linewidths=1.4)
    ax.add_collection3d(lc)
    c = info["center"]
    r = 120
    ax.set_xlim(c.X - r, c.X + r)
    ax.set_ylim(c.Y - r, c.Y + r)
    ax.set_zlim(c.Z - r, c.Z + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
fig.suptitle("方案b': 盘面=带面(yaw=0), 入口臂抬高D≈13mm 从交叉点上方压过, "
             "环首扇区115°内下潜回平面 (红=束心,蓝/绿=内外极限匝,橙=入口臂,紫=出口臂)")
out = Path(__file__).resolve().parents[1] / "tmp" / "proposal_b_sketch.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
