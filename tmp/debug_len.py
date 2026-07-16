"""调试:方案b' 鼻端段长随 shift 的变化。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tmp"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "prop", Path(__file__).resolve().parents[1] / "tmp" / "nose_proposal_b.py")
# 不能直接 import(脚本会执行主体)。改为复制关键函数太笨,直接
# 在这里内联最小版本。
import matplotlib
matplotlib.use("Agg")

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

D = 13.1
delta = math.radians(115.0)
xhat = b3d.Vector(1, 0, 0)
rn = (res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose
zn = res.l2 / 2 + res.cc
Rc = inp.rd_nose + res.wa_turn / 2
sin3, cos3 = math.sin(inp.seita3), math.cos(inp.seita3)
axis = b3d.Vector(0, cos3, sin3)
d_in = xhat
n = d_in.cross(axis)
th1, th2 = -res.fai1, +res.fai2
slot1 = m._cyl(res.rr1 + res.hc / 2, th1, +res.l2 / 2)
slot4 = m._cyl(res.rr2 + res.hc / 2, th2, +res.l2 / 2)


def compact_virtual(slot, endpoint, toward, radius2):
    def state(ext):
        virtual = endpoint - toward * ext
        incoming = virtual - slot
        u = incoming.normalized()
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
    _val, tau, virtual = state(hi)
    if tau > math.radians(120) + 1e-9:
        raise ValueError(f"tau={math.degrees(tau):.1f}")
    ratio = ((virtual - slot).length + hi) / (endpoint - slot).length
    if ratio > 1.5:
        raise ValueError(f"ratio={ratio:.2f}")
    return virtual


def nose_len(shift):
    center = b3d.Vector(0, rn, zn) - axis * shift
    r0 = d_in * math.cos(BETA) - axis * math.sin(BETA)
    Q2 = center + r0 * Rc + n * D
    r3u = (d_in * math.cos(BETA) + axis * math.sin(BETA)) * -1.0
    Q3 = center + r3u * Rc
    t2 = n.cross(r0).normalized()
    t3 = n.cross(r3u).normalized()
    p2 = compact_virtual(slot1, Q2, t2, inp.r_bend_nose)
    p3 = compact_virtual(slot4, Q3, t3 * -1.0, inp.r_bend_nose)
    fl2 = m._fillet_corner(slot1, p2, Q2, inp.r_bend_nose, 1.0)
    fl3 = m._fillet_corner(Q3, p3, slot4, inp.r_bend_nose, 1.0)

    def ring_point(theta, dy=0.0):
        rdir = m._rotv(r0, n, theta)
        h = D * (1 - smoothstep(theta / delta)) if theta < delta else 0.0
        return center + rdir * (Rc + dy) + n * h

    ring = [ring_point(SWEEP * k / 720) for k in range(721)]
    L = (fl2.ts - slot1).length
    L += (fl2.ts - fl2.c).length * fl2.tau
    L += (Q2 - fl2.te).length
    L += sum((b - a).length for a, b in zip(ring, ring[1:]))
    L += (fl3.ts - Q3).length
    L += (fl3.ts - fl3.c).length * fl3.tau
    L += (slot4 - fl3.te).length
    return L


for s in (-40, -20, 0, 10, 17, 30, 60, 100, 200):
    try:
        print(f"shift={s:6.1f}: len={nose_len(s):9.3f}")
    except ValueError as e:
        print(f"shift={s:6.1f}: {e}")
