"""调试:抬高 Q2 后 rd2 紧凑相切为何不可行。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

b3d = m.b3d
BETA = m._NOSE_CROSS_BETA

inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(80.0)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)

xhat = b3d.Vector(1, 0, 0)
rn = (res.rr1 + res.rr2) / 2 + res.hc / 2 + inp.f_nose
zn = res.l2 / 2 + res.cc
Rc = inp.rd_nose + res.wa_turn / 2
sin3, cos3 = math.sin(inp.seita3), math.cos(inp.seita3)
axis = b3d.Vector(0, cos3, sin3)
d_in = xhat
n = d_in.cross(axis)
shift = 17.0
center = b3d.Vector(0, rn, zn) - axis * shift
r0 = d_in * math.cos(BETA) - axis * math.sin(BETA)
t2 = n.cross(r0).normalized()

th1 = -res.fai1
slot1 = m._cyl(res.rr1 + res.hc / 2, th1, +res.l2 / 2)

for D in (0.0, 5.0, 13.1):
    Q2 = center + r0 * Rc + n * D
    for ext in (0.0, 10.0, 20.0, 30.0 * math.sqrt(3)):
        virtual = Q2 - t2 * ext
        u = (virtual - slot1).normalized()
        tau = math.degrees(math.acos(max(-1, min(1, u.dot(t2)))))
        val = ext - 30.0 * math.tan(math.radians(tau) / 2)
        print(f"D={D:5.1f} ext={ext:6.2f}: tau={tau:7.2f}°  val={val:8.3f}")
    print()
