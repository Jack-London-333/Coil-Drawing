"""定位 yaw=0 时交叉点的精确位置(相对环心的面内坐标/深度)。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(80.0)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)

yaw = 0.0
shift = m.nose_axial_shift(res, yaw)
pose = (shift, yaw)
nose = m._nose_layout(res, 0.0, pose)
arc = nose.pos
axis = (arc.ma - arc.c).normalized()
n = arc.n
xhat = m.b3d.Vector(1, 0, 0)
_c, fillets = m._loop_fillets(res, 0.0, pose)


def coords(p):
    v = p - arc.c
    return v.dot(xhat), v.dot(axis), v.dot(n)  # (面内x, 面内轴向, 深度)


def sample(chord_a, chord_b, fillet, arc_first, count=200):
    arc_pts = [("arc", fillet.c + m._rotv(fillet.ts - fillet.c, fillet.n,
                                          fillet.tau * k / count))
               for k in range(count + 1)]
    line_pts = [("line", chord_a + (chord_b - chord_a) * (k / count))
                for k in range(count + 1)]
    return (arc_pts + line_pts) if arc_first else (line_pts + arc_pts)


entry = sample(fillets[1].te, fillets[2].ts, fillets[2], False)
exit_ = sample(fillets[3].te, fillets[4].ts, fillets[3], True)
best = None
for ta, p in entry:
    for tb, q in exit_:
        d = (p - q).length
        if best is None or d < best[0]:
            best = (d, ta, p, tb, q)
d, ta, p, tb, q = best
print(f"最小距={d:.3f}")
print(f"入口臂({ta}) 面内x={coords(p)[0]:8.2f} 轴向={coords(p)[1]:8.2f} 深度={coords(p)[2]:7.2f}")
print(f"出口臂({tb}) 面内x={coords(q)[0]:8.2f} 轴向={coords(q)[1]:8.2f} 深度={coords(q)[2]:7.2f}")
print(f"Q2  面内x={coords(nose.q2)[0]:8.2f} 轴向={coords(nose.q2)[1]:8.2f}")
print(f"Q3  面内x={coords(nose.q3)[0]:8.2f} 轴向={coords(nose.q3)[1]:8.2f}")
print(f"P2(虚拟角) {tuple(round(v,1) for v in coords(nose.p2))}")
print(f"P3(虚拟角) {tuple(round(v,1) for v in coords(nose.p3))}")
print(f"slot1 {tuple(round(v,1) for v in coords(_c[1][0]))}")
print(f"slot4 {tuple(round(v,1) for v in coords(_c[4][0]))}")
print(f"环半径 Rc={(arc.ts-arc.c).length:.2f} 外匝环半径={(arc.ts-arc.c).length + 3.5*res.hbd:.2f}")
