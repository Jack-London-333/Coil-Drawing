"""诊断:纤维裕量的约束对(哪两匝、哪个段)。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from coildrawing.engine import CoilInput, compute, strand_grid
from coildrawing import model3d as m

inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(80.0)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)

n = inp.n_turns
w_env, h_env, _ = strand_grid(inp)
wrap = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
ground = sum(l.thickness for l in inp.layers if l.thickness > 0)
half_x = (w_env + 2 * wrap) / 2 + ground
h_cap = res.had
half_y = min(h_env + 2 * wrap, h_cap) / 2 + ground
print(f"half_x={half_x:.2f} half_y={half_y:.2f}")

for drop in (14.0, 20.0, 24.0, 28.58):
    shift = m.nose_axial_shift(res, drop)
    pose = (shift, drop)
    lf = m._loop_frames(res, pose)
    fl = lf.fl

    def dy_slot(i):
        return res.had * (i - (n - 1) / 2)

    def dy_nose(i):
        return res.hbd * (i - (n - 1) / 2)

    def slant_samples(key, dy0, dy1, count=32):
        f0, f1 = lf.slant_frames[key]
        law, _s = lf.slant_laws[key]
        out = []
        for k in range(count + 1):
            lam = k / count
            f = m._twist_frame_at(f0, f1, lam, law)
            dy = dy0 + (dy1 - dy0) * lam
            out.append((f.o + f.y * dy, f.x, f.y, f"slant{key}@{lam:.2f}"))
        return out

    def corner_samples(f, fil, dy, tagp, count=24):
        p0 = f.o + f.y * dy
        out = []
        for k in range(count + 1):
            a = fil.tau * k / count
            out.append((fil.c + m._rotv(p0 - fil.c, fil.n, a),
                        m._rotv(f.x, fil.n, a), m._rotv(f.y, fil.n, a),
                        f"{tagp}@{k/count:.2f}"))
        return out

    f12e = lf.slant_frames[(1, 2)][1]
    entry, exit_ = [], []
    for i in range(n):
        ds, dn = dy_slot(i), dy_nose(i)
        entry += [(p, x, y, i, t) for p, x, y, t in
                  slant_samples((1, 2), ds, dn)]
        entry += [(p, x, y, i, t) for p, x, y, t in
                  corner_samples(f12e, fl[2], dn, "rd2入口")]
        exit_ += [(p, x, y, i, t) for p, x, y, t in
                  corner_samples(lf.f_q3, fl[3], dn, "rd2出口")]
        exit_ += [(p, x, y, i, t) for p, x, y, t in
                  slant_samples((3, 4), dn, ds)]

    best = None
    for pa, xa, ya, ia, ta in entry:
        for pb, xb, yb, ib, tb in exit_:
            d = pa - pb
            dist = d.length
            if dist > 60:
                continue
            u = d.normalized()
            ra = abs(u.dot(xa)) * half_x + abs(u.dot(ya)) * half_y
            rb = abs(u.dot(xb)) * half_x + abs(u.dot(yb)) * half_y
            s = dist - ra - rb - 2 * m._NOSE_CROSS_MARGIN
            if best is None or s < best[0]:
                best = (s, ia, ta, ib, tb, dist, ra, rb)
    s, ia, ta, ib, tb, dist, ra, rb = best
    print(f"drop={drop:6.2f}: slack={s:7.3f}  匝{ia+1}{ta} × 匝{ib+1}{tb} "
          f"dist={dist:.2f} ra={ra:.2f} rb={rb:.2f}")
