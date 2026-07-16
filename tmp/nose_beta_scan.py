"""yaw=0 时扫描交叉半角 beta:交叉净距、外匝环-臂净距、LLM shift。"""
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
need = m._nose_cross_clearance_need(res)
rc = inp.rd_nose + res.wa_turn / 2
n_t = inp.n_turns
print(f"need={need:.2f}  Rc={rc:.2f}  HBD={res.hbd:.3f}")

def sample_path(chord_a, chord_b, fillet, arc_first, count=64):
    arc_pts = [fillet.c + m._rotv(fillet.ts - fillet.c, fillet.n,
                                  fillet.tau * k / count)
               for k in range(count + 1)]
    line_pts = [chord_a + (chord_b - chord_a) * (k / count)
                for k in range(count + 1)]
    return (arc_pts + line_pts) if arc_first else (line_pts + arc_pts)

def min_dist(pa, pb):
    return min((p - q).length for p in pa for q in pb)

orig_beta = m._NOSE_CROSS_BETA
for beta_deg in (25, 30, 35, 40, 45, 50):
    m._NOSE_CROSS_BETA = math.radians(beta_deg)
    try:
        if hasattr(res, "_nose_pose3d"):
            del res._nose_pose3d
        shift = m.nose_axial_shift(res, 0.0)
        pose = (shift, 0.0)
        _c, fillets = m._loop_fillets(res, 0.0, pose)
        nose = m._nose_layout(res, 0.0, pose)
        arc = nose.pos
        n = arc.n

        entry = sample_path(fillets[1].te, fillets[2].ts, fillets[2], False)
        exit_ = sample_path(fillets[3].te, fillets[4].ts, fillets[3], True)
        d_cross = min_dist(entry, exit_)

        # 臂(束中心线±极端匝偏移)与盘最外匝环纤维的最小距离。
        # 盘上最外匝(i=7): 环半径 rc+3.5*HBD,在环平面内。
        r_out = rc + (n_t - 1) / 2 * res.hbd
        y0 = (arc.ts - arc.c).normalized()
        ring_pts = [arc.c + m._rotv(y0, n, arc.tau * k / 128) * r_out
                    for k in range(129)]
        # 臂上各匝纤维: 中心线 + y方向 dy(近鼻端 y≈环半径方向,
        # 保守直接采样束中心线与极端匝纤维)。
        d_ring = min(min_dist(entry, ring_pts), min_dist(exit_, ring_pts))
        print(f"beta={beta_deg}: sweep={180+2*beta_deg}°  shift={shift:8.3f}  "
              f"臂-臂={d_cross:7.3f} (需{need:.1f})  "
              f"束心线-外匝环心线={d_ring:7.3f}")
    except ValueError as e:
        print(f"beta={beta_deg}: {e}")
m._NOSE_CROSS_BETA = orig_beta
