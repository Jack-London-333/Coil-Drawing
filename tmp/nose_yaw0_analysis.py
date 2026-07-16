"""侦察:yaw=0(盘面=带面)时交叉净距的天然值与缺口。

用 test 202607142334 的 config(N=8, seita3=80°, RD=15, F=20)计算:
1. 各 yaw 下交叉处入口/出口两臂中心线最小距离(现有采样法);
2. yaw=0 时按 LLM 反解 shift 后,入口/出口臂交叉点位置、
   两臂在交叉附近沿环法向(≈径向)的出平面深度;
3. seita3=70/80/90 时环平面法向的变化(确认姿态确实随 seita3 转)。
"""
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

print(f"LLM={res.llm:.4f}  Rc={inp.rd_nose + res.wa_turn/2:.3f}  "
      f"HBD={res.hbd:.3f}  WA={res.wa_turn:.3f}  HAD={res.had:.3f}")

need = m._nose_cross_clearance_need(res)
print(f"need(交叉净距需求)={need:.3f} mm")

pose_auto = m._nose_pose(res)
print(f"当前自动姿态: shift={pose_auto[0]:.3f}  yaw={math.degrees(pose_auto[1]):.2f}°")

# --- 各 yaw 下的净距 ---
def sample_path(chord_a, chord_b, fillet, arc_first, count=48):
    pts = []
    arc_pts = [fillet.c + m._rotv(fillet.ts - fillet.c, fillet.n,
                                  fillet.tau * k / count)
               for k in range(count + 1)]
    line_pts = [chord_a + (chord_b - chord_a) * (k / count)
                for k in range(count + 1)]
    return (arc_pts + line_pts) if arc_first else (line_pts + arc_pts)


def min_dist(pa, pb):
    best, arg = float("inf"), None
    for p in pa:
        for q in pb:
            d = (p - q).length
            if d < best:
                best, arg = d, (p, q)
    return best, arg


for yaw_deg in (0.0, 5.0, 10.0, 20.0, 30.0):
    yaw = math.radians(yaw_deg)
    try:
        shift = m.nose_axial_shift(res, yaw)
    except ValueError as e:
        print(f"yaw={yaw_deg:5.1f}°: shift 反解失败: {e}")
        continue
    _c, fillets = m._loop_fillets(res, 0.0, (shift, yaw))
    entry = sample_path(fillets[1].te, fillets[2].ts, fillets[2], False)
    exit_ = sample_path(fillets[3].te, fillets[4].ts, fillets[3], True)
    d, arg = min_dist(entry, exit_)
    print(f"yaw={yaw_deg:5.1f}°: shift={shift:8.3f}  交叉最小距={d:7.3f} mm"
          f"  (需 {need:.2f})")

# --- yaw=0 细看几何 ---
yaw = 0.0
shift = m.nose_axial_shift(res, yaw)
nose = m._nose_layout(res, 0.0, (shift, yaw))
arc = nose.pos
axis = (arc.ma - arc.c).normalized()
n = arc.n  # 环平面法向
print(f"\nyaw=0: 环心={tuple(round(v,2) for v in arc.c)}  "
      f"轴(apex方向)={tuple(round(v,3) for v in axis)}  "
      f"法向={tuple(round(v,3) for v in n)}")
_c, fillets = m._loop_fillets(res, 0.0, (shift, yaw))

def depth(p):
    """出平面深度: 沿环法向到环平面的有符号距离(正=沿n)。"""
    return (p - arc.c).dot(n)

# 入口臂(斜边 1→2 + rd2 弯角2)与出口臂(rd2 弯角3 + 斜边 3→4)
entry = sample_path(fillets[1].te, fillets[2].ts, fillets[2], False, 96)
exit_ = sample_path(fillets[3].te, fillets[4].ts, fillets[3], True, 96)
d, (pe, px) = min_dist(entry, exit_)
print(f"交叉最小距={d:.3f} mm")
print(f"  入口臂最近点 深度={depth(pe):8.3f} 环面内半径="
      f"{((pe-arc.c) - n*depth(pe)).length:8.3f}")
print(f"  出口臂最近点 深度={depth(px):8.3f} 环面内半径="
      f"{((px-arc.c) - n*depth(px)).length:8.3f}")
print(f"  斜边1→2 槽侧端深度={depth(fillets[1].te):.2f}  鼻侧端深度={depth(fillets[2].ts):.2f}")
print(f"  斜边3→4 鼻侧端深度={depth(fillets[3].te):.2f}  槽侧端深度={depth(fillets[4].ts):.2f}")
print(f"  slot1 上层槽口角深度={depth(_c[1][0]):.2f}")
print(f"  slot4 下层槽口角深度={depth(_c[4][0]):.2f}")

# rd2 弯角2/3 的圆弧平面与环平面的夹角
for i, tag in ((2, "入口 rd2"), (3, "出口 rd2")):
    ang = math.degrees(math.acos(max(-1, min(1, abs(fillets[i].n.dot(n))))))
    print(f"  {tag} 弯角平面与环平面夹角={ang:.1f}°  转角={math.degrees(fillets[i].tau):.1f}°")

# --- seita3 扫描(法向变化) ---
print()
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
        s2 = m.nose_axial_shift(r2, 0.0)
        arc2 = m._nose_layout(r2, 0.0, (s2, 0.0)).pos
        print(f"seita3={deg:4.0f}°: shift={s2:8.3f}  法向="
              f"{tuple(round(v,3) for v in arc2.n)}  "
              f"apex={tuple(round(v,3) for v in (arc2.ma-arc2.c).normalized())}")
    except ValueError as e:
        print(f"seita3={deg}°: {e}")
