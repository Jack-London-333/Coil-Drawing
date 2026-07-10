"""数值审计逐匝模型各部件：体积、包围盒、引线与方壳开孔检查。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, compute  # noqa: E402

inp = CoilInput()
inp.corona_on = True
res = compute(inp)

from coildrawing.model3d import (  # noqa: E402
    b3d, build_coil_parts, _wire_segments, _strand_grid)

_, info = _wire_segments(res)
tin, tout = info["tip_in"], info["tip_out"]
print(f"tip_in = ({tin.X:.1f}, {tin.Y:.1f}, {tin.Z:.1f})")
print(f"tip_out= ({tout.X:.1f}, {tout.Y:.1f}, {tout.Z:.1f})")
print(f"bare_len = {info['bare_len']:.1f}")

w_env, h_env, strands = _strand_grid(res)
parts = build_coil_parts(res, detailed=True)
for p in parts:
    bb = p.solid.bounding_box()
    sol = p.solid.solids()
    print(f"{p.name}: vol={p.solid.volume:,.0f} solids={len(sol)} "
          f"bbox z[{bb.min.Z:.0f},{bb.max.Z:.0f}] "
          f"xy[{bb.min.X:.0f},{bb.max.X:.0f}]x[{bb.min.Y:.0f},{bb.max.Y:.0f}]")

def inter_vol(a, b) -> float:
    try:
        inter = a & b
        if inter is None:
            return 0.0
        return inter.volume
    except Exception:
        return 0.0


# 各实体两两之间不应有体积重叠（共面接触允许）
wrap_total = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
names = [p.name for p in parts]
bad = 0
for i in range(len(parts)):
    for j in range(i + 1, len(parts)):
        v = inter_vol(parts[i].solid, parts[j].solid)
        if v > 0.5:
            bad += 1
            print(f"OVERLAP {names[i]} ∩ {names[j]} = {v:.2f}")
print(f"重叠检查完成，异常对数 = {bad}")

# 引线柱区（与方壳开孔同截面）与对地壳的残余交集应≈0
zn = res.l2 / 2 + res.cc
cut_w = w_env + 2 * wrap_total + 0.1
cut_h = h_env + 2 * wrap_total + 0.1
w_bundle = w_env + 2 * wrap_total
for p in parts:
    if not p.name.startswith("对地"):
        continue
    for tag, tip in (("in", tin), ("out", tout)):
        z0 = zn + w_bundle / 2 - 1.0
        hgt = tip.Z - z0
        probe = b3d.Pos(tip.X, tip.Y, z0 + hgt / 2) * b3d.Box(cut_w, cut_h, hgt)
        print(f"  {p.name} ∩ 引线柱[{tag}] vol={inter_vol(p.solid, probe):.3f}")
