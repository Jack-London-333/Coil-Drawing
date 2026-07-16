"""逐阶段计时 N=8 精细模型构建,找出慢点。"""
import math
import sys
import time
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

t0 = time.time()
segs, info = m._wire_segments(res)
print(f"segments: {len(segs)} in {time.time()-t0:.1f}s")
ml = [i for i, s in enumerate(segs) if isinstance(s, m._MultiLoft)]
print(f"MultiLofts: {len(ml)} (每个 {len(segs[ml[0]].sections)} 截面)")

w, h, strands = m._strand_grid(res)
s = strands[0]

t0 = time.time()
one = m._seg_solid(segs[ml[0]], s["b"], s["h"], s["x"], s["y"])
print(f"单个 nose MultiLoft 实体: {time.time()-t0:.1f}s, "
      f"faces={len(one.faces())}")

t0 = time.time()
solids = []
for g in segs:
    solids.append(m._seg_solid(g, s["b"], s["h"], s["x"], s["y"]))
print(f"全部分段实体: {time.time()-t0:.1f}s ({len(solids)})")

t0 = time.time()
copper = m._join(solids)
print(f"copper join: {time.time()-t0:.1f}s, solids={len(copper.solids())}, "
      f"faces={len(copper.faces())}")

t0 = time.time()
fp = m._finish_part(m.CoilPart("铜导线", copper, m.COPPER_COLOR))
print(f"finish: {time.time()-t0:.1f}s, faces={len(fp.solid.faces())}")
