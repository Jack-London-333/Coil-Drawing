"""逐阶段计时:匝绝缘 split 与对地壳。"""
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

segs, info = m._wire_segments(res)
w_env, h_env, strands = m._strand_grid(res)
layer = inp.turn_layers[0]
w2 = w_env + 2 * layer.thickness
h2 = min(h_env + 2 * layer.thickness, res.had)

t_all = time.time()
for start, end, material_i in info["turn_ranges"]:
    t0 = time.time()
    leaves = m._turn_mica_shell_parts(segs[start:end], w_env, h_env, w2, h2)
    print(f"匝{material_i+1}: {len(leaves)} leaves, {time.time()-t0:.1f}s")
print(f"匝绝缘总计 {time.time()-t_all:.1f}s")

# 对地壳
t0 = time.time()
loop_segs = m._loop_segments(res)
print(f"loop segs {len(loop_segs)}: {time.time()-t0:.1f}s")
wrap_total = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
w_in = w_env + 2 * wrap_total
h_in = (inp.n_turns - 1) * res.had + h_env + 2 * wrap_total
t0 = time.time()
env_w = w_env + 2 * wrap_total
env_h = min(h_env + 2 * wrap_total, res.had)
cutters = m._lead_path_cutters(info, env_w, env_h)
print(f"cutters: {time.time()-t0:.1f}s")
t0 = time.time()
gparts, ggrow = m._ground_parts(res, w_in, h_in, cutters)
print(f"ground: {time.time()-t0:.1f}s ({len(gparts)} parts)")
for p in gparts:
    t0 = time.time()
    fp = m._finish_part(p)
    print(f"  finish {p.name}: {time.time()-t0:.1f}s")
