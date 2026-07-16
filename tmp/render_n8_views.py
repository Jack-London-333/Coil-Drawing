"""按 test 202607142334 config(N=8)生成逐匝精细模型并渲染多视图。

用法: uv run python tmp/render_n8_views.py <seita3_deg> <outdir> [--step 路径]
生成:
  径向视图/轴向视图/切向视图/整体/鼻端特写 png
  可选 STEP 导出
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

deg = float(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
step_path = None
if "--step" in sys.argv:
    step_path = Path(sys.argv[sys.argv.index("--step") + 1])

inp = CoilInput()
inp.n_turns = 8
inp.f_nose = 20.0
inp.seita3 = math.radians(deg)
inp.rd_nose = 15.0
inp.r_bend_slot = 30.0
inp.r_bend_nose = 30.0
res = compute(inp)

t0 = time.time()
if step_path is not None:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    names = m.export_step(res, str(step_path))
    print(f"STEP {step_path} ({step_path.stat().st_size} bytes, "
          f"{len(names)} parts, {time.time()-t0:.0f}s)")
    parts = [m._finish_part(p) for p in m.build_coil_parts(res, detailed=True)]
else:
    parts = [m._finish_part(p) for p in m.build_coil_parts(res, detailed=True)]
print(f"parts ready {time.time()-t0:.0f}s")

nose = m._nose_layout(res)
nc = nose.pos.c

meshes = []
for part in parts:
    verts, tris = part.solid.tessellate(0.6)
    pts = [(v.X, v.Y, v.Z) for v in verts]
    meshes.append((part.name, part.color, pts, tris))


def render(view, center, r, title, fname, elev, azim):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, color, pts, tris in meshes:
        polys = [[pts[i] for i in tri] for tri in tris]
        pc = Poly3DCollection(polys, alpha=0.97 if name.startswith("铜") else 0.35)
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(outdir / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", outdir / fname)


all_pts = [p for _n, _c, pts, _t in meshes for p in pts]
cx = sum(p[0] for p in all_pts) / len(all_pts)
cy = sum(p[1] for p in all_pts) / len(all_pts)
cz = 0.0
big_r = max(max(abs(p[0] - cx) for p in all_pts),
            max(abs(p[1] - cy) for p in all_pts),
            max(abs(p[2]) for p in all_pts)) * 1.05

tag = f"seita{int(deg)}"
# 径向视图: 从电机外侧沿 -Y 向轴心看鼻端(+Z 鼻端在 y≈nc.Y)
render("radial", (nc.X, nc.Y, nc.Z), 90,
       f"径向视图(从电机外侧) seita3={deg:.0f}°",
       f"径向视图_{tag}.png", 0, -90)
# 轴向视图: 沿 -Z 看("人"字交叉)
render("axial", (nc.X, nc.Y, nc.Z), 130,
       f"轴向视图 seita3={deg:.0f}°",
       f"轴向视图_{tag}.png", 90, -90)
# 切向视图: 沿 X 看(侧面,鼻端厚度)
render("tangent", (nc.X, nc.Y, nc.Z), 90,
       f"切向视图(侧面) seita3={deg:.0f}°",
       f"切向视图_{tag}.png", 0, 0)
# 鼻端特写(斜视)
render("closeup", (nc.X, nc.Y, nc.Z), 65,
       f"鼻端特写 seita3={deg:.0f}°",
       f"鼻端特写_{tag}.png", 28, -55)
# 整体
render("all", (cx, cy, cz), big_r,
       f"整体 seita3={deg:.0f}°",
       f"整体_{tag}.png", 22, -60)
