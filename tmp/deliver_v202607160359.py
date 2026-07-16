"""交付 v202607160359:按 config 生成 STEP + 全套渲染。

用法:
  uv run python tmp/deliver_v202607160359.py step     # 80° STEP+渲染
  uv run python tmp/deliver_v202607160359.py closeups # 70°/90° 特写
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

from coildrawing.config_io import load_config
from coildrawing.engine import compute
from coildrawing import model3d as m

ROOT = Path(__file__).resolve().parents[1]
VDIR = ROOT / "some_user_problems" / "software-v202607160359-test"
PICS = VDIR / "pics"
CONF = VDIR / "test 202607160359" / "1" / "config.txt"
STEP = VDIR / "test 202607160359" / "1" / "output" / "coil_3d.step"
PICS.mkdir(parents=True, exist_ok=True)
STEP.parent.mkdir(parents=True, exist_ok=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "step"


def build(inp):
    res = compute(inp)
    parts = [m._finish_part(p)
             for p in m.build_coil_parts(res, detailed=True)]
    return res, parts


def render(parts, center, r, title, fname, elev, azim):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, color, pts, tris in parts:
        polys = [[pts[i] for i in tri] for tri in tris]
        pc = Poly3DCollection(
            polys, alpha=0.97 if name.startswith("铜") else 0.35)
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
    fig.savefig(PICS / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", PICS / fname, flush=True)


def tessellate(parts):
    out = []
    for part in parts:
        verts, tris = part.solid.tessellate(0.6)
        out.append((part.name, part.color,
                    [(v.X, v.Y, v.Z) for v in verts], tris))
    return out


if mode == "step":
    inp = load_config(CONF)
    res = compute(inp)
    t0 = time.time()
    names = m.export_step(res, str(STEP))
    print(f"STEP {STEP} ({STEP.stat().st_size} bytes, {len(names)} parts, "
          f"{time.time()-t0:.0f}s)", flush=True)
    import hashlib
    print("sha256:", hashlib.sha256(STEP.read_bytes()).hexdigest(),
          flush=True)
    shift, drop = m._nose_pose(res)
    print(f"pose: shift={shift:.6f} drop={drop:.6f} LLM={res.llm}",
          flush=True)

    parts = [m._finish_part(p)
             for p in m.build_coil_parts(res, detailed=True)]
    meshes = tessellate(parts)
    nose = m._nose_layout(res)
    nc = nose.pos.c
    all_pts = [p for _n, _c2, pts, _t in meshes for p in pts]
    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)
    big_r = max(max(abs(p[0] - cx) for p in all_pts),
                max(abs(p[1] - cy) for p in all_pts),
                max(abs(p[2]) for p in all_pts)) * 1.05
    render(meshes, (nc.X, nc.Y, nc.Z), 90,
           "径向视图(从电机外侧) seita3=80°", "径向视图_seita80.png", 0, -90)
    render(meshes, (nc.X, nc.Y, nc.Z), 130,
           "轴向视图 seita3=80°", "轴向视图_seita80.png", 90, -90)
    render(meshes, (nc.X, nc.Y, nc.Z), 90,
           "切向视图(侧面) seita3=80°", "切向视图_seita80.png", 0, 0)
    render(meshes, (nc.X, nc.Y, nc.Z), 65,
           "鼻端特写 seita3=80°", "鼻端特写_seita80.png", 28, -55)
    render(meshes, (cx, cy, 0.0), big_r,
           "整体 seita3=80°", "整体_seita80.png", 22, -60)
else:
    for deg in (70.0, 90.0):
        inp = load_config(CONF)
        inp.seita3 = math.radians(deg)
        res, parts = build(inp)
        shift, drop = m._nose_pose(res)
        print(f"seita3={deg}: shift={shift:.3f} drop={drop:.3f}",
              flush=True)
        meshes = tessellate(parts)
        nose = m._nose_layout(res)
        nc = nose.pos.c
        tag = f"seita{int(deg)}"
        render(meshes, (nc.X, nc.Y, nc.Z), 65,
               f"鼻端特写 seita3={deg:.0f}°", f"鼻端特写_{tag}.png", 28, -55)
        render(meshes, (nc.X, nc.Y, nc.Z), 90,
               f"切向视图(侧面) seita3={deg:.0f}°", f"切向视图_{tag}.png",
               0, 0)
