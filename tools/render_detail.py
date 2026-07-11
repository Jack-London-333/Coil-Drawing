"""逐匝精细模型渲染检查：整体、接线侧鼻端（引线）、槽内截面。

用法: python tools/render_detail.py [base|multi|dual]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from coildrawing.engine import CoilInput, WireSpec, compute  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "base"

inp = CoilInput()
inp.corona_on = True
inp.cs = 0.3            # 防晕层厚度=CS
if variant == "multi":
    inp.wire1 = WireSpec(b=4.0, h=1.6, t0=0.1, npd=2, ncd=2)
    inp.n_turns = 4
elif variant == "dual":
    inp.wire1 = WireSpec(b=8.2, h=2.0, t0=0.05, npd=1, ncd=1)
    inp.wire2 = WireSpec(b=7.0, h=1.2, t0=0.05, npd=1, ncd=1)
    inp.n_turns = 6

res = compute(inp)

from coildrawing.model3d import b3d, build_coil_parts, _wire_segments  # noqa: E402

_, info = _wire_segments(res)
parts = build_coil_parts(res, detailed=True)
tin, tout = info["tip_in"], info["tip_out"]
nose_c = ((tin.X + tout.X) / 2, (tin.Y + tout.Y) / 2, (tin.Z + tout.Z) / 2 - 30)

fig = plt.figure(figsize=(18, 7))

# ---- 视图1/2：三维渲染 ----
views = [(22, -55, "整体", None), (25, -35, "接线侧鼻端/引线", "nose")]
for iax, (elev, azim, title, zoom) in enumerate(views, start=1):
    ax = fig.add_subplot(1, 3, iax, projection="3d")
    all_pts = []
    for part in parts:
        verts, tris = part.solid.tessellate(1.0)
        pts = [(v.X, v.Y, v.Z) for v in verts]
        all_pts.extend(pts)
        polys = [[pts[i] for i in tri] for tri in tris]
        alpha = 0.95 if part.name.startswith("铜") else 0.40
        pc = Poly3DCollection(polys, alpha=alpha)
        pc.set_facecolor(part.color)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    xs, ys, zs = zip(*all_pts)
    if zoom == "nose":
        cx, cy, cz = nose_c
        r = 90
    else:
        cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0
        r = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

# ---- 视图3：槽内截面（z=0 切片，上层边局部）----
ax = fig.add_subplot(1, 3, 3)
slab = b3d.Pos(0, 0, 0) * b3d.Box(4000, 4000, 0.4)
for part in parts:
    piece = part.solid & slab
    if piece is None or getattr(piece, "wrapped", None) is None:
        continue
    try:
        verts, tris = piece.tessellate(0.3)
    except Exception:
        continue
    pts = [(v.X, v.Y) for v in verts]
    for tri in tris:
        poly = [pts[i] for i in tri]
        ax.fill(*zip(*poly), color=part.color, lw=0)
ax.set_aspect("equal")
# 聚焦上层边截面
import math  # noqa: E402

th1 = -res.fai1
cxy = (math.sin(th1) * (res.rr1 + res.hc / 2), math.cos(th1) * (res.rr1 + res.hc / 2))
half = max(res.h_slot, res.w_slot) * 0.75
ax.set_xlim(cxy[0] - half, cxy[0] + half)
ax.set_ylim(cxy[1] - half, cxy[1] + half)
ax.set_title("槽内截面 z=0（上层边）")

out = Path(__file__).resolve().parents[1] / "output" / f"render_detail_{variant}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")
