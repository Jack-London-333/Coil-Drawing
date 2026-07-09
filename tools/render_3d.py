"""把 3D 线圈模型渲染成 PNG 供检查（matplotlib 三角面片渲染）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from coildrawing.engine import CoilInput, compute  # noqa: E402
from coildrawing.model3d import build_coil_parts  # noqa: E402


def main() -> int:
    inp = CoilInput(d2=921.5)
    res = compute(inp)
    parts = build_coil_parts(res)

    fig = plt.figure(figsize=(16, 8))
    views = [(20, -60, "iso", None), (90, -90, "top(axial)", None),
             (15, -35, "end zoom", "end")]
    for iax, (elev, azim, title, zoom) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, iax, projection="3d")
        all_pts = []
        for part in parts:
            verts, tris = part.solid.tessellate(1.0)
            pts = [(v.X, v.Y, v.Z) for v in verts]
            all_pts.extend(pts)
            polys = [[pts[i] for i in tri] for tri in tris]
            pc = Poly3DCollection(polys, alpha=0.95 if part.name.startswith("铜") else 0.45)
            pc.set_facecolor(part.color)
            pc.set_edgecolor("none")
            ax.add_collection3d(pc)
        xs, ys, zs = zip(*all_pts)
        if zoom == "end":
            zmax = max(zs)
            cx, cy, cz = 0.0, (max(ys) + min(ys)) / 2, zmax - 60
            r = 160
        else:
            cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
            r = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    out = Path(__file__).resolve().parent.parent / "output" / "render_check.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
