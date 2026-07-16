"""N=8 鼻端干涉复核(分段级,快速):

对 ±Z 两个鼻端分别检查:
  1. 交叉处 入口臂(斜边+rd2) × 出口臂(rd2+斜边) 跨匝对;
  2. 臂 × 卷环盘(全部匝的整环/换匝 MultiLoft);
  3. 卷环盘 匝与匝 之间;
铜-铜、铜-匝绝缘、匝绝缘-匝绝缘 三种材料组合,Common 体积 >1e-6 mm³ 记干涉。
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

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
w, h, _ = m._strand_grid(res)
wrap = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
outer_w = w + 2 * wrap
outer_h = min(h + 2 * wrap, res.had)


def kind_of(i):
    s = segs[i]
    if isinstance(s, m._MultiLoft):
        return "ring"
    return type(s).__name__


# 按 turn_ranges 归类各匝的分段角色。
# 环 = MultiLoft;入口臂 = 环之前的 Loft/Rev 串(斜边1→2+rd2角2 或
# 换匝后的斜边3→4);出口臂 = 环之后的 Loft/Rev 串。粗略按位置切分:
# 我们直接取每匝内所有 Loft/Rev 段,与所有 MultiLoft 段做跨匝检查,
# 覆盖臂×环;臂×臂 交叉对由(+Z 附近的段)互查覆盖。
turn_pieces = {}
for start, end, material_i in info["turn_ranges"]:
    ids = list(range(start, end))
    turn_pieces[material_i] = ids

nose_ctr = m._nose_layout(res)


def near_nose(seg, arc, r=120.0):
    """分段包围盒中心是否落在鼻端附近(粗筛)。"""
    if isinstance(seg, m._MultiLoft):
        o = seg.sections[len(seg.sections) // 2].o
    elif isinstance(seg, m._Loft):
        o = (seg.f0.o + seg.f1.o) * 0.5
    elif isinstance(seg, m._Rev):
        o = seg.fl.ma
    else:
        o = seg.f.o + seg.f.t * (seg.length / 2)
    return (o - arc.c).length < r


def common_volume(a, b):
    op = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
    op.SetRunParallel(True)
    op.Build()
    assert op.IsDone()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(op.Shape(), props)
    return props.Mass()


def overlaps(a, b):
    ba, bb = a.bounding_box(), b.bounding_box()
    return (ba.min.X < bb.max.X and bb.min.X < ba.max.X and
            ba.min.Y < bb.max.Y and bb.min.Y < ba.max.Y and
            ba.min.Z < bb.max.Z and bb.min.Z < ba.max.Z)


t0 = time.time()
for tag, arc in (("+Z", nose_ctr.pos), ("-Z", nose_ctr.neg)):
    # 收集鼻端附近分段(带匝号)
    items = []   # (turn, seg_idx, copper_solid, mica_ring)
    for turn, ids in turn_pieces.items():
        for i in ids:
            seg = segs[i]
            if isinstance(seg, m._Prism):
                continue
            if not near_nose(seg, arc):
                continue
            copper = m._seg_solid(seg, w, h)
            mica = m._seg_ring(seg, w, h, outer_w, outer_h)
            items.append((turn, i, copper, mica))
    print(f"[{tag}] 鼻端附近分段 {len(items)} 个 "
          f"({time.time()-t0:.0f}s)", flush=True)
    bad = []
    for a in range(len(items)):
        for b_ in range(a + 1, len(items)):
            ta, ia, ca, ma_ = items[a]
            tb, ib, cb, mb = items[b_]
            if ta == tb and abs(ia - ib) <= 1:
                continue   # 同匝相邻分段共享端面
            for la, sa in (("铜", ca), ("云母", ma_)):
                for lb, sb in (("铜", cb), ("云母", mb)):
                    if sa is None or sb is None:
                        continue
                    if not overlaps(sa, sb):
                        continue
                    if ta == tb and la == lb == "云母":
                        continue  # 同匝云母各分段本来连成一体
                    v = common_volume(sa, sb)
                    if v > 1e-6:
                        bad.append((v, ta, ia, la, tb, ib, lb))
    bad.sort(reverse=True)
    total = sum(x[0] for x in bad)
    for v, ta, ia, la, tb, ib, lb in bad[:15]:
        print(f"  {v:10.4f} mm3  匝{ta+1}[{ia}]{la} × 匝{tb+1}[{ib}]{lb} "
              f"({kind_of(ia)}×{kind_of(ib)})", flush=True)
    print(f"[{tag}] 干涉对 {len(bad)}, 总体积 {total:.4f} mm3 "
          f"({time.time()-t0:.0f}s)", flush=True)
