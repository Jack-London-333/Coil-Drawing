"""快速验证新浅螺旋鼻端:pose 反解、LLM 守恒、wire 长度、净距。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

for deg in (70.0, 80.0, 90.0):
    inp = CoilInput()
    inp.n_turns = 8
    inp.f_nose = 20.0
    inp.seita3 = math.radians(deg)
    inp.rd_nose = 15.0
    inp.r_bend_slot = 30.0
    inp.r_bend_nose = 30.0
    res = compute(inp)
    shift, drop = m._nose_pose(res)
    length = m._centerline_length(res, (shift, drop))
    nose = m._nose_layout(res)
    arc = nose.pos
    apex = (arc.ma - arc.c).normalized()
    ang = math.degrees(math.acos(max(-1, min(1, apex.dot(m.b3d.Vector(0, 1, 0))))))
    slack = m._nose_cross_fiber_slack(res, (shift, drop))
    print(f"seita3={deg:4.0f}°: shift={shift:8.3f} drop={drop:7.3f} "
          f"len-LLM={length - res.llm:.2e} apex∠径向={ang:7.3f}° "
          f"纤维裕量={slack:.3f}")
    wire, *_ = m.build_centerline(res)
    print(f"   wire.length-LLM = {wire.length - res.llm:.3e} mm")
