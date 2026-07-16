import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, compute
from coildrawing import model3d as m

for rb in (100.0, 150.0, 200.0, 300.0):
    try:
        res = compute(CoilInput(seita3=math.radians(80.0), r_bend_nose=rb))
        shift, drop = m._nose_pose(res)
        L = m._centerline_length(res, (shift, drop))
        print(f"rd2={rb}: OK shift={shift:.2f} drop={drop:.2f} "
              f"len-LLM={L - res.llm:.1e}")
    except ValueError as e:
        print(f"rd2={rb}: raises: {str(e)[:80]}")

# 旧测试2: seita3=120°+F=200 仍应不可行?
try:
    res = compute(CoilInput(f_nose=200.0, rd_nose=15.0, r_bend_nose=30.0,
                            seita3=math.radians(120.0)))
    print("120°:", m._nose_pose(res))
except ValueError as e:
    print(f"120°: raises: {str(e)[:80]}")
