"""开发验证：逐匝精细模型 STEP 导出（专利算例 + 防晕层 + 变体）。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, WireSpec, compute  # noqa: E402

out = Path(__file__).resolve().parents[1] / "output" / "dev_detail"
out.mkdir(parents=True, exist_ok=True)

variant = sys.argv[1] if len(sys.argv) > 1 else "base"

inp = CoilInput()
inp.corona_on = True
if variant == "multi":
    # 并绕2根×2层 + 自身绝缘，检验多股/自身绝缘路径
    inp.wire1 = WireSpec(b=4.0, h=1.6, t0=0.1, npd=2, ncd=2)
    inp.n_turns = 4
elif variant == "dual":
    # 双线规混绕
    inp.wire1 = WireSpec(b=8.2, h=2.0, t0=0.05, npd=1, ncd=1)
    inp.wire2 = WireSpec(b=7.0, h=1.2, t0=0.05, npd=1, ncd=1)
    inp.n_turns = 6

res = compute(inp)
print(f"[{variant}] N={inp.n_turns} HAD={res.had:.3f} WD={res.wd:.2f} "
      f"warnings={len(res.warnings)}")
for w in res.warnings:
    print("  warn:", w)

from coildrawing.model3d import export_step  # noqa: E402

t0 = time.time()
path = out / f"detail_{variant}.step"
names = export_step(res, str(path), detailed=True)
dt = time.time() - t0
size_mb = path.stat().st_size / 1e6
print(f"OK {dt:.1f}s {size_mb:.1f}MB parts={len(names)}")
for n in names:
    print("  -", n)
