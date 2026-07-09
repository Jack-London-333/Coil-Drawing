"""3D 建模冒烟测试：用专利算例参数生成 STEP。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coildrawing.engine import CoilInput, compute  # noqa: E402


def main() -> int:
    t0 = time.time()
    inp = CoilInput(d2=921.5)
    res = compute(inp)
    print(f"engine ok: CC={res.cc:.1f} S1={res.s1:.1f} LLM={res.llm:.1f}")

    from coildrawing.model3d import export_step  # noqa: E402
    out = Path(__file__).resolve().parent.parent / "output" / "smoke_coil.step"
    out.parent.mkdir(exist_ok=True)
    names = export_step(res, str(out))
    print(f"exported: {out}")
    print(f"parts: {names}")
    print(f"size: {out.stat().st_size / 1024:.0f} KiB, took {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
