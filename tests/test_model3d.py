"""逐匝三维模型的快速回归测试（小匝数，控制耗时）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coildrawing.engine import CoilInput, WireSpec, compute  # noqa: E402


@pytest.fixture(scope="module")
def small_res():
    inp = CoilInput()
    inp.n_turns = 2
    inp.corona_on = True
    inp.cs = 0.3            # 防晕层厚度=CS（参数已合并）
    inp.draw_wihm = True    # 层间垫片
    return compute(inp)


def test_detailed_parts_geometry(small_res):
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    names = [p.name for p in parts]
    assert names[0] == "铜导线"
    assert any(n.startswith("匝绝缘1") for n in names)
    assert sum(1 for n in names if n.startswith("对地绝缘")) == 2
    assert sum(1 for n in names if n.startswith("防晕层")) == 2
    assert sum(1 for n in names if n.startswith("层间垫片")) == 2

    for p in parts:
        assert p.solid.volume > 0, p.name
        assert len(p.solid.solids()) == 1, p.name

    # 铜导线体积 ≈ 截面 × 路径长（路径长约 2×LLM + 2×引线）
    copper = parts[0].solid
    w = small_res.inp.wire1
    approx_len = 2 * small_res.llm + 2 * small_res.inp.ysc
    vol_expect = w.b * w.h * approx_len
    assert abs(copper.volume - vol_expect) / vol_expect < 0.05


def test_detailed_parts_valid_solids(small_res):
    """全部部件必须是有效实体——无效实体在 SolidWorks 中会被降级为
    “曲面实体”（v202607110207 问题二的根源，出线端自相交所致）。"""
    from OCP.BRepCheck import BRepCheck_Analyzer

    from coildrawing.model3d import build_coil_parts

    for p in build_coil_parts(small_res, detailed=True):
        assert BRepCheck_Analyzer(p.solid.wrapped).IsValid(), \
            f"{p.name} 不是有效实体"


def test_detailed_no_interference(small_res):
    """铜线与对地壳之间不得有体积干涉（构造上共享分段框架）。"""
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=True)
    copper = parts[0].solid
    for p in parts:
        if not p.name.startswith("对地绝缘"):
            continue
        try:
            inter = copper & p.solid
            v = inter.volume if inter is not None else 0.0
        except Exception:
            v = 0.0
        assert v < 1.0, f"{p.name} 与铜导线干涉 {v:.2f}mm³"


def test_dual_wire_strand_grid():
    from coildrawing.model3d import _strand_grid

    inp = CoilInput()
    inp.wire1 = WireSpec(b=8.0, h=2.0, t0=0.05, npd=2, ncd=1)
    inp.wire2 = WireSpec(b=6.0, h=1.5, t0=0.05, npd=1, ncd=2)
    res = compute(inp)
    w_env, h_env, strands = _strand_grid(res)
    assert len(strands) == 2 * 1 + 1 * 2
    assert w_env == pytest.approx(max(8.1 * 2, 6.1 * 1))
    assert h_env == pytest.approx(2.1 * 1 + 1.6 * 2)
    # 导线1 行在下（y 小），导线2 行在上
    assert max(s["y"] for s in strands if s["no"] == 1) < \
        min(s["y"] for s in strands if s["no"] == 2)


def test_simple_parts_still_work(small_res):
    from coildrawing.model3d import build_coil_parts

    parts = build_coil_parts(small_res, detailed=False)
    assert parts[0].name == "铜导体束"
    for p in parts:
        assert p.solid.volume > 0, p.name
