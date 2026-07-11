"""config.txt 读写往返测试。"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coildrawing.config_io import (  # noqa: E402
    _unused_field_check, config_text, parse_config_text)
from coildrawing.engine import CoilInput, InsulationLayer, WireSpec  # noqa: E402


def test_fields_covered():
    _unused_field_check()


def test_roundtrip_default():
    inp = CoilInput()
    back = parse_config_text(config_text(inp))
    assert asdict(back) == asdict(inp)


def test_roundtrip_modified():
    inp = CoilInput(
        d2=921.50000001, lc=1250.12345678, ns=72, taw=7, n_turns=3,
        wire1=WireSpec(b=4.0, h=1.6, t0=0.1, npd=2, ncd=2),
        wire2=WireSpec(b=7.0, h=1.2, t0=0.05, npd=1, ncd=1),
        cs=0.3, xi=1e-9, corona_on=True, corona_overhang=120.0,
        draw_wedge=True, draw_wihm=True, detail_3d=False,
        layers=[InsulationLayer("云母带 A", 0.5),
                InsulationLayer("云母带 B", 0.6)],
        turn_layers=[InsulationLayer("匝间带", 0.15000001)],
    )
    back = parse_config_text(config_text(inp))
    assert asdict(back) == asdict(inp)


def test_hand_edit_tolerance():
    """手工编辑常见写法：布尔多写法、行内注释、缺失键取默认。"""
    txt = """
[铁芯与槽]
D2 = 1000.5 ; 内径
NS = 96

[三维模型]
防晕层 = true
画层间垫片 = 开

[匝绝缘分层]
层1 = 0.2
"""
    inp = parse_config_text(txt)
    assert inp.d2 == 1000.5
    assert inp.ns == 96
    assert inp.corona_on is True
    assert inp.draw_wihm is True
    assert inp.lc == CoilInput().lc          # 缺失 → 默认
    assert inp.turn_layers[0].thickness == 0.2

def test_bad_value_reports_key():
    try:
        parse_config_text("[铁芯与槽]\nD2 = abc\n")
    except ValueError as exc:
        assert "D2" in str(exc)
    else:
        raise AssertionError("应抛 ValueError")
