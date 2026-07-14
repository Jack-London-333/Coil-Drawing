"""config.txt 读写往返测试。"""
import math
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
        cs=0.3, xi=1e-9, seita3=math.radians(37.5),
        corona_on=True, corona_overhang=120.0,
        draw_wedge=True, draw_wihm=True, detail_3d=False,
        lead_end_positive_z=False,
        layers=[InsulationLayer("云母带 A", 0.5),
                InsulationLayer("云母带 B", 0.6)],
        turn_layers=[InsulationLayer("匝间带", 0.15000001)],
    )
    back = parse_config_text(config_text(inp))
    assert asdict(back) == asdict(inp)


def test_seita3_new_config_uses_degrees():
    text = config_text(CoilInput())
    assert "seita3_deg" in text
    assert "seita3_deg      = 80" in text
    assert "鼻端中心线与径向直径夹角" in text
    assert "鼻端内弯半径" in text
    assert "\nseita3 " not in text
    assert math.isclose(parse_config_text(text).seita3,
                        math.radians(80.0), rel_tol=0, abs_tol=1e-15)


def test_seita3_legacy_radians_and_new_key_priority():
    legacy = parse_config_text("[端部结构]\nseita3 = 0.5\n")
    assert legacy.seita3 == 0.5

    both = parse_config_text(
        "[端部结构]\nseita3 = 0.5\nseita3_deg = 30\n")
    assert math.isclose(both.seita3, math.radians(30.0),
                        rel_tol=0, abs_tol=1e-15)


def test_bundled_template_includes_lead_end_choice():
    template = (Path(__file__).resolve().parents[1] /
                "docs" / "config_template.txt").read_text(encoding="utf-8")
    assert "出线端在正轴端" in template
    assert "鼻端中心线与径向直径夹角" in template
    assert "Rc=RD+WA/2" in template
    assert "Larm按LLM守恒自动反算" in template
    assert parse_config_text(template).lead_end_positive_z is True


def test_hand_edit_tolerance():
    """手工编辑常见写法：布尔多写法、行内注释、缺失键取默认。"""
    txt = """
[铁芯与槽]
D2 = 1000.5 ; 内径
NS = 96

[三维模型]
防晕层 = true
画层间垫片 = 开
出线端在正轴端 = 否

[匝绝缘分层]
层1 = 0.2
"""
    inp = parse_config_text(txt)
    assert inp.d2 == 1000.5
    assert inp.ns == 96
    assert inp.corona_on is True
    assert inp.draw_wihm is True
    assert inp.lead_end_positive_z is False
    assert inp.lc == CoilInput().lc          # 缺失 → 默认
    assert inp.turn_layers[0].thickness == 0.2

def test_bad_value_reports_key():
    try:
        parse_config_text("[铁芯与槽]\nD2 = abc\n")
    except ValueError as exc:
        assert "D2" in str(exc)
    else:
        raise AssertionError("应抛 ValueError")
