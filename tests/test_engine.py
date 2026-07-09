"""用专利 CN104965948B 具体实施方式 [0136]-[0237] 的算例验证计算引擎。

说明：专利算例正文存在多处自相矛盾（例如 D2=1180 与 RR1=467 不符、
Wa=0.2 与 WS-W=0.8 不符、fs=fai1·gf 与其打印值 107 不符、
LLM 打印值比权利要求公式少 π(RD+WA/2) 等）。本测试以权利要求公式为准：
  * 步骤1-3 的确定性量与算例完全一致的，按高精度断言；
  * 端部链路（CC/seita/S1/S2/X1/X2/K1/K2）通过 aa_override 注入算例
    反推出的 AA1=106.7 / AA2=123.0，按 ±1% 断言；
  * 算例自身矛盾的打印值（E1/hh1/LLM/Lm1 等）不做强断言，只验证公式自洽。
"""

import math

import pytest

from coildrawing.engine import CoilInput, WireSpec, compute

# 专利算例：RR1=467 ⇒ D2 = 2*(467 - HSD - hh1) = 2*(467-4-2.25) = 921.5
# （专利正文写 D2=1180，与其 RR1=467 矛盾，见模块 docstring）
PATENT_D2 = 921.5


def patent_input() -> CoilInput:
    return CoilInput(
        d2=PATENT_D2, lc=1250.0, ns=108, poles=12, taw=9,
        hs=74.0, ws=11.5, hsd=4.0, wihu=1.0, wihm=3.0, wihb=1.0,
        n_turns=8,
        wire1=WireSpec(b=8.2, h=3.35, t0=0.0, npd=1, ncd=1),
        wire2=WireSpec(),
        t1=0.15, t3=0.15, t2=1.1, t4=1.1, cs=0.0,
        ld=20.0, le=20.0, f_nose=20.0, seita3=0.349,
        rd_nose=15.0, rd1_conn=15.0, rd2_nonconn=15.0,
        r_bend_slot=30.0, r_bend_nose=30.0,
        ba=7.0, ysc=45.0, xi=0.01,
    )


def test_step1_step2_sections():
    r = compute(patent_input())
    assert r.wa_turn == pytest.approx(8.5)     # [0144]
    assert r.had == pytest.approx(3.65)        # [0146]
    assert r.w_slot == pytest.approx(10.7)     # [0148]
    assert r.h_slot == pytest.approx(31.4)     # [0149]
    assert r.hbd == pytest.approx(3.65)        # [0151]
    assert r.wb_turn == pytest.approx(8.5)     # [0152]
    assert r.hd == pytest.approx(31.4)         # [0154]
    assert r.wd == pytest.approx(10.7)         # [0155]
    assert r.wc == pytest.approx(8.2)          # [0157]
    assert r.hc == pytest.approx(28.9)         # [0158]


def test_step3_geometry():
    r = compute(patent_input())
    assert r.hh2 == pytest.approx(5.5)                    # [0166]
    assert r.rr1 == pytest.approx(467.0)                  # [0168]
    assert r.rr2 == pytest.approx(501.5, abs=0.15)        # [0170]
    assert r.fai == pytest.approx(0.4654, abs=2e-4)       # [0172]
    assert r.t_pitch1 == pytest.approx(27.17, abs=0.01)   # [0177]
    assert r.t_pitch2 == pytest.approx(29.18, abs=0.02)   # [0178]


def test_step4_bend_judgement():
    r = compute(patent_input())
    assert r.gf == pytest.approx(481.45, abs=0.01)        # [0181]
    assert r.gi == pytest.approx(515.95, abs=0.15)        # [0182]
    # 算例 D=463 < RR1=467，需弯弧 [0195]
    assert r.need_bend
    assert r.d_min < r.rr1
    assert r.d_min == pytest.approx(463.0, rel=0.02)


def test_step5_end_winding_with_patent_aa():
    """注入算例反推的 AA1/AA2，校验端部迭代链路（±1%）。"""
    r = compute(patent_input(), aa_override=(106.7, 123.0))
    assert r.cc == pytest.approx(92.0, rel=0.01)          # [0211]
    assert r.seita1 == pytest.approx(0.7112, rel=0.01)    # [0211]
    assert r.seita2 == pytest.approx(0.6422, rel=0.01)    # [0211]
    assert r.ba1 == pytest.approx(7.03, abs=0.1)          # [0211]
    assert r.s1 == pytest.approx(141.0, rel=0.01)         # [0213]
    assert r.s2 == pytest.approx(153.6, rel=0.01)         # [0214]
    assert r.l2 == pytest.approx(1330.0)                  # [0216]
    assert r.l3 == pytest.approx(1330.0)
    assert r.x1 == pytest.approx(1298.7, abs=0.5)         # [0218]
    assert r.x2 == pytest.approx(1295.8, abs=0.5)         # [0219]
    assert r.k1 == pytest.approx(54.97, abs=0.5)          # [0221]
    assert r.k2 == pytest.approx(58.6, abs=0.5)           # [0222]
    # LLM 按权利要求公式自洽（算例打印值 3358.2 比公式少 π(RD+WA/2)，不强断言）
    expected = (2 * r.s1 + 2 * r.s2 + r.x1 + r.x2 + r.k1 + r.k2
                + 2 * math.pi * (15.0 + r.wa_turn / 2))
    assert r.llm == pytest.approx(expected)


def test_step6_lozenge_with_patent_aa():
    r = compute(patent_input(), aa_override=(106.7, 123.0))
    assert r.xx1 == pytest.approx(r.s1)                   # [0227] XX1=S1=141
    assert r.l4 == pytest.approx(1330.0)                  # [0229] L4=L2
    assert r.l5 == pytest.approx(1636.0, rel=0.01)        # [0231] L5=L3+2*S2
    assert r.rd1_lozenge == pytest.approx(15.0)           # [0233]
    # h_ = sqrt(S2^2 + RD1^2 - XX1^2)：算例打印 79mm，但代入其自身数值
    # (153.6²+15²-141²)^0.5 = 62.75mm，打印值与公式矛盾，按公式自洽断言
    expected_h = math.sqrt(r.s2 ** 2 + r.rd1_lozenge ** 2 - r.xx1 ** 2)
    assert r.h_lozenge == pytest.approx(expected_h)
    # Lm1 公式自洽 [0237]
    expected = (r.l2 + r.l3 + 2 * r.s1 + 2 * r.s2
                + math.pi * (15.0 + r.hd / 2))
    assert r.lm1 == pytest.approx(expected)


def test_full_auto_run_no_override():
    """不注入 AA，全自动跑通并保持物理合理。"""
    r = compute(patent_input())
    assert r.iterations >= 1
    assert 60 < r.cc < 130
    assert 0.3 < r.seita1 < 1.2
    assert r.s1 > r.aa1
    # 一匝 = 上下层直线部(X1+X2) + 4 条端部斜边 + 弯弧修正 + 鼻端
    assert (r.l2 + r.l3) < r.llm < 2 * (r.l2 + r.l3)
    assert r.wire_total == pytest.approx(r.llm * 8 + 90.0)


def test_slot_fit_warnings():
    bad = patent_input()
    bad.n_turns = 12  # 塞不下
    r = compute(bad)
    assert any("Ha" in w or "槽深" in w for w in r.warnings)


def test_two_wire_specs():
    """两种线规并绕：取宽度大值、厚度叠加。"""
    inp = patent_input()
    inp.wire2 = WireSpec(b=7.0, h=2.0, t0=0.1, npd=2, ncd=1)
    r = compute(inp)
    assert r.wib2 == pytest.approx(7.2)
    assert r.wit2 == pytest.approx(2.2)
    assert r.wa_turn == pytest.approx(max(8.2, 7.2 * 2) + 0.3)
    assert r.had == pytest.approx(3.35 + 2.2 + 0.3)
