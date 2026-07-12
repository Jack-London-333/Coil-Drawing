"""交流电机定子成型线圈参数计算引擎。

公式体系来自专利 CN104965948B《一种交流电机定子绕组线圈参数的计算方法》
（湘潭电机股份有限公司，2015）。计算流程：

    步骤1  计算导线绝缘后尺寸
    步骤2  计算线圈槽内/端部截面尺寸，验证槽内剩余空间
    步骤3  建立线圈空间几何模型（RR1/RR2/fai/E1/E2/t1/t2）
    步骤4  判定线圈上下层边是否需要弯弧（D 与 RR1 比较），确定 AA1/AA2
    步骤5  迭代计算端部轴向投影长 CC、端部斜边弧长 S1/S2、直线部弯弧中心距
           X1/X2、平均匝长 LLM
    步骤6  计算线圈梯形梭形参数（绕线模大样）

单位约定：长度 mm，角度 rad。

与专利原文的两处有意识偏差（专利算例本身与其权利要求公式不一致，
以算例/物理含义为准）：
  * gf/gi（上下层线圈边中心半径）取 RR1+HC/2、RR2+HC/2，
    与算例 gf=481.45、gi=515.95 精确吻合；权利要求文本写作 RR1+hh1，
    与其自身算例矛盾。
  * 对地绝缘 T2/T4、防晕 CS、匝间 T1/T3 均按"单边厚度"输入，
    公式按权利要求的 2T 计入两侧。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


def fmt_num(v: float, nd: int = 8) -> str:
    """数值 → 文本：保留 nd 位小数并去尾零（至少保留 1 位整数）。"""
    txt = f"{v:.{nd}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt if txt not in ("", "-") else "0"


@dataclass
class WireSpec:
    """一种线规（成型线圈用扁铜线）。

    b: 裸线沿槽宽方向的尺寸 mm
    h: 裸线沿槽深方向的尺寸 mm
    t0: 导线自身绝缘(漆膜/薄膜)单边厚度 mm
    npd: 并绕根数（沿槽宽方向并列）
    ncd: 每匝线圈内沿槽深方向的层数
    """

    b: float = 0.0
    h: float = 0.0
    t0: float = 0.0
    npd: int = 0
    ncd: int = 0

    @property
    def bi(self) -> float:
        """绝缘后沿槽宽方向尺寸（专利 WIB）"""
        return self.b + 2 * self.t0 if self.npd > 0 else 0.0

    @property
    def hi(self) -> float:
        """绝缘后沿槽深方向尺寸（专利 WIT）"""
        return self.h + 2 * self.t0 if self.ncd > 0 else 0.0


@dataclass
class InsulationLayer:
    """线圈对地包扎的一层绝缘（用于三维模型分层显示）。"""

    name: str
    thickness: float  # 单边厚度 mm


@dataclass
class CoilInput:
    """计算输入。默认值取自专利 CN104965948B 具体实施方式 [0136]。"""

    # --- 铁芯与槽 ---
    d2: float = 1180.0       # 定子铁芯内径 mm
    lc: float = 1250.0       # 铁芯轴向长度 mm
    ns: int = 108            # 定子槽数
    poles: int = 12          # 极数（仅参考，不参与专利公式）
    taw: int = 9             # 线圈节距（槽数）
    hs: float = 74.0         # 槽深 mm
    ws: float = 11.5         # 槽宽 mm
    hsd: float = 4.0         # 槽楔厚度 mm
    wihu: float = 1.0        # 槽楔下绝缘垫片厚度 mm
    wihm: float = 3.0        # 槽内上下层线圈间垫片厚度 mm
    wihb: float = 1.0        # 槽底绝缘垫片厚度 mm

    # --- 绕组 ---
    n_turns: int = 8         # 线圈匝数 N
    wire1: WireSpec = field(default_factory=lambda: WireSpec(b=8.2, h=3.35, t0=0.0, npd=1, ncd=1))
    wire2: WireSpec = field(default_factory=WireSpec)

    # --- 绝缘（单边厚度） ---
    t1: float = 0.15         # 槽内匝间绝缘 mm
    t3: float = 0.15         # 端部匝间绝缘 mm
    t2: float = 1.1          # 槽内对地绝缘 mm
    t4: float = 1.1          # 端部对地绝缘 mm
    cs: float = 0.0          # 槽内防晕层 mm

    # --- 端部结构 ---
    ld: float = 20.0         # 铁芯端部齿压板轴向长度 mm
    le: float = 20.0         # 线圈直线部分伸出铁芯长度 mm
    f_nose: float = 20.0     # 线圈端部鼻端抬高 mm
    seita3: float = 0.349    # 鼻端中心线与过鼻端弯弧中心点直径的夹角 rad
    rd_nose: float = 15.0    # 线圈端部鼻端半径 RD mm
    rd1_conn: float = 15.0   # 接线侧弯弧半径 RD1 mm（用于 E1）
    rd2_nonconn: float = 15.0  # 非接线侧弯弧半径 RD2 mm（用于 E2）
    r_bend_slot: float = 30.0  # 直线部与端部斜边连接处弯曲半径 rd1 mm
    r_bend_nose: float = 30.0  # 端部斜边与鼻端连接处弯曲半径 rd2 mm
    ba: float = 7.0          # 线圈端部间隙给定值 mm
    ysc: float = 45.0        # 引线长 mm
    xi: float = 1e-9         # 端部迭代误差设定值 ξ

    # --- 三维模型的对地绝缘分层（单边厚度，供 3D/大样使用） ---
    layers: list[InsulationLayer] = field(default_factory=lambda: [
        InsulationLayer("对地云母带 1", 0.55),
        InsulationLayer("对地云母带 2", 0.55),
    ])

    # --- 三维逐匝精细建模 ---
    # 匝绝缘分层（包在每匝导线束外，由内到外，单边厚度，总和应≈T1）
    turn_layers: list[InsulationLayer] = field(default_factory=lambda: [
        InsulationLayer("匝间云母带", 0.15),
    ])
    lead_bend_r: float = 15.0     # 引线折弯半径 mm（引线伸出长度用 ysc）
    lead_bare: float = 30.0       # 引线端头裸铜长度 mm（0=不留）
    lead_end_positive_z: bool = True  # 出线端：True=轴向 +Z，False=轴向 -Z
    corona_on: bool = False       # 是否绘制槽部防晕层（厚度=CS）
    corona_overhang: float = 50.0  # 防晕层每端伸出铁芯长度 mm（沿导线，可越过弯角沿端臂延伸）
    detail_3d: bool = True        # 导出 STEP 时使用逐匝精细模型
    # --- 槽内固定件（垫片/槽楔）是否加入三维模型 ---
    draw_wedge: bool = False      # 槽楔 HSD
    draw_wihu: bool = False       # 槽楔下垫片 WIHU
    draw_wihm: bool = False       # 层间垫片 WIHM
    draw_wihb: bool = False       # 槽底垫片 WIHB


@dataclass
class CoilResult:
    """计算结果（字段名尽量沿用专利符号）。"""

    inp: CoilInput = None  # type: ignore[assignment]

    # 步骤1
    wit1: float = 0.0
    wib1: float = 0.0
    wit2: float = 0.0
    wib2: float = 0.0

    # 步骤2
    wa_turn: float = 0.0   # WA  每匝包匝间绝缘后宽度
    had: float = 0.0       # HAD 每匝包匝间绝缘后厚度(高度)
    w_slot: float = 0.0    # W   槽内截面宽度
    h_slot: float = 0.0    # H   槽内截面高度
    hbd: float = 0.0       # HBD 端部每匝高度
    wb_turn: float = 0.0   # WB  端部每匝宽度
    hd: float = 0.0        # HD  端部截面高度
    wd: float = 0.0        # WD  端部截面宽度
    wc: float = 0.0        # WC  槽内去除对地/防晕后宽度
    hc: float = 0.0        # HC  槽内去除对地/防晕后高度
    wa_margin: float = 0.0  # Wa 槽宽方向余量
    ha_margin: float = 0.0  # Ha 槽深方向余量

    # 步骤3
    hh1: float = 0.0
    hh2: float = 0.0
    rr1: float = 0.0       # RR1 上层边底部中点半径
    rr2: float = 0.0       # RR2 下层边底部中点半径
    fai: float = 0.0       # 上下层边间张角
    fai1: float = 0.0
    fai2: float = 0.0
    e1: float = 0.0        # 接线侧鼻端轴向投影长
    e2: float = 0.0        # 非接线侧鼻端轴向投影长
    t_pitch1: float = 0.0  # t1 上层边每槽弧长(齿距)
    t_pitch2: float = 0.0  # t2 下层边每槽弧长

    # 步骤4
    gf: float = 0.0        # 上层边中心半径
    gi: float = 0.0        # 下层边中心半径
    ls: float = 0.0        # 上层端部投影弧长
    lu: float = 0.0        # 下层端部投影弧长
    xe: float = 0.0
    ye: float = 0.0
    xk: float = 0.0
    yk: float = 0.0
    d_min: float = 0.0     # 电机中心至直线段 ek 最短距离
    need_bend: bool = False
    rr_bend1: float = 0.0  # rr1 上层边端部斜边弯弧半径
    rr_bend2: float = 0.0  # rr2 下层边端部斜边弯弧半径
    aa1: float = 0.0       # AA1 上层边弦长或弧长
    aa2: float = 0.0       # AA2 下层边弦长或弧长

    # 步骤5
    bd: float = 0.0
    seita_d: float = 0.0
    cc: float = 0.0        # 端部轴向投影长
    seita1: float = 0.0    # 上层边与铁芯端面夹角
    seita2: float = 0.0    # 下层边与铁芯端面夹角
    b1: float = 0.0
    b2: float = 0.0
    ba1: float = 0.0       # 上层边端部实际间隙
    ba2: float = 0.0       # 下层边端部实际间隙
    iterations: int = 0
    s1: float = 0.0        # 上层端部斜边弧长
    s2: float = 0.0        # 下层端部斜边弧长
    l2: float = 0.0        # 上层直线部中心线长
    l3: float = 0.0        # 下层直线部中心线长
    x1: float = 0.0        # 上层直线部两端弯弧中心距
    x2: float = 0.0        # 下层直线部两端弯弧中心距
    k1: float = 0.0        # 弯弧系数 K1
    k2: float = 0.0        # 弯弧系数 K2
    llm: float = 0.0       # 线圈平均匝长
    wire_total: float = 0.0  # 单个线圈用线长 ≈ LLM*N + 2*ysc

    # 步骤6 梯形梭形
    xx1: float = 0.0       # 梭形斜边长
    l4: float = 0.0        # 梭形上底长
    l5: float = 0.0        # 梭形下底长
    rd1_lozenge: float = 0.0  # 梭形斜边与下底处弯弧半径
    h_lozenge: float = 0.0    # 梭形(梯形)高度
    lm1: float = 0.0          # 梭长

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def rows(self) -> list[tuple[str, str, str, str]]:
        """生成 (符号, 名称, 数值, 单位) 结果表，供 UI 和导出使用。

        数值显示到小数点后 8 位并去掉无意义的尾零（内部计算为
        全精度 double，导出 Excel/CSV 时亦写出该 8 位小数文本）。
        """
        deg = 180.0 / math.pi
        r: list[tuple[str, str, str, str]] = []

        def add(sym: str, name: str, val, unit: str = "mm", nd: int = 8):
            if isinstance(val, bool):
                txt = "是" if val else "否"
            elif isinstance(val, int):
                txt = str(val)
            else:
                txt = fmt_num(val, nd)
            r.append((sym, name, txt, unit))

        r.append(("—", "— 步骤1 导线绝缘后尺寸 —", "", ""))
        add("WIT1", "导线1绝缘后厚度(槽深向)", self.wit1)
        add("WIB1", "导线1绝缘后宽度(槽宽向)", self.wib1)
        if self.inp.wire2.npd > 0 or self.inp.wire2.ncd > 0:
            add("WIT2", "导线2绝缘后厚度(槽深向)", self.wit2)
            add("WIB2", "导线2绝缘后宽度(槽宽向)", self.wib2)

        r.append(("—", "— 步骤2 截面尺寸与槽内空间 —", "", ""))
        add("WA", "每匝包匝间绝缘后宽度", self.wa_turn)
        add("HAD", "每匝包匝间绝缘后厚度", self.had)
        add("W", "线圈槽内截面宽度", self.w_slot)
        add("H", "线圈槽内截面高度", self.h_slot)
        add("HBD", "端部每匝高度", self.hbd)
        add("WB", "端部每匝宽度", self.wb_turn)
        add("HD", "线圈端部截面高度", self.hd)
        add("WD", "线圈端部截面宽度", self.wd)
        add("WC", "槽内裸组截面宽度", self.wc)
        add("HC", "槽内裸组截面高度", self.hc)
        add("Wa", "槽宽方向余量", self.wa_margin)
        add("Ha", "槽深方向余量", self.ha_margin)

        r.append(("—", "— 步骤3 空间几何模型 —", "", ""))
        add("hh1", "槽口处绝缘厚度", self.hh1)
        add("hh2", "槽内上下层线圈间绝缘厚度", self.hh2)
        add("RR1", "上层边底部中点半径", self.rr1)
        add("RR2", "下层边底部中点半径", self.rr2)
        add("fai", "上下层边间张角", self.fai, "rad")
        add("fai°", "上下层边间张角", self.fai * deg, "°")
        add("E1", "接线侧鼻端轴向投影长", self.e1)
        add("E2", "非接线侧鼻端轴向投影长", self.e2)
        add("t1", "上层边处齿距", self.t_pitch1)
        add("t2", "下层边处齿距", self.t_pitch2)

        r.append(("—", "— 步骤4 弯弧判定 —", "", ""))
        add("gf", "上层边中心半径", self.gf)
        add("gi", "下层边中心半径", self.gi)
        add("D", "中心至 ek 线最短距离", self.d_min)
        add("弯弧", "上下层边是否需要弯弧", self.need_bend, "")
        if self.need_bend:
            add("rr1", "上层边端部斜边弯弧半径", self.rr_bend1)
            add("rr2", "下层边端部斜边弯弧半径", self.rr_bend2)
        add("AA1", "上层边投影长(弦/弧)", self.aa1)
        add("AA2", "下层边投影长(弦/弧)", self.aa2)

        r.append(("—", "— 步骤5 端部与匝长 —", "", ""))
        add("CC", "端部轴向投影长", self.cc)
        add("seita1", "上层边与铁芯端面夹角", self.seita1, "rad")
        add("seita1°", "上层边与铁芯端面夹角", self.seita1 * deg, "°")
        add("seita2", "下层边与铁芯端面夹角", self.seita2, "rad")
        add("seita2°", "下层边与铁芯端面夹角", self.seita2 * deg, "°")
        add("Ba1", "上层边端部实际间隙", self.ba1)
        add("Ba2", "下层边端部实际间隙", self.ba2)
        add("S1", "上层端部斜边弧长", self.s1)
        add("S2", "下层端部斜边弧长", self.s2)
        add("L2", "上层直线部中心线长", self.l2)
        add("L3", "下层直线部中心线长", self.l3)
        add("X1", "上层直线部弯弧中心距", self.x1)
        add("X2", "下层直线部弯弧中心距", self.x2)
        add("K1", "弯弧系数 K1", self.k1)
        add("K2", "弯弧系数 K2", self.k2)
        add("LLM", "线圈平均匝长", self.llm)
        add("L总", "单线圈用线长(含引线)", self.wire_total)

        r.append(("—", "— 步骤6 梯形梭形(绕线模) —", "", ""))
        add("XX1", "梭形斜边长度", self.xx1)
        add("L4", "梭形上底长", self.l4)
        add("L5", "梭形下底长", self.l5)
        add("RD1", "梭形斜边与下底处弯弧半径", self.rd1_lozenge)
        add("h_", "梯形梭形高度", self.h_lozenge)
        add("Lm1", "梭长", self.lm1)
        return r


# ----------------------------------------------------------------------
def compute(inp: CoilInput, aa_override: tuple[float, float] | None = None) -> CoilResult:
    """执行专利步骤 1-6 的完整计算。

    aa_override: 直接指定 (AA1, AA2)，仅用于验证专利算例。
    """
    res = CoilResult(inp=replace(inp))
    w1, w2 = inp.wire1, inp.wire2

    # ---------- 步骤1 导线绝缘后尺寸 ----------
    res.wit1, res.wib1 = w1.hi, w1.bi
    res.wit2, res.wib2 = w2.hi, w2.bi

    # ---------- 步骤2 截面尺寸 ----------
    res.wa_turn = max(res.wib1 * w1.npd, res.wib2 * w2.npd) + 2 * inp.t1
    res.had = res.wit1 * w1.ncd + res.wit2 * w2.ncd + 2 * inp.t1
    n = inp.n_turns
    res.w_slot = res.wa_turn + 2 * inp.t2 + 2 * inp.cs
    res.h_slot = res.had * n + 2 * inp.t2 + 2 * inp.cs
    res.hbd = res.had - 2 * inp.t1 + 2 * inp.t3
    res.wb_turn = res.wa_turn - 2 * inp.t1 + 2 * inp.t3
    res.hd = res.hbd * n + 2 * inp.t4
    res.wd = res.wb_turn + 2 * inp.t4
    res.wc = res.wa_turn - 2 * inp.t1
    res.hc = res.had * n - 2 * inp.t1
    res.wa_margin = inp.ws - res.w_slot
    res.ha_margin = (inp.hs - 2 * res.h_slot - inp.hsd
                     - inp.wihu - inp.wihm - inp.wihb)
    if res.wa_margin < 0:
        res.warnings.append(
            f"槽宽方向余量 Wa={res.wa_margin:.2f}mm < 0，线圈放不进槽，请检查线规/绝缘/槽宽")
    if res.ha_margin < 0:
        res.warnings.append(
            f"槽深方向余量 Ha={res.ha_margin:.2f}mm < 0，线圈放不进槽，请检查匝数/线规/槽深")

    # ---------- 步骤3 空间几何模型 ----------
    res.hh1 = inp.wihu + inp.t2 + inp.t1 + inp.cs
    res.hh2 = inp.wihm + 2 * (inp.t2 + inp.t1 + inp.cs)
    res.rr1 = inp.d2 / 2 + inp.hsd + res.hh1
    res.rr2 = res.rr1 + res.hc + res.hh2
    res.fai = 2 * math.pi * (inp.taw - 1) / inp.ns
    res.fai1 = res.fai2 = res.fai / 2
    hb = res.hc  # HB: 槽内去除对地/防晕后截面高度
    res.e1 = inp.rd1_conn + hb - res.hbd - 2 * inp.t3
    res.e2 = inp.rd2_nonconn + hb - 2 * inp.t3
    res.t_pitch1 = 2 * math.pi * res.rr1 / inp.ns
    res.t_pitch2 = 2 * math.pi * res.rr2 / inp.ns

    # ---------- 步骤4 弯弧判定 ----------
    # gf/gi 取线圈边中心半径（与专利算例一致；权利要求文本 RR1+hh1 与算例矛盾）
    res.gf = res.rr1 + res.hc / 2
    res.gi = res.rr2 + res.hc / 2
    res.ls = res.fai1 * res.gf
    res.lu = res.fai2 * res.gi

    # e: 上层边底部中点；k: 上层边端部斜边与鼻端连接处底部中点
    res.xe = res.rr1 * math.sin(res.fai1)
    res.ye = res.rr1 * math.cos(res.fai1)
    rk = res.rr1 + inp.f_nose + res.hc * math.sin(inp.seita3)
    # k 点周向偏离线圈中心线约一个鼻端半径（与专利算例 Xk=15=RD 一致）
    res.xk = min(inp.rd_nose, 0.9 * rk)
    res.yk = math.sqrt(rk * rk - res.xk * res.xk)
    a_ln = (res.yk - res.ye) / (res.xk - res.xe)
    c_ln = res.ye - (res.yk - res.ye) * res.xe / (res.xk - res.xe)
    res.d_min = abs(c_ln) / math.sqrt(a_ln * a_ln + 1.0)
    res.need_bend = res.d_min <= res.rr1

    if res.need_bend:
        aa1, aa2 = res.ls, res.lu           # 弧长代入
    else:
        chord1 = math.hypot(res.xk - res.xe, res.yk - res.ye)
        xe2 = res.rr2 * math.sin(res.fai2)
        ye2 = res.rr2 * math.cos(res.fai2)
        rk2 = res.rr2 + inp.f_nose + res.hc * math.sin(inp.seita3)
        yk2 = math.sqrt(rk2 * rk2 - res.xk * res.xk)
        chord2 = math.hypot(res.xk - xe2, yk2 - ye2)
        aa1, aa2 = chord1, chord2           # 弦长代入
    if aa_override is not None:
        aa1, aa2 = aa_override
    res.aa1, res.aa2 = aa1, aa2

    # ---------- 步骤5 端部迭代 ----------
    res.bd = res.wd + inp.ba
    if res.bd >= res.t_pitch1:
        raise ValueError(
            f"端部中心距 BD={res.bd:.2f}mm ≥ 上层齿距 t1={res.t_pitch1:.2f}mm，"
            "端部间隙无法满足，请减小线圈宽度或端部间隙")
    res.seita_d = math.asin(res.bd / res.t_pitch1)
    cc = aa1 * math.tan(res.seita_d)
    b1 = b2 = ba1 = ba2 = 0.0
    seita1 = seita2 = 0.0
    for it in range(1, 101):
        seita1 = math.atan(cc / aa1)
        seita2 = math.atan(cc / aa2)
        b1 = res.t_pitch1 * math.sin(seita1)
        b2 = res.t_pitch2 * math.sin(seita2)
        ba1 = b1 - res.wd
        ba2 = b2 - res.wd
        res.iterations = it
        if abs(ba1 - inp.ba) / inp.ba < inp.xi:
            break
        cc = aa1 * math.tan(math.asin((res.wd + inp.ba) / res.t_pitch1))
    res.cc = cc
    res.seita1, res.seita2 = seita1, seita2
    res.b1, res.b2 = b1, b2
    res.ba1, res.ba2 = ba1, ba2
    if res.ba2 < inp.ba:
        res.warnings.append(
            f"下层边端部实际间隙 Ba2={res.ba2:.2f}mm 小于给定值 Ba={inp.ba:.2f}mm"
            "（专利算法仅按上层边收敛，此为正常现象，请确认工艺可接受）")

    # 弯弧半径（需 CC，故在此处计算）
    if res.need_bend:
        s_f1, c_f1 = math.sin(res.fai1), math.cos(res.fai1)
        num1 = (res.rr1 * s_f1) ** 2 + cc ** 2 + (res.rr1 - res.rr1 * c_f1) ** 2
        res.rr_bend1 = num1 / (2 * (res.rr1 - res.rr1 * c_f1))
        s_f2, c_f2 = math.sin(res.fai2), math.cos(res.fai2)
        num2 = (res.rr2 * s_f2) ** 2 + cc ** 2 + (res.rr2 - res.rr2 * c_f2) ** 2
        res.rr_bend2 = num2 / (2 * (res.rr2 - res.rr2 * c_f2))

    # 斜边弧长 / 直线部 / 匝长
    res.s1 = math.hypot(aa1, cc)
    res.s2 = math.hypot(aa2, cc)
    res.l2 = res.l3 = inp.lc + 2 * inp.ld + 2 * inp.le
    seita5, seita4 = res.seita1, res.seita2  # 专利中 seita5/seita4 即上/下层边夹角
    res.x1 = res.l2 - 2 * math.tan(math.pi / 4 - seita5 / 2) * (inp.r_bend_slot + res.wc / 2)
    res.x2 = res.l3 - 2 * math.tan(math.pi / 4 - seita4 / 2) * (inp.r_bend_nose + res.wc / 2)
    rsum = (inp.r_bend_slot + inp.r_bend_nose + res.wa_turn) / 2
    res.k1 = rsum * (4 * (math.pi / 2 - seita5) - 4 * math.tan(math.pi / 4 - res.seita1 / 2))
    res.k2 = rsum * (4 * (math.pi / 2 - seita4) - 4 * math.tan(math.pi / 4 - res.seita2 / 2))
    res.llm = (2 * res.s1 + 2 * res.s2 + res.x1 + res.x2 + res.k1 + res.k2
               + 2 * math.pi * (inp.rd_nose + res.wa_turn / 2))
    res.wire_total = res.llm * n + 2 * inp.ysc

    # ---------- 步骤6 梯形梭形 ----------
    res.xx1 = res.s1
    res.l4 = res.l2
    res.l5 = res.l3 + 2 * res.s2
    res.rd1_lozenge = inp.rd_nose
    h2 = res.s2 ** 2 + res.rd1_lozenge ** 2 - res.xx1 ** 2
    if h2 >= 0:
        res.h_lozenge = math.sqrt(h2)
    else:
        res.h_lozenge = 0.0
        res.warnings.append("梯形梭形高度 h_ 计算出现负平方（S2²+RD1²<XX1²），请核对端部参数")
    res.lm1 = (res.l2 + res.l3 + 2 * res.s1 + 2 * res.s2
               + math.pi * (inp.rd_nose + res.hd / 2))

    # ---------- 三维绝缘分层与对地绝缘厚度一致性检查 ----------
    layer_sum = sum(l.thickness for l in inp.layers if l.thickness > 0)
    if inp.layers and abs(layer_sum - inp.t2) > 0.051:
        res.warnings.append(
            f"三维绝缘分层总厚 {layer_sum:.2f}mm 与槽内对地绝缘 T2={inp.t2:.2f}mm 不一致，"
            "三维模型外形将与计算截面不符（如刻意为之可忽略）")

    turn_sum = sum(l.thickness for l in inp.turn_layers if l.thickness > 0)
    if inp.turn_layers and abs(turn_sum - inp.t1) > 0.051:
        res.warnings.append(
            f"匝绝缘分层总厚 {turn_sum:.2f}mm 与槽内匝间绝缘 T1={inp.t1:.2f}mm 不一致，"
            "逐匝三维模型匝廓将与计算截面不符（如刻意为之可忽略）")
    if inp.corona_on:
        if inp.cs <= 0:
            res.warnings.append(
                "已勾选绘制防晕层但槽内防晕层 CS=0，三维模型中防晕层无厚度不会生成；"
                "请在“绝缘”组中填入 CS（典型 0.2~0.5mm）")
        # 防晕层沿导线延伸的可用长度 ≈ 直线段伸出 + 槽口弯角弧长 + 斜边长
        arc1 = inp.r_bend_slot * (math.pi / 2 - res.seita1)
        reach = inp.ld + inp.le + arc1 + 0.9 * min(res.s1, res.s2)
        if inp.corona_overhang > reach:
            res.warnings.append(
                f"防晕层每端伸出 {fmt_num(inp.corona_overhang, 1)}mm 超过沿导线可延伸长度"
                f"（直线段+槽口弯角+斜边 ≈{fmt_num(reach, 1)}mm），三维模型中将截短")

    return res


def strand_grid(inp: CoilInput):
    """每匝内的股线排布（纯几何，供 2D 截面图与 3D 建模共用）。

    返回 (w_env, h_env, strands)。w_env/h_env 为一匝裸组（含股自身绝缘）
    的包络宽/高；strands 为字典列表：no/row/col/b/h/t0/bi/hi/x/y，
    x 沿槽宽（周向）、y 沿槽深（径向，向外为正），均相对匝中心。
    行序由内径侧到外径侧：导线1 各层在下，导线2 各层在上；
    较窄的行在宽度方向居中。
    """
    rows: list[tuple[int, WireSpec]] = []
    for no, w in ((1, inp.wire1), (2, inp.wire2)):
        if w.npd > 0 and w.ncd > 0 and w.b > 0 and w.h > 0:
            rows += [(no, w)] * w.ncd
    if not rows:
        raise ValueError("无有效线规（导线1/导线2 均为空），无法逐匝建模")
    h_env = sum(w.hi for _, w in rows)
    w_env = max(w.bi * w.npd for _, w in rows)
    strands = []
    y = -h_env / 2
    row_counter: dict[int, int] = {}
    for no, w in rows:
        i = row_counter.get(no, 0)
        row_counter[no] = i + 1
        yc = y + w.hi / 2
        for col in range(w.npd):
            xc = (col - (w.npd - 1) / 2) * w.bi
            strands.append(dict(no=no, row=i, col=col, b=w.b, h=w.h,
                                t0=w.t0, bi=w.bi, hi=w.hi, x=xc, y=yc))
        y += w.hi
    return w_env, h_env, strands


def patent_example_input() -> CoilInput:
    """专利 CN104965948B 具体实施方式 [0136] 的算例输入。

    注意：算例正文给出 D2=1180mm，但其 RR1=467mm 只能由 D2≈920.3mm 得到
    （专利算例内部不一致），此处按 RR1=467 反推 D2 以复现算例全链路。
    """
    return CoilInput()  # 默认值即算例值；D2 见 tests 中的说明
