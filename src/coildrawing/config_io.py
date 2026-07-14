"""工作空间 config.txt 的读写。

格式为 INI 风格纯文本（UTF-8，带中文注释），用户既可在软件内修改
（自动同步写回），也可用任意文本编辑器直接编辑后由软件载入。

写出时始终按固定模板重新生成全文（含注释），保证文件规整；
读取时大小写敏感（RD1 与 rd1 是两个不同参数，均来自专利符号）。
"""

from __future__ import annotations

import configparser
import io
import math
from dataclasses import fields as dc_fields

from .engine import CoilInput, InsulationLayer, WireSpec

CONFIG_NAME = "config.txt"

# (INI键, 属性路径, 注释, 类型)；类型: f=float, i=int, b=bool,
# deg=界面/配置用角度、CoilInput 内部用弧度
_SECTIONS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    ("铁芯与槽", [
        ("D2",   "d2",    "定子铁芯内径 mm", "f"),
        ("LC",   "lc",    "铁芯轴向长度 mm", "f"),
        ("NS",   "ns",    "定子槽数", "i"),
        ("2P",   "poles", "极数", "i"),
        ("TAW",  "taw",   "线圈节距(槽数)", "i"),
        ("HS",   "hs",    "槽深 mm", "f"),
        ("WS",   "ws",    "槽宽 mm", "f"),
        ("HSD",  "hsd",   "槽楔厚度 mm", "f"),
        ("WIHU", "wihu",  "槽楔下垫片厚度 mm", "f"),
        ("WIHM", "wihm",  "层间垫片厚度 mm", "f"),
        ("WIHB", "wihb",  "槽底垫片厚度 mm", "f"),
    ]),
    ("绕组与线规", [
        ("N",    "n_turns",   "线圈匝数", "i"),
        ("WB1",  "wire1.b",   "导线1 裸线宽 mm", "f"),
        ("WT1",  "wire1.h",   "导线1 裸线厚 mm", "f"),
        ("T01",  "wire1.t0",  "导线1 自身绝缘单边厚 mm", "f"),
        ("NPD1", "wire1.npd", "导线1 并绕根数", "i"),
        ("NCD1", "wire1.ncd", "导线1 每匝层数", "i"),
        ("WB2",  "wire2.b",   "导线2 裸线宽 mm（无第二线规则为 0）", "f"),
        ("WT2",  "wire2.h",   "导线2 裸线厚 mm", "f"),
        ("T02",  "wire2.t0",  "导线2 自身绝缘单边厚 mm", "f"),
        ("NPD2", "wire2.npd", "导线2 并绕根数", "i"),
        ("NCD2", "wire2.ncd", "导线2 每匝层数", "i"),
    ]),
    ("绝缘", [
        ("T1", "t1", "槽内匝间绝缘单边厚 mm（当前 3D 要求 T1=T3）", "f"),
        ("T3", "t3", "端部匝间绝缘单边厚 mm（当前 3D 要求 T3=T1）", "f"),
        ("T2", "t2", "槽内对地绝缘单边厚 mm（当前 3D 要求 T2=T4）", "f"),
        ("T4", "t4", "端部对地绝缘单边厚 mm（当前 3D 要求 T4=T2）", "f"),
        ("CS", "cs", "槽内防晕层单边厚 mm（三维防晕层厚度同此值）", "f"),
    ]),
    ("端部结构", [
        ("LD",     "ld",           "齿压板轴向长度 mm", "f"),
        ("LE",     "le",           "直线部伸出铁芯长度 mm", "f"),
        ("F",      "f_nose",       "鼻端抬高 mm", "f"),
        ("seita3_deg", "seita3",
         "鼻端中心线与径向直径夹角 °（常用 70–90°）", "deg"),
        ("RD",     "rd_nose",
         "鼻端内弯半径 mm（Rc=RD+WA/2；Larm按LLM守恒自动反算）", "f"),
        ("RD1",    "rd1_conn",     "接线侧弯弧半径 mm", "f"),
        ("RD2",    "rd2_nonconn",  "非接线侧弯弧半径 mm", "f"),
        ("rd1",    "r_bend_slot",  "直线部-斜边弯曲半径 mm（注意与 RD1 区分大小写）", "f"),
        ("rd2",    "r_bend_nose",  "斜边-鼻端弯曲半径 mm（注意与 RD2 区分大小写）", "f"),
        ("Ba",     "ba",           "端部间隙给定值 mm", "f"),
        ("ysc",    "ysc",          "引线折弯后轴向自由直段长 mm", "f"),
        ("xi",     "xi",           "端部迭代误差设定值 ξ", "f"),
    ]),
    ("三维模型", [
        ("逐匝精细", "detail_3d",       "是=逐匝精细模型，否=简化束模型", "b"),
        ("引线折弯半径", "lead_bend_r",  "两段反向等角错位圆弧半径 mm", "f"),
        ("引线端头裸铜长", "lead_bare",  "mm，0=不留", "f"),
        ("出线端在正轴端", "lead_end_positive_z",
         "是=端部侧视图右端（轴向 +Z），否=左端（轴向 -Z）", "b"),
        ("防晕层", "corona_on",         "是否绘制槽部防晕层（厚度取 CS）", "b"),
        ("防晕层每端伸出", "corona_overhang",
         "mm，沿导线计量，可越过槽口弯角沿端臂延伸", "f"),
        ("画槽楔", "draw_wedge",        "是否将槽楔加入三维模型", "b"),
        ("画槽楔下垫片", "draw_wihu",   "是否将槽楔下垫片加入三维模型", "b"),
        ("画层间垫片", "draw_wihm",     "是否将层间垫片加入三维模型", "b"),
        ("画槽底垫片", "draw_wihb",     "是否将槽底垫片加入三维模型", "b"),
    ]),
]

_HEADER = """\
; ============================================================
;  CoilDrawing 工作空间配置文件（config.txt）
;  * 单位：长度 mm、角度 °；数值支持最多 8 位小数（可用科学计数法）
;    旧版键 seita3（弧度）仍可读取；新键 seita3_deg 优先
;  * 开关量填：是/否（也接受 true/false、1/0、开/关）
;  * 分号 ; 之后为注释；本文件由软件自动维护，手工修改保存后
;    软件会检测到变化并提示是否载入
; ============================================================
"""

_LAYER_HELP = """\
; 每行一层，格式：层N = 名称 | 单边厚度mm （由内到外）
"""


def _num(v: float) -> str:
    return format(v, ".15g")


def _disp_w(s: str) -> int:
    """显示宽度：中文等全角字符按 2 计。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def _pad_to(s: str, width: int) -> str:
    return s + " " * max(1, width - _disp_w(s))


def _get(inp: CoilInput, path: str):
    obj = inp
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(inp: CoilInput, path: str, value) -> None:
    parts = path.split(".")
    obj = inp
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def config_text(inp: CoilInput) -> str:
    """按固定模板生成 config.txt 全文。"""
    out = io.StringIO()
    out.write(_HEADER)
    for section, entries in _SECTIONS:
        out.write(f"\n[{section}]\n")
        for key, path, comment, typ in entries:
            v = _get(inp, path)
            if typ == "b":
                txt = "是" if v else "否"
            elif typ == "i":
                txt = str(int(v))
            elif typ == "deg":
                txt = _num(math.degrees(float(v)))
            else:
                txt = _num(float(v))
            out.write(f"{_pad_to(key, 16)}= {_pad_to(txt, 18)}; {comment}\n")
    for section, layers in (("对地绝缘分层", inp.layers),
                            ("匝绝缘分层", inp.turn_layers)):
        out.write(f"\n[{section}]\n")
        out.write(_LAYER_HELP)
        for i, layer in enumerate(layers, start=1):
            out.write(f"层{i} = {layer.name} | {_num(layer.thickness)}\n")
    return out.getvalue()


def _parse_bool(txt: str) -> bool:
    t = txt.strip().lower()
    if t in ("是", "true", "1", "开", "yes", "y", "on"):
        return True
    if t in ("否", "false", "0", "关", "no", "n", "off"):
        return False
    raise ValueError(f"无法识别的开关量: {txt!r}（请填 是/否）")


def _parse_layers(sec: "configparser.SectionProxy") -> list[InsulationLayer]:
    layers = []
    for key in sec:
        if not key.startswith("层"):
            continue
        try:
            idx = int(key[1:])
        except ValueError:
            continue
        raw = sec[key]
        if "|" in raw:
            name, _, t_txt = raw.rpartition("|")
        else:  # 容错：只给厚度
            name, t_txt = f"绝缘层 {idx}", raw
        layers.append((idx, InsulationLayer(name.strip() or f"绝缘层 {idx}",
                                            float(t_txt.strip()))))
    return [l for _, l in sorted(layers, key=lambda x: x[0])]


def parse_config_text(text: str) -> CoilInput:
    """解析 config.txt 文本 → CoilInput。缺失项用默认值补齐。

    解析错误抛 ValueError（含出错的节/键名，便于用户定位）。
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"),
                                   interpolation=None, strict=False)
    cp.optionxform = str  # 大小写敏感（RD1 与 rd1 不同）
    try:
        cp.read_string(text)
    except configparser.Error as exc:
        raise ValueError(f"配置文件格式错误：{exc}") from exc

    inp = CoilInput()
    for section, entries in _SECTIONS:
        if not cp.has_section(section):
            continue
        sec = cp[section]
        for key, path, _comment, typ in entries:
            if key not in sec:
                continue
            raw = sec[key].strip()
            try:
                if typ == "b":
                    val = _parse_bool(raw)
                elif typ == "i":
                    val = int(float(raw))
                elif typ == "deg":
                    val = math.radians(float(raw))
                else:
                    val = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"[{section}] {key} = {raw!r} 解析失败：{exc}") from exc
            _set(inp, path, val)

    # 向后兼容 v202607130557 及更早版本：旧键 seita3 的单位为弧度。
    # 同时出现时，新键 seita3_deg 已在上方解析，应当优先。
    nose_section = "端部结构"
    if (cp.has_section(nose_section)
            and "seita3_deg" not in cp[nose_section]
            and "seita3" in cp[nose_section]):
        raw = cp[nose_section]["seita3"].strip()
        try:
            inp.seita3 = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"[{nose_section}] seita3 = {raw!r} 解析失败：{exc}") from exc
    try:
        if cp.has_section("对地绝缘分层"):
            inp.layers = _parse_layers(cp["对地绝缘分层"])
        if cp.has_section("匝绝缘分层"):
            inp.turn_layers = _parse_layers(cp["匝绝缘分层"])
    except ValueError as exc:
        raise ValueError(f"绝缘分层解析失败：{exc}") from exc
    return inp


def save_config(path, inp: CoilInput) -> str:
    """写 config.txt，返回写出的文本。"""
    text = config_text(inp)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_config(path) -> CoilInput:
    with open(path, encoding="utf-8") as f:
        return parse_config_text(f.read())


def _unused_field_check() -> None:
    """开发自检：确保 _SECTIONS 覆盖 CoilInput 的全部标量字段。"""
    covered = {e[1].split(".")[0] for _, es in _SECTIONS for e in es}
    covered |= {"layers", "turn_layers"}
    missing = [f.name for f in dc_fields(CoilInput) if f.name not in covered]
    assert not missing, f"config_io 未覆盖字段: {missing}"
