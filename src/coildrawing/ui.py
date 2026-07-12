"""CoilDrawing 桌面界面（PySide6）。

左侧：参数输入（分组表单 + 绝缘分层表格）
右侧：计算结果表 / 线圈大样图预览 两个标签页
底部：计算、导出 Excel/CSV、导出大样图、导出 STEP(3D)

工作空间机制：每次任务对应一个目录（可新建/切换），目录中的
config.txt 保存全部参数（INI 风格纯文本，可手工编辑，软件检测到
外部修改会提示载入）；各类导出默认放入工作空间的 output 子目录。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QLocale, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QDoubleValidator, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QStatusBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .config_io import CONFIG_NAME, load_config, parse_config_text, save_config
from .engine import CoilInput, CoilResult, InsulationLayer, WireSpec, compute
from .export import export_csv, export_xlsx

# matplotlib 嵌入
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg, NavigationToolbar2QT)

from .drawing2d import make_figure, save_figure  # noqa: E402

APP_STATE_FILE = Path.home() / ".coildrawing_app.json"   # 记住上次工作空间


# ----------------------------------------------------------------------
class StepExportWorker(QThread):
    """后台线程执行 3D 建模 + STEP 导出（首次导入 OCCT 内核较慢）。"""

    done = Signal(str, list)   # filepath, part names
    failed = Signal(str)

    def __init__(self, res: CoilResult, filepath: str, parent=None):
        super().__init__(parent)
        self._res = res
        self._path = filepath

    def run(self) -> None:
        try:
            from .model3d import export_step
            names = export_step(self._res, self._path)
            self.done.emit(self._path, names)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ----------------------------------------------------------------------
def _fmt(v: float) -> str:
    """数值 → 输入框文本（最多 15 位有效数字，无多余尾零）。"""
    return format(float(v), ".15g")


class NumEdit(QLineEdit):
    """自由文本数值输入框：支持 8 位小数与科学计数法，不响应滚轮。

    （QLineEdit 本身无滚轮改值行为，从根上避免误触。）
    """

    def __init__(self, val: float, lo: float = 0.0, hi: float = 1e6,
                 parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self._last_good = float(val)
        v = QDoubleValidator(lo, hi, 15, self)
        v.setNotation(QDoubleValidator.ScientificNotation)
        v.setLocale(QLocale.c())
        self.setValidator(v)
        self.setAlignment(Qt.AlignRight)
        self.setValue(val)
        self.editingFinished.connect(self._normalize)

    def value(self) -> float:
        try:
            x = float(self.text().replace("，", ".").replace(",", "."))
        except ValueError:
            return self._last_good
        x = min(max(x, self._lo), self._hi)
        self._last_good = x
        return x

    def setValue(self, v: float) -> None:
        self._last_good = float(v)
        self.setText(_fmt(v))

    def _normalize(self) -> None:
        self.setText(_fmt(self.value()))


class IntSpin(QSpinBox):
    """整数微调框：不响应滚轮（避免误触改值）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e) -> None:  # noqa: N802
        e.ignore()


class NoWheelComboBox(QComboBox):
    """下拉框：不响应滚轮（避免误触换项）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, e) -> None:  # noqa: N802
        e.ignore()


def _dnum(val: float, lo: float = 0.0, hi: float = 1e6) -> NumEdit:
    return NumEdit(val, lo, hi)


def _ispin(val: int, lo=0, hi=100000) -> IntSpin:
    sp = IntSpin()
    sp.setRange(lo, hi)
    sp.setValue(val)
    sp.setAlignment(Qt.AlignRight)
    return sp


class LayerTableGroup(QGroupBox):
    """绝缘分层编辑表（名称 + 单边厚度），带增删按钮与总厚提示。"""

    def __init__(self, title: str, sum_hint: str, parent=None):
        super().__init__(title, parent)
        self._sum_hint = sum_hint
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["层名称", "单边厚度 mm"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(90)
        v.addWidget(self.table)
        hb = QHBoxLayout()
        btn_add = QPushButton("＋ 添加层")
        btn_del = QPushButton("－ 删除选中层")
        btn_add.clicked.connect(self.add_row)
        btn_del.clicked.connect(self.del_row)
        hb.addWidget(btn_add)
        hb.addWidget(btn_del)
        hb.addStretch(1)
        self.lbl_sum = QLabel()
        hb.addWidget(self.lbl_sum)
        v.addLayout(hb)
        self.table.itemChanged.connect(lambda *_: self.update_sum())
        self.update_sum()

    def add_row(self, name: str | bool = False, t: float = 0.5) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        label = name if isinstance(name, str) else f"绝缘层 {row + 1}"
        self.table.setItem(row, 0, QTableWidgetItem(label))
        item_t = QTableWidgetItem(_fmt(t))
        item_t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 1, item_t)
        self.update_sum()

    def del_row(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self.update_sum()

    def set_layers(self, layers: list[InsulationLayer]) -> None:
        self.table.setRowCount(0)
        for layer in layers:
            self.add_row(layer.name, layer.thickness)

    def layers(self) -> list[InsulationLayer]:
        out = []
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, 0)
            t_item = self.table.item(r, 1)
            name = name_item.text().strip() if name_item else f"绝缘层 {r + 1}"
            try:
                t = float(t_item.text()) if t_item else 0.0
            except ValueError:
                t = 0.0
            out.append(InsulationLayer(name or f"绝缘层 {r + 1}", t))
        return out

    def update_sum(self) -> None:
        s = sum(l.thickness for l in self.layers())
        self.lbl_sum.setText(f"分层总厚：{_fmt(round(s, 8))} mm（{self._sum_hint}）")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1420, 880)
        self._result: CoilResult | None = None
        self._worker: StepExportWorker | None = None
        self.workspace: Path | None = None
        self._cfg_text_written = ""       # 最近一次自己写出的配置文本
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_config_file_changed)
        self._build_menu()
        self._build_ui()
        self._init_workspace()
        self.recalculate()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("文件(&F)")

        def act(text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            m_file.addAction(a)
            return a

        act("选择/新建工作空间(&W)…", self.on_choose_workspace)
        act("导入配置文件(&I)…（复制为 config.txt）", self.on_import_config)
        m_file.addSeparator()
        act("保存配置(&S)", self.on_save_config, "Ctrl+S")
        act("重新载入配置(&R)", self.on_reload_config, "F6")
        m_file.addSeparator()
        act("打开工作空间文件夹(&O)", self.on_open_workspace_dir)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # ============ 左侧：输入 ============
        form_host = QWidget()
        form_v = QVBoxLayout(form_host)
        form_v.setContentsMargins(8, 8, 8, 8)

        def group(title: str) -> tuple[QGroupBox, QFormLayout]:
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setLabelAlignment(Qt.AlignRight)
            form_v.addWidget(g)
            return g, f

        # --- 铁芯与槽 ---
        _, f1 = group("铁芯与槽（长度单位 mm）")
        self.in_d2 = _dnum(1180.0, 10, 20000)
        self.in_lc = _dnum(1250.0, 10, 20000)
        self.in_ns = _ispin(108, 6, 2000)
        self.in_poles = _ispin(12, 2, 200)
        self.in_taw = _ispin(9, 2, 200)
        self.in_hs = _dnum(74.0, 1, 500)
        self.in_ws = _dnum(11.5, 1, 100)
        self.in_hsd = _dnum(4.0, 0, 50)
        self.in_wihu = _dnum(1.0, 0, 20)
        self.in_wihm = _dnum(3.0, 0, 20)
        self.in_wihb = _dnum(1.0, 0, 20)
        f1.addRow("定子铁芯内径 D2", self.in_d2)
        f1.addRow("铁芯轴向长度 LC", self.in_lc)
        f1.addRow("定子槽数 NS", self.in_ns)
        f1.addRow("极数 2P", self.in_poles)
        f1.addRow("线圈节距 TAW (槽)", self.in_taw)
        f1.addRow("槽深 HS", self.in_hs)
        f1.addRow("槽宽 WS", self.in_ws)
        f1.addRow("槽楔厚度 HSD", self.in_hsd)
        f1.addRow("槽楔下垫片 WIHU", self.in_wihu)
        f1.addRow("层间垫片 WIHM", self.in_wihm)
        f1.addRow("槽底垫片 WIHB", self.in_wihb)

        # --- 绕组线规 ---
        _, f2 = group("绕组与线规（长度单位 mm）")
        self.in_turns = _ispin(8, 1, 200)
        self.in_w1b = _dnum(8.2, 0, 50)
        self.in_w1h = _dnum(3.35, 0, 50)
        self.in_w1t0 = _dnum(0.0, 0, 5)
        self.in_w1npd = _ispin(1, 0, 20)
        self.in_w1ncd = _ispin(1, 0, 20)
        self.in_w2b = _dnum(0.0, 0, 50)
        self.in_w2h = _dnum(0.0, 0, 50)
        self.in_w2t0 = _dnum(0.0, 0, 5)
        self.in_w2npd = _ispin(0, 0, 20)
        self.in_w2ncd = _ispin(0, 0, 20)
        f2.addRow("线圈匝数 N", self.in_turns)
        f2.addRow("导线1 裸线宽 WB1", self.in_w1b)
        f2.addRow("导线1 裸线厚 WT1", self.in_w1h)
        f2.addRow("导线1 自身绝缘 T01(单边)", self.in_w1t0)
        f2.addRow("导线1 并绕根数 NPD1", self.in_w1npd)
        f2.addRow("导线1 每匝层数 NCD1", self.in_w1ncd)
        f2.addRow("导线2 裸线宽 WB2 (无则0)", self.in_w2b)
        f2.addRow("导线2 裸线厚 WT2", self.in_w2h)
        f2.addRow("导线2 自身绝缘 T02(单边)", self.in_w2t0)
        f2.addRow("导线2 并绕根数 NPD2", self.in_w2npd)
        f2.addRow("导线2 每匝层数 NCD2", self.in_w2ncd)

        # --- 绝缘 ---
        _, f3 = group("绝缘（单边厚度，单位 mm）")
        self.in_t1 = _dnum(0.15, 0, 5)
        self.in_t3 = _dnum(0.15, 0, 5)
        self.in_t2 = _dnum(1.1, 0, 10)
        self.in_t4 = _dnum(1.1, 0, 10)
        self.in_cs = _dnum(0.0, 0, 5)
        f3.addRow("槽内匝间绝缘 T1", self.in_t1)
        f3.addRow("端部匝间绝缘 T3", self.in_t3)
        f3.addRow("槽内对地绝缘 T2", self.in_t2)
        f3.addRow("端部对地绝缘 T4", self.in_t4)
        f3.addRow("槽内防晕层 CS", self.in_cs)

        # --- 3D 绝缘分层表 ---
        self.grp_ground = LayerTableGroup(
            "三维模型对地绝缘分层（云母带等，由内到外）", "应≈对地绝缘 T2")
        form_v.addWidget(self.grp_ground)
        self.grp_ground.set_layers([InsulationLayer("对地云母带 1", 0.55),
                                    InsulationLayer("对地云母带 2", 0.55)])

        self.grp_turn = LayerTableGroup(
            "三维模型匝绝缘分层（包每匝导线束，由内到外）", "应≈匝间绝缘 T1")
        form_v.addWidget(self.grp_turn)
        self.grp_turn.set_layers([InsulationLayer("匝间云母带", 0.15)])

        # --- 三维模型选项 ---
        _, f4b = group("三维模型（逐匝精细建模）")
        self.in_detail = NoWheelComboBox()
        self.in_detail.addItems(["逐匝精细模型（铜线逐匝+分层绝缘+引线）",
                                 "简化束模型（等效整束，生成快）"])
        self.in_leadbr = _dnum(15.0, 1, 200)
        self.in_leadbare = _dnum(30.0, 0, 300)
        self.in_lead_end = NoWheelComboBox()
        self.in_lead_end.addItems([
            "端部侧视图右端（轴向 +Z）",
            "端部侧视图左端（轴向 -Z）",
        ])
        self.in_corona = QCheckBox("绘制槽部防晕层（黑色半导电层，厚度=CS）")
        self.in_corona.toggled.connect(self._on_corona_toggled)
        self.in_corona_ov = _dnum(50.0, 0, 1000)
        f4b.addRow("导出 STEP 模型", self.in_detail)
        f4b.addRow("引线折弯半径 (mm)", self.in_leadbr)
        f4b.addRow("引线端头裸铜长 (mm, 0=不留)", self.in_leadbare)
        f4b.addRow("出线端位置", self.in_lead_end)
        f4b.addRow(self.in_corona)
        f4b.addRow("防晕层每端伸出铁芯 (mm, 沿导线)", self.in_corona_ov)
        lbl_cor = QLabel("伸出超过直线段时自动越过槽口弯角、沿端臂向鼻端延伸")
        lbl_cor.setStyleSheet("color:#888; font-size:11px;")
        f4b.addRow("", lbl_cor)

        # 槽内固定件
        hw = QWidget()
        hw_grid = QGridLayout(hw)
        hw_grid.setContentsMargins(0, 0, 0, 0)
        self.in_draw_wedge = QCheckBox("槽楔 HSD")
        self.in_draw_wihu = QCheckBox("槽楔下垫片 WIHU")
        self.in_draw_wihm = QCheckBox("层间垫片 WIHM")
        self.in_draw_wihb = QCheckBox("槽底垫片 WIHB")
        hw_grid.addWidget(self.in_draw_wedge, 0, 0)
        hw_grid.addWidget(self.in_draw_wihu, 0, 1)
        hw_grid.addWidget(self.in_draw_wihm, 1, 0)
        hw_grid.addWidget(self.in_draw_wihb, 1, 1)
        f4b.addRow("槽内固定件加入模型", hw)

        # --- 端部结构 ---
        _, f5 = group("端部结构（长度单位 mm）")
        self.in_ld = _dnum(20.0, 0, 200)
        self.in_le = _dnum(20.0, 0, 200)
        self.in_f = _dnum(20.0, 0, 200)
        self.in_seita3 = _dnum(0.349, 0, 1.57)
        self.in_rd = _dnum(15.0, 1, 100)
        self.in_rd1 = _dnum(15.0, 0, 100)
        self.in_rd2 = _dnum(15.0, 0, 100)
        self.in_rbs = _dnum(30.0, 1, 200)
        self.in_rbn = _dnum(30.0, 1, 200)
        self.in_ba = _dnum(7.0, 0.1, 100)
        self.in_ysc = _dnum(45.0, 0, 500)
        self.in_xi = _dnum(1e-9, 1e-12, 0.02)
        f5.addRow("齿压板轴向长度 LD", self.in_ld)
        f5.addRow("直线部伸出铁芯 LE", self.in_le)
        f5.addRow("鼻端抬高 F", self.in_f)
        f5.addRow("鼻端中心线夹角 seita3 (rad)", self.in_seita3)
        f5.addRow("鼻端半径 RD", self.in_rd)
        f5.addRow("接线侧弯弧半径 RD1", self.in_rd1)
        f5.addRow("非接线侧弯弧半径 RD2", self.in_rd2)
        f5.addRow("直线部-斜边弯曲半径 rd1", self.in_rbs)
        f5.addRow("斜边-鼻端弯曲半径 rd2", self.in_rbn)
        f5.addRow("端部间隙给定值 Ba", self.in_ba)
        f5.addRow("引线长 ysc", self.in_ysc)
        f5.addRow("迭代误差 ξ", self.in_xi)

        form_v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(form_host)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(scroll)

        btn_bar = QHBoxLayout()
        self.btn_calc = QPushButton("计算 (F5)")
        self.btn_calc.setShortcut("F5")
        self.btn_calc.setStyleSheet("font-weight:bold; padding:6px 14px;")
        self.btn_calc.clicked.connect(self.recalculate)
        btn_bar.addWidget(self.btn_calc)
        self.btn_xlsx = QPushButton("导出 Excel")
        self.btn_xlsx.clicked.connect(self.on_export_xlsx)
        btn_bar.addWidget(self.btn_xlsx)
        self.btn_csv = QPushButton("导出 CSV")
        self.btn_csv.clicked.connect(self.on_export_csv)
        btn_bar.addWidget(self.btn_csv)
        self.btn_png = QPushButton("导出大样图")
        self.btn_png.clicked.connect(self.on_export_png)
        btn_bar.addWidget(self.btn_png)
        self.btn_step = QPushButton("导出 STEP (3D)")
        self.btn_step.clicked.connect(self.on_export_step)
        btn_bar.addWidget(self.btn_step)
        lv.addLayout(btn_bar)

        note = QLabel(
            "参数保存在工作空间 config.txt 中（可手工编辑，外部修改会提示载入）；"
            "导出默认到工作空间 output 目录。\n"
            "三维模型为 STEP (AP214) 实体。逐匝精细模型生成约需 15~60 秒；"
            "需要 Parasolid(.x_t) 时用 SolidWorks / NX 打开 STEP 另存即可。")
        note.setStyleSheet("color:#666; font-size:11px; padding:2px 6px;")
        note.setWordWrap(True)
        lv.addWidget(note)
        splitter.addWidget(left)

        # ============ 右侧：结果 ============
        self.tabs = QTabWidget()

        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["符号", "名称", "数值", "单位"])
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.result_table, "计算结果")

        canvas_host = QWidget()
        cv = QVBoxLayout(canvas_host)
        self.canvas = FigureCanvasQTAgg(matplotlib.figure.Figure())
        self.toolbar = NavigationToolbar2QT(self.canvas, canvas_host)
        cv.addWidget(self.toolbar)
        cv.addWidget(self.canvas, 1)
        self.tabs.addTab(canvas_host, "线圈大样图")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪 — 修改参数后点击“计算”或按 F5")

    # ------------------------------------------------------------------
    def _on_corona_toggled(self, checked: bool) -> None:
        if checked and self.in_cs.value() <= 0:
            self.in_cs.setValue(0.30)
            self.statusBar().showMessage(
                "已自动将槽内防晕层 CS 设为 0.30 mm（防晕层厚度=CS，"
                "计算与三维模型保持一致），请按 F5 重新计算")

    # ------------------------------------------------------------------
    def gather_input(self) -> CoilInput:
        return CoilInput(
            d2=self.in_d2.value(), lc=self.in_lc.value(),
            ns=self.in_ns.value(), poles=self.in_poles.value(),
            taw=self.in_taw.value(),
            hs=self.in_hs.value(), ws=self.in_ws.value(), hsd=self.in_hsd.value(),
            wihu=self.in_wihu.value(), wihm=self.in_wihm.value(), wihb=self.in_wihb.value(),
            n_turns=self.in_turns.value(),
            wire1=WireSpec(self.in_w1b.value(), self.in_w1h.value(),
                           self.in_w1t0.value(), self.in_w1npd.value(), self.in_w1ncd.value()),
            wire2=WireSpec(self.in_w2b.value(), self.in_w2h.value(),
                           self.in_w2t0.value(), self.in_w2npd.value(), self.in_w2ncd.value()),
            t1=self.in_t1.value(), t3=self.in_t3.value(),
            t2=self.in_t2.value(), t4=self.in_t4.value(), cs=self.in_cs.value(),
            ld=self.in_ld.value(), le=self.in_le.value(), f_nose=self.in_f.value(),
            seita3=self.in_seita3.value(), rd_nose=self.in_rd.value(),
            rd1_conn=self.in_rd1.value(), rd2_nonconn=self.in_rd2.value(),
            r_bend_slot=self.in_rbs.value(), r_bend_nose=self.in_rbn.value(),
            ba=self.in_ba.value(), ysc=self.in_ysc.value(), xi=self.in_xi.value(),
            layers=self.grp_ground.layers(),
            turn_layers=self.grp_turn.layers(),
            lead_bend_r=self.in_leadbr.value(),
            lead_bare=self.in_leadbare.value(),
            lead_end_positive_z=self.in_lead_end.currentIndex() == 0,
            corona_on=self.in_corona.isChecked(),
            corona_overhang=self.in_corona_ov.value(),
            detail_3d=self.in_detail.currentIndex() == 0,
            draw_wedge=self.in_draw_wedge.isChecked(),
            draw_wihu=self.in_draw_wihu.isChecked(),
            draw_wihm=self.in_draw_wihm.isChecked(),
            draw_wihb=self.in_draw_wihb.isChecked(),
        )

    def apply_input(self, inp: CoilInput) -> None:
        self.in_d2.setValue(inp.d2); self.in_lc.setValue(inp.lc)
        self.in_ns.setValue(inp.ns); self.in_poles.setValue(inp.poles)
        self.in_taw.setValue(inp.taw)
        self.in_hs.setValue(inp.hs); self.in_ws.setValue(inp.ws)
        self.in_hsd.setValue(inp.hsd); self.in_wihu.setValue(inp.wihu)
        self.in_wihm.setValue(inp.wihm); self.in_wihb.setValue(inp.wihb)
        self.in_turns.setValue(inp.n_turns)
        w1, w2 = inp.wire1, inp.wire2
        self.in_w1b.setValue(w1.b); self.in_w1h.setValue(w1.h)
        self.in_w1t0.setValue(w1.t0); self.in_w1npd.setValue(w1.npd); self.in_w1ncd.setValue(w1.ncd)
        self.in_w2b.setValue(w2.b); self.in_w2h.setValue(w2.h)
        self.in_w2t0.setValue(w2.t0); self.in_w2npd.setValue(w2.npd); self.in_w2ncd.setValue(w2.ncd)
        self.in_t1.setValue(inp.t1); self.in_t3.setValue(inp.t3)
        self.in_t2.setValue(inp.t2); self.in_t4.setValue(inp.t4); self.in_cs.setValue(inp.cs)
        self.in_ld.setValue(inp.ld); self.in_le.setValue(inp.le)
        self.in_f.setValue(inp.f_nose); self.in_seita3.setValue(inp.seita3)
        self.in_rd.setValue(inp.rd_nose)
        self.in_rd1.setValue(inp.rd1_conn); self.in_rd2.setValue(inp.rd2_nonconn)
        self.in_rbs.setValue(inp.r_bend_slot); self.in_rbn.setValue(inp.r_bend_nose)
        self.in_ba.setValue(inp.ba); self.in_ysc.setValue(inp.ysc); self.in_xi.setValue(inp.xi)
        self.grp_ground.set_layers(inp.layers)
        self.grp_turn.set_layers(inp.turn_layers)
        self.in_detail.setCurrentIndex(0 if inp.detail_3d else 1)
        self.in_leadbr.setValue(inp.lead_bend_r)
        self.in_leadbare.setValue(inp.lead_bare)
        self.in_lead_end.setCurrentIndex(0 if inp.lead_end_positive_z else 1)
        self.in_corona.setChecked(inp.corona_on)
        self.in_corona_ov.setValue(inp.corona_overhang)
        self.in_draw_wedge.setChecked(inp.draw_wedge)
        self.in_draw_wihu.setChecked(inp.draw_wihu)
        self.in_draw_wihm.setChecked(inp.draw_wihm)
        self.in_draw_wihb.setChecked(inp.draw_wihb)

    # ------------------------------------------------------------------
    def recalculate(self) -> None:
        try:
            inp = self.gather_input()
            self._result = compute(inp)
        except Exception as exc:
            self._result = None
            QMessageBox.critical(self, "计算失败", str(exc))
            self.statusBar().showMessage(f"计算失败：{exc}")
            return
        self._fill_result_table(self._result)
        self._draw_preview(self._result)
        self._write_config(inp)
        msg = (f"计算完成：平均匝长 LLM={self._result.llm:.1f} mm，"
               f"端部投影 CC={self._result.cc:.1f} mm，迭代 {self._result.iterations} 次")
        if self._result.warnings:
            msg += f" — ⚠ {len(self._result.warnings)} 条警告（见结果表底部）"
        self.statusBar().showMessage(msg)

    def _fill_result_table(self, res: CoilResult) -> None:
        rows = res.rows()
        warn_rows = [("⚠", w, "", "") for w in res.warnings]
        allrows = rows + warn_rows
        self.result_table.setRowCount(len(allrows))
        bold = QFont()
        bold.setBold(True)
        for i, (sym, name, val, unit) in enumerate(allrows):
            for j, txt in enumerate((sym, name, val, unit)):
                item = QTableWidgetItem(txt)
                if j == 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if sym == "—":
                    item.setBackground(QColor("#FFF2CC"))
                    item.setFont(bold)
                if sym == "⚠":
                    item.setBackground(QColor("#FDE9E9"))
                self.result_table.setItem(i, j, item)
        self.result_table.resizeColumnsToContents()
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def _draw_preview(self, res: CoilResult) -> None:
        fig = make_figure(res)
        self.canvas.figure.clf()
        # 用新 figure 替换画布内容
        self.canvas.figure = fig
        fig.set_canvas(self.canvas)
        self.canvas.draw_idle()

    # ==================================================================
    # 工作空间 / config.txt
    # ==================================================================
    def _load_app_state(self) -> dict:
        try:
            return json.loads(APP_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_app_state(self, **kw) -> None:
        state = self._load_app_state()
        state.update(kw)
        try:
            APP_STATE_FILE.write_text(
                json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _config_path(self) -> Path | None:
        return self.workspace / CONFIG_NAME if self.workspace else None

    def _init_workspace(self) -> None:
        """启动时确定工作空间：优先环境变量/上次的；否则询问；取消则用默认目录。"""
        env_ws = os.environ.get("COILDRAWING_WORKSPACE")
        if env_ws:   # 自检/脚本模式：无对话框直接使用
            self._set_workspace(Path(env_ws))
            return
        last = self._load_app_state().get("last_workspace", "")
        if last and Path(last).is_dir():
            self._set_workspace(Path(last))
            return
        QMessageBox.information(
            self, "选择工作空间",
            "请为本次任务选择（或新建）一个工作空间目录。\n\n"
            "目录中将生成参数配置文件 config.txt 与导出目录 output；\n"
            "以后启动会自动打开上次的工作空间，可用菜单 文件→选择/新建工作空间 切换。")
        path = QFileDialog.getExistingDirectory(
            self, "选择/新建工作空间目录（对话框内可新建文件夹）")
        if not path:
            path = str(Path.home() / "Documents" / "CoilDrawing")
            Path(path).mkdir(parents=True, exist_ok=True)
            QMessageBox.information(
                self, "使用默认工作空间", f"未选择目录，已使用默认工作空间：\n{path}")
        self._set_workspace(Path(path))

    def _set_workspace(self, ws: Path) -> None:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "output").mkdir(exist_ok=True)
        old_cfg = self._config_path()
        if old_cfg is not None and str(old_cfg) in self._watcher.files():
            self._watcher.removePath(str(old_cfg))
        self.workspace = ws
        cfg = self._config_path()
        if cfg.exists():
            try:
                self.apply_input(load_config(cfg))
                self._cfg_text_written = cfg.read_text(encoding="utf-8")
            except (ValueError, OSError) as exc:
                QMessageBox.warning(
                    self, "配置载入失败",
                    f"{cfg} 载入失败，界面保持当前参数：\n\n{exc}")
        else:
            self._write_config(self.gather_input())
        if cfg.exists():
            self._watcher.addPath(str(cfg))
        self._save_app_state(last_workspace=str(ws))
        self.setWindowTitle(
            f"CoilDrawing — 电机定子成型线圈计算与建模（CN104965948B 公式体系）"
            f"  [工作空间: {ws}]")

    def _write_config(self, inp: CoilInput) -> None:
        cfg = self._config_path()
        if cfg is None:
            return
        try:
            self._cfg_text_written = save_config(cfg, inp)
        except OSError as exc:
            self.statusBar().showMessage(f"配置写入失败：{exc}")
            return
        if str(cfg) not in self._watcher.files():
            self._watcher.addPath(str(cfg))

    def _on_config_file_changed(self, path: str) -> None:
        """外部修改 config.txt → 提示载入（自己写出的不提示）。"""
        cfg = self._config_path()
        if cfg is None or str(cfg) != path:
            return
        # 某些编辑器保存是“删除+重建”，需要重新挂监听
        if str(cfg) not in self._watcher.files() and cfg.exists():
            self._watcher.addPath(str(cfg))
        try:
            text = cfg.read_text(encoding="utf-8")
        except OSError:
            return
        if text == self._cfg_text_written:
            return
        ret = QMessageBox.question(
            self, "配置文件已修改",
            f"{cfg.name} 在软件外部被修改，是否载入新参数并重新计算？\n"
            "（选择“No”保留界面当前参数，下次保存将覆盖外部修改）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret != QMessageBox.Yes:
            self._cfg_text_written = text   # 视为已知内容，避免反复弹窗
            return
        try:
            self.apply_input(parse_config_text(text))
        except ValueError as exc:
            QMessageBox.warning(self, "配置解析失败", str(exc))
            self._cfg_text_written = text
            return
        self._cfg_text_written = text
        self.recalculate()

    # ---- 菜单动作 ----
    def on_choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择/新建工作空间目录（对话框内可新建文件夹）",
            str(self.workspace or Path.home()))
        if not path:
            return
        self._set_workspace(Path(path))
        self.recalculate()
        self.statusBar().showMessage(f"已切换工作空间：{path}")

    def on_import_config(self) -> None:
        if self.workspace is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "导入配置文件（将复制为工作空间的 config.txt）",
            str(self.workspace), "配置文本 (*.txt);;所有文件 (*)")
        if not path:
            return
        src = Path(path)
        cfg = self._config_path()
        try:
            inp = load_config(src)      # 先验证再覆盖
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "导入失败", f"{src} 不是有效的配置文件：\n\n{exc}")
            return
        if cfg.exists() and src.resolve() != cfg.resolve():
            ret = QMessageBox.question(
                self, "覆盖确认",
                f"工作空间已有 {CONFIG_NAME}，导入将覆盖它，继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret != QMessageBox.Yes:
                return
        if src.resolve() != cfg.resolve():
            shutil.copyfile(src, cfg)
        self._cfg_text_written = cfg.read_text(encoding="utf-8")
        self.apply_input(inp)
        self.recalculate()
        self.statusBar().showMessage(f"已导入 {src.name} → {cfg}")

    def on_save_config(self) -> None:
        self._write_config(self.gather_input())
        self.statusBar().showMessage(f"配置已保存：{self._config_path()}")

    def on_reload_config(self) -> None:
        cfg = self._config_path()
        if cfg is None or not cfg.exists():
            return
        try:
            inp = load_config(cfg)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "载入失败", str(exc))
            return
        self._cfg_text_written = cfg.read_text(encoding="utf-8")
        self.apply_input(inp)
        self.recalculate()
        self.statusBar().showMessage(f"已重新载入配置：{cfg}")

    def on_open_workspace_dir(self) -> None:
        if self.workspace is not None:
            os.startfile(str(self.workspace))  # noqa: S606

    # ==================================================================
    # 导出（默认到 工作空间/output）
    # ==================================================================
    def _ensure_result(self) -> CoilResult | None:
        if self._result is None:
            self.recalculate()
        return self._result

    def _export_dir(self) -> Path:
        d = (self.workspace / "output") if self.workspace else Path.cwd()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def on_export_xlsx(self) -> None:
        res = self._ensure_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel", str(self._export_dir() / "coil_result.xlsx"),
            "Excel (*.xlsx)")
        if not path:
            return
        try:
            export_xlsx(res, path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出 Excel：{path}")

    def on_export_csv(self) -> None:
        res = self._ensure_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", str(self._export_dir() / "coil_result.csv"),
            "CSV (*.csv)")
        if not path:
            return
        try:
            export_csv(res, path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出 CSV：{path}")

    def on_export_png(self) -> None:
        res = self._ensure_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出大样图", str(self._export_dir() / "coil_drawing.png"),
            "PNG 图片 (*.png);;PDF 矢量图 (*.pdf);;SVG 矢量图 (*.svg)")
        if not path:
            return
        try:
            save_figure(res, path)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出大样图：{path}")

    def on_export_step(self) -> None:
        res = self._ensure_result()
        if res is None:
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "请稍候", "上一个 STEP 导出仍在进行中")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 STEP 三维模型", str(self._export_dir() / "coil_3d.step"),
            "STEP (*.step *.stp)")
        if not path:
            return
        self.btn_step.setEnabled(False)
        self.btn_step.setText("正在生成 3D…")
        mode = "逐匝精细" if res.inp.detail_3d else "简化束"
        self.statusBar().showMessage(
            f"正在生成三维模型（{mode}，首次运行需加载几何内核；"
            "逐匝模型约 15~60 秒）…")
        self._worker = StepExportWorker(res, path, self)
        self._worker.done.connect(self._on_step_done)
        self._worker.failed.connect(self._on_step_failed)
        self._worker.start()

    def _on_step_done(self, path: str, names: list) -> None:
        self.btn_step.setEnabled(True)
        self.btn_step.setText("导出 STEP (3D)")
        self.statusBar().showMessage(f"已导出 STEP：{path}（部件：{'、'.join(names)}）")
        QMessageBox.information(
            self, "导出完成",
            f"三维模型已导出：\n{path}\n\n部件：{chr(10).join(names)}\n\n"
            "如需 Parasolid(.x_t)：请用 SolidWorks / NX / Solid Edge 打开此 STEP，"
            "另存为 Parasolid 即可（无损转换）。")

    def _on_step_failed(self, err: str) -> None:
        self.btn_step.setEnabled(True)
        self.btn_step.setText("导出 STEP (3D)")
        self.statusBar().showMessage("STEP 导出失败")
        QMessageBox.critical(self, "STEP 导出失败", err[-2000:])


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
