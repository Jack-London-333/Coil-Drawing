"""CoilDrawing 桌面界面（PySide6）。

左侧：参数输入（分组表单 + 绝缘分层表格）
右侧：计算结果表 / 线圈大样图预览 两个标签页
底部：计算、导出 Excel/CSV、导出大样图 PNG、导出 STEP(3D)
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QStatusBar, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .engine import CoilInput, CoilResult, InsulationLayer, WireSpec, compute
from .export import export_csv, export_xlsx

# matplotlib 嵌入
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg, NavigationToolbar2QT)

from .drawing2d import make_figure, save_png  # noqa: E402


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
def _dspin(val: float, lo=0.0, hi=1e6, step=0.1, dec=3, suffix=" mm") -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(dec)
    sp.setSingleStep(step)
    sp.setValue(val)
    if suffix:
        sp.setSuffix(suffix)
    sp.setAlignment(Qt.AlignRight)
    return sp


def _ispin(val: int, lo=0, hi=100000) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(val)
    sp.setAlignment(Qt.AlignRight)
    return sp


class MainWindow(QMainWindow):
    SETTINGS_NAME = "coildrawing_last_input.json"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CoilDrawing — 电机定子成型线圈计算与建模（CN104965948B 公式体系）")
        self.resize(1420, 860)
        self._result: CoilResult | None = None
        self._worker: StepExportWorker | None = None
        self._build_ui()
        self._load_last_input()
        self.recalculate()

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
        _, f1 = group("铁芯与槽")
        self.in_d2 = _dspin(1180.0, 10, 20000, 1, 1)
        self.in_lc = _dspin(1250.0, 10, 20000, 1, 1)
        self.in_ns = _ispin(108, 6, 2000)
        self.in_poles = _ispin(12, 2, 200)
        self.in_taw = _ispin(9, 2, 200)
        self.in_hs = _dspin(74.0, 1, 500, 0.5, 2)
        self.in_ws = _dspin(11.5, 1, 100, 0.1, 2)
        self.in_hsd = _dspin(4.0, 0, 50, 0.1, 2)
        self.in_wihu = _dspin(1.0, 0, 20, 0.1, 2)
        self.in_wihm = _dspin(3.0, 0, 20, 0.1, 2)
        self.in_wihb = _dspin(1.0, 0, 20, 0.1, 2)
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
        _, f2 = group("绕组与线规")
        self.in_turns = _ispin(8, 1, 200)
        self.in_w1b = _dspin(8.2, 0, 50, 0.05)
        self.in_w1h = _dspin(3.35, 0, 50, 0.05)
        self.in_w1t0 = _dspin(0.0, 0, 5, 0.01)
        self.in_w1npd = _ispin(1, 0, 20)
        self.in_w1ncd = _ispin(1, 0, 20)
        self.in_w2b = _dspin(0.0, 0, 50, 0.05)
        self.in_w2h = _dspin(0.0, 0, 50, 0.05)
        self.in_w2t0 = _dspin(0.0, 0, 5, 0.01)
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
        _, f3 = group("绝缘（单边厚度）")
        self.in_t1 = _dspin(0.15, 0, 5, 0.01)
        self.in_t3 = _dspin(0.15, 0, 5, 0.01)
        self.in_t2 = _dspin(1.1, 0, 10, 0.05)
        self.in_t4 = _dspin(1.1, 0, 10, 0.05)
        self.in_cs = _dspin(0.0, 0, 5, 0.05)
        f3.addRow("槽内匝间绝缘 T1", self.in_t1)
        f3.addRow("端部匝间绝缘 T3", self.in_t3)
        f3.addRow("槽内对地绝缘 T2", self.in_t2)
        f3.addRow("端部对地绝缘 T4", self.in_t4)
        f3.addRow("槽内防晕层 CS", self.in_cs)

        # --- 3D 绝缘分层表 ---
        g4 = QGroupBox("三维模型对地绝缘分层（云母带等，由内到外）")
        v4 = QVBoxLayout(g4)
        self.layer_table = QTableWidget(0, 2)
        self.layer_table.setHorizontalHeaderLabels(["层名称", "单边厚度 mm"])
        self.layer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setMinimumHeight(120)
        v4.addWidget(self.layer_table)
        hb = QHBoxLayout()
        btn_add = QPushButton("＋ 添加层")
        btn_del = QPushButton("－ 删除选中层")
        btn_add.clicked.connect(self._add_layer_row)
        btn_del.clicked.connect(self._del_layer_row)
        hb.addWidget(btn_add)
        hb.addWidget(btn_del)
        hb.addStretch(1)
        self.lbl_layer_sum = QLabel("分层总厚：0.00 mm")
        hb.addWidget(self.lbl_layer_sum)
        v4.addLayout(hb)
        form_v.addWidget(g4)
        self.layer_table.itemChanged.connect(lambda *_: self._update_layer_sum())
        for name, t in [("对地云母带 1", 0.55), ("对地云母带 2", 0.55)]:
            self._add_layer_row(name, t)

        # --- 端部结构 ---
        _, f5 = group("端部结构")
        self.in_ld = _dspin(20.0, 0, 200, 0.5)
        self.in_le = _dspin(20.0, 0, 200, 0.5)
        self.in_f = _dspin(20.0, 0, 200, 0.5)
        self.in_seita3 = _dspin(0.349, 0, 1.57, 0.001, 3, " rad")
        self.in_rd = _dspin(15.0, 1, 100, 0.5)
        self.in_rd1 = _dspin(15.0, 0, 100, 0.5)
        self.in_rd2 = _dspin(15.0, 0, 100, 0.5)
        self.in_rbs = _dspin(30.0, 1, 200, 0.5)
        self.in_rbn = _dspin(30.0, 1, 200, 0.5)
        self.in_ba = _dspin(7.0, 0.1, 100, 0.1)
        self.in_ysc = _dspin(45.0, 0, 500, 1)
        self.in_xi = _dspin(0.01, 0.001, 0.02, 0.001, 3, "")
        f5.addRow("齿压板轴向长度 LD", self.in_ld)
        f5.addRow("直线部伸出铁芯 LE", self.in_le)
        f5.addRow("鼻端抬高 F", self.in_f)
        f5.addRow("鼻端中心线夹角 seita3", self.in_seita3)
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
        scroll.setMinimumWidth(390)

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
            "三维模型为 STEP (AP214) 实体：铜导体束 + 逐层绝缘。\n"
            "Parasolid(.x_t) 为西门子私有格式：请在 SolidWorks / NX / Solid Edge 中打开\n"
            "STEP 后另存为 .x_t（这些软件即 Parasolid 内核，转换零损失）。")
        note.setStyleSheet("color:#666; font-size:11px; padding:2px 6px;")
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
    def _add_layer_row(self, name: str | bool = False, t: float = 0.5) -> None:
        row = self.layer_table.rowCount()
        self.layer_table.insertRow(row)
        label = name if isinstance(name, str) else f"绝缘层 {row + 1}"
        self.layer_table.setItem(row, 0, QTableWidgetItem(label))
        item_t = QTableWidgetItem(f"{t:g}")
        item_t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layer_table.setItem(row, 1, item_t)
        self._update_layer_sum()

    def _del_layer_row(self) -> None:
        rows = sorted({i.row() for i in self.layer_table.selectedItems()}, reverse=True)
        if not rows and self.layer_table.rowCount():
            rows = [self.layer_table.rowCount() - 1]
        for r in rows:
            self.layer_table.removeRow(r)
        self._update_layer_sum()

    def _layers(self) -> list[InsulationLayer]:
        out = []
        for r in range(self.layer_table.rowCount()):
            name_item = self.layer_table.item(r, 0)
            t_item = self.layer_table.item(r, 1)
            name = name_item.text().strip() if name_item else f"绝缘层 {r + 1}"
            try:
                t = float(t_item.text()) if t_item else 0.0
            except ValueError:
                t = 0.0
            out.append(InsulationLayer(name or f"绝缘层 {r + 1}", t))
        return out

    def _update_layer_sum(self) -> None:
        s = sum(l.thickness for l in self._layers())
        self.lbl_layer_sum.setText(f"分层总厚：{s:.2f} mm（应≈对地绝缘 T2）")

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
            layers=self._layers(),
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
        self.layer_table.setRowCount(0)
        for layer in inp.layers:
            self._add_layer_row(layer.name, layer.thickness)

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
        self._save_last_input(inp)
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

    # ------------------------------------------------------------------
    def _ensure_result(self) -> CoilResult | None:
        if self._result is None:
            self.recalculate()
        return self._result

    def on_export_xlsx(self) -> None:
        res = self._ensure_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel", "coil_result.xlsx", "Excel (*.xlsx)")
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
            self, "导出 CSV", "coil_result.csv", "CSV (*.csv)")
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
            self, "导出大样图", "coil_drawing.png", "PNG 图片 (*.png)")
        if not path:
            return
        try:
            save_png(res, path)
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
            self, "导出 STEP 三维模型", "coil_3d.step", "STEP (*.step *.stp)")
        if not path:
            return
        self.btn_step.setEnabled(False)
        self.btn_step.setText("正在生成 3D…")
        self.statusBar().showMessage("正在生成三维模型（首次运行需加载几何内核，约 10~30 秒）…")
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

    # ------------------------------------------------------------------
    def _settings_path(self) -> Path:
        return Path.home() / self.SETTINGS_NAME

    def _save_last_input(self, inp: CoilInput) -> None:
        try:
            self._settings_path().write_text(
                json.dumps(asdict(inp), ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError:
            pass

    def _load_last_input(self) -> None:
        p = self._settings_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["wire1"] = WireSpec(**data.get("wire1", {}))
            data["wire2"] = WireSpec(**data.get("wire2", {}))
            data["layers"] = [InsulationLayer(**d) for d in data.get("layers", [])]
            self.apply_input(CoilInput(**data))
        except Exception:
            pass  # 配置损坏则用默认值


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
