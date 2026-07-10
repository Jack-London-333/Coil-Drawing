"""CoilDrawing 启动入口。

正常启动：      uv run main.py
打包自检模式：  CoilDrawing.exe --smoke [输出目录]
    在无界面（offscreen）环境下依次验证：Qt 界面构建、计算引擎、
    大样图 PNG、三维建模 + STEP 导出，并把产物写入输出目录。
    全部成功时输出目录中生成 smoke_ok.txt，退出码为 0。
"""

from __future__ import annotations

import os
import sys


def run_smoke(out_dir: str) -> int:
    # 默认用原生平台插件（窗口不 show 不会显示）；无桌面环境时可设
    # COILDRAWING_SMOKE_OFFSCREEN=1 切换到 offscreen。
    if os.environ.get("COILDRAWING_SMOKE_OFFSCREEN") == "1":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    from PySide6.QtWidgets import QApplication

    from coildrawing.drawing2d import save_png
    from coildrawing.engine import CoilInput, compute
    from coildrawing.ui import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()          # 构建全部界面并完成一次默认计算
    win.grab().save(str(out / "smoke_ui.png"))
    report.append(f"UI ok: {win.windowTitle()}")

    res = compute(CoilInput())  # 专利算例
    report.append(f"engine ok: LLM={res.llm:.1f} mm, CC={res.cc:.1f} mm")

    save_png(res, str(out / "smoke_drawing.png"))
    report.append("drawing ok: smoke_drawing.png")

    from coildrawing.model3d import export_step

    names = export_step(res, str(out / "smoke_coil.step"))
    report.append(f"3d ok: {len(names)} parts -> smoke_coil.step")

    (out / "smoke_ok.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    del win, app
    return 0


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        idx = sys.argv.index("--smoke")
        target = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "smoke_out"
        try:
            sys.exit(run_smoke(target))
        except Exception:
            import traceback

            from pathlib import Path

            Path(target).mkdir(parents=True, exist_ok=True)
            Path(target, "smoke_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
            traceback.print_exc()
            sys.exit(1)

    from coildrawing.ui import main

    sys.exit(main())
