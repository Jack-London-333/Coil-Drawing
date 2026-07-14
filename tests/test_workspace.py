"""启动工作空间选择与初始化行为测试。"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

import coildrawing.ui as ui  # noqa: E402


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_normal_startup_cancel_exits_without_writes(tmp_path, monkeypatch):
    _app()
    previous = tmp_path / "previous-workspace"
    previous.mkdir()
    state = tmp_path / "state.json"
    original_state = json.dumps({"last_workspace": str(previous)},
                                ensure_ascii=False)
    state.write_text(original_state, encoding="utf-8")
    monkeypatch.setattr(ui, "APP_STATE_FILE", state)
    monkeypatch.delenv("COILDRAWING_WORKSPACE", raising=False)

    chooser_calls = []

    def cancel(*args):
        chooser_calls.append(args)
        return ""

    monkeypatch.setattr(ui.QFileDialog, "getExistingDirectory", cancel)
    win = ui.MainWindow()
    try:
        assert win.startup_cancelled is True
        assert win.workspace is None
        assert len(chooser_calls) == 1
        assert chooser_calls[0][2] == str(previous)
        assert state.read_text(encoding="utf-8") == original_state
        assert not (previous / "config.txt").exists()
        assert not (previous / "output").exists()
    finally:
        win.close()


def test_confirm_empty_workspace_initializes_config_and_output(tmp_path,
                                                               monkeypatch):
    _app()
    workspace = tmp_path / "empty-workspace"
    workspace.mkdir()
    state = tmp_path / "state.json"
    monkeypatch.setattr(ui, "APP_STATE_FILE", state)
    monkeypatch.delenv("COILDRAWING_WORKSPACE", raising=False)
    monkeypatch.setattr(
        ui.QFileDialog, "getExistingDirectory", lambda *args: str(workspace))

    win = ui.MainWindow()
    try:
        assert win.startup_cancelled is False
        assert win.workspace == workspace
        assert (workspace / "config.txt").is_file()
        assert (workspace / "output").is_dir()
        assert "seita3_deg" in (workspace / "config.txt").read_text(
            encoding="utf-8")
        assert math.isclose(win.gather_input().seita3, math.radians(80.0),
                            rel_tol=0, abs_tol=1e-15)
        assert json.loads(state.read_text(encoding="utf-8"))["last_workspace"] == str(workspace)
    finally:
        win.close()


def test_workspace_environment_bypasses_chooser_and_does_not_change_history(
        tmp_path, monkeypatch):
    _app()
    workspace = tmp_path / "automated-workspace"
    state = tmp_path / "state.json"
    original_state = '{"last_workspace": "user-workspace"}'
    state.write_text(original_state, encoding="utf-8")
    monkeypatch.setattr(ui, "APP_STATE_FILE", state)
    monkeypatch.setenv("COILDRAWING_WORKSPACE", str(workspace))

    def unexpected_chooser(*_args):
        raise AssertionError("COILDRAWING_WORKSPACE 模式不应弹出目录选择框")

    monkeypatch.setattr(ui.QFileDialog, "getExistingDirectory",
                        unexpected_chooser)
    win = ui.MainWindow()
    try:
        assert win.startup_cancelled is False
        assert win.workspace == workspace
        assert (workspace / "config.txt").is_file()
        assert (workspace / "output").is_dir()
        assert state.read_text(encoding="utf-8") == original_state
    finally:
        win.close()
