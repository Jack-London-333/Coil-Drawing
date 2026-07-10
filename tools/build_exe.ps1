# CoilDrawing 一键打包脚本（在项目根目录运行）：
#   powershell -ExecutionPolicy Bypass -File tools\build_exe.ps1
#
# 产物：dist\CoilDrawing\ 整个文件夹即为免安装软件，双击其中 CoilDrawing.exe 使用。

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== 1/3 同步依赖 (uv sync) ==" -ForegroundColor Cyan
uv sync

Write-Host "== 2/3 PyInstaller 打包 ==" -ForegroundColor Cyan
uv run pyinstaller CoilDrawing.spec --noconfirm --clean

Write-Host "== 3/3 附带说明文档 ==" -ForegroundColor Cyan
Copy-Item README.md dist\CoilDrawing\README.md -Force
New-Item -ItemType Directory -Force -Path dist\CoilDrawing\docs\images | Out-Null
Copy-Item docs\使用教程.md dist\CoilDrawing\docs\ -Force
Copy-Item docs\images\* dist\CoilDrawing\docs\images\ -Force

$size = [math]::Round((Get-ChildItem dist\CoilDrawing -Recurse -File | Measure-Object -Sum Length).Sum / 1MB)
Write-Host "完成：dist\CoilDrawing（约 $size MB）— 双击 CoilDrawing.exe 使用" -ForegroundColor Green
