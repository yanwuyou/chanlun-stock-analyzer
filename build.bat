@echo off
chcp 65001 >nul
echo ========================================
echo   《缠论》股析 - 打包脚本
echo ========================================
echo.

REM 检查 PyInstaller 是否安装
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] PyInstaller...
    pip install pyinstaller
    echo.
)

echo [清理] 旧构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "*.spec" del /q *.spec

echo [打包] 正在生成 exe（可能需要几分钟）...
echo.

pyinstaller ^
    --onefile ^
    --windowed ^
    --runtime-tmpdir "." ^
    --name "《缠论》股析" ^
    --additional-hooks-dir "pyinstaller_hooks" ^
    --runtime-hook "pyinstaller_hooks\runtime_tkinter.py" ^
    --add-data "chan_lib;chan_lib" ^
    --add-data "activation.py;." ^
    --hidden-import tkinter ^
    --hidden-import _tkinter ^
    --hidden-import baostock ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import chan_lib ^
    --hidden-import chan_lib.data ^
    --hidden-import chan_lib.morphology ^
    --hidden-import chan_lib.dynamics ^
    --hidden-import chan_lib.strategy ^
    --hidden-import chan_lib.visual ^
    --hidden-import mplfinance ^
    --hidden-import watchlist_manager ^
    --exclude-module torch ^
    --exclude-module torchaudio ^
    --exclude-module torchvision ^
    --exclude-module transformers ^
    --exclude-module tensorflow ^
    --exclude-module numba ^
    --exclude-module llvmlite ^
    gui_app.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   打包成功！
    echo   exe 位置: dist\《缠论》股析.exe
    echo ========================================
) else (
    echo.
    echo [失败] 打包出错，请检查上方日志
)

pause
