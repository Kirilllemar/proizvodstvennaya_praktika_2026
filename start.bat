@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python не найден. Установите Python 3.10+ с https://www.python.org/downloads/
    echo При установке отметьте "Add Python to PATH".
    pause
    exit /b 1
)

python start.py
if errorlevel 1 pause
