#!/usr/bin/env python3
"""Кроссплатформенный запуск FoodVision (Windows / macOS / Linux)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "venv")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")


def venv_python(root: str = ROOT) -> str:
    """Путь к интерпретатору Python внутри venv для текущей ОС."""
    if platform.system() == "Windows":
        return os.path.join(root, "venv", "Scripts", "python.exe")
    return os.path.join(root, "venv", "bin", "python")


def ensure_venv() -> str:
    """Создать venv и установить зависимости при первом запуске."""
    py = venv_python()
    if os.path.isfile(py):
        return py

    print("Создание виртуального окружения...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

    pip_cmd = [py, "-m", "pip", "install", "-q", "-U", "pip"]
    subprocess.check_call(pip_cmd)

    print("Установка зависимостей (Flask, PyTorch, ...)...")
    subprocess.check_call([py, "-m", "pip", "install", "-q", "-r", REQUIREMENTS])
    return py


def run_app() -> int:
    py = ensure_venv()
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Python: {py}")
    print("Запуск сервера...\n")
    return subprocess.call([py, os.path.join(ROOT, "run.py")], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(run_app())
