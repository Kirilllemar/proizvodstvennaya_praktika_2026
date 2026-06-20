#!/usr/bin/env python3
"""Запуск: python run.py  или  python start.py"""

import os

from app import create_app

app = create_app()

HOST = os.environ.get("FOODVISION_HOST", "127.0.0.1")
PORT = int(os.environ.get("FOODVISION_PORT", "5000"))
DEBUG = os.environ.get("FOODVISION_DEBUG", "1") not in ("0", "false", "False")

if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print("\n" + "=" * 50)
    print("  FoodVision — распознавание блюд (вариант 4)")
    print(f"  Откройте в браузере: {url}")
    print("=" * 50 + "\n")
    app.run(debug=DEBUG, host=HOST, port=PORT)
