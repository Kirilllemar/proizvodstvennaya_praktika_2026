#!/usr/bin/env sh
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  exec python3 start.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python start.py "$@"
else
  echo "Python не найден. Установите Python 3.10+." >&2
  exit 1
fi
