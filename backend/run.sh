#!/usr/bin/env bash
# Local development launcher.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

[ -f .env ] || cp .env.example .env

python scripts/verify_artifacts.py
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
