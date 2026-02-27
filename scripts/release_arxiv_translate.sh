#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Running test suite..."
uv run pytest -q

echo "[2/3] Building distributions..."
uv run --with build python -m build

echo "[3/3] Validating metadata rendering..."
uv run --with twine twine check dist/*

cat <<'EOF'

Build and checks passed.

Upload to TestPyPI:
  uv run --with twine twine upload --repository testpypi dist/*

Verify:
  python -m venv /tmp/arxiv-translate-testpypi
  source /tmp/arxiv-translate-testpypi/bin/activate
  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple arxiv-translate
  arx --help
  arxiv-translate --help
  deactivate

Upload to PyPI:
  uv run --with twine twine upload dist/*
EOF
