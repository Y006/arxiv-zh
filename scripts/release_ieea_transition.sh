#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$ROOT_DIR/legacy/ieea-transition"
cd "$PKG_DIR"

echo "[1/2] Building legacy transition package..."
uv run --no-project --with build python -m build

echo "[2/2] Validating metadata rendering..."
uv run --no-project --with twine twine check dist/*

cat <<'EOF'

Legacy transition package build/check passed.

Upload to TestPyPI:
  uv run --no-project --with twine twine upload --repository testpypi dist/*

Verify:
  python -m venv /tmp/ieea-bridge-testpypi
  source /tmp/ieea-bridge-testpypi/bin/activate
  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ieeA
  ieeA --help
  deactivate

Upload to PyPI:
  uv run --no-project --with twine twine upload dist/*
EOF
