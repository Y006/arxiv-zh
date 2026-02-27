# PyPI Release Guide

This document describes the release process for:

- Main package: `arxiv-translate`
- Legacy transitional package: `ieeA` (one-time bridge release)

## Prerequisites

```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-xxxxxxxxxxxxxxxx"
```

Install release tools:

```bash
uv sync --group dev
```

## Release `arxiv-translate`

From repository root:

```bash
uv run pytest -q
uv run --with build python -m build
uv run --with twine twine check dist/*
```

Upload to TestPyPI first:

```bash
uv run --with twine twine upload --repository testpypi dist/*
```

Verify install from TestPyPI:

```bash
python -m venv /tmp/arxiv-translate-testpypi
source /tmp/arxiv-translate-testpypi/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple arxiv-translate
arx --help
arxiv-translate --help
deactivate
```

Then upload to PyPI:

```bash
uv run --with twine twine upload dist/*
```

## Release legacy bridge `ieeA` (one-time)

`legacy/ieea-transition` is an independent package config. It publishes an `ieeA`
command that warns and forwards to `arxiv-translate`.

```bash
cd legacy/ieea-transition
uv run --no-project --with build python -m build
uv run --no-project --with twine twine check dist/*
uv run --no-project --with twine twine upload --repository testpypi dist/*
```

Verify on TestPyPI:

```bash
python -m venv /tmp/ieea-bridge-testpypi
source /tmp/ieea-bridge-testpypi/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ieeA
ieeA --help
deactivate
```

Publish after verification:

```bash
uv run --no-project --with twine twine upload dist/*
```
