# ieeA (Legacy Transitional Package)

`ieeA` is now a compatibility bridge package.

Please migrate to:

```bash
pip install -U arxiv-translate
```

Command migration:

- `ieeA ...` -> `arx ...`
- `ieeA ...` -> `arxiv-translate ...`

This transitional release keeps the `ieeA` command available temporarily and
forwards execution to `arxiv-translate` while printing a deprecation warning.
