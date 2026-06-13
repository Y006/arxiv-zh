# arxiv-zh

`arxiv-zh` 是一个面向本地使用的 arXiv LaTeX 论文中文翻译 CLI。它下载 arXiv 源码、解析可翻译文本块、调用 DeepSeek 翻译、重组 LaTeX，并可用 XeLaTeX 编译中文 PDF，它对agent的支持很友好。

## 快速开始

```bash
git clone https://github.com/Y006/arxiv-zh.git
cd arxiv-zh

uv sync
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
cp config.example.yaml config.yaml

uv run arxiv-zh --doctor --config config.yaml
uv run arxiv-zh 2501.12345 --config config.yaml
```

主入口是 `arxiv-zh`。`arx` 和 `arxiv-translate` 仍作为上游兼容入口保留。

## 输入和输出

输入：

```bash

>>> uv run arxiv-zh --help
                                                                     
 Usage: arxiv-zh [OPTIONS] [ARXIV_ID]                                
                                                                     
╭─ Arguments ───────────────────────────────────────────────────────╮
│   arxiv_id      [ARXIV_ID]  arXiv ID or URL                       │
╰───────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────╮
│ --config        PATH  Config YAML path. Defaults are used when    │
│                       omitted.                                    │
│ --doctor              Run environment checks and exit.            │
│ --help                Show this message and exit.                 │
╰───────────────────────────────────────────────────────────────────╯


```

输出目录：

```text
output/arxiv-<arxiv_id>/
├── source/
├── translated/main_zh.tex
├── pdf/main_zh.pdf
├── cache/
├── logs/translate.log
├── logs/compile.log
├── logs/compile_attempts.json
└── translation_report.md
```

## 配置

生产入口 `arxiv-zh` 只需要论文 ID 和一个配置文件。模型、输出目录、是否编译、并发数、字体、缓存和编译策略都写在同一个 YAML 中。

DeepSeek 密钥支持 shell 环境变量或项目根目录 `.env`，shell 环境变量优先。配置文件只保存环境变量名，不建议写入真实密钥。

```bash
export DEEPSEEK_API_KEY=sk-...
```

完整配置模板见 `config.example.yaml`。默认用户配置目录是 `~/.config/arxiv-translate/`；生产使用建议显式传 `--config config.yaml`。默认模型为 `deepseek-v4-flash`。

项目保留三枚默认 CJK 字体：`fonts/STSONG.TTF`、`fonts/STXIHEI.TTF`、`fonts/STKAITI.TTF`

如果配置里的 `fonts.auto_detect` 为 `true`，CLI 会优先扫描配置中的 `fonts.dir`，再回退到项目 `fonts/` 和系统字体。

编译默认按 TinyTeX 优先适配：配置会把常见 TinyTeX bin 目录加入 `PATH`，优先使用 `latexmk + xelatex`，遇到缺失 `.sty` / `.cls` 等文件时会尝试通过 `tlmgr search` 和 `tlmgr install` 自动安装缺包。首次编译需要下载包时可能较慢，`config.example.yaml` 已将编译超时放宽到 600 秒、缺包安装超时放宽到 1200 秒。