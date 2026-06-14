# arxiv-zh

`arxiv-zh` 是一个面向本地使用的 arXiv LaTeX 论文中文翻译 CLI。它下载 arXiv 源码、解析可翻译文本块、调用 DeepSeek 翻译、重组 LaTeX，并可用 XeLaTeX 编译中文 PDF，它对agent的支持很友好。

## 快速开始

推荐先安装 [Miniforge](https://github.com/conda-forge/miniforge)，使用 conda-forge 的 `mamba` 管理本地运行环境。

```bash
git clone https://github.com/Y006/arxiv-zh.git
cd arxiv-zh

mamba env create -f environment.yml
conda activate arxiv-zh
uv pip install -e .

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
cp config.example.yaml config.yaml

arxiv-zh --doctor --config config.yaml
arxiv-zh 2501.12345 --config config.yaml
arxiv-zh 2501.12345 --compile-only --config config.yaml
```

主入口是 `arxiv-zh`。`arx` 和 `arxiv-translate` 仍作为上游兼容入口保留。

## 输入和输出

输入：

```bash

>>> arxiv-zh --help
                                                                     
 Usage: arxiv-zh [OPTIONS] [ARXIV_ID]                                
                                                                     
╭─ Arguments ───────────────────────────────────────────────────────╮
│   arxiv_id      [ARXIV_ID]  arXiv ID or URL                       │
╰───────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────╮
│ --config        PATH  Config YAML path. Defaults are used when    │
│                       omitted.                                    │
│ --doctor              Run environment checks and exit.            │
│ --compile-only        Compile existing translated/main_zh.tex and │
│                       exit.                                       │
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
└── metadata.json
```

`metadata.json` 记录两类信息：

- `arxiv`：规范化后的 arXiv ID、base ID、版本、输入来源和输出目录，用于避免同一篇文章被不同 URL 识别成多个目录。
- `run`：`download`、`translation`、`compilation` 三个阶段的状态。普通重跑会跳过已完成阶段；编译失败后可用 `--compile-only` 只重试编译。

## 配置

生产入口 `arxiv-zh` 只需要论文 ID 和一个配置文件。模型、输出目录、是否编译、并发数、字体、缓存和编译策略都写在同一个 YAML 中。

DeepSeek 密钥支持 shell 环境变量或项目根目录 `.env`，shell 环境变量优先。配置文件只保存环境变量名，不建议写入真实密钥。

```bash
export DEEPSEEK_API_KEY=sk-...
```

完整配置模板见 `config.example.yaml`。默认用户配置目录是 `~/.config/arxiv-translate/`；生产使用建议显式传 `--config config.yaml`。默认模型为 `deepseek-v4-flash`。

环境分工：

- `conda` / `mamba` 管统一运行环境：Python、`uv`、R、R 包 `tinytex`、基础 LaTeX 命令。
- `uv pip` 管 Python 项目安装：`uv pip install -e .`。
- R 包 `tinytex` 管官方 LaTeX 编译 wrapper 和自动补包能力。
- TinyTeX / `tlmgr` 管 TeX 包，例如 `tlmgr install tex-gyre`。

项目保留三枚默认 CJK 字体：`fonts/STSONG.TTF`、`fonts/STXIHEI.TTF`、`fonts/STKAITI.TTF`

如果配置里的 `fonts.auto_detect` 为 `true`，CLI 会优先扫描配置中的 `fonts.dir`，再回退到项目 `fonts/` 和系统字体。项目自带的 `.TTF` 字体可直接通过文件路径注入 LaTeX，不要求系统安装 `fontconfig`。

编译默认按 TinyTeX 优先适配：配置会把常见 TinyTeX bin 目录加入 `PATH`。当 `Rscript` 和 R 包 `tinytex` 可用时，优先调用官方 `tinytex::latexmk(..., install_packages = TRUE)` 自动安装缺失 TeX 包；如果官方 wrapper 不可用，`auto` 只能降级为普通 `latexmk` 编译，不会再由项目自写 `tlmgr` 补包。首次补包可能较慢，`config.example.yaml` 已将单次编译超时放宽到 600 秒、R tinytex 总编译等待放宽到 7200 秒。

`arxiv-zh --doctor --config config.yaml` 会检查当前 conda 环境、`Rscript`、R 包 `tinytex`、`tlmgr repository`、`tlmgr search --global --file /tgpagella.sty` 和代理环境。项目不会自动安装 TinyTeX、不会自动换源，也不会永久修改 `tlmgr option repository`。若 TinyTeX 本体缺失，请按 doctor 提示运行：

```bash
Rscript -e 'tinytex::install_tinytex()'
```

若 `tlmgr` 网络检查失败，请配置代理或手动切换到可访问的 CTAN 镜像。
