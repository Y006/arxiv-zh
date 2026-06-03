# arxiv-zh 本地使用说明

`arxiv-zh` 是基于 `zcyisiee/arxiv-translate` 的本地自用入口。第一版只支持 DeepSeek，复用上游下载、LaTeX 解析、占位符保护、翻译缓存、断点续跑、重组和验证流程。

## 准备仓库

```bash
mkdir -p ~/Developer/arxiv-zh-work
cd ~/Developer/arxiv-zh-work

git clone https://github.com/zcyisiee/arxiv-translate.git
git clone https://gist.github.com/58821d077b4b54d80c20daaf970fb133.git gist-git-guide
git clone https://gist.github.com/de1348f3e212e61f87fd2cf47306b0ca.git gist-latexmk-makefile

cd arxiv-translate
git checkout -b local-deepseek-v1
```

## 创建环境

```bash
uv sync
```

或使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 设置 DeepSeek API Key

```bash
export DEEPSEEK_API_KEY=你的_key
```

不要把 API key 写进仓库文件、配置样例或日志。

## 检查环境

```bash
python scripts/check_env.py
```

重点确认：

- `DEEPSEEK_API_KEY` 存在。
- `xelatex` 可用。
- 优先有 `latexmk`；没有时会回退到多轮 `xelatex`。
- TinyTeX 常见路径存在：
  - `~/Library/TinyTeX/bin/universal-darwin/`
  - `/Library/TeX/texbin/`

## 快速测试

```bash
arxiv-zh 2501.12345 --provider deepseek --compile --max-chunks 2 --output ./output/test-paper
```

`--max-chunks 2` 只翻译前两个 chunk，适合验证 API、缓存、重组和编译链路。

默认模型是 `deepseek-chat`。需要切换模型时使用 `--model`：

```bash
arxiv-zh 2501.12345 --provider deepseek --model deepseek-reasoner --max-chunks 2 --output ./output/reasoner-test
```

## 本地字体

项目内 `fonts/` 是默认本地字体目录。`arxiv-zh` 会优先扫描这里的 `.ttf`、`.ttc`、`.otf` 文件，并把该目录加入 `OSFONTDIR`，供 XeLaTeX 和 `fontspec` 查找；如果本地目录没有可用字体，再回退系统字体。

当前样本字体的推荐 family name：

- main: `STSong`，可回退 `SimSun`
- sans: `STXihei`，可回退 `SimHei`
- mono: `STKaiti`，可回退 `KaiTi`

可以完全通过 CLI 控制字体：

```bash
arxiv-zh 2605.28486 \
  --provider deepseek \
  --compile \
  --max-chunks 2 \
  --output ./output/mag-vla-font-test \
  --font-dir ./fonts \
  --cjk-main-font STSong \
  --cjk-sans-font STXihei \
  --cjk-mono-font STKaiti
```

可用参数：

- `--font-dir PATH`：指定本地字体目录。
- `--cjk-main-font TEXT`：指定中文正文字体。
- `--cjk-sans-font TEXT`：指定中文无衬线字体。
- `--cjk-mono-font TEXT`：指定中文等宽/备用字体。
- `--font-auto / --no-font-auto`：开启或关闭自动字体检测。

优先级为 CLI 显式参数 → `--config` 配置 → 项目本地 `fonts/` 自动检测 → 系统字体自动检测。

## 完整翻译

```bash
arxiv-zh 2501.12345 --provider deepseek --compile --output ./output/2501.12345
```

不需要 PDF 时去掉 `--compile`：

```bash
arxiv-zh 2501.12345 --provider deepseek --output ./output/2501.12345
```

## 输出目录

成功时：

```text
output/<paper>/
├── source/
├── translated/
│   └── main_zh.tex
├── pdf/
│   └── main_zh.pdf
├── cache/
├── logs/
│   ├── translate.log
│   └── compile.log
└── translation_report.md
```

编译失败时仍会保留：

```text
output/<paper>/
├── translated/main_zh.tex
├── cache/
├── logs/
│   ├── translate.log
│   ├── compile.log
│   └── compile_error_summary.md
└── translation_report.md
```

## 排错

- 翻译失败：看 `logs/translate.log`。
- 编译失败：先看 `logs/compile_error_summary.md`，再看完整 `logs/compile.log`。
- 断点续跑：再次运行同一个 `--output` 目录，翻译状态和本地缓存会继续使用 `cache/` 下的文件。

## 验收

```bash
python scripts/check_env.py
uv run --with pytest --with pytest-asyncio python -m pytest
arxiv-zh 2501.12345 --provider deepseek --compile --max-chunks 2 --output ./output/test-paper
```
