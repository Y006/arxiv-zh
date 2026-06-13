# AGENTS.md

## arxiv-zh 本地使用速记

- 当前本地主入口是 `arxiv-zh`；在开发环境里优先使用 `uv run arxiv-zh ...`。
- 生产使用建议显式传入配置文件：
  ```bash
  uv run arxiv-zh <arxiv_id_or_url> --config config.yaml
  ```
- 本地首次使用：
  ```bash
  uv sync
  cp .env.example .env
  cp config.example.yaml config.yaml
  ```
- DeepSeek 密钥可写入项目根目录 `.env`，或通过 shell 环境变量提供：
  ```bash
  export DEEPSEEK_API_KEY=sk-...
  ```
- 默认只支持 DeepSeek；默认模型以 `config.yaml` / `config.example.yaml` 中的 `llm.models` 为准。
- 快速验证命令：
  ```bash
  uv run arxiv-zh 2501.12345 --config config.yaml
  ```
- URL 输入也会先解析为 arXiv ID：
  ```bash
  uv run arxiv-zh https://arxiv.org/html/2410.24164v1 --config config.yaml
  ```
- 默认输出目录格式：
  ```text
  output/arxiv-<arxiv_id>/
  ```
- 编译策略：
  - 优先 `latexmk + xelatex`
  - 失败后会进行安全修复并回退到 `lualatex`
  - `pdflatex` 默认不用来编译含中文文档，除非配置显式开启
- 编译诊断优先看：
  - `logs/compile_attempts.json`
  - `logs/compile_error_summary.md`
  - `logs/compile.log`
- 如果自动修复后的译文成功编译，`translated/` 下可能额外出现 `main_zh.before_compile.tex` 备份文件。
