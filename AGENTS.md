# AGENTS.md

## arxiv-zh agent 协助规则

本文件用于指导 agent 协助用户本地运行、配置和排错 `arxiv-zh`。

## 1. 优先运行环境诊断

当用户询问如何运行、配置或排错时，agent 应优先运行：

```bash
uv run arxiv-zh --doctor --config config.yaml
````

## 2. DeepSeek 和 API Key 规则

当前版本仅支持 DeepSeek。

项目通过根目录 `.env` 读取 DeepSeek API Key。agent 可以提示用户执行：

```bash
cp .env.example .env
```

然后让用户自行打开 `.env` 填写：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
```

agent 不应替用户填写、保存、回显真实 API Key，也不要要求用户把完整密钥发到聊天中。

## 3. TeX 环境规则

如果 `--doctor` 检查发现缺少 `latexmk`、`xelatex` 或 `lualatex`，agent 应判断当前环境可能无法正常编译 PDF。

TinyTeX 推荐补包链路是 `Rscript` + R 包 `tinytex`。如果 `--doctor` 发现 `Rscript` 或 `R tinytex` 不可用，agent 应说明当前会降级到 `latexmk + tlmgr` hook，推荐用户安装 R 包 `tinytex` 以启用官方 `tinytex::latexmk(..., install_packages = TRUE)` 自动补包能力。

如果 `--doctor` 中 `tlmgr repository` 或 `tlmgr 缺包搜索` 失败，agent 应优先判断是 CTAN 仓库或网络代理问题。agent 可以提示用户检查 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`，或手动切换到可访问的 CTAN 镜像，但不得未经确认永久执行 `tlmgr option repository ...`。

此时应建议用户安装 TinyTeX：

```text
https://yihui.org/tinytex/
```

只有在用户明确同意后，agent 才能协助安装 TinyTeX。不得未经确认自动安装 TeX 发行版。

## 4. 配置解释规则

agent可以帮助用户配置config.yaml文件，询问用户选用默认配置（直接复制example）还是要查看配置项自己配置？如果用户询问配置含义，或修改 `config.yaml` 后需要确认配置时，agent 应询问：需要我解释当前 config.yaml 的配置含义吗？如果用户需要，应读取当前配置，并用表格说明，下面是表格样式例子，具体要读取config.yaml文件。

| 配置项                           | 当前值  | 中文解释                           |
| ----------------------------- | ---- | ------------------------------ |
| `llm.sdk`                     | 读取配置 | 当前使用的 LLM SDK；当前版本仅支持 DeepSeek |
| `llm.models`                  | 读取配置 | 调用的 DeepSeek 模型                |

解释时应说明配置对速度、稳定性、翻译质量或编译结果的影响，不要只复述字段名。

## 5. 排错规则

编译失败、PDF 没生成或 LaTeX 报错时，优先查看：

```text
logs/compile_error_summary.md
logs/compile_attempts.json
logs/compile.log
```

翻译不完整、英文残留或 LaTeX 结构异常时，优先查看：

```text
metadata.json
logs/translate.log
```

`metadata.json` 中的 `run.download`、`run.translation`、`run.compilation` 会记录三个阶段是否完成。若翻译已完成但编译失败，优先使用：

```bash
uv run arxiv-zh <arxiv_id_or_url> --compile-only --config config.yaml
```

该命令只读取 `translated/main_zh.tex` 并重新编译，不应重新下载或重新翻译。

常见审计信息含义：

```text
placeholder audit   占位符可能被破坏
brace audit         LaTeX 花括号结构可能异常
line_end audit      行尾结构可能异常
untranslated audit  可能存在未翻译内容
```

如果出现：

```text
brace retry exhausted
```

表示部分 chunk 多次修复后仍未完全通过括号检查，程序会保留最后一次翻译结果。此时应检查对应 chunk 或最终 `.tex` 文件。

## 6. agent 行为边界

agent 应优先自己运行 `--doctor`、读取配置和查看日志，不要把机械检查步骤交给用户。

涉及以下行为时，必须先说明影响并等待用户确认：

* 安装 TinyTeX
* 修改 `.env`
* 修改 `config.yaml`
* 删除输出目录
* 清理缓存
* 重新编译
* 使用 `--compile-only` 重新编译
* 重新翻译

涉及 API Key 时，只说明 `.env` 配置方法，不索要、不保存、不回显完整密钥。
