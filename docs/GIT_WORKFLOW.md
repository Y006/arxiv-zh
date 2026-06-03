# arxiv-zh 本地开发 Git 工作流

本规范用于 `arxiv-zh` 自用第一版开发。目标是让每一次改动都容易回看、回滚和验证。

## 分支

当前自用开发分支固定为：

```bash
local-deepseek-v1
```

开始开发前确认：

```bash
git branch --show-current
```

如果不在该分支：

```bash
git checkout local-deepseek-v1
```

## 提交

- 每完成一个独立任务单独 commit。
- commit message 使用中文。
- 不要把多个大功能堆在一次提交。
- 推荐第一行格式：

```text
[YYYY-MM-DD HH:MM] 简短说明
```

示例：

```bash
git add src/arxiv_translate/translator/deepseek_provider.py tests/test_deepseek_provider.py
git commit -m "[2026-06-03 21:30] 增加 DeepSeek 翻译 Provider"
```

## 建议提交边界

- DeepSeek provider 与 factory 测试。
- `arxiv-zh` CLI 入口与输出目录。
- LaTeX 编译日志与错误摘要。
- 环境检查脚本。
- 本地使用文档与配置示例。

## 稳定标签

第一版本地验收稳定后打 tag：

```bash
git tag v1-local-ready
```

如需推送到个人远端：

```bash
git push origin local-deepseek-v1
git push origin v1-local-ready
```
