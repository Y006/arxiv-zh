# Configuration Guide

This guide covers all configuration options for arxiv-translate.

## Configuration Hierarchy

arxiv-translate uses a layered configuration system (later layers override earlier):

1. Built-in defaults: `src/arxiv_translate/defaults/config.yaml`
2. User config: `~/.config/arxiv-translate/config.yaml`
3. Command-line flags: override selected settings

## Creating a Configuration File

### User Configuration

Create `~/.config/arxiv-translate/config.yaml`:

```yaml
llm:
  # SDK: openai | openai-coding | anthropic | anthropic-coding | bailian | null
  # (null = direct HTTP)
  sdk: null
  # Model name or list (first item is used)
  models: openai/gpt-5-mini
  # API key (required when sdk is not null)
  key: ""
  # Optional custom endpoint
  endpoint: https://openrouter.ai/api/v1/chat/completions
  temperature: 0.1
  max_tokens: 4000

compilation:
  engine: xelatex
  timeout: 120
  clean_aux: true

paths:
  output_dir: output
  cache_dir: .cache

fonts:
  auto_detect: true
  # Optional manual overrides
  main: null
  sans: null
  mono: null

translation:
  custom_system_prompt: null
  custom_user_prompt: null
  preserve_terms: []
  quality_mode: standard
  examples_path: null

parser:
  extra_protected_environments: []
  extra_translatable_environments: []
```

## Configuration Options

### LLM Settings

```yaml
llm:
  sdk: openai
  models: gpt-4o-mini
  key: "sk-..."
  endpoint: null
  temperature: 0.1
  max_tokens: 4000
```

Notes:
- `sdk` must be `openai`, `openai-coding`, `anthropic`, `anthropic-coding`, `bailian`, or `null`.
- `models` can be a string or a list; lists use the first item.
- When `sdk` is not null, `key` is required (or provide `--key` on the CLI).
- Ark endpoint auto-routing is enabled when host matches `ark.*.volces.com`.
  Use openai-style sdk (`openai`, `openai-coding`, or `null`) + Ark endpoint.
  `sdk=ark` is no longer supported.

### Compilation Settings

```yaml
compilation:
  engine: xelatex
  timeout: 120
  clean_aux: true
```

### Path Settings

```yaml
paths:
  output_dir: output
  cache_dir: .cache
```

### Font Settings

```yaml
fonts:
  auto_detect: true
  main: null
  sans: null
  mono: null
```

### Translation Settings

```yaml
translation:
  custom_system_prompt: null
  custom_user_prompt: null
  preserve_terms: []
  quality_mode: standard
  examples_path: null
```

### Parser Settings

```yaml
parser:
  extra_protected_environments: []
  extra_translatable_environments: []
```

### Glossary File (separate)

The glossary is loaded from `~/.config/arxiv-translate/glossary.yaml` and is not part of `config.yaml`.
See [Custom Rules & Glossary](custom-rules.md).

## Command-Line Overrides

Selected settings can be overridden via command line:

```bash
# Override SDK, model, key, endpoint
arx translate paper.tex --sdk anthropic --model claude-3-sonnet-20240229 --key "sk-..."
arx translate paper.tex --endpoint https://openrouter.ai/api/v1/chat/completions

# Override output directory
arx translate paper.tex --output-dir ./my-output/

# Control concurrency
arx translate paper.tex --concurrency 20

# Enable high-quality translation mode
arx translate https://arxiv.org/abs/2301.07041 --high-quality

# Provide custom abstract for context
arx translate https://arxiv.org/abs/2301.07041 --high-quality --abstract "This paper proposes..."

# Skip compilation or keep source
arx translate paper.tex --no-compile --keep-source
```

## Complete Example Configuration

```yaml
# ~/.config/arxiv-translate/config.yaml - Full example

llm:
  sdk: openai
  models: gpt-4o-mini
  key: "sk-..."
  endpoint: null
  temperature: 0.1
  max_tokens: 4000

compilation:
  engine: xelatex
  timeout: 120
  clean_aux: true

paths:
  output_dir: ~/Documents/translations
  cache_dir: ~/.config/arxiv-translate/cache

fonts:
  auto_detect: true
  main: null
  sans: null
  mono: null

translation:
  custom_system_prompt: null
  custom_user_prompt: null
  preserve_terms: []
  quality_mode: standard
  examples_path: null

parser:
  extra_protected_environments: []
  extra_translatable_environments: []
```

## Configuration Validation

arx validates configuration on startup. Invalid settings will produce clear error messages:

```
Error: Invalid configuration
  llm.sdk: sdk must be 'openai', 'openai-coding', 'anthropic', 'anthropic-coding', 'bailian', or None, got 'invalid'
  llm.models: models cannot be empty
```

## Next Steps

- [Custom Rules & Glossary](custom-rules.md) - Create custom translation rules
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
