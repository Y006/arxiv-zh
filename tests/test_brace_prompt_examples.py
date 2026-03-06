from arxiv_translate.rules.examples import load_builtin_examples
from arxiv_translate.translator.prompts import build_system_prompt


def test_system_prompt_contains_brace_preservation_examples():
    prompt = build_system_prompt()
    assert "转义花括号" in prompt
    assert r'输入：Return JSON: \{"name": "search"' in prompt
    assert r'输出：返回 JSON：\{"name": "search"' in prompt


def test_builtin_examples_include_brace_structure_case():
    examples = load_builtin_examples()
    matched = [
        ex
        for ex in examples
        if isinstance(ex, dict)
        and "Tool call payload" in str(ex.get("source", ""))
        and r'\{"name": "search"' in str(ex.get("source", ""))
    ]
    assert len(matched) == 1
    target = str(matched[0].get("target", ""))
    assert "工具调用载荷" in target
    assert r'\{"name": "search"' in target
