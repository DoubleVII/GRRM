from utils.config import LANG_MAP


def build_translation_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    return f"""Translate the complete source text from {source_lang} into {target_lang}.

First analyze the source in detail, including its meaning, context, tone, ambiguities, terminology, entities, numbers, logical relationships, and relevant formatting. Then produce one faithful, complete, natural, and internally consistent translation. Do not add unsupported content or omit source information.

Output exactly this Markdown structure:
# Step-by-step Analysis

[detailed step-by-step analysis]

# Final Translation

```
[one complete final translation in {target_lang}]
```


The final translation must be the last content in the response and enclosed in code blocks.

Source text:

```
{source_text}
```"""
