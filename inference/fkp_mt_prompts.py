from utils.config import LANG_MAP


def build_fkp_mt_prompt(source_lang: str, target_lang: str, source_text: str) -> str:
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    return f"""Translate this text from {source_lang} to {target_lang}. First identify compact source spans that are difficult to translate, explain the risks, and suggest faithful local translations. Then use these key points in context to produce one complete, faithful, natural translation of the whole text.

# Source

```
{source_text}
```"""
