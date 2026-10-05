from utils.config import LANG_MAP


def _language_name(language: str) -> str:
    return LANG_MAP.get(language, language)


def build_key_points_prompt(source_lang: str, target_lang: str, source_text: str) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    return f'''You are a translation difficulty analysis agent working from {source_lang} to {target_lang}.

Identify only compact source spans that are likely to be mistranslated. Prefer one or two words, or a short phrase, rather than a clause or sentence. Include a span only when it presents a real semantic, terminological, idiomatic, cultural, numerical, ambiguity, register, or tone-related risk. Do not annotate ordinary easy wording, and do not translate the full source.

For every key point:
- Copy source_span exactly and contiguously from the source text.
- Explain briefly why this span is easy to mistranslate in this context.
- Give one or more natural {target_lang} translations. Every translation must preserve the source meaning, facts, entities, quantities, polarity, degree, and register.
- Keep translations meaningfully distinct; do not create superficial punctuation variants.

If there are no genuine translation difficulties, return an empty key_points list. Return only one valid JSON object with exactly this schema:
{{
  "key_points": [
    {{
      "source_span": "exact text copied from the source",
      "explanation": "short explanation of the translation risk",
      "translations": ["translation 1", "translation 2"]
    }}
  ]
}}

Source language: {source_lang}
Target language: {target_lang}
Source text:
<source>
{source_text}
</source>'''
