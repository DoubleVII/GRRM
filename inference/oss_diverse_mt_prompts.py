import json

from utils.config import LANG_MAP


def _language_name(language: str) -> str:
    return LANG_MAP.get(language, language)


def build_divergent_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    min_candidates: int = 3,
    max_candidates: int = 6,
    prompt_type: str = "json",
) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    shared_task = f"""You are performing the divergent first stage of a translation task from {source_lang} to {target_lang}.

Analyze the source, divide it into ordered, non-overlapping translation units, and propose diverse local translation candidates for every unit. A unit may be a clause, idiom, term, name, discourse marker, or other span that should be translated together.

The purpose of this stage is exploration, not selection:
- Cover all lexical content in the source and preserve its order. Boundary punctuation may remain between units, but no words or meaning may be omitted.
- Normally give {min_candidates} to {max_candidates} genuinely distinct candidates per unit. Explore plausible differences in sense, register, syntax, idiom handling, terminology, and target-language naturalness.
- Do not create superficial variants that differ only in punctuation or one interchangeable function word.
- Give fewer than {min_candidates} candidates only when there is truly no meaningful alternative; briefly state why in the unit analysis.
- A candidate must translate only its own source span, not the entire source.
- Do not choose or compose a final translation in this stage.
"""

    codeblock_source_section = f"""Source text:
```
{source_text}
```"""

    if prompt_type == "codeblock":
        return f"""{shared_task}

Output format requirements:
1. First output a brief step-by-step analysis of the source's meaning, context, tone, ambiguities, and translation constraints.
2. After that, output the exploratory segmentation and candidates as a single Markdown code block.
3. Inside the code block, organize the segments, local candidates, and their distinct interpretation or style angles in any clear natural-language format you find useful. You do not need to follow JSON, XML, or any fixed schema.
4. Do not put a final full translation in the code block or anywhere else.
5. Do not output anything after the closing code fence.

{codeblock_source_section}"""

    if prompt_type != "json":
        raise ValueError("prompt_type must be one of: json, codeblock")

    return f"""{shared_task}

Return only one valid JSON object. Do not use Markdown fences or add text before or after it. Follow this schema:
{{
  "source_analysis": "brief global analysis of meaning, context, tone, ambiguities, and constraints",
  "segments": [
    {{
      "segment_id": 1,
      "source_span": "an exact contiguous lexical span copied from the source (boundary punctuation may be excluded)",
      "analysis": "meaning, ambiguity, role, and relevant constraints",
      "candidates": [
        {{"translation": "local {target_lang} translation", "angle": "what interpretation or style this explores"}}
      ]
    }}
  ]
}}

Source text:
<source>
{source_text}
</source>"""


def build_convergent_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    divergent_result,
) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    if isinstance(divergent_result, dict):
        divergent_context = json.dumps(divergent_result, ensure_ascii=False, indent=2)
    elif isinstance(divergent_result, str) and divergent_result.strip():
        divergent_context = divergent_result.strip()
    else:
        raise ValueError("divergent_result must be a non-empty dict or string")
    return f"""You are performing the convergent second stage of a translation task from {source_lang} to {target_lang}.

The source and a divergent analysis with local candidates are provided below. Critically compare the candidates against the source, reject mistranslations, and select, edit, or combine the strongest choices into one coherent translation. Resolve cross-segment dependencies such as terminology, pronouns, tense, register, syntax, and discourse flow. The candidate set is exploratory evidence, not an authority: repair it whenever needed.

Requirements:
- Translate the complete source faithfully without additions or omissions.
- Produce a natural, internally consistent {target_lang} text, not a stitched list of fragments.
- Preserve the source's intended tone and formatting where appropriate.
- Output only the final translation between the exact tags below. Do not include analysis, labels, Markdown fences, or any text outside the tags.

<source>
{source_text}
</source>

<divergent_analysis>
{divergent_context}
</divergent_analysis>

<final_translation>...</final_translation>"""


def build_direct_prompt(source_lang: str, target_lang: str, source_text: str) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    return f"""Translate the complete source text from {source_lang} into {target_lang}. Preserve its meaning, tone, and relevant formatting, and make the translation natural and internally consistent.

Output only the translation between the exact tags below. Do not include analysis, labels, Markdown fences, or any text outside the tags.

<source>
{source_text}
</source>

<final_translation>...</final_translation>"""
