from utils.config import LANG_MAP
from inference.prompts import (
    candidate_prompt,
    oss_group_post_edit_prompt_templates,
)


PROMPT_TYPES = {"markdown", "adaptive", "fixed_4", "fixed_16"}


def build_post_edit_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidates: list[str],
    notes=None,
) -> str:
    if notes is not None:
        raise ValueError("FlashGPE does not support post-edit notes")
    if len(candidates) < 2:
        raise ValueError("FlashGPE post-editing requires at least 2 candidates")
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    rendered = "".join(
        candidate_prompt.format(index + 1, value)
        for index, value in enumerate(candidates)
    )
    return f"""You are a translation post-editing agent.

Read the source and all numbered candidates. Select the best candidate or carefully combine them, correcting errors while preserving every source fact.

Output exactly this Markdown structure:
# Step-by-step Analysis

[step-by-step analysis]

# Final Translation

[one complete final translation in {target_lang}]

Do not use code fences. Do not add text outside these two sections.

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{rendered}"""


def validate_prompt_type(prompt_type: str, max_candidates: int) -> None:
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(
            "prompt_type must be one of: markdown, adaptive, fixed_4, fixed_16"
        )
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    if prompt_type == "fixed_4" and max_candidates != 4:
        raise ValueError("fixed_4 prompt_type requires max_candidates=4")
    if prompt_type == "fixed_16" and max_candidates != 16:
        raise ValueError("fixed_16 prompt_type requires max_candidates=16")


def build_candidate_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    max_candidates: int = 4,
    prompt_type: str = "markdown",
    candidate_count: int | None = None,
) -> str:
    validate_prompt_type(prompt_type, max_candidates)
    if candidate_count is not None:
        if not 2 <= candidate_count <= max_candidates:
            raise ValueError("candidate_count must be between 2 and max_candidates")
        max_candidates = candidate_count
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    if prompt_type == "markdown":
        return f"""Translate the source text from {source_lang} to {target_lang} and produce exactly {max_candidates} complete translation candidates.

The candidates will be compared by a separate post-editing stage. Make them genuinely diverse while keeping every candidate independently correct and complete. Preserve all source facts, entities, numbers, units, polarity, degree, time, and logical relationships. Diversity never permits mistranslation, omission, unsupported content, or commentary inside a translation.

Output only the following Markdown structure. Use exactly the consecutive headings `# Candidate 1` through `# Candidate {max_candidates}`. Put one complete {target_lang} translation under each heading. A translation may contain multiple lines. Do not add any other headings, analysis, code fences, or text before or after the candidates.

# Candidate 1

first complete {target_lang} translation

# Candidate 2

second complete {target_lang} translation

Source text:
<source>
{source_text}
</source>"""
    if prompt_type == "fixed_4":
        return f"""Translate the source text from {source_lang} to {target_lang} and produce exactly 4 complete translation candidates in one response.

The candidates will be compared by a separate post-editing stage. Make them genuinely diverse while keeping every candidate independently correct and complete:
- Explore meaningful differences in wording, syntax, register, idiom handling, and information structure.
- Do not create superficial variants that differ only in punctuation or interchangeable function words.
- Every candidate must preserve all source facts, entities, numbers, units, polarity, degree, time, and logical relationships.
- Diversity never permits deliberate mistranslation, omission, unsupported content, or commentary inside a translation.
- Each entry must contain only one full translation in {target_lang}.

Return only one valid JSON object with exactly this schema and exactly 4 strings. Do not use Markdown fences or add text before or after the JSON:
{{
  "translations": [
    "{target_lang} translation 1",
    "{target_lang} translation 2",
    "{target_lang} translation 3",
    "{target_lang} translation 4"
  ]
}}

Source text:
<source>
{source_text}
</source>"""
    if prompt_type == "adaptive":
        count_instruction = (
            f"produce between 2 and {max_candidates} complete translation candidates"
        )
        count_guidance = f"""Decide how many candidates are useful based on the source's length, complexity, ambiguity, idioms, terminology, register, and the number of genuinely meaningful translation choices:
- For a simple or very short source with few meaningful choices, normally return 2 to 4 candidates.
- For a moderately complex source, return more candidates only when they explore real translation choices.
- For a long, difficult, or highly ambiguous source, return as many as useful, but never more than {max_candidates}.
- The maximum is a ceiling, not a target. Do not pad the list with weak or superficial variants."""
        output_count = f"at least 2 and at most {max_candidates} strings"
    else:
        count_instruction = "produce between 2 and 16 complete translation candidates"
        count_guidance = """Decide how many candidates are useful based on the source's length, complexity, ambiguity, idioms, terminology, register, and the number of genuinely meaningful translation choices. Use these ranges as a calibration guide:
- Return 2 to 3 candidates only for exceptionally short or trivial sources with almost no meaningful variation.
- For an ordinary simple source, normally return 6 to 8 candidates.
- For a moderately complex source, normally return 8 to 12 candidates.
- For a long, difficult, terminology-heavy, stylistically rich, or highly ambiguous source, normally return 12 to 16 candidates.
- If the source does not clearly fit a category, prefer 8 candidates rather than a smaller set.
- The maximum of 16 is a ceiling, not a target. Do not pad the list with weak or superficial variants."""
        output_count = "at least 2 and at most 16 strings"

    return f"""Translate the source text from {source_lang} to {target_lang} and {count_instruction} in one response.

{count_guidance}

The candidates will be compared by a separate post-editing stage. Keep every candidate independently correct and complete:
- Explore meaningful differences in wording, syntax, register, idiom handling, and information structure.
- Do not create superficial variants that differ only in punctuation or interchangeable function words.
- Every candidate must preserve all source facts, entities, numbers, units, polarity, degree, time, and logical relationships.
- Diversity never permits deliberate mistranslation, omission, unsupported content, or commentary inside a translation.
- Each entry must contain only one full translation in {target_lang}.

Return only one valid JSON object with this schema. The translations array must contain {output_count}. Do not use Markdown fences or add text before or after the JSON:
{{
  "translations": [
    "first complete {target_lang} translation",
    "second complete {target_lang} translation"
  ]
}}

Source text:
<source>
{source_text}
</source>"""
