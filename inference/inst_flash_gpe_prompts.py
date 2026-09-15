from utils.config import LANG_MAP


def validate_prompt_type(prompt_type: str, max_candidates: int) -> None:
    if prompt_type != "markdown":
        raise ValueError("Instruct FlashGPE only supports prompt_type='markdown'")
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")


def build_candidate_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    max_candidates: int = 8,
    candidate_count: int | None = None,
    explicit_analysis: bool = False,
) -> str:
    validate_prompt_type("markdown", max_candidates)
    if candidate_count is not None:
        if not 2 <= candidate_count <= max_candidates:
            raise ValueError("candidate_count must be between 2 and max_candidates")
        max_candidates = candidate_count
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    analysis_instruction = (
        "First output a detailed step-by-step analysis under the exact heading "
        "`# Step-by-step Analysis`. Then output the candidate headings. The analysis "
        "must be outside the translations and must not contain candidate headings."
        if explicit_analysis else
        "Do not output analysis, code fences, or any text before or after the candidates."
    )
    prefix = "# Step-by-step Analysis\n\n[detailed analysis]\n\n" if explicit_analysis else ""
    return f"""Translate the source text from {source_lang} to {target_lang} and produce exactly {max_candidates} complete translation candidates.

The candidates will be compared by a separate post-editing stage. Make them genuinely diverse while keeping every candidate independently correct and complete. Preserve all source facts, entities, numbers, units, polarity, degree, time, and logical relationships. Diversity never permits mistranslation, omission, unsupported content, or commentary inside a translation.

{analysis_instruction} Use exactly the consecutive headings `# Candidate 1` through `# Candidate {max_candidates}`. Put one complete {target_lang} translation under each heading. A translation may contain multiple lines.

{prefix} 
# Candidate 1

first complete {target_lang} translation

# Candidate 2

second complete {target_lang} translation

Source text:
<source>
{source_text}
</source>"""


def build_post_edit_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidates: list[str],
    explicit_analysis: bool = False,
) -> str:
    if len(candidates) < 2:
        raise ValueError("post-editing requires at least 2 candidates")
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    rendered = "\n".join(
        f"Candidate {index}:\n```\n{value}\n```"
        for index, value in enumerate(candidates, 1)
    )
    output_format = (
        "First output a detailed step-by-step analysis under `# Step-by-step Analysis`, "
        "then output the final translation under `# Final Translation`."
        if explicit_analysis else
        "Think through the comparison internally, then output only `# Final Translation`."
    )
    visible_analysis = "# Step-by-step Analysis\n\n[detailed analysis]\n\n" if explicit_analysis else ""
    return f"""You are a translation post-editing agent.

Read the source and all numbered candidates. Select the best candidate or carefully combine them, correcting errors while preserving every source fact.

{output_format}
{visible_analysis}
# Final Translation

[one complete final translation in {target_lang}]

The final translation must be the last content in the response.

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{rendered}"""
