from inference.inst_flash_gpe_prompts import build_candidate_prompt
from utils.config import LANG_MAP, candidate_identifiers


def validate_candidate_count(candidate_count: int) -> None:
    if not 2 <= candidate_count <= len(candidate_identifiers):
        raise ValueError(
            f"candidate_count must be between 2 and {len(candidate_identifiers)}"
        )


def build_gqm_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidates: list[str],
    explicit_analysis: bool = False,
) -> str:
    validate_candidate_count(len(candidates))
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    identifiers = candidate_identifiers[:len(candidates)]
    rendered = "\n".join(
        f"Translation {identifier}:\n```\n{candidate}\n```"
        for identifier, candidate in zip(identifiers, candidates)
    )
    score_example = ", ".join(
        f"{identifier}: {max(0, 9 - index)}"
        for index, identifier in enumerate(identifiers)
    )
    ranking_example = " > ".join(identifiers)
    analysis_instruction = (
        "First output a detailed comparison under the exact heading "
        "`# Step-by-step Analysis`. "
        if explicit_analysis
        else "Think through the comparison internally. Do not output analysis. "
    )
    visible_analysis = (
        "# Step-by-step Analysis\n\n[detailed comparison]\n\n"
        if explicit_analysis else ""
    )
    return f"""Given a source text in {source_lang} and multiple translation candidates in {target_lang}, rank and score every candidate by translation quality.

Judge faithfulness, completeness, terminology, grammar, fluency, and style. Assign each candidate an integer score from 0 to 10. Candidates with the same score must be tied with `=` in the ranking; candidates with different scores must be ordered from highest to lowest with `>`.

{analysis_instruction}Then output exactly the two sections shown below. Include every candidate identifier exactly once in each section. The scores line must use plain ASCII text in the format `{score_example}`. Do not use code fences around the ranking or scores, and do not output any content after the scores line.

{visible_analysis}# Final Ranking

{ranking_example}

# Scores

{score_example}

Source text:
```
{source_text}
```

{rendered}"""


__all__ = ["build_candidate_prompt", "build_gqm_prompt", "validate_candidate_count"]
