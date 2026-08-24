from utils.config import LANG_MAP, candidate_identifiers
from typing import Optional

Output_example = {
    "score": "Output the scores on the last line, for example: `A: 4, B: 9, C: 7, D: 9`.",
    "ranking": "Output the rankings in descending order on the last line, for example: `B > A = D > C`.",
    "ranking_score": "At the end section, first output the rankings in descending order, for example: `B > A = D > C`. Then, on the last line, output the scores, for example: `B: 9, A: 7, D: 7, C: 2`.",
}


Task_format = {
    "score": "Finally, score the candidates with integer scores on a scale from 0 to 10.",
    "ranking": "Finally, rank the candidates in order of quality from best to worst.",
    "ranking_score": "Finally, rank and score the candidates with integer scores on a scale from 0 to 10.",
}


GQM_prompt_template = """Given a source text in {source_lang} and multiple translation candidates in {target_lang}. Perform a step by step analysis and comparison of the translation quality for the candidates. {task_prompt}

Source text:
```
{source_text}
```

{candidate_prompts}{reference_prompt}{notes_prompt}"""


notes_prompt_template = """

You may refer to the following notes if necessary.

Notes:
```
{notes}
```
"""


reference_prompt_template = """

You may refer to the following reference if necessary.

{} reference:
```
{}
```
"""


candidate_prompt = """Translation {}:
```
{}
```
"""


FFGPE_PROMPT_TYPES = {"markdown", "adaptive", "fixed_4", "fixed_16"}


def validate_ffgpe_prompt_type(prompt_type: str, max_candidates: int) -> None:
    if prompt_type not in FFGPE_PROMPT_TYPES:
        raise ValueError(
            "prompt_type must be one of: markdown, adaptive, fixed_4, fixed_16"
        )
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")
    if prompt_type == "fixed_4" and max_candidates != 4:
        raise ValueError("fixed_4 prompt_type requires max_candidates=4")
    if prompt_type == "fixed_16" and max_candidates != 16:
        raise ValueError("fixed_16 prompt_type requires max_candidates=16")


def build_ffgpe_prompt(
    source_lang: str,
    target_lang: str,
    source_text: str,
    max_candidates: int = 4,
    prompt_type: str = "fixed_4",
) -> str:
    """Build the shared prompt used by FFGPE SFT, RL, and inference."""
    validate_ffgpe_prompt_type(prompt_type, max_candidates)
    source_lang = LANG_MAP.get(source_lang, source_lang)
    target_lang = LANG_MAP.get(target_lang, target_lang)
    exact_count = prompt_type in {"markdown", "fixed_4", "fixed_16"}
    count = (
        f"exactly {max_candidates}"
        if exact_count
        else f"as many as useful, up to {max_candidates}"
    )
    return f"""Translate this text from {source_lang} to {target_lang}. First produce {count} meaningfully different complete translations, keeping each one faithful and natural. Then review those candidates, correct their errors, and produce the best final translation.

For the candidate section, output exactly consecutive Markdown headings `# Candidate 1` through `# Candidate {max_candidates}`, with one complete translation under each heading. Candidates may contain multiple lines. Do not use code fences or add unrelated headings.

Source:
{source_text}"""


def get_task_prompt(prompt_format: str, add_example: bool = False):
    if prompt_format not in Task_format:
        raise ValueError(f"prompt_format must be one of {Task_format.keys()}")
    task_prompt = Task_format[prompt_format]
    if add_example:
        task_prompt += f" {Output_example[prompt_format]}"
    return task_prompt


def build_notes_prompt(notes: str = None) -> str:
    if notes is None:
        return ""
    notes = notes.strip()
    if not notes:
        return ""
    return notes_prompt_template.format(notes=notes)


def build_reference_prompt(ref_text: str = None, ref_lang: str = None) -> str:
    if ref_text is None or ref_lang is None:
        return ""
    ref_text = ref_text.strip()
    ref_lang = ref_lang.strip()
    if not ref_text or not ref_lang:
        return ""
    if len(ref_lang) == 2 and ref_lang in LANG_MAP:
        ref_lang = LANG_MAP[ref_lang]
    return reference_prompt_template.format(ref_lang, ref_text)


def get_GQM_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    prompt_format: str,
    add_example: bool = False,
    notes: str = None,
    ref_text: str = None,
    ref_lang: str = None,
):
    if len(source_lang) == 2 and source_lang in LANG_MAP:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2 and target_lang in LANG_MAP:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")

    task_prompt = get_task_prompt(prompt_format, add_example)

    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    notes_prompt = build_notes_prompt(notes)
    reference_prompt = build_reference_prompt(ref_text, ref_lang)

    return GQM_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        task_prompt=task_prompt,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
        reference_prompt=reference_prompt,
        notes_prompt=notes_prompt,
    )


GQMPE_prompt_template = """Given a source text in {source_lang} and multiple translation candidates in {target_lang}. Perform a step by step analysis and comparison of the translation quality for the candidates. {task_prompt} Then provide a detailed post-edit analysis and a final improved translation in {target_lang}, selecting, combining, or editing the candidates as appropriate.

Source text:
```
{source_text}
```

{candidate_prompts}"""


def get_GQMPE_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    prompt_format: str = "ranking_score",
):
    if len(source_lang) == 2 and source_lang in LANG_MAP:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2 and target_lang in LANG_MAP:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")

    task_prompt = get_task_prompt(prompt_format, add_example=False)
    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_text)
        for i, mt_text in enumerate(mt_texts)
    )
    return GQMPE_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        task_prompt=task_prompt,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
    )


get_GQM_with_notes_prompt = get_GQM_prompt




oss_group_post_edit_with_notes_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and strictly following the provided translation notes.

Your priority is to respect the notes and make the smallest necessary edits.
Do not be creative for its own sake.
Do not add content that is not supported by the source text or the notes.
Do not ignore the notes when choosing between candidate phrasings.

You will receive:
- source language
- target language
- source text
- 1 to 4 translation candidates
- translation notes

Some candidate fields may be empty or missing meaningful content. Ignore unusable candidates.

Your job:
1. Read the source text and the notes carefully.
2. Compare the translation candidates against the source text and the notes.
3. Identify which candidate is the best base, or whether a careful combination of candidates is needed.
4. Perform minimal, targeted post-editing to produce the best final translation.
5. Prioritize correctness, faithfulness, fluency, and compliance with the notes.
6. If the notes specify terminology, tone, named-entity handling, ambiguity, register, formatting, or other constraints, follow them closely.
7. Do not produce multiple alternative translations.
8. Do not output back-translations, annotations, or extra commentary outside the required format.

Be especially careful about:
- preserving meaning from the source text
- following note-specified terminology or interpretations
- preserving tone/register/style when mentioned in the notes
- not introducing unsupported improvements

Output format requirements:
1. Output the final post-edited translation as a single code block.
2. Inside that code block, output only the final translation and nothing else.

Now perform the task on the following input.

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{candidate_prompts}

Notes:
```
{notes}
```
"""

oss_group_post_edit_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and selecting or carefully combining the best parts of the candidates.


Your job:
1. Read the source text carefully.
2. Identify which candidate is the best base, or whether a careful combination of candidates is needed.
3. Perform minimal, targeted post-editing to produce the best final translation.
4. Provide a step-by-step analysis in the output before the final translation.


Output format requirements:
1. First output your step-by-step analysis in Markdown.
2. After that, output the final post-edited translation as a single Markdown code block. Inside that code block, output only the final translation and nothing else.

Output format:
# Step-by-step Analysis
[analysis/breakdown]

# Final post-edited translation
```
[your translation]
```

---

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{candidate_prompts}
"""


def get_oss_group_post_edit_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    notes: str = None,
    ):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")


    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    if notes is not None:
        return oss_group_post_edit_with_notes_prompt_templates.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            candidate_prompts=candidate_prompts,
            notes=notes,
        )
    else:
        return oss_group_post_edit_prompt_templates.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            candidate_prompts=candidate_prompts,
        )




oss_GQM_post_edit_completion_prompt_template = """You are completing an existing translation-quality response with a post-editing decision.

The source text, translation candidates, and the existing GQM response are provided below. The existing GQM response already contains the detailed candidate evaluation and scores. Treat it as the completed first part of the response: do not repeat, rewrite, summarize, or rescore it.

Use the existing GQM analysis to decide the best post-editing strategy for the specific input. You may preserve the strongest candidate, combine the best-supported parts of multiple candidates, or rewrite a flawed passage when necessary. Produce the most accurate and natural final translation supported by the source text.

Prefer minimal edits when a candidate is already accurate, but do not preserve a candidate's wording when doing so would reduce accuracy, completeness, or fluency. Do not introduce information unsupported by the source text. The source text remains authoritative if any candidate or evaluation detail is inconsistent with it.

Language and style requirements:
- Write the post-edit analysis in English, regardless of the source and target languages.
- When referring to specific words or passages from the source text or translation candidates, preserve those quoted passages in their original language; the surrounding analysis must remain in English.
- The final translation is the only section that must be in {target_lang}.
- Match the terminology, tone, and level of detail of the existing GQM analysis.
- Do not omit the post-edit analysis.
- Do not provide alternative final translations.

Output exactly these two sections and nothing before or after them:

# Post-edit Analysis
[detailed analysis of the selection and edits]

# Final post-edited translation
```
[one final translation in {target_lang}]
```

---

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{candidate_prompts}
<existing_gqm_response>
{gqm_response}
</existing_gqm_response>
"""


def format_GQM_scores(scores, candidate_count: int) -> str:
    if len(scores) != candidate_count:
        raise ValueError(
            f"Expected {candidate_count} GQM scores, but got {len(scores)}."
        )
    identifiers = candidate_identifiers[:candidate_count]
    return ", ".join(
        f"{identifier}: {int(score)}"
        for identifier, score in zip(identifiers, scores)
    )


def get_oss_GQM_post_edit_completion_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    gqm_analysis,
    gqm_scores,
):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")
    if not isinstance(gqm_analysis, str) or not gqm_analysis.strip():
        raise ValueError("GQM analysis must be a non-empty string.")

    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_text)
        for i, mt_text in enumerate(mt_texts)
    )
    score_line = format_GQM_scores(gqm_scores, len(mt_texts))
    gqm_response = f"{gqm_analysis.strip()}\n\n### Scores:\n\n{score_line}"

    return oss_GQM_post_edit_completion_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
        gqm_response=gqm_response,
    )



group_post_edit_with_notes_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and strictly following the provided translation notes.

Your priority is to respect the notes and make the smallest necessary edits.

---

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{candidate_prompts}

Notes:
```
{notes}
```
"""




group_post_edit_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and selecting or carefully combining the best parts of the candidates.

---

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

{candidate_prompts}
"""

def get_group_post_edit_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    notes: str = None,
    ):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    # if len(mt_texts) == 1:
    #     raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")


    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    if notes is not None:
        return group_post_edit_with_notes_prompt_templates.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            candidate_prompts=candidate_prompts,
            notes=notes,
        )
    else:
        return group_post_edit_prompt_templates.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            candidate_prompts=candidate_prompts,
        )


post_edit_prompt_templates = """You are a translation post-editing agent.

Your task is to improve a given translation candidate using the source text and the provided translation notes. The notes come from an earlier translation-prep stage and should be treated as the primary editing guidance.

Input will contain:
- source language
- target language
- source text
- translation candidate
- translation notes

Your job:
1. Read the source text, translation candidate, and notes carefully.
2. Evaluate step by step whether the translation candidate already satisfies the notes.
3. If the candidate translation already follows the notes well enough, keep it unchanged.
4. If changes are needed, revise only what is necessary to better satisfy the notes and improve faithfulness to the source text.
5. Prioritize the notes over your own preferences. Do not introduce stylistic rewrites or unnecessary improvements beyond what the notes require.
6. Do not ignore the candidate and retranslate from scratch unless the candidate clearly fails to satisfy the notes or seriously misrepresents the source text.
7. Preserve the intended meaning, tone, register, formatting, named entities, and special handling requirements indicated in the notes.

Important constraints:
- Be conservative.
- Do not over-edit.
- Do not add information not present in the source text.
- Do not remove meaning unless the candidate added unsupported content.
- Follow the notes closely, especially for slang, idioms, ambiguity, tone, named entities, terminology, formatting, and transliteration/translation choices.
- If the notes and candidate are already aligned, output the original candidate exactly.
- Do not output a full comparison table.

Output format requirements:
1. Output the final post-edited translation as a single code block.
2. The code block must contain only the final translation text, with no labels or commentary inside it.

Now perform the task on the following input.

Source language: {source_lang}
Target language: {target_lang}

Source text:
```
{source_text}
```

Translation Candidate:
```
{translation_candidate}
```

Notes:
```
{notes}
```
"""


def get_post_edit_prompt(source_lang, target_lang, source_text, mt_text, notes):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    return post_edit_prompt_templates.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_candidate=mt_text,
        notes=notes,
    )


prep_notes_task_prompt = """You are a translation-prep agent. Your task is not to fully translate the source text, but to analyze it and produce only a concise list of translation-relevant notes that may help a downstream translation agent.

Input will contain:
- source language
- target language
- source text

Your job:
1. Read the source text and assess whether it contains any translation difficulties, special handling requirements, or notable stylistic/semantic risks.
2. First, provide a brief step-by-step analysis of whether any special translation guidance is needed and why.
3. In that analysis, assign a translation difficulty score from 0 to 10, where:
   - 0 = trivial to translate, no special handling needed
   - 2 = very easy, standard translation knowledge is sufficient
   - 4 = mostly straightforward, but with minor points worth noticing
   - 6 = moderately difficult, with some non-obvious translation risks
   - 8 = difficult, with clear issues such as slang, ambiguity, cultural references, or style-sensitive language
   - 10 = extremely difficult, with multiple serious translation challenges
4. Then summarize the useful translation notes into a short natural-language checklist for a downstream translator.
5. You may include recommended translations for specific words or short phrases when helpful, but do not produce a full translation of the text.
6. Do not include obvious points that a generally capable translation model would already know.

Be conservative:
- In most ordinary cases, no notes are needed.
- Avoid over-annotating simple sentences.
- Output notes only when they are genuinely useful.
- The difficulty score should reflect actual translation difficulty, not just text length or topic complexity.

Include notes only when necessary, for example:
- slang, memes, or highly colloquial expressions
- idioms, proverbs, wordplay, puns, or double meanings
- culturally specific references
- named entities, brands, titles, product names, organizations, places, or historical references
- terminology needing domain-specific handling
- ambiguous pronouns or unclear referents that require caution
- important tone/register constraints
- literary, rhetorical, poetic, humorous, sarcastic, or emotionally marked language worth preserving
- formatting-sensitive elements such as quotes, lists, UI strings, slogans, hashtags, or line breaks
- intentionally ungrammatical, stylized, ironic, or character-voiced language
- cases where transliteration vs translation may matter

Output format requirements:
1. First output your step-by-step analysis.
2. The analysis must explicitly include a line in the form:
   Difficulty score: X/10
3. After that, output the final result as a single Markdown code block.
4. Inside the code block, provide either:
   a) a numbered list of concise translation notes, or
   b) `No special translation notes needed.`

Good output example:

Step-by-step analysis
1. The text contains a slang expression that should not be translated literally.
2. The second sentence carries sarcasm, which may be easy to lose in translation.
3. A named entity appears and should be interpreted correctly.
Difficulty score: 7/10

```markdown
1. "spill the tea" is slang meaning to reveal gossip; translate by meaning rather than literally. Possible rendering: "爆料" / "说八卦" depending on target-language style.
2. Keep the sarcastic tone in the second sentence.
3. "Apple" here refers to the company, not the fruit.
```

Example when no notes are needed:

Step-by-step analysis
1. The text is straightforward and literal.
2. There are no unusual idioms, cultural references, ambiguities, or style-sensitive expressions.
3. Standard translation ability should be sufficient.
Difficulty score: 1/10

```markdown
No special translation notes needed.
```

Bad output examples:
- giving a full translation of the text
- explaining every sentence in excessive detail
- listing trivial grammar points
- omitting the difficulty score
- omitting the step-by-step analysis
- putting the final notes outside the code block

Your goal is to maximize usefulness while minimizing unnecessary guidance.
"""


prep_notes_simple_task_prompt = """You are a translation-prep agent. Your task is not to fully translate the source text, but to analyze it and produce only a concise list of translation-relevant notes that may help a downstream translation agent.

Your job:
1. Read the source text and assess whether it contains any translation difficulties, special handling requirements, or notable stylistic/semantic risks.
2. First, provide a brief step-by-step analysis of whether any special translation guidance is needed and why.
3. In that analysis, assign a translation difficulty score from 0 to 10.
4. Then summarize the useful translation notes into a short natural-language checklist for a downstream translator.
5. You may include recommended translations for specific words or short phrases when helpful, but do not produce a full translation of the text.
"""

prep_notes_prompt_template = """
{}

Now analyze the following source text.
Source language: {}
Target language: {}

Source text:
```
{}
```
"""


def get_prep_notes_prompt(source_lang, target_lang, source_text, use_simple_prompt=False):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]

    return prep_notes_prompt_template.format(prep_notes_task_prompt if not use_simple_prompt else prep_notes_simple_task_prompt, source_lang, target_lang, source_text)


if __name__ == "__main__":
    # print(get_GQM_prompt("en", "zh", "Hello, world!", ["你好，世界！", "您好，砸瓦鲁多"], "ranking_score", add_example=False, notes="A helpful note."))
    
    print(get_group_post_edit_prompt("en", "zh", "Hello, world!", ["你好，世界！", "您好，砸瓦鲁多"], notes="A helpful note."))

    # print(get_post_edit_prompt("en", "zh", "Hello, world!", "你好，世界！", "A helpful note."))




teacher_GQM_prompt_template = """You are given a source text in {source_lang}, multiple {target_lang} translation candidates, and a set of Notes. Your task is to evaluate the translation quality of all candidates step by step, compare them, and then rank and score them.

Important instructions:
1. Treat the Notes as an important and authoritative reference for evaluation.
2. For each translation candidate, explicitly check whether it is consistent with the Notes.
3. If your own interpretation conflicts with the Notes on any point, follow the Notes.
4. Only evaluate based on your own language understanding for aspects that are not covered by the Notes.
5. In the analysis section, clearly state how well each candidate matches the Notes, and use this as one of the key factors in the final ranking.
6. Compare candidates one by one in terms of:
   - accuracy of meaning
   - consistency with the Notes
   - completeness
   - fluency and naturalness
   - whether the wording introduces mistranslation, omission, or misleading information
7. Keep the analysis concise but explicit.

Output format:
### Step-by-step Analysis
[Source text analysis/breakdown]

[Analysis for each candidate, including whether and how it matches the Notes]

### Conclusion
[Overall comparison and final judgment, explicitly referring to consistency with the Notes as one of the ranking reasons]

### Final Ranking
[Output on a single line in descending order, e.g. `B > A = D > C`]

### Scores
[Output on a single line in descending order, e.g. `B: 9, A: 7, D: 7, C: 2`]

---

Source text:
```
{source_text}
```

{candidate_prompts}

Notes:
```
{notes}
```
"""


def get_teacher_GQM_with_notes_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    notes: str = None,
    **kwargs,
):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]
    if len(mt_texts) == 1:
        raise ValueError("Only support multiple candidates.")
    if len(mt_texts) > len(candidate_identifiers):
        raise ValueError(f"Only support {len(candidate_identifiers)} candidates.")


    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    return teacher_GQM_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
        notes=notes,
    )




def get_mt_privileged_prompt(
    prompt_type, src_lang, trg_lang, src_text, ref_text: str = None, ref_lang: str = None, notes: Optional[str] = None
):
    if notes is not None and prompt_type != "codeblock-think":
        raise ValueError("only codeblock-think prompt can use notes input")
    
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP[trg_lang]
    if ref_lang is not None and len(ref_lang) == 2:
        ref_lang = LANG_MAP[ref_lang]
    
    if prompt_type == "codeblock-think":
        reference_prompt = ""
        if ref_text is not None and ref_lang is not None:
            reference_prompt = f"""

Here is a reference translation in {ref_lang}:
```
{ref_text}
```
Try to provide your own analysis and final translation based on the reference.
"""
        return f"""Translate the following text from {src_lang} into {trg_lang}. Perform a step by step analysis and output the final translation in a code block.

Source text:
```
{src_text}
```{build_notes_prompt(notes)}
{reference_prompt}"""
    else:
        raise NotImplementedError



def get_GQM_GPE_prompt():
    return "Using the source text, candidate translations, and evaluation above, provide the final improved translation in the target language. Include a concise step-by-step analysis, then output only the final translation in a single Markdown code block."
