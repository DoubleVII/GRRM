from utils.config import LANG_MAP, candidate_identifiers


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

{candidate_prompts}{notes_prompt}"""


notes_prompt_template = """

You may refer to the following notes, if helpful, when evaluating the translations.

Notes:
```
{notes}
```
"""


candidate_prompt = """Translation {}:
```
{}
```
"""


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


def get_GQM_with_notes_prompt(
    source_lang,
    target_lang,
    source_text,
    mt_texts,
    prompt_format: str,
    add_example: bool = False,
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

    task_prompt = get_task_prompt(prompt_format, add_example)

    candidate_prompts = "".join(
        candidate_prompt.format(candidate_identifiers[i], mt_texts[i])
        for i in range(len(mt_texts))
    )

    notes_prompt = build_notes_prompt(notes)

    return GQM_prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        task_prompt=task_prompt,
        source_text=source_text,
        candidate_prompts=candidate_prompts,
        notes_prompt=notes_prompt,
    )




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

You will receive:
- source language
- target language
- source text
- 1 to 4 translation candidates

Your job:
1. Read the source text carefully.
2. Compare the translation candidates against the source text.
3. Identify which candidate is the best base, or whether a careful combination of candidates is needed.
4. Perform minimal, targeted post-editing to produce the best final translation.
5. Prioritize correctness, faithfulness, fluency, consistency, and naturalness in the target language.
6. Do not produce multiple alternative translations.

Be especially careful about:
- preserving meaning from the source text
- resolving omissions, mistranslations, and overtranslations
- choosing the most accurate wording among candidates
- maintaining grammatical correctness and fluency
- improving your translation if necessary

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



group_post_edit_with_notes_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and strictly following the provided translation notes.

Your priority is to respect the notes and make the smallest necessary edits.

Be especially careful about:
- preserving meaning from the source text
- preserving tone/register/style when mentioned in the notes
- not introducing unsupported improvements

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


group_post_edit_prompt_templates = """You are a translation post-editing agent.

Your task is to produce a final improved translation in the target language by:
- using the provided source text,
- reviewing the available translation candidates,
- and selecting or carefully combining the best parts of the candidates.


Be especially careful about:
- preserving meaning from the source text
- resolving omissions, mistranslations, and overtranslations
- choosing the most accurate wording among candidates
- maintaining grammatical correctness and fluency
- improving your translation if necessary

Now perform the task on the following input.

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


def get_prep_notes_prompt(source_lang, target_lang, source_text):
    if len(source_lang) == 2:
        source_lang = LANG_MAP[source_lang]
    if len(target_lang) == 2:
        target_lang = LANG_MAP[target_lang]

    return prep_notes_prompt_template.format(prep_notes_task_prompt, source_lang, target_lang, source_text)


if __name__ == "__main__":
    # print(get_GQM_with_notes_prompt("en", "zh", "Hello, world!", ["你好，世界！", "您好，砸瓦鲁多"], "ranking_score", add_example=False, notes="A helpful note."))
    
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