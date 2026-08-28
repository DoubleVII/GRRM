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
    candidate_confidence: bool = False,
    max_decision_points: int = 4,
) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    if prompt_type == "decision_points":
        if candidate_confidence:
            raise ValueError(
                "candidate_confidence is not supported for decision_points"
            )
        return f"""You are performing the exploratory first stage of a translation task from {source_lang} to {target_lang}.

Do not translate the full source and do not divide it into exhaustive translation segments. Instead, identify only high-impact translation decision points where two or more contextually plausible choices could materially change correctness. A decision point may be a genuinely ambiguous term, idiom, culturally specific expression, difficult attachment, or another compact source span whose best rendering cannot be chosen mechanically from the full context.

Requirements:
- Return zero decision points when the source has no meaningful local ambiguity or terminology risk. Do not invent decision points for easy wording.
- Normally return 0 to 2 decision points. Return 3 or 4 only for unusually difficult sources with that many independent, consequential choices. The limit of {max_decision_points} is a hard ceiling, not a target or desired count.
- Do not create a decision point merely because a span is a name, date, number, ordinary technical term, or register marker. Include one only when there is a real risk that alternatives would lead to materially different translations.
- Decision points do not need to cover the source, preserve source order, or be mutually exclusive; overlapping spans are allowed when they represent different decisions.
- Copy each source_span exactly from the source and keep it as compact as possible while retaining enough context to make the decision meaningful.
- Normally give {min_candidates} to {max_candidates} genuinely distinct local {target_lang} candidates per decision point. Give fewer only when fewer meaningful alternatives exist.
- Candidate differences must concern meaning, terminology, idiom handling, register, or another consequential translation choice, not superficial punctuation or function-word variation.
- Every candidate must preserve the same source facts. Diversity never permits changing an entity, number, unit, polarity, degree, time, or logical relationship. Discard a candidate if it is fluent but factually inconsistent with the source.
- Before proposing candidates involving quantities, explicitly verify the magnitude and unit. For example, when translating into Chinese, $10 billion is 100亿美元, not 10亿美元. All candidates must use a factually equivalent quantity.
- Add global_constraints only for whole-text relationships that Stage 2 must preserve, such as negation or quantifier scope, entity and number consistency, long-distance modification, condition, causality, comparison, coreference, or discourse relations. State the intended source meaning, not a target-language translation.
- In global_constraints, spell out exact numeric magnitudes, units, dates, and entity relationships whenever they are easy to confuse. Do not merely repeat an ambiguous surface form.
- Do not assign confidence labels, select a winner, compose a draft, or output a final translation.

Return only one valid JSON object with this schema:
{{
  "source_analysis": "brief global analysis of meaning, context, tone, and the main translation risks",
  "global_constraints": [
    "a concise whole-text semantic constraint that Stage 2 must preserve"
  ],
  "decision_points": [
    {{
      "decision_point_id": 1,
      "source_span": "an exact compact span copied from the source",
      "issue_type": "terminology, idiom, entity, word_sense, attachment, register, culture, or another short label",
      "analysis": "why this decision matters in the full source context",
      "candidates": [
        {{"translation": "local {target_lang} translation", "angle": "the consequential interpretation or rendering this candidate explores"}}
      ]
    }}
  ]
}}

Source text:
<source>
{source_text}
</source>"""

    if prompt_type == "semantic_units":
        unit_task = """Analyze the source, identify ordered semantic translation units, and propose diverse translation candidates for every unit.

A unit may range from a single word or term through a phrase or clause to a complete sentence. Choose the span needed to preserve a coherent meaning or translation decision; do not prefer small units merely because they are easy to isolate. Use a larger phrase, clause, or sentence when splitting it would weaken negation or quantifier scope, attachment, causality, comparison, coreference, idiomatic meaning, tone, or another cross-word dependency.

Units may overlap, and parent-child units are encouraged when they support exploration at complementary levels. A parent unit should preserve the complete meaning and dependencies of a larger phrase, clause, or sentence. Its child units may separately explore important terms, idioms, entities, or local ambiguities inside that span. Give candidates for both levels: parent candidates explore coherent renderings of the complete span, while child candidates preserve local alternatives that Stage 2 may use when composing or polishing the parent meaning. Order units by where they begin in the source; for units with the same start, place the larger unit first. Avoid duplicate units that explore the same span at the same semantic level."""
        coverage_task = """- Collectively cover all lexical content in the source, but do not force a partition: meaningful overlap is allowed. Boundary punctuation may remain outside units, but no words or meaning may be omitted.
- Use semantically complete parent units to cover the source, then add focused child units where local exploration is useful. Do not fragment ordinary compositional wording into child units that offer no meaningful translation choice."""
    else:
        unit_task = """Analyze the source, divide it into ordered, non-overlapping translation units, and propose diverse local translation candidates for every unit. A unit may be a clause, idiom, term, name, discourse marker, or other span that should be translated together."""
        coverage_task = """- Cover all lexical content in the source and preserve its order. Boundary punctuation may remain between units, but no words or meaning may be omitted."""

    shared_task = f"""You are performing the divergent first stage of a translation task from {source_lang} to {target_lang}.

{unit_task}

The purpose of this stage is exploration, not selection:
{coverage_task}
- Normally give {min_candidates} to {max_candidates} genuinely distinct candidates per unit. Explore plausible differences in sense, register, syntax, idiom handling, terminology, and target-language naturalness.
- Do not create superficial variants that differ only in punctuation or one interchangeable function word.
- Give fewer than {min_candidates} candidates only when there is truly no meaningful alternative; briefly state why in the unit analysis.
- A candidate must translate only its own source span, not the entire source.
- Do not choose or compose a final translation in this stage.
"""

    confidence_task = ""
    if candidate_confidence:
        confidence_task = """
Assign every candidate a confidence level based on how reliably it conveys its source span in the full source context, not on stylistic preference:
- high: strongly supported by the source and context; safe as a default choice.
- medium: plausible, but depends on an unresolved ambiguity or contextual assumption.
- low: speculative or risky; include it only when it represents a genuinely useful alternative that stage 2 should verify carefully.
Confidence is not a diversity dimension and must not be balanced across candidates. Most candidates for an unambiguous unit should normally be high confidence, even when they differ in style, register, or wording. Use medium or low only for a genuine semantic or contextual uncertainty supported by the source. Never lower confidence merely because a candidate is idiomatic, less literal, or more creative. Do not force every unit to contain all three levels, and do not invent or downgrade candidates merely to populate a level. Explain the concrete uncertainty or risk for every medium- and low-confidence candidate in its angle.
"""
    shared_task += confidence_task

    codeblock_source_section = f"""Source text:
```
{source_text}
```"""

    if prompt_type == "codeblock":
        confidence_format = (
            " Explicitly label every candidate as confidence: high, medium, or low."
            if candidate_confidence
            else ""
        )
        return f"""{shared_task}

Output format requirements:
1. First output a brief step-by-step analysis of the source's meaning, context, tone, ambiguities, and translation constraints.
2. After that, output the exploratory segmentation and candidates as a single Markdown code block.
3. Inside the code block, organize the segments, local candidates, and their distinct interpretation or style angles in any clear natural-language format you find useful.{confidence_format} You do not need to follow JSON, XML, or any fixed schema.
4. Do not put a final full translation in the code block or anywhere else.
5. Do not output anything after the closing code fence.

{codeblock_source_section}"""

    if prompt_type not in {"json", "semantic_units"}:
        raise ValueError(
            "prompt_type must be one of: json, codeblock, semantic_units, "
            "decision_points"
        )

    confidence_schema = (
        ', "confidence": "high"'
        if candidate_confidence
        else ""
    )
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
        {{"translation": "local {target_lang} translation", "angle": "what interpretation or style this explores"{confidence_schema}}}
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
    *,
    polish: bool = True,
    candidate_confidence: bool = False,
    prompt_type: str = "json",
) -> str:
    source_lang = _language_name(source_lang)
    target_lang = _language_name(target_lang)
    if isinstance(divergent_result, dict):
        divergent_context = json.dumps(divergent_result, ensure_ascii=False, indent=2)
    elif isinstance(divergent_result, str) and divergent_result.strip():
        divergent_context = divergent_result.strip()
    else:
        raise ValueError("divergent_result must be a non-empty dict or string")
    if prompt_type == "decision_points":
        if candidate_confidence:
            raise ValueError(
                "candidate_confidence is not supported for decision_points"
            )
        polish_requirement = (
            f"""- Polish the complete draft for idiomatic phrasing, sentence structure, cohesion, punctuation, and consistent register so it reads as an originally written {target_lang} text.
"""
            if polish
            else ""
        )
        return f"""You are performing the final stage of a translation task from {source_lang} to {target_lang}.

The source and a selective analysis of high-impact translation decision points are provided below. The analysis deliberately does not cover the full source and must never be treated as pieces to concatenate.

Work in this order:
1. Read the complete source and independently construct a faithful, coherent full-text draft without relying on the decision-point candidates.
2. Check every global constraint against that draft, especially negation and quantifier scope, entities, numbers, units, long-distance modification, condition, causality, comparison, and coreference. Independently recompute numeric magnitudes from the source instead of trusting the analysis or candidates.
3. Consult each decision point only as optional evidence. Use, edit, or reject its candidates according to the full source context. Reject any candidate that changes a fact even if it is fluent or repeated by several candidates. Keep the independent draft when none of the candidates improves it.
4. Re-read the complete source and perform a final fidelity audit. Local improvements must not damage global meaning or relationships.

Requirements:
- Translate the complete source without additions or omissions.
- The source is authoritative; the selective analysis may be incomplete or wrong.
- Preserve every number and unit at the correct magnitude. For example, when translating into Chinese, $10 billion is 100亿美元, not 10亿美元.
{polish_requirement}- Preserve the source's intended tone and formatting where appropriate.
- Output only the final translation between the exact tags below. Do not include analysis, labels, Markdown fences, or text outside the tags.

<source>
{source_text}
</source>

<decision_point_analysis>
{divergent_context}
</decision_point_analysis>

<final_translation>...</final_translation>"""
    if prompt_type not in {"json", "codeblock", "semantic_units"}:
        raise ValueError(
            "prompt_type must be one of: json, codeblock, semantic_units, "
            "decision_points"
        )
    confidence_requirement = ""
    if candidate_confidence:
        confidence_requirement = """
- Use candidate confidence as a calibrated aid, not as ground truth. Prefer high-confidence candidates when the source supports them; independently verify medium- and low-confidence candidates before using them. Reject even a high-confidence candidate when it conflicts with the source or global context.
"""
    if polish:
        composition_requirements = f"""- Treat segment boundaries and local candidates only as analysis aids. First construct a complete draft, freely rewriting across segment boundaries instead of concatenating candidate phrases.
- Then perform a mandatory whole-text polish pass: improve idiomatic phrasing, collocations, sentence structure, cohesion, punctuation, and register so the result reads as an originally written {target_lang} text rather than a stitched list of fragments.
- After polishing, check the full translation against the source again. Fluency edits must not change meaning, factual details, named entities, numbers, negation, emphasis, or tone.
"""
    else:
        composition_requirements = f"""- Produce a natural, internally consistent {target_lang} text, not a stitched list of fragments.
"""
    return f"""You are performing the convergent second stage of a translation task from {source_lang} to {target_lang}.

The source and a divergent analysis with local candidates are provided below. Critically compare the candidates against the source, reject mistranslations, and select, edit, or combine the strongest choices into one coherent translation. Resolve cross-segment dependencies such as terminology, pronouns, tense, register, syntax, and discourse flow. The candidate set is exploratory evidence, not an authority: repair it whenever needed.

Requirements:
- Translate the complete source faithfully without additions or omissions.
{confidence_requirement}{composition_requirements}- Preserve the source's intended tone and formatting where appropriate.
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
