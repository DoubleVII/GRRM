"""Candidate-independent difficulty annotation with gpt-oss-120b."""
import json

from inference.hard_generation import generate
from inference.hard_prompts import build_difficulty_prompt



def validate_annotation(value, item, *, max_span_chars=48, max_span_words=8):
    """Validate exact source quotes, bounded lookup anchors, and score consistency."""
    if (not isinstance(value, dict) or not isinstance(value.get("analysis"), str)
            or not value["analysis"].strip() or type(value.get("score")) is not int
            or not 0 <= value["score"] <= 10 or not isinstance(value.get("spans"), list)):
        return None
    spans, seen = [], set()
    for span in value["spans"]:
        if not isinstance(span, dict):
            return None
        quote = span.get("source_span")
        if (not isinstance(quote, str) or not quote.strip() or quote != quote.strip()
                or quote not in item["src_text"] or quote in seen
                or "\n" in quote or "\r" in quote
                or len(quote) > max_span_chars or len(quote.split()) > max_span_words
                or type(span.get("score")) is not int or not 1 <= span["score"] <= 10
                or not isinstance(span.get("explanation"), str) or not span["explanation"].strip()):
            return None
        seen.add(quote)
        spans.append({key: span[key] for key in ("source_span", "score", "explanation")})
    if value["score"] != max((span["score"] for span in spans), default=0):
        return None
    return dict(analysis=value["analysis"], score=value["score"], spans=spans)


def annotate(items, model_path=None, *, model=None, examples_per_score=2,
             max_span_chars=48, max_span_words=8, **generation_options):
    """Annotate reference-assisted items; invalid responses use the shared OSS retry loop."""
    for name, limit in (("max_span_chars", max_span_chars), ("max_span_words", max_span_words)):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    sources, prompts = [], []
    for item in items:
        for key in ("src_text", "src_lang", "trg_lang", "ref_text"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        source = {key: item[key] for key in
                  ("data_id", "row_id", "ltb_id", "src_text", "src_lang", "trg_lang", "ref_text") if key in item}
        comment = item.get("comment")
        if comment is not None:
            if not isinstance(comment, str):
                raise ValueError("comment must be a string or None")
            if comment.strip():
                source["comment"] = comment
        sources.append(source)
        prompt = build_difficulty_prompt(source, examples_per_score=examples_per_score)
        prompts.append(prompt + f"\nAdditional output limits: each source_span must contain at most "
                       f"{max_span_chars} characters and {max_span_words} whitespace-separated words, "
                       "and no line breaks. These are hard upper bounds, not target lengths; "
                       "prefer much shorter lookup expressions as described above.")
    results = generate(prompts,
        [lambda value, item=item: validate_annotation(value, item,
            max_span_chars=max_span_chars, max_span_words=max_span_words) for item in sources],
        backend="oss", model_path=model_path, model=model, **generation_options)
    if len(results) != len(sources):
        raise ValueError("Annotation output count mismatch")
    return [{**item, "status": "ok" if result.get("parsed") is not None else "failed",
             "result": result.get("parsed"), "generation": result,
             "error": None if result.get("parsed") is not None else "Invalid annotation after retries"}
            for item, result in zip(sources, results)]


def main(src_text, src_lang, trg_lang, ref_text, model_path, temperature=0.9,
         top_p=1.0, max_tokens=8192, retry=3, reasoning_effort="medium",
         examples_per_score=2, max_span_chars=48, max_span_words=8, vllm_kwargs=None, comment=None):
    """Annotate one source text and print the complete JSON result."""
    output = annotate([dict(src_text=src_text, src_lang=src_lang, trg_lang=trg_lang,
                            ref_text=ref_text, comment=comment)],
        model_path, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        retry=retry, reasoning_effort=reasoning_effort, examples_per_score=examples_per_score,
        max_span_chars=max_span_chars, max_span_words=max_span_words, vllm_kwargs=vllm_kwargs)
    print(json.dumps(output[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
