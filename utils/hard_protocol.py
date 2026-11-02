"""Versioned artifacts and validated source spans for translation difficulty."""
import json
import re
from pathlib import Path

VERSION = 1
PROMPT_VERSION = "hard-v2"
IDENTITY = ("data_id", "row_id", "src_text", "src_lang", "trg_lang",
            "ref_text", "comment", "candidate")


def parse_data_ids(data_id):
    values = data_id.split(",") if isinstance(data_id, str) else list(data_id)
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError("Provide nonempty, distinct data_ids")
    ids = tuple(value.strip() for value in values)
    if (len(set(ids)) != len(ids) or any(not did or did in {".", ".."}
            or "/" in did or "\\" in did for did in ids)):
        raise ValueError("Provide distinct data_ids usable as filenames")
    return ids


def artifact_path(directory, data_id, stage, mode=None):
    parse_data_ids([data_id])
    if stage not in {"candidates", "predictions", "golden", "evaluation"}:
        raise ValueError("Unknown artifact stage")
    if stage in {"predictions", "evaluation"}:
        if mode not in {"direct", "qe"}:
            raise ValueError("mode must be direct or qe")
        suffix = f"{mode}.{stage}"
    else:
        suffix = stage
    return Path(directory) / f"{data_id}.{suffix}.json"


def output_paths(data_ids, output_dir, stage, mode=None):
    paths = {did: artifact_path(output_dir, did, stage, mode) for did in data_ids}
    for path in paths.values():
        check_output(path)
    return paths


def load_dataset_artifacts(data_ids, directory, stage, mode=None):
    items, configs = [], {}
    for did in data_ids:
        path = artifact_path(directory, did, stage, mode)
        payload = load(path, stage)
        if not payload["items"] or any(item["data_id"] != did for item in payload["items"]):
            raise ValueError(f"Wrong or empty data_id contents: {path}")
        if stage in {"predictions", "evaluation"} and payload["config"].get("mode") != mode:
            raise ValueError(f"Prediction mode mismatch: {path}")
        items.extend(payload["items"])
        configs[did] = payload["config"]
    return items, configs


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError("enable_thinking must be true or false")


def check_output(path):
    if Path(path).exists():
        raise FileExistsError(path)


def save(path, stage, config, items, **extra):
    check_output(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(dict(version=VERSION, prompt_version=PROMPT_VERSION,
                       stage=stage, config=config, items=items, **extra),
                  stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(path, stage):
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != VERSION or payload.get("stage") != stage:
        raise ValueError(f"Unsupported artifact version/stage: {path}")
    indexed(payload["items"])
    return payload


def indexed(items):
    result = {}
    for item in items:
        if any(field not in item for field in IDENTITY):
            raise ValueError("Missing item identity fields")
        if not isinstance(item["data_id"], str) or type(item["row_id"]) is not int:
            raise ValueError("Invalid data_id/row_id")
        if (not item["data_id"].strip() or item["row_id"] < 0
                or any(not isinstance(item[field], str) or not item[field].strip()
                       for field in ("src_text", "src_lang", "trg_lang", "ref_text"))
                or (item["candidate"] is not None and not isinstance(item["candidate"], str))
                or (item["comment"] is not None and not isinstance(item["comment"], str))):
            raise ValueError("Invalid item text, language, or context fields")
        key = (item["data_id"], item["row_id"])
        if key in result:
            raise ValueError(f"Duplicate item: {key}")
        result[key] = item
    return result


def align(predictions, golden):
    left, right = indexed(predictions), indexed(golden)
    if left.keys() != right.keys():
        raise ValueError("Prediction/golden item sets differ")
    for key, item in left.items():
        if any(item[field] != right[key][field] for field in IDENTITY):
            raise ValueError(f"Prediction/golden candidate or context mismatch: {key}")
    return [(item, right[key]) for key, item in left.items()]


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def canonical_quote(quote, text):
    """Recover the original substring, tolerating case and line/space formatting.

    The second pass also accepts line breaks inserted inside a word. Punctuation
    and non-whitespace characters must still match; this is not fuzzy matching.
    """
    if not isinstance(quote, str) or not quote.strip():
        return None
    if quote in text:
        return quote

    def normalize(value, join_lines):
        chars, positions = [], []
        for match in re.finditer(r"\s+|\S", value):
            token = match.group()
            if token.isspace():
                if join_lines and ("\n" in token or "\r" in token):
                    continue
                normalized = " "
            else:
                normalized = token.casefold()
            for char in normalized:
                chars.append(char)
                positions.append((match.start(), match.end()))
        return "".join(chars), positions

    for join_lines in (False, True):
        needle, _ = normalize(quote, join_lines)
        needle = needle.strip()
        haystack, positions = normalize(text, join_lines)
        start = haystack.find(needle)
        while needle and start >= 0:
            end = start + len(needle)
            # Do not match just half of a Unicode casefold expansion (e.g. ß).
            if ((start == 0 or positions[start - 1] != positions[start])
                    and (end == len(positions) or positions[end - 1] != positions[end])):
                return text[positions[start][0]:positions[end - 1][1]]
            start = haystack.find(needle, start + 1)
    return None


def validate_spans(value, item, golden=False):
    if not isinstance(value, dict) or not isinstance(value.get("analysis"), str):
        return None
    spans = value.get("spans")
    if not isinstance(spans, list):
        return None
    seen = set()
    canonical_spans = []
    for span in spans:
        if not isinstance(span, dict):
            return None
        text = canonical_quote(span.get("source_span"), item["src_text"])
        if (text is None
                or not isinstance(span.get("explanation"), str)
                or not span["explanation"].strip()):
            return None
        marker = (text, span["explanation"])
        if marker in seen:
            return None
        seen.add(marker)
        canonical = {**span, "source_span": text}
        if golden:
            error = span.get("translation_span")
            if (not isinstance(error, str)
                    or span.get("error_type") not in {"omission", "addition", "other"}):
                return None
            if error == "" and span["error_type"] == "omission":
                canonical["translation_span"] = ""
            else:
                error = canonical_quote(error, item["candidate"])
                if error is None:
                    return None
                canonical["translation_span"] = error
        canonical_spans.append(canonical)
    return {**value, "spans": canonical_spans}


def valid_candidate(item):
    return isinstance(item.get("candidate"), str) and bool(item["candidate"].strip())


def result_item(item, result=None, error=None):
    result = result or {"parsed": None, "response": None, "thinking": None}
    valid = result.get("parsed") is not None
    return {**{key: item[key] for key in IDENTITY}, "status": "ok" if valid else "failed",
            "result": result.get("parsed"), "generation": result,
            "error": None if valid else (error or "Invalid model response after retries")}
