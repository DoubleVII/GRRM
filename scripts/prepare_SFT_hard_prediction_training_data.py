"""Convert one collected direct-prediction JSON artifact to native-thinking SFT.

Example (run from the repository root)::

    .venv/bin/python -m scripts.prepare_SFT_hard_prediction_training_data \
        --data_path=/path/to/dataset.direct.predictions.json \
        --output_path=/path/to/hard.train.parquet --tokenizer_path=/path/to/model
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from inference.hard_prompts import DIFFICULTY_RUBRIC_VERSION
from inference.run_oss_hard_annotation import validate_annotation
from inference.sft_hard_prompts import SFT_HARD_PROMPT_VERSION, build_sft_hard_prompt
from utils.difficulty_protocol import PROTOCOL, PROMPT_VERSION, indexed, same_source, validate_runs
from utils.difficulty_response import parse_prediction_response
from utils.hard_protocol import normalize_bool
from utils.helpers import split_thinking


def load_prediction_file(data_path):
    """Require one concrete current-version direct predictions artifact."""
    path = Path(data_path)
    if path.suffix != ".json" or not path.is_file():
        raise ValueError("data_path must be a single existing JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or payload.get("version") != 1
            or payload.get("protocol") != PROTOCOL
            or payload.get("prompt_version") != PROMPT_VERSION
            or payload.get("rubric_version") != DIFFICULTY_RUBRIC_VERSION
            or payload.get("stage") != "predictions"
            or payload.get("config", {}).get("mode") != "direct"):
        raise ValueError("Expected a current-version direct predictions artifact")
    items, config = payload["items"], payload["config"]
    if not isinstance(items, list) or not items:
        raise ValueError("Prediction artifact has no items")
    indexed(items, repeated=True)
    runs = validate_runs(config.get("runs", 1))
    groups = {}
    for item in items:
        if item["data_id"] != config.get("data_id"):
            raise ValueError("Item data_id does not match artifact config")
        key = (item["data_id"], item["row_id"])
        group = groups.setdefault(key, [])
        if group:
            same_source(group[0], item)
        group.append(item)
    if any({item.get("run_id", 0) for item in group} != set(range(runs))
           for group in groups.values()):
        raise ValueError("Incomplete or unexpected prediction runs")
    return items, config


def build_records(items, config, tokenizer, *, max_length=32768,
                  include_calibration=False, examples_per_score=2):
    """Keep strict teacher answers only; never repair labels or truncate thinking."""
    records, counts = [], Counter()
    limits = {key: config[key] for key in ("max_span_chars", "max_span_words") if key in config}
    for item in items:
        counts["total"] += 1
        if (item.get("status") != "ok" or item.get("acceptance") != "strict"
                or item.get("strict_valid") is not True):
            counts["not_strict"] += 1
            continue
        result = validate_annotation(item.get("result"), item, **limits)
        if result is None:
            counts["invalid_result"] += 1
            continue
        generation = item.get("generation") or {}
        thinking, response = generation.get("thinking"), generation.get("response")
        if not isinstance(thinking, str) or not thinking.strip():
            counts["missing_thinking"] += 1
            continue
        if not isinstance(response, str) or not response.strip():
            counts["invalid_response"] += 1
            continue
        parsed = validate_annotation(parse_prediction_response(
            response, generation.get("response_format", "json")), item, **limits)
        if parsed != result:
            counts["inconsistent_response"] += 1
            continue
        assistant = f"<think>\n{thinking.strip()}\n</think>\n{response.strip()}"
        if split_thinking(assistant) != (thinking.strip(), response.strip()):
            counts["invalid_thinking_tags"] += 1
            continue
        messages = [
            {"role": "user", "content": build_sft_hard_prompt(item,
                include_calibration=include_calibration, examples_per_score=examples_per_score)},
            {"role": "assistant", "content": assistant},
        ]
        length = len(tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=False, enable_thinking=True))
        if length > max_length:
            counts["too_long"] += 1
            continue
        records.append(dict(messages=messages, task="hard_prediction",
            data_id=item["data_id"], row_id=item["row_id"], run_id=item.get("run_id", 0),
            src_lang=item["src_lang"], trg_lang=item["trg_lang"], sequence_length=length,
            prompt_version=SFT_HARD_PROMPT_VERSION,
            include_calibration=include_calibration, examples_per_score=examples_per_score))
        counts["saved"] += 1
    return records, counts


def main(data_path: str, output_path: str,
         tokenizer_path: str = "/home/nfs06/yangs/LLM/Qwen/Qwen3-8B",
         max_length: int = 32768, max_samples: int = 0, seed: int = 114514,
         include_calibration=False, examples_per_score: int = 2):
    """Build one Parquet file; no merging, split creation, or answer truncation."""
    from transformers import AutoTokenizer

    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    include_calibration = normalize_bool(include_calibration)
    # Validate prompt options even when all input records will be filtered out.
    build_sft_hard_prompt(dict(src_text="check", src_lang="en", trg_lang="zh"),
        include_calibration=include_calibration, examples_per_score=examples_per_score)
    destination = Path(output_path)
    if destination.suffix != ".parquet":
        raise ValueError("output_path must end with .parquet")
    items, config = load_prediction_file(data_path)
    if max_samples:
        items = items[:max_samples]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    records, counts = build_records(items, config, tokenizer, max_length=max_length,
        include_calibration=include_calibration, examples_per_score=examples_per_score)
    print(json.dumps(dict(counts), ensure_ascii=False, sort_keys=True))
    if not records:
        raise ValueError("No valid SFT examples remain after filtering")
    frame = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        frame.to_parquet(stream, index=False)
    print(f"Saved {len(frame)} hard_prediction examples to {destination}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
