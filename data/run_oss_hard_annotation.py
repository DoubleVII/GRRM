"""Annotate raw MT datasets without candidate translations or references."""
import json
from pathlib import Path

from inference.hard_prompts import DIFFICULTY_RUBRIC_VERSION
from inference.run_oss_hard_annotation import DEFAULT_MODEL, annotate
from utils.hard_protocol import check_output, parse_data_ids


def load_items(data_id, data_dir=None, max_samples=0):
    """Read configured paths, or a relocated dataset root / flat custom directory."""
    import pandas as pd
    from utils.config import MT_TEST_DATA_META_INFO

    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    items = []
    for did in parse_data_ids(data_id):
        meta = MT_TEST_DATA_META_INFO.get(did)
        if data_dir is not None:
            path = Path(data_dir) / f"{did}.parquet"
            if not path.exists() and meta:
                relative = Path(meta["path"]).relative_to("parquet_data/test_data_mt")
                path = Path(data_dir) / relative
        elif meta:
            path = Path(meta["path"])
        else:
            raise ValueError(f"Unknown data_id: {did}; supply data_dir for custom datasets")
        frame = pd.read_parquet(path)
        if "src_text" not in frame.columns:
            raise ValueError(f"Missing src_text in {did}")
        if max_samples:
            frame = frame.head(max_samples)
        if frame.empty:
            raise ValueError(f"No dataset rows selected for {did}")
        for row_id, row in enumerate(frame.to_dict("records")):
            item = dict(data_id=did, row_id=row_id, src_text=row["src_text"])
            for key in ("src_lang", "trg_lang"):
                item[key] = row.get(key) or (meta or {}).get(key)
            for key in ("src_text", "src_lang", "trg_lang"):
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ValueError(f"Invalid {key} in {did}, row {row_id}")
            if "ltb_id" in row and pd.notna(row["ltb_id"]):
                item["ltb_id"] = int(row["ltb_id"])
            items.append(item)
    return items


def main(data_id, output_dir, model_path=DEFAULT_MODEL, data_dir=None, max_samples=0,
         temperature=0.0, top_p=1.0, max_tokens=8192, retry=3, reasoning_effort="medium",
         examples_per_score=2, max_span_chars=48, max_span_words=8, vllm_kwargs=None):
    """Write one independent difficulty_annotations JSON artifact per dataset."""
    data_ids = parse_data_ids(data_id)
    paths = {did: Path(output_dir) / f"{did}.oss.difficulty_annotations.json" for did in data_ids}
    for path in paths.values():
        check_output(path)
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    items = load_items(data_ids, data_dir, max_samples)
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, reasoning_effort=reasoning_effort,
                   examples_per_score=examples_per_score, max_span_chars=max_span_chars,
                   max_span_words=max_span_words, vllm_kwargs=vllm_kwargs)
    output = annotate(items, model_path, **options)
    summary = {}
    for did, path in paths.items():
        subset = [item for item in output if item["data_id"] == did]
        counts = dict(total=len(subset), valid=sum(item["status"] == "ok" for item in subset),
                      failed=sum(item["status"] == "failed" for item in subset))
        payload = dict(version=1, prompt_version=DIFFICULTY_RUBRIC_VERSION,
                       stage="difficulty_annotations", config=dict(model_path=model_path,
                       data_id=did, data_dir=str(data_dir) if data_dir is not None else None,
                       max_samples=max_samples, **options), summary=counts, items=subset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        summary[did] = dict(path=str(path), **counts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
