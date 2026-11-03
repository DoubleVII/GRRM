"""Annotate MT difficulty using reference translations and optional expert comments."""
import json
from pathlib import Path

from inference.hard_prompts import DIFFICULTY_RUBRIC_VERSION
from inference.run_oss_hard_annotation import annotate
from utils.hard_protocol import check_output, parse_data_ids


def _input_paths(data_id, data_dir=None, input_file=None):
    from utils.config import MT_TEST_DATA_META_INFO

    if input_file is not None:
        if data_dir is not None:
            raise ValueError("input_file and data_dir cannot be used together")
        path = Path(input_file)
        data_ids = parse_data_ids(path.stem if data_id is None else data_id)
        if len(data_ids) != 1:
            raise ValueError("input_file requires exactly one data_id")
        return {data_ids[0]: path}
    if data_id is None:
        raise ValueError("Provide data_id or input_file")
    paths = {}
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
        paths[did] = path
    return paths


def load_items(data_id=None, data_dir=None, max_samples=0, input_file=None):
    """Read a direct parquet file, configured paths, or a custom dataset directory."""
    import pandas as pd
    from utils.config import MT_TEST_DATA_META_INFO

    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    items = []
    for did, path in _input_paths(data_id, data_dir, input_file).items():
        meta = MT_TEST_DATA_META_INFO.get(did)
        frame = pd.read_parquet(path)
        required = {"src_text", "trg_text"}
        if required - set(frame.columns):
            raise ValueError(f"Missing columns in {did}: {sorted(required - set(frame.columns))}")
        if max_samples:
            frame = frame.head(max_samples)
        if frame.empty:
            raise ValueError(f"No dataset rows selected for {did}")
        for row_id, row in enumerate(frame.to_dict("records")):
            item = dict(data_id=did, row_id=row_id, src_text=row["src_text"],
                        ref_text=row["trg_text"])
            for key in ("src_lang", "trg_lang"):
                item[key] = row.get(key) or (meta or {}).get(key)
            for key in ("src_text", "src_lang", "trg_lang", "ref_text"):
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ValueError(f"Invalid {key} in {did}, row {row_id}")
            comment = row.get("comment")
            if comment is not None and pd.notna(comment) and str(comment).strip():
                item["comment"] = str(comment)
            if "ltb_id" in row and pd.notna(row["ltb_id"]):
                item["ltb_id"] = int(row["ltb_id"])
            items.append(item)
    return items


def main(data_id=None, output_dir=None, model_path=None, data_dir=None, max_samples=0,
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3, reasoning_effort="medium",
         examples_per_score=2, max_span_chars=48, max_span_words=8, vllm_kwargs=None,
         input_file=None):
    """Annotate configured datasets or --input_file parquet; requires explicit model_path/output_dir.

    With input_file, data_id defaults to the filename without its extension.
    """
    if output_dir is None:
        raise ValueError("Provide output_dir")
    if model_path is None or not str(model_path).strip():
        raise ValueError("Provide an explicit model_path")
    data_ids = tuple(_input_paths(data_id, data_dir, input_file))
    paths = {did: Path(output_dir) / f"{did}.oss.difficulty_annotations.json" for did in data_ids}
    for path in paths.values():
        check_output(path)
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, or high")
    items = load_items(data_ids, data_dir, max_samples, input_file=input_file)
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
        if input_file is not None:
            payload["config"]["input_file"] = str(input_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        summary[did] = dict(path=str(path), **counts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
