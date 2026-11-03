"""Generate and freeze one instruct translation per dataset row."""
from pathlib import Path

from utils.hard_protocol import normalize_bool, output_paths, parse_data_ids, save


def load_items(data_id, data_dir=None, max_samples=0):
    import pandas as pd
    from utils.config import MT_TEST_DATA_META_INFO
    ids = parse_data_ids(data_id)
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    items = []
    for did in ids:
        if data_dir:
            path = Path(data_dir) / f"{did}.parquet"
        else:
            if did not in MT_TEST_DATA_META_INFO:
                raise ValueError(f"Unknown data_id: {did}")
            path = Path(MT_TEST_DATA_META_INFO[did]["path"])
        frame = pd.read_parquet(path)
        required = {"src_text", "src_lang", "trg_lang", "trg_text"}
        if required - set(frame.columns):
            raise ValueError(f"Missing columns in {did}: {sorted(required - set(frame.columns))}")
        if max_samples:
            frame = frame.head(max_samples)
        if frame.empty:
            raise ValueError(f"No dataset rows selected for {did}")
        for row_id, row in enumerate(frame.to_dict("records")):
            for field in required:
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"Invalid {field} in {did}, row {row_id}")
            comment = row.get("comment")
            if comment is None or pd.isna(comment) or not str(comment).strip():
                comment = None
            else:
                comment = str(comment)
            items.append(dict(data_id=did, row_id=row_id, src_text=row["src_text"],
                              src_lang=row["src_lang"], trg_lang=row["trg_lang"],
                              ref_text=row["trg_text"], comment=comment, candidate=None))
    if not items:
        raise ValueError("No dataset rows selected")
    return items


def make_candidates(items, model_path, model=None, vllm_kwargs=None, **options):
    from inference.run_inst_mt import init_inst_model, run_translation_stage
    engine = model if model is not None else init_inst_model(model_path, **(vllm_kwargs or {}))
    result = run_translation_stage(
        [item["src_text"] for item in items], [item["src_lang"] for item in items],
        [item["trg_lang"] for item in items], model=engine, **options)
    if len(result["translations"]) != len(items):
        raise ValueError("Translation output count mismatch")
    output = []
    for i, item in enumerate(items):
        candidate = result["translations"][i]
        valid = isinstance(candidate, str) and bool(candidate.strip())
        output.append({**item, "candidate": candidate if valid else None,
                       "status": "ok" if valid else "failed",
                       "error": None if valid else "Translation failed after retries",
                       "generation": {key: values[i] for key, values in result.items()}})
    return output


def main(data_id, output_dir, model_path, data_dir=None, max_samples=0,
         temperature=0.9, top_p=1.0, max_tokens=8192, retry=3,
         enable_thinking=False, prompt_version="codeblock", vllm_kwargs=None):
    data_ids = parse_data_ids(data_id)
    paths = output_paths(data_ids, output_dir, "candidates")
    if retry < 0:
        raise ValueError("retry must be nonnegative")
    options = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                   retry=retry, enable_thinking=normalize_bool(enable_thinking),
                   prompt_version=prompt_version)
    items = load_items(data_ids, data_dir, max_samples)
    output = make_candidates(items, model_path, vllm_kwargs=vllm_kwargs, **options)
    for did, path in paths.items():
        save(path, "candidates", dict(model_path=model_path, data_id=did,
             data_dir=data_dir, max_samples=max_samples, vllm_kwargs=vllm_kwargs, **options),
             [item for item in output if item["data_id"] == did])


if __name__ == "__main__":
    import fire
    fire.Fire(main)
