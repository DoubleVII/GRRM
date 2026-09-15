import json
import random
import statistics
from pathlib import Path

import pandas as pd

from inference.run_inst_flash_gpe_mt import init_inst_model, run_pipeline


DEFAULT_DATA = "/home/nfs06/yangs/data/parquet_data/raw/towerx_v2_all.parquet"


def _rate(values):
    return sum(values) / len(values) if values else 0.0


def _token_summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None,
                "p95": None, "min": None, "max": None}

    def percentile(percent):
        index = min(len(values) - 1, int((len(values) - 1) * percent))
        return values[index]

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": values[0],
        "max": values[-1],
    }


def main(data_path: str = DEFAULT_DATA,
         model_path: str = None,
         output_path: str = "results/benchmark_inst_flash_gpe_mt.json",
         max_samples: int = 256, seed: int = 42, runs: int = 4,
         min_candidates: int = 2, max_candidates: int = 8, retry: int = 0,
         enable_thinking: bool = True,
         candidate_temperature: float = 1.0, candidate_top_p: float = 0.95,
         candidate_top_k: int = 20, candidate_presence_penalty: float = 1.5,
         candidate_repetition_penalty: float = 1.0,
         candidate_max_tokens: int = 8192, post_edit_temperature: float = 1.0,
         post_edit_top_p: float = 0.95, post_edit_top_k: int = 20,
         post_edit_presence_penalty: float = 1.5,
         post_edit_repetition_penalty: float = 1.0,
         post_edit_max_tokens: int = 8192,
         gpu_memory_utilization: float = 0.9, max_model_len: int = 32768):
    if not model_path:
        raise ValueError("model_path is required")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if retry != 0:
        raise ValueError("benchmark retry must be 0 so runs remain independent")
    if not 2 <= min_candidates <= max_candidates:
        raise ValueError("Expected 2 <= min_candidates <= max_candidates")
    frame = pd.read_parquet(data_path)
    required = {"src_text", "src_lang", "trg_lang"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_samples > 0 and len(frame) > max_samples:
        frame = frame.sample(max_samples, random_state=seed)
    frame = frame.reset_index(drop=True)
    model = init_inst_model(model_path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    rng = random.Random(seed)
    details = [{"index": i, "src_text": row.src_text, "src_lang": row.src_lang,
                "trg_lang": row.trg_lang, "runs": []} for i, row in frame.iterrows()]
    stage1_all, stage2_all, overall_all = [], [], []
    candidate_tokens_all, post_edit_tokens_all = [], []
    per_run_token_summaries = []
    for run_index in range(runs):
        cycle = list(range(min_candidates, max_candidates + 1))
        rng.shuffle(cycle)
        candidate_counts = [cycle[i % len(cycle)] for i in range(len(frame))]
        result = run_pipeline(frame.src_text.tolist(), frame.src_lang.tolist(), frame.trg_lang.tolist(),
                              model=model, max_candidates=max_candidates,
                              candidate_counts=candidate_counts, retry=retry,
                              enable_thinking=enable_thinking, candidate_temperature=candidate_temperature,
                              candidate_top_p=candidate_top_p, candidate_top_k=candidate_top_k,
                              candidate_presence_penalty=candidate_presence_penalty,
                              candidate_repetition_penalty=candidate_repetition_penalty,
                              candidate_max_tokens=candidate_max_tokens,
                              post_edit_temperature=post_edit_temperature, post_edit_top_p=post_edit_top_p,
                              post_edit_top_k=post_edit_top_k,
                              post_edit_presence_penalty=post_edit_presence_penalty,
                              post_edit_repetition_penalty=post_edit_repetition_penalty,
                              post_edit_max_tokens=post_edit_max_tokens)
        for i in range(len(frame)):
            stage1 = result["usable_candidate_counts"][i] >= 2
            stage2 = bool(result["post_edit"]["translations"][i])
            overall = stage1 and stage2
            stage1_all.append(stage1); stage2_all.append(stage2); overall_all.append(overall)
            candidate_tokens = result["candidate_generation"]["output_tokens"][i]
            post_edit_tokens = result["post_edit"]["output_tokens"][i]
            if stage1:
                candidate_tokens_all.append(candidate_tokens)
            if stage2:
                post_edit_tokens_all.append(post_edit_tokens)
            details[i]["runs"].append({"run": run_index, "stage1_pass": stage1,
                                       "stage2_pass": stage2, "overall_pass": overall,
                                       "usable_candidate_count": result["usable_candidate_counts"][i],
                                       "candidates": result["candidate_generation"]["translations"][i],
                                       "candidate_response": result["candidate_generation"]["responses"][i],
                                       "candidate_thinking": result["candidate_generation"]["thinking"][i],
                                       "candidate_output_tokens": candidate_tokens,
                                       "post_edit_response": result["post_edit"]["responses"][i],
                                       "post_edit_thinking": result["post_edit"]["thinking"][i],
                                       "post_edit_output_tokens": post_edit_tokens,
                                       "translation": result["post_edit"]["translations"][i]})
        per_run_token_summaries.append({
            "run": run_index,
            "stage1_output_tokens": _token_summary([
                result["candidate_generation"]["output_tokens"][i]
                for i in range(len(frame))
                if result["usable_candidate_counts"][i] >= 2
            ]),
            "stage2_output_tokens": _token_summary([
                result["post_edit"]["output_tokens"][i]
                for i in range(len(frame))
                if result["post_edit"]["translations"][i]
            ]),
        })
    def summary(values):
        return {"total": len(values), "passed": sum(values), "failed": len(values) - sum(values), "pass_rate": _rate(values)}
    payload = {"method": "inst_flash_gpe_format_benchmark", "data_path": data_path,
               "model_path": model_path, "settings": {"max_samples": max_samples, "seed": seed,
               "runs": runs, "min_candidates": min_candidates, "max_candidates": max_candidates,
               "retry": retry, "enable_thinking": enable_thinking,
               "candidate_temperature": candidate_temperature,
               "candidate_top_p": candidate_top_p, "candidate_top_k": candidate_top_k,
               "candidate_presence_penalty": candidate_presence_penalty,
               "candidate_repetition_penalty": candidate_repetition_penalty,
               "candidate_max_tokens": candidate_max_tokens,
               "post_edit_temperature": post_edit_temperature,
               "post_edit_top_p": post_edit_top_p, "post_edit_top_k": post_edit_top_k,
               "post_edit_presence_penalty": post_edit_presence_penalty,
               "post_edit_repetition_penalty": post_edit_repetition_penalty,
               "post_edit_max_tokens": post_edit_max_tokens},
               "summary": {"stage1": summary(stage1_all), "stage2": summary(stage2_all),
                            "overall": summary(overall_all),
                            "stage1_output_tokens": _token_summary(candidate_tokens_all),
                            "stage2_output_tokens": _token_summary(post_edit_tokens_all),
                            "per_run_output_tokens": per_run_token_summaries},
               "items": details}
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved benchmark results to {destination}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
