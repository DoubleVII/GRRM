"""Source-only identities and versioned difficulty prediction artifacts."""
import json
from pathlib import Path

from inference.hard_prompts import DIFFICULTY_EXAMPLES, DIFFICULTY_RUBRIC_VERSION
from utils.hard_protocol import check_output, parse_data_ids

PROTOCOL = "difficulty-prediction-v1"
PROMPT_VERSION = "source-difficulty-v1"
SOURCE_FIELDS = ("data_id", "row_id", "src_text", "src_lang", "trg_lang")


def indexed(items, *, repeated=False):
    result = {}
    for item in items:
        if (any(not isinstance(item.get(k), str) or not item[k].strip()
                for k in ("data_id", "src_text", "src_lang", "trg_lang"))
                or type(item.get("row_id")) is not int or item["row_id"] < 0):
            raise ValueError("Invalid difficulty source identity")
        key = (item["data_id"], item["row_id"])
        if repeated:
            run_id = item.get('run_id', 0)
            if type(run_id) is not int or run_id < 0:
                raise ValueError('run_id must be a nonnegative integer')
            key += (run_id,)
        if key in result:
            raise ValueError(f"Duplicate difficulty item: {key}")
        result[key] = item
    return result


def validate_runs(runs):
    if type(runs) is not int or runs < 1:
        raise ValueError('runs must be a positive integer')
    return runs


def same_source(left, right):
    if any(left[k] != right[k] for k in SOURCE_FIELDS):
        raise ValueError(f"Source/direction mismatch: {(left['data_id'], left['row_id'])}")


def source_item(item, mode="direct"):
    value = {k: item[k] for k in SOURCE_FIELDS}
    if "ltb_id" in item:
        value["ltb_id"] = item["ltb_id"]
    if mode == "qe":
        value["candidate"] = item.get("candidate")
    return value


def calibration_reason(item):
    for example in DIFFICULTY_EXAMPLES:
        if item["src_text"] == example["src_text"]:
            return "calibration_source_text"
        if (item["data_id"], item["row_id"]) == (example["data_id"], example["row_id"]):
            return "calibration_provenance"
        if item.get("ltb_id") is not None and item.get("ltb_id") == example.get("ltb_id"):
            return "calibration_ltb_id"
    return None


def load_annotations(data_ids, directory):
    items, configs = [], {}
    for did in parse_data_ids(data_ids):
        path = Path(directory) / f"{did}.oss.difficulty_annotations.json"
        payload = json.loads(path.read_text())
        if (payload.get("version") != 1 or payload.get("stage") != "difficulty_annotations"
                or payload.get("prompt_version") != DIFFICULTY_RUBRIC_VERSION):
            raise ValueError(f"Incompatible difficulty annotations: {path}")
        subset = payload["items"]
        indexed(subset)
        if not subset or any(x["data_id"] != did for x in subset):
            raise ValueError(f"Wrong or empty dataset: {path}")
        items.extend(subset)
        configs[did] = payload["config"]
    indexed(items)
    return items, configs


def load_candidates(data_ids, directory):
    """Absent dataset files represent missing candidates, not missing sources."""
    from utils.hard_protocol import load
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Candidate directory does not exist: {directory}")
    items, configs = [], {}
    for did in parse_data_ids(data_ids):
        path = root / f"{did}.candidates.json"
        if not path.exists():
            configs[did] = None
            continue
        payload = load(path, "candidates")
        if any(x["data_id"] != did for x in payload["items"]):
            raise ValueError(f"Wrong candidate dataset: {path}")
        items.extend(payload["items"])
        configs[did] = payload["config"]
    indexed(items)
    return items, configs


def prediction_sources(annotations, mode, candidates=None):
    if mode not in {"direct", "qe"}:
        raise ValueError("mode must be direct or qe")
    indexed(annotations)
    lookup = indexed(candidates or []) if mode == "qe" else {}
    output = []
    for annotation in annotations:
        item = source_item(annotation)
        if mode == "qe":
            candidate = lookup.get((item["data_id"], item["row_id"]))
            if candidate is not None:
                same_source(item, candidate)
            item["candidate"] = (candidate.get("candidate") if candidate is not None
                                 and candidate.get("status") == "ok" else None)
        output.append(item)
    return output


def save_artifact(path, stage, config, items, **extra):
    check_output(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(dict(version=1, protocol=PROTOCOL, prompt_version=PROMPT_VERSION,
                       rubric_version=DIFFICULTY_RUBRIC_VERSION, stage=stage,
                       config=config, items=items, **extra), stream,
                  ensure_ascii=False, indent=2, allow_nan=False)


def load_predictions(data_ids, directory, mode):
    items, configs = [], {}
    for did in parse_data_ids(data_ids):
        path = Path(directory) / f"{did}.{mode}.predictions.json"
        payload = json.loads(path.read_text())
        if (payload.get("version") != 1 or payload.get("protocol") != PROTOCOL
                or payload.get("prompt_version") != PROMPT_VERSION
                or payload.get("rubric_version") != DIFFICULTY_RUBRIC_VERSION
                or payload.get("stage") != "predictions"
                or payload.get("config", {}).get("mode") != mode):
            raise ValueError(f"Incompatible difficulty predictions (legacy files unsupported): {path}")
        subset = payload["items"]
        if not subset or any(x["data_id"] != did for x in subset):
            raise ValueError(f"Wrong or empty dataset: {path}")
        runs = validate_runs(payload['config'].get('runs', 1))
        indexed(subset, repeated=True)
        by_source = {}
        for item in subset:
            key = (item['data_id'], item['row_id'])
            group = by_source.setdefault(key, [])
            if group:
                same_source(group[0], item)
            group.append(item)
        if any({x.get('run_id', 0) for x in group} != set(range(runs)) for group in by_source.values()):
            raise ValueError(f'Incomplete or unexpected prediction runs: {path}')
        items.extend(subset)
        configs[did] = payload["config"]
    indexed(items, repeated=True)
    return items, configs
