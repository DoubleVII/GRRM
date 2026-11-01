"""Golden annotation and semantic matching; scoring is deterministic Python."""
from inference.hard_generation import generate
from inference.hard_prompts import golden_prompt, matching_prompt
from utils.hard_protocol import (align, result_item, valid_candidate, validate_spans)

DEFAULT_MODEL = "openai/gpt-oss-120b"


def annotate(items, model_path=DEFAULT_MODEL, **options):
    eligible = [item for item in items if valid_candidate(item)]
    results = generate(
        [golden_prompt(item) for item in eligible],
        [lambda value, item=item: validate_spans(value, item, golden=True) for item in eligible],
        backend="oss", model_path=model_path, **options)
    iterator = iter(results)
    return [result_item(item, next(iterator)) if valid_candidate(item)
            else result_item(item, error="Candidate translation failed") for item in items]


def validate_pairs(value, predicted_count, golden_count):
    if not isinstance(value, dict) or not isinstance(value.get("analysis"), str):
        return None
    pairs = value.get("pairs")
    if not isinstance(pairs, list):
        return None
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(index) is not int for index in pair)
                or not 0 <= pair[0] < predicted_count or not 0 <= pair[1] < golden_count
                or tuple(pair) in seen):
            return None
        seen.add(tuple(pair))
    return value


def maximum_matching(pairs, predicted_count):
    adjacency = [[] for _ in range(predicted_count)]
    for left, right in pairs:
        adjacency[left].append(right)
    owners = {}

    def augment(left, seen):
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in owners or augment(owners[right], seen):
                owners[right] = left
                return True
        return False

    for left in range(predicted_count):
        augment(left, set())
    return sorted([[left, right] for right, left in owners.items()])


def metrics(tp, fp, fn, *, empty_item=False):
    precision = tp / (tp + fp) if tp + fp else float(empty_item)
    recall = tp / (tp + fn) if tp + fn else float(empty_item)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float(empty_item)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def summarize(items):
    valid = [item for item in items if item["status"] == "ok"]
    counts = {key: sum(item["metrics"][key] for item in valid) for key in ("tp", "fp", "fn")}
    return dict(**metrics(**counts), total=len(items), valid=len(valid),
                failed=len(items) - len(valid), coverage=len(valid) / len(items) if items else 0.0,
                both_empty=sum(item["both_empty"] for item in valid))


def evaluate(predictions, golden, model_path=DEFAULT_MODEL, **options):
    aligned = align(predictions, golden)
    items, pending, prompts, validators = [], [], [], []
    for prediction, gold in aligned:
        item = result_item(prediction, error="Invalid upstream prediction or golden")
        item["metrics"] = None
        item["both_empty"] = False
        items.append(item)
        if (not valid_candidate(prediction) or prediction.get("status") != "ok"
                or gold.get("status") != "ok"
                or validate_spans(prediction.get("result"), prediction) is None
                or validate_spans(gold.get("result"), gold, golden=True) is None):
            continue
        n, m = len(prediction["result"]["spans"]), len(gold["result"]["spans"])
        if not n or not m:
            item.update(status="ok", error=None, result={"pairs": [], "matching": []},
                        metrics=metrics(0, n, m, empty_item=(n == m == 0)), both_empty=n == m == 0)
            continue
        pending.append((len(items) - 1, n, m))
        prompts.append(matching_prompt(prediction, gold))
        validators.append(lambda value, n=n, m=m: validate_pairs(value, n, m))
    results = generate(prompts, validators, backend="oss", model_path=model_path, **options)
    for (index, n, m), result in zip(pending, results):
        item = items[index]
        item["generation"] = result
        if result["parsed"] is None:
            item["error"] = "Invalid semantic matching response after retries"
            continue
        matching = maximum_matching(result["parsed"]["pairs"], n)
        tp = len(matching)
        item.update(status="ok", error=None, result={**result["parsed"], "matching": matching},
                    metrics=metrics(tp, n - tp, m - tp))
    by_dataset = {did: summarize([item for item in items if item["data_id"] == did])
                  for did in dict.fromkeys(item["data_id"] for item in items)}
    return items, dict(overall=summarize(items), by_dataset=by_dataset)
