"""Deterministic semantic-edge validation, maximum matching and span metrics."""

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

