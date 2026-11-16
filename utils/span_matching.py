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



def iou(a, b):
    if a.get('source_start') is None or b.get('source_start') is None:
        return 0.
    overlap = max(0, min(a['source_end'], b['source_end']) - max(a['source_start'], b['source_start']))
    union = max(a['source_end'], b['source_end']) - min(a['source_start'], b['source_start'])
    return overlap / union if union else 0.


def location_edges(predicted, golden, mode='overlap'):
    edges = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(golden):
            if mode == 'exact':
                match = p.get('source_start') is not None and (p['source_start'], p['source_end']) == (g['source_start'], g['source_end'])
            elif mode == 'overlap':
                # Deterministic diagnostic only; semantic-location graph is primary.
                match = iou(p, g) >= .5
            else:
                raise ValueError(mode)
            if match:
                edges.append([i, j])
    return edges
