"""Shared JSONL reading and atomic artifact writing."""

import json
from pathlib import Path


def iter_jsonl(path, *, require_object=False):
    """Yield nonblank JSONL values with file and line context on invalid input."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if require_object and not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield row


def read_jsonl(path):
    """Read nonblank JSONL values into a list."""
    return list(iter_jsonl(path))


def iter_jsonl_objects(data_path):
    """Stream annotation objects from an existing JSONL file."""
    path = Path(data_path)
    if path.suffix != ".jsonl" or not path.is_file():
        raise ValueError("data_path must be an existing JSONL file")
    yield from iter_jsonl(path, require_object=True)


def write_jsonl(path, rows):
    """Replace a complete artifact after generation; never resume or append requests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                 for row in rows), encoding='utf-8')
    temporary.replace(path)
