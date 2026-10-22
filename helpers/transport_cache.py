"""Portable array-only caches; expensive cache misses require explicit opt-in."""
from functools import wraps
import hashlib
import json
from pathlib import Path
import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def cache_path(directory, revision, name, args, kwargs):
    description = json.dumps([revision, name, _jsonable(args), _jsonable(kwargs)], sort_keys=True)
    digest = hashlib.sha256(description.encode()).hexdigest()
    return Path(directory) / f"{name}_{digest}.npz"


def save_result(path, result):
    if isinstance(result, dict):
        payload = result.copy()
        kind = "dict"
    elif isinstance(result, tuple):
        payload = {f"item_{i}": value for i, value in enumerate(result)}
        kind = "tuple"
    else:
        payload = {"value": result}
        kind = "array"
    assert all(np.asarray(v).dtype.kind != "O" for v in payload.values())
    np.savez_compressed(path, _result_kind=np.array(kind), **payload)


def load_result(path):
    with np.load(path, allow_pickle=False) as data:
        kind = data['_result_kind'].item()
        values = {key: data[key].item() if data[key].ndim == 0 else data[key].copy()
                  for key in data.files if key != '_result_kind'}
    if kind == 'dict':
        return values
    if kind == 'tuple':
        return tuple(values[f'item_{i}'] for i in range(len(values)))
    return values['value']


def make_cache(directory, revision, allow_recompute):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            path = cache_path(directory, revision, function.__name__, args, kwargs)
            if path.exists():
                return load_result(path)
            if not allow_recompute():
                raise FileNotFoundError(
                    f"Missing transport cache {path.name}. Restore the distributed data folder "
                    "or explicitly set ALLOW_EXPENSIVE_RECOMPUTE = True."
                )
            print(f"Computing uncached {function.__name__}; result will be retained.", flush=True)
            result = function(*args, **kwargs)
            save_result(path, result)
            return result
        return run
    return decorate
