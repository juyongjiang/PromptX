import os
import gzip
import json
import jsonlines
from typing import List


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items

def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(path: str, data: List[dict], append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)

def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v

# generator that returns the index and item in the dataset.
# if the results_path exists, it will skip all items that have been processed before.
def enumerate_resume(dataset, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        # count the number of items in the results file
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1
        # yield the remaining items
        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item

def resume_success_count(results_path) -> int:
    count = 0
    if os.path.exists(results_path):
        with jsonlines.open(results_path) as reader:
            for item in reader:
                if "is_solved" in item and item["is_solved"]:
                    count += 1
    return count