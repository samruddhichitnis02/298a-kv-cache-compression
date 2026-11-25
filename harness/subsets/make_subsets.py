"""Fixed record lists shared by every member, so timing and adaptive fitting use the same inputs.

    python harness/subsets/make_subsets.py

timing_100.json  100 validation records for serving measurements: 6 per task spread from the shortest
                 to the longest prompt, plus the 4 longest prompts not already picked.
train_300.json   300 records from outside the validation and test splits, for fitting the adaptive
                 rule: 18 or 19 per task, chosen with a fixed seed. Run them with
                 --data data/processed_longbench/processed_longbench.jsonl --records harness/subsets/train_300.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "harness"))

import needles  # noqa: E402

DATA = REPO / "data" / "processed_longbench"


def by_task(records):
    tasks = defaultdict(list)
    for record in records:
        tasks[record["task"]].append(record)
    return dict(sorted(tasks.items()))


def main():
    dev = [json.loads(line) for line in open(DATA / "processed_longbench_dev.jsonl", encoding="utf-8")]
    picked = []
    for records in by_task(dev).values():
        records.sort(key=lambda r: (r["prompt_tokens_after"], r["record_id"]))
        picked += [records[round(q * (len(records) - 1))] for q in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    chosen = {r["record_id"] for r in picked}
    rest = sorted((r for r in dev if r["record_id"] not in chosen),
                  key=lambda r: (-r["prompt_tokens_after"], r["record_id"]))
    picked += rest[:100 - len(picked)]
    (HERE / "timing_100.json").write_text(json.dumps(sorted(r["record_id"] for r in picked), indent=0))

    held_out = needles.held_out_ids()
    assert len(held_out) == 1600, "validation and test id lists not found"
    train = [r for r in map(json.loads, open(DATA / "processed_longbench.jsonl", encoding="utf-8"))
             if r["record_id"] not in held_out]
    rng = random.Random(0)
    sample = []
    for index, records in enumerate(by_task(train).values()):
        records.sort(key=lambda r: r["record_id"])
        sample += rng.sample(records, min(19 if index < 12 else 18, len(records)))
    (HERE / "train_300.json").write_text(json.dumps(sorted(r["record_id"] for r in sample), indent=0))

    tokens = sorted(r["prompt_tokens_after"] for r in picked)
    print(f"timing_100.json: {len(picked)} records, {tokens[0]} to {tokens[-1]} tokens, median {tokens[len(tokens) // 2]}")
    print(f"train_300.json: {len(sample)} records from {len(train)} outside validation and test, "
          f"{len(by_task(sample))} tasks")


if __name__ == "__main__":
    main()
