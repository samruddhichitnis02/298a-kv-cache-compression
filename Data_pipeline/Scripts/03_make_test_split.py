"""Fix a held-out test split for DATA 298B, disjoint from the 800-record dev set.

Run from the repo root after unzipping processed_longbench.zip into data/:

    python Data_pipeline/Scripts/03_make_test_split.py

Picks 50 records per task (seed 42) from the 2,931 cleaned records outside the dev set.
Records whose document (whitespace-normalised context) also appears in the dev set are used
only when a task has fewer than 50 document-disjoint records (NarrativeQA has none, QMSum 49),
and every such record is listed in the output so the overlap is disclosed, not hidden.
Writes Data_pipeline/processed_longbench/test_ids.json.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed_longbench"
OUT = REPO / "Data_pipeline" / "processed_longbench" / "test_ids.json"
PER_TASK, SEED = 50, 42


def doc_hash(record):
    return hashlib.sha256(" ".join(record["context"].split()).encode()).hexdigest()


def main():
    full = [json.loads(line) for line in open(DATA / "processed_longbench.jsonl")]
    dev = [json.loads(line) for line in open(DATA / "processed_longbench_dev.jsonl")]
    dev_ids = {r["record_id"] for r in dev}
    dev_docs = {doc_hash(r) for r in dev}
    by_task = defaultdict(lambda: ([], []))  # task -> (document-disjoint, shares a dev document)
    for r in full:
        if r["record_id"] not in dev_ids:
            by_task[r["task"]][doc_hash(r) in dev_docs].append(r["record_id"])

    rng = random.Random(SEED)
    test, overlap, per_task = [], [], {}
    for task in sorted(by_task):
        disjoint, shared = (sorted(ids) for ids in by_task[task])
        pick = rng.sample(disjoint, min(PER_TASK, len(disjoint)))
        extra = rng.sample(shared, PER_TASK - len(pick))
        test += pick + extra
        overlap += extra
        per_task[task] = {"records": len(pick) + len(extra), "shares_dev_document": len(extra)}

    assert len(test) == len(set(test)) == PER_TASK * len(by_task)
    assert not set(test) & dev_ids
    OUT.write_text(json.dumps({
        "seed": SEED, "per_task": PER_TASK, "n_records": len(test),
        "source": "processed_longbench.jsonl minus the 800 dev records",
        "leakage_check": {"record_id_overlap_with_dev": 0,
                          "records_sharing_a_dev_document": len(overlap),
                          "per_task": per_task},
        "shares_dev_document": sorted(overlap),
        "test_ids": sorted(test),
    }, indent=1) + "\n")
    print(f"{len(test)} test records, {len(overlap)} share a document with dev; wrote {OUT}")
    for task, row in per_task.items():
        print(f"  {task:22s} {row}")


if __name__ == "__main__":
    main()
