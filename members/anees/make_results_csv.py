"""Write Anees's frozen results (Nov 13) to results/anees.csv in the team format.

    python members/anees/make_results_csv.py

One row per run and task, plus one "macro" row per run (the task average used in
every table). Columns: method, budget, dataset, task, n, score, bytes_per_token,
kv_cache_gb, peak_gpu_gb, context_length.

- budget: the compression ratio (share of tokens dropped); 0.0 is the full cache.
- score: mean task score x100 (LongBench scorer or RULER string match).
- bytes_per_token: mean over records of kv_cache_gb / prompt tokens. RULER rows
  do not log prompt tokens, so the nominal context length is used there.
- kv_cache_gb, peak_gpu_gb: means over the records.
- context_length: mean prompt tokens (LongBench) or the nominal length (RULER).
"""

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "anees.csv"
SOURCES = [("longbench", REPO / "results" / "longbench" / "Llama-3.1-8B-Instruct_none_0.0.jsonl")]
SOURCES += [("longbench", Path(f)) for f in sorted(glob.glob(str(REPO / "results/restorekv/longbench/*.jsonl")))]
SOURCES += [("ruler", Path(f)) for f in sorted(glob.glob(str(REPO / "results/restorekv/ruler/*.jsonl")))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    out = []
    for dataset, path in SOURCES:
        name = path.stem.replace("Llama-3.1-8B-Instruct_", "")
        length = None
        if dataset == "ruler":
            length, name = name.split("_", 1)
            dataset = f"ruler_{length}"
        method, budget = name.rsplit("_", 1)
        rows = [json.loads(line) for line in open(path) if line.strip()]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        task_rows = []
        for task, rs in sorted(by_task.items()):
            tokens = [r.get("prompt_tokens") or int(length) for r in rs]
            task_rows.append({
                "method": method, "budget": float(budget), "dataset": dataset, "task": task, "n": len(rs),
                "score": round(100 * mean([r["score"] for r in rs]), 2),
                "bytes_per_token": round(mean([r["kv_cache_gb"] * 1e9 / t for r, t in zip(rs, tokens)
                                               if r.get("kv_cache_gb") is not None]) or 0, 1),
                "kv_cache_gb": round(mean([r.get("kv_cache_gb") for r in rs]), 4),
                "peak_gpu_gb": round(mean([r.get("peak_gpu_gb") for r in rs]), 2),
                "context_length": round(mean(tokens)),
            })
        macro = {k: task_rows[0][k] for k in ("method", "budget", "dataset")}
        macro.update(task="macro", n=len(rows),
                     score=round(100 * mean([mean([r["score"] for r in rs]) for rs in by_task.values()]), 2),
                     bytes_per_token=round(mean([t["bytes_per_token"] for t in task_rows]), 1),
                     kv_cache_gb=round(mean([r.get("kv_cache_gb") for r in rows]), 4),
                     peak_gpu_gb=round(mean([r.get("peak_gpu_gb") for r in rows]), 2),
                     context_length=round(mean([r.get("prompt_tokens") or int(length) for r in rows])))
        out += [macro] + task_rows
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {OUT}")
    for r in out:
        if r["task"] == "macro":
            print(f"{r['dataset']:12s} {r['method']:16s} {r['budget']:<5} {r['score']:6.2f}  {r['bytes_per_token']:8.1f} B/token")


if __name__ == "__main__":
    main()
