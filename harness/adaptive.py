"""Adaptive budgets: choose the compression setting per prompt, at a fixed average memory, without the task label.

The rule looks only at the prompt (its length and a few cheap text statistics, plus an optional
signal each method can supply). Prompts are grouped into content types by k-means on those features,
and each type gets one setting, chosen so the average memory equals that of a fixed setting. At run
time a new prompt goes to its nearest type and takes that type's setting.

Evaluate on result files that already exist (no GPU): each record has been run at every setting, and
decoding is greedy, so "what would the adaptive rule have scored" is just a per-record lookup.

    # held-out comparison of fixed, adaptive and the upper bounds, for one method
    python harness/adaptive.py evaluate --runs results/restorekv/longbench/Llama-3.1-8B-Instruct_restorekv_{0.75,0.8,0.9,0.95}.jsonl --target 0.9

    # fit a policy on training-sample runs and use it live
    python harness/adaptive.py fit --runs <one file per setting> --target 0.9 --out policy.json
    python harness/run_longbench.py --press restorekv --adaptive-policy policy.json

A setting is one result file. Runs of one press are named by their `compression_ratio` ("0.9"); when the
files mix presses or bit-widths they are named "<press>@<ratio>" ("restorekv_4bit@0.95"), so the rule can
choose the strategy as well as the budget. `--signal` adds a method's own per-record numbers (JSON:
record_id -> number or list) to the features. `--repeats` averages the held-out comparison over that many
random splits, because one split of 800 records is noisy.
"""

import argparse
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
UNIT = 0.02   # GB step of the allocation table
CLUSTERS = 24   # content types; gains level off between 16 and 32 on the validation results
SEED = 0


# --------------------------------------------------------------------------------------------
# Features: computed from the prompt alone
# --------------------------------------------------------------------------------------------
FEATURES = ["log_tokens", "log_question_chars", "compressibility", "digit_share", "symbol_share",
            "newline_share", "upper_share", "mean_word_length", "passage_markers", "question_mark"]


def features(record, signal=None):
    """Label-free description of one prompt. `record` needs `context`, `prompt` and a token count."""
    context = record["context"]
    question = record["prompt"][record["prompt"].rfind(context[-200:].strip()) + len(context[-200:].strip()):]
    sample = context[:20000]                      # the start of the document is enough for text statistics
    n = max(len(sample), 1)
    words = sample.split()
    values = [
        math.log(max(record.get("prompt_tokens_after") or len(record["prompt"]) / 4, 1)),
        math.log(max(len(question), 1)),
        len(zlib.compress(sample.encode("utf-8", "ignore"))) / n,
        sum(c.isdigit() for c in sample) / n,
        sum(c in "{}[]();=<>_#/\\*+|&" for c in sample) / n,
        sample.count("\n") / n,
        sum(c.isupper() for c in sample) / n,
        sum(map(len, words)) / max(len(words), 1),
        (sample.count("Passage ") + sample.count("Paragraph ")) / n * 1000,
        float("?" in question),
    ]
    if signal is not None:
        values += list(signal) if isinstance(signal, (list, tuple)) else [signal]
    return values


def kmeans(x, k, seed=SEED, steps=50):
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(steps):
        assign = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([x[assign == j].mean(0) if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers


def nearest(x, centers):
    return ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)


# --------------------------------------------------------------------------------------------
# Allocation: one setting per group, total memory under the budget
# --------------------------------------------------------------------------------------------
def memory(row):
    """Cache size of one record in GB: measured if the run packed it, else computed."""
    return row.get("rdkv_kv_cache_gb") or row["kv_cache_gb"]


def group_tables(runs, groups):
    """setting -> group -> (sum of scores, sum of GB) over the records of each group."""
    tables = {}
    for setting, rows in runs.items():
        table = defaultdict(lambda: [0.0, 0.0])
        for record_id, group in groups.items():
            table[group][0] += rows[record_id]["score"]
            table[group][1] += memory(rows[record_id])
        tables[setting] = {g: tuple(v) for g, v in table.items()}
    return tables


TOLERANCE = 0.001   # the allocation may exceed the budget by this share, to absorb rounding to UNIT


def allocate(tables, budget_gb, fallback=None):
    """Pick one setting per group to maximise the total score with total memory within the budget.

    Exact dynamic programme over memory in steps of UNIT. If rounding leaves no feasible choice,
    every group gets `fallback` (the fixed setting the budget came from).
    """
    group_ids = sorted(next(iter(tables.values())))
    steps = int(budget_gb * (1 + TOLERANCE) / UNIT)
    best = [None] * (steps + 1)
    best[0] = (0.0, {})
    for group in group_ids:
        nxt = [None] * (steps + 1)
        for setting, table in tables.items():
            score, gb = table[group]
            cost = round(gb / UNIT)
            for used in range(steps - cost + 1):
                if best[used] is None:
                    continue
                cand = best[used][0] + score
                if nxt[used + cost] is None or cand > nxt[used + cost][0]:
                    nxt[used + cost] = (cand, {**best[used][1], group: setting})
        best = nxt
    feasible = [b for b in best if b is not None]
    if not feasible:
        if fallback is None:
            fallback = min(tables, key=lambda s: sum(gb for _, gb in tables[s].values()))
        return {g: fallback for g in group_ids}
    return max(feasible, key=lambda b: b[0])[1]


def outcome(runs, choice):
    """Macro score (x100, mean over tasks) and total GB when each record uses its chosen setting."""
    by_task, gb = defaultdict(list), 0.0
    for record_id, setting in choice.items():
        row = runs[setting][record_id]
        by_task[row["task"]].append(row["score"])
        gb += memory(row)
    return 100 * sum(sum(v) / len(v) for v in by_task.values()) / len(by_task), gb


# --------------------------------------------------------------------------------------------
# Policy: fit, save, apply
# --------------------------------------------------------------------------------------------
def fit_policy(runs, records, target, ids, signals=None, clusters=CLUSTERS, seed=SEED, points=None):
    """Group the prompts in `ids` by content, then give each group a setting under the target's memory."""
    ids = list(ids)
    if points is None:
        points = {i: features(records[i], (signals or {}).get(i)) for i in ids}
    x = np.array([points[i] for i in ids], dtype=float)
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    centers = kmeans((x - mean) / std, clusters, seed)
    groups = dict(zip(ids, nearest((x - mean) / std, centers).tolist()))
    budget = sum(memory(runs[target][i]) for i in ids)
    choice = allocate(group_tables(runs, groups), budget, target)
    settings = [choice.get(g, target) for g in range(len(centers))]
    return {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "centers": centers.tolist(),
            "settings": settings, "budget": target, "uses_signal": bool(signals)}


def choose_setting(policy, record, signal=None):
    """The setting (a result-file name such as "0.9" or "restorekv_4bit@0.95") the policy gives this prompt."""
    x = (np.array([features(record, signal)], dtype=float) - np.array(policy["mean"])) / np.array(policy["std"])
    return policy["settings"][int(nearest(x, np.array(policy["centers"]))[0])]


def choose_ratio(policy, record, signal=None):
    """The compression ratio the policy gives this prompt, for policies over one press."""
    return float(str(choose_setting(policy, record, signal)).split("@")[-1])


def load_policy(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------------
# Evaluation on existing result files
# --------------------------------------------------------------------------------------------
def load_runs(paths):
    """setting name -> record_id -> row, for records present in every file.

    The name is the compression ratio when all files are one press at different ratios, and
    "<press>@<ratio>" otherwise.
    """
    loaded = []
    for path in paths:
        rows = {r["record_id"]: r for r in map(json.loads, open(path)) if not r.get("oom")}
        first = next(iter(rows.values()))
        loaded.append((first["press"], str(first["compression_ratio"]), rows))
    one_press = len({press for press, _, _ in loaded}) == 1 and len({ratio for _, ratio, _ in loaded}) == len(loaded)
    runs = {(ratio if one_press else f"{press}@{ratio}"): rows for press, ratio, rows in loaded}
    if len(runs) != len(loaded):
        raise ValueError("two result files have the same press and ratio")
    shared = set.intersection(*(set(rows) for rows in runs.values()))
    return {s: {i: rows[i] for i in shared} for s, rows in runs.items()}


def find_setting(runs, wanted):
    """The setting called `wanted`, or the one whose ratio is nearest to it."""
    if wanted in runs:
        return wanted
    try:
        return min(runs, key=lambda s: abs(float(s.split("@")[-1]) - float(wanted)))
    except ValueError:
        raise SystemExit(f"no setting named {wanted}; have {sorted(runs)}")


def halves(runs, seed=None):
    """Two halves with every task split evenly: by record id order, or shuffled within each task by `seed`."""
    by_task = defaultdict(list)
    for record_id, row in sorted(next(iter(runs.values())).items()):
        by_task[row["task"]].append(record_id)
    if seed is not None:
        rng = random.Random(seed)
        for ids in by_task.values():
            rng.shuffle(ids)
    return [[i for ids in by_task.values() for i in ids[k::2]] for k in (0, 1)]


def evaluate(runs, records, target, signals=None, clusters=CLUSTERS, repeats=1):
    """Fixed vs adaptive (label-free, held out) vs the two task-label bounds, at the target's memory.

    With repeats > 1 the held-out numbers are the mean over that many random splits, and
    `adaptive_gain_sd` is the spread of the adaptive gain across them.
    """
    ids = sorted(next(iter(runs.values())))
    fixed, fixed_gb = outcome(runs, {i: target for i in ids})
    tasks = {i: runs[target][i]["task"] for i in ids}
    upper = outcome(runs, _spread(allocate(group_tables(runs, tasks), fixed_gb, target), tasks))[0]
    points = {i: features(records[i], (signals or {}).get(i)) for i in ids}   # computed once, reused by every split

    adaptive, per_task, memory_share = [], [], []
    for repeat in range(repeats):
        first, second = halves(runs, None if repeats == 1 else repeat)
        scores, by_label, used, allowed = [], [], 0.0, 0.0
        for fit, test in ((first, second), (second, first)):
            policy = fit_policy(runs, records, target, fit, signals, clusters, points=points)
            choice = dict(zip(test, apply_policy(policy, [points[i] for i in test])))
            score, gb = outcome(runs, choice)
            scores.append(score)
            used += gb
            allowed += outcome(runs, {i: target for i in test})[1]
            budget = sum(memory(runs[target][i]) for i in fit)
            task_choice = allocate(group_tables(runs, {i: tasks[i] for i in fit}), budget, target)
            # a task with no records in the fitting half keeps the fixed setting
            by_label.append(outcome(runs, {i: task_choice.get(tasks[i], target) for i in test})[0])
        adaptive.append(sum(scores) / 2)
        per_task.append(sum(by_label) / 2)
        memory_share.append(used / allowed)
    gains = [a - fixed for a in adaptive]
    return {"target": target, "records": len(ids), "memory_gb_mean": fixed_gb / len(ids), "fixed": fixed,
            "adaptive_held_out": statistics.mean(adaptive), "per_task_held_out": statistics.mean(per_task),
            "per_task_upper_bound": upper, "adaptive_memory_vs_fixed": statistics.mean(memory_share),
            "adaptive_gain_sd": statistics.stdev(gains) if repeats > 1 else None, "repeats": repeats}


def _spread(choice, groups):
    return {i: choice[g] for i, g in groups.items()}


def apply_policy(policy, points):
    """The policy's setting for each feature vector in `points`."""
    x = (np.array(points, dtype=float) - np.array(policy["mean"])) / np.array(policy["std"])
    return [policy["settings"][j] for j in nearest(x, np.array(policy["centers"])).tolist()]


def read_records(path, ids=None):
    records = {}
    for line in open(path, encoding="utf-8"):
        record = json.loads(line)
        if ids is None or record["record_id"] in ids:
            records[record["record_id"]] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["evaluate", "fit"])
    parser.add_argument("--runs", nargs="+", required=True, help="one result .jsonl per setting, same records")
    parser.add_argument("--target", required=True, help="the fixed setting whose memory is the budget, e.g. 0.9")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="processed LongBench jsonl holding these records")
    parser.add_argument("--signal", type=Path, help="JSON: record_id -> number or list, a method's own signal")
    parser.add_argument("--clusters", type=int, default=CLUSTERS)
    parser.add_argument("--repeats", type=int, default=1, help="evaluate: average over this many random splits")
    parser.add_argument("--out", type=Path, help="fit: where to write the policy")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    target = find_setting(runs, args.target)
    ids = set(next(iter(runs.values())))
    records = read_records(args.data, ids)
    missing = ids - set(records)
    if missing:
        raise SystemExit(f"{len(missing)} records in the runs are not in {args.data}")
    signals = json.loads(args.signal.read_text()) if args.signal else None

    if args.command == "evaluate":
        result = evaluate(runs, records, target, signals, args.clusters, args.repeats)
        print(json.dumps(result, indent=1))
        print(f"\nfixed {result['fixed']:.2f} | adaptive, no task label, held out {result['adaptive_held_out']:.2f} "
              f"({result['adaptive_held_out'] - result['fixed']:+.2f}) at {100 * result['adaptive_memory_vs_fixed']:.0f}% "
              f"of the fixed memory | per task, held out {result['per_task_held_out']:.2f} | "
              f"per task, upper bound {result['per_task_upper_bound']:.2f}")
    else:
        if not args.out:
            parser.error("fit needs --out")
        policy = fit_policy(runs, records, target, sorted(ids), signals, args.clusters)
        policy["budget"] = target
        args.out.write_text(json.dumps(policy, indent=1))
        counts = defaultdict(int)
        for s in policy["settings"]:
            counts[s] += 1
        print(f"wrote {args.out}: {len(policy['centers'])} content types, settings {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
