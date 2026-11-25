"""Do different task types need different cache strategies? Analysis of committed LongBench runs.

    python results/comparison/task_adaptive.py

Writes results/comparison/TASK_ADAPTIVE.md and task_adaptive.json. No GPU and no new runs: it only
reads the 800-record result files already in the repo.

Part 1, strategy by task type. For each compression ratio, the score lost against the full cache
in each LongBench category, for every method that was run at that ratio.

Part 2, budget by task. For each method, every task is allowed its own compression ratio, with the
total memory held at or below that of one fixed ratio. Two numbers are reported:
  - upper bound: the allocation is chosen and scored on the same records (optimistic);
  - held out: each task's records are split in two halves, the allocation is chosen on one half
    and scored on the other, and the two directions are averaged.
The allocation uses the task label, which a deployed system does not have, so both numbers bound
what a label-free rule can reach.

Part 3, the rule without the task label (harness/adaptive.py). Prompts are grouped into content types
from the prompt text alone, and each type gets one setting at the same average memory. Held-out
numbers are the mean over 20 random splits. This part also covers the families where the choice is
the bit-width or the method variant rather than the keep ratio.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from compare_methods import BASELINE, CATEGORIES, REPO, load

sys.path.insert(0, str(REPO / "harness"))
import adaptive  # noqa: E402

OUT = Path(__file__).resolve().parent
RATIOS = ["0.75", "0.8", "0.9", "0.95"]
# name -> (label, results folder, file tag)
METHODS = {
    "restorekv_4bit": ("RestoreKV 4-bit", "restorekv", "restorekv_4bit"),
    "restorekv_2bit": ("RestoreKV 2-bit", "restorekv", "restorekv_2bit"),
    "restorekv": ("RestoreKV", "restorekv", "restorekv"),
    "kvzip": ("KVzip", "restorekv", "kvzip"),
    "rdkv": ("RDKV", "rdkv", "rdkv"),
    "snapkv": ("SnapKV", "restorekv", "snapkv"),
    "triattention": ("TriAttention", "triattention", "triattention"),
}
BUDGET_METHODS = ["restorekv", "kvzip", "rdkv", "snapkv", "triattention"]  # run at all four ratios
UNIT = 0.02  # GB step of the allocation table
REPEATS = 20
DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"


def result(folder, tag, ratio):
    return REPO / folder / f"Llama-3.1-8B-Instruct_{tag}_{ratio}.jsonl"


def ratios(folder, tag):
    return [result(folder, tag, r) for r in RATIOS]


RESTOREKV = "results/restorekv/longbench"
TURBOQUANT = "members/vedant/results/sem1/fixed"
# label, what the rule chooses between, result files, the fixed settings whose memory is the budget
RULE_FAMILIES = [
    ("RestoreKV", "keep ratio", ratios(RESTOREKV, "restorekv"), ["0.9", "0.8"]),
    ("KVzip", "keep ratio", ratios(RESTOREKV, "kvzip"), ["0.9", "0.8"]),
    ("RDKV", "bit budget", ratios("results/rdkv/longbench", "rdkv"), ["0.9", "0.8"]),
    ("SnapKV", "keep ratio", ratios(RESTOREKV, "snapkv"), ["0.9", "0.8"]),
    ("TriAttention", "keep ratio", ratios("results/triattention/longbench", "triattention"), ["0.9", "0.8"]),
    ("RestoreKV with quantization", "keep ratio and bit-width (16, 4 or 2)",
     ratios(RESTOREKV, "restorekv") + [result(RESTOREKV, f"restorekv_{b}bit", r) for b in (4, 2) for r in ("0.9", "0.95")],
     ["restorekv_4bit@0.95", "restorekv_4bit@0.9"]),
    ("TurboQuant (MSE)", "bit setting (2.5, 3, 3.5 or 4)",
     [result(TURBOQUANT, f"turboquant_mse_{b}bit", "0.0") for b in ("2.5", "3", "3.5", "4")],
     ["turboquant_mse_3bit@0.0"]),
    ("ResidualKV", "rebuild ratio", ratios("results/residualkv/longbench", "residualkv"), ["0.9"]),
]


def memory(row):
    """Computed cache size of one record; RDKV's own column includes its quantizer metadata."""
    return row.get("rdkv_kv_cache_gb", row["kv_cache_gb"])


def run_file(method, ratio):
    _, folder, tag = METHODS[method]
    return REPO / "results" / folder / "longbench" / f"Llama-3.1-8B-Instruct_{tag}_{ratio}.jsonl"


def task_table(rows, ids):
    """task -> (mean score 0-100, total GB) over the given record ids."""
    score, gb, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for i in ids:
        r = rows[i]
        score[r["task"]] += 100 * r["score"]
        gb[r["task"]] += memory(r)
        n[r["task"]] += 1
    return {t: (score[t] / n[t], gb[t]) for t in n}


def allocate(tables, budget_gb):
    """Pick one ratio per task to maximise the macro score with total memory <= budget_gb.

    tables: ratio -> task -> (score, GB). Exact dynamic programme over memory in UNIT steps.
    """
    tasks = sorted(next(iter(tables.values())))
    steps = round(budget_gb / UNIT)
    best = [None] * (steps + 1)
    best[0] = (0.0, {})
    for task in tasks:
        nxt = [None] * (steps + 1)
        for ratio, table in tables.items():
            score, gb = table[task]
            cost = round(gb / UNIT)
            for used in range(steps - cost + 1):
                if best[used] is None:
                    continue
                cand = best[used][0] + score
                if nxt[used + cost] is None or cand > nxt[used + cost][0]:
                    nxt[used + cost] = (cand, {**best[used][1], task: ratio})
        best = nxt
    return max((b for b in best if b is not None), key=lambda b: b[0])[1]


def evaluate(tables, choice):
    tasks = list(choice)
    score = sum(tables[choice[t]][t][0] for t in tasks) / len(tasks)
    gb = sum(tables[choice[t]][t][1] for t in tasks)
    return score, gb


def main():
    base = load(BASELINE)
    ids = sorted(base)
    by_task = defaultdict(list)
    for i in ids:
        by_task[base[i]["task"]].append(i)
    halves = [[i for t in by_task.values() for i in t[k::2]] for k in (0, 1)]
    base_table = task_table(base, ids)
    full_gb = sum(gb for _, gb in base_table.values())
    full_score = sum(s for s, _ in base_table.values()) / len(base_table)

    runs = {}
    for method in METHODS:
        for ratio in RATIOS:
            path = run_file(method, ratio)
            if path.exists():
                rows = load(path)
                if set(rows) == set(base):
                    runs[method, ratio] = rows

    def category_scores(table):
        return {c: sum(table[t][0] for t in ts) / len(ts) for c, ts in CATEGORIES.items()}

    full_cat = category_scores(base_table)
    cats = list(CATEGORIES)
    md = ["# Task-adaptive analysis on the LongBench validation set", "",
          "Generated by `results/comparison/task_adaptive.py` from committed result files (800 records, "
          "16 tasks). No new runs.", "",
          "## Part 1. Strategy by task type", "",
          "Points lost against the full cache in each category (negative is worse). The best method in "
          "each column is in bold. Categories hold two or three tasks (100 to 150 records), so "
          "differences of one or two points are within noise.", "",
          "Full cache: " + ", ".join(f"{c} {full_cat[c]:.1f}" for c in cats) + ".", ""]
    part1 = {}
    for ratio in reversed(RATIOS):
        present = [m for m in METHODS if (m, ratio) in runs]
        tables = {m: task_table(runs[m, ratio], ids) for m in present}
        drops = {m: {c: v - full_cat[c] for c, v in category_scores(tables[m]).items()} for m in present}
        mem = {m: 100 * sum(gb for _, gb in tables[m].values()) / full_gb for m in present}
        top = {c: max(present, key=lambda m: drops[m][c]) for c in cats}
        part1[ratio] = {"memory_pct": mem, "drops": drops, "best": top}
        md += [f"### Ratio {ratio} (about {min(mem.values()):.0f}% of the full cache)", "",
               "| Method | Memory | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
        for m in present:
            cells = [f"**{drops[m][c]:+.1f}**" if top[c] == m else f"{drops[m][c]:+.1f}" for c in cats]
            md.append(f"| {METHODS[m][0]} | {mem[m]:.1f}% | " + " | ".join(cells) + " |")
        md.append("")

    md += ["## Part 2. Budget by task", "",
           "Each task may use its own ratio (0.75, 0.8, 0.9 or 0.95) of the same method, with total memory "
           "held at or below that of one fixed ratio. \"Upper bound\" chooses and scores on the same "
           "records. \"Held out\" chooses on half of each task's records and scores on the other half.", "",
           "| Method | Budget | Fixed ratio | Per-task, held out | Per-task, upper bound | Held-out memory |",
           "|---|---|---|---|---|---|"]
    part2 = {}
    for method in BUDGET_METHODS:
        if not all((method, r) in runs for r in RATIOS):
            continue
        whole = {r: task_table(runs[method, r], ids) for r in RATIOS}
        half = [{r: task_table(runs[method, r], h) for r in RATIOS} for h in halves]
        for target in ("0.9", "0.8"):
            fixed, budget = evaluate(whole, {t: target for t in base_table})
            choice = allocate(whole, budget)
            upper, _ = evaluate(whole, choice)
            held, held_gb, fixed_gb = [], 0.0, 0.0
            for fit, test in ((0, 1), (1, 0)):
                _, b = evaluate(half[fit], {t: target for t in base_table})
                c = allocate(half[fit], b)
                s, gb = evaluate(half[test], c)
                held.append(s)
                held_gb += gb
                fixed_gb += evaluate(half[test], {t: target for t in base_table})[1]
            held_score = sum(held) / 2
            part2[f"{method}@{target}"] = {
                "budget_pct": 100 * budget / full_gb, "fixed": fixed, "held_out": held_score,
                "upper_bound": upper, "held_out_memory_vs_fixed_pct": 100 * held_gb / fixed_gb,
                "choice": choice}
            md.append(f"| {METHODS[method][0]} | {100 * budget / full_gb:.1f}% | {fixed:.2f} | "
                      f"{held_score:.2f} ({held_score - fixed:+.2f}) | {upper:.2f} ({upper - fixed:+.2f}) | "
                      f"{100 * held_gb / fixed_gb:.0f}% of fixed |")
    md += ["", f"Full cache: {full_score:.2f}.", "",
           "### Which tasks get more memory", "",
           "Ratio chosen for each task at the budget of fixed ratio 0.9 (upper-bound allocation). "
           "A lower ratio means more of the cache is kept.", "",
           "| Task | " + " | ".join(METHODS[m][0] for m in BUDGET_METHODS if f"{m}@0.9" in part2) + " |",
           "|---|" + "---|" * sum(f"{m}@0.9" in part2 for m in BUDGET_METHODS)]
    for category, tasks in CATEGORIES.items():
        for task in tasks:
            cells = [part2[f"{m}@0.9"]["choice"][task] for m in BUDGET_METHODS if f"{m}@0.9" in part2]
            md.append(f"| {task} ({category}) | " + " | ".join(cells) + " |")
    md += ["", "## Part 3. The rule without the task label", "",
           f"Prompts are grouped into {adaptive.CLUSTERS} content types from the prompt text alone (length and text "
           "statistics), and each type gets one setting, with average memory equal to the fixed setting's. "
           f"\"Adaptive\" is scored on records the rule was not fitted on, averaged over {REPEATS} random splits; "
           "the ± is the spread of the gain across those splits. \"Per task\" is the same test with the task label known.", "",
           "| Method | The rule chooses | Memory | Fixed | Adaptive, no task label | Per task, held out | Per task, upper bound |",
           "|---|---|---|---|---|---|---|"]
    part3 = {}
    records = adaptive.read_records(DATA) if DATA.exists() else None
    if records is None:
        md += [f"| (not computed: {DATA.relative_to(REPO)} is not present; the prompt text is needed) | | | | | | |"]
    else:
        for label, chooses, files, targets in RULE_FAMILIES:
            if not all(f.exists() for f in files):
                continue
            family = adaptive.load_runs(files)
            for target in targets:
                r = adaptive.evaluate(family, records, target, repeats=REPEATS)
                gain = r["adaptive_held_out"] - r["fixed"]
                part3[f"{label}@{target}"] = r
                md.append(f"| {label} | {chooses} | {100 * r['memory_gb_mean'] * len(ids) / full_gb:.1f}% | {r['fixed']:.2f} | "
                          f"{r['adaptive_held_out']:.2f} ({gain:+.2f} ± {r['adaptive_gain_sd']:.2f}) | "
                          f"{r['per_task_held_out']:.2f} | {r['per_task_upper_bound']:.2f} |")
    md += ["", "## Limits", "",
           "- Part 2 uses the task label. Part 3 does not, but its number of content types was chosen on these "
           "validation results, where gains level off between 16 and 32 types.",
           "- Memory is the computed cache size (document plus uncompressed question), not measured GPU memory.",
           "- ResidualKV and TurboQuant are in Part 3 only: their memory does not line up with the four ratios.",
           "- TriAttention uses the Nov 10 calibration, which likely understates the method.", ""]

    (OUT / "TASK_ADAPTIVE.md").write_text("\n".join(md), encoding="utf-8")
    json.dump({"full_cache": {"score": full_score, "categories": full_cat}, "by_task_type": part1,
               "by_task_budget": part2, "rule_without_task_label": part3},
              open(OUT / "task_adaptive.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
