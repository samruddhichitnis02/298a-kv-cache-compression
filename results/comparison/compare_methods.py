"""Compare every completed LongBench run against the full-cache baseline, at matched memory.

    python results/comparison/compare_methods.py

Reads every 800-record run under results/ and members/*/results/, keeps runs whose record IDs
match the baseline exactly, and writes results/comparison/COMPARISON.md, comparison.json and,
if matplotlib is installed, score_vs_memory.png and category_scores.png.

Memory is the run's mean computed kv_cache_gb as a share of the full-cache baseline's (document
plus uncompressed question; quantized methods include their packed codes and metadata). Methods
are compared at the memory they actually use, not at nominal ratios. Intervals are 95% paired
bootstrap intervals: records are resampled within each task, then task differences are
macro-averaged, so each interval matches the macro score it describes.
"""

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = REPO / "results" / "longbench" / "Llama-3.1-8B-Instruct_none_0.0.jsonl"
DRAWS, SEED = 2000, 0
CATEGORIES = {
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}
FAMILY = {"restorekv": "RestoreKV", "kvzip": "KVzip", "snapkv": "SnapKV", "triattention": "TriAttention",
          "turboquant": "TurboQuant", "rdkv": "RDKV", "residualkv": "ResidualKV"}


def load(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["record_id"]] = r
    return rows


def per_task(rows, ids=None):
    by = defaultdict(list)
    for i, r in rows.items():
        if ids is None or i in ids:
            by[r["task"]].append(100 * r["score"])
    return {t: sum(v) / len(v) for t, v in by.items()}


def macro(rows):
    t = per_task(rows)
    return sum(t.values()) / len(t)


def paired_ci(run, base, tasks):
    rng = random.Random(SEED)
    diffs = []
    for _ in range(DRAWS):
        d = []
        for ids in tasks.values():
            pick = [rng.choice(ids) for _ in ids]
            d.append(100 * sum(run[i]["score"] - base[i]["score"] for i in pick) / len(pick))
        diffs.append(sum(d) / len(d))
    diffs.sort()
    return diffs[int(0.025 * DRAWS)], diffs[int(0.975 * DRAWS) - 1]


def label(name):
    """Llama-3.1-8B-Instruct_turboquant_mse_3bit_0.0 -> ('TurboQuant', 'MSE 3-bit')"""
    body = name.replace("Llama-3.1-8B-Instruct_", "")
    press, ratio = body.rsplit("_", 1)
    family = FAMILY.get(press.split("_")[0], press)
    if press.startswith("turboquant"):
        setting = press.replace("turboquant_", "").replace("mse_", "MSE ").replace("bit", "-bit")
        setting = setting if setting.startswith("MSE") or setting == "adaptive" else "QJL " + setting
        return family, setting
    if press.startswith("restorekv_") and press.endswith("bit"):
        return family, f"{press.split('_')[1].replace('bit', '-bit')}, ratio {ratio}"
    return family, f"ratio {ratio}"


def main():
    base = load(BASELINE)
    base_kv = sum(r["kv_cache_gb"] for r in base.values()) / len(base)
    tasks = defaultdict(list)
    for i, r in base.items():
        tasks[r["task"]].append(i)
    files = sorted(set(glob.glob(str(REPO / "results" / "**" / "*.jsonl"), recursive=True)
                       + glob.glob(str(REPO / "members" / "*" / "results" / "**" / "*.jsonl"), recursive=True)))
    runs, skipped = [], []
    for f in files:
        name = Path(f).stem
        if f == str(BASELINE) or "ruler" in f or "smoke" in f or "followup" in f:
            continue
        rows = load(f)
        if set(rows) != set(base) or "score" not in next(iter(rows.values())):
            skipped.append((name, len(rows)))
            continue
        family, setting = label(name)
        kv = [r.get("kv_cache_gb") for r in rows.values()]
        mem = 100 * (sum(kv) / len(kv)) / base_kv if all(k is not None for k in kv) else None
        lo, hi = paired_ci(rows, base, tasks)
        pt = per_task(rows)
        runs.append({"run": name, "family": family, "setting": setting, "score": macro(rows),
                     "delta": macro(rows) - macro(base), "ci": [lo, hi], "memory_pct": mem,
                     "oom": sum(bool(r.get("oom")) for r in rows.values()),
                     "categories": {c: sum(pt[t] for t in ts) / len(ts) for c, ts in CATEGORIES.items()},
                     "source": str(Path(f).relative_to(REPO))})
    runs.sort(key=lambda r: (r["memory_pct"] or 0, -r["score"]))
    full_pt = per_task(base)
    full_cat = {c: sum(full_pt[t] for t in ts) / len(ts) for c, ts in CATEGORIES.items()}
    json.dump({"baseline": {"score": macro(base), "kv_cache_gb": base_kv, "categories": full_cat},
               "runs": runs, "skipped": skipped}, open(OUT / "comparison.json", "w"), indent=1)

    bands = [(0, 9, "About 6% of the full cache"), (9, 14, "About 11%"), (14, 17, "About 15%"),
             (17, 23, "About 18-22%"), (23, 32, "About 24-30%")]
    md = ["# Method comparison on the LongBench validation set", "",
          "Generated by `results/comparison/compare_methods.py` from every completed 800-record run. "
          f"Full cache (no compression): **{macro(base):.2f}**, mean KV cache {base_kv:.3f} GB.", "",
          "Memory = mean computed KV-cache size as a share of the full cache (document + uncompressed question). "
          "Δ = difference from full cache on the same 800 records; 95% paired bootstrap interval in brackets. "
          "An interval that contains 0 means no significant loss.", ""]
    for lo_b, hi_b, title in bands:
        sel = [r for r in runs if r["memory_pct"] is not None and lo_b <= r["memory_pct"] < hi_b]
        if not sel:
            continue
        md += [f"## {title}", "", "| Method | Setting | Memory | Score | Δ vs full [95% CI] | OOM |", "|---|---|---|---|---|---|"]
        for r in sorted(sel, key=lambda r: -r["score"]):
            md.append(f"| {r['family']} | {r['setting']} | {r['memory_pct']:.1f}% | {r['score']:.2f} | "
                      f"{r['delta']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['oom']} |")
        md.append("")
    md += ["## Scores by task category", "", "| Method | Setting | Memory | " + " | ".join(CATEGORIES) + " |",
           "|---|---|---|" + "---|" * len(CATEGORIES),
           "| Full cache | — | 100% | " + " | ".join(f"{full_cat[c]:.1f}" for c in CATEGORIES) + " |"]
    for r in runs:
        md.append(f"| {r['family']} | {r['setting']} | {r['memory_pct']:.1f}% | "
                  + " | ".join(f"{r['categories'][c]:.1f}" for c in CATEGORIES) + " |")
    md += ["", "## Not included yet", "",
           "Runs are added automatically when their 800-record files are committed. Missing at the time of writing: "
           "RDKV (result files not yet committed). ResidualKV compresses the question with the document and keeps about 30% of the cache at every ratio; see members/charvee/README.md.", ""]
    if skipped:
        md += ["Files skipped because their record IDs do not match the 800-record baseline: "
               + ", ".join(f"{n} ({k} rows)" for n, k in skipped), ""]
    (OUT / "COMPARISON.md").write_text("\n".join(md))
    print(f"{len(runs)} runs compared; {len(skipped)} skipped; wrote {OUT / 'COMPARISON.md'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped the figures")
        return
    colors = {"RestoreKV": "#2E5C8A", "KVzip": "#7FA7CF", "SnapKV": "#9A9A9A", "TriAttention": "#6BA368",
              "TurboQuant": "#E08A3C", "RDKV": "#B04A8F", "ResidualKV": "#8C6D31"}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for fam in colors:
        pts = [r for r in runs if r["family"] == fam]
        if pts:
            ax.scatter([r["memory_pct"] for r in pts], [r["score"] for r in pts], s=42, color=colors[fam], label=fam,
                       zorder=3, edgecolor="white", linewidth=0.6)
    ax.axhline(macro(base), color="#B33", ls="--", lw=1)
    ax.text(31.5, macro(base) + 0.6, f"full cache {macro(base):.2f}", ha="right", color="#B33", fontsize=8)
    best = max((r for r in runs if r["memory_pct"] < 9), key=lambda r: r["score"], default=None)
    if best:
        ax.annotate(f"{best['family']} {best['setting'].split(',')[0]}", (best["memory_pct"], best["score"]),
                    (best["memory_pct"] + 2, best["score"] - 6), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("KV-cache memory actually used (% of full cache)")
    ax.set_ylabel("LongBench macro score (800 records)")
    ax.set_xlim(0, 32)
    ax.set_title("Score vs memory for every completed run")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "score_vs_memory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote score_vs_memory.png")


if __name__ == "__main__":
    main()
