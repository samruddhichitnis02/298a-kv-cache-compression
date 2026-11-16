"""Figures for the RestoreKV report section (SAM-100).

    python members/anees/make_figures.py

Reads committed result files only and writes three PNGs to report/figures/:

- restorekv_score_vs_memory.png: RestoreKV (bf16, 4-bit, 2-bit), KVzip and SnapKV
  on LongBench, score against memory actually used.
- restorekv_quantization.png: at equal memory, bf16 vs 4-bit vs 2-bit RestoreKV,
  with 95% paired bootstrap intervals for the change from the full cache.
- restorekv_length_sweep.png: RULER at 4K, 8K and 16K for full cache, RestoreKV
  and KVzip at ratio 0.95.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "results" / "restorekv" / "longbench"
OUT = REPO / "report" / "figures"
MODEL = "Llama-3.1-8B-Instruct"
COLORS = {"RestoreKV": "#2E5C8A", "RestoreKV 4-bit": "#4E9A57", "RestoreKV 2-bit": "#9BC47A",
          "KVzip": "#7FA7CF", "SnapKV": "#9A9A9A", "Full cache": "#B33333"}


def load(path):
    return {r["record_id"]: r for r in map(json.loads, open(path)) if r}


BASE = load(REPO / "results" / "longbench" / f"{MODEL}_none_0.0.jsonl")
TASKS = defaultdict(list)
for i, r in BASE.items():
    TASKS[r["task"]].append(i)
BASE_KV = sum(r["kv_cache_gb"] for r in BASE.values()) / len(BASE)


def macro(rows):
    return sum(100 * sum(rows[i]["score"] for i in ids) / len(ids) for ids in TASKS.values()) / len(TASKS)


def ci_vs_base(rows, draws=2000, seed=0):
    rng, diffs = random.Random(seed), []
    for _ in range(draws):
        d = []
        for ids in TASKS.values():
            pick = [rng.choice(ids) for _ in ids]
            d.append(100 * sum(rows[i]["score"] - BASE[i]["score"] for i in pick) / len(pick))
        diffs.append(sum(d) / len(d))
    diffs.sort()
    return diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def memory_pct(rows):
    return 100 * sum(r["kv_cache_gb"] for r in rows.values()) / len(rows) / BASE_KV


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)


def score_vs_memory():
    series = {"RestoreKV": "restorekv_{}", "KVzip": "kvzip_{}", "SnapKV": "snapkv_{}"}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, pat in series.items():
        pts = sorted((memory_pct(rows), macro(rows)) for rows in
                     (load(LB / f"{MODEL}_{pat.format(r)}.jsonl") for r in ("0.75", "0.8", "0.9", "0.95")))
        ax.plot(*zip(*pts), "-o", color=COLORS[name], label=name, ms=5)
    for bits in ("4", "2"):
        pts = sorted((memory_pct(rows), macro(rows)) for rows in
                     (load(LB / f"{MODEL}_restorekv_{bits}bit_{r}.jsonl") for r in ("0.9", "0.95")))
        ax.plot(*zip(*pts), "--s", color=COLORS[f"RestoreKV {bits}-bit"], label=f"RestoreKV {bits}-bit", ms=6)
    ax.axhline(macro(BASE), color=COLORS["Full cache"], ls=":", lw=1.2)
    ax.text(27.5, macro(BASE) + 0.5, f"full cache {macro(BASE):.2f}", ha="right", color=COLORS["Full cache"], fontsize=8)
    ax.set_xlim(4, 28)
    ax.set_xlabel("KV-cache memory used (% of full cache)")
    ax.set_ylabel("LongBench macro score (800 records)")
    ax.set_title("RestoreKV against eviction baselines")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.savefig(OUT / "restorekv_score_vs_memory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def quantization():
    budgets = [("0.95", "6.4%"), ("0.9", "11.4%")]
    variants = [("bf16", "restorekv_{}", "RestoreKV"), ("4-bit", "restorekv_4bit_{}", "RestoreKV 4-bit"),
                ("2-bit", "restorekv_2bit_{}", "RestoreKV 2-bit")]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.26
    for j, (label, pat, color_key) in enumerate(variants):
        for k, (ratio, mem) in enumerate(budgets):
            rows = load(LB / f"{MODEL}_{pat.format(ratio)}.jsonl")
            delta = macro(rows) - macro(BASE)
            lo, hi = ci_vs_base(rows)
            x = k + (j - 1) * width
            ax.bar(x, delta, width, color=COLORS[color_key], label=label if k == 0 else None)
            ax.errorbar(x, delta, yerr=[[delta - lo], [hi - delta]], color="#333333", capsize=3, lw=1)
            ax.text(x, min(lo, delta) - 0.6, f"{macro(rows):.2f}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(budgets)), [f"{mem} memory\n(ratio {r})" for r, mem in budgets])
    ax.set_ylabel("Change from full cache (points, 95% CI)")
    ax.set_title("Same memory: few bf16 tokens vs more quantized tokens")
    ax.set_ylim(-16, 2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.savefig(OUT / "restorekv_quantization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def length_sweep():
    lengths = ["4096", "8192", "16384"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, stem in (("Full cache", "none_0.0"), ("RestoreKV", "restorekv_0.95"), ("KVzip", "kvzip_0.95")):
        ys = [json.load(open(REPO / "results" / "restorekv" / "ruler" / f"{MODEL}_{n}_{stem}.summary.json"))
              ["scores"]["average"] for n in lengths]
        ax.plot(range(len(lengths)), ys, "-o", color=COLORS[name], label=name, ms=5)
        for x, y in enumerate(ys):
            ax.text(x, y + 1.5, f"{y:.1f}", ha="center", fontsize=8, color=COLORS[name])
    ax.set_xticks(range(len(lengths)), ["4K", "8K", "16K"])
    ax.set_ylim(50, 102)
    ax.set_xlabel("Context length (RULER, 13 tasks)")
    ax.set_ylabel("RULER average")
    ax.set_title("RestoreKV vs KVzip at 5% budget as context grows")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.savefig(OUT / "restorekv_length_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    score_vs_memory()
    quantization()
    length_sweep()
    print(f"wrote 3 figures to {OUT}")
