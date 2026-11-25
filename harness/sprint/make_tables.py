"""Turn the result files of the four review points into the tables for the report.

    python harness/sprint/make_tables.py            # every point that has results
    python harness/sprint/make_tables.py timing     # one point

Reads results/hybrid, results/timing, results/many and results/long, and writes one Markdown table per
point to results/sprint/<point>.md. Every number comes from the result files; nothing is estimated.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FULL_GB_PER_TOKEN = 2 * 32 * 8 * 128 * 2 / 1e9     # Llama-3.1-8B in bf16: keys and values, all layers
WORKING_GB = 2.0                                   # GPU memory left free for activations in the session count
NAMES = {"none": "Full cache", "snapkv": "SnapKV", "kvzip": "KVzip", "restorekv": "RestoreKV", "rdkv": "RDKV",
         "triattention": "TriAttention"}


def rows_of(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def macro(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["score"])
    return 100 * sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def method(press):
    return NAMES.get(press, "TurboQuant " + press.replace("turboquant_", "").replace("_", " ")
                     if press.startswith("turboquant_") else press)


def hybrid(folder):
    lines = ["| Method | Strategy | Records | Measured document cache | Score |", "|---|---|---|---|---|"]
    table = []
    for path in sorted(folder.glob("*.jsonl")):
        rows = [r for r in rows_of(path) if not r.get("oom")]
        if not rows or rows[0].get("document_cache_share_measured") is None:
            continue
        press, bits, name = rows[0]["press"], rows[0].get("pack_bits"), path.stem
        if press == "rdkv":
            strategy = "eviction only (bit-widths 0 and 16)" if "_bits0-16" in name else "hybrid, mixed bit-widths"
        elif press == "none":
            strategy = f"quantization only, {bits}-bit"
        else:
            strategy = "eviction only" if bits == 16 else f"hybrid, {bits}-bit"
        share = mean([r["document_cache_share_measured"] for r in rows])
        table.append((share, f"| {method(press)} | {strategy} | {len(rows)} | {100 * share:.1f}% | {macro(rows):.2f} |"))
    return lines + [line for _, line in sorted(table)]


def timing(folder):
    runs = {}
    for path in sorted(folder.glob("*_timing.jsonl")):
        rows = [r for r in rows_of(path) if not r.get("oom") and r.get("timing")]
        if rows:
            runs[path.stem] = rows
    base = next((rows for rows in runs.values() if rows[0]["press"] == "none"), None)
    base_prefill = {r["record_id"]: r["timing"]["prefill_s"] for r in base} if base else {}
    lines = ["| Method | Records | Prefill (s) | Overhead vs full cache (s) | Time to first token (s) | Tokens per second | "
             "Measured document cache | Memory | Sessions in 32 GB at 8K / 16K / 32K |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in runs.items():
        press = rows[0]["press"]
        t = lambda key: mean([r["timing"][key] for r in rows])                     # noqa: E731
        overhead = mean([r["timing"]["prefill_s"] - base_prefill[r["record_id"]] for r in rows
                         if r["record_id"] in base_prefill]) if base and press != "none" else None
        share = mean([r.get("document_cache_share_measured") for r in rows])
        packed = rows[0].get("pack_bits")
        label = method(press) + (f", packed {packed}-bit" if packed else "")
        real = press == "none" or packed or share < 0.99
        memory = "measured" if real else "simulated: tokens hidden, not freed"
        if rows[0].get("decoded_copy_gb") is not None:
            memory = "packed codes measured; a decoded copy is also held"
        model_gb = mean([r.get("model_gpu_gb") for r in rows]) or 16.0
        sessions = " / ".join(str(int((32 - model_gb - WORKING_GB) // (share * tokens * FULL_GB_PER_TOKEN)))
                              for tokens in (8192, 16384, 32768))
        lines.append(f"| {label} | {len(rows)} | {fmt(t('prefill_s'))} | {fmt(overhead)} | {fmt(t('ttft_s'))} | "
                     f"{fmt(t('tokens_per_s'), 1)} | {100 * share:.1f}% | {memory} | {sessions} |")
    lines += ["", f"Sessions = (32 GB - model - {WORKING_GB:.0f} GB working memory) / measured document cache at that length. "
                  "For methods marked simulated the count is for the cache as it is really held today."]
    return lines


def needle_runs(folder):
    for path in sorted(folder.glob("*.jsonl")):
        rows = rows_of(path)
        if rows:
            first = rows[0]
            label = method(first["press"])
            if "_firstq" in path.stem:
                label += ", compressed with the first question"
            packed = re.search(r"_pack(\d+)", path.stem)
            if packed:
                label += f", packed {packed.group(1)}-bit"
            yield label, first["length"], rows


def many(folder):
    lines = ["| Length | Method | Documents | Score over the first 1 / 2 / 4 / 8 / 16 questions | Worst question (16) | "
             "Seconds per question at 1 / 16 |", "|---|---|---|---|---|---|"]
    for label, length, rows in needle_runs(folder):
        rows = [r for r in rows if not r["oom"] and len(r["scores"]) > 1]
        if not rows:
            continue
        counts = [c for c in (1, 2, 4, 8, 16) if all(len(r["scores"]) >= c for r in rows)]
        score = lambda c: 100 * mean([sum(r["scores"][:c]) / c for r in rows])     # noqa: E731
        per_q = lambda c: mean([(r["prefill_s"] + r["pack_s"] + sum(q["first_token_s"] + q["decode_s"]
                                for q in r["question_timings"][:c])) / c for r in rows])   # noqa: E731
        last = counts[-1]
        lines.append(f"| {length} | {label} | {len(rows)} | {' / '.join(fmt(score(c), 1) for c in counts)} | "
                     f"{fmt(100 * mean([min(r['scores'][:last]) for r in rows]), 1)} | {fmt(per_q(1))} / {fmt(per_q(last))} |")
    return lines


def long_documents(folder):
    full = {}
    table = []
    for label, length, rows in needle_runs(folder):
        done = [r for r in rows if not r["oom"]]
        score = 100 * mean([sum(r["scores"]) / len(r["scores"]) for r in done]) if done else None
        if label == "Full cache":
            full[length] = score
        table.append((length, label, len(rows), len(rows) - len(done), score,
                      max((r["peak_gpu_gb"] or 0) for r in rows) or None))
    lines = ["| Length | Method | Documents | Out of memory | Score | Against full cache | Peak GPU (GB) |", "|---|---|---|---|---|---|---|"]
    for length, label, n, oom, score, peak in sorted(table, key=lambda t: (t[0], t[1] != "Full cache", t[1])):
        against = None if score is None or full.get(length) is None else score - full[length]
        lines.append(f"| {length} | {label} | {n} | {oom} | {fmt(score, 1)} | "
                     f"{'n/a' if against is None else f'{against:+.1f}'} | {fmt(peak, 1)} |")
    return lines


POINTS = {"hybrid": ("hybrid/longbench", hybrid, "Point 1: hybrid eviction + quantization at measured bytes"),
          "timing": ("timing/longbench", timing, "Point 3: serving measurements"),
          "many": ("many", many, "Point 4: many questions from one compressed cache"),
          "long": ("long", long_documents, "Point 4: long documents")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("points", nargs="*", default=list(POINTS))
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    out_dir = args.results / "sprint"
    for point in args.points:
        folder, build, title = POINTS[point]
        folder = args.results / folder
        if not folder.exists() or not any(folder.glob("*.jsonl")):
            print(f"{point}: no result files in {folder}")
            continue
        text = "\n".join([f"# {title}", "", f"From the result files in `{folder.relative_to(args.results.parent)}`.", ""] + build(folder)) + "\n"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{point}.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
