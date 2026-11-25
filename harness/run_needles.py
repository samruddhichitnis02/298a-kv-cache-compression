"""Long documents and many questions per cache: run a method on the needle documents from harness/needles.py.

Each document is compressed once, without seeing any question, and every question for it is answered
from that one cache. The `many` task has 16 questions per document; the other tasks have one.

    python harness/needles.py --length 32768                       # once per length; writes data/needles/
    python harness/run_needles.py --length 32768 --press kvzip --compression-ratio 0.9
    python harness/run_needles.py --length 65536 --press none --chunk-prefill 8192
    python harness/run_needles.py --length 32768 --press restorekv --pack-bits 4 --memory-budget 0.1 --tasks many

`--compress-with-first-question` shows what a question-aware method loses on later questions: the first
question is appended to the document before compressing, and the others must use what was kept for it.

Results go to results/needles/<model>_<length>_<press>_<setting>.jsonl, one line per document; rerunning
skips documents already done. The summary gives the score per task and, for `many`, the score over the
first 1, 2, 4, 8 and 16 questions with the time per question when the compression cost is shared.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

import needles  # noqa: E402
import serve  # noqa: E402
from run_longbench import code_commit, kv_bytes_per_token, make_press, pick_device  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "needles"
FOLLOW_UP = "\n\nNow answer a different question about the same text."
COUNTS = (1, 2, 4, 8, 16)


def summarize(rows):
    """Score per task (x100) and, for documents with several questions, how score and time change with their number."""
    by_task = defaultdict(list)
    for row in rows:
        if not row["oom"]:
            by_task[row["task"]].append(sum(row["scores"]) / len(row["scores"]))
    summary = {task: round(100 * sum(v) / len(v), 2) for task, v in sorted(by_task.items())}
    summary["average"] = round(sum(summary.values()) / len(summary), 2) if summary else 0.0
    many = [row for row in rows if not row["oom"] and len(row["scores"]) > 1]
    if many:
        curve = {}
        for count in COUNTS:
            use = [row for row in many if len(row["scores"]) >= count]
            if not use:
                continue
            seconds = [(row["prefill_s"] + row["pack_s"] + sum(q["first_token_s"] + q["decode_s"]
                                                              for q in row["question_timings"][:count])) / count
                       for row in use]
            curve[str(count)] = {
                "score": round(100 * sum(sum(row["scores"][:count]) / count for row in use) / len(use), 2),
                "worst_question": round(100 * sum(min(row["scores"][:count]) for row in use) / len(use), 2),
                "seconds_per_question": round(sum(seconds) / len(seconds), 3)}
        summary["by_question_count"] = curve
        last = [row["scores"][-1] for row in many]
        first = [row["scores"][0] for row in many]
        summary["first_question"] = round(100 * sum(first) / len(first), 2)
        summary["last_question"] = round(100 * sum(last) / len(last), 2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, required=True, help="document length in tokens; needs data/needles/needles_<length>.jsonl")
    parser.add_argument("--data-dir", type=Path, default=needles.OUT_DIR)
    parser.add_argument("--press", default="none")
    parser.add_argument("--compression-ratio", type=float, default=0.0)
    parser.add_argument("--tasks", default="", help="comma-separated: single, multikey, multivalue, multiquery, many (default: all)")
    parser.add_argument("--per-task", type=int, default=None)
    parser.add_argument("--pack-bits", type=int, default=None, choices=[2, 4, 8, 16])
    parser.add_argument("--memory-budget", type=float, default=None)
    parser.add_argument("--chunk-prefill", type=int, default=None, help="prefill in chunks of this many tokens (none, kvzip, restorekv)")
    parser.add_argument("--compress-with-first-question", action="store_true")
    parser.add_argument("--triattention-stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-tag", default="")
    args = parser.parse_args()
    if args.memory_budget is not None and not args.pack_bits:
        parser.error("--memory-budget needs --pack-bits")

    data_file = args.data_dir / f"needles_{args.length}.jsonl"
    if not data_file.exists():
        parser.error(f"{data_file} not found; run: python harness/needles.py --length {args.length}")
    tasks = set(args.tasks.split(",")) if args.tasks else None
    counts, examples = defaultdict(int), []
    for line in data_file.open(encoding="utf-8"):
        example = json.loads(line)
        if tasks and example["task"] not in tasks:
            continue
        if args.per_task is not None and counts[example["task"]] >= args.per_task:
            continue
        counts[example["task"]] += 1
        examples.append(example)

    ratio = args.compression_ratio if args.press != "none" else 0.0
    setting = f"budget{args.memory_budget}" if args.memory_budget is not None else str(ratio)
    mode = (f"_pack{args.pack_bits}" if args.pack_bits else "") + ("_firstq" if args.compress_with_first_question else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{args.model.split('/')[-1]}_{args.length}_{args.press}_{setting}{mode}{args.out_tag}.jsonl"
    done = {json.loads(line)["id"] for line in out_file.open()} if out_file.exists() else set()
    todo = [e for e in examples if e["id"] not in done]

    device = pick_device()
    print(f"device={device}  model={args.model}  length={args.length}  press={args.press}  setting={setting}{mode}  "
          f"documents={len(examples)} ({len(done)} already done)")
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    press = make_press(args.press, ratio, None, args.triattention_stats)
    if args.press.startswith("restorekv"):
        press.post_init_from_model(pipe.model)
        pipe.model.to(device)
    full = kv_bytes_per_token(pipe.model.config)

    with out_file.open("a") as out:
        for i, example in enumerate(todo, 1):
            context = example["context"]
            questions = [q["question"] for q in example["questions"]]
            prefixes = [q["answer_prefix"] for q in example["questions"]]
            if args.compress_with_first_question:
                context += questions[0]
                questions = [""] + [FOLLOW_UP + q for q in questions[1:]]
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                served = serve.serve(pipe, press, context, questions, prefixes, example["max_new_tokens"],
                                     args.pack_bits, args.memory_budget, args.chunk_prefill)
                oom = False
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                served, oom = None, True
                serve.clear_masks(pipe.model)
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{i}/{len(todo)}] {example['task']:11s} OOM")
            answers = served["answers"] if served else [""] * len(questions)
            scores = [needles.ruler_score(a, q["answers"]) for a, q in zip(answers, example["questions"])]
            row = {
                "id": example["id"], "task": example["task"], "length": args.length, "press": args.press,
                "compression_ratio": ratio, "answers": answers, "scores": scores, "oom": oom,
                "seconds": round(time.perf_counter() - start, 2),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            }
            if served:
                row.update({
                    "context_tokens": served["context_tokens"],
                    "prefill_s": round(served["prefill_s"], 4), "pack_s": round(served["pack_s"], 4),
                    "ttft_s": round(served["ttft_s"], 4),
                    "question_timings": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}
                                         for q in served["questions"]],
                    "eviction_ratio": served["eviction_ratio"],
                    "document_cache_gb_measured": round(served["document_cache_bytes"] / 1e9, 6),
                    "document_cache_share_measured": round(served["document_cache_bytes"]
                                                           / served["full_document_cache_bytes"], 4),
                    # computed size, as in the LongBench runs: kept share of the document plus one question
                    "kv_cache_gb": round((round((1 - ratio) * served["context_tokens"])
                                          + served["question_tokens"][0]) * full / 1e9, 4),
                })
            out.write(json.dumps(row) + "\n")
            out.flush()
            if not oom:
                print(f"[{i}/{len(todo)}] {example['task']:11s} score={sum(scores) / len(scores):.2f} "
                      f"({len(scores)} questions)  ttft={row['ttft_s']:.2f}s  {row['seconds']:6.1f}s")

    wanted = {e["id"] for e in examples}
    rows = [row for row in map(json.loads, out_file.open()) if row["id"] in wanted]
    summary = summarize(rows)
    out_file.with_suffix(".summary.json").write_text(json.dumps(
        {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "device": device,
         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
         "code_commit": code_commit(),
         "n_documents": len(rows), "oom": sum(r["oom"] for r in rows), "scores": summary}, indent=2))
    for key, value in summary.items():
        print(f"{key:20s} {value}")
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
