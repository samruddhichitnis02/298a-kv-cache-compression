"""AdaptiveKV (CacheTune, arXiv 2605.24022) on the shared LongBench dev set.

Each record's document is split into reusable chunks (one per "Passage N:" for the
multi-hop QA tasks, fixed-size token blocks otherwise). Chunks are encoded on their
own and ranked offline, then reused at their new positions with the top `ratio`
share of their tokens recomputed. The instruction before the document and the
question after it are uncached text and always computed.

    # smoke test: 2 records per task on the paper's five tasks, r = 15 %
    python members/charvee/adaptive_kv/run_adaptive_kv.py --per-task 2

    # the paper's baselines on the same chunking
    python members/charvee/adaptive_kv/run_adaptive_kv.py --mode full_recompute
    python members/charvee/adaptive_kv/run_adaptive_kv.py --mode full_reuse

    # selection ablations (Sec. 5.4)
    python members/charvee/adaptive_kv/run_adaptive_kv.py --selection random

    # hardware-aware ratio for a disk pool (Sec. 4.3): calibrate on 10 samsum records
    python members/charvee/adaptive_kv/run_adaptive_kv.py --tier disk --calibrate 10

Output: results/adaptive_kv/longbench/<model>_<mode>_<selection>_<tier>_r<ratio>.jsonl
(+ .summary.json). Reruns skip record_ids already written.

Note: AdaptiveKV does not shrink the KV cache. It cuts prefill compute (TTFT) for
reused documents; the cache held during decoding is full size. Compare it with
the team's compression methods on score and TTFT, not on kv_cache_gb.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(HERE))

import longbench  # noqa: E402
from adaptive_kv import AdaptiveKVEngine, KVChunk, calibrate_ratio  # noqa: E402

DEFAULT_DATA = REPO / "data" / "processed_longbench" / "processed_longbench_dev.jsonl"
OUT_DIR = REPO / "results" / "adaptive_kv" / "longbench"
PAPER_TASKS = "samsum,multi_news,musique,2wikimqa,hotpotqa"
PASSAGE = re.compile(r"(?=Passage \d+:\n)")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_segments(tok, record, chunk_tokens):
    """Tokenise a record as [uncached prefix, chunk text pieces..., uncached suffix].

    Pieces are tokenised separately (as an independently cached chunk would be),
    so the full-recompute baseline uses exactly the same token ids.
    """
    prompt = record["prompt"]
    no_template = record["task"] in longbench.NO_CHAT_TEMPLATE_TASKS
    if no_template:
        text = prompt
    else:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    ctx = record["context"]
    head, tail = ctx[:200].strip(), ctx[-200:].strip()
    start = text.find(head) if head else -1
    end = text.rfind(tail) + len(tail) if tail else -1
    if start < 0 or end <= start:
        raise ValueError(f"context not found in prompt for {record['record_id']}")
    pre, doc, post = text[:start], text[start:end], text[end:]

    pieces = [p for p in PASSAGE.split(doc) if p] if PASSAGE.search(doc) else [doc]
    chunks = []
    for p in pieces:
        ids = tok(p, add_special_tokens=False).input_ids
        chunks.extend(ids[i:i + chunk_tokens] for i in range(0, len(ids), chunk_tokens))
    pre_ids = tok(pre, add_special_tokens=no_template).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids
    return pre_ids, chunks, post_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tasks", default=PAPER_TASKS, help="comma-separated tasks, or 'all'")
    ap.add_argument("--per-task", type=int, default=None)
    ap.add_argument("--max-prompt-tokens", type=int, default=None)
    ap.add_argument("--mode", default="adaptivekv", choices=["adaptivekv", "full_recompute", "full_reuse"])
    ap.add_argument("--ratio", type=float, default=0.15, help="recompute ratio r (paper: 0.15)")
    ap.add_argument("--alpha", type=float, default=0.5, help="low-frequency cutoff (paper: 0.5)")
    ap.add_argument("--selection", default="low", choices=["low", "high", "l2", "random"])
    ap.add_argument("--tier", default="cpu", choices=["gpu", "cpu", "disk"])
    ap.add_argument("--pool-dir", default=str(REPO / "results" / "adaptive_kv" / "pool"))
    ap.add_argument("--chunk-tokens", type=int, default=1024, help="max tokens per cached chunk")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="pick r by golden-section search on this many samsum records first")
    ap.add_argument("--r-min", type=float, default=0.15)
    args = ap.parse_args()

    tasks = None if args.tasks == "all" else set(args.tasks.split(","))
    records = longbench.load_records(args.data, tasks=tasks, per_task=args.per_task,
                                     max_prompt_tokens=args.max_prompt_tokens)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    engine = AdaptiveKVEngine(model, tier=args.tier, alpha=args.alpha, selection=args.selection,
                              pool_dir=args.pool_dir)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    ratio = {"full_reuse": 0.0, "full_recompute": 1.0}.get(args.mode, args.ratio)
    calibration = None
    if args.mode == "adaptivekv" and args.calibrate:
        cal_records = longbench.load_records(args.data, tasks={"samsum"}, per_task=args.calibrate)
        cal_reqs = []
        for rec in cal_records:
            pre, chunks, post = build_segments(tok, rec, args.chunk_tokens)
            cal_reqs.append([pre, *[engine.encode_chunk(c, bos) for c in chunks], post])
        calibration = calibrate_ratio(engine, cal_reqs, r_min=args.r_min)
        ratio = calibration["r_star"]
        print(f"calibrated r* = {ratio:.3f}  (r0 = {calibration['r0']:.3f}, "
              f"t_c = {calibration['t_c']:.2e}, t_i = {calibration['t_i']:.2e})")
        engine.clear_pool()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.mode if args.mode != "adaptivekv" else f"adaptivekv_{args.selection}"
    rtag = "cal" if calibration else f"{ratio:g}"
    out_file = OUT_DIR / f"{args.model.split('/')[-1]}_{tag}_{args.tier}_r{rtag}.jsonl"
    done = {json.loads(l)["record_id"] for l in out_file.open()} if out_file.exists() else set()
    todo = [r for r in records if r["record_id"] not in done]
    print(f"device={device} mode={args.mode} selection={args.selection} tier={args.tier} "
          f"ratio={ratio:.3f} records={len(records)} ({len(done)} done)")

    with out_file.open("a") as out:
        for i, rec in enumerate(todo, 1):
            pre, chunks, post = build_segments(tok, rec, args.chunk_tokens)
            start = time.perf_counter()
            try:
                if args.mode == "full_recompute":
                    ids = pre + [t for c in chunks for t in c] + post
                    res, encode_s = engine.full_prefill(ids), 0.0
                else:
                    kv = [engine.encode_chunk(c, bos) for c in chunks]      # offline step
                    encode_s = sum(k.encode_seconds + k.rank_seconds for k in kv)
                    res = engine.prefill([pre, *kv, post], ratio=ratio)
                gen = engine.generate(res, longbench.MAX_NEW_TOKENS[rec["task"]], eos)
                prediction = tok.decode(gen, skip_special_tokens=True)
                oom = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prediction, oom, res, encode_s = "", True, None, 0.0
            engine.clear_pool()          # LongBench documents are not shared across records
            row = {
                "record_id": rec["record_id"],
                "task": rec["task"],
                "press": tag,
                "recompute_ratio": round(ratio, 4),
                "tier": args.tier,
                "prompt_tokens": res.seq_len if res else rec["prompt_tokens_after"],
                "n_chunks": len(chunks),
                "n_cached_tokens": res.n_cached if res else None,
                "n_recomputed_tokens": res.n_recomputed if res else None,
                "loaded_kv_gb": round(res.loaded_bytes / 1e9, 4) if res else None,
                "prediction": prediction,
                "score": 0.0 if oom else longbench.score(rec["task"], prediction, rec["answers"],
                                                         rec["all_classes"]),
                "oom": oom,
                "ttft_s": round(res.ttft, 4) if res else None,
                "offline_encode_s": round(encode_s, 4),
                "seconds": round(time.perf_counter() - start, 2),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {rec['task']:18s} score={row['score']:.3f} "
                  f"ttft={row['ttft_s']}s chunks={len(chunks)} tok={row['prompt_tokens']}")

    ids = {r["record_id"] for r in records}
    rows = [json.loads(l) for l in out_file.open()]
    rows = [r for r in rows if r["record_id"] in ids]
    summary = longbench.summarize(rows)
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    summary_file = out_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps({
        "args": vars(args), "device": device, "recompute_ratio": ratio, "calibration": calibration,
        "n_records": len(rows), "mean_ttft_s": round(sum(ttfts) / max(len(ttfts), 1), 4),
        "scores": summary}, indent=2, default=str))
    for task, value in summary.items():
        print(f"{task:22s} {value:6.2f}")
    print(f"mean TTFT {sum(ttfts) / max(len(ttfts), 1):.3f}s\nsaved {out_file} and {summary_file}")


if __name__ == "__main__":
    main()
