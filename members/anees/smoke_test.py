"""Run RestoreKV on one long prompt and compare it with the full cache and KVzip (SAM-93).

Builds a needle-in-a-haystack prompt, answers it under each setting, and records
the answer, whether the needle was found, wall time and peak GPU memory.

    # Lab GPU, Llama-3.1-8B-Instruct, ~8K-token prompt, keep 10% of the KV cache
    python members/anees/smoke_test.py

    # Quick check of the script on a small model (RestoreKV adapters exist only
    # for Llama-3.1-8B-Instruct and Qwen3-8B, so skip it here)
    python members/anees/smoke_test.py --model Qwen/Qwen2.5-0.5B-Instruct --presses none,kvzip --tokens 2000
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import pipeline

from kvpress import KVzipPress, RestoreKVPress

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NEEDLE = "The secret code for the vault is {code}. "
QUESTION = "What is the secret code for the vault? Answer with the number only."
OUT_DIR = Path(__file__).parent / "outputs"


def build_context(tokenizer, target_tokens, depth, code):
    filler_tokens = len(tokenizer(FILLER, add_special_tokens=False).input_ids)
    parts = [FILLER] * max(1, target_tokens // filler_tokens)
    parts.insert(int(len(parts) * depth), NEEDLE.format(code=code))
    return "".join(parts)


def make_press(name, compression_ratio):
    if name == "none":
        return None
    if name == "kvzip":
        return KVzipPress(compression_ratio=compression_ratio)
    if name == "restorekv":
        return RestoreKVPress(compression_ratio=compression_ratio)
    if name == "restorekv_plus":
        return RestoreKVPress(compression_ratio=compression_ratio, kvzip_plus_normalization=True)
    raise ValueError(f"unknown press: {name}")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--presses", default="none,kvzip,restorekv")
    parser.add_argument("--tokens", type=int, default=8000, help="approximate context length")
    parser.add_argument("--compression-ratio", type=float, default=0.9, help="fraction of the KV cache removed")
    parser.add_argument("--depth", type=float, default=0.5, help="needle position, 0 = start, 1 = end")
    parser.add_argument("--code", default="7341")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    device = pick_device()
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, dtype=torch.bfloat16)
    context = build_context(pipe.tokenizer, args.tokens, args.depth, args.code)
    context_tokens = len(pipe.tokenizer(context, add_special_tokens=False).input_ids)
    print(f"device={device}  model={args.model}  context={context_tokens} tokens  "
          f"compression_ratio={args.compression_ratio}")

    rows = []
    for name in args.presses.split(","):
        press = make_press(name, args.compression_ratio)
        if isinstance(press, RestoreKVPress):
            # Load the LoRA adapter now and move it to the model's device; otherwise it
            # is created on the CPU when the pipeline runs on Apple GPUs (mps).
            press.post_init_from_model(pipe.model)
            pipe.model.to(device)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        answer = pipe(context, question=QUESTION, press=press, max_new_tokens=args.max_new_tokens)["answer"]
        seconds = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
        row = {
            "press": name,
            "answer": answer.strip(),
            "found_needle": args.code in answer,
            "seconds": round(seconds, 2),
            "peak_gpu_gb": round(peak_gb, 2) if peak_gb is not None else None,
        }
        rows.append(row)
        print(f"{name:15s} found={row['found_needle']!s:5s} {row['seconds']:7.2f}s  "
              f"peak={row['peak_gpu_gb']} GB  answer={row['answer']!r}")

    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"smoke_{args.model.split('/')[-1]}_{context_tokens}tok.json"
    out_file.write_text(json.dumps({"args": vars(args), "device": device, "context_tokens": context_tokens,
                                    "results": rows}, indent=2))
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
