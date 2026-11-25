"""ResidualKV on the needle documents: many questions from one compressed cache, and long documents.

The shared runner (harness/run_needles.py) cannot drive ResidualKV, which runs on the authors' code.
This script does the same experiment with that code:

  1. the document is run through ResidualKV on its own, with no question in view, and the compressed
     cache is kept;
  2. each question is answered from a copy of that cache, so one answer cannot affect the next.

    python harness/needles.py --length 16384                                   # once; builds the documents
    python members/charvee/run_residualkv_needles.py --length 16384 --per-task 2      # check it runs
    python members/charvee/run_residualkv_needles.py --length 16384                   # the `many` task
    python members/charvee/run_residualkv_needles.py --length 32768 --tasks single,multikey,multivalue,multiquery

Results go to results/many/ in the same format as harness/run_needles.py, so harness/sprint/make_tables.py
reads them with the other methods. Rerunning skips documents already done.

NOT YET RUN ON A GPU. It was written from the authors' generation code (deltakv/get_chat_api.py,
`manual_generate`), which accepts an existing cache. Two things to check on the first run:
  - Step 2 passes the authors' compressed cache to `copy.deepcopy`. If that fails, the cache type is not
    copyable and this approach needs their help or a different one.
  - With `--per-task 2`, the scores should be far above zero. If every answer is empty or wrong, the
    question is probably not being attached to the cached document correctly.
"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(HERE))

import needles  # noqa: E402
from run_residualkv import BASE_CONFIG, COMPRESSOR, import_deltakv, tensor_bytes  # noqa: E402

OUT_DIR = REPO / "results" / "many"


class NoBos:
    """The tokenizer, but reporting no BOS token, so the authors' code does not put one before a question."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.bos_token = None

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def split_template(tokenizer, context):
    """The document wrapped in the chat template, and the text that follows a question (as kvpress does)."""
    separator = "#" * (len(context) + 10)
    text = tokenizer.apply_chat_template([{"role": "user", "content": context + separator}],
                                         add_generation_prompt=True, tokenize=False)
    document, suffix = text.split(separator)
    return document, suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--compressor", default=COMPRESSOR)
    parser.add_argument("--deltakv-repo", default=str(HERE / "DeltaKV"))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=needles.OUT_DIR)
    parser.add_argument("--compression-ratio", type=float, default=0.9)
    parser.add_argument("--tasks", default="many")
    parser.add_argument("--per-task", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    data_file = args.data_dir / f"needles_{args.length}.jsonl"
    if not data_file.exists():
        parser.error(f"{data_file} not found; run: python harness/needles.py --length {args.length}")
    tasks, counts, examples = set(args.tasks.split(",")), defaultdict(int), []
    for line in data_file.open(encoding="utf-8"):
        example = json.loads(line)
        if example["task"] in tasks and (args.per_task is None or counts[example["task"]] < args.per_task):
            counts[example["task"]] += 1
            examples.append(example)

    ratio = args.compression_ratio
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{args.model.split('/')[-1]}_{args.length}_residualkv_{ratio}.jsonl"
    done = {json.loads(line)["id"] for line in out_file.open()} if out_file.exists() else set()
    todo = [e for e in examples if e["id"] not in done]
    print(f"residualkv ratio={ratio} length={args.length} documents={len(examples)} ({len(done)} already done)", flush=True)

    get_generate_api = import_deltakv(args.deltakv_repo)
    from deltakv.get_chat_api import manual_generate
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(args.model)
    config = dict(BASE_CONFIG, decode_keep_tokens=1 - ratio, prefill_keep_tokens=1 - ratio)
    _, model = get_generate_api(model_path, config, deltakv_checkpoint_path=snapshot_download(args.compressor),
                                tokenizer_path=model_path, backend="hf", return_kv_cache=True, return_model=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    no_bos = NoBos(tokenizer)
    stop = model.generation_config.eos_token_id
    stop = stop if isinstance(stop, list) else [stop]
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    full_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    with out_file.open("a") as out:
        for i, example in enumerate(todo, 1):
            document, suffix = split_template(tokenizer, example["context"])
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            answers, timings, cache, oom = [], [], None, False
            try:
                # 1. the document alone: one forward pass builds and compresses the cache.
                #    The single token this generates is discarded and never enters the cache.
                _, cache = manual_generate(model, tokenizer, document, return_kv_cache=True,
                                           max_new_tokens=1, do_sample=False, eos_token_id=stop)
                torch.cuda.synchronize()
                prefill_s = time.perf_counter() - start
                # 2. every question from its own copy of that cache
                for question in example["questions"]:
                    asked = time.perf_counter()
                    text = manual_generate(model, no_bos, question["question"] + suffix + question["answer_prefix"],
                                           past_key_values=copy.deepcopy(cache), return_kv_cache=False,
                                           max_new_tokens=example["max_new_tokens"], do_sample=False,
                                           eos_token_id=stop)
                    torch.cuda.synchronize()
                    answers.append(text)
                    timings.append({"first_token_s": None, "decode_s": round(time.perf_counter() - asked, 4)})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom, answers = True, [""] * len(example["questions"])
                print(f"[{i}/{len(todo)}] {example['task']:11s} OOM", flush=True)
            scores = [needles.ruler_score(a, q["answers"]) for a, q in zip(answers, example["questions"])]
            tokens = len(tokenizer(document, add_special_tokens=False).input_ids)
            row = {"id": example["id"], "task": example["task"], "length": args.length, "press": "residualkv",
                   "compression_ratio": ratio, "answers": answers, "scores": scores, "oom": oom,
                   "seconds": round(time.perf_counter() - start, 2),
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            if not oom:
                # per-question time here covers the whole answer; first_token_s is filled so the shared tables can add it
                for t in timings:
                    t["first_token_s"] = 0.0
                measured = tensor_bytes(cache)
                row.update({"context_tokens": tokens, "prefill_s": round(prefill_s, 4), "pack_s": 0.0,
                            "question_timings": timings,
                            "document_cache_gb_measured": round(measured / 1e9, 6),
                            "document_cache_share_measured": round(measured / (tokens * full_per_token), 4)})
                print(f"[{i}/{len(todo)}] {example['task']:11s} score={sum(scores) / len(scores):.2f} "
                      f"({len(scores)} questions)  {row['seconds']:6.1f}s", flush=True)
            out.write(json.dumps(row) + "\n")
            out.flush()
            del cache
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
