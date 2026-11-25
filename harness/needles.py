"""Needle documents for long contexts and many questions per cache (RULER-style, generated here).

The RULER build the team uses (simonjegou/ruler) stops at 16K tokens and asks one question per
document. This generator makes documents of any length, with up to 16 separate questions on one
document, so a compressed cache can be tested at 32K, 64K and beyond, and on many questions.

A document is real prose with short facts ("needles") inserted at spread-out depths:

    One of the special magic numbers for amber-lantern is: 4815162.

The prose comes from LongBench documents that are in neither the validation nor the test split, so
nothing here overlaps with the records methods are scored on. Tasks:

    single       1 needle, 1 question
    multikey     4 needles with different keys, 1 question about one of them
    multivalue   4 needles with the same key, 1 question asking for all four numbers
    multiquery   4 needles, 1 question asking for all four keys
    many         16 needles, 16 separate questions (one per needle), for many-questions-per-cache tests

    python harness/needles.py --length 32768 --per-task 20
    -> data/needles/needles_32768.jsonl   (data/ is not in git; the same command gives the same file)

Scoring is RULER's: the share of expected numbers that appear in the answer.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data" / "processed_longbench" / "processed_longbench.jsonl"
SPLIT_FILES = [REPO / "data" / "processed_longbench" / "dev_ids.json",
               REPO / "Data_pipeline" / "processed_longbench" / "test_ids.json"]
OUT_DIR = REPO / "data" / "needles"
PROSE_TASKS = {"narrativeqa", "gov_report", "qmsum", "multi_news", "multifieldqa_en", "qasper"}
TASKS = {"single": (1, 1), "multikey": (4, 1), "multivalue": (4, 1), "multiquery": (4, 1), "many": (16, 16)}
INSTRUCTION = ("Some special magic numbers are hidden within the following text. Make sure to memorize them. "
               "I will quiz you about the numbers afterwards.\n\n")
NEEDLE = "One of the special magic numbers for {key} is: {value}."
ADJECTIVES = ("amber brisk calm dusty eager faint gentle hollow icy jolly keen lunar mellow nimble odd proud quiet "
              "rustic silent tidy urban vivid windy young zesty ancient bold crimson dapper elder").split()
NOUNS = ("anchor basket candle dagger engine falcon garden harbor island jacket kettle lantern meadow needle orchard "
         "pillow quarry ribbon saddle tunnel umbrella valley wagon yacht zipper beacon cradle drawer ember fountain").split()
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def ruler_score(prediction, answers):
    """Share of the expected strings found in the prediction, in [0, 1]."""
    prediction = prediction.lower()
    return sum(answer.lower() in prediction for answer in answers) / len(answers)


def held_out_ids():
    ids = set()
    for path in SPLIT_FILES:
        if path.exists():
            loaded = json.loads(path.read_text())
            lists = [loaded] if isinstance(loaded, list) else [v for k, v in loaded.items()
                                                               if k.endswith("ids") and isinstance(v, list)]
            for values in lists:
                ids |= set(values)
    return ids


def prose_documents(source=SOURCE):
    """Distinct prose documents outside the validation and test splits, in a fixed order."""
    excluded, documents = held_out_ids(), []
    if len(excluded) < 1600:
        raise SystemExit("validation and test id lists not found; refusing to build haystacks that might overlap them")
    records = [json.loads(line) for line in open(source, encoding="utf-8")]
    digest = lambda r: hashlib.md5(r["context"].encode()).hexdigest()  # noqa: E731
    # A document can come with several questions; skip it if any of its records is held out.
    seen = {digest(r) for r in records if r["record_id"] in excluded}
    for record in records:
        if record["task"] not in PROSE_TASKS or digest(record) in seen:
            continue
        seen.add(digest(record))
        documents.append(" ".join(record["context"].split()))
    return sorted(documents, key=lambda d: hashlib.md5(d.encode()).hexdigest())


def haystack(documents, tokenizer, target_tokens, rng):
    """Sentences of prose adding up to about `target_tokens` tokens."""
    sentences, tokens = [], 0
    order = list(range(len(documents)))
    rng.shuffle(order)
    for index in order:
        for sentence in SENTENCE_END.split(documents[index]):
            if not sentence:
                continue
            count = len(tokenizer(" " + sentence, add_special_tokens=False).input_ids)
            if tokens + count > target_tokens:
                return sentences
            sentences.append(sentence)
            tokens += count
    return sentences


def make_example(task, index, length, documents, tokenizer, seed):
    rng = random.Random(f"{seed}-{task}-{length}-{index}")
    n_needles, n_questions = TASKS[task]
    keys = rng.sample([f"{a}-{n}" for a in ADJECTIVES for n in NOUNS], n_needles)
    if task == "multivalue":
        keys = [keys[0]] * n_needles
    values = [str(v) for v in rng.sample(range(1000000, 9999999), n_needles)]
    needles = [NEEDLE.format(key=k, value=v) for k, v in zip(keys, values)]
    # leave room for the instruction, the needles and the chat template around the document
    budget = length - 100 - sum(len(tokenizer(" " + n, add_special_tokens=False).input_ids) for n in needles)
    sentences = haystack(documents, tokenizer, budget, rng)
    depths = [(i + rng.uniform(0.1, 0.9)) / n_needles for i in range(n_needles)]
    for needle, depth in sorted(zip(needles, depths), key=lambda pair: -pair[1]):   # back to front keeps positions valid
        sentences.insert(round(depth * len(sentences)), needle)

    def question(asked_keys, asked_values):
        if task == "multivalue":
            text = f"What are all the special magic numbers for {asked_keys[0]} mentioned in the provided text?"
            prefix = f" The special magic numbers for {asked_keys[0]} mentioned in the provided text are"
        elif len(asked_keys) > 1:
            names = ", ".join(asked_keys)
            text = f"What are all the special magic numbers for {names} mentioned in the provided text?"
            prefix = f" The special magic numbers for {names} mentioned in the provided text are"
        else:
            text = f"What is the special magic number for {asked_keys[0]} mentioned in the provided text?"
            prefix = f" The special magic number for {asked_keys[0]} mentioned in the provided text is"
        return {"question": "\n\n" + text, "answer_prefix": prefix, "answers": asked_values}

    if task in ("multivalue", "multiquery"):
        questions = [question(keys, values)]
    elif task == "many":
        order = list(range(n_needles))
        rng.shuffle(order)                 # questions do not follow document order
        questions = [question([keys[i]], [values[i]]) for i in order]
    else:
        target = rng.randrange(n_needles)
        questions = [question([keys[target]], [values[target]])]
    return {"id": f"needles_{length}_{task}_{index}", "task": task, "length": length,
            "context": INSTRUCTION + " ".join(sentences), "questions": questions[:n_questions],
            "needle_depths": [round(d, 3) for d in depths],
            "max_new_tokens": 64 if len(questions[0]["answers"]) > 1 else 24}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length", type=int, required=True, help="document length in tokens, e.g. 32768 or 65536")
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    documents = prose_documents(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"needles_{args.length}.jsonl"
    lengths = []
    with out_file.open("w", encoding="utf-8") as out:
        for task in args.tasks.split(","):
            for index in range(args.per_task):
                example = make_example(task, index, args.length, documents, tokenizer, args.seed)
                lengths.append(len(tokenizer(example["context"], add_special_tokens=False).input_ids))
                out.write(json.dumps(example) + "\n")
    digest = hashlib.sha256(out_file.read_bytes()).hexdigest()[:16]
    print(f"wrote {out_file}: {len(lengths)} documents from {len(documents)} source texts, "
          f"{min(lengths)} to {max(lengths)} tokens, sha256 {digest}")


if __name__ == "__main__":
    main()
