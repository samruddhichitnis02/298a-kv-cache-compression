"""Checks for the adaptive rule and the needle documents. CPU only, no model or data needed.

    python -m pytest -q harness/test_tools.py
"""
import random
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive  # noqa: E402
import needles  # noqa: E402


def synthetic_runs(n=240, seed=0):
    """Two kinds of prompt: prose that survives heavy compression and tables that do not."""
    rng = random.Random(seed)
    records, runs = {}, {"0.5": {}, "0.9": {}}
    for i in range(n):
        fragile = i % 2 == 0
        text = ("12 34 | 56 78 | 90\n" if fragile else "the quick brown fox jumps over the lazy dog. ") * 200
        rid = f"r{i}"
        records[rid] = {"record_id": rid, "context": text, "prompt": text + "\n\nQuestion: what?",
                        "prompt_tokens_after": 4000 + rng.randrange(50)}
        for ratio in ("0.5", "0.9"):
            hurt = fragile and ratio == "0.9"
            runs[ratio][rid] = {"record_id": rid, "task": "table" if fragile else "prose", "compression_ratio": float(ratio),
                                "score": 0.1 if hurt else 0.8, "kv_cache_gb": 1.0 - float(ratio)}
    return runs, records


def test_allocation_stays_under_budget_and_beats_fixed():
    runs, _ = synthetic_runs()
    ids = sorted(runs["0.9"])
    tasks = {i: runs["0.9"][i]["task"] for i in ids}
    budget = 0.5 * (sum(r["kv_cache_gb"] for r in runs["0.5"].values()) + sum(r["kv_cache_gb"] for r in runs["0.9"].values()))
    choice = adaptive.allocate(adaptive.group_tables(runs, tasks), budget)
    assert choice == {"prose": "0.9", "table": "0.5"}
    score, gb = adaptive.outcome(runs, {i: choice[tasks[i]] for i in ids})
    assert gb <= budget + 1e-6 and score == pytest.approx(80.0)


def test_policy_finds_the_fragile_prompts_without_labels():
    runs, records = synthetic_runs()
    ids = sorted(records)
    # budget: what a half-and-half mix would use
    runs["0.7"] = {i: {**runs["0.9"][i], "compression_ratio": 0.7, "kv_cache_gb": 0.3, "score": 0.45} for i in ids}
    policy = adaptive.fit_policy(runs, records, "0.7", ids, clusters=4)
    picked = {i: adaptive.choose_ratio(policy, records[i]) for i in ids}
    assert all(picked[i] == 0.5 for i in ids if runs["0.9"][i]["task"] == "table")
    assert all(picked[i] == 0.9 for i in ids if runs["0.9"][i]["task"] == "prose")


def test_evaluate_reports_gain_at_equal_memory():
    runs, records = synthetic_runs()
    ids = sorted(records)
    runs["0.7"] = {i: {**runs["0.9"][i], "compression_ratio": 0.7, "kv_cache_gb": 0.3, "score": 0.45} for i in ids}
    result = adaptive.evaluate(runs, records, "0.7", clusters=4)
    assert result["fixed"] == pytest.approx(45.0)
    assert result["adaptive_held_out"] == pytest.approx(80.0)
    assert result["adaptive_memory_vs_fixed"] <= 1.0 + 1e-9


class WordTokenizer:
    """Stand-in tokenizer: one token per word."""
    def __call__(self, text, add_special_tokens=False):
        return type("Encoding", (), {"input_ids": text.split()})()


def test_needle_scoring():
    assert needles.ruler_score("The numbers are 1234567 and 7654321.", ["1234567", "7654321"]) == 1.0
    assert needles.ruler_score("It is 1234567.", ["1234567", "7654321"]) == 0.5
    assert needles.ruler_score("I do not know.", ["1234567"]) == 0.0


def test_needle_documents_hold_their_answers_and_hit_the_length():
    documents = [" ".join(f"Sentence {d} number {i} is here." for i in range(400)) for d in range(6)]
    tokenizer = WordTokenizer()
    for task, (n_needles, n_questions) in needles.TASKS.items():
        example = needles.make_example(task, 0, 2000, documents, tokenizer, seed=0)
        again = needles.make_example(task, 0, 2000, documents, tokenizer, seed=0)
        assert example == again                                   # same seed, same document
        assert len(example["questions"]) == n_questions
        assert 1800 < len(example["context"].split()) <= 2000
        answers = [a for q in example["questions"] for a in q["answers"]]
        assert all(a in example["context"] for a in answers)
        assert example["context"].count("special magic number") >= n_needles
    many = needles.make_example("many", 1, 2000, documents, tokenizer, seed=0)
    assert len({q["answers"][0] for q in many["questions"]}) == 16


def test_runs_that_mix_presses_get_named_settings(tmp_path):
    import json
    files = []
    for press, ratio in (("restorekv", 0.95), ("restorekv_4bit", 0.95), ("restorekv", 0.9)):
        path = tmp_path / f"{press}_{ratio}.jsonl"
        path.write_text(json.dumps({"record_id": "a", "task": "t", "press": press, "compression_ratio": ratio,
                                    "score": 0.5, "kv_cache_gb": 0.1}) + "\n")
        files.append(path)
    mixed = adaptive.load_runs(files)
    assert sorted(mixed) == ["restorekv@0.9", "restorekv@0.95", "restorekv_4bit@0.95"]
    assert adaptive.find_setting(mixed, "restorekv_4bit@0.95") == "restorekv_4bit@0.95"
    one_press = adaptive.load_runs([files[0], files[2]])
    assert sorted(one_press) == ["0.9", "0.95"]
    assert adaptive.find_setting(one_press, "0.9") == "0.9"


def test_allocation_keeps_the_fixed_setting_when_it_is_the_only_one_that_fits():
    tables = {"small": {"g1": (10.0, 1.0), "g2": (10.0, 1.0)}, "large": {"g1": (20.0, 5.0), "g2": (20.0, 5.0)}}
    assert adaptive.allocate(tables, 2.0, "small") == {"g1": "small", "g2": "small"}
    # equal memory, different scores: the better one wins without exceeding the budget
    same = {"plain": {"g1": (10.0, 1.0)}, "quantized": {"g1": (30.0, 1.0)}}
    assert adaptive.allocate(same, 1.0, "plain") == {"g1": "quantized"}


def test_evaluate_copes_with_a_task_missing_from_one_half():
    runs, records = synthetic_runs(40)
    ids = sorted(records)
    runs["0.7"] = {i: {**runs["0.9"][i], "compression_ratio": 0.7, "kv_cache_gb": 0.3, "score": 0.45} for i in ids}
    for setting in runs:                      # one record of a task nobody else has
        runs[setting][ids[0]] = {**runs[setting][ids[0]], "task": "lonely"}
    result = adaptive.evaluate(runs, records, "0.7", clusters=4, repeats=3)
    assert result["per_task_held_out"] > 0
