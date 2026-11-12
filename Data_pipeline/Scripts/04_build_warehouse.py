"""Team 6 data warehouse and pipeline monitor (Workbook 2, data pipeline demo).

Loads every pipeline stage into one SQLite warehouse, runs validation checks on it and
writes a monitoring report. Re-running rebuilds the warehouse from scratch, so it is safe
to run after every new result.

    python Data_pipeline/Scripts/04_build_warehouse.py --raw path/to/longbench_raw/data

Inputs:  raw LongBench JSONL files (--raw), data/processed_longbench/processed_longbench.jsonl,
         the split ID lists in Data_pipeline/processed_longbench/, and every results/**/*.jsonl
         and members/*/results/**/*.jsonl run file.
Outputs: data/warehouse/team6.db and data/warehouse/monitoring_report.md (both outside Git).

Tables:  raw_records, clean_records, splits, runs, pipeline_log. Full document text is not
copied into the warehouse; each document is identified by the SHA-256 of its context.
"""

import argparse
import datetime as dt
import glob
import hashlib
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed_longbench" / "processed_longbench.jsonl"
SPLITS = REPO / "Data_pipeline" / "processed_longbench"
OUT_DIR = REPO / "data" / "warehouse"
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
         "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
EXPECTED = {"raw": 3750, "clean": 3731, "train": 2131, "validation": 800, "test": 800}

SCHEMA = """
CREATE TABLE raw_records (record_id TEXT PRIMARY KEY, task TEXT, language TEXT, length_words INTEGER,
                          context_chars INTEGER, context_sha TEXT);
CREATE TABLE clean_records (record_id TEXT PRIMARY KEY, task TEXT, category TEXT, metric TEXT,
                            context_words INTEGER, prompt_tokens_before INTEGER, prompt_tokens_after INTEGER,
                            truncated INTEGER, context_sha TEXT);
CREATE TABLE splits (record_id TEXT PRIMARY KEY, split TEXT);
CREATE TABLE runs (run TEXT, method TEXT, compression_ratio REAL, record_id TEXT, task TEXT, score REAL,
                   kv_cache_gb REAL, peak_gpu_gb REAL, seconds REAL, oom INTEGER, source_file TEXT);
CREATE TABLE pipeline_log (run_at TEXT, stage TEXT, check_name TEXT, expected TEXT, observed TEXT, status TEXT);
"""


def sha(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the raw LongBench task JSONL files")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUT_DIR / "team6.db"
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    log = []

    def check(stage, name, expected, observed):
        status = "PASS" if expected == observed else "FAIL"
        log.append((now, stage, name, str(expected), str(observed), status))

    # 1. Ingestion: raw archive
    raw_rows = []
    for task in TASKS:
        for line in open(Path(args.raw) / f"{task}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            raw_rows.append((r["_id"], task, r["language"], r["length"], len(r["context"]), sha(r["context"])))
    db.executemany("INSERT INTO raw_records VALUES (?,?,?,?,?,?)", raw_rows)
    check("ingestion", "raw record count", EXPECTED["raw"], len(raw_rows))
    check("ingestion", "unique raw IDs", len(raw_rows), len({r[0] for r in raw_rows}))

    # 2. Cleaning + transformation: processed export
    clean_rows = []
    for line in open(PROCESSED, encoding="utf-8"):
        r = json.loads(line)
        clean_rows.append((r["record_id"], r["task"], r["category"], r["metric"], r["context_words"],
                           r["prompt_tokens_before"], r["prompt_tokens_after"], int(bool(r["truncated"])), sha(r["context"])))
    db.executemany("INSERT INTO clean_records VALUES (?,?,?,?,?,?,?,?,?)", clean_rows)
    check("cleaning", "clean record count", EXPECTED["clean"], len(clean_rows))
    check("cleaning", "rows removed by cleaning", 19, len(raw_rows) - len(clean_rows))
    check("cleaning", "clean IDs all come from raw", 0,
          db.execute("SELECT COUNT(*) FROM clean_records WHERE record_id NOT IN (SELECT record_id FROM raw_records)").fetchone()[0])
    check("transformation", "prompts over the 32,000-token cap", 0,
          db.execute("SELECT COUNT(*) FROM clean_records WHERE prompt_tokens_after > 32000").fetchone()[0])
    check("transformation", "truncated prompts", 88, db.execute("SELECT SUM(truncated) FROM clean_records").fetchone()[0])

    # 3. Preparation: fixed splits
    dev = json.load(open(SPLITS / "dev_ids.json"))
    dev_ids = dev if isinstance(dev, list) else next(v for v in dev.values() if isinstance(v, list))
    split_rows = [(i, "validation") for i in dev_ids]
    split_rows += [(i, "test") for i in json.load(open(SPLITS / "test_ids.json"))["test_ids"]]
    split_rows += [(i, "train") for i in json.load(open(SPLITS / "train_ids.json"))["train_ids"]]
    db.executemany("INSERT INTO splits VALUES (?,?)", split_rows)  # PRIMARY KEY rejects any ID in two splits
    for name in ("train", "validation", "test"):
        check("preparation", f"{name} rows", EXPECTED[name],
              db.execute("SELECT COUNT(*) FROM splits WHERE split=?", (name,)).fetchone()[0])
    check("preparation", "split IDs missing from clean data", 0,
          db.execute("SELECT COUNT(*) FROM splits WHERE record_id NOT IN (SELECT record_id FROM clean_records)").fetchone()[0])
    check("preparation", "records per task in validation", "50 x 16", "50 x 16" if all(n == 50 for (n,) in db.execute(
        "SELECT COUNT(*) FROM splits s JOIN clean_records c USING(record_id) WHERE split='validation' GROUP BY task")) else "unbalanced")

    # 4. Results: every stored run
    files = sorted(set(glob.glob(str(REPO / "results" / "**" / "*.jsonl"), recursive=True)
                       + glob.glob(str(REPO / "members" / "*" / "results" / "**" / "*.jsonl"), recursive=True)))
    for f in files:
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if not rows or "record_id" not in rows[0] or "score" not in rows[0]:
            continue
        run = Path(f).stem
        db.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            (run, r.get("press"), r.get("compression_ratio"), r["record_id"], r["task"], r["score"],
             r.get("kv_cache_gb"), r.get("peak_gpu_gb"), r.get("seconds"), int(bool(r.get("oom"))),
             str(Path(f).relative_to(REPO))) for r in rows])
        outside = db.execute("SELECT COUNT(*) FROM runs WHERE run=? AND record_id NOT IN "
                             "(SELECT record_id FROM splits WHERE split='validation')", (run,)).fetchone()[0]
        check("results", f"{run}: records outside validation", 0, outside)
    db.executemany("INSERT INTO pipeline_log VALUES (?,?,?,?,?,?)", log)
    db.commit()

    # 5. Monitoring report
    lines = [f"# Team 6 pipeline monitoring report", "", f"Built {now} from {len(files)} result files.", "",
             "## Stage counts", "", "| Stage | Rows |", "|---|---|"]
    for name, q in (("Raw", "SELECT COUNT(*) FROM raw_records"), ("Cleaned / transformed", "SELECT COUNT(*) FROM clean_records"),
                    ("Train / calibration", "SELECT COUNT(*) FROM splits WHERE split='train'"),
                    ("Validation", "SELECT COUNT(*) FROM splits WHERE split='validation'"),
                    ("Test", "SELECT COUNT(*) FROM splits WHERE split='test'")):
        lines.append(f"| {name} | {db.execute(q).fetchone()[0]:,} |")
    lines += ["", "## Runs", "", "| Run | Records | OOM | Macro score | Mean KV cache (GB) |", "|---|---|---|---|---|"]
    for run, n, oom, kv in db.execute("SELECT run, COUNT(*), SUM(oom), AVG(kv_cache_gb) FROM runs GROUP BY run ORDER BY run"):
        tasks = db.execute("SELECT AVG(score) FROM runs WHERE run=? GROUP BY task", (run,)).fetchall()
        macro = 100 * sum(t for (t,) in tasks) / len(tasks)
        lines.append(f"| {run} | {n} | {oom} | {macro:.2f} | {kv:.3f} |" if kv is not None else f"| {run} | {n} | {oom} | {macro:.2f} | — |")
    fails = [r for r in log if r[5] == "FAIL"]
    lines += ["", f"## Checks: {len(log) - len(fails)} passed, {len(fails)} failed", "",
              "| Stage | Check | Expected | Observed | Status |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {c} | {e} | {o} | {st} |" for (_, s, c, e, o, st) in log]
    (OUT_DIR / "monitoring_report.md").write_text("\n".join(lines) + "\n")
    print(f"warehouse: {db_path}\nreport:    {OUT_DIR / 'monitoring_report.md'}\nchecks:    {len(log) - len(fails)} passed, {len(fails)} failed")
    for f_ in fails:
        print("  FAIL", f_[1], f_[2], "expected", f_[3], "observed", f_[4])


if __name__ == "__main__":
    main()
