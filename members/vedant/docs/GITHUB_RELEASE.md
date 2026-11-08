# GitHub source release

This source release contains the local implementation and validation evidence.
RTX 5090 and full Llama 3.1 8B results are pending and belong in a later results
commit. The original project remains at the same path.

## Before publishing

```bash
python -m pytest -q
python -m build --no-isolation
 git status --short
 git diff --check
```

This package lives in the team repository under `members/vedant/`. Work on a branch
and open a pull request on `samruddhichitnis02/298a-kv-cache-compression`; never push
to `main` directly:

```bash
git switch -c vedant/<topic>
git add members/vedant
git diff --cached --stat
git commit -m "<what changed>"
git push -u origin vedant/<topic>
```

Do not add the old transfer ZIP, `.venv`, credentials, model weights or dataset
text. The 800-record dataset remains a team input with a frozen identity.
Regenerate calibration/context JSONL using `turboquant-split` after obtaining it.
The vendored official LongBench code retains its upstream MIT license; this
notice does not assign a new license to the team's own code.

## Later GPU results commit

Include the source commit, environment, model revision, data identity, commands,
logs, paired predictions/scoring, observed storage and peak memory, reports and
confidence intervals. Remove secrets and model weights. Review file sizes and
sharing permissions. Intentionally allow only the result artifacts you want to
publish; do not disable the whole results ignore rule.
