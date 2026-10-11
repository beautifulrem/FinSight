# Operation wordings, independent set v1

`operation_wordings_independent_v1.jsonl` is 234 comparison and non-comparison questions (117 Chinese, 117 English).
Each one is labelled with the arithmetic operation its writer meant. It tests the comparison grammar
(`query_intelligence/agent/frame.py`) and the operation classifier (`models/operation_classifier.joblib`) on wordings
that the author of their templates did not write (round-12 review L17).

## How it was written

- **Writer.** A separate agent (a fresh Claude sub-agent with no conversation context) wrote the set on 2026-10-11,
  while the repository was at `c39008b`. It received only the task text described below. It was told not to read,
  list or search any file on the machine and not to use web search. It used one tool call, the one that wrote the
  file. It never saw `frame.py`, the classifier's training templates (`training/operation_classifier/train.py`), the
  tests, the dev / ext / held-out task sets or the review probes.
- **Input.**
  - The eleven labels, each with a one-line definition: `difference`, `ratio`, `relative`, `share`, `share_total`,
    `sum`, `mean`, `spread`, `which`, `total_vs_one` and `none`.
  - The thirteen company names (Chinese and English).
  - Twelve metric names (Chinese and English).
  - A request for at least 230 questions: at least 19 per operation (at least 9 in each language) and at least 30
    `none`.
  - Instructions to vary register, length and word order, to include lead-in clauses and instructions, and to give
    one operation per question.
- **The writer's own notes on labels it was unsure of.**
  - A yes/no direction question ("…比…高吗") is labelled `which`.
  - "Which is better to invest in" is labelled `none`.
  - 几成 is labelled `relative` in "多出几成" and `share` in "是…的几成".
  - `relative` is used for amounts and `share` where B is the base.
  - `total_vs_one` includes "by how much" questions.
  - The rows are shuffled with a fixed seed.
- **Fields.** `id`, `lang`, `question`, `operation`, `companies` (canonical Chinese names, in order of first mention)
  and `metric` (a frame metric key or `null`).
- **Integrity.** The file was copied unchanged from the writer's output. Its sha256 is
  `faabe7d90d1cf300bb6d8570e0f8f6456f087d6c8debf87c4bf23d58c2b56a92`. The labels are the writer's; the engineers did
  not change any of them.

## How it is scored

```bash
python -m training.operation_classifier.evaluate_independent --label <label>
```

The script compares three readings of the operation with the writer's labels:

- the grammar alone (run-time path with the classifier off);
- the classifier alone (its top label);
- the two combined (run-time path: the grammar, then the tie-breaker and the fallback).

It also runs the frame step with the offline NLU and checks the operation together with the operands. It counts how
often the classifier fires and how often that changes the frame request, both on this set and on the dev and ext task
questions. Results are written to `evaluation/results/operation_classifier-independent-v1-<label>.json`.

- **`first-run`** was measured at `c39008b`, from an exported tree (`git archive c39008b`), before any round-16
  change. Nobody had read the questions before this run.
- **Every later run is after exposure.** The engineers read the misses of the first run while working on round 16, so
  later numbers do not measure generalisation to unseen wordings.
