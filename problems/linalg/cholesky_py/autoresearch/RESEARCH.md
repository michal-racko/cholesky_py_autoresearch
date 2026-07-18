# Autoresearch iteration — batched Cholesky on B200

You are one iteration of an automated performance-research loop for a GPU MODE
leaderboard submission. Your job in this session: interpret the latest results,
pick ONE hypothesis, implement it in `submission.py`, and declare it in
`autoresearch/CANDIDATE.json`. A deterministic harness (not you) will then run
correctness tests and benchmarks on a Modal B200, decide keep-or-revert, and
append the outcome to the ledger.

## The task being optimized

`custom_kernel(data)` in `submission.py` receives a `batch x n x n` FP32 CUDA
tensor of SPD matrices and must return the lower-triangular Cholesky factor L
(FP32, positive diagonal). See `task.yml` for the full spec and `reference.py`
for the checker.

- Ranking = geometric mean of runtime over the 15 benchmark entries in
  `task.yml` (`benchmarks:` section). Every entry counts equally: a 2x win on
  one entry is worth the same wherever it happens.
- Correctness gate: reconstruction residual `||L@L.T - A||_1 <= 20 * n * eps *
  ||A||_1` per matrix, checked in strict FP32 (TF32 disabled in the checker),
  plus lower-triangularity and positive diagonal. Slack grows with n — that is
  why reduced-precision tricks work at large n but fail at small n.
- All benchmark inputs are dense SPD (`cond: 2`); diagonal/tridiagonal cases
  appear only in the correctness tests, so structure detection does not pay.
- Hardware: B200 (Blackwell), CUDA 13, torch 2.9.1 cu130, triton available.

## Step 1 — read the state

Read, in this order:

1. `autoresearch/ledger.jsonl` — experiment history. Fields per line: `id`,
   `parent`, `hypothesis`, `verdict` (`baseline` / `improved` /
   `inconclusive` / `worse` / `fail-correctness` / `fail-benchmark` /
   `fail-compile` / `error`), `geomean_ns`, `delta_vs_best`, and
   `per_entry_delta` (relative speedup per benchmark spec, >0 = faster).
2. `autoresearch/BACKLOG.md` — the idea backlog. This file is YOURS to
   maintain.
3. `submission.py` — the current best-known implementation. The harness has
   already restored it if the previous experiment lost, so it is always your
   correct starting point.

## Step 2 — interpret the previous experiment

Update `autoresearch/BACKLOG.md` based on the newest ledger entry:

- Move its backlog item to the Tried section with a one-line finding,
  including WHY it won or lost (use `per_entry_delta` to see which entries
  moved). Failed ideas must record enough detail that future iterations do not
  retry them blindly.
- Add any new ideas the result suggests. Re-rank Open items if warranted.
- An `inconclusive` verdict (within the ~2% noise floor) may deserve one
  retry as a refined variant, but never more than one.

## Step 3 — pick ONE hypothesis

- If the Open backlog is empty or thin, first do your own analysis: study
  `submission.py` (its docstrings record measured dispatch trade-offs), the
  benchmark grid in `task.yml`, and the per-entry timings in the ledger and
  `autoresearch/best/best.json`. Which entries get the least-optimized
  treatment relative to their share of the geomean? Write the ideas you
  generate into the Open backlog, ranked, so future iterations inherit them.
- Take the top viable item from the Open backlog unless the latest result
  clearly motivates a refinement of the previous experiment.
- One conceptual change per iteration. Never bundle unrelated changes —
  attribution is the whole point of the loop.
- Parameter-tuning hypotheses (e.g. "block size 2048 instead of 1024") are
  legitimate experiments.

## Step 4 — implement it in `submission.py`

- Keep the `custom_kernel(data: input_t) -> output_t` entry point.
- You have NO GPU here. Do not attempt to run or benchmark the code. You may
  syntax-check with `python -m py_compile submission.py`.
- Correctness costs an entire Modal round-trip: be conservative about the
  residual gate at small n, guard new paths by dispatch conditions, and keep
  the proven fallbacks for sizes you are not targeting.
- Write clean code with brief comments explaining dispatch decisions —
  future iterations read this file cold.

## Step 5 — declare the candidate

Write `autoresearch/CANDIDATE.json`:

```json
{
  "hypothesis": "one sentence: what change and why it should be faster",
  "what_changed": "2-4 sentences of implementation detail",
  "backlog_item": "title of the backlog item this came from",
  "expected_entries": ["which benchmark specs should improve"]
}
```

If and only if the backlog is exhausted and you cannot form a credible new
hypothesis, instead write `{"stop": true, "reason": "..."}` and leave
`submission.py` untouched.

## Hard rules

- Strictly follow the fair-play rules enforced by KernelGuard
  (https://github.com/gpu-mode/kernelguard). Read that repository (README and
  the detection rules in `kernelguard.py`) before implementing, and never use
  the techniques it detects — e.g. timer/evaluator monkeypatching, spoofed
  benchmark output, caching and replaying output tensors across calls,
  unsynchronized multi-stream dispatch that escapes the timed region, or any
  other trick that games the measurement instead of doing the computation.
  Every optimization must genuinely compute the Cholesky factor of the given
  input within the timed call.
- Do NOT run `run_modal.py`, `modal`, or any benchmark yourself.
- Do NOT edit: `eval.py`, `reference.py`, `task.py`, `task.yml`,
  `run_modal.py`, `autoresearch/run_experiment.py`, `autoresearch.sh`,
  `autoresearch/ledger.jsonl`, anything under `autoresearch/best/` or
  `autoresearch/experiments/`.
- Only `submission.py`, `autoresearch/BACKLOG.md`, and
  `autoresearch/CANDIDATE.json` may change in this session.