#!/usr/bin/env python3
"""Deterministic half of the autoresearch loop.

Measures whatever is currently in ../submission.py on Modal, decides
keep-or-revert against the best-known snapshot, and appends the result to the
ledger. All state lives next to this file:

    ledger.jsonl        append-only experiment history (this script is the
                        only writer)
    best/               best-known submission.py + best.json (its timings)
    experiments/eNNN/   snapshot of every attempt: submission.py,
                        candidate.json, raw popcorn output, result.json

Modes:
    --baseline            measure current submission.py, install it as best
    --candidate FILE      evaluate current submission.py against best; FILE is
                          the CANDIDATE.json the agent wrote (hypothesis etc.)
    --summary             print the ledger as a table; no Modal runs

Environment:
    AR_GPU                GPU type passed to run_modal.py (default B200)
    AR_IMPROVE_THRESHOLD  relative geomean improvement required to accept a
                          candidate (default 0.02 = 2%, ~ run-to-run noise)
"""

import argparse
import json
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

AR = Path(__file__).parent.resolve()
ROOT = AR.parent
SUBMISSION = ROOT / "submission.py"
LEDGER = AR / "ledger.jsonl"
BEST_DIR = AR / "best"
EXP_DIR = AR / "experiments"

GPU = os.environ.get("AR_GPU", "B200")
IMPROVE_THRESHOLD = float(os.environ.get("AR_IMPROVE_THRESHOLD", "0.02"))

_POPCORN_KEY = re.compile(r"^((?:test|benchmark)[.\w-]*|check|test-count|benchmark-count): (.*)$")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def append_ledger(entry: dict) -> None:
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def run_modal(mode: str) -> tuple[int, str]:
    """Run run_modal.py <mode> and return (returncode, combined output)."""
    proc = subprocess.run(
        [sys.executable, "run_modal.py", mode, GPU],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    if proc.stderr.strip():
        out += "\n--- run_modal stderr ---\n" + proc.stderr
    return proc.returncode, out


def parse_popcorn(text: str) -> dict[str, str]:
    """Extract `key: value` popcorn lines from run_modal.py output."""
    # eval.py's own stderr is echoed after this marker; don't parse it.
    text = text.split("\n--- stderr ---")[0]
    kv = {}
    for line in text.splitlines():
        m = _POPCORN_KEY.match(line)
        if m:
            kv[m.group(1)] = m.group(2)
    return kv


def parse_benchmark_entries(kv: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Return (entries, errors). Each entry: {spec, mean_ns, err_ns, best_ns}."""
    count = int(kv.get("benchmark-count", 0))
    entries, errors = [], []
    for i in range(count):
        spec = kv.get(f"benchmark.{i}.spec", f"<entry {i}>")
        if kv.get(f"benchmark.{i}.status") == "fail":
            errors.append(f"{spec}: {kv.get(f'benchmark.{i}.error', 'unknown error')}")
            continue
        mean = kv.get(f"benchmark.{i}.mean")
        if mean is None:
            errors.append(f"{spec}: no timing reported")
            continue
        entries.append(
            {
                "spec": spec,
                "mean_ns": float(mean),
                "err_ns": float(kv.get(f"benchmark.{i}.err", "nan")),
                "best_ns": float(kv.get(f"benchmark.{i}.best", "nan")),
            }
        )
    return entries, errors


def test_errors(kv: dict[str, str]) -> list[str]:
    errors = []
    for i in range(int(kv.get("test-count", 0))):
        if kv.get(f"test.{i}.status") == "fail":
            spec = kv.get(f"test.{i}.spec", f"<test {i}>")
            errors.append(f"{spec}: {kv.get(f'test.{i}.error', 'unknown error')}")
    return errors


def geomean_ns(entries: list[dict]) -> float:
    return math.exp(sum(math.log(e["mean_ns"]) for e in entries) / len(entries))


def next_id() -> str:
    return f"e{len(read_ledger()):03d}"


def snapshot_dir(exp_id: str) -> Path:
    d = EXP_DIR / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def restore_best() -> None:
    best_sub = BEST_DIR / "submission.py"
    if best_sub.exists():
        shutil.copy2(best_sub, SUBMISSION)


def install_best(exp_id: str, geo_ns: float, entries: list[dict]) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUBMISSION, BEST_DIR / "submission.py")
    (BEST_DIR / "best.json").write_text(
        json.dumps({"id": exp_id, "geomean_ns": geo_ns, "entries": entries}, indent=2)
    )


def load_best() -> dict | None:
    path = BEST_DIR / "best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def finish(entry: dict, snap: Path) -> None:
    """Record the experiment and print a one-line outcome."""
    (snap / "result.json").write_text(json.dumps(entry, indent=2))
    append_ledger(entry)
    geo = entry.get("geomean_ns")
    geo_str = f" geomean={geo / 1e6:.3f}ms" if geo else ""
    delta = entry.get("delta_vs_best")
    delta_str = f" delta={delta * 100:+.2f}%" if delta is not None else ""
    print(f"[{entry['id']}] {entry['verdict']}{geo_str}{delta_str}")
    for err in entry.get("errors", [])[:5]:
        print(f"    error: {err}")


def evaluate(candidate_file: str | None, baseline: bool) -> int:
    exp_id = next_id()
    snap = snapshot_dir(exp_id)
    shutil.copy2(SUBMISSION, snap / "submission.py")

    candidate = {}
    if candidate_file:
        try:
            candidate = json.loads(Path(candidate_file).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            candidate = {"hypothesis": f"<unreadable candidate file: {exc}>"}
        shutil.copy2(candidate_file, snap / "candidate.json")

    best = load_best()
    entry = {
        "id": exp_id,
        "parent": None if baseline else (best or {}).get("id"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": "baseline" if baseline else candidate.get("hypothesis", "<missing>"),
        "what_changed": candidate.get("what_changed", ""),
        "gpu": GPU,
    }

    # Cheap local gate before spending Modal money.
    try:
        py_compile.compile(str(SUBMISSION), doraise=True)
    except py_compile.PyCompileError as exc:
        entry.update(verdict="fail-compile", errors=[str(exc)])
        restore_best()
        finish(entry, snap)
        return 0

    print(f"[{exp_id}] correctness tests on {GPU} ...")
    rc, out = run_modal("test")
    (snap / "test.txt").write_text(out)
    kv = parse_popcorn(out)
    if "check" not in kv:
        entry.update(verdict="error", errors=[f"test run produced no results (rc={rc})", out[-2000:]])
        restore_best()
        finish(entry, snap)
        return 0
    if kv["check"] != "pass":
        entry.update(verdict="fail-correctness", errors=test_errors(kv))
        restore_best()
        finish(entry, snap)
        return 0

    print(f"[{exp_id}] benchmarks on {GPU} ...")
    rc, out = run_modal("benchmark")
    (snap / "benchmark.txt").write_text(out)
    kv = parse_popcorn(out)
    entries, errors = parse_benchmark_entries(kv)
    if "check" not in kv:
        entry.update(verdict="error", errors=[f"benchmark run produced no results (rc={rc})", out[-2000:]])
        restore_best()
        finish(entry, snap)
        return 0
    if kv["check"] != "pass" or errors or not entries:
        entry.update(verdict="fail-benchmark", errors=errors or ["no benchmark entries parsed"])
        restore_best()
        finish(entry, snap)
        return 0

    geo = geomean_ns(entries)
    entry.update(geomean_ns=geo, entries=entries, errors=[])

    if baseline or best is None:
        entry["verdict"] = "baseline"
        install_best(exp_id, geo, entries)
        finish(entry, snap)
        return 0

    delta = (best["geomean_ns"] - geo) / best["geomean_ns"]  # >0 means faster
    entry["delta_vs_best"] = delta
    best_by_spec = {e["spec"]: e["mean_ns"] for e in best.get("entries", [])}
    entry["per_entry_delta"] = {
        e["spec"]: (best_by_spec[e["spec"]] - e["mean_ns"]) / best_by_spec[e["spec"]]
        for e in entries
        if e["spec"] in best_by_spec
    }

    if delta > IMPROVE_THRESHOLD:
        entry["verdict"] = "improved"
        install_best(exp_id, geo, entries)
    else:
        entry["verdict"] = "worse" if delta < -IMPROVE_THRESHOLD else "inconclusive"
        restore_best()
    finish(entry, snap)
    return 0


def summary() -> int:
    ledger = read_ledger()
    if not ledger:
        print("ledger is empty")
        return 0
    best = load_best() or {}
    print(f"{'id':<6}{'verdict':<18}{'geomean':>12}{'delta':>9}  hypothesis")
    for e in ledger:
        geo = f"{e['geomean_ns'] / 1e6:.3f}ms" if e.get("geomean_ns") else "-"
        delta = f"{e['delta_vs_best'] * 100:+.2f}%" if e.get("delta_vs_best") is not None else "-"
        mark = " *" if e["id"] == best.get("id") else ""
        print(f"{e['id']:<6}{e['verdict']:<18}{geo:>12}{delta:>9}  {e['hypothesis'][:70]}{mark}")
    if best:
        print(f"\nbest: {best['id']} geomean={best['geomean_ns'] / 1e6:.3f}ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--candidate", metavar="FILE")
    group.add_argument("--summary", action="store_true")
    ns = parser.parse_args()

    if ns.summary:
        return summary()
    return evaluate(ns.candidate, ns.baseline)


if __name__ == "__main__":
    sys.exit(main())