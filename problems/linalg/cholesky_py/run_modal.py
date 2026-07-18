"""Run the Cholesky submission on a Modal GPU.

This drives eval.py exactly like the real grader does: it materializes the
test/benchmark specs from task.yml into the line format get_test_cases()
expects, then runs eval.py over the POPCORN_FD protocol on a cloud GPU.

Usage (from this directory, with the venv active):
    python run_modal.py test          # correctness tests
    python run_modal.py benchmark     # timed benchmarks
    python run_modal.py test H100     # override GPU (default: B200)

Requires a one-time `modal setup` to authenticate.
"""

import sys
from pathlib import Path

import modal

# NB: keep module-level imports to what the REMOTE container also has. Modal
# re-imports this file inside the container to locate run_eval, so anything
# imported here (e.g. yaml) must either be in the image or imported lazily in
# main(). yaml is only needed locally to parse task.yml -> imported in main().

HERE = Path(__file__).parent.resolve()
UTILS = (HERE / ".." / ".." / "pmpp_v2" / "utils.py").resolve()

# CUDA 13 + torch cu130 wheel: required for Blackwell (B200) support.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("numpy")
    .add_local_file(HERE / "eval.py", "/root/eval.py")
    .add_local_file(HERE / "reference.py", "/root/reference.py")
    .add_local_file(HERE / "task.py", "/root/task.py")
    .add_local_file(HERE / "submission.py", "/root/submission.py")
    .add_local_file(UTILS, "/root/utils.py")
)

app = modal.App("cholesky-eval", image=image)


def _specs_to_text(specs: list[dict]) -> str:
    """Render task.yml spec dicts into eval.py's `key: value; ...` line format."""
    return "\n".join("; ".join(f"{k}: {v}" for k, v in spec.items()) for spec in specs)


@app.function(gpu="B200", timeout=1800)
def run_eval(mode: str, specs_text: str, seed: int | None = None):
    import os
    import subprocess

    Path("/root/specs.txt").write_text(specs_text)

    # eval.py writes its structured results to the fd named by POPCORN_FD.
    out_path = "/root/popcorn_out.txt"
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.set_inheritable(fd, True)
    env = dict(os.environ)
    env["POPCORN_FD"] = str(fd)
    if seed is not None:
        env["POPCORN_SEED"] = str(seed)

    proc = subprocess.run(
        ["python", "eval.py", mode, "specs.txt"],
        cwd="/root",
        env=env,
        pass_fds=[fd],
        capture_output=True,
        text=True,
    )
    try:
        os.close(fd)  # child holds its own copy; safe if it already closed
    except OSError:
        pass

    return {
        "returncode": proc.returncode,
        "popcorn": Path(out_path).read_text(),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Run the Cholesky submission on a Modal GPU.")
    parser.add_argument("mode", nargs="?", default="test",
                        choices=["test", "benchmark", "leaderboard"])
    parser.add_argument("gpu", nargs="?", default="B200", help="GPU type (default: B200)")
    parser.add_argument("--seed", type=int, default=None)
    ns = parser.parse_args()

    key = {"test": "tests", "benchmark": "benchmarks", "leaderboard": "benchmarks"}[ns.mode]
    task = yaml.safe_load((HERE / "task.yml").read_text())
    specs_text = _specs_to_text(task[key])

    fn = run_eval.with_options(gpu=ns.gpu)  # allow CLI override of the default B200
    with app.run():
        result = fn.remote(ns.mode, specs_text, ns.seed)

    print(f"=== eval.py {ns.mode} on {ns.gpu} (exit {result['returncode']}) ===\n")
    print(result["popcorn"])
    if result["stderr"].strip():
        print("\n--- stderr ---\n" + result["stderr"])
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
