"""Command-line entry points for reproducible, plot-free experiments."""
import argparse
from pathlib import Path

from . import native
from .runner import METHODS, run, run_job, write


def demo_config():
    return {"graphs": [{"name": "synthetic-smoke", "format": "synthetic", "n": 24}],
            "scenarios": ["balanced-reference"], "budgets": [5], "theta_factors": [4],
            "methods": list(METHODS), "repeats": 1, "evaluation_trials": 500,
            "forward_batches": 32, "seed": 42}


def main():
    parser = argparse.ArgumentParser(description="Coupon influence maximization experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Build optional C++17 acceleration")
    builder.add_argument("--build-dir", type=Path, default=Path("build"))
    builder.add_argument("--compiler", help="GCC/Clang executable; otherwise CXX or PATH")
    for command in ("run", "demo"):
        p = commands.add_parser(command, help="Run a JSON experiment" if command == "run" else "Run generated smoke graph")
        if command == "run":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        else:
            p.add_argument("--output", type=Path, default=Path("results/demo"))
        p.add_argument("--backend", choices=("python", "cpp"), default="python")
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--build-dir", type=Path, default=Path("build"))
        p.add_argument("--timeout", type=float, help="Per-job seconds; failed/timed-out jobs can be resumed")
    worker = commands.add_parser("_job", help=argparse.SUPPRESS)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            native.build(args.build_dir, args.compiler)
            print(f"C++ kernels ready in {args.build_dir.resolve()}")
        elif args.command == "_job":
            run_job(args.output, args.job_id, args.build_dir)
        else:
            if args.command == "demo":
                args.config = args.output / "demo-config.json"
                if not args.config.exists():
                    write(args.config, demo_config())
            raise SystemExit(run(args.config, args.output, args.backend, args.workers, args.build_dir, args.timeout))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
