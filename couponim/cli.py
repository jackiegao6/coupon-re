"""Command-line entry points for reproducible, plot-free experiments."""
import argparse
from pathlib import Path

from . import native
from .runner import run, run_job, write
from .selection import DEFAULT_METHODS


def demo_config():
    return {"graphs": [{"name": "synthetic-smoke", "format": "synthetic", "n": 24}],
            "scenarios": ["balanced-reference"], "budgets": [5], "theta_factors": [4],
            "methods": list(DEFAULT_METHODS), "repeats": 1, "evaluation_trials": 500,
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
    for name in ("inspect", "diagnose", "sampler-ablation"):
        p = commands.add_parser(name, help="Graph statistics, exact small-graph diagnosis, or fixed-seed sampler comparison")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--max-nodes", type=int, default=16, help="Exact DP size limit")
    p = commands.add_parser("reevaluate", help="Evaluate stored seeds again without repeating selection")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=5000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--backend", choices=("python", "cpp"), default="python")
    p.add_argument("--build-dir", type=Path, default=Path("build"))
    commands.add_parser("methods", help="List available selection methods")
    p = commands.add_parser("convert", help="Convert your edge list or MATLAB adjacency to CSR NPZ")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-format", choices=("edge-list", "mat"), default="edge-list")
    direction = p.add_mutually_exclusive_group(required=True)
    direction.add_argument("--directed", action="store_true")
    direction.add_argument("--undirected", action="store_true")
    p.add_argument("--nodes", type=int)
    p.add_argument("--matrix-key", default="network")
    worker = commands.add_parser("_job", help=argparse.SUPPRESS)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            native.build(args.build_dir, args.compiler)
            print(f"C++ kernels ready in {args.build_dir.resolve()}")
        elif args.command == "convert":
            from .convert import convert_graph
            print(convert_graph(args.input, args.output, args.input_format,
                                args.directed, args.nodes, args.matrix_key))
        elif args.command == "methods":
            from .selection import METHODS
            print("\\n".join(METHODS))
        elif args.command in {"inspect", "diagnose", "sampler-ablation"}:
            from .diagnostics import run_diagnostic
            run_diagnostic(args.config, args.output, args.command, args.max_nodes)
            print(f"Saved diagnostics to {args.output}")
        elif args.command == "reevaluate":
            from .reevaluate import reevaluate
            count = reevaluate(args.source, args.output, args.trials, args.backend, args.workers, args.build_dir)
            print(f"Re-evaluated {count} saved selections in {args.output}")
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
