"""Aggregate independent validated-study job files into auditable CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="experiments/results/validated-v2"
    )
    parser.add_argument("--output-dir", help="Separate output directory; defaults to input directory")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    jobs = sorted((input_dir / "jobs").glob("*/*/*.json"))
    if not jobs:
        raise SystemExit(f"No completed jobs under {input_dir}")

    raw: list[dict[str, object]] = []
    for path in jobs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "REAL_EXPERIMENT":
            raise ValueError(f"Incomplete job: {path}")
        key = payload["key"]
        for method in payload["methods"]:
            raw.append(
                {
                    **key,
                    "evaluation_seed": payload["evaluation_seed"],
                    **{k: v for k, v in method.items() if k != "seeds"},
                    "seeds": json.dumps(method["seeds"], separators=(",", ":")),
                    "status": "REAL_EXPERIMENT",
                }
            )

    raw.sort(
        key=lambda r: (
            str(r["scenario"]),
            str(r["dataset"]),
            int(r["k"]),
            str(r["method"]),
            int(r["selection_seed"]),
        )
    )
    raw_fields = list(raw[0])
    write_csv(output_dir / "validated_raw.csv", raw_fields, raw)

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        groups[(row["dataset"], row["scenario"], row["k"], row["method"])].append(row)

    summary: list[dict[str, object]] = []
    for (dataset, scenario, k, method), rows in sorted(groups.items()):
        values = [float(row["mean_adopters"]) for row in rows]
        redemptions = [float(row["mean_redemptions"]) for row in rows]
        selection_times = [
            float(row["selection_seconds"])
            for row in rows
            if row["selection_seconds"] not in (None, "")
        ]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ci95_selection = (
            float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
            if len(values) > 1 else ""
        )
        summary.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "k": k,
                "method": method,
                "selection_repeats": len(values),
                "mean_adopters": statistics.mean(values),
                "std_across_selection_runs": std,
                "ci95_across_selection_runs": ci95_selection,
                "ci_method": "student_t_two_sided_95" if len(values) > 1 else "unavailable_single_repeat",
                "min_adopters": min(values),
                "max_adopters": max(values),
                "mean_redemptions": statistics.mean(redemptions),
                "mean_selection_seconds": (
                    statistics.mean(selection_times) if selection_times else ""
                ),
                "status": "REAL_EXPERIMENT",
            }
        )
    summary_fields = list(summary[0])
    write_csv(output_dir / "validated_summary.csv", summary_fields, summary)
    print(f"Aggregated {len(jobs)} jobs, {len(raw)} raw rows, {len(summary)} summaries")


if __name__ == "__main__":
    main()
