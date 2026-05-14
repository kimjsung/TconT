#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


RESULT_PATTERN = re.compile(
    r"^TCCG_RESULT equation=(?P<equation>\d+) precision=(?P<precision>\w+) "
    r"operations=(?P<operations>\d+) time_ms=(?P<time_ms>[0-9.]+) "
    r"gflops=(?P<gflops>[0-9.]+) validation=(?P<validation>\w+)$")

PRECISIONS = ("fp64", )

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_binary = repo_root / "build" / "benchmarks" / "tccg" / "bench_tccg"
    default_output_dir = Path(__file__).resolve().parent / "results"

    parser = argparse.ArgumentParser(
        description="Run all TCCG benchmarks for FP64 and TF32, and save per-precision CSV files.")
    parser.add_argument(
        "--binary",
        type=Path,
        default=default_binary,
        help=f"Path to the bench_tccg binary (default: {default_binary})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to store result CSV files (default: {default_output_dir})")
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First equation index to run (default: 1)")
    parser.add_argument(
        "--end",
        type=int,
        default=48,
        help="Last equation index to run, inclusive (default: 48)")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip validation and mark validation as SKIP")
    return parser.parse_args()


def run_benchmark(binary: Path, equation: int, precision: str, verify: bool) -> dict:
    cmd = [str(binary), "-b", str(equation), f"--{precision}"]
    if verify:
        cmd.append("--verify")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    output = completed.stdout + completed.stderr

    if completed.returncode != 0:
        return {
            "equation": equation,
            "precision": precision,
            "operation_count": "",
            "time_ms": "",
            "gflops": "",
            "validation": "ERROR",
            "error_message": f"exit_code={completed.returncode}",
            "raw_output": output.strip(),
        }

    for line in output.splitlines():
        match = RESULT_PATTERN.match(line.strip())
        if match:
            return {
                "equation": int(match.group("equation")),
                "precision": match.group("precision"),
                "operation_count": int(match.group("operations")),
                "time_ms": float(match.group("time_ms")),
                "gflops": float(match.group("gflops")),
                "validation": match.group("validation"),
                "error_message": "",
                "raw_output": output.strip(),
            }

    return {
        "equation": equation,
        "precision": precision,
        "operation_count": "",
        "time_ms": "",
        "gflops": "",
        "validation": "ERROR",
        "error_message": "summary_not_found",
        "raw_output": output.strip(),
    }


def log(message: str) -> None:
    print(message, flush=True)


def write_results(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "equation",
        "precision",
        "operation_count",
        "time_ms",
        "gflops",
        "validation",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    args = parse_args()

    if not args.binary.exists():
        raise FileNotFoundError(f"bench_tccg binary not found: {args.binary}")
    if args.start < 0 or args.end < args.start:
        raise ValueError("Expected 0 <= start <= end")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify = not args.no_verify

    log(f"Python executable: {sys.executable}")
    log(f"Benchmark binary : {args.binary}")
    log(f"Output directory : {args.output_dir}")
    log(f"Equation range   : {args.start}..{args.end}")
    log(f"Verification     : {'enabled' if verify else 'disabled'}")

    for precision in PRECISIONS:
        rows = []
        output_path = args.output_dir / f"tccg_{precision}_results.csv"
        log(f"Starting precision sweep: {precision}")

        for equation in range(args.start, args.end + 1):
            log(f"[{precision}] starting equation={equation}")
            row = run_benchmark(args.binary, equation, precision, verify)
            rows.append(row)
            log(
                f"[{precision}] equation={row['equation']} "
                f"operations={row['operation_count']} time_ms={row['time_ms']} "
                f"gflops={row['gflops']} validation={row['validation']}")
            if row["error_message"]:
                log(f"[{precision}] equation={row['equation']} error={row['error_message']}")
                if row["raw_output"]:
                    log(f"[{precision}] equation={row['equation']} raw_output:\n{row['raw_output']}")

        write_results(output_path, rows)
        log(f"Saved {precision} results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
