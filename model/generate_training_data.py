#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


BACKENDS = ("cogent", "ttgt")
DEFAULT_INPUT_SETS = ("rand_problems", "rand_problems_new", "rand_problems_new2")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "model" / "results"

    parser = argparse.ArgumentParser(
        description="Collect joined Cogent/TTGT benchmark data for model training."
    )
    parser.add_argument("--binary", type=Path, required=True, help="Path to bench_rand_problem binary")
    parser.add_argument(
        "--input-root",
        dest="input_roots",
        action="append",
        type=Path,
        help="Input root containing *.in files. Repeat to include multiple roots.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to write dataset files")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS),
        choices=BACKENDS,
        help="Backends to benchmark for each problem",
    )
    parser.add_argument("--precision", choices=["fp64", "tf32"], default="fp64", help="Precision to benchmark")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations per backend")
    parser.add_argument("--repeats", type=int, default=200, help="Timed iterations per backend")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of problems to process after sharding")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this run")
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.02,
        help="Relative time difference below which the label is considered a tie",
    )
    parser.add_argument("--timeout-sec", type=int, default=0, help="Optional per-backend subprocess timeout")
    parser.add_argument("--append", action="store_true", help="Append to an existing dataset and skip completed inputs")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    return parser.parse_args()


def log(message, verbose=True):
    if verbose:
        print(message, flush=True)


def detect_input_roots(explicit_roots):
    if explicit_roots:
        return [root.resolve() for root in explicit_roots]

    repo_root = Path(__file__).resolve().parents[1]
    gen_input_root = repo_root.parent / "gen_input"
    roots = []
    for name in DEFAULT_INPUT_SETS:
        candidate = gen_input_root / name
        if candidate.exists():
            roots.append(candidate.resolve())
    if not roots:
        raise FileNotFoundError(
            f"No default input roots found under {gen_input_root}. "
            "Pass --input-root explicitly if your layout is different."
        )
    return roots


def collect_problem_files(input_roots):
    problem_files = []
    for input_root in input_roots:
        if not input_root.exists():
            raise FileNotFoundError(f"Input root not found: {input_root}")
        problem_files.extend(sorted(input_root.rglob("*.in")))
    seen = set()
    unique_files = []
    for path in sorted(problem_files):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(resolved)
    return unique_files


def apply_sharding(problem_files, num_shards, shard_index):
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Expected 0 <= shard-index < num-shards")
    return [path for index, path in enumerate(problem_files) if index % num_shards == shard_index]


def run_problem(binary, input_path, backend, precision, warmup, repeats, timeout_sec):
    command = [
        str(binary),
        "--input",
        str(input_path),
        "--backend",
        backend,
        "--precision",
        precision,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "input_path": str(input_path),
            "requested_backend": backend,
            "precision": precision,
            "status": "ERROR",
            "error_message": f"timeout after {exc.timeout} seconds",
        }

    if result.returncode != 0:
        return {
            "input_path": str(input_path),
            "requested_backend": backend,
            "precision": precision,
            "status": "ERROR",
            "error_message": result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}",
        }

    stdout = result.stdout.strip()
    try:
        return parse_json_payload(stdout)
    except json.JSONDecodeError as exc:
        return {
            "input_path": str(input_path),
            "requested_backend": backend,
            "precision": precision,
            "status": "ERROR",
            "error_message": f"invalid json output: {exc}; stdout={stdout}",
        }


def parse_json_payload(text):
    stripped = text.strip()
    if not stripped:
        raise json.JSONDecodeError("empty output", text, 0)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise json.JSONDecodeError("no JSON object found in output", text, 0)


def json_compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def infer_label(backend_results, tie_tolerance):
    ok_results = {
        backend: row
        for backend, row in backend_results.items()
        if row.get("status") == "OK" and isinstance(row.get("avg_ms"), (int, float)) and row.get("avg_ms", 0) > 0
    }
    if len(ok_results) != 2:
        return {
            "label": "",
            "winner_backend": "",
            "winner_speedup": "",
            "time_ratio_ttgt_over_cogent": "",
            "is_tie": False,
        }

    cogent_ms = float(ok_results["cogent"]["avg_ms"])
    ttgt_ms = float(ok_results["ttgt"]["avg_ms"])
    relative_gap = abs(cogent_ms - ttgt_ms) / min(cogent_ms, ttgt_ms)
    time_ratio = ttgt_ms / cogent_ms

    if relative_gap <= tie_tolerance:
        return {
            "label": "tie",
            "winner_backend": "tie",
            "winner_speedup": 1.0,
            "time_ratio_ttgt_over_cogent": time_ratio,
            "is_tie": True,
        }

    if cogent_ms < ttgt_ms:
        return {
            "label": "cogent",
            "winner_backend": "cogent",
            "winner_speedup": ttgt_ms / cogent_ms,
            "time_ratio_ttgt_over_cogent": time_ratio,
            "is_tie": False,
        }

    return {
        "label": "ttgt",
        "winner_backend": "ttgt",
        "winner_speedup": cogent_ms / ttgt_ms,
        "time_ratio_ttgt_over_cogent": time_ratio,
        "is_tie": False,
    }


def join_problem_rows(input_path, precision, backend_rows, tie_tolerance):
    first_row = next(iter(backend_rows.values()))
    joined = {
        "input_path": str(input_path),
        "precision": precision,
        "problem_metadata": first_row.get("problem_metadata", {}),
        "equation_features": first_row.get("equation_features", {}),
        "backend_results": backend_rows,
    }
    joined.update(infer_label(backend_rows, tie_tolerance))
    return joined


def flatten_joined_row(row, backends):
    flat = {
        "input_path": row.get("input_path", ""),
        "precision": row.get("precision", ""),
        "label": row.get("label", ""),
        "winner_backend": row.get("winner_backend", ""),
        "winner_speedup": row.get("winner_speedup", ""),
        "time_ratio_ttgt_over_cogent": row.get("time_ratio_ttgt_over_cogent", ""),
        "is_tie": row.get("is_tie", False),
    }

    metadata = row.get("problem_metadata", {})
    for key in ("source_collection", "equation_group", "problem_file", "problem_id"):
        flat[key] = metadata.get(key, "")

    equation = row.get("equation_features", {})
    scalar_keys = (
        "equation_name",
        "equation_string",
        "modeC",
        "modeA",
        "modeB",
        "num_modes_C",
        "num_modes_A",
        "num_modes_B",
        "num_unique_modes",
        "num_internal_modes",
        "num_output_modes",
        "num_output_only_left_modes",
        "num_output_only_right_modes",
        "num_output_shared_modes",
        "size_A",
        "size_B",
        "size_C",
        "output_volume",
        "internal_volume",
        "operation_count",
        "min_extent",
        "max_extent",
        "mean_extent",
        "std_extent",
        "extent_ratio_max_min",
        "log2_size_A",
        "log2_size_B",
        "log2_size_C",
        "log2_internal_volume",
    )
    for key in scalar_keys:
        flat[key] = equation.get(key, "")
    for key in ("mode_extents", "sorted_extents", "internal_modes"):
        flat[key] = json_compact(equation.get(key, {} if key == "mode_extents" else []))

    for backend in backends:
        prefix = f"{backend}_"
        backend_row = row.get("backend_results", {}).get(backend, {})
        flat[prefix + "status"] = backend_row.get("status", "")
        flat[prefix + "resolved_backend"] = backend_row.get("resolved_backend", "")
        flat[prefix + "avg_ms"] = backend_row.get("avg_ms", "")
        flat[prefix + "gflops"] = backend_row.get("gflops", "")
        flat[prefix + "error_message"] = backend_row.get("error_message", "")

        stage_times = backend_row.get("stage_times", {})
        for key in ("available", "total_ms", "transpose_a_ms", "transpose_b_ms", "gemm_ms", "transpose_c_ms"):
            flat[prefix + "stage_" + key] = stage_times.get(key, "")

        plan = backend_row.get("plan_features", {})
        for key, value in sorted(plan.items()):
            flat[prefix + "plan_" + key] = value if isinstance(value, (str, int, float, bool)) else json_compact(value)

    return flat


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows, backends):
    flattened_rows = [flatten_joined_row(row, backends) for row in rows]
    fieldnames = []
    for row in flattened_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow(row)


def load_existing_rows(path):
    if not path.exists():
        return []
    deduped_rows = []
    row_index_by_key = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                key = (row.get("input_path", ""), row.get("precision", ""))
                if key in row_index_by_key:
                    deduped_rows[row_index_by_key[key]] = row
                else:
                    row_index_by_key[key] = len(deduped_rows)
                    deduped_rows.append(row)
    return deduped_rows


def completed_input_keys(rows, backends):
    completed = set()
    for row in rows:
        key = (row.get("input_path", ""), row.get("precision", ""))
        backend_results = row.get("backend_results", {})
        if all(
            backend in backend_results and backend_results[backend].get("status") == "OK"
            for backend in backends
        ):
            completed.add(key)
    return completed


def write_manifest(path, args, input_roots, total_inputs, rows):
    ok_labels = sum(1 for row in rows if row.get("label") in {"cogent", "ttgt", "tie"})
    manifest = {
        "binary": str(args.binary),
        "input_roots": [str(root) for root in input_roots],
        "output_dir": str(args.output_dir),
        "backends": list(args.backends),
        "precision": args.precision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "tie_tolerance": args.tie_tolerance,
        "total_inputs_seen": total_inputs,
        "rows_written": len(rows),
        "rows_with_label": ok_labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    if not args.binary.exists():
        raise FileNotFoundError(f"Binary not found: {args.binary}")

    input_roots = detect_input_roots(args.input_roots)
    all_problem_files = collect_problem_files(input_roots)
    sharded_files = apply_sharding(all_problem_files, args.num_shards, args.shard_index)
    if args.limit > 0:
        sharded_files = sharded_files[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = args.output_dir / f"training_data_{args.precision}_shard{args.shard_index:03d}-of-{args.num_shards:03d}.jsonl"
    dataset_csv = args.output_dir / f"training_data_{args.precision}_shard{args.shard_index:03d}-of-{args.num_shards:03d}.csv"
    manifest_path = args.output_dir / f"training_data_{args.precision}_shard{args.shard_index:03d}-of-{args.num_shards:03d}.manifest.json"

    if not args.append and dataset_jsonl.exists():
        dataset_jsonl.unlink()

    rows = load_existing_rows(dataset_jsonl) if args.append else []
    row_index_by_key = {
        (row.get("input_path", ""), row.get("precision", "")): index
        for index, row in enumerate(rows)
    }
    completed = completed_input_keys(rows, args.backends) if args.append else set()

    log(f"Python executable : {sys.executable}", args.verbose)
    log(f"Benchmark binary  : {args.binary}", args.verbose)
    log(f"Input roots       : {', '.join(str(root) for root in input_roots)}", args.verbose)
    log(f"Shard             : {args.shard_index}/{args.num_shards}", args.verbose)
    log(f"Problems in shard : {len(sharded_files)}", args.verbose)
    log(f"Output JSONL      : {dataset_jsonl}", args.verbose)
    log(f"Output CSV        : {dataset_csv}", args.verbose)

    for index, input_path in enumerate(sharded_files, start=1):
        key = (str(input_path), args.precision)
        if key in completed:
            log(f"[skip {index}/{len(sharded_files)}] {input_path}", args.verbose)
            continue

        log(f"[run {index}/{len(sharded_files)}] {input_path}", args.verbose)
        backend_rows = {}
        for backend in args.backends:
            backend_row = run_problem(
                args.binary,
                input_path,
                backend,
                args.precision,
                args.warmup,
                args.repeats,
                args.timeout_sec,
            )
            backend_rows[backend] = backend_row
            summary = backend_row.get("status", "UNKNOWN")
            if backend_row.get("status") == "OK":
                summary += f" avg_ms={backend_row.get('avg_ms')} gflops={backend_row.get('gflops')}"
            elif backend_row.get("error_message"):
                summary += f" error={backend_row['error_message']}"
            log(f"  [{backend}] {summary}", args.verbose)

        joined = join_problem_rows(input_path, args.precision, backend_rows, args.tie_tolerance)
        if key in row_index_by_key:
            rows[row_index_by_key[key]] = joined
        else:
            row_index_by_key[key] = len(rows)
            rows.append(joined)

    write_jsonl(dataset_jsonl, rows)
    write_csv(dataset_csv, rows, args.backends)
    write_manifest(manifest_path, args, input_roots, len(sharded_files), rows)

    print(f"Wrote JSONL dataset to {dataset_jsonl}")
    print(f"Wrote CSV dataset to {dataset_csv}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
