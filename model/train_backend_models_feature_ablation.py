#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_backend_models import (
    build_feature_inventory,
    build_group_splits,
    build_numeric_feature_frame,
    choose_feature_columns,
    evaluate_on_tccg_benchmark,
    extract_tccg_benchmark_rows,
    load_training_dataframe,
    train_classification_model,
    train_full_classification,
    train_full_regression,
    train_regression_models,
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze low-importance features and run feature-set ablations without modifying the base trainer."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "model" / "data",
        help="Root directory containing collected training CSV files.",
    )
    parser.add_argument(
        "--base-results-dir",
        type=Path,
        default=repo_root / "model" / "results" / "backend_selector",
        help="Directory containing feature inventory and importance CSVs from the base trainer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "model" / "results" / "backend_selector_feature_ablation",
        help="Directory to write pruning analysis and ablation results.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=repo_root / "model" / "tcont_best.csv",
        help="Measured TCCG benchmark results used only for final evaluation.",
    )
    parser.add_argument("--precision", default="fp64", choices=["fp64"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--paper-top-k-per-model", type=int, default=8)
    parser.add_argument("--near-zero-regret-threshold", type=float, default=1e-5)
    parser.add_argument("--near-zero-accuracy-threshold", type=float, default=1e-4)
    parser.add_argument("--near-zero-gflops-threshold", type=float, default=1e-5)
    return parser.parse_args()


def load_importance_tables(base_results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(base_results_dir / "feature_inventory.csv")
    classification = pd.read_csv(base_results_dir / "feature_importance_classification.csv")
    regression = pd.read_csv(base_results_dir / "feature_importance_regression_selector.csv")
    return inventory, classification, regression


def build_combined_importance(
    inventory: pd.DataFrame,
    classification: pd.DataFrame,
    regression: pd.DataFrame,
) -> pd.DataFrame:
    combined = inventory.copy()
    cls = classification.add_prefix("classification_").rename(
        columns={"classification_feature_name": "feature_name"}
    )
    reg = regression.add_prefix("regression_").rename(
        columns={"regression_feature_name": "feature_name"}
    )
    combined = combined.merge(cls, on="feature_name", how="left")
    combined = combined.merge(reg, on="feature_name", how="left")
    return combined


def add_pruning_flags(
    combined: pd.DataFrame,
    regret_threshold: float,
    accuracy_threshold: float,
    gflops_threshold: float,
) -> pd.DataFrame:
    out = combined.copy()
    out["classification_nonpositive"] = (
        (out["classification_mean_total_time_regret_increase"] <= 0.0)
        & (out["classification_mean_accuracy_drop"] <= 0.0)
        & (out["classification_mean_total_gflops_efficiency_drop"] <= 0.0)
    )
    out["regression_nonpositive"] = (
        (out["regression_mean_total_time_regret_increase"] <= 0.0)
        & (out["regression_mean_accuracy_drop"] <= 0.0)
        & (out["regression_mean_total_gflops_efficiency_drop"] <= 0.0)
    )
    out["nonpositive_both_models"] = out["classification_nonpositive"] & out["regression_nonpositive"]

    out["classification_near_zero"] = (
        (out["classification_mean_total_time_regret_increase"] <= regret_threshold)
        & (out["classification_mean_accuracy_drop"] <= accuracy_threshold)
        & (out["classification_mean_total_gflops_efficiency_drop"] <= gflops_threshold)
    )
    out["regression_near_zero"] = (
        (out["regression_mean_total_time_regret_increase"] <= regret_threshold)
        & (out["regression_mean_accuracy_drop"] <= accuracy_threshold)
        & (out["regression_mean_total_gflops_efficiency_drop"] <= gflops_threshold)
    )
    out["near_zero_both_models"] = out["classification_near_zero"] & out["regression_near_zero"]
    return out


def rank_feature_names(df: pd.DataFrame, prefix: str, top_k: int) -> list[str]:
    ranked = df.sort_values(
        [f"{prefix}_mean_total_time_regret_increase", f"{prefix}_mean_accuracy_drop"],
        ascending=[False, False],
    )
    return ranked.head(top_k)["feature_name"].tolist()


def build_paper_feature_table(combined: pd.DataFrame, top_k_per_model: int) -> pd.DataFrame:
    top_cls = rank_feature_names(combined, "classification", top_k_per_model)
    top_reg = rank_feature_names(combined, "regression", top_k_per_model)
    selected = []
    for name in top_cls + top_reg:
        if name not in selected:
            selected.append(name)

    paper = combined[combined["feature_name"].isin(selected)].copy()
    paper["selected_by_classification_topk"] = paper["feature_name"].isin(top_cls)
    paper["selected_by_regression_topk"] = paper["feature_name"].isin(top_reg)
    paper["paper_priority_score"] = (
        paper["classification_mean_total_time_regret_increase"].fillna(0.0)
        + paper["regression_mean_total_time_regret_increase"].fillna(0.0)
    )
    paper = paper.sort_values("paper_priority_score", ascending=False)
    return paper[
        [
            "feature_name",
            "feature_group",
            "description",
            "selected_by_classification_topk",
            "selected_by_regression_topk",
            "classification_mean_total_time_regret_increase",
            "classification_mean_accuracy_drop",
            "regression_mean_total_time_regret_increase",
            "regression_mean_accuracy_drop",
        ]
    ]


def write_markdown_table(df: pd.DataFrame, path: Path):
    headers = [
        "Feature",
        "Group",
        "Description",
        "Cls top-k",
        "Reg top-k",
        "Cls regret +",
        "Reg regret +",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature_name"]),
                    str(row["feature_group"]),
                    str(row["description"]),
                    "Y" if row["selected_by_classification_topk"] else "",
                    "Y" if row["selected_by_regression_topk"] else "",
                    f"{row['classification_mean_total_time_regret_increase']:.6g}",
                    f"{row['regression_mean_total_time_regret_increase']:.6g}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_feature_subset(
    subset_name: str,
    selected_columns: list[str],
    full_feature_df: pd.DataFrame,
    df: pd.DataFrame,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    benchmark_csv: Path,
    precision: str,
) -> dict:
    subset_feature_df = full_feature_df[selected_columns].copy()
    split = build_group_splits(df, seed, train_fraction, val_fraction)

    regression_artifacts, regression_val, regression_test = train_regression_models(
        subset_feature_df, df, split, seed
    )
    classification_artifacts, classification_val, classification_test = train_classification_model(
        subset_feature_df, df, split, seed
    )

    benchmark_df = extract_tccg_benchmark_rows(
        precision,
        Path(__file__).resolve().parents[1] / "benchmarks" / "tccg" / "tccg_cases.cpp",
    )
    benchmark_feature_full = build_numeric_feature_frame(benchmark_df)
    benchmark_feature_subset = benchmark_feature_full[selected_columns].copy()

    full_regression = train_full_regression(subset_feature_df, df, seed)
    full_classification = train_full_classification(subset_feature_df, df, seed)
    _, benchmark_summary = evaluate_on_tccg_benchmark(
        benchmark_df,
        benchmark_csv,
        full_regression,
        full_classification,
        benchmark_feature_subset,
    )

    return {
        "subset_name": subset_name,
        "feature_count": int(len(selected_columns)),
        "validation_regression_total_time_regret_ratio": regression_val["total_time_regret_ratio"],
        "validation_classification_total_time_regret_ratio": classification_val["total_time_regret_ratio"],
        "test_regression_total_time_regret_ratio": regression_test["total_time_regret_ratio"],
        "test_classification_total_time_regret_ratio": classification_test["total_time_regret_ratio"],
        "test_regression_accuracy": regression_test["accuracy"],
        "test_classification_accuracy": classification_test["accuracy"],
        "test_regression_total_gflops_efficiency": regression_test["total_gflops_efficiency"],
        "test_classification_total_gflops_efficiency": classification_test["total_gflops_efficiency"],
        "tccg_regression_agreement": benchmark_summary["regression"]["agreement_with_tcont_best"],
        "tccg_classification_agreement": benchmark_summary["classification"]["agreement_with_tcont_best"],
        "tccg_regression_total_gflops_efficiency": benchmark_summary["regression"]["total_gflops_efficiency"],
        "tccg_classification_total_gflops_efficiency": benchmark_summary["classification"]["total_gflops_efficiency"],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory, classification, regression = load_importance_tables(args.base_results_dir)
    combined = build_combined_importance(inventory, classification, regression)
    combined = add_pruning_flags(
        combined,
        regret_threshold=args.near_zero_regret_threshold,
        accuracy_threshold=args.near_zero_accuracy_threshold,
        gflops_threshold=args.near_zero_gflops_threshold,
    )

    combined_path = args.output_dir / "feature_importance_combined.csv"
    combined.to_csv(combined_path, index=False)

    removal_candidates = combined[
        [
            "feature_name",
            "feature_group",
            "description",
            "nonpositive_both_models",
            "near_zero_both_models",
            "classification_mean_total_time_regret_increase",
            "classification_mean_accuracy_drop",
            "regression_mean_total_time_regret_increase",
            "regression_mean_accuracy_drop",
        ]
    ].sort_values(
        ["near_zero_both_models", "nonpositive_both_models", "feature_group", "feature_name"],
        ascending=[False, False, True, True],
    )
    removal_candidates_path = args.output_dir / "feature_pruning_candidates.csv"
    removal_candidates.to_csv(removal_candidates_path, index=False)

    paper_table = build_paper_feature_table(combined, args.paper_top_k_per_model)
    paper_table_path = args.output_dir / "paper_feature_table.csv"
    paper_table.to_csv(paper_table_path, index=False)
    write_markdown_table(paper_table, args.output_dir / "paper_feature_table.md")

    df = load_training_dataframe(args.data_root)
    full_feature_df = build_numeric_feature_frame(df)
    selected_feature_columns = choose_feature_columns(full_feature_df)
    feature_inventory = build_feature_inventory(selected_feature_columns)
    all_columns = feature_inventory["feature_name"].tolist()

    near_zero_remove = combined.loc[combined["near_zero_both_models"], "feature_name"].tolist()
    nonpositive_remove = combined.loc[combined["nonpositive_both_models"], "feature_name"].tolist()
    paper_columns = paper_table["feature_name"].tolist()

    feature_sets = {
        "full": all_columns,
        "drop_nonpositive_both": [column for column in all_columns if column not in nonpositive_remove],
        "drop_near_zero_both": [column for column in all_columns if column not in near_zero_remove],
        "paper_top_union": paper_columns,
    }

    ablation_rows = []
    for subset_name, selected_columns in feature_sets.items():
        ablation_rows.append(
            evaluate_feature_subset(
                subset_name=subset_name,
                selected_columns=selected_columns,
                full_feature_df=full_feature_df,
                df=df,
                seed=args.seed,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                benchmark_csv=args.benchmark_csv,
                precision=args.precision,
            )
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("feature_count")
    ablation_path = args.output_dir / "feature_set_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)

    report = {
        "input_results_dir": str(args.base_results_dir),
        "feature_counts": {
            "full": int(len(all_columns)),
            "nonpositive_both_models": int(len(nonpositive_remove)),
            "near_zero_both_models": int(len(near_zero_remove)),
            "paper_top_union": int(len(paper_columns)),
        },
        "files": {
            "combined_importance": str(combined_path),
            "pruning_candidates": str(removal_candidates_path),
            "paper_feature_table_csv": str(paper_table_path),
            "paper_feature_table_md": str(args.output_dir / "paper_feature_table.md"),
            "feature_set_ablation": str(ablation_path),
        },
        "thresholds": {
            "near_zero_regret_threshold": args.near_zero_regret_threshold,
            "near_zero_accuracy_threshold": args.near_zero_accuracy_threshold,
            "near_zero_gflops_threshold": args.near_zero_gflops_threshold,
        },
        "paper_feature_names": paper_columns,
        "ablation_summary": ablation_rows,
    }
    (args.output_dir / "feature_ablation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
