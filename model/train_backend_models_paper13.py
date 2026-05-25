#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from train_backend_models import (
    build_feature_inventory,
    build_group_splits,
    build_numeric_feature_frame,
    build_primary_metric_summary,
    collect_benchmark_bucket_metrics,
    collect_internal_split_metrics,
    compute_classification_scores,
    compute_regression_selector_scores,
    ensure_fractions,
    evaluate_on_tccg_benchmark,
    extract_tccg_benchmark_rows,
    index_for_groups,
    load_training_dataframe,
    permutation_feature_importance,
    predict_regression_times,
    run_group_cross_validation,
    save_pickle,
    summarize_top_feature_importance,
    train_classification_model,
    train_full_classification,
    train_full_regression,
    train_regression_models,
)


PAPER_13_FEATURES = [
    "internal_volume",
    "operation_count",
    "size_C",
    "size_A",
    "modeB_pattern_id_sum",
    "num_internal_modes",
    "size_B",
    "modeA_pattern_pos_1",
    "modeB_pattern_id_prod",
    "mean_extent",
    "modeA_pattern_pos_2",
    "modeB_pattern_pos_2",
    "modeA_pattern_id_last",
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train backend-selection models using only the 13 paper-selected features."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "model" / "data",
        help="Root directory containing collected training CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "model" / "results" / "backend_selector_paper13",
        help="Directory to write models, metrics, and reports.",
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
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--group-cv-folds", type=int, default=5)
    parser.add_argument("--skip-benchmark-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_fractions(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_training_dataframe(args.data_root)
    full_feature_df = build_numeric_feature_frame(df)
    missing = [feature for feature in PAPER_13_FEATURES if feature not in full_feature_df.columns]
    if missing:
        raise KeyError(f"Missing expected paper features: {missing}")
    feature_df = full_feature_df[PAPER_13_FEATURES].copy()

    feature_inventory_df = build_feature_inventory(PAPER_13_FEATURES)
    feature_inventory_path = args.output_dir / "feature_inventory.csv"
    feature_inventory_df.to_csv(feature_inventory_path, index=False)

    split = build_group_splits(df, args.seed, args.train_fraction, args.val_fraction)
    split_payload = {
        "train_groups": split.train_groups,
        "val_groups": split.val_groups,
        "test_groups": split.test_groups,
    }
    (args.output_dir / "split_groups.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    regression_artifacts, regression_val, regression_test = train_regression_models(
        feature_df, df, split, args.seed
    )
    classification_artifacts, classification_val, classification_test = train_classification_model(
        feature_df, df, split, args.seed
    )

    test_idx = index_for_groups(df, split.test_groups)
    X_test = feature_df.loc[test_idx, PAPER_13_FEATURES].copy()
    y_test_df = df.loc[test_idx].copy()

    classification_importance_df = permutation_feature_importance(
        X_test,
        scorer=lambda X: compute_classification_scores(
            y_test_df,
            classification_artifacts["model"].predict(
                classification_artifacts["imputer"].transform(X[classification_artifacts["columns"]])
            ),
        ),
        rng=rng,
    )
    classification_importance_path = args.output_dir / "feature_importance_classification.csv"
    classification_importance_df.merge(
        feature_inventory_df[["feature_name", "feature_group", "description"]],
        on="feature_name",
        how="left",
    ).to_csv(classification_importance_path, index=False)

    regression_importance_df = permutation_feature_importance(
        X_test,
        scorer=lambda X: compute_regression_selector_scores(y_test_df, X, regression_artifacts),
        rng=rng,
    )
    regression_importance_path = args.output_dir / "feature_importance_regression_selector.csv"
    regression_importance_df.merge(
        feature_inventory_df[["feature_name", "feature_group", "description"]],
        on="feature_name",
        how="left",
    ).to_csv(regression_importance_path, index=False)

    split_metrics = {
        "train": collect_internal_split_metrics(
            df,
            feature_df,
            index_for_groups(df, split.train_groups),
            regression_artifacts,
            classification_artifacts,
        ),
        "validation": collect_internal_split_metrics(
            df,
            feature_df,
            index_for_groups(df, split.val_groups),
            regression_artifacts,
            classification_artifacts,
        ),
        "test": collect_internal_split_metrics(
            df,
            feature_df,
            index_for_groups(df, split.test_groups),
            regression_artifacts,
            classification_artifacts,
        ),
    }

    report = {
        "dataset": {
            "rows": int(len(df)),
            "equation_groups": int(df["equation_group"].nunique()),
            "label_distribution": df["exact_winner"].value_counts().to_dict(),
            "tie_label_distribution": df["label"].value_counts().to_dict(),
        },
        "feature_policy": {
            "name": "paper_13_feature_subset",
            "uses_bench_rand_problem_cpp": False,
            "uses_backend_plan_features": False,
            "selected_features": PAPER_13_FEATURES,
        },
        "feature_summary": {
            "feature_count": len(PAPER_13_FEATURES),
            "feature_inventory_file": str(feature_inventory_path),
            "feature_group_counts": feature_inventory_df["feature_group"].value_counts().to_dict(),
            "top_classification_feature_importance": summarize_top_feature_importance(
                classification_importance_df, feature_inventory_df, top_k=len(PAPER_13_FEATURES)
            ),
            "top_regression_selector_feature_importance": summarize_top_feature_importance(
                regression_importance_df, feature_inventory_df, top_k=len(PAPER_13_FEATURES)
            ),
        },
        "split_distribution_summary": {
            "train": split_metrics["train"]["distribution"],
            "validation": split_metrics["validation"]["distribution"],
            "test": split_metrics["test"]["distribution"],
        },
        "validation": {
            "regression": regression_val,
            "classification": classification_val,
        },
        "test": {
            "regression": regression_test,
            "classification": classification_test,
        },
        "bucketed_internal_performance": {
            "train": split_metrics["train"]["bucketed"],
            "validation": split_metrics["validation"]["bucketed"],
            "test": split_metrics["test"]["bucketed"],
        },
    }

    preferred_strategy = (
        "classification"
        if classification_val["mean_time_regret_ratio"] <= regression_val["mean_time_regret_ratio"]
        else "regression"
    )
    report["model_selection"] = {
        "selected_on_validation": preferred_strategy,
        "selection_metric": "mean_time_regret_ratio",
    }

    full_regression_artifacts = train_full_regression(feature_df, df, args.seed)
    full_classification_artifacts = train_full_classification(feature_df, df, args.seed)

    save_pickle(args.output_dir / "regression_artifacts.pkl", full_regression_artifacts)
    save_pickle(args.output_dir / "classification_artifacts.pkl", full_classification_artifacts)
    save_pickle(args.output_dir / "feature_columns.pkl", PAPER_13_FEATURES)
    report["artifacts"] = {
        "feature_inventory_file": str(feature_inventory_path),
        "classification_feature_importance_file": str(classification_importance_path),
        "regression_selector_feature_importance_file": str(regression_importance_path),
        "regression_artifacts_file": str(args.output_dir / "regression_artifacts.pkl"),
        "classification_artifacts_file": str(args.output_dir / "classification_artifacts.pkl"),
    }

    benchmark_summary = None
    if not args.skip_benchmark_eval:
        try:
            benchmark_df = extract_tccg_benchmark_rows(
                args.precision,
                Path(__file__).resolve().parents[1] / "benchmarks" / "tccg" / "tccg_cases.cpp",
            )
            benchmark_feature_full = build_numeric_feature_frame(benchmark_df)
            benchmark_feature_df = benchmark_feature_full[PAPER_13_FEATURES].copy()
            comparison_df, benchmark_summary = evaluate_on_tccg_benchmark(
                benchmark_df,
                args.benchmark_csv,
                full_regression_artifacts,
                full_classification_artifacts,
                benchmark_feature_df,
            )
            regression_pred_cogent_ms, regression_pred_ttgt_ms = predict_regression_times(
                full_regression_artifacts, benchmark_feature_df
            )
            comparison_df["regression_pred_cogent_ms"] = regression_pred_cogent_ms
            comparison_df["regression_pred_ttgt_ms"] = regression_pred_ttgt_ms
            comparison_df.to_csv(args.output_dir / "tccg_benchmark_comparison.csv", index=False)
            comparison_df.to_csv(args.output_dir / "tccg_benchmark_detailed_results.csv", index=False)
            report["tccg_benchmark"] = benchmark_summary
            report["tccg_benchmark"]["detailed_results_file"] = str(
                args.output_dir / "tccg_benchmark_detailed_results.csv"
            )
            report["tccg_benchmark"]["bucketed_performance"] = collect_benchmark_bucket_metrics(comparison_df)
        except Exception as exc:
            report["tccg_benchmark"] = {
                "status": "ERROR",
                "error_message": str(exc),
            }

    report["primary_metrics"] = build_primary_metric_summary(split_metrics, benchmark_summary)
    report["group_cross_validation"] = run_group_cross_validation(df, feature_df, args.seed, args.group_cv_folds)

    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
