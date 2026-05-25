#!/usr/bin/env python3
import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


COMMON_NUMERIC_COLUMNS = [
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
]

COGENT_STAGE_COLUMNS = [
    "cogent_stage_available",
    "cogent_stage_total_ms",
    "cogent_stage_transpose_a_ms",
    "cogent_stage_transpose_b_ms",
    "cogent_stage_gemm_ms",
    "cogent_stage_transpose_c_ms",
]

TTGT_STAGE_COLUMNS = [
    "ttgt_stage_available",
    "ttgt_stage_total_ms",
    "ttgt_stage_transpose_a_ms",
    "ttgt_stage_transpose_b_ms",
    "ttgt_stage_gemm_ms",
    "ttgt_stage_transpose_c_ms",
]

MAX_TENSOR_RANK = 6
SIZE_BUCKET_LABELS = ["small", "medium", "large", "xlarge"]
SIZE_BUCKET_BINS = [-np.inf, 0.1, 1.0, 10.0, np.inf]

FEATURE_DESCRIPTIONS = {
    "num_modes_C": "Number of output tensor modes in C.",
    "num_modes_A": "Number of left input tensor modes in A.",
    "num_modes_B": "Number of right input tensor modes in B.",
    "num_unique_modes": "Number of unique indices appearing in the contraction.",
    "num_internal_modes": "Number of contracted indices removed from the output.",
    "num_output_modes": "Number of output indices in the result tensor.",
    "num_output_only_left_modes": "Output indices that appear only in A and C.",
    "num_output_only_right_modes": "Output indices that appear only in B and C.",
    "num_output_shared_modes": "Output indices shared across A, B, and C.",
    "size_A": "Total element count of tensor A.",
    "size_B": "Total element count of tensor B.",
    "size_C": "Total element count of tensor C.",
    "output_volume": "Total element count of the output tensor.",
    "internal_volume": "Product of contracted index extents.",
    "operation_count": "Estimated floating-point work, 2 * product of all extents.",
    "min_extent": "Minimum index extent in the contraction.",
    "max_extent": "Maximum index extent in the contraction.",
    "mean_extent": "Mean index extent in the contraction.",
    "std_extent": "Standard deviation of index extents.",
    "extent_ratio_max_min": "Largest extent divided by smallest extent.",
    "log2_size_A": "Base-2 log of tensor A size.",
    "log2_size_B": "Base-2 log of tensor B size.",
    "log2_size_C": "Base-2 log of tensor C size.",
    "log2_internal_volume": "Base-2 log of contracted volume.",
}


@dataclass
class DatasetSplit:
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and compare backend-selection models for Cogent vs TTGT."
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
        default=repo_root / "model" / "training_runs" / "backend_selector",
        help="Directory to write models, metrics, and prediction reports.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=repo_root / "model" / "tcont_best.csv",
        help="Measured TCCG benchmark results used only for final evaluation.",
    )
    parser.add_argument("--precision", default="fp64", choices=["fp64"], help="Benchmark precision.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Fraction of groups used for train split.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of groups used for validation split.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of groups used for test split.")
    parser.add_argument("--group-cv-folds", type=int, default=5, help="Number of group CV folds. Use 0 to disable.")
    parser.add_argument(
        "--skip-benchmark-eval",
        action="store_true",
        help="Skip final evaluation against model/tcont_best.csv.",
    )
    return parser.parse_args()


def ensure_fractions(args):
    total = args.train_fraction + args.val_fraction + args.test_fraction
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train/val/test fractions must sum to 1.0")


def load_training_dataframe(data_root: Path) -> pd.DataFrame:
    csv_paths = sorted(data_root.glob("**/training_data_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No training CSV files found under {data_root}")
    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["cogent_status"] == "OK") & (df["ttgt_status"] == "OK")].copy()
    if df.empty:
        raise ValueError("No rows with both backends in OK status.")
    df["exact_winner"] = np.where(df["cogent_avg_ms"] <= df["ttgt_avg_ms"], "cogent", "ttgt")
    df["exact_winner_is_ttgt"] = (df["exact_winner"] == "ttgt").astype(int)
    df["speedup_over_oracle"] = np.where(
        df["cogent_avg_ms"] <= df["ttgt_avg_ms"],
        df["ttgt_avg_ms"] / df["cogent_avg_ms"],
        df["cogent_avg_ms"] / df["ttgt_avg_ms"],
    )
    return df.reset_index(drop=True)


def parse_json_value(value, default):
    if pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sequence_summary(prefix: str, seq: Iterable[float]) -> dict[str, float]:
    seq = list(seq)
    if not seq:
        return {
            f"{prefix}_len": 0.0,
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_prod": 0.0,
            f"{prefix}_first": 0.0,
            f"{prefix}_last": 0.0,
            f"{prefix}_spread": 0.0,
        }
    product = 1.0
    for value in seq:
        product *= float(value)
    return {
        f"{prefix}_len": float(len(seq)),
        f"{prefix}_sum": float(sum(seq)),
        f"{prefix}_mean": float(sum(seq) / len(seq)),
        f"{prefix}_min": float(min(seq)),
        f"{prefix}_max": float(max(seq)),
        f"{prefix}_prod": float(product),
        f"{prefix}_first": float(seq[0]),
        f"{prefix}_last": float(seq[-1]),
        f"{prefix}_spread": float(max(seq) - min(seq)),
    }


def inversion_count(seq: list[int]) -> int:
    count = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                count += 1
    return count


def canonical_mode_pattern_features(mode_c: str, mode_a: str, mode_b: str) -> dict[str, float]:
    mode_order = []
    mapping = {}
    for mode in list(mode_c) + list(mode_a) + list(mode_b):
        if mode not in mapping:
            mapping[mode] = len(mapping) + 1
            mode_order.append(mode)

    features = {
        "pattern_unique_mode_count": float(len(mapping)),
        "pattern_mode_reuse_count": float(len(mode_c) + len(mode_a) + len(mode_b) - len(mapping)),
    }

    for tensor_name, mode_string in (("modeC", mode_c), ("modeA", mode_a), ("modeB", mode_b)):
        ids = [mapping[mode] for mode in mode_string]
        features.update(sequence_summary(f"{tensor_name}_pattern_id", ids))
        features[f"{tensor_name}_rank"] = float(len(mode_string))
        features[f"{tensor_name}_adjacent_repeat_count"] = float(
            sum(1 for left, right in zip(mode_string, mode_string[1:]) if left == right)
        )
        for index in range(MAX_TENSOR_RANK):
            features[f"{tensor_name}_pattern_pos_{index}"] = float(ids[index]) if index < len(ids) else 0.0

    return features


def build_numeric_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        features = {column: safe_float(row[column]) for column in COMMON_NUMERIC_COLUMNS}
        features.update(canonical_mode_pattern_features(str(row["modeC"]), str(row["modeA"]), str(row["modeB"])))
        rows.append(features)
    return pd.DataFrame(rows).fillna(0.0)


def choose_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in feature_df.columns
        if column in COMMON_NUMERIC_COLUMNS
        or column.startswith("pattern_")
        or column.startswith("modeA_pattern_")
        or column.startswith("modeB_pattern_")
        or column.startswith("modeC_pattern_")
    ]


def infer_feature_category(feature_name: str) -> str:
    if feature_name in COMMON_NUMERIC_COLUMNS:
        return "numeric_problem_size"
    if feature_name.startswith("pattern_"):
        return "equation_pattern_global"
    if feature_name.startswith("modeA_pattern_"):
        return "equation_pattern_A"
    if feature_name.startswith("modeB_pattern_"):
        return "equation_pattern_B"
    if feature_name.startswith("modeC_pattern_"):
        return "equation_pattern_C"
    return "other"


def describe_feature(feature_name: str) -> str:
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]

    pattern_match = re.match(r"mode([ABC])_pattern_pos_(\d+)", feature_name)
    if pattern_match:
        tensor_name, index = pattern_match.groups()
        return f"Canonicalized mode ID at position {int(index)} of tensor {tensor_name}."

    if feature_name.endswith("_pattern_id_len"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Length of the canonical mode-ID sequence for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_sum"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Sum of canonical mode IDs for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_mean"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Mean canonical mode ID for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_min"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Minimum canonical mode ID for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_max"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Maximum canonical mode ID for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_prod"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Product of canonical mode IDs for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_first"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"First canonical mode ID for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_last"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Last canonical mode ID for tensor {tensor_name}."
    if feature_name.endswith("_pattern_id_spread"):
        tensor_name = feature_name.split("_")[0][-1]
        return f"Range of canonical mode IDs for tensor {tensor_name}."
    if feature_name.endswith("_rank"):
        tensor_name = feature_name[4]
        return f"Tensor rank for tensor {tensor_name}."
    if feature_name.endswith("_adjacent_repeat_count"):
        tensor_name = feature_name[4]
        return f"Count of adjacent repeated modes in tensor {tensor_name}."
    if feature_name == "pattern_unique_mode_count":
        return "Number of unique canonical mode IDs across C, A, and B."
    if feature_name == "pattern_mode_reuse_count":
        return "Total repeated mode occurrences after canonicalization."
    return "Derived structural feature from equation and problem size."


def build_feature_inventory(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for order, feature_name in enumerate(feature_columns):
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": infer_feature_category(feature_name),
                "description": describe_feature(feature_name),
                "used_by_regression": True,
                "used_by_classification": True,
                "feature_order": order,
            }
        )
    return pd.DataFrame(rows)


def build_group_splits(df: pd.DataFrame, seed: int, train_fraction: float, val_fraction: float) -> DatasetSplit:
    group_stats = (
        df.groupby("equation_group")["exact_winner_is_ttgt"]
        .mean()
        .reset_index()
        .rename(columns={"exact_winner_is_ttgt": "ttgt_ratio"})
    )
    group_stats["stratify_label"] = (group_stats["ttgt_ratio"] >= 0.5).astype(int)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    group_names = group_stats["equation_group"].to_numpy()
    stratify_labels = group_stats["stratify_label"].to_numpy()
    train_idx, holdout_idx = next(splitter.split(group_names, stratify_labels))

    train_groups = sorted(group_names[train_idx].tolist())
    holdout_groups = group_stats.iloc[holdout_idx].reset_index(drop=True)

    val_share_of_holdout = val_fraction / (1.0 - train_fraction)
    holdout_splitter = StratifiedShuffleSplit(n_splits=1, train_size=val_share_of_holdout, random_state=seed + 1)
    val_idx, test_idx = next(
        holdout_splitter.split(
            holdout_groups["equation_group"].to_numpy(),
            holdout_groups["stratify_label"].to_numpy(),
        )
    )

    val_groups = sorted(holdout_groups.iloc[val_idx]["equation_group"].tolist())
    test_groups = sorted(holdout_groups.iloc[test_idx]["equation_group"].tolist())
    return DatasetSplit(train_groups=train_groups, val_groups=val_groups, test_groups=test_groups)


def index_for_groups(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
    return df.index[df["equation_group"].isin(groups)].to_numpy()


def add_oracle_columns(
    df: pd.DataFrame,
    cogent_ms_col: str = "cogent_avg_ms",
    ttgt_ms_col: str = "ttgt_avg_ms",
    truth_col: str = "exact_winner",
) -> pd.DataFrame:
    df = df.copy()
    df["oracle_ms"] = np.minimum(df[cogent_ms_col].to_numpy(), df[ttgt_ms_col].to_numpy())
    if truth_col not in df.columns:
        df[truth_col] = np.where(df[cogent_ms_col] <= df[ttgt_ms_col], "cogent", "ttgt")
    return df


def assign_size_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=SIZE_BUCKET_BINS, labels=SIZE_BUCKET_LABELS, right=False)


def summarize_numeric_series(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p10": float(series.quantile(0.10)),
        "p90": float(series.quantile(0.90)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def summarize_split_distribution(df: pd.DataFrame) -> dict:
    df = add_oracle_columns(df)
    return {
        "rows": int(len(df)),
        "equation_groups": int(df["equation_group"].nunique()),
        "winner_distribution": df["exact_winner"].value_counts().to_dict(),
        "winner_proportion": df["exact_winner"].value_counts(normalize=True).to_dict(),
        "oracle_ms": summarize_numeric_series(df["oracle_ms"]),
        "operation_count": summarize_numeric_series(df["operation_count"]),
        "size_A": summarize_numeric_series(df["size_A"]),
        "size_B": summarize_numeric_series(df["size_B"]),
        "size_C": summarize_numeric_series(df["size_C"]),
        "output_volume": summarize_numeric_series(df["output_volume"]),
        "internal_volume": summarize_numeric_series(df["internal_volume"]),
    }


def evaluate_backend_predictions_generic(
    df: pd.DataFrame,
    predicted_backend: np.ndarray,
    truth_col: str,
    cogent_ms_col: str,
    ttgt_ms_col: str,
    cogent_gflops_col: str,
    ttgt_gflops_col: str,
    name: str,
) -> dict:
    truth = df[truth_col].to_numpy()
    labels = ["cogent", "ttgt"]
    selected_times = np.where(predicted_backend == "cogent", df[cogent_ms_col].to_numpy(), df[ttgt_ms_col].to_numpy())
    best_times = np.minimum(df[cogent_ms_col].to_numpy(), df[ttgt_ms_col].to_numpy())
    selected_gflops = np.where(
        predicted_backend == "cogent",
        df[cogent_gflops_col].to_numpy(),
        df[ttgt_gflops_col].to_numpy(),
    )
    best_gflops = np.maximum(df[cogent_gflops_col].to_numpy(), df[ttgt_gflops_col].to_numpy())

    accuracy = float(accuracy_score(truth, predicted_backend))
    unique_truth = np.unique(truth)
    unique_pred = np.unique(predicted_backend)
    if len(np.union1d(unique_truth, unique_pred)) < 2:
        balanced_accuracy = accuracy
    else:
        balanced_accuracy = float(balanced_accuracy_score(truth, predicted_backend))
    macro_f1 = float(f1_score(truth, predicted_backend, labels=labels, average="macro", zero_division=0))

    return {
        "strategy": name,
        "rows": int(len(df)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mean_selected_ms": float(np.mean(selected_times)),
        "mean_oracle_ms": float(np.mean(best_times)),
        "mean_time_regret_ratio": float(np.mean(selected_times / best_times)),
        "total_time_regret_ratio": float(np.sum(selected_times) / np.sum(best_times)),
        "mean_selected_gflops": float(np.mean(selected_gflops)),
        "mean_oracle_gflops": float(np.mean(best_gflops)),
        "mean_gflops_efficiency": float(np.mean(selected_gflops / best_gflops)),
        "total_gflops_efficiency": float(np.sum(selected_gflops) / np.sum(best_gflops)),
    }


def evaluate_backend_predictions(df: pd.DataFrame, predicted_backend: np.ndarray, name: str) -> dict:
    return evaluate_backend_predictions_generic(
        df,
        predicted_backend,
        truth_col="exact_winner",
        cogent_ms_col="cogent_avg_ms",
        ttgt_ms_col="ttgt_avg_ms",
        cogent_gflops_col="cogent_gflops",
        ttgt_gflops_col="ttgt_gflops",
        name=name,
    )


def bucketize_evaluation(
    df: pd.DataFrame,
    predicted_backend: np.ndarray,
    strategy_name: str,
    truth_col: str,
    cogent_ms_col: str,
    ttgt_ms_col: str,
    cogent_gflops_col: str,
    ttgt_gflops_col: str,
    oracle_col: str,
) -> dict:
    tmp = df.copy()
    tmp["_predicted_backend"] = predicted_backend
    tmp["_size_bucket"] = assign_size_bucket(tmp[oracle_col])

    bucket_report = {}
    for bucket in SIZE_BUCKET_LABELS:
        sub = tmp[tmp["_size_bucket"] == bucket]
        if sub.empty:
            continue
        metrics = evaluate_backend_predictions_generic(
            sub,
            sub["_predicted_backend"].to_numpy(),
            truth_col=truth_col,
            cogent_ms_col=cogent_ms_col,
            ttgt_ms_col=ttgt_ms_col,
            cogent_gflops_col=cogent_gflops_col,
            ttgt_gflops_col=ttgt_gflops_col,
            name=strategy_name,
        )
        metrics["oracle_ms_distribution"] = summarize_numeric_series(sub[oracle_col])
        bucket_report[bucket] = metrics
    return bucket_report


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def train_regression_models(
    feature_df: pd.DataFrame,
    df: pd.DataFrame,
    split: DatasetSplit,
    seed: int,
) -> tuple[dict, dict, dict]:
    train_idx = index_for_groups(df, split.train_groups)
    val_idx = index_for_groups(df, split.val_groups)
    test_idx = index_for_groups(df, split.test_groups)

    common_columns = choose_feature_columns(feature_df)
    cogent_columns = common_columns
    ttgt_columns = common_columns

    X_train_cogent = feature_df.loc[train_idx, cogent_columns]
    X_train_ttgt = feature_df.loc[train_idx, ttgt_columns]
    y_train_cogent = np.log(df.loc[train_idx, "cogent_avg_ms"].to_numpy())
    y_train_ttgt = np.log(df.loc[train_idx, "ttgt_avg_ms"].to_numpy())

    cogent_imputer = fit_imputer(X_train_cogent)
    ttgt_imputer = fit_imputer(X_train_ttgt)

    cogent_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed,
    )
    ttgt_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed + 1,
    )

    cogent_model.fit(cogent_imputer.transform(X_train_cogent), y_train_cogent)
    ttgt_model.fit(ttgt_imputer.transform(X_train_ttgt), y_train_ttgt)

    artifacts = {
        "cogent_model": cogent_model,
        "ttgt_model": ttgt_model,
        "cogent_imputer": cogent_imputer,
        "ttgt_imputer": ttgt_imputer,
        "cogent_columns": cogent_columns,
        "ttgt_columns": ttgt_columns,
    }

    def predict_backend(index: np.ndarray) -> np.ndarray:
        cogent_pred = np.exp(
            cogent_model.predict(cogent_imputer.transform(feature_df.loc[index, cogent_columns]))
        )
        ttgt_pred = np.exp(ttgt_model.predict(ttgt_imputer.transform(feature_df.loc[index, ttgt_columns])))
        return np.where(cogent_pred <= ttgt_pred, "cogent", "ttgt")

    val_metrics = evaluate_backend_predictions(df.loc[val_idx], predict_backend(val_idx), "regression")
    test_metrics = evaluate_backend_predictions(df.loc[test_idx], predict_backend(test_idx), "regression")
    return artifacts, val_metrics, test_metrics


def train_classification_model(
    feature_df: pd.DataFrame,
    df: pd.DataFrame,
    split: DatasetSplit,
    seed: int,
) -> tuple[dict, dict, dict]:
    train_idx = index_for_groups(df, split.train_groups)
    val_idx = index_for_groups(df, split.val_groups)
    test_idx = index_for_groups(df, split.test_groups)

    columns = choose_feature_columns(feature_df)
    X_train = feature_df.loc[train_idx, columns]
    y_train = df.loc[train_idx, "exact_winner"].to_numpy()

    imputer = fit_imputer(X_train)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed,
    )
    model.fit(imputer.transform(X_train), y_train)

    artifacts = {
        "model": model,
        "imputer": imputer,
        "columns": columns,
    }

    def predict_backend(index: np.ndarray) -> np.ndarray:
        return model.predict(imputer.transform(feature_df.loc[index, columns]))

    val_metrics = evaluate_backend_predictions(df.loc[val_idx], predict_backend(val_idx), "classification")
    test_metrics = evaluate_backend_predictions(df.loc[test_idx], predict_backend(test_idx), "classification")
    return artifacts, val_metrics, test_metrics


def train_full_regression(feature_df: pd.DataFrame, df: pd.DataFrame, seed: int) -> dict:
    common_columns = choose_feature_columns(feature_df)
    cogent_columns = common_columns
    ttgt_columns = common_columns

    cogent_imputer = fit_imputer(feature_df[cogent_columns])
    ttgt_imputer = fit_imputer(feature_df[ttgt_columns])

    cogent_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed,
    )
    ttgt_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed + 1,
    )
    cogent_model.fit(cogent_imputer.transform(feature_df[cogent_columns]), np.log(df["cogent_avg_ms"].to_numpy()))
    ttgt_model.fit(ttgt_imputer.transform(feature_df[ttgt_columns]), np.log(df["ttgt_avg_ms"].to_numpy()))

    return {
        "cogent_model": cogent_model,
        "ttgt_model": ttgt_model,
        "cogent_imputer": cogent_imputer,
        "ttgt_imputer": ttgt_imputer,
        "cogent_columns": cogent_columns,
        "ttgt_columns": ttgt_columns,
    }


def train_full_classification(feature_df: pd.DataFrame, df: pd.DataFrame, seed: int) -> dict:
    columns = choose_feature_columns(feature_df)
    imputer = fit_imputer(feature_df[columns])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=40,
        random_state=seed,
    )
    model.fit(imputer.transform(feature_df[columns]), df["exact_winner"].to_numpy())
    return {
        "model": model,
        "imputer": imputer,
        "columns": columns,
    }


def normalize_best_method(value: str) -> str:
    value = value.strip().lower()
    if value == "direct":
        return "cogent"
    if value == "ttgt":
        return "ttgt"
    raise ValueError(f"Unexpected best_method value: {value}")


def parse_tccg_cases(cases_path: Path) -> list[dict]:
    text = cases_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"//\s*(?P<case_id>\d+):\s*(?P<comment>[^\n]+)\n\s*\{\s*"
        r"\{\s*(?P<modeC>[^}]*)\}\s*,\s*"
        r"\{\s*(?P<modeA>[^}]*)\}\s*,\s*"
        r"\{\s*(?P<modeB>[^}]*)\}\s*,\s*"
        r"'(?P<op>.)'\s*,\s*TconT::ScalarType::Float64,\s*"
        r"\{\s*(?P<extents>.*?)\s*\}\s*"
        r"\},",
        re.S,
    )
    cases = []
    for match in pattern.finditer(text):
        case_id = int(match.group("case_id"))
        if case_id == 0:
            continue
        mode_c = re.findall(r"'([A-Za-z])'", match.group("modeC"))
        mode_a = re.findall(r"'([A-Za-z])'", match.group("modeA"))
        mode_b = re.findall(r"'([A-Za-z])'", match.group("modeB"))
        extents = {
            mode: int(value)
            for mode, value in re.findall(r"\{'([A-Za-z])',\s*(\d+)\}", match.group("extents"))
        }
        cases.append(
            {
                "equation": case_id,
                "comment": match.group("comment").strip(),
                "modeC": mode_c,
                "modeA": mode_a,
                "modeB": mode_b,
                "extents": extents,
            }
        )
    if len(cases) != 48:
        raise ValueError(f"Expected 48 TCCG benchmark cases, found {len(cases)}")
    return cases


def build_equation_features_from_modes(
    mode_c: list[str],
    mode_a: list[str],
    mode_b: list[str],
    extents: dict[str, int],
    equation_name: str,
    equation_string: str,
) -> dict:
    size_a = math.prod(extents[mode] for mode in mode_a)
    size_b = math.prod(extents[mode] for mode in mode_b)
    size_c = math.prod(extents[mode] for mode in mode_c)

    unique_modes = sorted(extents.keys())
    internal_modes = sorted(mode for mode in mode_a if mode in mode_b and mode not in mode_c)
    output_only_left = sorted(mode for mode in mode_c if mode in mode_a and mode not in mode_b)
    output_only_right = sorted(mode for mode in mode_c if mode in mode_b and mode not in mode_a)
    output_shared = sorted(mode for mode in mode_c if mode in mode_a and mode in mode_b)

    internal_volume = math.prod(extents[mode] for mode in internal_modes) if internal_modes else 1
    operation_count = 2.0
    for mode in unique_modes:
        operation_count *= extents[mode]

    sorted_extents = sorted(extents.values())
    mean_extent = float(sum(sorted_extents) / len(sorted_extents))
    variance = sum((value - mean_extent) ** 2 for value in sorted_extents) / len(sorted_extents)

    return {
        "equation_name": equation_name,
        "equation_string": equation_string,
        "modeC": "".join(mode_c),
        "modeA": "".join(mode_a),
        "modeB": "".join(mode_b),
        "num_modes_C": len(mode_c),
        "num_modes_A": len(mode_a),
        "num_modes_B": len(mode_b),
        "num_unique_modes": len(unique_modes),
        "num_internal_modes": len(internal_modes),
        "num_output_modes": len(mode_c),
        "num_output_only_left_modes": len(output_only_left),
        "num_output_only_right_modes": len(output_only_right),
        "num_output_shared_modes": len(output_shared),
        "size_A": size_a,
        "size_B": size_b,
        "size_C": size_c,
        "output_volume": size_c,
        "internal_volume": internal_volume,
        "operation_count": float(operation_count),
        "min_extent": min(sorted_extents),
        "max_extent": max(sorted_extents),
        "mean_extent": mean_extent,
        "std_extent": math.sqrt(variance),
        "extent_ratio_max_min": max(sorted_extents) / min(sorted_extents),
        "log2_size_A": math.log2(size_a),
        "log2_size_B": math.log2(size_b),
        "log2_size_C": math.log2(size_c),
        "log2_internal_volume": math.log2(internal_volume),
        "mode_extents": json.dumps(extents),
        "sorted_extents": json.dumps(sorted_extents),
        "internal_modes": json.dumps(internal_modes),
    }


def build_tccg_equation_string(case: dict) -> str:
    mode_c = case["modeC"]
    mode_a = case["modeA"]
    mode_b = case["modeB"]
    extents = case["extents"]
    internal_modes = [mode for mode in mode_a if mode in mode_b and mode not in mode_c]
    sum_clause = ""
    if internal_modes:
        sum_items = ",".join(f"{mode},{extents[mode]}" for mode in internal_modes)
        sum_clause = f"sum({sum_items}) * "
    lhs = ",".join(f"{mode},{extents[mode]}" for mode in mode_c)
    rhs_a = ",".join(mode_a)
    rhs_b = ",".join(mode_b)
    return f"t3 [{lhs}] += {sum_clause}t2 [{rhs_a}] * v2 [{rhs_b}];"


def extract_tccg_benchmark_rows(precision: str, cases_path: Path) -> pd.DataFrame:
    cases = parse_tccg_cases(cases_path)
    flat_rows = []
    for case in cases:
        equation_string = build_tccg_equation_string(case)
        features = build_equation_features_from_modes(
            case["modeC"],
            case["modeA"],
            case["modeB"],
            case["extents"],
            case["comment"],
            equation_string,
        )
        flat = {
            "input_path": f"tccg://{case['equation']}",
            "precision": precision,
            "source_collection": "tccg_benchmark",
            "equation_group": f"tccg_{case['equation']}",
            "problem_file": f"tccg_{case['equation']}.in",
            "problem_id": case["equation"],
            "equation": case["equation"],
        }
        flat.update(features)
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)


def predict_with_regression(artifacts: dict, feature_df: pd.DataFrame) -> np.ndarray:
    cogent_pred, ttgt_pred = predict_regression_times(artifacts, feature_df)
    return np.where(cogent_pred <= ttgt_pred, "cogent", "ttgt")


def predict_with_classification(artifacts: dict, feature_df: pd.DataFrame) -> np.ndarray:
    return artifacts["model"].predict(artifacts["imputer"].transform(feature_df[artifacts["columns"]]))


def predict_regression_times(artifacts: dict, feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cogent_pred = np.exp(
        artifacts["cogent_model"].predict(
            artifacts["cogent_imputer"].transform(feature_df[artifacts["cogent_columns"]])
        )
    )
    ttgt_pred = np.exp(
        artifacts["ttgt_model"].predict(
            artifacts["ttgt_imputer"].transform(feature_df[artifacts["ttgt_columns"]])
        )
    )
    return cogent_pred, ttgt_pred


def compute_classification_scores(df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    metrics = evaluate_backend_predictions(df, predictions, "classification")
    return {
        "accuracy": float(metrics["accuracy"]),
        "total_time_regret_ratio": float(metrics["total_time_regret_ratio"]),
        "total_gflops_efficiency": float(metrics["total_gflops_efficiency"]),
    }


def compute_regression_selector_scores(df: pd.DataFrame, feature_df: pd.DataFrame, artifacts: dict) -> dict[str, float]:
    predictions = predict_with_regression(artifacts, feature_df)
    metrics = evaluate_backend_predictions(df, predictions, "regression")
    return {
        "accuracy": float(metrics["accuracy"]),
        "total_time_regret_ratio": float(metrics["total_time_regret_ratio"]),
        "total_gflops_efficiency": float(metrics["total_gflops_efficiency"]),
    }


def permutation_feature_importance(
    feature_df: pd.DataFrame,
    scorer,
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> pd.DataFrame:
    baseline = scorer(feature_df)
    rows = []

    for feature_name in feature_df.columns:
        accuracy_drops = []
        regret_increases = []
        efficiency_drops = []
        for _ in range(n_repeats):
            permuted = feature_df.copy()
            permuted[feature_name] = rng.permutation(permuted[feature_name].to_numpy())
            permuted_scores = scorer(permuted)
            accuracy_drops.append(baseline["accuracy"] - permuted_scores["accuracy"])
            regret_increases.append(
                permuted_scores["total_time_regret_ratio"] - baseline["total_time_regret_ratio"]
            )
            efficiency_drops.append(
                baseline["total_gflops_efficiency"] - permuted_scores["total_gflops_efficiency"]
            )

        rows.append(
            {
                "feature_name": feature_name,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_total_time_regret_ratio": baseline["total_time_regret_ratio"],
                "baseline_total_gflops_efficiency": baseline["total_gflops_efficiency"],
                "mean_accuracy_drop": float(np.mean(accuracy_drops)),
                "std_accuracy_drop": float(np.std(accuracy_drops)),
                "mean_total_time_regret_increase": float(np.mean(regret_increases)),
                "std_total_time_regret_increase": float(np.std(regret_increases)),
                "mean_total_gflops_efficiency_drop": float(np.mean(efficiency_drops)),
                "std_total_gflops_efficiency_drop": float(np.std(efficiency_drops)),
            }
        )

    importance_df = pd.DataFrame(rows)
    return importance_df.sort_values(
        ["mean_total_time_regret_increase", "mean_accuracy_drop"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_top_feature_importance(
    importance_df: pd.DataFrame,
    feature_inventory_df: pd.DataFrame,
    top_k: int = 15,
) -> list[dict]:
    merged = importance_df.merge(
        feature_inventory_df[["feature_name", "feature_group", "description"]],
        on="feature_name",
        how="left",
    )
    top = merged.head(top_k)
    return top.to_dict(orient="records")


def gflops_to_ms(operation_count: pd.Series | np.ndarray, gflops: pd.Series | np.ndarray) -> np.ndarray:
    operation_count = np.asarray(operation_count, dtype=float)
    gflops = np.asarray(gflops, dtype=float)
    return operation_count / (gflops * 1.0e6)


def evaluate_on_tccg_benchmark(
    benchmark_df: pd.DataFrame,
    tcont_best_csv: Path,
    regression_artifacts: dict,
    classification_artifacts: dict,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    truth_df = pd.read_csv(tcont_best_csv).copy()
    truth_df["equation"] = truth_df["equation"].astype(int)
    truth_df["actual_best_backend"] = truth_df["best_method"].map(normalize_best_method)

    comparison = benchmark_df[
        [
            "equation",
            "equation_group",
            "equation_name",
            "operation_count",
            "size_A",
            "size_B",
            "size_C",
            "output_volume",
            "internal_volume",
        ]
    ].copy()
    regression_pred_cogent_ms, regression_pred_ttgt_ms = predict_regression_times(regression_artifacts, feature_df)
    comparison["regression_pred_cogent_ms"] = regression_pred_cogent_ms
    comparison["regression_pred_ttgt_ms"] = regression_pred_ttgt_ms
    comparison["regression_pred_backend"] = np.where(
        regression_pred_cogent_ms <= regression_pred_ttgt_ms, "cogent", "ttgt"
    )
    comparison["classification_pred_backend"] = predict_with_classification(classification_artifacts, feature_df)

    comparison = comparison.merge(
        truth_df[
            [
                "equation",
                "actual_best_backend",
                "best_gflops",
                "direct_gflops",
                "ttgt_gflops",
            ]
        ],
        on="equation",
        how="left",
    )

    comparison["actual_cogent_ms"] = gflops_to_ms(comparison["operation_count"], comparison["direct_gflops"])
    comparison["actual_ttgt_ms"] = gflops_to_ms(comparison["operation_count"], comparison["ttgt_gflops"])
    comparison["actual_best_ms"] = np.minimum(comparison["actual_cogent_ms"], comparison["actual_ttgt_ms"])
    comparison["actual_cogent_backend"] = "cogent"
    comparison["actual_ttgt_backend"] = "ttgt"
    comparison["actual_best_gflops"] = comparison["best_gflops"]
    comparison["actual_cogent_gflops"] = comparison["direct_gflops"]
    comparison["actual_ttgt_gflops"] = comparison["ttgt_gflops"]
    comparison["actual_gflops_gap"] = comparison["best_gflops"] - np.minimum(
        comparison["direct_gflops"], comparison["ttgt_gflops"]
    )
    comparison["actual_speedup_best_over_other"] = np.where(
        comparison["actual_best_backend"] == "cogent",
        comparison["direct_gflops"] / comparison["ttgt_gflops"],
        comparison["ttgt_gflops"] / comparison["direct_gflops"],
    )

    for strategy in ("regression", "classification"):
        predicted = comparison[f"{strategy}_pred_backend"]
        comparison[f"{strategy}_matches_best"] = predicted == comparison["actual_best_backend"]
        comparison[f"{strategy}_predicted_gflops"] = np.where(
            predicted == "cogent",
            comparison["direct_gflops"],
            comparison["ttgt_gflops"],
        )
        comparison[f"{strategy}_gflops_efficiency"] = (
            comparison[f"{strategy}_predicted_gflops"] / comparison["best_gflops"]
        )
        comparison[f"{strategy}_gflops_regret"] = comparison["best_gflops"] - comparison[f"{strategy}_predicted_gflops"]

    comparison["regression_selected_backend_actual_gflops"] = comparison["regression_predicted_gflops"]
    comparison["classification_selected_backend_actual_gflops"] = comparison["classification_predicted_gflops"]

    summary = {}
    for strategy in ("regression", "classification"):
        summary[strategy] = {
            "benchmark_rows": int(len(comparison)),
            "agreement_with_tcont_best": float(comparison[f"{strategy}_matches_best"].mean()),
            "mean_gflops_efficiency": float(comparison[f"{strategy}_gflops_efficiency"].mean()),
            "total_gflops_efficiency": float(
                comparison[f"{strategy}_predicted_gflops"].sum() / comparison["best_gflops"].sum()
            ),
        }
    return comparison, summary


def save_pickle(path: Path, obj):
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def collect_internal_split_metrics(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    indices: np.ndarray,
    regression_artifacts: dict,
    classification_artifacts: dict,
) -> dict:
    sub_df = add_oracle_columns(df.loc[indices].copy())
    sub_feat = feature_df.loc[indices]
    regression_pred = predict_with_regression(regression_artifacts, sub_feat)
    classification_pred = predict_with_classification(classification_artifacts, sub_feat)

    return {
        "distribution": summarize_split_distribution(sub_df),
        "regression": evaluate_backend_predictions(sub_df, regression_pred, "regression"),
        "classification": evaluate_backend_predictions(sub_df, classification_pred, "classification"),
        "bucketed": {
            "regression": bucketize_evaluation(
                sub_df,
                regression_pred,
                "regression",
                truth_col="exact_winner",
                cogent_ms_col="cogent_avg_ms",
                ttgt_ms_col="ttgt_avg_ms",
                cogent_gflops_col="cogent_gflops",
                ttgt_gflops_col="ttgt_gflops",
                oracle_col="oracle_ms",
            ),
            "classification": bucketize_evaluation(
                sub_df,
                classification_pred,
                "classification",
                truth_col="exact_winner",
                cogent_ms_col="cogent_avg_ms",
                ttgt_ms_col="ttgt_avg_ms",
                cogent_gflops_col="cogent_gflops",
                ttgt_gflops_col="ttgt_gflops",
                oracle_col="oracle_ms",
            ),
        },
    }


def collect_benchmark_bucket_metrics(comparison_df: pd.DataFrame) -> dict:
    return {
        "distribution": summarize_split_distribution(
            comparison_df.rename(
                columns={
                    "actual_best_backend": "exact_winner",
                    "actual_cogent_ms": "cogent_avg_ms",
                    "actual_ttgt_ms": "ttgt_avg_ms",
                    "actual_cogent_gflops": "cogent_gflops",
                    "actual_ttgt_gflops": "ttgt_gflops",
                }
            )
        ),
        "regression": bucketize_evaluation(
            comparison_df,
            comparison_df["regression_pred_backend"].to_numpy(),
            "regression",
            truth_col="actual_best_backend",
            cogent_ms_col="actual_cogent_ms",
            ttgt_ms_col="actual_ttgt_ms",
            cogent_gflops_col="actual_cogent_gflops",
            ttgt_gflops_col="actual_ttgt_gflops",
            oracle_col="actual_best_ms",
        ),
        "classification": bucketize_evaluation(
            comparison_df,
            comparison_df["classification_pred_backend"].to_numpy(),
            "classification",
            truth_col="actual_best_backend",
            cogent_ms_col="actual_cogent_ms",
            ttgt_ms_col="actual_ttgt_ms",
            cogent_gflops_col="actual_cogent_gflops",
            ttgt_gflops_col="actual_ttgt_gflops",
            oracle_col="actual_best_ms",
        ),
    }


def build_primary_metric_summary(split_metrics: dict, benchmark_summary: dict | None) -> dict:
    summary = {
        "selection_metric": "total_time_regret_ratio",
        "validation": {
            "regression_total_time_regret_ratio": split_metrics["validation"]["regression"]["total_time_regret_ratio"],
            "classification_total_time_regret_ratio": split_metrics["validation"]["classification"]["total_time_regret_ratio"],
            "regression_total_gflops_efficiency": split_metrics["validation"]["regression"]["total_gflops_efficiency"],
            "classification_total_gflops_efficiency": split_metrics["validation"]["classification"]["total_gflops_efficiency"],
        },
        "test": {
            "regression_total_time_regret_ratio": split_metrics["test"]["regression"]["total_time_regret_ratio"],
            "classification_total_time_regret_ratio": split_metrics["test"]["classification"]["total_time_regret_ratio"],
            "regression_total_gflops_efficiency": split_metrics["test"]["regression"]["total_gflops_efficiency"],
            "classification_total_gflops_efficiency": split_metrics["test"]["classification"]["total_gflops_efficiency"],
        },
    }
    if benchmark_summary and "regression" in benchmark_summary and "classification" in benchmark_summary:
        summary["tccg_benchmark"] = {
            "regression_total_gflops_efficiency": benchmark_summary["regression"]["total_gflops_efficiency"],
            "classification_total_gflops_efficiency": benchmark_summary["classification"]["total_gflops_efficiency"],
            "regression_agreement_with_tcont_best": benchmark_summary["regression"]["agreement_with_tcont_best"],
            "classification_agreement_with_tcont_best": benchmark_summary["classification"]["agreement_with_tcont_best"],
        }
    return summary


def run_group_cross_validation(df: pd.DataFrame, feature_df: pd.DataFrame, seed: int, n_splits: int) -> dict:
    if n_splits <= 1:
        return {"status": "SKIPPED", "reason": "group_cv_folds <= 1"}

    y = df["exact_winner"].to_numpy()
    groups = df["equation_group"].to_numpy()
    indices = np.arange(len(df))

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iterator = splitter.split(indices, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        fold_iterator = splitter.split(indices, y, groups)

    fold_reports = []
    for fold_id, (train_idx, test_idx) in enumerate(fold_iterator, start=1):
        train_groups = sorted(df.loc[train_idx, "equation_group"].unique().tolist())
        test_groups = sorted(df.loc[test_idx, "equation_group"].unique().tolist())
        split = DatasetSplit(train_groups=train_groups, val_groups=test_groups, test_groups=test_groups)

        regression_artifacts, _, regression_metrics = train_regression_models(feature_df, df, split, seed + fold_id)
        classification_artifacts, _, classification_metrics = train_classification_model(
            feature_df, df, split, seed + fold_id
        )
        fold_reports.append(
            {
                "fold": fold_id,
                "test_groups": test_groups,
                "regression": regression_metrics,
                "classification": classification_metrics,
            }
        )

    def aggregate(strategy: str, metric: str) -> dict:
        values = [fold[strategy][metric] for fold in fold_reports]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    metrics_to_track = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mean_time_regret_ratio",
        "total_time_regret_ratio",
        "mean_gflops_efficiency",
        "total_gflops_efficiency",
    ]
    aggregate_report = {
        strategy: {metric: aggregate(strategy, metric) for metric in metrics_to_track}
        for strategy in ("regression", "classification")
    }
    return {
        "status": "OK",
        "folds": n_splits,
        "per_fold": fold_reports,
        "aggregate": aggregate_report,
    }


def main():
    args = parse_args()
    ensure_fractions(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_training_dataframe(args.data_root)
    feature_df = build_numeric_feature_frame(df)
    feature_columns = choose_feature_columns(feature_df)
    feature_inventory_df = build_feature_inventory(feature_columns)
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
    X_test = feature_df.loc[test_idx, feature_columns].copy()
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
            "name": "common_deployable_features_only",
            "uses_bench_rand_problem_cpp": False,
            "uses_backend_plan_features": False,
        },
        "feature_summary": {
            "feature_count": int(len(feature_columns)),
            "feature_inventory_file": str(feature_inventory_path),
            "feature_group_counts": feature_inventory_df["feature_group"].value_counts().to_dict(),
            "top_classification_feature_importance": summarize_top_feature_importance(
                classification_importance_df, feature_inventory_df
            ),
            "top_regression_selector_feature_importance": summarize_top_feature_importance(
                regression_importance_df, feature_inventory_df
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
    save_pickle(args.output_dir / "feature_columns.pkl", list(feature_df.columns))
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
            benchmark_feature_df = build_numeric_feature_frame(benchmark_df)
            comparison_df, benchmark_summary = evaluate_on_tccg_benchmark(
                benchmark_df,
                args.benchmark_csv,
                full_regression_artifacts,
                full_classification_artifacts,
                benchmark_feature_df,
            )
            comparison_df.to_csv(args.output_dir / "tccg_benchmark_comparison.csv", index=False)
            comparison_df.to_csv(args.output_dir / "tccg_benchmark_detailed_results.csv", index=False)
            benchmark_bucketed = collect_benchmark_bucket_metrics(comparison_df)
            report["tccg_benchmark"] = benchmark_summary
            report["tccg_benchmark"]["detailed_results_file"] = str(
                args.output_dir / "tccg_benchmark_detailed_results.csv"
            )
            report["tccg_benchmark"]["bucketed_performance"] = benchmark_bucketed
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
