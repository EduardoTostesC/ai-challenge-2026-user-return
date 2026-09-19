#!/usr/bin/env python3
"""
AI Challenge 2026 — "I'll Come Back Later"
Final/reconstructed V5-style training pipeline for user-retention prediction.

The pipeline mirrors the final strategy used in the project:
- raw and engineered feature sets;
- CatBoost (raw + engineered);
- ExtraTrees;
- LightGBM;
- XGBoost;
- cross-validated probability averaging;
- rank-based ensemble for ROC-AUC;
- optional 15% blend with a previous V4 submission.

IMPORTANT
---------
The exact competition leaderboard CSV is preserved separately in
submissions/submission_retention_v5.csv. This script is a reproducible
implementation of the final methodology. Small differences can occur across
library versions/platforms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


TARGET = "retention"
ID_COL = "id"
SEED = 2026
N_SPLITS = 5

BASE_FEATURES = [
    "sessions_count",
    "avg_session_time",
    "days_since_last_activity",
    "purchases_count",
    "avg_purchase_value",
    "active_days",
    "session_std",
    "is_weekend_user",
]


def find_file(data_dir: Path, candidates: Iterable[str]) -> Path:
    """Find the first existing file among candidate names."""
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of {list(candidates)} inside {data_dir.resolve()}"
    )


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test/sample files, supporting names with or without .csv."""
    train_path = find_file(data_dir, ["train.csv", "train"])
    test_path = find_file(data_dir, ["test.csv", "test"])
    sample_path = find_file(
        data_dir,
        ["sample_submission.csv", "sample_submission"],
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    expected_train = {ID_COL, TARGET, *BASE_FEATURES}
    expected_test = {ID_COL, *BASE_FEATURES}

    missing_train = expected_train.difference(train.columns)
    missing_test = expected_test.difference(test.columns)

    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test)}")

    return train, test, sample


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the behavioral features used in the final ensemble."""
    x = df.copy()
    eps = 1e-6

    x["sessions_per_active"] = x["sessions_count"] / (x["active_days"] + 1.0)
    x["purchases_per_session"] = x["purchases_count"] / (x["sessions_count"] + 1.0)
    x["purchases_per_active"] = x["purchases_count"] / (x["active_days"] + 1.0)
    x["recency_per_active"] = x["days_since_last_activity"] / (x["active_days"] + 1.0)
    x["session_cv"] = x["session_std"] / (x["avg_session_time"].abs() + eps)
    x["total_session_time"] = x["sessions_count"] * x["avg_session_time"]
    x["purchase_total"] = x["purchases_count"] * x["avg_purchase_value"]
    x["activity_balance"] = x["active_days"] / (
        x["active_days"] + x["days_since_last_activity"] + 1.0
    )
    x["value_per_time"] = x["purchase_total"] / (x["total_session_time"].abs() + 1.0)

    # Mild nonlinear transformations that work well with tabular models.
    for col in [
        "sessions_count",
        "days_since_last_activity",
        "purchases_count",
        "avg_purchase_value",
        "active_days",
        "session_std",
        "total_session_time",
        "purchase_total",
    ]:
        values = x[col].clip(lower=0)
        x[f"log1p_{col}"] = np.log1p(values)

    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to evenly spaced ranks in (0, 1)."""
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / (len(values) + 1.0)


def make_models(seed: int) -> Dict[str, object]:
    """Build the five model families used by the final ensemble."""
    return {
        "cat_raw": CatBoostClassifier(
            iterations=800,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=6.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        "cat_eng": CatBoostClassifier(
            iterations=950,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=7.5,
            random_seed=seed + 17,
            verbose=False,
            allow_writing_files=False,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=3,
            max_features=0.85,
            bootstrap=False,
            n_jobs=-1,
            random_state=seed + 31,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=23,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.35,
            reg_lambda=3.0,
            random_state=seed + 43,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.025,
            min_child_weight=4.0,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.25,
            reg_lambda=4.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed + 59,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def fit_cv_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Train all model families with shared stratified folds."""
    y = train[TARGET].astype(int).to_numpy()

    raw_cols = BASE_FEATURES
    train_eng = add_features(train[BASE_FEATURES])
    test_eng = add_features(test[BASE_FEATURES])

    X_raw = train[raw_cols].copy()
    X_test_raw = test[raw_cols].copy()
    X_eng = train_eng.copy()
    X_test_eng = test_eng.copy()

    model_names = ["cat_raw", "cat_eng", "extra_trees", "lightgbm", "xgboost"]
    oof = {name: np.zeros(len(train), dtype=float) for name in model_names}
    test_pred = {name: np.zeros(len(test), dtype=float) for name in model_names}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        models = make_models(seed + fold * 100)

        for name, model in models.items():
            if name == "cat_raw":
                X_tr, X_va, X_te = X_raw.iloc[tr_idx], X_raw.iloc[va_idx], X_test_raw
            else:
                X_tr, X_va, X_te = X_eng.iloc[tr_idx], X_eng.iloc[va_idx], X_test_eng

            model.fit(X_tr, y[tr_idx])
            oof[name][va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred[name] += model.predict_proba(X_te)[:, 1] / n_splits

            fold_auc = roc_auc_score(y[va_idx], oof[name][va_idx])
            print(f"{name:12s} fold AUC: {fold_auc:.6f}")

    aucs = {name: roc_auc_score(y, pred) for name, pred in oof.items()}
    print("\nOOF model AUCs:")
    for name, score in aucs.items():
        print(f"  {name:12s}: {score:.6f}")

    # V5-style rank ensemble. Weights emphasize CatBoost while retaining model diversity.
    weights = {
        "cat_raw": 0.24,
        "cat_eng": 0.28,
        "extra_trees": 0.18,
        "lightgbm": 0.16,
        "xgboost": 0.14,
    }

    oof_blend = np.zeros(len(train), dtype=float)
    test_blend = np.zeros(len(test), dtype=float)

    for name, weight in weights.items():
        oof_blend += weight * rank_normalize(oof[name])
        test_blend += weight * rank_normalize(test_pred[name])

    # Final rank normalization is useful for ROC-AUC and reproduces the competition-file style.
    oof_final = rank_normalize(oof_blend)
    test_final = rank_normalize(test_blend)

    ensemble_auc = roc_auc_score(y, oof_final)
    aucs["rank_ensemble"] = ensemble_auc
    print(f"\nRank ensemble OOF AUC: {ensemble_auc:.6f}")

    return oof_final, test_final, aucs


def blend_with_v4(
    test_pred: np.ndarray,
    test_ids: pd.Series,
    v4_path: Path,
    new_weight: float = 0.85,
) -> np.ndarray:
    """Optional final V5 step: 85% new ensemble + 15% V4, then rank normalize."""
    v4 = pd.read_csv(v4_path)
    if not {ID_COL, TARGET}.issubset(v4.columns):
        raise ValueError(f"{v4_path} must contain columns {ID_COL!r} and {TARGET!r}.")

    aligned = pd.DataFrame({ID_COL: test_ids.astype(int)}).merge(
        v4[[ID_COL, TARGET]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if aligned[TARGET].isna().any():
        raise ValueError("V4 submission could not be aligned to all test IDs.")

    v4_rank = rank_normalize(aligned[TARGET].to_numpy())
    mixed = new_weight * rank_normalize(test_pred) + (1.0 - new_weight) * v4_rank
    return rank_normalize(mixed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train/test/sample_submission files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_retention_v5_reproduced.csv"),
        help="Output submission path.",
    )
    parser.add_argument(
        "--v4-path",
        type=Path,
        default=None,
        help="Optional V4 submission used for the final 85/15 V5 blend.",
    )
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train, test, sample = load_data(args.data_dir)
    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    print(f"Positive rate: {train[TARGET].mean():.4f}")

    _, test_pred, _ = fit_cv_ensemble(
        train=train,
        test=test,
        n_splits=args.folds,
        seed=args.seed,
    )

    if args.v4_path is not None:
        print(f"\nApplying 85/15 blend with: {args.v4_path}")
        test_pred = blend_with_v4(test_pred, test[ID_COL], args.v4_path)

    # Preserve sample/test order.
    submission = sample.copy()
    if ID_COL in submission.columns:
        pred_map = dict(zip(test[ID_COL], test_pred))
        submission[TARGET] = submission[ID_COL].map(pred_map)
        if submission[TARGET].isna().any():
            raise ValueError("Could not map predictions to every sample_submission ID.")
    else:
        submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COL, TARGET]].to_csv(args.output, index=False)
    print(f"\nSaved: {args.output.resolve()}")
    print(submission[[ID_COL, TARGET]].head())


if __name__ == "__main__":
    main()
