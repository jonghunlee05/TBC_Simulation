import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    IMPORTANCE_DIR,
    LHS_DIR,
    ML_DIR,
    ensure_output_tree,
)


TARGETS = [
    "ysz_tgo_max_sigma_yy",
    "ysz_tgo_max_tau_xy",
    "ysz_tgo_mean_sed",
    "tgo_bc_max_sigma_yy",
    "tgo_bc_max_tau_xy",
    "tgo_bc_mean_sed",
]


def _select_inputs(df):
    candidates = [
        "delta_t",
        "tgo_thickness_um",
        "ysz_thickness_um",
        "bond_thickness_um",
        "alpha_scale",
        "e_scale",
        "roughness_amplitude_um",
        "roughness_wavelength_um",
    ]
    return [c for c in candidates if c in df.columns]


def main():
    parser = argparse.ArgumentParser(description="Run Part I ML models.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(LHS_DIR, "partI_lhs_dataset.csv"),
        help="LHS dataset CSV",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    ensure_output_tree()
    df = pd.read_csv(args.dataset)
    inputs = _select_inputs(df)
    if not inputs:
        raise ValueError("No input columns found in dataset.")

    perf_rows = []

    for target in TARGETS:
        if target not in df.columns:
            continue
        subset = df.dropna(subset=inputs + [target]).reset_index(drop=True)
        if subset.empty:
            continue

        X = subset[inputs]
        y = subset[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state
        )

        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=args.random_state)),
            ]
        )
        ridge.fit(X_train, y_train)

        rf = RandomForestRegressor(
            n_estimators=300, random_state=args.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        for name, model in [("ridge", ridge), ("rf", rf)]:
            y_pred = model.predict(X_test)
            perf_rows.append(
                {
                    "target": target,
                    "model": name,
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                }
            )

        ridge_model = ridge.named_steps["model"]
        ridge_coeffs = pd.DataFrame(
            {"feature": inputs, "coef": ridge_model.coef_}
        )
        ridge_coeffs.to_csv(Path(ML_DIR) / f"{target}_ridge_coeffs.csv", index=False)

        rf_importance = pd.DataFrame(
            {"feature": inputs, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)
        rf_importance.to_csv(
            Path(IMPORTANCE_DIR) / f"{target}_rf_importance.csv", index=False
        )

        ridge_importance = ridge_coeffs.copy()
        ridge_importance["importance"] = ridge_importance["coef"].abs()
        ridge_importance = ridge_importance.sort_values("importance", ascending=False)
        ridge_importance.to_csv(
            Path(IMPORTANCE_DIR) / f"{target}_ridge_importance.csv", index=False
        )

        joblib.dump(rf, Path(ML_DIR) / f"{target}_rf_model.joblib")

    pd.DataFrame(perf_rows).to_csv(Path(ML_DIR) / "model_performance.csv", index=False)


if __name__ == "__main__":
    main()
