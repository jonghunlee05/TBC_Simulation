import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import IMPORTANCE_DIR, LHS_DIR, ML_DIR, ensure_output_tree


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
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)

    for target in TARGETS:
        if target not in df.columns:
            continue
        subset = df.dropna(subset=inputs + [target]).reset_index(drop=True)
        if subset.empty:
            continue

        X = subset[inputs]
        y = subset[target]
        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=args.random_state)),
            ]
        )
        rf = RandomForestRegressor(
            n_estimators=300, random_state=args.random_state, n_jobs=-1
        )

        for name, model in [("ridge", ridge), ("rf", rf)]:
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            mae_scores = cross_val_score(
                model, X, y, cv=cv, scoring="neg_mean_absolute_error"
            )
            perf_rows.append(
                {
                    "target": target,
                    "model": name,
                    "r2_mean": float(np.mean(r2_scores)),
                    "r2_std": float(np.std(r2_scores)),
                    "mae_mean": float(-np.mean(mae_scores)),
                    "mae_std": float(np.std(mae_scores)),
                }
            )

        ridge.fit(X, y)
        rf.fit(X, y)
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

        # Save the better-performing model for PDPs.
        r2_ridge = perf_rows[-2]["r2_mean"]
        r2_rf = perf_rows[-1]["r2_mean"]
        best_model = rf if r2_rf >= r2_ridge else ridge
        model_tag = "rf" if best_model is rf else "ridge"
        model_out = Path(ML_DIR) / f"{target}_{model_tag}_model.joblib"
        model_out.parent.mkdir(parents=True, exist_ok=True)
        best_model.fit(X, y)
        joblib.dump(best_model, model_out)

    pd.DataFrame(perf_rows).to_csv(
        Path(ML_DIR) / "model_performance.csv", index=False
    )


if __name__ == "__main__":
    main()
