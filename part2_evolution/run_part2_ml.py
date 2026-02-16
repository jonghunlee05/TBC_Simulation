import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["R_final", "R_max", "slope_early", "slope_late", "n_star"]


def _select_inputs(df):
    candidate_inputs = [
        "delta_t",
        "k_p_m2_s",
        "initial_tgo_thickness_um",
        "cycle_dt_s",
        "t_min_C",
        "t_max_C",
        "n_cycles",
        "roughness_amplitude_um",
        "roughness_wavelength_um",
        "growth_eigenstrain_coeff",
    ]
    return [c for c in candidate_inputs if c in df.columns]


def _model_importance(model, X, y, random_state):
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_)
    else:
        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=random_state
        )
        importances = perm.importances_mean
    return importances


def _save_feature_importance(model, X, y, out_csv, random_state):
    importances = _model_importance(model, X, y, random_state)
    df = pd.DataFrame({"feature": X.columns, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(out_csv, index=False)
    return df


def _save_ridge_coeffs(model, X, out_csv):
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    else:
        estimator = model
    if not hasattr(estimator, "coef_"):
        return None
    df = pd.DataFrame({"feature": X.columns, "coef": estimator.coef_})
    df.to_csv(out_csv, index=False)
    return df


def _save_partial_dependence(model, X, features, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features=features, ax=ax
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run Part II ML on descriptors.")
    parser.add_argument(
        "--descriptors_csv",
        default=os.path.join("05_results", "part2_evolution", "part2_descriptors.csv"),
        help="Path to part2_descriptors.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("05_results", "part2_evolution", "ml"),
        help="Directory for ML outputs",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.descriptors_csv, comment="#")
    inputs = _select_inputs(df)
    if not inputs:
        raise ValueError("No input columns found for ML analysis.")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_lines = []
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

        target_dir = Path(args.output_dir) / target
        target_dir.mkdir(parents=True, exist_ok=True)

        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        summary_lines.append(f"{target}: RF R2={r2:.3f}, MAE={mae:.3e}")

        rf_importance = _save_feature_importance(
            rf,
            X,
            y,
            target_dir / "rf_importance.csv",
            args.random_state,
        )
        _save_feature_importance(
            ridge,
            X,
            y,
            target_dir / "ridge_importance.csv",
            args.random_state,
        )
        _save_ridge_coeffs(ridge, X, target_dir / "ridge_coeffs.csv")

        top_features = rf_importance["feature"].tolist()[:2]
        for feat in top_features:
            _save_partial_dependence(
                rf,
                X,
                [feat],
                target_dir / f"pdp_{feat}.png",
            )

    with open(Path(args.output_dir) / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


if __name__ == "__main__":
    main()
