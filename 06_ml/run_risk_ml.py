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


RISK_OPTIONS = ("sed_max", "weighted")


def _load_feature_csvs(features_dir, include_cyclewise, include_prefixes):
    csvs = sorted(Path(features_dir).glob("*.csv"))
    frames = []
    for csv_path in csvs:
        if include_prefixes and not any(
            csv_path.name.startswith(prefix) for prefix in include_prefixes
        ):
            continue
        if not include_cyclewise and "cyclewise" in csv_path.name:
            continue
        df = pd.read_csv(csv_path, comment="#")
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {features_dir}")
    return pd.concat(frames, ignore_index=True)


def _compute_risk(df, mode):
    """Compute delamination risk proxy from interface metrics."""
    if mode == "sed_max":
        # Dissertation-locked risk metric: max mean SED at critical interfaces.
        df["risk"] = df[["ysz_tgo_mean_sed", "tgo_bc_mean_sed"]].max(axis=1)
    elif mode == "weighted":
        # Non-dissertation option: requires explicit normalization before use.
        sigma = df[["ysz_tgo_max_sigma_yy", "tgo_bc_max_sigma_yy"]].abs().max(axis=1)
        tau = df[["ysz_tgo_max_tau_xy", "tgo_bc_max_tau_xy"]].abs().max(axis=1)
        sed = df[["ysz_tgo_mean_sed", "tgo_bc_mean_sed"]].max(axis=1)
        df["risk"] = 0.5 * sed + 0.25 * sigma + 0.25 * tau
    else:
        raise ValueError(f"Unknown risk mode: {mode}")
    return df


def _select_inputs(df):
    candidate_inputs = [
        "delta_t",
        "tgo_thickness_um",
        "ysz_thickness_um",
        "bondcoat_thickness_um",
        "k_p_m2_s",
    ]
    return [c for c in candidate_inputs if c in df.columns]


def _train_models(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    linear = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )
    linear.fit(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=300, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    results = {}
    for name, model in [("ridge", linear), ("rf", rf)]:
        y_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "X_test": X_test,
            "y_test": y_test,
        }
    return results


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


def _save_partial_dependence(model, X, features, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features=features, ax=ax
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_interaction(model, X, f1, f2, out_path):
    if X[f1].nunique() < 2 or X[f2].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features=[(f1, f2)], ax=ax
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _write_summary(out_path, summary_lines):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def main():
    """Run ML analysis with a fixed, SED-based risk proxy."""
    parser = argparse.ArgumentParser(description="Risk proxy + ML analysis.")
    parser.add_argument(
        "--features_dir",
        default=os.path.join("05_outputs", "features"),
        help="Directory containing feature CSVs",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("06_ml", "outputs"),
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--risk_mode",
        choices=RISK_OPTIONS,
        default="sed_max",
        help="Risk proxy definition",
    )
    parser.add_argument(
        "--include_cyclewise",
        action="store_true",
        help="Include features_cyclewise.csv in analysis",
    )
    parser.add_argument(
        "--include_prefix",
        action="append",
        default=["sweep_dataset_lhs_final"],
        help="CSV filename prefix to include (repeatable)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = _load_feature_csvs(
        args.features_dir,
        include_cyclewise=args.include_cyclewise,
        include_prefixes=args.include_prefix,
    )
    df = _compute_risk(df, args.risk_mode)
    inputs = _select_inputs(df)

    if not inputs:
        raise ValueError("No input columns found for ML analysis.")

    df = df.dropna(subset=inputs + ["risk"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.output_dir, "dataset_with_risk.csv"), index=False)

    X = df[inputs]
    y = df["risk"]

    results = _train_models(X, y, args.random_state)

    summary_lines = []
    for name, info in results.items():
        summary_lines.append(f"{name}: R2={info['r2']:.3f}, MAE={info['mae']:.3e}")

    ridge_model = results["ridge"]["model"]
    rf_model = results["rf"]["model"]

    ridge_importance = _save_feature_importance(
        ridge_model,
        X,
        y,
        os.path.join(args.output_dir, "ridge_importance.csv"),
        args.random_state,
    )
    rf_importance = _save_feature_importance(
        rf_model,
        X,
        y,
        os.path.join(args.output_dir, "rf_importance.csv"),
        args.random_state,
    )

    summary_lines.append(
        f"Top ridge feature: {ridge_importance.iloc[0]['feature']}"
    )
    summary_lines.append(f"Top rf feature: {rf_importance.iloc[0]['feature']}")

    if "delta_t" in inputs:
        _save_partial_dependence(
            rf_model,
            X,
            ["delta_t"],
            os.path.join(args.output_dir, "pdp_delta_t.png"),
        )
    if "tgo_thickness_um" in inputs:
        _save_partial_dependence(
            rf_model,
            X,
            ["tgo_thickness_um"],
            os.path.join(args.output_dir, "pdp_tgo_thickness.png"),
        )
    if "delta_t" in inputs and "tgo_thickness_um" in inputs:
        _save_interaction(
            rf_model,
            X,
            "delta_t",
            "tgo_thickness_um",
            os.path.join(args.output_dir, "interaction_delta_t_tgo.png"),
        )

    _write_summary(os.path.join(args.output_dir, "summary.txt"), summary_lines)


if __name__ == "__main__":
    main()
