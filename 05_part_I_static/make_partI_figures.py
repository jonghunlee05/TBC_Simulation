import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from utils import (
    IMPORTANCE_DIR,
    INTERACTIONS_DIR,
    LHS_DIR,
    ML_DIR,
    OAT_DIR,
    SUMMARY_DIR,
    ensure_output_tree,
    plot_triplet,
)


TARGETS = [
    "ysz_tgo_max_sigma_yy",
    "ysz_tgo_max_tau_xy",
    "ysz_tgo_mean_sed",
    "tgo_bc_max_sigma_yy",
    "tgo_bc_max_tau_xy",
    "tgo_bc_mean_sed",
]


def _metric_columns(interface):
    return [
        f"{interface}_max_sigma_yy",
        f"{interface}_max_tau_xy",
        f"{interface}_mean_sed",
    ]


def _plot_oat(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    param = df["sweep_param"].iloc[0]
    x = df["value"].to_numpy()
    xlabel_map = {
        "deltaT": "delta_T (C)",
        "delta_t": "delta_T (C)",
        "tgo_thickness_um": "TGO thickness (um)",
        "alpha_scale": "CTE scale (-)",
        "e_scale": "E scale (-)",
        "roughness_amplitude": "roughness amplitude (um)",
        "roughness_wavelength": "roughness wavelength (um)",
    }
    xlabel = xlabel_map.get(param, param)

    for interface in ["ysz_tgo", "tgo_bc"]:
        cols = _metric_columns(interface)
        series = {
            "max_sigma_yy": [(df[cols[0]].to_numpy(), "max")],
            "max_tau_xy": [(df[cols[1]].to_numpy(), "max")],
            "mean_sed": [(df[cols[2]].to_numpy(), "mean")],
        }
        out_path = Path(out_dir) / f"oat_{param}_{interface}.png"
        plot_triplet(x, series, xlabel, f"OAT sweep: {param} ({interface})", out_path)


def _plot_importance(csv_path, out_path, title):
    df = pd.read_csv(csv_path).sort_values("importance", ascending=True)
    plt.figure(figsize=(6, 4))
    plt.barh(df["feature"], df["importance"])
    plt.title(title)
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_pdp(model_path, X, features, out_path, title):
    model = joblib.load(model_path)
    fig, axes = plt.subplots(len(features), 1, figsize=(7, 3 * len(features)))
    if len(features) == 1:
        axes = [axes]
    PartialDependenceDisplay.from_estimator(model, X, features=features, ax=axes)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make Part I figures.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(LHS_DIR, "partI_lhs_dataset.csv"),
        help="LHS dataset CSV",
    )
    args = parser.parse_args()

    ensure_output_tree()
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in Path(OAT_DIR).glob("oat_*.csv"):
        _plot_oat(csv_path, OAT_DIR)

    summary_rows = []
    df = pd.read_csv(args.dataset)
    input_cols = [
        c
        for c in df.columns
        if c
        in [
            "delta_t",
            "tgo_thickness_um",
            "ysz_thickness_um",
            "bond_thickness_um",
            "alpha_scale",
            "e_scale",
            "roughness_amplitude_um",
            "roughness_wavelength_um",
        ]
    ]

    for target in TARGETS:
        rf_path = Path(IMPORTANCE_DIR) / f"{target}_rf_importance.csv"
        ridge_path = Path(IMPORTANCE_DIR) / f"{target}_ridge_importance.csv"
        if not rf_path.exists() or not ridge_path.exists():
            continue

        _plot_importance(
            rf_path,
            Path(IMPORTANCE_DIR) / f"{target}_rf_importance.png",
            f"RF importance: {target}",
        )
        _plot_importance(
            ridge_path,
            Path(IMPORTANCE_DIR) / f"{target}_ridge_importance.png",
            f"Ridge importance: {target}",
        )

        rf_df = pd.read_csv(rf_path)
        top3 = rf_df["feature"].head(3).tolist()
        if len(top3) == 3:
            summary_rows.append(
                {"target": target, "top1": top3[0], "top2": top3[1], "top3": top3[2]}
            )

        model_candidates = sorted(Path(ML_DIR).glob(f"{target}_*_model.joblib"))
        if model_candidates and input_cols and len(top3) > 0:
            model_path = model_candidates[0]
            _plot_pdp(
                model_path,
                df[input_cols],
                top3,
                Path(INTERACTIONS_DIR) / f"{target}_pdp.png",
                f"PDP: {target}",
            )

            fig_out = Path(figures_dir) / f"partI_v2_pdp_{target}.png"
            _plot_pdp(
                model_path,
                df[input_cols],
                top3,
                fig_out,
                f"PDP: {target}",
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(SUMMARY_DIR) / "top3_drivers.csv"
    summary_df.to_csv(summary_csv, index=False)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * len(summary_df)))
        ax.axis("off")
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        fig.tight_layout()
        fig.savefig(Path(SUMMARY_DIR) / "top3_drivers.png", dpi=200)
        plt.close(fig)

    # Check PDP trends for alpha_scale and e_scale
    trend_rows = []
    for target in TARGETS:
        model_candidates = sorted(Path(ML_DIR).glob(f"{target}_*_model.joblib"))
        if not model_candidates or not input_cols:
            continue
        model_path = model_candidates[0]
        model = joblib.load(model_path)
        for feature in ("alpha_scale", "e_scale"):
            if feature not in input_cols:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            disp = PartialDependenceDisplay.from_estimator(
                model, df[input_cols], features=[feature], ax=ax
            )
            values = disp.pd_results[0]["average"].ravel()
            trend_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pdp_range": float(np.nanmax(values) - np.nanmin(values)),
                }
            )
            plt.close(fig)
    if trend_rows:
        pd.DataFrame(trend_rows).to_csv(
            Path(SUMMARY_DIR) / "pdp_trend_check.csv", index=False
        )


if __name__ == "__main__":
    main()
