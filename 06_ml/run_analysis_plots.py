import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _risk_from_sed(df):
    """Risk proxy: maximum mean SED at critical interfaces."""
    return df[["ysz_tgo_mean_sed", "tgo_bc_mean_sed"]].max(axis=1)


def _plot_mesh_convergence(mesh_csv, out_dir):
    df = pd.read_csv(mesh_csv)
    df["risk"] = _risk_from_sed(df)
    plt.figure()
    plt.plot(df["nx"], df["risk"], marker="o")
    plt.xlabel("nx")
    plt.ylabel("Risk (max mean SED)")
    plt.title("Mesh Convergence (Risk vs Mesh Refinement)")
    plt.savefig(Path(out_dir) / "mesh_convergence_risk.png", dpi=200)
    plt.close()


def _plot_oat(csv_path, x_col, out_path, xlabel):
    df = pd.read_csv(csv_path)
    df["risk"] = _risk_from_sed(df)
    plt.figure()
    plt.plot(df[x_col], df["risk"], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Risk (max mean SED)")
    plt.title(f"Risk vs {xlabel}")
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_interaction_heatmap(csv_path, out_path, n_bins=20):
    df = pd.read_csv(csv_path, comment="#")
    df["risk"] = _risk_from_sed(df)
    x = df["delta_t"]
    y = df["tgo_thickness_um"]
    z = df["risk"]

    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)

    heat = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (x >= x_bins[i])
                & (x < x_bins[i + 1])
                & (y >= y_bins[j])
                & (y < y_bins[j + 1])
            )
            if mask.any():
                heat[j, i] = z[mask].mean()

    plt.figure()
    plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
    )
    plt.colorbar(label="Risk (max mean SED)")
    plt.xlabel("delta_t")
    plt.ylabel("tgo_thickness_um")
    plt.title("Risk Interaction Heatmap (ΔT × TGO)")
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_feature_importance(importance_csv, out_path, title):
    df = pd.read_csv(importance_csv)
    plt.figure()
    plt.bar(df["feature"], df["importance"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_summary(out_path, lines):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Generate dissertation figures from sweep and ML outputs."""
    parser = argparse.ArgumentParser(description="Generate dissertation plots.")
    parser.add_argument(
        "--features_dir",
        default=os.path.join("05_outputs", "features"),
        help="Directory containing feature CSVs",
    )
    parser.add_argument(
        "--ml_dir",
        default=os.path.join("06_ml", "outputs"),
        help="Directory with ML outputs",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("06_ml", "outputs", "figures"),
        help="Directory to store figures",
    )
    parser.add_argument(
        "--lhs_csv",
        default=os.path.join("05_outputs", "features", "sweep_dataset_lhs_final.csv"),
        help="LHS sweep CSV for interaction plot",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _plot_mesh_convergence(
        Path(args.features_dir) / "mesh_convergence_baseline.csv", args.out_dir
    )
    _plot_oat(
        Path(args.features_dir) / "sweep_oat_delta_t.csv",
        "delta_t",
        Path(args.out_dir) / "risk_vs_delta_t.png",
        "delta_t",
    )
    _plot_oat(
        Path(args.features_dir) / "sweep_oat_tgo.csv",
        "tgo_thickness_um",
        Path(args.out_dir) / "risk_vs_tgo.png",
        "tgo_thickness_um",
    )
    _plot_interaction_heatmap(args.lhs_csv, Path(args.out_dir) / "risk_heatmap.png")
    _plot_feature_importance(
        Path(args.ml_dir) / "rf_importance.csv",
        Path(args.out_dir) / "feature_importance_rf.png",
        "RF Feature Importance",
    )

    summary_lines = [
        "Dominant parameters:",
        "- Use rf_importance.csv for ranked drivers.",
        "",
        "Interaction evidence:",
        "- Risk heatmap reflects joint dependence on delta_t and TGO thickness.",
        "",
        "Limitations:",
        "- Linear elasticity, no creep or cracking.",
        "- Uniform temperature per cycle state.",
        "- Risk is an elastic upper-bound proxy.",
    ]
    _write_summary(Path(args.out_dir) / "interpretation_summary.txt", summary_lines)


if __name__ == "__main__":
    main()
