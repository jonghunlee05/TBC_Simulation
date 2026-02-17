import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIG = {
    # TODO: Replace with FE reference values if available.
    "E_ref_Pa": 200e9,  # Placeholder: 200 GPa
    "nu_ref": 0.23,  # Placeholder Poisson's ratio
    "delta_alpha_ref_1K": 1.0e-5,  # Placeholder CTE mismatch (1/K)
}


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _summary_stats(series):
    return {
        "mean": float(np.nanmean(series)),
        "std": float(np.nanstd(series)),
        "min": float(np.nanmin(series)),
        "max": float(np.nanmax(series)),
    }


def _save_hist(data, title, xlabel, out_path, bins=30):
    plt.figure()
    plt.hist(data, bins=bins, edgecolor="k")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_scatter(x, y, title, xlabel, ylabel, out_path, line_y_eq_x=False):
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.7)
    if line_y_eq_x:
        lims = [
            np.nanmin([np.nanmin(x), np.nanmin(y)]),
            np.nanmax([np.nanmax(x), np.nanmax(y)]),
        ]
        plt.plot(lims, lims, "--", color="k", linewidth=1)
        plt.xlim(lims)
        plt.ylim(lims)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    data_path = os.path.join(
        "05_results",
        "02_part_I_static_ranking",
        "02_lhs_dataset",
        "partI_lhs_dataset_v2.csv",
    )
    out_dir = Path("04_verification") / "partI_verification_math"
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    df = pd.read_csv(data_path)

    E_ref = float(CONFIG["E_ref_Pa"])
    nu_ref = float(CONFIG["nu_ref"])
    delta_alpha_ref = float(CONFIG["delta_alpha_ref_1K"])

    df["E_eff"] = df["e_scale"] * E_ref
    df["delta_alpha_eff"] = df["alpha_scale"] * delta_alpha_ref
    df["stress_scale"] = (df["E_eff"] / (1.0 - nu_ref)) * df["delta_alpha_eff"] * df["delta_t"]

    df["Pi_sigma_ysz"] = df["ysz_tgo_max_sigma_yy"] / df["stress_scale"]
    df["Pi_sigma_tgo"] = df["tgo_bc_max_sigma_yy"] / df["stress_scale"]

    df["u_proxy_ysz"] = (df["ysz_tgo_max_sigma_yy"] ** 2) / (2.0 * df["E_eff"])
    df["u_proxy_tgo"] = (df["tgo_bc_max_sigma_yy"] ** 2) / (2.0 * df["E_eff"])
    df["Pi_u_ysz"] = df["ysz_tgo_mean_sed"] / df["u_proxy_ysz"]
    df["Pi_u_tgo"] = df["tgo_bc_mean_sed"] / df["u_proxy_tgo"]

    df["psi_proxy_ysz"] = np.arctan(
        df["ysz_tgo_max_tau_xy"] / df["ysz_tgo_max_sigma_yy"]
    )
    df["psi_proxy_tgo"] = np.arctan(
        df["tgo_bc_max_tau_xy"] / df["tgo_bc_max_sigma_yy"]
    )

    _save_scatter(
        df["stress_scale"],
        df["ysz_tgo_max_sigma_yy"],
        "YSZ/TGO sigma_max vs stress_scale",
        "stress_scale (Pa)",
        "sigma_max (Pa)",
        fig_dir / "scatter_sigma_ysz_vs_scale.png",
        line_y_eq_x=True,
    )
    _save_scatter(
        df["stress_scale"],
        df["tgo_bc_max_sigma_yy"],
        "TGO/BC sigma_max vs stress_scale",
        "stress_scale (Pa)",
        "sigma_max (Pa)",
        fig_dir / "scatter_sigma_tgo_vs_scale.png",
        line_y_eq_x=True,
    )
    _save_hist(
        df["Pi_sigma_ysz"],
        "Pi_sigma (YSZ/TGO)",
        "Pi_sigma_ysz",
        fig_dir / "hist_pi_sigma_ysz.png",
    )
    _save_hist(
        df["Pi_sigma_tgo"],
        "Pi_sigma (TGO/BC)",
        "Pi_sigma_tgo",
        fig_dir / "hist_pi_sigma_tgo.png",
    )

    _save_scatter(
        df["u_proxy_ysz"],
        df["ysz_tgo_mean_sed"],
        "YSZ/TGO mean SED vs u_proxy",
        "u_proxy (J/m^3)",
        "mean SED (J/m^3)",
        fig_dir / "scatter_sed_ysz_vs_proxy.png",
        line_y_eq_x=True,
    )
    _save_scatter(
        df["u_proxy_tgo"],
        df["tgo_bc_mean_sed"],
        "TGO/BC mean SED vs u_proxy",
        "u_proxy (J/m^3)",
        "mean SED (J/m^3)",
        fig_dir / "scatter_sed_tgo_vs_proxy.png",
        line_y_eq_x=True,
    )
    _save_hist(
        df["Pi_u_ysz"],
        "Pi_u (YSZ/TGO)",
        "Pi_u_ysz",
        fig_dir / "hist_pi_u_ysz.png",
    )
    _save_hist(
        df["Pi_u_tgo"],
        "Pi_u (TGO/BC)",
        "Pi_u_tgo",
        fig_dir / "hist_pi_u_tgo.png",
    )

    _save_hist(
        df["psi_proxy_ysz"],
        "psi_proxy (YSZ/TGO)",
        "psi_proxy_ysz (rad)",
        fig_dir / "hist_psi_ysz.png",
    )
    _save_hist(
        df["psi_proxy_tgo"],
        "psi_proxy (TGO/BC)",
        "psi_proxy_tgo (rad)",
        fig_dir / "hist_psi_tgo.png",
    )

    _save_scatter(
        df["delta_t"],
        df["Pi_sigma_ysz"],
        "Pi_sigma_ysz vs delta_t",
        "delta_t (C)",
        "Pi_sigma_ysz",
        fig_dir / "pi_sigma_ysz_vs_delta_t.png",
    )
    _save_scatter(
        df["e_scale"],
        df["Pi_sigma_ysz"],
        "Pi_sigma_ysz vs e_scale",
        "e_scale (-)",
        "Pi_sigma_ysz",
        fig_dir / "pi_sigma_ysz_vs_e_scale.png",
    )
    _save_scatter(
        df["alpha_scale"],
        df["Pi_sigma_ysz"],
        "Pi_sigma_ysz vs alpha_scale",
        "alpha_scale (-)",
        "Pi_sigma_ysz",
        fig_dir / "pi_sigma_ysz_vs_alpha_scale.png",
    )

    summary = {
        "ysz_pi_sigma": _summary_stats(df["Pi_sigma_ysz"]),
        "tgo_pi_sigma": _summary_stats(df["Pi_sigma_tgo"]),
        "ysz_pi_u": _summary_stats(df["Pi_u_ysz"]),
        "tgo_pi_u": _summary_stats(df["Pi_u_tgo"]),
        "corr_sigma_ysz": float(df[["ysz_tgo_max_sigma_yy", "stress_scale"]].corr().iloc[0, 1]),
        "corr_sigma_tgo": float(df[["tgo_bc_max_sigma_yy", "stress_scale"]].corr().iloc[0, 1]),
        "corr_sed_ysz": float(df[["ysz_tgo_mean_sed", "u_proxy_ysz"]].corr().iloc[0, 1]),
        "corr_sed_tgo": float(df[["tgo_bc_mean_sed", "u_proxy_tgo"]].corr().iloc[0, 1]),
    }

    rows = []
    for key, stats in summary.items():
        if isinstance(stats, dict):
            row = {"metric": key}
            row.update(stats)
            rows.append(row)
        else:
            rows.append({"metric": key, "mean": stats, "std": None, "min": None, "max": None})
    pd.DataFrame(rows).to_csv(out_dir / "verification_summary.csv", index=False)


if __name__ == "__main__":
    main()
