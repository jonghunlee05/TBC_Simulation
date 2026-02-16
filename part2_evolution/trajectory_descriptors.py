import argparse
import os

import numpy as np
import pandas as pd


def _risk_from_sed(df):
    if "ysz_tgo_mean_sed" in df.columns and "tgo_bc_mean_sed" in df.columns:
        return df[["ysz_tgo_mean_sed", "tgo_bc_mean_sed"]].max(axis=1)
    raise ValueError("Required mean SED columns not found in cyclewise CSV.")


def compute_descriptors(cyclewise_csv, threshold_ratio=0.8):
    df = pd.read_csv(cyclewise_csv, comment="#")
    if df.empty:
        raise ValueError(f"No rows found in {cyclewise_csv}")

    if "cycle_id" in df.columns:
        cycles = df["cycle_id"].to_numpy(dtype=float)
    else:
        cycles = np.arange(1, len(df) + 1, dtype=float)

    R = _risk_from_sed(df).to_numpy(dtype=float)
    R_max = float(np.nanmax(R))
    R_final = float(R[-1])

    n_early = min(5, len(R))
    n_late = min(5, len(R))
    slope_early = float(np.polyfit(cycles[:n_early], R[:n_early], 1)[0])
    slope_late = float(np.polyfit(cycles[-n_late:], R[-n_late:], 1)[0])

    threshold = threshold_ratio * R_max
    above = np.where(R >= threshold)[0]
    n_star = int(cycles[above[0]]) if len(above) > 0 else None

    if "tgo_thickness_um" in df.columns:
        final_tgo = float(df["tgo_thickness_um"].iloc[-1])
    else:
        final_tgo = None

    sigma_cols = [c for c in df.columns if c in ("ysz_tgo_max_sigma_yy", "tgo_bc_max_sigma_yy")]
    tau_cols = [c for c in df.columns if c in ("ysz_tgo_max_tau_xy", "tgo_bc_max_tau_xy")]
    max_sigma = float(df[sigma_cols].max(axis=1).max()) if sigma_cols else None
    max_tau = float(df[tau_cols].max(axis=1).max()) if tau_cols else None

    return {
        "R_final": R_final,
        "R_max": R_max,
        "slope_early": slope_early,
        "slope_late": slope_late,
        "n_star": n_star,
        "threshold_ratio": threshold_ratio,
        "threshold_value": threshold,
        "final_tgo_thickness": final_tgo,
        "max_sigma_yy_over_life": max_sigma,
        "max_tau_xy_over_life": max_tau,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute trajectory descriptors.")
    parser.add_argument(
        "--cyclewise_csv",
        required=True,
        help="Cyclewise CSV path for a single case",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV path for descriptors",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.8,
        help="Threshold ratio for n_star (default 0.8)",
    )
    args = parser.parse_args()

    descriptors = compute_descriptors(args.cyclewise_csv, args.threshold_ratio)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame([descriptors]).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
