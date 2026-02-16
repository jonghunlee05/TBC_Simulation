import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "02_mesh"))
sys.path.append(str(repo_root / "03_solver"))

from make_mesh import build_mesh
from thermoelastic_solver import build_case_context, solve_delta_t
from extract_features import extract_features
from utils import (
    ensure_dir,
    linear_scaling_check,
    plot_metrics_triplet,
    run_single_case,
    save_csv,
    scale_materials_alpha,
    load_materials,
)


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")
    delta_t = 900.0

    out_dir = os.path.join("05_results", "01_verification", "cte_scaling")
    ensure_dir(out_dir)

    scale_factors = [0.7, 0.9, 1.0, 1.1, 1.3]

    mesh_path = os.path.join(out_dir, "tbc_2d.mesh")
    build_mesh(geometry, mesh_path, nx=200, dy_scale=1.0)

    mats_base = load_materials(materials)
    alpha_sub = float(mats_base["materials"]["substrate"]["alpha_1K"])
    alpha_ysz_base = float(mats_base["materials"]["YSZ"]["alpha_1K"])

    delta_alpha_vals = []
    sigma_vals = []
    tau_vals = []
    sed_vals = []
    rows = []
    for scale in scale_factors:
        mats_path = os.path.join(out_dir, f"materials_alphaScale_{scale:.2f}.yaml")
        scale_materials_alpha(materials, mats_path, scale)
        alpha_ysz = alpha_ysz_base * scale
        delta_alpha = abs(alpha_ysz - alpha_sub)
        max_sigma, max_tau, mean_sed = run_single_case(
            geometry,
            mats_path,
            mesh_path,
            delta_t,
            build_case_context,
            solve_delta_t,
            extract_features,
        )
        delta_alpha_vals.append(delta_alpha)
        sigma_vals.append(max_sigma)
        tau_vals.append(max_tau)
        sed_vals.append(mean_sed)
        rows.append(
            {
                "alpha_scale_ysz": scale,
                "delta_alpha": delta_alpha,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "mean_sed": mean_sed,
            }
        )

    save_csv(rows, os.path.join(out_dir, "cte_scaling_results.csv"))
    plot_metrics_triplet(
        delta_alpha_vals,
        [(sigma_vals, "max")],
        [(tau_vals, "max")],
        [(sed_vals, "mean")],
        "delta_alpha (1/K)",
        "CTE Scaling",
        os.path.join(out_dir, "cte_scaling_plot.png"),
    )

    monotonic = np.all(np.diff(sigma_vals) >= 0.0)
    linear_ok, r2 = linear_scaling_check(
        np.array(delta_alpha_vals), np.array(sigma_vals)
    )
    print("CTE scaling summary:")
    print(f"- Monotonic increase: {bool(monotonic)}")
    print(f"- Linear scaling (R2>=0.9): {linear_ok} (R2={r2})")


if __name__ == "__main__":
    main()
