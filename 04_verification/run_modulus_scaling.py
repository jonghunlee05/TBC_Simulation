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
    plot_metrics,
    run_single_case,
    save_csv,
    scale_materials_E,
)


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")
    delta_t = 900.0

    out_dir = os.path.join("05_results", "01_verification", "modulus_scaling")
    ensure_dir(out_dir)

    scale_factors = [0.5, 1.0, 1.5, 2.0]

    mesh_path = os.path.join(out_dir, "tbc_2d.mesh")
    build_mesh(geometry, mesh_path, nx=200, dy_scale=1.0)

    sigma_vals = []
    rows = []
    for scale in scale_factors:
        mats_path = os.path.join(out_dir, f"materials_Escale_{scale:.2f}.yaml")
        scale_materials_E(materials, mats_path, scale)
        max_sigma, max_tau, mean_sed = run_single_case(
            geometry,
            mats_path,
            mesh_path,
            delta_t,
            build_case_context,
            solve_delta_t,
            extract_features,
        )
        sigma_vals.append(max_sigma)
        rows.append(
            {
                "E_scale": scale,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "mean_sed": mean_sed,
            }
        )

    save_csv(rows, os.path.join(out_dir, "modulus_scaling_results.csv"))
    plot_metrics(
        scale_factors,
        [sigma_vals],
        ["max_sigma_yy"],
        "E_scale",
        "max_sigma_yy",
        "Modulus Scaling",
        os.path.join(out_dir, "modulus_scaling_plot.png"),
    )

    monotonic = np.all(np.diff(sigma_vals) >= 0.0)
    linear_ok, r2 = linear_scaling_check(np.array(scale_factors), np.array(sigma_vals))
    print("Modulus scaling summary:")
    print(f"- Monotonic increase: {bool(monotonic)}")
    print(f"- Linear scaling (R2>=0.9): {linear_ok} (R2={r2})")


if __name__ == "__main__":
    main()
