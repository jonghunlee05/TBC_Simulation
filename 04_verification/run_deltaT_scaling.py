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
from utils import ensure_dir, linear_scaling_check, plot_metrics, run_single_case, save_csv


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")

    out_dir = os.path.join("05_results", "01_verification", "deltaT_scaling")
    ensure_dir(out_dir)

    dt_values = [300.0, 500.0, 700.0, 900.0, 1100.0]

    mesh_path = os.path.join(out_dir, "tbc_2d.mesh")
    build_mesh(geometry, mesh_path, nx=200, dy_scale=1.0)

    sigma_vals = []
    rows = []
    for dT in dt_values:
        max_sigma, max_tau, mean_sed = run_single_case(
            geometry,
            materials,
            mesh_path,
            dT,
            build_case_context,
            solve_delta_t,
            extract_features,
        )
        sigma_vals.append(max_sigma)
        rows.append(
            {
                "delta_t": dT,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "mean_sed": mean_sed,
            }
        )

    save_csv(rows, os.path.join(out_dir, "deltaT_scaling_results.csv"))
    plot_metrics(
        dt_values,
        [sigma_vals],
        ["max_sigma_yy"],
        "delta_t",
        "max_sigma_yy",
        "DeltaT Scaling",
        os.path.join(out_dir, "deltaT_scaling_plot.png"),
    )

    monotonic = np.all(np.diff(sigma_vals) >= 0.0)
    linear_ok, r2 = linear_scaling_check(np.array(dt_values), np.array(sigma_vals))
    print("DeltaT scaling summary:")
    print(f"- Monotonic increase: {bool(monotonic)}")
    print(f"- Linear scaling (R2>=0.9): {linear_ok} (R2={r2})")


if __name__ == "__main__":
    main()
