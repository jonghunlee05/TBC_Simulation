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
    plot_metrics,
    run_single_case,
    save_csv,
    update_tgo_thickness,
)


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")
    delta_t = 900.0

    out_dir = os.path.join("05_results", "01_verification", "thickness_sanity")
    ensure_dir(out_dir)

    tgo_values = [0.5, 1.0, 2.0, 4.0, 6.0]

    sigma_vals = []
    rows = []
    for tgo_th in tgo_values:
        geom_path = os.path.join(out_dir, f"geometry_tgo_{tgo_th:.2f}.yaml")
        update_tgo_thickness(geometry, geom_path, tgo_th)
        mesh_path = os.path.join(out_dir, f"tbc_2d_tgo_{tgo_th:.2f}.mesh")
        build_mesh(geom_path, mesh_path, nx=200, dy_scale=1.0)

        max_sigma, max_tau, mean_sed = run_single_case(
            geom_path,
            materials,
            mesh_path,
            delta_t,
            build_case_context,
            solve_delta_t,
            extract_features,
        )
        sigma_vals.append(max_sigma)
        rows.append(
            {
                "tgo_thickness_um": tgo_th,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "mean_sed": mean_sed,
            }
        )

    save_csv(rows, os.path.join(out_dir, "thickness_sanity_results.csv"))
    plot_metrics(
        tgo_values,
        [sigma_vals],
        ["max_sigma_yy"],
        "tgo_thickness_um",
        "max_sigma_yy",
        "TGO Thickness Sanity",
        os.path.join(out_dir, "thickness_sanity_plot.png"),
    )

    monotonic = np.all(np.diff(sigma_vals) >= 0.0)
    print("TGO thickness sanity summary:")
    print(f"- Monotonic increase: {bool(monotonic)}")
    print("- Expected behavior: thicker TGO generally increases mismatch stresses.")


if __name__ == "__main__":
    main()
