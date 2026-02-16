import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "02_mesh"))
sys.path.append(str(repo_root / "03_solver"))

from utils import ensure_dir, load_geometry, plot_metrics, run_single_case, save_csv
from make_mesh import build_mesh
from thermoelastic_solver import build_case_context, solve_delta_t
from extract_features import extract_features


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")
    delta_t = 900.0

    out_dir = os.path.join("05_results", "01_verification", "mesh_convergence")
    ensure_dir(out_dir)

    geom, width = load_geometry(geometry)

    levels = [
        (120, 1.4),
        (160, 1.2),
        (200, 1.0),
        (300, 0.8),
    ]

    rows = []
    element_sizes = []
    sigma_vals = []
    tau_vals = []
    sed_vals = []

    for nx, dy_scale in levels:
        mesh_path = os.path.join(out_dir, f"tbc_2d_nx{nx}_dy{dy_scale}.mesh")
        build_mesh(geometry, mesh_path, nx=nx, dy_scale=dy_scale)

        max_sigma, max_tau, mean_sed = run_single_case(
            geometry,
            materials,
            mesh_path,
            delta_t,
            build_case_context,
            solve_delta_t,
            extract_features,
        )

        element_size = width / float(nx)
        rows.append(
            {
                "nx": nx,
                "dy_scale": dy_scale,
                "element_size_um": element_size,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "mean_sed": mean_sed,
            }
        )
        element_sizes.append(element_size)
        sigma_vals.append(max_sigma)
        tau_vals.append(max_tau)
        sed_vals.append(mean_sed)

    save_csv(rows, os.path.join(out_dir, "mesh_convergence_results.csv"))
    plot_metrics(
        element_sizes,
        [sigma_vals, tau_vals, sed_vals],
        ["max_sigma_yy", "max_tau_xy", "mean_sed"],
        "element_size_um",
        "metric",
        "Mesh Convergence",
        os.path.join(out_dir, "convergence_plot.png"),
    )

    monotonic = all(
        np.all(np.diff(vals) <= 0.0) for vals in (sigma_vals, tau_vals, sed_vals)
    )
    print("Mesh convergence summary:")
    print(f"- Monotonic decrease with refinement: {monotonic}")
    print("- Expected behavior: metrics stabilize as element size decreases.")


if __name__ == "__main__":
    main()
