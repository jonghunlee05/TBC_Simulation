import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "02_mesh"))
sys.path.append(str(repo_root / "03_solver"))

from utils import (
    compute_interface_samples,
    ensure_dir,
    load_geometry,
    plot_metrics_triplet,
    save_csv,
)
from make_mesh import build_mesh
from thermoelastic_solver import build_case_context, solve_delta_t


def main():
    geometry = os.path.join("00_inputs", "geometry_spec.yaml")
    materials = os.path.join("00_inputs", "materials.yaml")
    delta_t = 900.0

    out_dir = os.path.join("05_results", "01_verification", "mesh_convergence")
    ensure_dir(out_dir)

    _, width = load_geometry(geometry)

    levels = [
        (120, 1.4),
        (160, 1.2),
        (200, 1.0),
        (300, 0.8),
        (400, 0.6),
    ]

    rows = []
    element_sizes = []
    sigma_vals = []
    tau_vals = []
    sed_vals = []
    sigma_p95_vals = []
    tau_p95_vals = []

    for nx, dy_scale in levels:
        mesh_path = os.path.join(out_dir, f"tbc_2d_nx{nx}_dy{dy_scale}.mesh")
        build_mesh(geometry, mesh_path, nx=nx, dy_scale=dy_scale)

        context = build_case_context(geometry, materials, mesh_path)
        pb, state, out = solve_delta_t(context, delta_t, growth_strain=0.0, bc_variant="fixed")
        samples = compute_interface_samples(
            pb,
            context["regions"],
            {
                "substrate": {
                    "lam": context["props"]["substrate"]["lam"],
                    "mu": context["props"]["substrate"]["mu"],
                },
                "bondcoat": {
                    "lam": context["props"]["bondcoat"]["lam"],
                    "mu": context["props"]["bondcoat"]["mu"],
                },
                "tgo": {
                    "lam": context["props"]["tgo"]["lam"],
                    "mu": context["props"]["tgo"]["mu"],
                },
                "ysz": {
                    "lam": context["props"]["ysz"]["lam"],
                    "mu": context["props"]["ysz"]["mu"],
                },
            },
            context["y2"],
            context["y3"],
            n_select=200,
        )

        sigma_ysz = samples["ysz_tgo"]["sigma_yy"]
        sigma_tgo = samples["tgo_bc"]["sigma_yy"]
        tau_ysz = np.abs(samples["ysz_tgo"]["tau_xy"])
        tau_tgo = np.abs(samples["tgo_bc"]["tau_xy"])
        sed_ysz = samples["ysz_tgo"]["sed"]
        sed_tgo = samples["tgo_bc"]["sed"]

        max_sigma = np.nanmax([np.nanmax(sigma_ysz), np.nanmax(sigma_tgo)])
        max_tau = np.nanmax([np.nanmax(tau_ysz), np.nanmax(tau_tgo)])
        mean_sed = np.nanmax([np.nanmean(sed_ysz), np.nanmean(sed_tgo)])
        sigma_p95 = np.nanmax(
            [np.nanpercentile(sigma_ysz, 95), np.nanpercentile(sigma_tgo, 95)]
        )
        tau_p95 = np.nanmax(
            [np.nanpercentile(tau_ysz, 95), np.nanpercentile(tau_tgo, 95)]
        )

        element_size = width / float(nx)
        rows.append(
            {
                "nx": nx,
                "dy_scale": dy_scale,
                "element_size_um": element_size,
                "max_sigma_yy": max_sigma,
                "max_tau_xy": max_tau,
                "p95_sigma_yy": sigma_p95,
                "p95_tau_xy": tau_p95,
                "mean_sed": mean_sed,
            }
        )
        element_sizes.append(element_size)
        sigma_vals.append(max_sigma)
        tau_vals.append(max_tau)
        sed_vals.append(mean_sed)
        sigma_p95_vals.append(sigma_p95)
        tau_p95_vals.append(tau_p95)

    save_csv(rows, os.path.join(out_dir, "mesh_convergence_results.csv"))
    plot_metrics_triplet(
        element_sizes,
        [(sigma_vals, "max"), (sigma_p95_vals, "p95")],
        [(tau_vals, "max"), (tau_p95_vals, "p95")],
        [(sed_vals, "mean")],
        "element size (um)",
        "Mesh Convergence",
        os.path.join(out_dir, "convergence_plot.png"),
        invert_x=True,
    )

    monotonic = all(
        np.all(np.diff(vals) <= 0.0) for vals in (sigma_vals, tau_vals, sed_vals)
    )
    print("Mesh convergence summary:")
    print(f"- Monotonic decrease with refinement: {monotonic}")
    print("- Expected behavior: metrics stabilize as element size decreases.")

    def _pct_change(prev, curr):
        if prev is None or curr is None:
            return np.nan
        denom = abs(prev)
        if denom == 0:
            return np.nan
        return 100.0 * (curr - prev) / denom

    if len(sigma_vals) >= 2:
        print("Last-step percent change:")
        print(f"- max_sigma_yy: {_pct_change(sigma_vals[-2], sigma_vals[-1]):.2f}%")
        print(f"- max_tau_xy: {_pct_change(tau_vals[-2], tau_vals[-1]):.2f}%")
        print(f"- p95_tau_xy: {_pct_change(tau_p95_vals[-2], tau_p95_vals[-1]):.2f}%")
        print(f"- mean_sed: {_pct_change(sed_vals[-2], sed_vals[-1]):.2f}%")


if __name__ == "__main__":
    main()
