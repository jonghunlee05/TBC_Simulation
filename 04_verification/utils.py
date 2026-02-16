import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_geometry(geometry_path):
    with open(geometry_path, "r", encoding="utf-8") as f:
        geom = yaml.safe_load(f)
    width = float(geom["domain"]["width_um"])
    return geom, width


def load_materials(materials_path):
    with open(materials_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def extract_metrics(features):
    sigma_candidates = []
    tau_candidates = []
    sed_candidates = []

    for key in ("ysz_tgo_max_sigma_yy", "tgo_bc_max_sigma_yy"):
        if key in features:
            sigma_candidates.append(features[key])
    for key in ("ysz_tgo_max_tau_xy", "tgo_bc_max_tau_xy"):
        if key in features:
            tau_candidates.append(abs(features[key]))
    for key in ("ysz_tgo_mean_sed", "tgo_bc_mean_sed"):
        if key in features:
            sed_candidates.append(features[key])

    max_sigma = max(sigma_candidates) if sigma_candidates else None
    max_tau = max(tau_candidates) if tau_candidates else None
    mean_sed = max(sed_candidates) if sed_candidates else None
    return max_sigma, max_tau, mean_sed


def _evaluate_region_stress_strain(pb, region, lam, mu, out_stress, out_strain):
    if region.cells.shape[0] == 0:
        return

    strain = pb.evaluate(
        f"ev_cauchy_strain.2.{region.name}(u)",
        mode="el_avg",
        verbose=False,
    )

    # Shapes are (n_el, 1, 3, 1) -> squeeze to (n_el, 3)
    strain = np.asarray(strain)[:, 0, :, 0]

    # Plane strain isotropic stress from strain.
    exx = strain[:, 0]
    eyy = strain[:, 1]
    exy = strain[:, 2]
    sxx = (lam + 2.0 * mu) * exx + lam * eyy
    syy = lam * exx + (lam + 2.0 * mu) * eyy
    sxy = 2.0 * mu * exy
    stress = np.column_stack([sxx, syy, sxy])

    out_stress[region.cells] = stress
    out_strain[region.cells] = strain


def _nearest_element_indices(y_centroids, y_interface, n_select):
    n_select = min(n_select, y_centroids.shape[0])
    dist = np.abs(y_centroids - y_interface)
    return np.argsort(dist)[:n_select]


def compute_interface_samples(pb, regions, materials, y2, y3, n_select=200):
    mesh = pb.domain.mesh
    n_cells = mesh.n_el

    stress = np.full((n_cells, 3), np.nan, dtype=np.float64)
    strain = np.full((n_cells, 3), np.nan, dtype=np.float64)

    _evaluate_region_stress_strain(
        pb,
        regions["substrate"],
        materials["substrate"]["lam"],
        materials["substrate"]["mu"],
        stress,
        strain,
    )
    _evaluate_region_stress_strain(
        pb,
        regions["bondcoat"],
        materials["bondcoat"]["lam"],
        materials["bondcoat"]["mu"],
        stress,
        strain,
    )
    _evaluate_region_stress_strain(
        pb,
        regions["tgo"],
        materials["tgo"]["lam"],
        materials["tgo"]["mu"],
        stress,
        strain,
    )
    _evaluate_region_stress_strain(
        pb,
        regions["ysz"],
        materials["ysz"]["lam"],
        materials["ysz"]["mu"],
        stress,
        strain,
    )

    sigma_yy = stress[:, 1]
    tau_xy = stress[:, 2]
    sed = 0.5 * (
        stress[:, 0] * strain[:, 0]
        + stress[:, 1] * strain[:, 1]
        + 2.0 * stress[:, 2] * strain[:, 2]
    )

    conn = mesh.get_conn(mesh.descs[0])
    centroids = mesh.coors[conn].mean(axis=1)
    y_centroids = centroids[:, 1]
    idx_ysz_tgo = _nearest_element_indices(y_centroids, y3, n_select)
    idx_tgo_bc = _nearest_element_indices(y_centroids, y2, n_select)

    return {
        "ysz_tgo": {
            "sigma_yy": sigma_yy[idx_ysz_tgo],
            "tau_xy": tau_xy[idx_ysz_tgo],
            "sed": sed[idx_ysz_tgo],
        },
        "tgo_bc": {
            "sigma_yy": sigma_yy[idx_tgo_bc],
            "tau_xy": tau_xy[idx_tgo_bc],
            "sed": sed[idx_tgo_bc],
        },
    }


def save_csv(rows, csv_path):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def plot_metrics_triplet(
    x,
    sigma_series,
    tau_series,
    sed_series,
    xlabel,
    title,
    out_path,
    invert_x=False,
):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 9))

    for y, label in sigma_series:
        axes[0].plot(x, y, marker="o", label=label)
    axes[0].set_ylabel("sigma_yy (Pa)")
    if sigma_series:
        axes[0].legend()

    for y, label in tau_series:
        axes[1].plot(x, y, marker="o", label=label)
    axes[1].set_ylabel("|tau_xy| (Pa)")
    if tau_series:
        axes[1].legend()

    for y, label in sed_series:
        axes[2].plot(x, y, marker="o", label=label)
    axes[2].set_ylabel("mean SED (J/m^3)")
    if sed_series:
        axes[2].legend()

    axes[2].set_xlabel(xlabel)
    fig.suptitle(title)
    if invert_x:
        for ax in axes:
            ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def is_monotonic_non_decreasing(values, tol=1e-9):
    if len(values) < 2:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs >= -tol))


def linear_scaling_check(x, y):
    if len(x) < 2:
        return False, None
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return (r2 is not None and r2 >= 0.9), r2


def run_single_case(
    geometry_path,
    materials_path,
    mesh_path,
    delta_t,
    build_case_context,
    solve_delta_t,
    extract_features,
):
    context = build_case_context(geometry_path, materials_path, mesh_path)
    pb, state, out = solve_delta_t(context, delta_t, growth_strain=0.0, bc_variant="fixed")
    features = extract_features(
        pb,
        regions=context["regions"],
        materials={
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
        y2=context["y2"],
        y3=context["y3"],
        output_csv=os.devnull,
        delta_t=delta_t,
    )
    return extract_metrics(features)


def scale_materials_E(materials_path, out_path, scale):
    mats = load_materials(materials_path)
    for mat in mats["materials"].values():
        if "E_GPa" in mat:
            mat["E_GPa"] = float(mat["E_GPa"]) * float(scale)
    write_yaml(mats, out_path)


def scale_materials_alpha(materials_path, out_path, scale_ysz):
    mats = load_materials(materials_path)
    if "YSZ" in mats["materials"]:
        mats["materials"]["YSZ"]["alpha_1K"] = (
            float(mats["materials"]["YSZ"]["alpha_1K"]) * float(scale_ysz)
        )
    write_yaml(mats, out_path)


def update_tgo_thickness(geometry_path, out_path, tgo_thickness_um):
    geom = load_geometry(geometry_path)[0]
    layers = geom["layers"]
    for layer in layers:
        if "tgo" in layer["name"].lower():
            layer["thickness_um"] = float(tgo_thickness_um)
            break
    write_yaml(geom, out_path)
