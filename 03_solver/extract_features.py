import csv
import os

import numpy as np


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


def extract_features(pb, regions, materials, y2, y3, output_csv, delta_t=None, n_select=200):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    mesh = pb.domain.mesh
    n_cells = mesh.n_el

    stress = np.full((n_cells, 3), np.nan, dtype=np.float64)
    strain = np.full((n_cells, 3), np.nan, dtype=np.float64)

    _evaluate_region_stress_strain(
        pb, regions["substrate"], materials["substrate"]["lam"], materials["substrate"]["mu"], stress, strain
    )
    _evaluate_region_stress_strain(
        pb, regions["bondcoat"], materials["bondcoat"]["lam"], materials["bondcoat"]["mu"], stress, strain
    )
    _evaluate_region_stress_strain(
        pb, regions["tgo"], materials["tgo"]["lam"], materials["tgo"]["mu"], stress, strain
    )
    _evaluate_region_stress_strain(
        pb, regions["ysz"], materials["ysz"]["lam"], materials["ysz"]["mu"], stress, strain
    )

    # Stress/strain components: [xx, yy, xy] in 2D
    sigma_yy = stress[:, 1]
    tau_xy = stress[:, 2]
    sed = 0.5 * (
        stress[:, 0] * strain[:, 0]
        + stress[:, 1] * strain[:, 1]
        + 2.0 * stress[:, 2] * strain[:, 2]
    )

    mesh = pb.domain.mesh
    conn = mesh.get_conn(mesh.descs[0])
    centroids = mesh.coors[conn].mean(axis=1)
    y_centroids = centroids[:, 1]

    idx_ysz_tgo = _nearest_element_indices(y_centroids, y3, n_select)
    idx_tgo_bc = _nearest_element_indices(y_centroids, y2, n_select)

    features = {
        "delta_t": delta_t,
        "ysz_tgo_max_sigma_yy": np.nanmax(sigma_yy[idx_ysz_tgo]),
        "ysz_tgo_max_tau_xy": np.nanmax(np.abs(tau_xy[idx_ysz_tgo])),
        "ysz_tgo_mean_sed": np.nanmean(sed[idx_ysz_tgo]),
        "tgo_bc_max_sigma_yy": np.nanmax(sigma_yy[idx_tgo_bc]),
        "tgo_bc_max_tau_xy": np.nanmax(np.abs(tau_xy[idx_tgo_bc])),
        "tgo_bc_mean_sed": np.nanmean(sed[idx_tgo_bc]),
    }

    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(features)
