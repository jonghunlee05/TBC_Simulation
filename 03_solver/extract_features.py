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


def _interface_mask(ys, y_interface, dy):
    tol = 0.5 * dy
    return np.abs(ys - y_interface) <= tol


def _safe_nanmax(values, mask):
    if not np.any(mask):
        return np.nan
    return np.nanmax(values[mask])


def _safe_nanmean(values, mask):
    if not np.any(mask):
        return np.nan
    return np.nanmean(values[mask])


def extract_features(pb, regions, materials, y2, y3, output_csv, delta_t=None):
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
    sed = 0.5 * (stress[:, 0] * strain[:, 0] + stress[:, 1] * strain[:, 1] + 2.0 * stress[:, 2] * strain[:, 2])

    centroids = pb.domain.get_centroids(dim=2)
    ys = centroids[:, 1]
    unique_y = np.unique(ys)
    dy = np.median(np.diff(unique_y)) if unique_y.size > 1 else 1.0

    mask_y2 = _interface_mask(ys, y2, dy)
    mask_y3 = _interface_mask(ys, y3, dy)

    features = {
        "delta_t": delta_t,
        "ysz_tgo_max_sigma_yy": _safe_nanmax(sigma_yy, mask_y3),
        "ysz_tgo_max_tau_xy": _safe_nanmax(np.abs(tau_xy), mask_y3),
        "ysz_tgo_mean_sed": _safe_nanmean(sed, mask_y3),
        "tgo_bc_max_sigma_yy": _safe_nanmax(sigma_yy, mask_y2),
        "tgo_bc_max_tau_xy": _safe_nanmax(np.abs(tau_xy), mask_y2),
        "tgo_bc_mean_sed": _safe_nanmean(sed, mask_y2),
    }

    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(features)
