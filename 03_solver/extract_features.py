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


def _interface_band_mask(y_centroids, y_interface, band):
    """
    Return a boolean mask for elements within a fixed band of an interface.
    """
    return (y_centroids >= (y_interface - band)) & (y_centroids <= (y_interface + band))


def _mask_node_count(conn, mask):
    if not np.any(mask):
        return 0
    nodes = np.unique(conn[mask].ravel())
    return int(nodes.size)


def _max_loc_um(values, centroids, mask):
    """
    Return the (x, y) centroid location (um) for the max value in a mask.
    """
    masked = np.where(mask, values, -np.inf)
    if not np.any(mask):
        return np.nan, np.nan
    idx = int(np.nanargmax(masked))
    return float(centroids[idx, 0]), float(centroids[idx, 1])


def extract_features(
    pb,
    regions,
    materials,
    y2,
    y3,
    output_csv,
    delta_t=None,
    n_select=200,
    extra_fields=None,
):
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

    total_thickness = float(mesh.coors[:, 1].max() - mesh.coors[:, 1].min())
    unique_y = np.unique(mesh.coors[:, 1])
    dy_min = float(np.min(np.diff(unique_y))) if unique_y.size > 1 else total_thickness
    band = max(2.0 * dy_min, 1.0e-6 * total_thickness)

    # Layer masks for interface restrictions.
    mask_ysz = np.zeros(n_cells, dtype=bool)
    mask_tgo = np.zeros(n_cells, dtype=bool)
    mask_bond = np.zeros(n_cells, dtype=bool)
    if "ysz" in regions:
        mask_ysz[regions["ysz"].cells] = True
    if "tgo" in regions:
        mask_tgo[regions["tgo"].cells] = True
    if "bondcoat" in regions:
        mask_bond[regions["bondcoat"].cells] = True

    band_ysz_tgo = _interface_band_mask(y_centroids, y3, band)
    band_tgo_bc = _interface_band_mask(y_centroids, y2, band)

    mask_ysz_tgo = band_ysz_tgo & (mask_ysz | mask_tgo)
    mask_tgo_bc = band_tgo_bc & (mask_tgo | mask_bond)

    n_ysz_tgo = int(np.sum(mask_ysz_tgo))
    n_tgo_bc = int(np.sum(mask_tgo_bc))
    if n_ysz_tgo <= 0 or n_tgo_bc <= 0:
        raise ValueError(
            f"Interface selection failed: n_ysz_tgo={n_ysz_tgo}, n_tgo_bc={n_tgo_bc}"
        )

    ysz_tgo_loc_x_um, ysz_tgo_loc_y_um = _max_loc_um(
        sigma_yy, centroids, mask_ysz_tgo
    )
    tgo_bc_loc_x_um, tgo_bc_loc_y_um = _max_loc_um(
        sigma_yy, centroids, mask_tgo_bc
    )

    features = {
        "delta_t": delta_t,
        "ysz_tgo_max_sigma_yy": np.nanmax(sigma_yy[mask_ysz_tgo]),
        "ysz_tgo_max_tau_xy": np.nanmax(np.abs(tau_xy[mask_ysz_tgo])),
        "ysz_tgo_mean_sed": np.nanmean(sed[mask_ysz_tgo]),
        "tgo_bc_max_sigma_yy": np.nanmax(sigma_yy[mask_tgo_bc]),
        "tgo_bc_max_tau_xy": np.nanmax(np.abs(tau_xy[mask_tgo_bc])),
        "tgo_bc_mean_sed": np.nanmean(sed[mask_tgo_bc]),
        "ysz_tgo_n_elements": n_ysz_tgo,
        "tgo_bc_n_elements": n_tgo_bc,
        "ysz_tgo_n_nodes": _mask_node_count(conn, mask_ysz_tgo),
        "tgo_bc_n_nodes": _mask_node_count(conn, mask_tgo_bc),
        "ysz_tgo_max_loc_x_um": ysz_tgo_loc_x_um,
        "ysz_tgo_max_loc_y_um": ysz_tgo_loc_y_um,
        "tgo_bc_max_loc_x_um": tgo_bc_loc_x_um,
        "tgo_bc_max_loc_y_um": tgo_bc_loc_y_um,
    }
    if extra_fields is not None:
        features.update(extra_fields)

    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(features)
    return features