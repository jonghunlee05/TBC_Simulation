import argparse
import os
from pathlib import Path

import numpy as np

from utils import (
    OAT_DIR,
    ensure_output_tree,
    extract_case_metrics,
    load_yaml,
    run_case,
    write_csv,
)


def _linspace(bounds, n_points):
    if n_points <= 1:
        return [float(bounds[0])]
    return np.linspace(float(bounds[0]), float(bounds[1]), n_points).tolist()


def _find_layer_thickness(geom, name_fragment):
    for layer in geom["layers"]:
        if name_fragment.lower() in layer["name"].lower():
            return float(layer["thickness_um"])
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def main():
    parser = argparse.ArgumentParser(description="Run Part I OAT sweeps.")
    parser.add_argument(
        "--geometry",
        default=os.path.join("00_inputs", "geometry_spec.yaml"),
        help="Geometry spec YAML",
    )
    parser.add_argument(
        "--materials",
        default=os.path.join("00_inputs", "materials.yaml"),
        help="Materials YAML",
    )
    parser.add_argument(
        "--bounds",
        default=os.path.join("00_inputs", "parameters_bounds.yaml"),
        help="Bounds YAML",
    )
    parser.add_argument("--enable_roughness", action="store_true")
    parser.add_argument("--n_select", type=int, default=200)
    args = parser.parse_args()

    ensure_output_tree()

    bounds = load_yaml(args.bounds)["bounds"]
    base_geom = load_yaml(args.geometry)
    base_tgo = _find_layer_thickness(base_geom, "tgo")
    base_ysz = _find_layer_thickness(base_geom, "ysz")
    base_bond = _find_layer_thickness(base_geom, "bond")

    baseline = {
        "delta_t": 900.0,
        "tgo_thickness_um": base_tgo,
        "ysz_thickness_um": base_ysz,
        "bond_thickness_um": base_bond,
        "alpha_scale": 1.0,
        "e_scale": 1.0,
        "roughness_amplitude": 0.0,
        "roughness_wavelength": 100.0,
    }

    sweeps = {
        "deltaT": [300.0, 500.0, 700.0, 900.0, 1100.0],
        "tgo_thickness_um": [0.5, 1.0, 2.0, 4.0, 6.0],
        "alpha_scale": _linspace(bounds.get("alpha_ysz_scale", [0.8, 1.2]), 5),
        "e_scale": _linspace(bounds.get("E_ysz_scale", [0.5, 2.0]), 4),
    }

    rough_amp_bounds = bounds.get("roughness_amplitude_um", [0.0, 10.0])
    rough_wave_bounds = bounds.get("roughness_wavelength_um", [50.0, 500.0])
    rough_amp_mid = 0.5 * (float(rough_amp_bounds[0]) + float(rough_amp_bounds[1]))

    if args.enable_roughness:
        sweeps["roughness_amplitude"] = _linspace(rough_amp_bounds, 5)
        sweeps["roughness_wavelength"] = _linspace(rough_wave_bounds, 5)

    for sweep_name, values in sweeps.items():
        rows = []
        for value in values:
            params = dict(baseline)
            if sweep_name == "deltaT":
                params["delta_t"] = value
            else:
                params[sweep_name] = value
            if sweep_name == "roughness_wavelength" and params["roughness_amplitude"] == 0.0:
                params["roughness_amplitude"] = rough_amp_mid

            features = run_case(
                args.geometry,
                args.materials,
                mesh_path=Path(OAT_DIR) / f"mesh_{sweep_name}_{value:.3f}.mesh",
                delta_t=params["delta_t"],
                tgo_thickness_um=params["tgo_thickness_um"],
                ysz_thickness_um=params["ysz_thickness_um"],
                bond_thickness_um=params["bond_thickness_um"],
                alpha_scale=params["alpha_scale"],
                e_scale=params["e_scale"],
                enable_roughness=args.enable_roughness,
                roughness_amplitude=params["roughness_amplitude"],
                roughness_wavelength=params["roughness_wavelength"],
                n_select=args.n_select,
            )

            row = {
                "sweep_param": sweep_name,
                "value": value,
                "delta_t": params["delta_t"],
                "tgo_thickness_um": params["tgo_thickness_um"],
                "ysz_thickness_um": params["ysz_thickness_um"],
                "bond_thickness_um": params["bond_thickness_um"],
                "alpha_scale": params["alpha_scale"],
                "e_scale": params["e_scale"],
                "roughness_amplitude_um": params["roughness_amplitude"],
                "roughness_wavelength_um": params["roughness_wavelength"],
            }
            row.update(extract_case_metrics(features))
            rows.append(row)

        out_csv = Path(OAT_DIR) / f"oat_{sweep_name}.csv"
        write_csv(rows, out_csv)
        print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
