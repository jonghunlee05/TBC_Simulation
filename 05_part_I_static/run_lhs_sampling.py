import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    LHS_DIR,
    ensure_output_tree,
    extract_case_metrics,
    lhs_samples,
    load_yaml,
    run_case,
    save_input_correlation,
    save_input_histograms,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Run Part I LHS sampling.")
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
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable_roughness", action="store_true")
    parser.add_argument("--n_select", type=int, default=200)
    args = parser.parse_args()

    ensure_output_tree()
    bounds = load_yaml(args.bounds)["bounds"]
    rng = np.random.default_rng(args.seed)

    inputs = {
        "delta_t": lhs_samples(bounds["deltaT_C"], args.n_samples, rng),
        "tgo_thickness_um": lhs_samples(bounds["initial_TGO_thickness_um"], args.n_samples, rng),
        "alpha_scale": lhs_samples(bounds.get("alpha_ysz_scale", [0.8, 1.2]), args.n_samples, rng),
        "e_scale": lhs_samples(bounds.get("E_ysz_scale", [0.5, 2.0]), args.n_samples, rng),
    }

    if "YSZ_thickness_um" in bounds:
        inputs["ysz_thickness_um"] = lhs_samples(bounds["YSZ_thickness_um"], args.n_samples, rng)
    if "bondcoat_thickness_um" in bounds:
        inputs["bond_thickness_um"] = lhs_samples(
            bounds["bondcoat_thickness_um"], args.n_samples, rng
        )

    if args.enable_roughness:
        inputs["roughness_amplitude_um"] = lhs_samples(
            bounds.get("roughness_amplitude_um", [0.0, 10.0]), args.n_samples, rng
        )
        inputs["roughness_wavelength_um"] = lhs_samples(
            bounds.get("roughness_wavelength_um", [50.0, 500.0]), args.n_samples, rng
        )

    rows = []
    for idx in range(args.n_samples):
        params = {k: v[idx] for k, v in inputs.items()}
        case_dir = Path(LHS_DIR) / f"case_{idx + 1:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        features = run_case(
            args.geometry,
            args.materials,
            mesh_path=case_dir / "tbc_2d.mesh",
            delta_t=params["delta_t"],
            tgo_thickness_um=params.get("tgo_thickness_um"),
            ysz_thickness_um=params.get("ysz_thickness_um"),
            bond_thickness_um=params.get("bond_thickness_um"),
            alpha_scale=params["alpha_scale"],
            e_scale=params["e_scale"],
            enable_roughness=args.enable_roughness,
            roughness_amplitude=params.get("roughness_amplitude_um", 0.0),
            roughness_wavelength=params.get("roughness_wavelength_um", 100.0),
            n_select=args.n_select,
        )
        row = dict(params)
        row.update(extract_case_metrics(features))
        rows.append(row)
        print(f"LHS case {idx + 1}/{args.n_samples}")

    out_csv = Path(LHS_DIR) / "partI_lhs_dataset.csv"
    write_csv(rows, out_csv)
    print(f"Saved dataset to {out_csv}")

    input_df = pd.DataFrame({k: v for k, v in inputs.items()})
    save_input_histograms(input_df, LHS_DIR)
    save_input_correlation(input_df, Path(LHS_DIR) / "input_correlation.png")


if __name__ == "__main__":
    main()
