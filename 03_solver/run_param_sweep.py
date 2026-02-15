import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "02_mesh"))
from make_mesh import build_mesh  # noqa: E402
from run_one_case import main as run_one_case_main  # noqa: E402


def _load_bounds(bounds_path):
    with open(bounds_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["bounds"]


def _load_geometry(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_materials(materials_path):
    with open(materials_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_materials(spec, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _write_geometry(spec, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _linspace_from_bounds(bounds, n_points):
    if n_points <= 1:
        return [float(bounds[0])]
    return np.linspace(float(bounds[0]), float(bounds[1]), n_points).tolist()


def _find_layer_thickness(layers, name_fragment):
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            return float(layer["thickness_um"])
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")

def _set_layer_thickness(layers, name_fragment, thickness):
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            layer["thickness_um"] = float(thickness)
            return
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _lhs_samples(bounds, n_samples, rng):
    low = float(bounds[0])
    high = float(bounds[1])
    if n_samples <= 1:
        return [low]
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return (low + (high - low) * u).tolist()


def main():
    parser = argparse.ArgumentParser(description="Run param sweep for deltaT and TGO.")
    parser.add_argument(
        "--bounds",
        default=os.path.join("00_inputs", "parameters_bounds.yaml"),
        help="Path to parameters_bounds.yaml",
    )
    parser.add_argument(
        "--geometry",
        default=os.path.join("00_inputs", "geometry_spec.yaml"),
        help="Base geometry_spec.yaml",
    )
    parser.add_argument(
        "--materials",
        default=os.path.join("00_inputs", "materials.yaml"),
        help="Path to materials.yaml",
    )
    parser.add_argument("--n_dt", type=int, default=4, help="Number of deltaT samples")
    parser.add_argument("--n_tgo", type=int, default=4, help="Number of TGO samples")
    parser.add_argument(
        "--sampling",
        choices=[
            "grid",
            "lhs",
            "lhs_extended",
            "oat_delta_t",
            "oat_tgo",
            "oat_ysz",
            "oat_bond",
            "oat_alpha",
            "oat_alpha_sub",
            "oat_alpha_bond",
            "oat_alpha_tgo",
            "oat_E",
            "oat_E_sub",
            "oat_E_bond",
            "oat_E_tgo",
            "oat_roughness",
            "oat_roughness_wavelength",
        ],
        default="grid",
        help="Sampling mode for deltaT and TGO",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of LHS samples when sampling=lhs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for LHS sampling",
    )
    parser.add_argument(
        "--fixed_delta_t",
        type=float,
        default=900.0,
        help="Fixed deltaT (C) for oat_tgo",
    )
    parser.add_argument(
        "--fixed_tgo",
        type=float,
        default=None,
        help="Fixed TGO thickness (um) for oat_delta_t",
    )
    parser.add_argument(
        "--fixed_ysz",
        type=float,
        default=None,
        help="Fixed YSZ thickness (um) for OAT modes",
    )
    parser.add_argument(
        "--fixed_bond",
        type=float,
        default=None,
        help="Fixed bondcoat thickness (um) for OAT modes",
    )
    parser.add_argument(
        "--alpha_scale_ysz",
        type=float,
        default=1.0,
        help="YSZ CTE scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--alpha_scale_sub",
        type=float,
        default=1.0,
        help="Substrate CTE scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--alpha_scale_bond",
        type=float,
        default=1.0,
        help="Bondcoat CTE scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--alpha_scale_tgo",
        type=float,
        default=1.0,
        help="TGO CTE scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--E_scale_ysz",
        type=float,
        default=1.0,
        help="YSZ elastic modulus scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--E_scale_sub",
        type=float,
        default=1.0,
        help="Substrate elastic modulus scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--E_scale_bond",
        type=float,
        default=1.0,
        help="Bondcoat elastic modulus scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--E_scale_tgo",
        type=float,
        default=1.0,
        help="TGO elastic modulus scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--enable_roughness",
        action="store_true",
        help="Enable sinusoidal roughness at YSZ/TGO interface",
    )
    parser.add_argument(
        "--roughness_amplitude",
        type=float,
        default=0.0,
        help="Roughness amplitude (um)",
    )
    parser.add_argument(
        "--roughness_wavelength",
        type=float,
        default=100.0,
        help="Roughness wavelength (um)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("05_outputs", "features", "sweep_dataset.csv"),
        help="Output dataset CSV",
    )
    parser.add_argument(
        "--cases_dir",
        default=os.path.join("04_runs", "param_sweep"),
        help="Directory for case-specific meshes/specs",
    )
    args = parser.parse_args()

    os.chdir(repo_root)
    bounds = _load_bounds(repo_root / args.bounds)
    base_geom = _load_geometry(repo_root / args.geometry)
    base_mats = _load_materials(repo_root / args.materials)

    base_tgo = _find_layer_thickness(base_geom["layers"], "tgo")
    base_ysz = _find_layer_thickness(base_geom["layers"], "ysz")
    base_bond = _find_layer_thickness(base_geom["layers"], "bond")
    fixed_tgo = base_tgo if args.fixed_tgo is None else args.fixed_tgo
    fixed_ysz = base_ysz if args.fixed_ysz is None else args.fixed_ysz
    fixed_bond = base_bond if args.fixed_bond is None else args.fixed_bond

    fixed_alpha_ysz = args.alpha_scale_ysz
    fixed_alpha_sub = args.alpha_scale_sub
    fixed_alpha_bond = args.alpha_scale_bond
    fixed_alpha_tgo = args.alpha_scale_tgo
    fixed_E_ysz = args.E_scale_ysz
    fixed_E_sub = args.E_scale_sub
    fixed_E_bond = args.E_scale_bond
    fixed_E_tgo = args.E_scale_tgo
    fixed_rough_amp = args.roughness_amplitude
    fixed_rough_wave = args.roughness_wavelength

    if args.sampling == "lhs":
        rng = np.random.default_rng(args.seed)
        dt_values = _lhs_samples(bounds["deltaT_C"], args.n_samples, rng)
        tgo_values = _lhs_samples(
            bounds["initial_TGO_thickness_um"], args.n_samples, rng
        )
        pairs = [
            (
                dt,
                tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for dt, tgo in zip(dt_values, tgo_values)
        ]
    elif args.sampling == "lhs_extended":
        rng = np.random.default_rng(args.seed)
        dt_values = _lhs_samples(bounds["deltaT_C"], args.n_samples, rng)
        tgo_values = _lhs_samples(
            bounds["initial_TGO_thickness_um"], args.n_samples, rng
        )
        alpha_ysz_values = _lhs_samples(bounds["alpha_ysz_scale"], args.n_samples, rng)
        alpha_sub_values = _lhs_samples(bounds["alpha_sub_scale"], args.n_samples, rng)
        alpha_bond_values = _lhs_samples(bounds["alpha_bond_scale"], args.n_samples, rng)
        alpha_tgo_values = _lhs_samples(bounds["alpha_tgo_scale"], args.n_samples, rng)
        E_ysz_values = _lhs_samples(bounds["E_ysz_scale"], args.n_samples, rng)
        E_sub_values = _lhs_samples(bounds["E_sub_scale"], args.n_samples, rng)
        E_bond_values = _lhs_samples(bounds["E_bond_scale"], args.n_samples, rng)
        E_tgo_values = _lhs_samples(bounds["E_tgo_scale"], args.n_samples, rng)
        rough_amp_values = _lhs_samples(
            bounds["roughness_amplitude_um"], args.n_samples, rng
        )
        rough_wave_values = _lhs_samples(
            bounds["roughness_wavelength_um"], args.n_samples, rng
        )
        pairs = list(
            zip(
                dt_values,
                tgo_values,
                [fixed_ysz] * args.n_samples,
                [fixed_bond] * args.n_samples,
                alpha_ysz_values,
                alpha_sub_values,
                alpha_bond_values,
                alpha_tgo_values,
                E_ysz_values,
                E_sub_values,
                E_bond_values,
                E_tgo_values,
                rough_amp_values,
                rough_wave_values,
            )
        )
    elif args.sampling == "oat_delta_t":
        dt_values = _linspace_from_bounds(bounds["deltaT_C"], args.n_dt)
        pairs = [
            (
                dt,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for dt in dt_values
        ]
    elif args.sampling == "oat_tgo":
        tgo_values = _linspace_from_bounds(bounds["initial_TGO_thickness_um"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for tgo in tgo_values
        ]
    elif args.sampling == "oat_ysz":
        ysz_values = _linspace_from_bounds(bounds["YSZ_thickness_um"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for ysz in ysz_values
        ]
    elif args.sampling == "oat_bond":
        bond_values = _linspace_from_bounds(
            bounds["bondcoat_thickness_um"], args.n_tgo
        )
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for bond in bond_values
        ]
    elif args.sampling == "oat_alpha":
        alpha_ysz_values = _linspace_from_bounds(bounds["alpha_ysz_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for alpha_ysz in alpha_ysz_values
        ]
    elif args.sampling == "oat_alpha_sub":
        alpha_sub_values = _linspace_from_bounds(bounds["alpha_sub_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for alpha_sub in alpha_sub_values
        ]
    elif args.sampling == "oat_alpha_bond":
        alpha_bond_values = _linspace_from_bounds(
            bounds["alpha_bond_scale"], args.n_tgo
        )
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for alpha_bond in alpha_bond_values
        ]
    elif args.sampling == "oat_alpha_tgo":
        alpha_tgo_values = _linspace_from_bounds(
            bounds["alpha_tgo_scale"], args.n_tgo
        )
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for alpha_tgo in alpha_tgo_values
        ]
    elif args.sampling == "oat_E":
        E_ysz_values = _linspace_from_bounds(bounds["E_ysz_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for E_ysz in E_ysz_values
        ]
    elif args.sampling == "oat_E_sub":
        E_sub_values = _linspace_from_bounds(bounds["E_sub_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for E_sub in E_sub_values
        ]
    elif args.sampling == "oat_E_bond":
        E_bond_values = _linspace_from_bounds(bounds["E_bond_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for E_bond in E_bond_values
        ]
    elif args.sampling == "oat_E_tgo":
        E_tgo_values = _linspace_from_bounds(bounds["E_tgo_scale"], args.n_tgo)
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for E_tgo in E_tgo_values
        ]
    elif args.sampling == "oat_roughness":
        rough_amp_values = _linspace_from_bounds(
            bounds["roughness_amplitude_um"], args.n_tgo
        )
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                rough_amp,
                fixed_rough_wave,
            )
            for rough_amp in rough_amp_values
        ]
    elif args.sampling == "oat_roughness_wavelength":
        rough_wave_values = _linspace_from_bounds(
            bounds["roughness_wavelength_um"], args.n_tgo
        )
        pairs = [
            (
                args.fixed_delta_t,
                fixed_tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                rough_wave,
            )
            for rough_wave in rough_wave_values
        ]
    else:
        dt_values = _linspace_from_bounds(bounds["deltaT_C"], args.n_dt)
        tgo_values = _linspace_from_bounds(bounds["initial_TGO_thickness_um"], args.n_tgo)
        pairs = list(itertools.product(dt_values, tgo_values))
        pairs = [
            (
                dt,
                tgo,
                fixed_ysz,
                fixed_bond,
                fixed_alpha_ysz,
                fixed_alpha_sub,
                fixed_alpha_bond,
                fixed_alpha_tgo,
                fixed_E_ysz,
                fixed_E_sub,
                fixed_E_bond,
                fixed_E_tgo,
                fixed_rough_amp,
                fixed_rough_wave,
            )
            for dt, tgo in pairs
        ]

    cases_dir = repo_root / args.cases_dir
    cases_dir.mkdir(parents=True, exist_ok=True)

    total_cases = len(pairs)
    for idx, (
        delta_t,
        tgo_th,
        ysz_th,
        bond_th,
        alpha_ysz,
        alpha_sub,
        alpha_bond,
        alpha_tgo,
        E_ysz,
        E_sub,
        E_bond,
        E_tgo,
        rough_amp,
        rough_wave,
    ) in enumerate(pairs, 1):
        print(f"Case {idx}/{total_cases}")
        case_dir = cases_dir / f"case_{idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        geom = dict(base_geom)
        geom_layers = []
        for layer in base_geom["layers"]:
            geom_layers.append(dict(layer))
        geom["layers"] = geom_layers
        _set_layer_thickness(geom["layers"], "tgo", tgo_th)
        _set_layer_thickness(geom["layers"], "ysz", ysz_th)
        _set_layer_thickness(geom["layers"], "bond", bond_th)

        geom_path = case_dir / "geometry_spec.yaml"
        mesh_path = case_dir / "tbc_2d.mesh"
        mats_path = case_dir / "materials.yaml"
        fields_dir = case_dir / "fields"
        fields_dir.mkdir(exist_ok=True)
        _write_geometry(geom, geom_path)

        enable_roughness = args.enable_roughness or rough_amp > 0.0
        build_mesh(
            str(geom_path),
            str(mesh_path),
            enable_roughness=enable_roughness,
            roughness_amplitude=rough_amp,
            roughness_wavelength=rough_wave,
        )

        mats = yaml.safe_load(yaml.safe_dump(base_mats))
        mats["materials"]["YSZ"]["alpha_1K"] = (
            float(mats["materials"]["YSZ"]["alpha_1K"]) * float(alpha_ysz)
        )
        mats["materials"]["substrate"]["alpha_1K"] = (
            float(mats["materials"]["substrate"]["alpha_1K"]) * float(alpha_sub)
        )
        mats["materials"]["bondcoat"]["alpha_1K"] = (
            float(mats["materials"]["bondcoat"]["alpha_1K"]) * float(alpha_bond)
        )
        mats["materials"]["TGO_Al2O3"]["alpha_1K"] = (
            float(mats["materials"]["TGO_Al2O3"]["alpha_1K"]) * float(alpha_tgo)
        )
        mats["materials"]["YSZ"]["E_GPa"] = (
            float(mats["materials"]["YSZ"]["E_GPa"]) * float(E_ysz)
        )
        mats["materials"]["substrate"]["E_GPa"] = (
            float(mats["materials"]["substrate"]["E_GPa"]) * float(E_sub)
        )
        mats["materials"]["bondcoat"]["E_GPa"] = (
            float(mats["materials"]["bondcoat"]["E_GPa"]) * float(E_bond)
        )
        mats["materials"]["TGO_Al2O3"]["E_GPa"] = (
            float(mats["materials"]["TGO_Al2O3"]["E_GPa"]) * float(E_tgo)
        )
        _write_materials(mats, mats_path)

        # Call run_one_case via its CLI entrypoint.
        os.environ["PYTHONPATH"] = str(repo_root)
        run_one_case_main_args = [
            "--geometry",
            str(geom_path),
            "--materials",
            str(mats_path),
            "--mesh",
            str(mesh_path),
            "--delta_t",
            str(delta_t),
            "--output_csv",
            str(repo_root / args.output_csv),
            "--fields_dir",
            str(fields_dir),
            "--alpha_scale_ysz",
            str(alpha_ysz),
            "--alpha_scale_sub",
            str(alpha_sub),
            "--alpha_scale_bond",
            str(alpha_bond),
            "--alpha_scale_tgo",
            str(alpha_tgo),
            "--E_scale_ysz",
            str(E_ysz),
            "--E_scale_sub",
            str(E_sub),
            "--E_scale_bond",
            str(E_bond),
            "--E_scale_tgo",
            str(E_tgo),
            "--roughness_amplitude",
            str(rough_amp),
            "--roughness_wavelength",
            str(rough_wave),
        ]

        # Simulate CLI args for run_one_case
        saved_argv = sys.argv
        sys.argv = ["run_one_case.py"] + run_one_case_main_args
        try:
            run_one_case_main()
        finally:
            sys.argv = saved_argv


if __name__ == "__main__":
    main()
