import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "02_mesh"))
sys.path.append(str(repo_root / "03_solver"))

from make_mesh import build_mesh  # noqa: E402
from run_one_case import main as run_one_case_main  # noqa: E402


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _load_bounds(bounds_path):
    return _load_yaml(bounds_path)["bounds"]


def _load_geometry(spec_path):
    return _load_yaml(spec_path)


def _load_materials(materials_path):
    return _load_yaml(materials_path)


def _linspace_from_bounds(bounds, n_points):
    if n_points <= 1:
        return [float(bounds[0])]
    return np.linspace(float(bounds[0]), float(bounds[1]), n_points).tolist()


def _lhs_samples(bounds, n_samples, rng):
    low = float(bounds[0])
    high = float(bounds[1])
    if n_samples <= 1:
        return [low]
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(u)
    return (low + (high - low) * u).tolist()


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


def _build_pairs(config, bounds, base_geom):
    sampling = config["sampling"]
    fixed = config["fixed"]
    scales = config["scales"]
    rough = config["roughness"]

    base_tgo = _find_layer_thickness(base_geom["layers"], "tgo")
    base_ysz = _find_layer_thickness(base_geom["layers"], "ysz")
    base_bond = _find_layer_thickness(base_geom["layers"], "bond")

    fixed_tgo = base_tgo if fixed["tgo_thickness_um"] is None else fixed["tgo_thickness_um"]
    fixed_ysz = base_ysz if fixed["ysz_thickness_um"] is None else fixed["ysz_thickness_um"]
    fixed_bond = base_bond if fixed["bondcoat_thickness_um"] is None else fixed["bondcoat_thickness_um"]

    method = sampling["method"]
    n_samples = int(sampling["n_samples"])
    n_dt = int(sampling["n_dt"])
    n_tgo = int(sampling["n_tgo"])
    seed = int(sampling["seed"])

    fixed_params = (
        fixed_tgo,
        fixed_ysz,
        fixed_bond,
        scales["alpha_ysz"],
        scales["alpha_sub"],
        scales["alpha_bond"],
        scales["alpha_tgo"],
        scales["E_ysz"],
        scales["E_sub"],
        scales["E_bond"],
        scales["E_tgo"],
        rough["amplitude_um"],
        rough["wavelength_um"],
    )

    if method == "lhs":
        rng = np.random.default_rng(seed)
        dt_values = _lhs_samples(bounds["deltaT_C"], n_samples, rng)
        tgo_values = _lhs_samples(bounds["initial_TGO_thickness_um"], n_samples, rng)
        pairs = [
            (dt, tgo, fixed_ysz, fixed_bond, *fixed_params[3:])
            for dt, tgo in zip(dt_values, tgo_values)
        ]
    elif method == "lhs_extended":
        rng = np.random.default_rng(seed)
        dt_values = _lhs_samples(bounds["deltaT_C"], n_samples, rng)
        tgo_values = _lhs_samples(bounds["initial_TGO_thickness_um"], n_samples, rng)
        alpha_ysz_values = _lhs_samples(bounds["alpha_ysz_scale"], n_samples, rng)
        alpha_sub_values = _lhs_samples(bounds["alpha_sub_scale"], n_samples, rng)
        alpha_bond_values = _lhs_samples(bounds["alpha_bond_scale"], n_samples, rng)
        alpha_tgo_values = _lhs_samples(bounds["alpha_tgo_scale"], n_samples, rng)
        E_ysz_values = _lhs_samples(bounds["E_ysz_scale"], n_samples, rng)
        E_sub_values = _lhs_samples(bounds["E_sub_scale"], n_samples, rng)
        E_bond_values = _lhs_samples(bounds["E_bond_scale"], n_samples, rng)
        E_tgo_values = _lhs_samples(bounds["E_tgo_scale"], n_samples, rng)
        rough_amp_values = _lhs_samples(
            bounds["roughness_amplitude_um"], n_samples, rng
        )
        rough_wave_values = _lhs_samples(
            bounds["roughness_wavelength_um"], n_samples, rng
        )
        pairs = list(
            zip(
                dt_values,
                tgo_values,
                [fixed_ysz] * n_samples,
                [fixed_bond] * n_samples,
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
    elif method == "oat_delta_t":
        dt_values = _linspace_from_bounds(bounds["deltaT_C"], n_dt)
        pairs = [(dt, *fixed_params) for dt in dt_values]
    elif method == "oat_tgo":
        tgo_values = _linspace_from_bounds(bounds["initial_TGO_thickness_um"], n_tgo)
        pairs = [(fixed["delta_t"], tgo, fixed_ysz, fixed_bond, *fixed_params[3:]) for tgo in tgo_values]
    elif method == "oat_ysz":
        ysz_values = _linspace_from_bounds(bounds["YSZ_thickness_um"], n_tgo)
        pairs = [(fixed["delta_t"], fixed_tgo, ysz, fixed_bond, *fixed_params[3:]) for ysz in ysz_values]
    elif method == "oat_bond":
        bond_values = _linspace_from_bounds(bounds["bondcoat_thickness_um"], n_tgo)
        pairs = [(fixed["delta_t"], fixed_tgo, fixed_ysz, bond, *fixed_params[3:]) for bond in bond_values]
    elif method == "oat_alpha_ysz":
        alpha_values = _linspace_from_bounds(bounds["alpha_ysz_scale"], n_tgo)
        pairs = [(fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, a, *fixed_params[4:]) for a in alpha_values]
    elif method == "oat_alpha_sub":
        alpha_values = _linspace_from_bounds(bounds["alpha_sub_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, fixed_params[3], a, *fixed_params[5:])
            for a in alpha_values
        ]
    elif method == "oat_alpha_bond":
        alpha_values = _linspace_from_bounds(bounds["alpha_bond_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, fixed_params[3], fixed_params[4], a, *fixed_params[6:])
            for a in alpha_values
        ]
    elif method == "oat_alpha_tgo":
        alpha_values = _linspace_from_bounds(bounds["alpha_tgo_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, fixed_params[3], fixed_params[4], fixed_params[5], a, *fixed_params[7:])
            for a in alpha_values
        ]
    elif method == "oat_E_ysz":
        E_values = _linspace_from_bounds(bounds["E_ysz_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:7], e, *fixed_params[8:])
            for e in E_values
        ]
    elif method == "oat_E_sub":
        E_values = _linspace_from_bounds(bounds["E_sub_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:8], e, *fixed_params[9:])
            for e in E_values
        ]
    elif method == "oat_E_bond":
        E_values = _linspace_from_bounds(bounds["E_bond_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:9], e, *fixed_params[10:])
            for e in E_values
        ]
    elif method == "oat_E_tgo":
        E_values = _linspace_from_bounds(bounds["E_tgo_scale"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:10], e, *fixed_params[11:])
            for e in E_values
        ]
    elif method == "oat_roughness_amp":
        amp_values = _linspace_from_bounds(bounds["roughness_amplitude_um"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:11], a, fixed_params[12])
            for a in amp_values
        ]
    elif method == "oat_roughness_wave":
        wave_values = _linspace_from_bounds(bounds["roughness_wavelength_um"], n_tgo)
        pairs = [
            (fixed["delta_t"], fixed_tgo, fixed_ysz, fixed_bond, *fixed_params[:12], w)
            for w in wave_values
        ]
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Run Part I state-driver dataset.")
    parser.add_argument(
        "--config",
        default=str(repo_root / "part1_state" / "config_part1.yaml"),
        help="Path to Part I config YAML",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    paths = config["paths"]

    os.makedirs(paths["output_dir"], exist_ok=True)
    os.makedirs(paths["cases_dir"], exist_ok=True)

    bounds = _load_bounds(repo_root / paths["bounds"])
    base_geom = _load_geometry(repo_root / paths["geometry"])
    base_mats = _load_materials(repo_root / paths["materials"])

    pairs = _build_pairs(config, bounds, base_geom)
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
        case_dir = Path(paths["cases_dir"]) / f"case_{idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        geom = dict(base_geom)
        geom_layers = [dict(layer) for layer in base_geom["layers"]]
        geom["layers"] = geom_layers
        _set_layer_thickness(geom["layers"], "tgo", tgo_th)
        _set_layer_thickness(geom["layers"], "ysz", ysz_th)
        _set_layer_thickness(geom["layers"], "bond", bond_th)

        geom_path = case_dir / "geometry_spec.yaml"
        mesh_path = case_dir / "tbc_2d.mesh"
        mats_path = case_dir / "materials.yaml"
        fields_dir = case_dir / "fields"
        fields_dir.mkdir(exist_ok=True)
        _write_yaml(geom, geom_path)

        enable_roughness = config["roughness"]["enable"] or rough_amp > 0.0
        build_mesh(
            str(geom_path),
            str(mesh_path),
            nx=int(config["mesh"]["nx"]),
            dy_scale=float(config["mesh"]["dy_scale"]),
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
        _write_yaml(mats, mats_path)

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
            str(repo_root / paths["output_csv"]),
            "--fields_dir",
            str(fields_dir),
            "--n_select",
            str(config["mesh"]["n_select"]),
            "--bc_variant",
            str(config["fixed"]["bc_variant"]),
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

        saved_argv = sys.argv
        sys.argv = ["run_one_case.py"] + run_one_case_main_args
        try:
            run_one_case_main()
        finally:
            sys.argv = saved_argv


if __name__ == "__main__":
    main()
