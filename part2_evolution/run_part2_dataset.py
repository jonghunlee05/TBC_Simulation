import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "03_solver"))
sys.path.append(str(repo_root / "02_mesh"))
from run_thermal_cycles import main as run_thermal_cycles_main  # noqa: E402
from trajectory_descriptors import compute_descriptors  # noqa: E402


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _load_bounds(bounds_path):
    return _load_yaml(bounds_path)["bounds"]


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


def _append_row(csv_path, row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def main():
    parser = argparse.ArgumentParser(description="Run Part II evolution dataset.")
    parser.add_argument(
        "--config",
        default=str(repo_root / "part2_evolution" / "config_part2.yaml"),
        help="Path to Part II config YAML",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    paths = config["paths"]
    bounds = _load_bounds(repo_root / paths["bounds"])
    base_geom = _load_yaml(repo_root / paths["geometry"])

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(config["sampling"]["seed"]))
    n_cases = int(config["sampling"]["n_cases"])

    dt_values = _lhs_samples(bounds["deltaT_C"], n_cases, rng)
    tgo_values = _lhs_samples(
        bounds["initial_TGO_thickness_um"], n_cases, rng
    )
    kp_values = _lhs_samples(bounds["k_p"], n_cases, rng)
    rough_amp_values = _lhs_samples(bounds["roughness_amplitude_um"], n_cases, rng)
    rough_wave_values = _lhs_samples(bounds["roughness_wavelength_um"], n_cases, rng)
    cycle_dt_values = _lhs_samples(
        config["sampling"]["cycle_dt_bounds_s"], n_cases, rng
    )

    descriptors_csv = output_dir / "part2_descriptors.csv"

    for case_id in range(1, n_cases + 1):
        case_dir = output_dir / f"case_{case_id:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        delta_t = float(dt_values[case_id - 1])
        tgo_th = float(tgo_values[case_id - 1])
        k_p = float(kp_values[case_id - 1])
        rough_amp = float(rough_amp_values[case_id - 1])
        rough_wave = float(rough_wave_values[case_id - 1])
        cycle_dt_s = float(cycle_dt_values[case_id - 1])

        geom = dict(base_geom)
        geom_layers = [dict(layer) for layer in base_geom["layers"]]
        geom["layers"] = geom_layers
        _set_layer_thickness(geom["layers"], "tgo", tgo_th)
        geom_path = case_dir / "geometry_spec.yaml"
        _write_yaml(geom, geom_path)

        t_min = float(config["cycle"]["t_min_C"])
        t_max = t_min + delta_t
        cycle_spec = {
            "thermal_cycle": {
                "T_min_C": t_min,
                "T_max_C": t_max,
                "heating_time_s": float(config["cycle"]["heating_time_s"]),
                "cooling_time_s": float(config["cycle"]["cooling_time_s"]),
                "hold_time_s": cycle_dt_s,
                "n_cycles": int(config["cycle"]["n_cycles"]),
            }
        }
        cycle_path = case_dir / "cycle.yaml"
        _write_yaml(cycle_spec, cycle_path)

        output_csv = case_dir / "cyclewise.csv"
        runs_dir = case_dir / "runs"
        fields_dir = case_dir / "fields"
        os.makedirs(runs_dir, exist_ok=True)

        enable_roughness = config["roughness"]["enable"] or rough_amp > 0.0

        run_args = [
            "--geometry",
            str(geom_path),
            "--materials",
            str(repo_root / paths["materials"]),
            "--cycle",
            str(cycle_path),
            "--k_p",
            str(k_p),
            "--output_csv",
            str(output_csv),
            "--runs_dir",
            str(runs_dir),
            "--fields_dir",
            str(fields_dir),
            "--nx",
            str(config["mesh"]["nx"]),
            "--dy_scale",
            str(config["mesh"]["dy_scale"]),
            "--growth_eigenstrain_coeff",
            str(config["growth"]["growth_eigenstrain_coeff"]),
            "--roughness_amplitude",
            str(rough_amp),
            "--roughness_wavelength",
            str(rough_wave),
        ]
        if enable_roughness:
            run_args.append("--enable_roughness")

        saved_argv = sys.argv
        sys.argv = ["run_thermal_cycles.py"] + run_args
        try:
            run_thermal_cycles_main()
        finally:
            sys.argv = saved_argv

        descriptors = compute_descriptors(str(output_csv))
        summary = {
            "case_id": case_id,
            "delta_t": delta_t,
            "k_p_m2_s": k_p,
            "initial_tgo_thickness_um": tgo_th,
            "cycle_dt_s": cycle_dt_s,
            "t_min_C": t_min,
            "t_max_C": t_max,
            "n_cycles": int(config["cycle"]["n_cycles"]),
            "roughness_amplitude_um": rough_amp,
            "roughness_wavelength_um": rough_wave,
            "growth_eigenstrain_coeff": float(config["growth"]["growth_eigenstrain_coeff"]),
        }
        summary.update(descriptors)

        summary_csv = case_dir / "summary.csv"
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        _append_row(descriptors_csv, summary)


if __name__ == "__main__":
    main()
