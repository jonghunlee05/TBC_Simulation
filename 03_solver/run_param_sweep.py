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
        choices=["grid", "lhs", "oat_delta_t", "oat_tgo"],
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

    base_tgo = _find_layer_thickness(base_geom["layers"], "tgo")
    fixed_tgo = base_tgo if args.fixed_tgo is None else args.fixed_tgo

    if args.sampling == "lhs":
        rng = np.random.default_rng(args.seed)
        dt_values = _lhs_samples(bounds["deltaT_C"], args.n_samples, rng)
        tgo_values = _lhs_samples(
            bounds["initial_TGO_thickness_um"], args.n_samples, rng
        )
        pairs = list(zip(dt_values, tgo_values))
    elif args.sampling == "oat_delta_t":
        dt_values = _linspace_from_bounds(bounds["deltaT_C"], args.n_dt)
        pairs = [(dt, fixed_tgo) for dt in dt_values]
    elif args.sampling == "oat_tgo":
        tgo_values = _linspace_from_bounds(bounds["initial_TGO_thickness_um"], args.n_tgo)
        pairs = [(args.fixed_delta_t, tgo) for tgo in tgo_values]
    else:
        dt_values = _linspace_from_bounds(bounds["deltaT_C"], args.n_dt)
        tgo_values = _linspace_from_bounds(bounds["initial_TGO_thickness_um"], args.n_tgo)
        pairs = list(itertools.product(dt_values, tgo_values))

    cases_dir = repo_root / args.cases_dir
    cases_dir.mkdir(parents=True, exist_ok=True)

    for idx, (delta_t, tgo_th) in enumerate(pairs, 1):
        case_dir = cases_dir / f"case_{idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        geom = dict(base_geom)
        geom_layers = []
        for layer in base_geom["layers"]:
            if "tgo" in layer["name"].lower():
                new_layer = dict(layer)
                new_layer["thickness_um"] = float(tgo_th)
                geom_layers.append(new_layer)
            else:
                geom_layers.append(dict(layer))
        geom["layers"] = geom_layers

        geom_path = case_dir / "geometry_spec.yaml"
        mesh_path = case_dir / "tbc_2d.mesh"
        fields_dir = case_dir / "fields"
        fields_dir.mkdir(exist_ok=True)
        _write_geometry(geom, geom_path)

        build_mesh(str(geom_path), str(mesh_path))

        # Call run_one_case via its CLI entrypoint.
        os.environ["PYTHONPATH"] = str(repo_root)
        run_one_case_main_args = [
            "--geometry",
            str(geom_path),
            "--materials",
            str(repo_root / args.materials),
            "--mesh",
            str(mesh_path),
            "--delta_t",
            str(delta_t),
            "--output_csv",
            str(repo_root / args.output_csv),
            "--fields_dir",
            str(fields_dir),
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
