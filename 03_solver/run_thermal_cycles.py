import argparse
import math
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_mesh"))

from extract_features import extract_features  # noqa: E402
from make_mesh import build_mesh  # noqa: E402
from thermoelastic_solver import build_case_context, solve_delta_t  # noqa: E402


def _load_cycle(cycle_path):
    """Load thermal cycle metadata with fallbacks for legacy keys."""
    with open(cycle_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)["thermal_cycle"]

    t_min = float(spec.get("T_min_C", spec.get("T_room_C", 25.0)))
    t_max = float(spec.get("T_max_C", spec.get("T_peak_C", 900.0)))
    hold_time = float(spec.get("hold_time_s", 0.0))
    n_cycles = int(spec.get("n_cycles", 1))

    if "heating_rate_Cps" in spec:
        heating_time = abs(t_max - t_min) / float(spec["heating_rate_Cps"])
    else:
        heating_time = float(spec.get("heating_time_s", 0.0))

    if "cooling_rate_Cps" in spec:
        cooling_time = abs(t_max - t_min) / float(spec["cooling_rate_Cps"])
    else:
        cooling_time = float(spec.get("cooling_time_s", 0.0))

    return {
        "t_min": t_min,
        "t_max": t_max,
        "hold_time": hold_time,
        "heating_time": heating_time,
        "cooling_time": cooling_time,
        "n_cycles": n_cycles,
    }


def _load_geometry(geometry_path):
    """Load geometry specification from YAML."""
    with open(geometry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_geometry(spec, output_path):
    """Persist a geometry spec YAML."""
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _find_layer_thickness(layers, name_fragment):
    """Find layer thickness by name fragment match."""
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            return float(layer["thickness_um"])
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _set_layer_thickness(layers, name_fragment, thickness):
    """Set layer thickness by name fragment match."""
    for layer in layers:
        if name_fragment.lower() in layer["name"].lower():
            layer["thickness_um"] = float(thickness)
            return
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _load_kp(bounds_path):
    """Load a default k_p from bounds using geometric mean."""
    with open(bounds_path, "r", encoding="utf-8") as f:
        bounds = yaml.safe_load(f)["bounds"]
    k_min, k_max = bounds["k_p"]
    # Geometric mean for a neutral default.
    return math.sqrt(float(k_min) * float(k_max))


def _growth_update(h_um, k_p_m2_s, dt_s):
    """Apply parabolic growth update in micrometers."""
    # Convert k_p (m^2/s) to um^2/s before applying.
    k_p_um2_s = k_p_m2_s * 1.0e12
    h_new = math.sqrt(max(0.0, h_um * h_um + k_p_um2_s * dt_s))
    return h_new


def main():
    """Run cycle-resolved thermoelastic solves with TGO growth."""
    parser = argparse.ArgumentParser(description="Run thermal cycles with TGO growth.")
    parser.add_argument(
        "--geometry",
        default=os.path.join("00_inputs", "geometry_spec.yaml"),
        help="Path to geometry_spec.yaml",
    )
    parser.add_argument(
        "--materials",
        default=os.path.join("00_inputs", "materials.yaml"),
        help="Path to materials.yaml",
    )
    parser.add_argument(
        "--cycle",
        default=os.path.join("00_inputs", "cycle.yaml"),
        help="Path to cycle.yaml",
    )
    parser.add_argument(
        "--bounds",
        default=os.path.join("00_inputs", "parameters_bounds.yaml"),
        help="Path to parameters_bounds.yaml (for default k_p)",
    )
    parser.add_argument(
        "--k_p",
        type=float,
        default=None,
        help="Parabolic growth constant (m^2/s)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("05_outputs", "features", "features_cyclewise.csv"),
        help="Cyclewise feature output CSV",
    )
    parser.add_argument(
        "--runs_dir",
        default=os.path.join("04_runs", "thermal_cycles"),
        help="Directory for cycle meshes/specs",
    )
    parser.add_argument(
        "--fields_dir",
        default=os.path.join("04_runs", "thermal_cycles", "fields"),
        help="Directory for VTK outputs",
    )
    parser.add_argument("--nx", type=int, default=200, help="Number of x divisions")
    parser.add_argument(
        "--dy_scale", type=float, default=1.0, help="Scale factor for y spacing"
    )
    parser.add_argument(
        "--save_fields",
        action="store_true",
        help="Save VTK fields per cycle state",
    )
    args = parser.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.save_fields:
        os.makedirs(args.fields_dir, exist_ok=True)

    cycle = _load_cycle(args.cycle)
    base_geom = _load_geometry(args.geometry)

    k_p = args.k_p if args.k_p is not None else _load_kp(args.bounds)

    tgo_th = _find_layer_thickness(base_geom["layers"], "tgo")
    ysz_th = _find_layer_thickness(base_geom["layers"], "ysz")
    bond_th = _find_layer_thickness(base_geom["layers"], "bond")

    dt_cycle = cycle["heating_time"] + cycle["hold_time"] + cycle["cooling_time"]
    for cycle_id in range(1, cycle["n_cycles"] + 1):
        # Parabolic growth once per cycle (quasi-static approximation).
        tgo_next = _growth_update(tgo_th, k_p, dt_cycle)
        delta_h = max(0.0, tgo_next - tgo_th)
        growth_strain = delta_h / max(tgo_th, 1e-12)
        tgo_th = tgo_next

        cycle_dir = os.path.join(args.runs_dir, f"cycle_{cycle_id:04d}")
        os.makedirs(cycle_dir, exist_ok=True)
        geom = dict(base_geom)
        geom_layers = [dict(layer) for layer in base_geom["layers"]]
        geom["layers"] = geom_layers
        _set_layer_thickness(geom["layers"], "tgo", tgo_th)
        _set_layer_thickness(geom["layers"], "ysz", ysz_th)
        _set_layer_thickness(geom["layers"], "bond", bond_th)
        # TODO: add roughness amplitude/wavelength to geometry perturbation here.

        geom_path = os.path.join(cycle_dir, "geometry_spec.yaml")
        mesh_path = os.path.join(cycle_dir, "tbc_2d.mesh")
        _write_geometry(geom, geom_path)
        build_mesh(geom_path, mesh_path, nx=args.nx, dy_scale=args.dy_scale)

        context = build_case_context(geom_path, args.materials, mesh_path)
        for t_state, t_val in (("min", cycle["t_min"]), ("max", cycle["t_max"])):
            # Uniform temperature relative to the minimum state.
            delta_t = t_val - cycle["t_min"]
            pb, state, out = solve_delta_t(context, delta_t, growth_strain=growth_strain)

            if args.save_fields:
                vtk_path = os.path.join(
                    args.fields_dir,
                    f"u_snapshot_cycle_{cycle_id:04d}_{t_state}.vtk",
                )
                pb.save_state(vtk_path, state, out=out)

            extract_features(
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
                output_csv=args.output_csv,
                delta_t=delta_t,
                extra_fields={
                    "cycle_id": cycle_id,
                    "t_state": t_state,
                    "tgo_thickness_um": tgo_th,
                    "ysz_thickness_um": ysz_th,
                    "bondcoat_thickness_um": bond_th,
                    "k_p_m2_s": k_p,
                    "growth_strain": growth_strain,
                },
            )


if __name__ == "__main__":
    main()
