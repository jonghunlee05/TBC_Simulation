import argparse
import logging
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
    """Apply parabolic growth update in micrometers (um)."""
    # Convert k_p (m^2/s) to um^2/s before applying.
    k_p_um2_s = k_p_m2_s * 1.0e12
    h_new = math.sqrt(max(0.0, h_um * h_um + k_p_um2_s * dt_s))
    return h_new


def _setup_logger(log_path):
    logger = logging.getLogger("thermal_cycles")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _validate_growth(tgo_history):
    if any(tgo_history[i] < tgo_history[i - 1] for i in range(1, len(tgo_history))):
        raise ValueError("TGO thickness is not monotonically increasing.")


def _check_feature_variation(feature_rows, tol=1e-6):
    by_state = {"min": [], "max": []}
    for row in feature_rows:
        by_state[row["t_state"]].append(row)

    def _all_close(values):
        if not values:
            return True
        v0 = values[0]
        return all(abs(v - v0) <= tol for v in values[1:])

    warnings = []
    for t_state, rows in by_state.items():
        sigma_vals = [r["ysz_tgo_max_sigma_yy"] for r in rows]
        tau_vals = [r["ysz_tgo_max_tau_xy"] for r in rows]
        if _all_close(sigma_vals):
            warnings.append(f"{t_state}: ysz_tgo_max_sigma_yy appears constant")
        if _all_close(tau_vals):
            warnings.append(f"{t_state}: ysz_tgo_max_tau_xy appears constant")
    return warnings


def _write_validation_plots(feature_rows, plot_dir):
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    cycles = sorted({row["cycle_id"] for row in feature_rows})
    tgo_by_cycle = {c: None for c in cycles}
    for row in feature_rows:
        tgo_by_cycle[row["cycle_id"]] = row["tgo_thickness_um"]
    plt.figure()
    plt.plot(cycles, [tgo_by_cycle[c] for c in cycles], marker="o")
    plt.xlabel("Cycle")
    plt.ylabel("TGO thickness (um)")
    plt.title("TGO Thickness vs Cycle")
    plt.savefig(os.path.join(plot_dir, "tgo_thickness_vs_cycle.png"), dpi=200)
    plt.close()

    for metric, fname, ylabel in [
        ("ysz_tgo_max_sigma_yy", "sigma_yy_vs_cycle.png", "Max sigma_yy (Pa)"),
        ("ysz_tgo_max_tau_xy", "tau_xy_vs_cycle.png", "Max tau_xy (Pa)"),
    ]:
        plt.figure()
        for t_state in ("min", "max"):
            values = [
                row[metric]
                for row in feature_rows
                if row["t_state"] == t_state
            ]
            plt.plot(cycles, values, marker="o", label=f"{t_state} state")
        plt.xlabel("Cycle")
        plt.ylabel(ylabel)
        plt.title(metric.replace("_", " ").title())
        plt.legend()
        plt.savefig(os.path.join(plot_dir, fname), dpi=200)
        plt.close()


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
        "--arrhenius_growth",
        action="store_true",
        help="Enable Arrhenius temperature-dependent k_p",
    )
    parser.add_argument(
        "--kp0",
        type=float,
        default=None,
        help="Arrhenius pre-exponential k_p0 (m^2/s)",
    )
    parser.add_argument(
        "--Q",
        type=float,
        default=None,
        help="Arrhenius activation energy Q (J/mol)",
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
    parser.add_argument(
        "--disable_growth_strain",
        action="store_true",
        help="Disable TGO growth eigenstrain for comparison runs",
    )
    parser.add_argument(
        "--growth_eigenstrain_coeff",
        type=float,
        default=1e-3,
        help=(
            "Scale factor for growth eigenstrain: eps_g = coeff * (delta_h / h_initial). "
            "Keep small (e.g., 1e-3 or 1e-4) to bound prestress."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.save_fields:
        os.makedirs(args.fields_dir, exist_ok=True)

    logger = _setup_logger(os.path.join(args.runs_dir, "thermal_cycles.log"))

    cycle = _load_cycle(args.cycle)
    base_geom = _load_geometry(args.geometry)

    k_p = args.k_p if args.k_p is not None else _load_kp(args.bounds)
    if args.arrhenius_growth:
        if args.kp0 is None or args.Q is None:
            raise ValueError("Arrhenius growth requires --kp0 and --Q.")
        k_p = None

    tgo_th = _find_layer_thickness(base_geom["layers"], "tgo")
    tgo_initial = tgo_th
    ysz_th = _find_layer_thickness(base_geom["layers"], "ysz")
    bond_th = _find_layer_thickness(base_geom["layers"], "bond")

    dt_cycle = cycle["heating_time"] + cycle["hold_time"] + cycle["cooling_time"]
    if dt_cycle <= 0.0:
        raise ValueError("Cycle time must be positive for TGO growth.")

    feature_rows = []
    tgo_history = []
    for cycle_id in range(1, cycle["n_cycles"] + 1):
        if args.arrhenius_growth:
            # Use cycle maximum temperature in Kelvin for Arrhenius k_p.
            t_max_k = cycle["t_max"] + 273.15
            k_p_eff = args.kp0 * math.exp(-args.Q / (8.314 * t_max_k))
        else:
            k_p_eff = k_p
        # Parabolic growth once per cycle (quasi-static approximation).
        tgo_next = _growth_update(tgo_th, k_p_eff, dt_cycle)
        delta_h = max(0.0, tgo_next - tgo_th)
        # Effective eigenstrain uses initial TGO thickness as reference.
        # delta_h and h_initial are in um, so the ratio is dimensionless.
        growth_strain = args.growth_eigenstrain_coeff * (
            delta_h / max(tgo_initial, 1e-12)
        )
        tgo_th = tgo_next
        tgo_history.append(tgo_th)

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
            applied_growth = 0.0 if args.disable_growth_strain else growth_strain
            logger.info(
                "cycle=%s state=%s TGO=%.6f um growth_strain=%.6e coeff=%.3e",
                cycle_id,
                t_state,
                tgo_th,
                applied_growth,
                args.growth_eigenstrain_coeff,
            )
            pb, state, out = solve_delta_t(
                context, delta_t, growth_strain=applied_growth
            )

            if args.save_fields:
                vtk_path = os.path.join(
                    args.fields_dir,
                    f"u_snapshot_cycle_{cycle_id:04d}_{t_state}.vtk",
                )
                pb.save_state(vtk_path, state, out=out)

            features = extract_features(
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
                    "k_p_m2_s": k_p_eff,
                    "growth_strain": applied_growth,
                    "growth_eigenstrain_coeff": args.growth_eigenstrain_coeff,
                    "arrhenius_growth": bool(args.arrhenius_growth),
                    "kp0_m2_s": args.kp0 if args.arrhenius_growth else None,
                    "Q_J_mol": args.Q if args.arrhenius_growth else None,
                },
            )
            max_sigma = features["ysz_tgo_max_sigma_yy"]
            max_tau = features["ysz_tgo_max_tau_xy"]
            logger.info(
                "cycle=%s state=%s TGO=%.6f um delta_h=%.6f um eps_g=%.6e "
                "max_sigma_yy=%.6e Pa max_tau_xy=%.6e Pa",
                cycle_id,
                t_state,
                tgo_th,
                delta_h,
                applied_growth,
                max_sigma,
                max_tau,
            )
            if max(abs(max_sigma), abs(max_tau)) > 5.0e9:
                logger.warning(
                    "Warning: stress exceeds typical TBC material strength range."
                )
            feature_rows.append(features)

    _validate_growth(tgo_history)
    warnings = _check_feature_variation(feature_rows)
    if warnings:
        for warning in warnings:
            logger.warning("Feature variation check: %s", warning)
    _write_validation_plots(
        feature_rows, os.path.join("05_outputs", "validation_plots")
    )


if __name__ == "__main__":
    main()
