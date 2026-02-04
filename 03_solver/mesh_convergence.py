import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_mesh"))

from extract_features import extract_features  # noqa: E402
from make_mesh import build_mesh  # noqa: E402
from thermoelastic_solver import build_case_context, solve_delta_t  # noqa: E402


def _parse_levels(levels_str):
    levels = []
    for item in levels_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid level '{item}'. Use 'nx,dy_scale'.")
        nx = int(parts[0])
        dy_scale = float(parts[1])
        levels.append((nx, dy_scale))
    if not levels:
        raise ValueError("No mesh levels provided.")
    return levels


def main():
    parser = argparse.ArgumentParser(description="Run mesh convergence study.")
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
        "--delta_t",
        type=float,
        required=True,
        help="Delta-T value (C) for convergence test",
    )
    parser.add_argument(
        "--levels",
        default="200,1.0;400,0.5",
        help="Semicolon-separated levels: 'nx,dy_scale;nx,dy_scale;...'",
    )
    parser.add_argument(
        "--mesh_dir",
        default=os.path.join("04_runs", "mesh_convergence"),
        help="Directory to store generated meshes",
    )
    parser.add_argument(
        "--fields_dir",
        default=os.path.join("04_runs", "mesh_convergence", "fields"),
        help="Directory to store VTK outputs",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("05_outputs", "features", "mesh_convergence.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--save_fields",
        action="store_true",
        help="Save VTK fields for each mesh level",
    )
    args = parser.parse_args()

    os.makedirs(args.mesh_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.save_fields:
        os.makedirs(args.fields_dir, exist_ok=True)

    levels = _parse_levels(args.levels)
    for nx, dy_scale in levels:
        mesh_path = os.path.join(args.mesh_dir, f"tbc_2d_nx{nx}_dy{dy_scale}.mesh")
        build_mesh(args.geometry, mesh_path, nx=nx, dy_scale=dy_scale)

        context = build_case_context(args.geometry, args.materials, mesh_path)
        pb, state, out = solve_delta_t(context, args.delta_t)

        if args.save_fields:
            vtk_path = os.path.join(
                args.fields_dir, f"u_snapshot_dT_{int(args.delta_t)}_nx{nx}.vtk"
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
                "tgo": {"lam": context["props"]["tgo"]["lam"], "mu": context["props"]["tgo"]["mu"]},
                "ysz": {"lam": context["props"]["ysz"]["lam"], "mu": context["props"]["ysz"]["mu"]},
            },
            y2=context["y2"],
            y3=context["y3"],
            output_csv=args.output_csv,
            delta_t=args.delta_t,
            extra_fields={
                "tgo_thickness_um": context["tgo_th"],
                "nx": nx,
                "dy_scale": dy_scale,
            },
        )


if __name__ == "__main__":
    main()
