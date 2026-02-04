import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import extract_features
from thermoelastic_solver import build_case_context, solve_delta_t


def main():
    parser = argparse.ArgumentParser(description="Run one thermoelastic case.")
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
        "--mesh",
        default=os.path.join("02_mesh", "tbc_2d.mesh"),
        help="Mesh path",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        nargs="+",
        default=None,
        help="One or more delta-T values (C)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("05_outputs", "features", "sensitivity_deltaT.csv"),
        help="CSV output path for interface features",
    )
    parser.add_argument(
        "--fields_dir",
        default=os.path.join("05_outputs", "fields"),
        help="Directory for VTK field outputs",
    )
    parser.add_argument(
        "--n_select",
        type=int,
        default=200,
        help="Number of elements nearest each interface to sample",
    )
    args = parser.parse_args()

    os.makedirs(args.fields_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    context = build_case_context(args.geometry, args.materials, args.mesh)
    dT_values = args.delta_t or [600.0, 750.0, 900.0, 1050.0]
    for dT in dT_values:
        pb, state, out = solve_delta_t(context, dT)
        tgo_th = context["tgo_th"]

        vtk_path = os.path.join(args.fields_dir, f"u_snapshot_dT_{int(dT)}.vtk")
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
                "tgo": {"lam": context["props"]["tgo"]["lam"], "mu": context["props"]["tgo"]["mu"]},
                "ysz": {"lam": context["props"]["ysz"]["lam"], "mu": context["props"]["ysz"]["mu"]},
            },
            y2=context["y2"],
            y3=context["y3"],
            output_csv=args.output_csv,
            delta_t=dT,
            n_select=args.n_select,
            extra_fields={"tgo_thickness_um": tgo_th},
        )

        scale_sub = context["props"]["substrate"]["E"] * context["props"]["substrate"]["alpha"] * dT
        scale_bond = context["props"]["bondcoat"]["E"] * context["props"]["bondcoat"]["alpha"] * dT
        scale_tgo = context["props"]["tgo"]["E"] * context["props"]["tgo"]["alpha"] * dT
        scale_ysz = context["props"]["ysz"]["E"] * context["props"]["ysz"]["alpha"] * dT
        scale_max = max(scale_sub, scale_bond, scale_tgo, scale_ysz)
        ysz_tgo_sigma = features["ysz_tgo_max_sigma_yy"]
        tgo_bc_sigma = features["tgo_bc_max_sigma_yy"]

        print(
            "Sanity dT={:.1f}C | E*a*dT GPa: sub={:.2f}, bond={:.2f}, tgo={:.2f}, ysz={:.2f} "
            "| sigma_yy_max GPa: ysz_tgo={:.2f}, tgo_bc={:.2f} | ratios to max scale: {:.2f}, {:.2f}".format(
                dT,
                scale_sub / 1e9,
                scale_bond / 1e9,
                scale_tgo / 1e9,
                scale_ysz / 1e9,
                ysz_tgo_sigma / 1e9,
                tgo_bc_sigma / 1e9,
                ysz_tgo_sigma / scale_max,
                tgo_bc_sigma / scale_max,
            )
        )

        print(f"Saved: {vtk_path}")


if __name__ == "__main__":
    main()
