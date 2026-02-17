import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "03_solver"))

from extract_features import extract_features  # noqa: E402
from thermoelastic_solver import build_case_context, solve_delta_t  # noqa: E402


def _summary(arr):
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))


def main():
    parser = argparse.ArgumentParser(
        description="Recompute Part I LHS metrics with updated interface masks."
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join(
            "05_results",
            "02_part_I_static_ranking",
            "02_lhs_dataset",
            "partI_lhs_dataset_v2.csv",
        ),
        help="Existing LHS dataset CSV (inputs)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(
            "05_results",
            "02_part_I_static_ranking",
            "02_lhs_dataset",
            "partI_lhs_dataset_v3.csv",
        ),
        help="Output CSV with updated interface metrics",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional cap on number of cases to recompute",
    )
    parser.add_argument("--n_select", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_rows = []
    n_ysz = []
    n_tgo = []
    n_ysz_nodes = []
    n_tgo_nodes = []

    if args.max_cases is not None:
        df = df.iloc[: args.max_cases].reset_index(drop=True)

    for idx, row in df.iterrows():
        case_dir = Path("05_results") / "02_part_I_static_ranking" / "02_lhs_dataset" / f"case_{idx + 1:04d}"
        geom_path = case_dir / "tbc_2d.geometry.yaml"
        mats_path = case_dir / "tbc_2d.materials.yaml"
        mesh_path = case_dir / "tbc_2d.mesh"

        if not (geom_path.exists() and mats_path.exists() and mesh_path.exists()):
            raise FileNotFoundError(
                f"Missing case files in {case_dir} (geometry/materials/mesh)."
            )

        context = build_case_context(str(geom_path), str(mats_path), str(mesh_path))
        pb, state, out = solve_delta_t(
            context, float(row["delta_t"]), growth_strain=0.0, bc_variant="fixed"
        )
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
            output_csv=os.devnull,
            delta_t=float(row["delta_t"]),
            n_select=args.n_select,
        )

        new_row = dict(row)
        new_row.update(features)
        out_rows.append(new_row)
        n_ysz.append(features.get("ysz_tgo_n_elements", np.nan))
        n_tgo.append(features.get("tgo_bc_n_elements", np.nan))
        n_ysz_nodes.append(features.get("ysz_tgo_n_nodes", np.nan))
        n_tgo_nodes.append(features.get("tgo_bc_n_nodes", np.nan))
        print(f"Recomputed case {idx + 1}/{len(df)}")

    pd.DataFrame(out_rows).to_csv(args.output_csv, index=False)

    n_ysz = np.array(n_ysz, dtype=float)
    n_tgo = np.array(n_tgo, dtype=float)
    n_ysz_nodes = np.array(n_ysz_nodes, dtype=float)
    n_tgo_nodes = np.array(n_tgo_nodes, dtype=float)
    print("Selection counts summary:")
    print(f"- ysz_tgo_n_elements min/mean/max: {_summary(n_ysz)}")
    print(f"- tgo_bc_n_elements min/mean/max: {_summary(n_tgo)}")
    print(f"- ysz_tgo_n_nodes min/mean/max: {_summary(n_ysz_nodes)}")
    print(f"- tgo_bc_n_nodes min/mean/max: {_summary(n_tgo_nodes)}")


if __name__ == "__main__":
    main()
