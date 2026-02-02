import os
import sys

import numpy as np
import yaml  # (we won't use yet, placeholder)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.base.base import Struct
from sfepy.discrete import (
    FieldVariable,
    Material,
    Integral,
    Equation,
    Equations,
    Problem,
)
from sfepy.terms import Term, Terms
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton


sys.path.append(os.path.dirname(__file__))
from regions import layer_limits
from extract_features import extract_features


def main():
    os.makedirs("05_outputs/fields", exist_ok=True)
    os.makedirs("05_outputs/features", exist_ok=True)

    mesh = Mesh.from_file("02_mesh/tbc_2d.mesh")
    domain = FEDomain("domain", mesh)

    omega = domain.create_region("Omega", "all")

    y0, y1, y2, y3, y4 = layer_limits()

    def cells_in_y(min_y, max_y, include_top=False):
        def _selector(coors, domain=None):
            ys = coors[:, 1]
            if include_top:
                return np.where((ys >= min_y) & (ys <= max_y))[0]
            return np.where((ys >= min_y) & (ys < max_y))[0]

        return _selector

    region_functions = {
        "cells_substrate": cells_in_y(y0, y1),
        "cells_bondcoat": cells_in_y(y1, y2),
        "cells_tgo": cells_in_y(y2, y3),
        "cells_ysz": cells_in_y(y3, y4, include_top=True),
    }

    substrate = domain.create_region(
        "Substrate", "cells by cells_substrate", "cell", functions=region_functions
    )
    bondcoat = domain.create_region(
        "Bondcoat", "cells by cells_bondcoat", "cell", functions=region_functions
    )
    tgo = domain.create_region(
        "TGO", "cells by cells_tgo", "cell", functions=region_functions, allow_empty=True
    )
    ysz = domain.create_region(
        "YSZ", "cells by cells_ysz", "cell", functions=region_functions
    )

    # Boundaries
    left = domain.create_region("Left", "vertices in x < 1e-6", "facet")
    right = domain.create_region("Right", "vertices in x > 1999.999", "facet")
    bottom = domain.create_region("Bottom", "vertices in y < 1e-6", "facet")

    # Field: displacement (2D)
    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)

    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    # ---- Materials (placeholders, per-layer for this run) ----
    def lame_from_E_nu(E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    lam_sub, mu_sub = lame_from_E_nu(200e9, 0.30)
    alpha_sub = 13e-6
    m_sub = Material("m_sub", lam=lam_sub, mu=mu_sub, alpha=alpha_sub)

    lam_bond, mu_bond = lame_from_E_nu(180e9, 0.30)
    alpha_bond = 14e-6
    m_bond = Material("m_bond", lam=lam_bond, mu=mu_bond, alpha=alpha_bond)

    lam_tgo, mu_tgo = lame_from_E_nu(350e9, 0.25)
    alpha_tgo = 8e-6
    m_tgo = Material("m_tgo", lam=lam_tgo, mu=mu_tgo, alpha=alpha_tgo)

    lam_ysz, mu_ysz = lame_from_E_nu(200e9, 0.23)
    alpha_ysz = 10e-6
    m_ysz = Material("m_ysz", lam=lam_ysz, mu=mu_ysz, alpha=alpha_ysz)

    def thermal_stress(lam, mu, alpha, dT):
        stress_th = (3.0 * lam + 2.0 * mu) * alpha * dT
        return np.array([[[stress_th], [stress_th], [0.0]]], dtype=np.float64)

    integral = Integral("i", order=2)

    dT_values = [600.0, 750.0, 900.0, 1050.0]
    for dT in dT_values:
        th_sub = Material("th_sub", stress=thermal_stress(lam_sub, mu_sub, alpha_sub, dT))
        th_bond = Material(
            "th_bond", stress=thermal_stress(lam_bond, mu_bond, alpha_bond, dT)
        )
        th_tgo = Material("th_tgo", stress=thermal_stress(lam_tgo, mu_tgo, alpha_tgo, dT))
        th_ysz = Material("th_ysz", stress=thermal_stress(lam_ysz, mu_ysz, alpha_ysz, dT))

        # Thermoelasticity weak form:
        # ∫ sigma(u):eps(v) dΩ - ∫ sigma_th:eps(v) dΩ = 0
        t1_terms = [
            Term.new(
                "dw_lin_elastic_iso(m_sub.lam, m_sub.mu, v, u)",
                integral,
                substrate,
                m_sub=m_sub,
                v=v,
                u=u,
            ),
            Term.new(
                "dw_lin_elastic_iso(m_bond.lam, m_bond.mu, v, u)",
                integral,
                bondcoat,
                m_bond=m_bond,
                v=v,
                u=u,
            ),
            Term.new(
                "dw_lin_elastic_iso(m_ysz.lam, m_ysz.mu, v, u)",
                integral,
                ysz,
                m_ysz=m_ysz,
                v=v,
                u=u,
            ),
        ]

        t2_terms = [
            Term.new(
                "dw_lin_prestress(th_sub.stress, v)",
                integral,
                substrate,
                th_sub=th_sub,
                v=v,
            ),
            Term.new(
                "dw_lin_prestress(th_bond.stress, v)",
                integral,
                bondcoat,
                th_bond=th_bond,
                v=v,
            ),
            Term.new(
                "dw_lin_prestress(th_ysz.stress, v)", integral, ysz, th_ysz=th_ysz, v=v
            ),
        ]

        if tgo.cells.shape[0] > 0:
            t1_terms.append(
                Term.new(
                    "dw_lin_elastic_iso(m_tgo.lam, m_tgo.mu, v, u)",
                    integral,
                    tgo,
                    m_tgo=m_tgo,
                    v=v,
                    u=u,
                )
            )
            t2_terms.append(
                Term.new(
                    "dw_lin_prestress(th_tgo.stress, v)",
                    integral,
                    tgo,
                    th_tgo=th_tgo,
                    v=v,
                )
            )

        t1 = Terms(t1_terms)
        t2 = Terms(t2_terms)

        eq = Equation("balance", t1 - t2)
        eqs = Equations([eq])

        # Boundary conditions: fix bottom in y, fix one point in x (avoid rigid body)
        bc_bottom = EssentialBC("bc_bottom", bottom, {"u.1": 0.0})
        bc_left = EssentialBC("bc_left", left, {"u.0": 0.0})

        pb = Problem(f"tbc_thermoelastic_snapshot_dT_{int(dT)}", equations=eqs)
        pb.time_update(ebcs=Conditions([bc_bottom, bc_left]))

        pb.set_solver(Newton({}, lin_solver=ScipyDirect({})))

        state = pb.solve()

        out = state.create_output()

        # Add a cell-wise layer id so ParaView can visualize the regions.
        n_cells = mesh.n_el
        layer_id = np.full(n_cells, -1, dtype=np.int32)
        layer_id[substrate.cells] = 0
        layer_id[bondcoat.cells] = 1
        if tgo.cells.shape[0] > 0:
            layer_id[tgo.cells] = 2
        layer_id[ysz.cells] = 3

        layer_id_simple = layer_id.astype(np.float64)[:, None]
        out["layer_id"] = Struct(
            name="layer_id",
            mode="cell",
            data=layer_id_simple[:, None, :, None],
            var_name="layer_id",
        )

        vtk_path = f"05_outputs/fields/u_snapshot_dT_{int(dT)}.vtk"
        pb.save_state(vtk_path, state, out=out)

        extract_features(
            pb,
            regions={
                "substrate": substrate,
                "bondcoat": bondcoat,
                "tgo": tgo,
                "ysz": ysz,
            },
            materials={
                "substrate": {"lam": lam_sub, "mu": mu_sub},
                "bondcoat": {"lam": lam_bond, "mu": mu_bond},
                "tgo": {"lam": lam_tgo, "mu": mu_tgo},
                "ysz": {"lam": lam_ysz, "mu": mu_ysz},
            },
            y2=y2,
            y3=y3,
            output_csv="05_outputs/features/sensitivity_deltaT.csv",
            delta_t=dT,
        )

        print(f"Saved: {vtk_path}")


if __name__ == "__main__":
    main()
