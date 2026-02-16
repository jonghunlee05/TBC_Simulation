import os

import numpy as np
import yaml
from sfepy.base.base import Struct
from sfepy.discrete import (
    Equation,
    Equations,
    FieldVariable,
    Integral,
    Material,
    Problem,
)
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term, Terms

from regions import layer_limits


def _find_layer_thickness(layers, name_fragment):
    for name, thickness in layers.items():
        if name_fragment.lower() in name.lower():
            return thickness
    raise KeyError(f"Layer with name containing '{name_fragment}' not found.")


def _load_geometry(geometry_path):
    with open(geometry_path, "r", encoding="utf-8") as f:
        geom_spec = yaml.safe_load(f)

    layers = {layer["name"]: float(layer["thickness_um"]) for layer in geom_spec["layers"]}
    width = float(geom_spec["domain"]["width_um"])
    thicknesses = {
        "ysz": _find_layer_thickness(layers, "ysz"),
        "tgo": _find_layer_thickness(layers, "tgo"),
        "bond": _find_layer_thickness(layers, "bond"),
        "sub": _find_layer_thickness(layers, "substrate"),
    }
    return width, thicknesses


def _load_materials(materials_path):
    with open(materials_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["materials"]


def _lame_from_E_nu(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def _thermal_stress(lam, mu, alpha, dT):
    stress_th = (3.0 * lam + 2.0 * mu) * alpha * dT
    return np.array([[[stress_th], [stress_th], [0.0]]], dtype=np.float64)


def _cells_in_y(min_y, max_y, include_top=False):
    def _selector(coors, domain=None):
        ys = coors[:, 1]
        if include_top:
            return np.where((ys >= min_y) & (ys <= max_y))[0]
        return np.where((ys >= min_y) & (ys < max_y))[0]

    return _selector


def build_case_context(geometry_path, materials_path, mesh_path, tol=1e-3):
    width, thicknesses = _load_geometry(geometry_path)
    mats_spec = _load_materials(materials_path)

    mesh = Mesh.from_file(mesh_path)
    coors = mesh.coors
    x_min, x_max = float(coors[:, 0].min()), float(coors[:, 0].max())
    y_min, y_max = float(coors[:, 1].min()), float(coors[:, 1].max())
    exp_height = (
        thicknesses["sub"] + thicknesses["bond"] + thicknesses["tgo"] + thicknesses["ysz"]
    )
    if abs(x_min) > tol or abs(y_min) > tol:
        raise ValueError(
            f"Mesh origin mismatch: min coords ({x_min:.6g}, {y_min:.6g}) not near 0."
        )
    if abs(x_max - width) > tol:
        raise ValueError(
            f"Mesh width mismatch: mesh x_max={x_max:.6g} vs width={width:.6g}."
        )
    if abs(y_max - exp_height) > tol:
        raise ValueError(
            f"Mesh height mismatch: mesh y_max={y_max:.6g} vs height={exp_height:.6g}."
        )
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all")

    y0, y1, y2, y3, y4 = layer_limits(
        ysz=thicknesses["ysz"],
        tgo=thicknesses["tgo"],
        bond=thicknesses["bond"],
        sub=thicknesses["sub"],
    )

    region_functions = {
        "cells_substrate": _cells_in_y(y0, y1),
        "cells_bondcoat": _cells_in_y(y1, y2),
        "cells_tgo": _cells_in_y(y2, y3),
        "cells_ysz": _cells_in_y(y3, y4, include_top=True),
    }

    regions = {
        "substrate": domain.create_region(
            "Substrate", "cells by cells_substrate", "cell", functions=region_functions
        ),
        "bondcoat": domain.create_region(
            "Bondcoat", "cells by cells_bondcoat", "cell", functions=region_functions
        ),
        "tgo": domain.create_region(
            "TGO",
            "cells by cells_tgo",
            "cell",
            functions=region_functions,
            allow_empty=True,
        ),
        "ysz": domain.create_region(
            "YSZ", "cells by cells_ysz", "cell", functions=region_functions
        ),
    }

    boundaries = {
        "left": domain.create_region("Left", "vertices in x < 1e-6", "facet"),
        "right": domain.create_region("Right", f"vertices in x > {width - 1e-6}", "facet"),
        "bottom": domain.create_region("Bottom", "vertices in y < 1e-6", "facet"),
        "origin": domain.create_region(
            "Origin", "vertices in (x < 1e-6) & (y < 1e-6)", "vertex"
        ),
    }

    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    def _mat_props(key):
        mat = mats_spec[key]
        return float(mat["E_GPa"]) * 1e9, float(mat["nu"]), float(mat["alpha_1K"])

    E_sub, nu_sub, alpha_sub = _mat_props("substrate")
    lam_sub, mu_sub = _lame_from_E_nu(E_sub, nu_sub)
    E_bond, nu_bond, alpha_bond = _mat_props("bondcoat")
    lam_bond, mu_bond = _lame_from_E_nu(E_bond, nu_bond)
    E_tgo, nu_tgo, alpha_tgo = _mat_props("TGO_Al2O3")
    lam_tgo, mu_tgo = _lame_from_E_nu(E_tgo, nu_tgo)
    E_ysz, nu_ysz, alpha_ysz = _mat_props("YSZ")
    lam_ysz, mu_ysz = _lame_from_E_nu(E_ysz, nu_ysz)

    materials = {
        "m_sub": Material("m_sub", lam=lam_sub, mu=mu_sub, alpha=alpha_sub),
        "m_bond": Material("m_bond", lam=lam_bond, mu=mu_bond, alpha=alpha_bond),
        "m_tgo": Material("m_tgo", lam=lam_tgo, mu=mu_tgo, alpha=alpha_tgo),
        "m_ysz": Material("m_ysz", lam=lam_ysz, mu=mu_ysz, alpha=alpha_ysz),
    }

    return {
        "mesh": mesh,
        "domain": domain,
        "regions": regions,
        "boundaries": boundaries,
        "field": field,
        "u": u,
        "v": v,
        "integral": Integral("i", order=2),
        "y2": y2,
        "y3": y3,
        "tgo_th": thicknesses["tgo"],
        "ysz_th": thicknesses["ysz"],
        "bond_th": thicknesses["bond"],
        "sub_th": thicknesses["sub"],
        "props": {
            "substrate": {"E": E_sub, "alpha": alpha_sub, "lam": lam_sub, "mu": mu_sub},
            "bondcoat": {"E": E_bond, "alpha": alpha_bond, "lam": lam_bond, "mu": mu_bond},
            "tgo": {"E": E_tgo, "alpha": alpha_tgo, "lam": lam_tgo, "mu": mu_tgo},
            "ysz": {"E": E_ysz, "alpha": alpha_ysz, "lam": lam_ysz, "mu": mu_ysz},
        },
        "materials": materials,
    }


def solve_delta_t(context, dT, growth_strain=0.0, bc_variant="fixed"):
    """
    Solve a uniform temperature step with optional TGO growth eigenstrain.

    Growth strain is treated as an isotropic eigenstrain in the TGO. This is a
    first-order approximation that captures volumetric expansion due to oxidation
    without modeling full inelasticity or damage.

    Boundary condition variants:
    - "fixed": left edge u_x=0 and bottom edge u_y=0 (default)
    - "free_edge": only pin the origin to remove rigid body motion
    """
    regions = context["regions"]
    u = context["u"]
    v = context["v"]
    integral = context["integral"]

    props = context["props"]
    materials = context["materials"]

    th_sub = Material(
        "th_sub",
        stress=_thermal_stress(props["substrate"]["lam"], props["substrate"]["mu"], props["substrate"]["alpha"], dT),
    )
    th_bond = Material(
        "th_bond",
        stress=_thermal_stress(props["bondcoat"]["lam"], props["bondcoat"]["mu"], props["bondcoat"]["alpha"], dT),
    )
    th_tgo = Material(
        "th_tgo",
        stress=_thermal_stress(
            props["tgo"]["lam"], props["tgo"]["mu"], props["tgo"]["alpha"], dT
        ),
    )
    th_ysz = Material(
        "th_ysz",
        stress=_thermal_stress(props["ysz"]["lam"], props["ysz"]["mu"], props["ysz"]["alpha"], dT),
    )

    t1_terms = [
        Term.new(
            "dw_lin_elastic_iso(m_sub.lam, m_sub.mu, v, u)",
            integral,
            regions["substrate"],
            m_sub=materials["m_sub"],
            v=v,
            u=u,
        ),
        Term.new(
            "dw_lin_elastic_iso(m_bond.lam, m_bond.mu, v, u)",
            integral,
            regions["bondcoat"],
            m_bond=materials["m_bond"],
            v=v,
            u=u,
        ),
        Term.new(
            "dw_lin_elastic_iso(m_ysz.lam, m_ysz.mu, v, u)",
            integral,
            regions["ysz"],
            m_ysz=materials["m_ysz"],
            v=v,
            u=u,
        ),
    ]

    t2_terms = [
        Term.new(
            "dw_lin_prestress(th_sub.stress, v)",
            integral,
            regions["substrate"],
            th_sub=th_sub,
            v=v,
        ),
        Term.new(
            "dw_lin_prestress(th_bond.stress, v)",
            integral,
            regions["bondcoat"],
            th_bond=th_bond,
            v=v,
        ),
        Term.new(
            "dw_lin_prestress(th_ysz.stress, v)",
            integral,
            regions["ysz"],
            th_ysz=th_ysz,
            v=v,
        ),
    ]

    if regions["tgo"].cells.shape[0] > 0:
        t1_terms.append(
            Term.new(
                "dw_lin_elastic_iso(m_tgo.lam, m_tgo.mu, v, u)",
                integral,
                regions["tgo"],
                m_tgo=materials["m_tgo"],
                v=v,
                u=u,
            )
        )
        t2_terms.append(
            Term.new(
                "dw_lin_prestress(th_tgo.stress, v)",
                integral,
                regions["tgo"],
                th_tgo=th_tgo,
                v=v,
            )
        )
        if growth_strain > 0.0:
            # Growth eigenstrain modeled as isotropic prestress in the TGO.
            # This keeps the model linear while capturing oxidation expansion.
            gr_tgo = Material(
                "gr_tgo",
                stress=_thermal_stress(
                    props["tgo"]["lam"], props["tgo"]["mu"], growth_strain, 1.0
                ),
            )
            t2_terms.append(
                Term.new(
                    "dw_lin_prestress(gr_tgo.stress, v)",
                    integral,
                    regions["tgo"],
                    gr_tgo=gr_tgo,
                    v=v,
                )
            )

    eq = Equation("balance", Terms(t1_terms) - Terms(t2_terms))
    eqs = Equations([eq])

    bc_bottom = EssentialBC("bc_bottom", context["boundaries"]["bottom"], {"u.1": 0.0})
    bc_left = EssentialBC("bc_left", context["boundaries"]["left"], {"u.0": 0.0})
    bc_origin = EssentialBC(
        "bc_origin", context["boundaries"]["origin"], {"u.0": 0.0, "u.1": 0.0}
    )

    pb = Problem(f"tbc_thermoelastic_snapshot_dT_{int(dT)}", equations=eqs)
    if bc_variant == "free_edge":
        ebcs = Conditions([bc_origin])
    else:
        ebcs = Conditions([bc_bottom, bc_left])
    pb.time_update(ebcs=ebcs)
    pb.set_solver(Newton({}, lin_solver=ScipyDirect({})))
    state = pb.solve()

    out = state.create_output()

    mesh = context["mesh"]
    n_cells = mesh.n_el
    layer_id = np.full(n_cells, -1, dtype=np.int32)
    layer_id[regions["substrate"].cells] = 0
    layer_id[regions["bondcoat"].cells] = 1
    if regions["tgo"].cells.shape[0] > 0:
        layer_id[regions["tgo"].cells] = 2
    layer_id[regions["ysz"].cells] = 3

    layer_id_simple = layer_id.astype(np.float64)[:, None]
    out["layer_id"] = Struct(
        name="layer_id",
        mode="cell",
        data=layer_id_simple[:, None, :, None],
        var_name="layer_id",
    )

    return pb, state, out
