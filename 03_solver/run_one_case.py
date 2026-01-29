import numpy as np
import yaml  # (we won't use yet, placeholder)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import (
    FieldVariable,
    Material,
    Integral,
    Equation,
    Equations,
    Problem,
)
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton

from regions import layer_limits


def main():
    mesh = Mesh.from_file("02_mesh/tbc_2d.mesh")
    domain = FEDomain("domain", mesh)

    omega = domain.create_region("Omega", "all")

    # Boundaries
    left = domain.create_region("Left", "vertices in x < 1e-6", "facet")
    right = domain.create_region("Right", "vertices in x > 1999.999", "facet")
    bottom = domain.create_region("Bottom", "vertices in y < 1e-6", "facet")

    # Field: displacement (2D)
    field = Field.from_args("displacement", np.float64, "vector", omega, approx_order=1)

    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    # ---- Materials (placeholders, same for whole domain for FIRST run) ----
    E = 200e9
    nu = 0.30
    alpha = 12e-6

    # Plane strain stiffness (using sfepy isotropic convenience through lame)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    m = Material("m", lam=lam, mu=mu, alpha=alpha)

    # Thermal load: uniform DeltaT for first run
    dT = 900.0 - 25.0  # C, treated like K difference
    stress_th = (3.0 * lam + 2.0 * mu) * alpha * dT
    th_stress = np.array([[[stress_th], [stress_th], [0.0]]], dtype=np.float64)
    th = Material("th", stress=th_stress)

    integral = Integral("i", order=2)

    # Thermoelasticity weak form:
    # ∫ sigma(u):eps(v) dΩ - ∫ sigma_th:eps(v) dΩ = 0
    t1 = Term.new(
        "dw_lin_elastic_iso(m.lam, m.mu, v, u)", integral, omega, m=m, v=v, u=u
    )
    t2 = Term.new("dw_lin_prestress(th.stress, v)", integral, omega, th=th, v=v)

    eq = Equation("balance", t1 - t2)
    eqs = Equations([eq])

    # Boundary conditions: fix bottom in y, fix one point in x (avoid rigid body)
    bc_bottom = EssentialBC("bc_bottom", bottom, {"u.1": 0.0})
    bc_left = EssentialBC("bc_left", left, {"u.0": 0.0})

    pb = Problem("tbc_thermoelastic_snapshot", equations=eqs)
    pb.time_update(ebcs=Conditions([bc_bottom, bc_left]))

    pb.set_solver(Newton({}, lin_solver=ScipyDirect({})))

    state = pb.solve()

    out = state.create_output()
    pb.save_state("05_outputs/fields/u_snapshot.vtk", state)

    print("Saved: 05_outputs/fields/u_snapshot.vtk")


if __name__ == "__main__":
    main()
