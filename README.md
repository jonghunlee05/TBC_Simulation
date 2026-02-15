Thermal Barrier Coating (TBC) FEM Simulation

This repository implements a 2D plane-strain, linear thermo-elastic model of a
multilayer thermal barrier coating (YSZ / TGO / bondcoat / substrate). The code
supports cycle-resolved thermal loading with oxidation-driven TGO growth and
extracts interface stress/energy metrics used as delamination proxies.

Physics Loop (Cycle-Resolved)
- Read thermal cycle metadata from `00_inputs/cycle.yaml`.
- For each cycle, update TGO thickness using parabolic growth:
  h_TGO(t + dt)^2 = h_TGO(t)^2 + k_p * dt
- Regenerate the mesh with the updated TGO thickness.
- Solve two quasi-static states per cycle: T_min and T_max (uniform temperature).
- Apply growth strain in the TGO as an isotropic eigenstrain (first-order model).
- Extract interface metrics per cycle and temperature state.

Modeling Assumptions
- Quasi-static thermal loading (no transient heat conduction).
- Uniform temperature field per solve.
- 2D plane strain.
- Linear thermo-elasticity (no creep, cracking, or delamination).
- Isotropic growth eigenstrain in TGO (first-order oxidation expansion).

How to Run
Single case (static delta-T snapshots):
- `python 03_solver/run_one_case.py --delta_t 600 900`

Multi-cycle simulation (growth + cyclewise features):
- `python 03_solver/run_thermal_cycles.py --cycle 00_inputs/cycle.yaml --output_csv 05_outputs/features/features_cyclewise.csv`

Outputs
- `05_outputs/features/` contains CSVs of interface metrics.
- `04_runs/thermal_cycles/` stores per-cycle meshes (optional VTK fields if enabled).

Sanity Checks
- `python 07_sanity_checks/check_magnitudes.py --results_dir 05_outputs/features`
- `python 07_sanity_checks/check_trends.py --results_dir 05_outputs/features`
- `python 07_sanity_checks/check_closed_form.py --delta_t 600`
- `python 07_sanity_checks/check_mesh_sensitivity.py --delta_t 600`
- `python 07_sanity_checks/run_all.py --delta_t 600 --include_mesh`

Notes
- `k_p` is in m^2/s and converted to um^2/s internally for geometry updates.
- Roughness hooks are present but not implemented yet.