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
- `python 03_solver/run_thermal_cycles.py --cycle 00_inputs/cycle.yaml --output_csv 05_results/features/features_cyclewise.csv`

Outputs
- `05_results/features/` contains CSVs of interface metrics.
- `04_runs/thermal_cycles/` stores per-cycle meshes (optional VTK fields if enabled).
- `05_results/part1_state/` contains Part I datasets and case outputs.
- `05_results/part2_evolution/` contains Part II case runs, descriptors, and ML outputs.
- `05_results/validation_plots/` contains cyclewise diagnostic plots.
- `05_results/ml_outputs/` contains Part I ML outputs and figures.

Dissertation Workflow
Part I (state drivers, growth disabled):
- `python part1_state/run_part1_dataset.py --config part1_state/config_part1.yaml`
- `python 06_ml/run_risk_ml.py --features_dir 05_results/part1_state --include_prefix part1_state_dataset`

Part I (static ranking v2):
- Wide ΔT sweeps are reserved for verification scaling checks.
- Narrow ΔT (800–1000 °C) is used in Part I to isolate secondary drivers
  like CTE and modulus scaling without ΔT dominating the response.

Part II (evolution drivers, growth enabled):
- `python part2_evolution/run_part2_dataset.py --config part2_evolution/config_part2.yaml`
- `python part2_evolution/run_part2_ml.py`

Notes
- `k_p` is in m^2/s and converted to um^2/s internally for geometry updates.
- Roughness can be enabled at the YSZ/TGO interface in mesh generation and sweeps.