# Part I Verification Checks (Math)

This module performs post-processing checks on the Part I LHS dataset to verify
thermoelastic scaling and energy consistency.

## Governing Relations

Thermoelastic stress:

sigma = C : (eps - eps_th), where eps_th = alpha * DeltaT

For mismatch scaling, the characteristic stress scale is approximated by:

sigma_scale ~ E / (1 - nu) * DeltaAlpha * DeltaT

Strain energy density (SED) is:

u = 1/2 * sigma : eps_e

We use a proxy based on the maximum normal stress:

u_proxy ~ sigma_max^2 / (2 * E)

Mode-mix proxy:

psi = atan(tau / sigma)

## Outputs

- Dimensionless Pi terms:
  - Pi_sigma = sigma_max / sigma_scale
  - Pi_u = mean_sed / u_proxy
- Mode mix proxies:
  - psi = atan(tau_max / sigma_max)

Figures and summaries are saved in `04_verification/partI_verification_math/figures/`
and `04_verification/partI_verification_math/verification_summary.csv`.
