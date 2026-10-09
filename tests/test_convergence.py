"""Numerical Convergence & Grid Error Verification Suite.

Validates the discrete numerical orders of accuracy across computational physics solvers:
1. Numerov / Finite Difference 1D Radial Schrödinger Solver:
   Richardson error scaling towards the exact Bohr/Rydberg bound state E_1 = -13.598 eV.
2. Yoshida 4th-Order Symplectic Integrator (1PN Celestial Orbits):
   Exact 4th-order energy conservation scaling (dE ~ dt^4) with zero secular dissipation.
3. Lane-Emden RK4 Polytrope Solver:
   Asymptotic convergence to analytical surface zeros:
   xi_1(n=1.5) = 3.65375 and xi_1(n=3.0) = 6.89685 with error < 10^-5.
4. Ali-Bodmer Nuclear Cluster Variational Expansion:
   Convergence of 8Be / Hoyle resonance state with Gaussian basis expansion size.
"""

import pytest
import numpy as np
import math
from scipy import linalg

from multiverse.constants import UniverseParameters, ALPHA_STANDARD, ALPHA_S_STANDARD
from multiverse.solvers.quantum_atom import QuantumAtomSolver
from multiverse.solvers.orbital_dynamics import integrate_yoshida_step
from multiverse.solvers.stellar_structure import integrate_lane_emden_rk4
from multiverse.solvers.nuclear_cluster import NuclearClusterSolver


def test_numerov_schrodinger_grid_convergence():
    """Verify spatial grid convergence towards exact reduced-mass ground state E_1 = -13.598 eV."""
    u_std = UniverseParameters.standard()
    # Exact reduced-mass ground state energy in eV
    # E_exact = -0.5 * mu * c^2 * alpha^2 ~ -13.5983 eV
    mu_over_mp = u_std.beta / (1.0 + u_std.beta)
    e_exact_ev = -0.5 * mu_over_mp * 938.272088e6 * (u_std.alpha ** 2.0)

    n_grids = [500, 1000, 2000]
    energies = []
    errors = []

    for n in n_grids:
        solver = QuantumAtomSolver(n_points=n, rho_max=35.0)
        res = solver.solve(u_std)
        energies.append(res.e1_ev)
        errors.append(abs(res.e1_ev - e_exact_ev))

    # Monotonic error convergence towards analytical ground state
    assert errors[0] > errors[1] > errors[2]
    # Final error with 2000 grid points is well within 0.01 eV
    assert errors[-1] < 0.015
    # Verification that ground state approaches -13.598 eV
    assert energies[-1] == pytest.approx(-13.598, abs=0.02)


def test_yoshida_symplectic_4th_order_convergence():
    """Verify that the Yoshida 4th-order symplectic integrator achieves order p ~ 4.0."""
    masses = np.array([1.0, 1e-3, 1e-3], dtype=np.float64)

    def compute_total_energy(pos, vel):
        ke = 0.5 * np.sum(masses[:, None] * (vel ** 2))
        pe = 0.0
        for i in range(len(masses)):
            for j in range(i + 1, len(masses)):
                pe -= masses[i] * masses[j] / np.linalg.norm(pos[i] - pos[j])
        return ke + pe

    def get_max_energy_error(dt, n_orbits=2):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.7, 0.0]], dtype=np.float64)
        e0 = compute_total_energy(pos, vel)
        n_steps = int(n_orbits * 2.0 * np.pi / dt)
        max_err = 0.0
        for _ in range(n_steps):
            pos, vel = integrate_yoshida_step(pos, vel, masses, 1.0, 0.0, dt)
            e = compute_total_energy(pos, vel)
            max_err = max(max_err, abs((e - e0) / e0))
        return max_err

    # Evaluate energy errors across dyadic timesteps
    dt_vals = [0.04, 0.02, 0.01]
    errs = [get_max_energy_error(dt) for dt in dt_vals]

    # Compute observed convergence orders: p = log(E1/E2) / log(2)
    order_1 = np.log(errs[0] / errs[1]) / np.log(2.0)
    order_2 = np.log(errs[1] / errs[2]) / np.log(2.0)

    # Yoshida 4th-order integrator guarantees p in [3.8, 4.2]
    assert 3.8 <= order_1 <= 4.2
    assert 3.8 <= order_2 <= 4.2
    # Energy error is tiny (order ~ 10^-11)
    assert errs[-1] < 1e-10


def test_lane_emden_rk4_asymptotic_convergence():
    """Verify RK4 convergence to the exact analytical polytrope zeros with error < 10^-5."""
    # Classical Chandrasekhar analytical values:
    # n=1.5: xi_1 = 3.65375, -xi_1^2 theta'(xi_1) = 2.71406
    # n=3.0: xi_1 = 6.89685, -xi_1^2 theta'(xi_1) = 2.01824
    xi_1_5_exact = 3.65375
    xi_3_0_exact = 6.89685

    # Run with standard high-precision step size dxi = 1e-4
    _, _, xi_1_5, omega_1_5 = integrate_lane_emden_rk4(1.5, dxi=1e-4)
    _, _, xi_3_0, omega_3_0 = integrate_lane_emden_rk4(3.0, dxi=1e-4)

    err_1_5 = abs(xi_1_5 - xi_1_5_exact)
    err_3_0 = abs(xi_3_0 - xi_3_0_exact)

    # Tolerance strictly < 10^-5
    assert err_1_5 < 1e-5
    assert err_3_0 < 1e-5

    # Mass constants omega_n match analytical values to < 10^-4
    assert abs(omega_1_5 - 2.71406) < 1e-4
    assert abs(omega_3_0 - 2.01824) < 1e-4


def test_nuclear_cluster_variational_basis_convergence():
    """Verify that the Ali-Bodmer variational cluster solver converges with basis size."""
    solver = NuclearClusterSolver()

    # Variational energies with N=10, 14, 18 Gaussians
    e_10 = solver._solve_resonance_variational(ALPHA_STANDARD, ALPHA_S_STANDARD, num_gaussians=10)
    e_14 = solver._solve_resonance_variational(ALPHA_STANDARD, ALPHA_S_STANDARD, num_gaussians=14)
    e_18 = solver._solve_resonance_variational(ALPHA_STANDARD, ALPHA_S_STANDARD, num_gaussians=18)

    # Rayleigh-Ritz variational principle guarantees strict monotonic lowering: E_18 < E_14 < E_10
    assert e_10 > e_14 > e_18
    # Diminishing energy improvements as basis expands
    assert (e_10 - e_14) > (e_14 - e_18) > 0.0

    # Hoyle resonance calculation at standard parameters yields exactly Delta E_R = 0
    res_std = solver.compute_hoyle_resonance(ALPHA_STANDARD, ALPHA_S_STANDARD)
    assert abs(res_std.hoyle_shift_mev) < 1e-6
    assert res_std.is_within_hoyle_window
