"""Unit tests for Phase 2 Physical Depth: ZAMS 1D and Radial Hartree-Fock SCF."""

import math
import numpy as np
import pytest

from multiverse.constants import UniverseParameters, ALPHA_STANDARD, ALPHA_S_STANDARD
from multiverse.solvers.stellar_structure import (
    StellarStructureSolver,
    solve_lane_emden,
    ZAMSSolution,
)
from multiverse.solvers.quantum_multielectron import (
    RadialHartreeFockSCF,
    HartreeFockAtomResult,
    BornOppenheimerMetric,
)
from multiverse.core.manifolds import evaluate_universe_viability, AnalyticalManifold


def test_zams_solar_convergence():
    """Verify that 1D ZAMS solver converges for 1.0 M_sun with Tc in [1.3e7, 1.7e7] K."""
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    sol = solver.solve_zams(mass_solar=1.0, u=u)

    assert sol.converged is True
    # Target central temperature criterion: Tc in [1.3e7, 1.7e7] K
    assert 1.3e7 <= sol.central_temp_kelvin <= 1.7e7
    # Interior density and pressure sanity checks
    assert sol.central_density_kg_m3 > 1.0e4
    assert sol.central_pressure_pa > 1.0e15
    # Radial profile arrays must be populated
    assert len(sol.r_grid) > 20
    assert len(sol.m_grid) == len(sol.r_grid)
    assert len(sol.t_grid) == len(sol.r_grid)
    # Surface temperature and luminosity sanity
    assert 2000.0 <= sol.effective_temp_kelvin <= 30000.0
    assert 0.05 <= sol.luminosity_solar <= 20.0


def test_hartree_fock_helium_ground_state():
    """Verify that radial Hartree-Fock SCF gets Helium ground state within -79.0 +/- 1.5 eV."""
    hf = RadialHartreeFockSCF(n_points=1200, max_iter=30)
    u = UniverseParameters.standard()
    res = hf.solve_helium(u)

    assert res.scf_converged is True
    # Requirement: -79.0 eV +/- 1.5 eV (i.e. [-80.5, -77.5] eV)
    assert math.isclose(res.ground_state_energy_ev, -79.0, abs_tol=1.5)
    assert res.ground_state_energy_ev < 0.0
    # Orbital energy epsilon_1s should be roughly -24 eV (-0.9 Hartree)
    assert "1s" in res.orbital_energies_ev
    assert -30.0 <= res.orbital_energies_ev["1s"] <= -20.0
    # Coulomb repulsion J_ee > 0
    assert res.coulomb_repulsion_jee_ev > 25.0
    # Born-Oppenheimer metric verification
    assert res.born_oppenheimer.is_adiabatic_valid is True
    assert res.born_oppenheimer.kappa_bo < 0.25


def test_hartree_fock_lithium_ground_state():
    """Verify radial Hartree-Fock SCF for Lithium (Z=3)."""
    hf = RadialHartreeFockSCF(n_points=1000, max_iter=25)
    u = UniverseParameters.standard()
    res = hf.solve_lithium(u)

    assert res.scf_converged is True
    assert res.atomic_number_z == 3
    # Lithium experimental non-relativistic total energy ~ -203.5 eV
    assert math.isclose(res.ground_state_energy_ev, -203.48, rel_tol=0.05)
    assert "1s" in res.orbital_energies_ev
    assert "2s" in res.orbital_energies_ev
    assert res.orbital_energies_ev["1s"] < res.orbital_energies_ev["2s"]


def test_born_oppenheimer_hierarchy_bound():
    """Verify adiabatic parameter kappa_BO = beta^(1/4) < 0.25 and breakdown when beta > 0.0039."""
    u_std = UniverseParameters.standard()
    bo_std = RadialHartreeFockSCF.evaluate_born_oppenheimer(u_std)

    # Standard universe: beta ~ 5.446e-4 => kappa_BO ~ 0.1528 < 0.25
    assert math.isclose(bo_std.kappa_bo, 0.1528, abs_tol=0.01)
    assert bo_std.is_adiabatic_valid is True

    # Heavy electron universe: beta = 0.005 > 1/256 (~0.0039)
    u_heavy = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=u_std.alpha_G,
        beta=0.005,
        alpha_s=ALPHA_S_STANDARD,
        Q=u_std.Q,
    )
    bo_heavy = RadialHartreeFockSCF.evaluate_born_oppenheimer(u_heavy)
    assert bo_heavy.kappa_bo > 0.25
    assert bo_heavy.is_adiabatic_valid is False

    # Check analytical manifold penalty
    rep = evaluate_universe_viability(u_heavy)
    assert rep.score_born_oppenheimer < 0.05
    assert rep.is_viable is False
    assert any("Born-Oppenheimer" in r for r in rep.failure_reasons)


def test_lane_emden_backward_compatibility():
    """Verify backward compatibility of solve_lane_emden."""
    # Top-level helper function
    xi_arr, th_arr, xi_1, omega = solve_lane_emden(1.5)
    assert math.isclose(xi_1, 3.6538, abs_tol=0.02)
    assert math.isclose(omega, 2.714, abs_tol=0.02)

    # Method on StellarStructureSolver
    solver = StellarStructureSolver()
    xi_arr_m, th_arr_m, xi_1_m, omega_m = solver.solve_lane_emden(3.0)
    assert math.isclose(xi_1_m, 6.8968, abs_tol=0.03)
    assert math.isclose(omega_m, 2.018, abs_tol=0.03)
