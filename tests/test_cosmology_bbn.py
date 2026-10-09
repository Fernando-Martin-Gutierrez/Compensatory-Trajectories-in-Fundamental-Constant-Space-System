"""Unit tests for the Cosmology and BBN Solver."""

import pytest
import math
from multiverse.constants import UniverseParameters, ALPHA_STANDARD, ALPHA_G_STANDARD, BETA_STANDARD, ALPHA_S_STANDARD, Q_STANDARD
from multiverse.solvers.cosmology_bbn import CosmologyBBNSolver, BBNResults


def test_standard_universe_bbn_abundances():
    """Verify that the standard universe produces ~24-25% Helium-4 and ~75-76% Hydrogen."""
    solver = CosmologyBBNSolver()
    res = solver.solve_bbn(UniverseParameters.standard())

    # Check freeze-out
    assert 0.70 < res.t_freeze_out_mev < 0.90
    assert 0.15 < res.np_ratio_frozen < 0.25

    # Check post-bottleneck ratios
    assert 0.10 < res.np_ratio_bbn < 0.18

    # Check primordial mass fractions
    assert 0.22 < res.helium4_mass_fraction_yp < 0.27
    assert 0.73 < res.hydrogen_mass_fraction_xh < 0.78
    assert res.helium4_mass_fraction_yp + res.hydrogen_mass_fraction_xh == pytest.approx(1.0, rel=1e-5)

    # Check trace Deuterium
    assert 1.0e-5 < res.deuterium_abundance_dh < 5.0e-5

    # Check overall viability
    assert res.is_bbn_viable
    assert res.score_bbn_viability > 0.90


def test_diproton_bound_causes_all_helium_catastrophe():
    """If diproton binds (e.g. strong force alpha_s is significantly increased),
    100% of hydrogen is converted to helium instantaneously.
    """
    solver = CosmologyBBNSolver()
    # High alpha_s binds the diproton
    u_strong = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=ALPHA_G_STANDARD,
        beta=BETA_STANDARD,
        alpha_s=0.135,
        Q=Q_STANDARD
    )
    res = solver.solve_bbn(u_strong)

    assert res.helium4_mass_fraction_yp == 1.0
    assert res.hydrogen_mass_fraction_xh == 0.0
    assert not res.is_bbn_viable
    assert any("Diproton" in r or "All-Helium" in r for r in res.failure_reasons)


def test_unbound_deuteron_prevents_nucleosynthesis():
    """If alpha_s is low enough that deuteron cannot bind, nucleosynthesis fails."""
    solver = CosmologyBBNSolver()
    # Low alpha_s unbinds deuteron (alpha_s < ~0.09)
    u_weak = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=ALPHA_G_STANDARD,
        beta=BETA_STANDARD,
        alpha_s=0.080,
        Q=Q_STANDARD
    )
    res = solver.solve_bbn(u_weak)

    assert res.helium4_mass_fraction_yp == 0.0
    assert not res.is_bbn_viable
    assert any("Deuteron is unbound" in r for r in res.failure_reasons)


def test_gravity_coupling_scales_freeze_out():
    """Higher alpha_G accelerates cosmological expansion, increasing freeze-out temperature."""
    solver = CosmologyBBNSolver()
    u_std = UniverseParameters.standard()
    u_high_g = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=ALPHA_G_STANDARD * 100.0,
        beta=BETA_STANDARD,
        alpha_s=ALPHA_S_STANDARD,
        Q=Q_STANDARD
    )

    res_std = solver.solve_bbn(u_std)
    res_high_g = solver.solve_bbn(u_high_g)

    # T_f proportional to alpha_G^(1/6) => T_f(100*G) > T_f(G)
    assert res_high_g.t_freeze_out_mev > res_std.t_freeze_out_mev
    # Faster expansion means higher freeze-out ratio (n/p)_f
    assert res_high_g.np_ratio_frozen > res_std.np_ratio_frozen
