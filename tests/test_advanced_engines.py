"""Unit tests for the advanced ab initio physics engines:

1. Chiral Nuclear Effective Field Theory (chi-EFT) Engine:
   - Folded 1PE + 2PE chiral nuclear cluster potential
   - Pion mass m_pi(alpha_s) and decay constant f_pi(alpha_s) scaling
   - Variational Gaussian Rayleigh-Ritz eigenvalue solving
   - Hoyle resonance state shift Delta E_R and sensitivity derivatives
   - Emergent compensatory scaling slope mu_chiEFT

2. Hydrodynamic Photo-evaporative Atmospheric Escape Engine:
   - Stellar high-energy XUV irradiance evolution
   - Erkaev Roche-lobe tidal correction factor K_tide
   - Transonic energy-limited mass loss dM/dt
   - Hunten diffusion-limited bottleneck
   - Geological volatile reservoir integration (terrestrial ocean equivalents)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    UniverseParameters,
)
from multiverse.solvers.chiral_nuclear import (
    ChiralNuclearEngine,
    ChiralLowEnergyConstants,
    ChiralResonanceResult,
    ChiralPotentialComponents,
)
from multiverse.solvers.hydrodynamic_escape import (
    HydrodynamicAtmosphericEscapeEngine,
    HydrodynamicEscapeResult,
    EARTH_MASS_KG,
    EARTH_RADIUS_M,
)
from multiverse.solvers.nuclear_cluster import NuclearClusterSolver
from multiverse.solvers.planetary_habitability import PlanetaryHabitabilitySolver
from multiverse.core.manifolds import AnalyticalManifold


class TestChiralNuclearEngine:
    """Verifies the Chiral EFT variational cluster engine."""

    def test_chiral_scales_scaling(self):
        engine = ChiralNuclearEngine()
        m_pi_std, f_pi_std = engine.compute_chiral_scales(ALPHA_S_STANDARD)
        assert pytest.approx(m_pi_std, rel=1e-3) == 138.04
        assert pytest.approx(f_pi_std, rel=1e-3) == 92.4

        # Strong coupling increase should increase pion mass
        m_pi_high, f_pi_high = engine.compute_chiral_scales(ALPHA_S_STANDARD * 1.10)
        assert m_pi_high > m_pi_std
        assert f_pi_high > f_pi_std

    def test_folded_potential_components(self):
        engine = ChiralNuclearEngine()
        r_grid = np.linspace(0.1, 8.0, 100)
        pot = engine.evaluate_folded_chiral_potential(r_grid, ALPHA_STANDARD, ALPHA_S_STANDARD)

        assert isinstance(pot, ChiralPotentialComponents)
        assert pot.v_total_mev.shape == (100,)
        # Short range must be strongly repulsive due to Pauli core
        assert pot.v_core_mev[0] > 50.0
        # Medium range (~2 fm) must have significant attractive depth
        idx_2fm = np.argmin(np.abs(r_grid - 2.0))
        assert pot.v_att_chiral_mev[idx_2fm] < -10.0
        # Coulomb barrier exists
        assert pot.barrier_height_mev > 0.0

    def test_standard_universe_hoyle_resonance(self):
        engine = ChiralNuclearEngine()
        res = engine.solve_hoyle_resonance(ALPHA_STANDARD, ALPHA_S_STANDARD)

        assert isinstance(res, ChiralResonanceResult)
        assert pytest.approx(res.hoyle_resonance_energy_mev, abs=1e-4) == 0.3795
        assert pytest.approx(res.hoyle_shift_mev, abs=1e-4) == 0.0
        assert res.is_within_hoyle_window is True
        assert res.compensatory_slope_mu > 0.0

    def test_chiral_compensatory_derivatives(self):
        engine = ChiralNuclearEngine()
        res = engine.solve_hoyle_resonance(ALPHA_STANDARD, ALPHA_S_STANDARD)

        # dE/d(ln alpha) > 0 (increasing Coulomb raises resonance energy)
        assert res.dE_d_ln_alpha > 0.0
        # dE/d(ln alpha_s) < 0 (increasing nuclear attraction lowers resonance energy)
        assert res.dE_d_ln_alpha_s < 0.0

        # Emergent slope mu = - (dE/d ln alpha) / (dE/d ln alpha_s)
        expected_mu = - res.dE_d_ln_alpha / res.dE_d_ln_alpha_s
        assert pytest.approx(res.compensatory_slope_mu, rel=1e-5) == expected_mu
        assert 0.4 < res.compensatory_slope_mu < 1.2

    def test_chiral_cluster_solver_integration(self):
        solver = NuclearClusterSolver()
        shift, viable = solver.compute_chiral_eft_hoyle_resonance(ALPHA_STANDARD, ALPHA_S_STANDARD)
        assert pytest.approx(shift, abs=1e-4) == 0.0
        assert viable is True

        res = solver.solve_chiral_eft(ALPHA_STANDARD, ALPHA_S_STANDARD)
        assert isinstance(res, ChiralResonanceResult)
        assert res.is_within_hoyle_window is True

    def test_analytical_manifold_with_chiral_eft(self):
        manifold = AnalyticalManifold(use_chiral_eft=True)
        u_std = UniverseParameters.standard()
        report = manifold.evaluate(u_std)

        assert report.is_viable is True
        assert report.total_score > 0.95
        assert pytest.approx(report.hoyle_resonance_shift_mev, abs=1e-3) == 0.0


class TestHydrodynamicAtmosphericEscapeEngine:
    """Verifies the Hydrodynamic atmospheric escape and volatile evolution engine."""

    def test_stellar_xuv_evolution(self):
        engine = HydrodynamicAtmosphericEscapeEngine()
        l_bol = 3.828e26  # Solar luminosity

        # Age <= 100 Myr: saturated phase
        t_sat = 50.0 * 1e6 * 3.15576e7  # 50 Myr
        l_xuv_sat = engine.compute_stellar_xuv_luminosity(l_bol, t_sat)
        assert pytest.approx(l_xuv_sat, rel=1e-4) == 1.0e-3 * l_bol

        # Age = 1 Gyr: decayed phase
        t_1gyr = 1.0 * 1e9 * 3.15576e7
        l_xuv_1gyr = engine.compute_stellar_xuv_luminosity(l_bol, t_1gyr)
        assert l_xuv_1gyr < l_xuv_sat

    def test_erkaev_tidal_correction(self):
        engine = HydrodynamicAtmosphericEscapeEngine()
        r_p = EARTH_RADIUS_M
        m_p = EARTH_MASS_KG
        m_star = 1.98847e30
        a_orb = 1.495978707e11  # 1 AU

        r_hill, k_tide = engine.compute_tidal_correction(r_p, m_p, m_star, a_orb)
        assert r_hill > r_p
        # For Earth at 1 AU, Hill sphere is huge, so K_tide ~ 0.99
        assert 0.95 < k_tide <= 1.0

        # Close-in planet at 0.02 AU: Roche-lobe factor should decrease (enhancing escape)
        _, k_tide_close = engine.compute_tidal_correction(r_p, m_p, m_star, 0.02 * a_orb)
        assert k_tide_close < k_tide

    def test_earth_analog_hydrodynamic_escape(self):
        engine = HydrodynamicAtmosphericEscapeEngine()
        res = engine.evaluate_planetary_escape()

        assert isinstance(res, HydrodynamicEscapeResult)
        # Earth retains volatile water inventory
        assert res.is_ocean_preserved is True
        # Loses between 2 and 5 terrestrial oceans over 10 Gyr
        assert 1.0 < res.oceans_equivalent_lost < 5.0
        # Positive instantaneous mass loss
        assert res.instantaneous_mass_loss_kg_s > 0.0
        # Tidal factor close to 1
        assert res.tidal_correction_k_tide > 0.9

    def test_extreme_sub_earth_stripping(self):
        engine = HydrodynamicAtmosphericEscapeEngine()
        # Very low mass planet (0.05 Earth masses) close to star (0.2 AU)
        res = engine.evaluate_planetary_escape(
            planet_mass_earth=0.05,
            a_orb_au=0.20,
            stellar_lifetime_yr=1.0e10,
        )
        # Severe atmospheric photo-evaporation: oceans lost exceeds initial 5 oceans reservoir
        assert res.oceans_equivalent_lost > 5.0
        assert res.is_ocean_preserved is False

    def test_planetary_habitability_solver_integration(self):
        solver = PlanetaryHabitabilitySolver()
        u_std = UniverseParameters.standard()
        res = solver.evaluate_habitability(u_std)

        assert res.is_atmosphere_viable is True
        assert res.hydrodynamic_escape is not None
        assert res.hydrodynamic_escape.is_ocean_preserved is True
        assert "oceans_lost" in res.diagnostics
        assert "hydro_loss_rate_kg_s" in res.diagnostics
        assert "tidal_k_tide" in res.diagnostics
