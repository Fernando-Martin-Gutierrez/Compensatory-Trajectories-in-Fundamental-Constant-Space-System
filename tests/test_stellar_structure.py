import pytest
import math
import numpy as np

from multiverse.constants import UniverseParameters
from multiverse.solvers.stellar_structure import (
    StellarStructureSolver,
    integrate_lane_emden_rk4,
    SOLAR_MASS_KG,
)


def test_lane_emden_rk4_solutions():
    # Polytrope n = 1.5 known exact: xi_1 ~ 3.6538, omega ~ 2.714
    xi_arr_15, th_arr_15, xi_1_15, omega_15 = integrate_lane_emden_rk4(1.5, dxi=1e-4)
    assert math.isclose(xi_1_15, 3.6538, abs_tol=0.01)
    assert math.isclose(omega_15, 2.714, abs_tol=0.02)

    # Polytrope n = 3.0 known exact: xi_1 ~ 6.8968, omega ~ 2.018
    xi_arr_30, th_arr_30, xi_1_30, omega_30 = integrate_lane_emden_rk4(3.0, dxi=1e-4)
    assert math.isclose(xi_1_30, 6.8968, abs_tol=0.02)
    assert math.isclose(omega_30, 2.018, abs_tol=0.02)


def test_stellar_stability_window():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    res = solver.evaluate_stellar_viability(u)

    # In our universe: Chandrasekhar ~ 1.44 Msun, min ignition ~ 0.08 Msun
    assert math.isclose(res.chandrasekhar_mass_solar, 1.44, rel_tol=0.05)
    assert math.isclose(res.min_ignition_mass_solar, 0.08, rel_tol=0.05)
    assert res.has_stable_nuclear_burning is True
    assert res.mass_window_ratio > 10.0


def test_gamow_tunneling_parameters():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    # Solar core temperature T = 1.57e7 K
    E_G_kev, E_0_kev, delta_E_kev, tau_pp = solver.compute_gamow_parameters(1.57e7, u)

    # Standard physics: E_G ~ 493.2 keV, E_0 ~ 6.09 keV, delta_E ~ 6.63 keV, tau_pp ~ 13.5
    assert math.isclose(E_G_kev, 493.18, rel_tol=0.01)
    assert math.isclose(E_0_kev, 6.09, rel_tol=0.02)
    assert math.isclose(delta_E_kev, 6.63, rel_tol=0.02)
    assert math.isclose(tau_pp, 13.50, rel_tol=0.02)


def test_nuclear_generation_rates():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()

    # At solar core (rho = 1.5e5 kg/m^3, T = 1.57e7 K)
    eps_pp, eps_cno, eps_tot = solver.compute_nuclear_generation_rates(1.5e5, 1.57e7, u)
    assert math.isclose(eps_pp, 0.017, rel_tol=0.05)
    assert eps_pp > eps_cno  # pp chain dominates in Sun
    assert math.isclose(eps_tot, eps_pp + eps_cno, rel_tol=1e-5)

    # At massive star core (T = 3.5e7 K): CNO dominates due to steep temperature sensitivity
    eps_pp_hot, eps_cno_hot, _ = solver.compute_nuclear_generation_rates(1.5e5, 3.5e7, u)
    assert eps_cno_hot > eps_pp_hot


def test_solar_analog_equilibrium():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    res = solver.evaluate_stellar_viability(u)

    assert res.solar_analog is not None
    star = res.solar_analog

    # Standard solar calibrations
    assert math.isclose(star.stellar_mass_solar, 1.0, rel_tol=0.01)
    assert 0.8 <= star.radius_solar <= 1.2
    assert 0.5 <= star.luminosity_solar <= 2.0
    assert 1.2e7 <= star.central_temp_kelvin <= 2.0e7
    assert 5000.0 <= star.effective_temp_kelvin <= 7000.0
    assert star.pp_luminosity_fraction > 0.8
    assert star.main_sequence_lifetime_gyr > 5.0
    assert star.spectral_class in ["F", "G"]


def test_stellar_population_spectrum():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    res = solver.evaluate_stellar_viability(u)

    assert len(res.stellar_population) >= 4
    masses = [s.stellar_mass_solar for s in res.stellar_population]
    assert sorted(masses) == masses  # strictly increasing masses

    # Luminosity must increase monotonically with mass (L ~ M^3 - M^4)
    lums = [s.luminosity_solar for s in res.stellar_population]
    assert all(lums[i] < lums[i + 1] for i in range(len(lums) - 1))


def test_stellar_interior_1d_profile():
    solver = StellarStructureSolver()
    u = UniverseParameters.standard()
    star = solver.solve_stellar_equilibrium(mass_solar=1.0, u=u, compute_profile=True)

    assert star.profile_1d is not None
    prof = star.profile_1d

    assert len(prof.r_array_m) == 250
    # Temperature and density must decrease monotonically outward
    assert prof.temperature_kelvin[0] > prof.temperature_kelvin[-1]
    assert prof.density_kg_m3[0] > prof.density_kg_m3[-1]
    assert np.all(np.diff(prof.temperature_kelvin) <= 0.0)
    assert np.all(np.diff(prof.density_kg_m3) <= 0.0)

    # Enclosed mass must start near 0 and reach stellar mass
    assert prof.enclosed_mass_kg[0] < 0.01 * star.stellar_mass_kg
    assert math.isclose(prof.enclosed_mass_kg[-1], star.stellar_mass_kg, rel_tol=0.02)
