"""Unit tests for Planetary Habitability, CHZ, and Atmospheric Retention Solver."""

import pytest
import math

from multiverse.constants import UniverseParameters
from multiverse.solvers.planetary_habitability import (
    PlanetaryHabitabilitySolver,
    HabitableZoneBoundaries,
    AtmosphericGasRetention,
    PlanetaryHabitabilityResult,
)


def test_standard_earth_habitable_zone_boundaries():
    """Verify Kasting-Kopparapu boundaries for a solar analog star."""
    chz = PlanetaryHabitabilitySolver.compute_habitable_zone(
        luminosity_solar=1.0,
        effective_temp_kelvin=5778.0,
    )
    # Conservative boundaries for 1 Solar Luminosity:
    # Runaway greenhouse ~ 0.95 AU, Maximum greenhouse ~ 1.68 AU
    assert 0.90 < chz.runaway_greenhouse_au < 1.00
    assert 1.60 < chz.maximum_greenhouse_au < 1.75
    assert chz.habitable_width_au > 0.65
    assert chz.runaway_greenhouse_au < 1.0 < chz.maximum_greenhouse_au
    assert chz.recent_venus_au < chz.runaway_greenhouse_au
    assert chz.maximum_greenhouse_au < chz.early_mars_au


def test_rocky_planet_interior_and_escape_velocity():
    """Verify terrestrial mass-radius scaling and escape velocity in standard universe."""
    u = UniverseParameters.standard()
    R_m, R_earth, g_ms2, v_esc_kms = PlanetaryHabitabilitySolver.compute_rocky_planet_structure(1.0, u)

    assert abs(R_earth - 1.0) < 0.02
    assert abs(g_ms2 - 9.81) < 0.20
    assert abs(v_esc_kms - 11.19) < 0.25

    # Test super-Earth scaling (5 M_earth)
    R_m_5, R_earth_5, g_ms2_5, v_esc_kms_5 = PlanetaryHabitabilitySolver.compute_rocky_planet_structure(5.0, u)
    assert R_earth_5 > R_earth
    assert v_esc_kms_5 > v_esc_kms
    expected_r = 5.0 ** 0.27
    assert abs(R_earth_5 - expected_r) < 0.05


def test_atmospheric_jeans_volatile_retention():
    """Verify selective Jeans escape: H2 escapes while H2O/N2/CO2 are retained."""
    u = UniverseParameters.standard()
    T_eq = 254.0  # Earth radiative equilibrium temperature
    v_esc = 11.19  # Earth escape velocity in km/s

    gases = PlanetaryHabitabilitySolver.evaluate_atmospheric_retention(T_eq, v_esc, u)

    assert "H2" in gases
    assert "H2O" in gases
    assert "N2" in gases
    assert "CO2" in gases

    # Hydrogen escapes (prevents dense runaway primordial atmosphere)
    assert not gases["H2"].is_retained_longterm
    assert gases["H2"].velocity_ratio < 6.0

    # Water is retained over geological timescales (> 4 Gyr)
    assert gases["H2O"].is_retained_longterm
    assert gases["H2O"].velocity_ratio >= 6.0

    # Nitrogen and Carbon dioxide are securely retained
    assert gases["N2"].is_retained_longterm
    assert gases["CO2"].is_retained_longterm
    assert gases["N2"].velocity_ratio > gases["H2O"].velocity_ratio
    assert gases["CO2"].velocity_ratio > gases["N2"].velocity_ratio


def test_end_to_end_habitability_standard_earth():
    """Verify comprehensive habitability result for standard Earth-Sun baseline."""
    solver = PlanetaryHabitabilitySolver()
    u = UniverseParameters.standard()

    res = solver.evaluate_habitability(u, planet_mass_earth=1.0, orbital_radius_au=1.0)

    assert res.is_in_habitable_zone
    assert res.is_water_retained
    assert res.is_hydrogen_escaped
    assert res.is_atmosphere_viable
    assert res.habitability_score > 0.95
    assert abs(res.surface_temp_k - 288.0) < 5.0


def test_weak_gravity_causes_ocean_desiccation():
    """Verify that severely reducing alpha_G causes loss of water retention."""
    import dataclasses
    solver = PlanetaryHabitabilitySolver()
    # Universe with 100x weaker gravitational coupling
    u_std = UniverseParameters.standard()
    u_weak = dataclasses.replace(u_std, alpha_G=u_std.alpha_G * 0.01)

    res = solver.evaluate_habitability(u_weak, planet_mass_earth=1.0, orbital_radius_au=1.0)

    # With 100x weaker gravity, escape velocity drops sharply
    assert res.escape_velocity_kms < 5.0
    assert not res.is_water_retained
    assert res.habitability_score == 0.0


def test_luminous_star_shifts_chz_outward():
    """Verify that a higher stellar luminosity pushes the habitable zone outward."""
    chz_bright = PlanetaryHabitabilitySolver.compute_habitable_zone(
        luminosity_solar=4.0,
        effective_temp_kelvin=7000.0,
    )
    # At L = 4 L_sun, r_CHZ ~ sqrt(4) = 2x larger
    assert chz_bright.runaway_greenhouse_au > 1.7
    assert chz_bright.maximum_greenhouse_au > 3.0
    # Planet at 1.0 AU would experience runaway greenhouse
    assert 1.0 < chz_bright.runaway_greenhouse_au

