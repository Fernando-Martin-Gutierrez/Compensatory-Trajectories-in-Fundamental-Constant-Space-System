"""Circumstellar Habitable Zone (CHZ), Planetary Geophysics, and Atmospheric Retention Solver.

Models the physical prerequisites for planetary habitability across varying fundamental constants:
  1. Kasting-Kopparapu photoclimatic circumstellar habitable zone:
         S_eff(T_*) = S_eff0 + a (T_* - 5780) + b (T_* - 5780)^2 + c (T_* - 5780)^3
         r_CHZ = sqrt( (L_* / L_sun) / S_eff ) AU
     determining runaway greenhouse (inner edge) and maximum CO2 greenhouse (outer edge).
  2. Terrestrial rocky planet mass-radius equation of state and surface gravity:
         R_p ~ R_earth * (M_p / M_earth)^0.27 * (alpha0 / alpha) * (beta0 / beta) * (alpha_G0 / alpha_G)^0.05
         v_esc = sqrt( 2 G M_p / R_p )
  3. Atmospheric Jeans thermal escape and volatile retention:
         v_th(species) = sqrt( 3 k_B T_eq / (mu m_p) )
         lambda_J = v_esc^2 / (2 v_th^2)
     Requires v_esc / v_th >= 6.0 for biogenic volatiles (H2O, N2, CO2, O2)
     while allowing H2 escape to avoid runaway hydrogen gas envelope accretion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    PROTON_MASS_SI,
    GRAVITATIONAL_CONSTANT_SI,
    SPEED_OF_LIGHT_SI,
    HBAR_SI,
    UniverseParameters,
)
from multiverse.solvers.stellar_structure import (
    SOLAR_LUMINOSITY_W,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    BOLTZMANN_K_SI,
    STEFAN_BOLTZMANN_SI,
    StellarEquilibriumSolution,
)

# Reference Astronomical & Geophyical Constants in SI
EARTH_MASS_KG: float = 5.9722e24
EARTH_RADIUS_M: float = 6.371e6
ASTRONOMICAL_UNIT_M: float = 1.495978707e11   # 1 AU in meters
EARTH_ALBEDO_BOND: float = 0.306             # Bond albedo of Earth
EARTH_GREENHOUSE_DT_K: float = 33.0          # Natural greenhouse temperature boost


@dataclass
class HabitableZoneBoundaries:
    """Circumstellar Habitable Zone (CHZ) orbital radii in AU and meters."""
    recent_venus_au: float           # Optimistic inner edge
    runaway_greenhouse_au: float     # Conservative inner edge
    maximum_greenhouse_au: float     # Conservative outer edge
    early_mars_au: float             # Optimistic outer edge
    habitable_width_au: float        # r_max_gh - r_runaway_gh
    stellar_flux_inner_solar: float
    stellar_flux_outer_solar: float


@dataclass
class AtmosphericGasRetention:
    """Jeans thermal escape diagnostic for an atmospheric volatile species."""
    species_name: str
    molar_mass_amu: float
    thermal_velocity_kms: float
    escape_velocity_kms: float
    velocity_ratio: float           # v_esc / v_th (>= 6 for geological retention)
    jeans_parameter_lambda: float   # lambda_J = v_esc^2 / (2 v_th^2)
    is_retained_longterm: bool


@dataclass
class PlanetaryHabitabilityResult:
    """Comprehensive habitability assessment for a planet-star system."""
    universe: UniverseParameters
    stellar_luminosity_solar: float
    stellar_effective_temp_k: float
    planet_mass_earth: float
    planet_radius_earth: float
    planet_radius_m: float
    surface_gravity_ms2: float
    surface_gravity_earth: float
    escape_velocity_kms: float
    orbital_radius_au: float
    equilibrium_temp_k: float
    surface_temp_k: float
    habitable_zone: HabitableZoneBoundaries
    is_in_habitable_zone: bool
    is_water_retained: bool
    is_hydrogen_escaped: bool
    is_atmosphere_viable: bool
    gas_retention: Dict[str, AtmosphericGasRetention]
    habitability_score: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


class PlanetaryHabitabilitySolver:
    """Evaluates photoclimatic habitable zones and atmospheric volatile retention."""

    # Volatile species: name -> molar mass in atomic mass units (amu)
    ATMOSPHERIC_VOLATILES: Dict[str, float] = {
        "H2": 2.016,
        "He": 4.0026,
        "H2O": 18.015,
        "N2": 28.013,
        "O2": 31.999,
        "CO2": 44.010,
    }

    # Kopparapu et al. (2013, 2014) polynomial coefficients for effective solar flux S_eff:
    # S_eff = S_eff0 + a * t_* + b * t_*^2 + c * t_*^3 + d * t_*^4, where t_* = T_* - 5780 K
    KOPPARAPU_COEFFS = {
        "recent_venus": {
            "s0": 1.7763,
            "a": 1.4335e-4,
            "b": 3.3954e-9,
            "c": -7.6364e-12,
            "d": -1.1950e-15,
        },
        "runaway_greenhouse": {
            "s0": 1.107,
            "a": 1.332e-4,
            "b": 1.580e-8,
            "c": -8.308e-12,
            "d": -1.931e-15,
        },
        "maximum_greenhouse": {
            "s0": 0.356,
            "a": 6.171e-5,
            "b": 1.698e-9,
            "c": -3.198e-12,
            "d": -5.575e-16,
        },
        "early_mars": {
            "s0": 0.320,
            "a": 5.547e-5,
            "b": 1.526e-9,
            "c": -2.874e-12,
            "d": -5.011e-16,
        },
    }

    @classmethod
    def compute_habitable_zone(
        cls,
        luminosity_solar: float,
        effective_temp_kelvin: float,
    ) -> HabitableZoneBoundaries:
        """Computes Kasting-Kopparapu photoclimatic habitable zone boundaries (AU)."""
        L_rel = max(1e-10, luminosity_solar)
        T_diff = max(-3000.0, min(5000.0, effective_temp_kelvin - 5780.0))

        def get_seff(limit_name: str) -> float:
            c = cls.KOPPARAPU_COEFFS[limit_name]
            s = c["s0"] + c["a"] * T_diff + c["b"] * (T_diff ** 2) + c["c"] * (T_diff ** 3) + c["d"] * (T_diff ** 4)
            return max(0.01, s)

        s_rv = get_seff("recent_venus")
        s_rg = get_seff("runaway_greenhouse")
        s_mg = get_seff("maximum_greenhouse")
        s_em = get_seff("early_mars")

        # Orbital distance in AU: r = sqrt(L_* / S_eff)
        r_rv = math.sqrt(L_rel / s_rv)
        r_rg = math.sqrt(L_rel / s_rg)
        r_mg = math.sqrt(L_rel / s_mg)
        r_em = math.sqrt(L_rel / s_em)

        width = max(0.0, r_mg - r_rg)

        return HabitableZoneBoundaries(
            recent_venus_au=r_rv,
            runaway_greenhouse_au=r_rg,
            maximum_greenhouse_au=r_mg,
            early_mars_au=r_em,
            habitable_width_au=width,
            stellar_flux_inner_solar=s_rg,
            stellar_flux_outer_solar=s_mg,
        )

    @staticmethod
    def compute_rocky_planet_structure(
        mass_earth: float,
        u: UniverseParameters,
    ) -> Tuple[float, float, float, float]:
        """Calculates terrestrial planet radius, surface gravity, and escape velocity.
        
        Returns:
            (R_planet_m, R_planet_earth, g_surface_ms2, v_escape_kms)
        """
        # Quantum electrodynamic + gravitational condensed matter scaling:
        # Electronic Bohr atomic spacing scales as 1 / (alpha * beta * m_p)
        # Relativistic and gravitational compression scales with alpha_G
        scale_bohr = (ALPHA_STANDARD / u.alpha) * (BETA_STANDARD / u.beta)
        scale_grav = (ALPHA_G_STANDARD / u.alpha_G) ** 0.05

        # Empirical terrestrial planet mass-radius index M^0.27
        m_rel = max(1e-4, mass_earth)
        r_rel = (m_rel ** 0.27) * scale_bohr * scale_grav

        R_planet_m = r_rel * EARTH_RADIUS_M
        M_planet_kg = m_rel * EARTH_MASS_KG

        # Gravitational coupling in candidate universe
        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio

        g_surface = G_u * M_planet_kg / (R_planet_m ** 2.0)
        v_esc_ms = math.sqrt(2.0 * G_u * M_planet_kg / R_planet_m)
        v_esc_kms = v_esc_ms / 1000.0

        return R_planet_m, r_rel, g_surface, v_esc_kms

    @classmethod
    def evaluate_atmospheric_retention(
        cls,
        T_eq_kelvin: float,
        v_escape_kms: float,
        u: UniverseParameters,
    ) -> Dict[str, AtmosphericGasRetention]:
        """Evaluates Jeans thermal escape criterion for standard volatile gases.
        
        Uses exospheric base temperature T_exo ~ 3.5 * T_eq driven by stellar EUV/X-ray
        heating in the upper thermosphere where particle escape takes place.
        """
        v_esc_ms = v_escape_kms * 1000.0
        results: Dict[str, AtmosphericGasRetention] = {}

        # Proton mass in candidate universe
        m_p_si = PROTON_MASS_SI
        T_exo = 3.5 * max(10.0, T_eq_kelvin)

        for gas, molar_amu in cls.ATMOSPHERIC_VOLATILES.items():
            molecular_mass = molar_amu * m_p_si
            # Root-mean-square thermal speed at exobase: v_th = sqrt(3 k_B T_exo / m)
            v_th_ms = math.sqrt((3.0 * BOLTZMANN_K_SI * T_exo) / molecular_mass)
            v_th_kms = v_th_ms / 1000.0

            ratio = v_esc_ms / max(v_th_ms, 1e-10)
            jeans_lambda = (v_esc_ms ** 2.0) / (2.0 * max(v_th_ms ** 2.0, 1e-10))

            # Classical Jeans retention threshold: v_esc / v_th >= 6.0 (lambda_J >= 18)
            is_retained = ratio >= 6.0

            results[gas] = AtmosphericGasRetention(
                species_name=gas,
                molar_mass_amu=molar_amu,
                thermal_velocity_kms=v_th_kms,
                escape_velocity_kms=v_escape_kms,
                velocity_ratio=ratio,
                jeans_parameter_lambda=jeans_lambda,
                is_retained_longterm=is_retained,
            )

        return results

    def evaluate_habitability(
        self,
        u: UniverseParameters,
        star_solution: Optional[StellarEquilibriumSolution] = None,
        planet_mass_earth: float = 1.0,
        orbital_radius_au: float = 1.0,
    ) -> PlanetaryHabitabilityResult:
        """Performs full end-to-end planetary habitability and atmospheric analysis."""
        # Fallback to solar analog if star_solution not provided
        if star_solution is not None:
            L_solar = star_solution.luminosity_solar
            T_eff = star_solution.effective_temp_kelvin
        else:
            L_solar = 1.0
            T_eff = 5778.0

        # 1. Circumstellar Habitable Zone boundaries
        chz = self.compute_habitable_zone(L_solar, T_eff)

        # 2. Planetary Geophysics & Escape Velocity
        R_p_m, R_p_earth, g_ms2, v_esc_kms = self.compute_rocky_planet_structure(planet_mass_earth, u)
        g_earth = g_ms2 / 9.80665

        # 3. Radiation Balance & Equilibrium Temperature
        # L_* (1 - A) / (16 pi sigma d^2)
        L_watts = L_solar * SOLAR_LUMINOSITY_W
        d_m = orbital_radius_au * ASTRONOMICAL_UNIT_M
        T_eq = ((L_watts * (1.0 - EARTH_ALBEDO_BOND)) / (16.0 * math.pi * STEFAN_BOLTZMANN_SI * (d_m ** 2.0))) ** 0.25

        # Surface temperature with greenhouse effect
        T_surface = T_eq + EARTH_GREENHOUSE_DT_K

        # Check CHZ location: conservative (runaway gh to max gh) or optimistic
        in_chz = chz.runaway_greenhouse_au <= orbital_radius_au <= chz.maximum_greenhouse_au

        # 4. Atmospheric Volatiles Retention Analysis
        gases = self.evaluate_atmospheric_retention(T_eq, v_esc_kms, u)
        water_retained = gases["H2O"].is_retained_longterm
        hydrogen_escaped = not gases["H2"].is_retained_longterm

        # An atmosphere is biologically viable if it retains heavy volatiles (H2O, N2, CO2)
        # and has escaped light hydrogen gas (preventing hydrogen asphyxiation)
        atmosphere_viable = water_retained and hydrogen_escaped and gases["N2"].is_retained_longterm

        # 5. Composite Habitability Score in [0, 1]
        # Temperature viability window around liquid water [260 K, 380 K]
        temp_score = math.exp(-0.5 * (((T_surface - 288.15) / 35.0) ** 2.0))

        # Position score within CHZ
        if in_chz:
            chz_score = 1.0
        else:
            # Distance from closest boundary in AU
            dist = min(abs(orbital_radius_au - chz.runaway_greenhouse_au), abs(orbital_radius_au - chz.maximum_greenhouse_au))
            chz_score = math.exp(-dist / 0.25)

        # Volatiles retention score
        ret_score = 1.0 if atmosphere_viable else 0.1
        if not water_retained:
            ret_score = 0.0

        habitability_score = (temp_score * chz_score * ret_score) ** (1.0 / 3.0)

        diagnostics = {
            "chz_inner_au": chz.runaway_greenhouse_au,
            "chz_outer_au": chz.maximum_greenhouse_au,
            "chz_width_au": chz.habitable_width_au,
            "t_eq_k": T_eq,
            "t_surf_k": T_surface,
            "v_esc_kms": v_esc_kms,
            "h2o_v_ratio": gases["H2O"].velocity_ratio,
            "h2_v_ratio": gases["H2"].velocity_ratio,
            "habitability_score": habitability_score,
        }

        return PlanetaryHabitabilityResult(
            universe=u,
            stellar_luminosity_solar=L_solar,
            stellar_effective_temp_k=T_eff,
            planet_mass_earth=planet_mass_earth,
            planet_radius_earth=R_p_earth,
            planet_radius_m=R_p_m,
            surface_gravity_ms2=g_ms2,
            surface_gravity_earth=g_earth,
            escape_velocity_kms=v_esc_kms,
            orbital_radius_au=orbital_radius_au,
            equilibrium_temp_k=T_eq,
            surface_temp_k=T_surface,
            habitable_zone=chz,
            is_in_habitable_zone=in_chz,
            is_water_retained=water_retained,
            is_hydrogen_escaped=hydrogen_escaped,
            is_atmosphere_viable=atmosphere_viable,
            gas_retention=gases,
            habitability_score=habitability_score,
            diagnostics=diagnostics,
        )
