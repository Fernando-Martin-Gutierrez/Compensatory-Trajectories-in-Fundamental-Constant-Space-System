"""Hydrodynamic Atmospheric Escape and Planetary Volatile Evolution Engine.

Implements ab initio hydrodynamic energy-limited and photo-evaporative atmospheric
escape models (Watson et al. 1981, Erkaev et al. 2007, Luger & Barnes 2015):

1. High-Energy Stellar Spectral Irradiance Evolution:
     F_XUV(t) = (L_XUV(t) / 4 pi a_orb^2)
     where L_XUV(t) / L_bol exhibits a saturated phase for t <= t_sat (~100 Myr)
     followed by power-law decay: L_XUV / L_bol ~ f_sat * (t / t_sat)^(-beta_XUV).

2. Energy-Limited Hydrodynamic Transonic Photo-evaporation:
     dM_dt_hydro = (epsilon_XUV * pi * R_p * R_XUV^2 * F_XUV) / (G * M_p * K_tide)
     where:
       - epsilon_XUV is the photo-evaporative heating efficiency (~0.15)
       - R_XUV is the effective EUV/XUV absorption radius (~1.1 R_p)
       - K_tide(xi) = 1 - 1.5/xi + 0.5/xi^3 is the Roche-lobe / Hill sphere tidal factor

3. Diffusion-Limited Escape (Hunten 1973 bottleneck):
     Phi_diff = (b_i / H_scale) * (Delta m / m_bg) * X_i

4. Geological Volatile Inventory Integration:
     Integrates total water mass lost across the stellar main-sequence lifetime:
     M_lost_total = int_0^tau_* dM_dt_hydro(t) dt
     quantified in Earth Ocean Equivalents (1 TO = 1.4e21 kg).
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
    BOLTZMANN_K_SI,
    YEAR_IN_SECONDS,
    UniverseParameters,
)
from multiverse.solvers.stellar_structure import (
    SOLAR_LUMINOSITY_W,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    STEFAN_BOLTZMANN_SI,
)

# Reference Geophysical and Astrophysical Constants
EARTH_MASS_KG: float = 5.9722e24
EARTH_RADIUS_M: float = 6.371e6
EARTH_OCEAN_MASS_KG: float = 1.4e21       # 1 Terrestrial Ocean (TO) mass in kg
ASTRONOMICAL_UNIT_M: float = 1.495978707e11
MYR_IN_SECONDS: float = 1e6 * YEAR_IN_SECONDS
GYR_IN_SECONDS: float = 1e9 * YEAR_IN_SECONDS


@dataclass
class HydrodynamicEscapeResult:
    """Detailed diagnostics of hydrodynamic and photochemical atmospheric escape."""
    planet_mass_kg: float
    planet_radius_m: float
    escape_velocity_kms: float
    hill_radius_m: float
    tidal_correction_k_tide: float
    stellar_xuv_flux_w_m2: float            # Contemporary / median XUV irradiance
    instantaneous_mass_loss_kg_s: float     # dM/dt in kg/s
    cumulative_mass_lost_kg: float          # Total volatile mass lost over tau_*
    oceans_equivalent_lost: float           # M_lost / M_earth_ocean
    is_ocean_preserved: bool                # Retains >= 1 ocean (lost < initial reservoir)
    diffusion_limited_loss_kg_s: float      # Hunten diffusion limit
    is_diffusion_limited: bool              # True if hydro loss exceeds diffusion capacity
    jeans_lambda_parameter: float           # Classical Jeans parameter for comparison
    effective_xuv_absorption_radius_m: float


class HydrodynamicAtmosphericEscapeEngine:
    """Solves photo-evaporative hydrodynamic and diffusion-limited volatile escape."""

    def __init__(
        self,
        photo_efficiency_epsilon: float = 0.15,
        xuv_absorption_ratio: float = 1.10,      # R_XUV / R_p
        f_sat_xuv: float = 1.0e-3,                # Saturation XUV luminosity fraction
        t_sat_myr: float = 100.0,                 # Saturation timescale in Myr
        beta_decay_xuv: float = 1.23,             # Ribas et al. power-law decay index
        initial_ocean_inventory_to: float = 5.0,  # Assumed initial volatile reservoir (5 oceans)
    ) -> None:
        self.epsilon_xuv = photo_efficiency_epsilon
        self.r_xuv_ratio = xuv_absorption_ratio
        self.f_sat = f_sat_xuv
        self.t_sat_s = t_sat_myr * MYR_IN_SECONDS
        self.beta_decay = beta_decay_xuv
        self.initial_oceans = initial_ocean_inventory_to

    def compute_stellar_xuv_luminosity(
        self,
        l_bol_watts: float,
        age_seconds: float,
    ) -> float:
        """Computes stellar EUV/X-ray luminosity L_XUV(t) via saturated power-law evolution."""
        if age_seconds <= self.t_sat_s:
            return self.f_sat * l_bol_watts
        decay_factor = (age_seconds / self.t_sat_s) ** (-self.beta_decay)
        return self.f_sat * l_bol_watts * decay_factor

    def compute_tidal_correction(
        self,
        r_p_m: float,
        m_p_kg: float,
        m_star_kg: float,
        a_orb_m: float,
    ) -> Tuple[float, float]:
        """Calculates Hill radius and Erkaev tidal correction factor K_tide."""
        # Hill radius: r_H = a * (M_p / 3 M_*)^(1/3)
        r_hill = a_orb_m * ((m_p_kg / (3.0 * m_star_kg)) ** (1.0 / 3.0))
        xi = max(1.5, r_hill / max(1.0, r_p_m))

        # K_tide(xi) = 1 - 3 / (2 xi) + 1 / (2 xi^3)
        k_tide = 1.0 - (1.5 / xi) + (0.5 / (xi ** 3.0))
        k_tide = max(0.10, min(1.0, k_tide))
        return r_hill, k_tide

    def compute_hydrodynamic_loss_rate(
        self,
        f_xuv_w_m2: float,
        r_p_m: float,
        m_p_kg: float,
        k_tide: float,
        u: UniverseParameters,
    ) -> float:
        """Evaluates instantaneous energy-limited mass loss dM/dt in kg/s."""
        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio
        r_xuv = self.r_xuv_ratio * r_p_m

        # dM/dt = (epsilon * pi * R_p * R_XUV^2 * F_XUV) / (G * M_p * K_tide)
        numerator = self.epsilon_xuv * math.pi * r_p_m * (r_xuv ** 2.0) * f_xuv_w_m2
        denominator = G_u * m_p_kg * k_tide
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def compute_diffusion_limit(
        self,
        t_eq_k: float,
        m_p_kg: float,
        r_p_m: float,
        u: UniverseParameters,
        species_molar_amu: float = 18.0,  # H2O
        mixing_ratio: float = 0.20,
    ) -> float:
        """Computes Hunten diffusion-limited escape flux in kg/s."""
        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio
        g_surf = G_u * m_p_kg / (r_p_m ** 2.0)

        # Scale height: H = k_B T / (mu m_p g)
        mu_bg = 28.0 * PROTON_MASS_SI  # N2 background
        h_scale = (BOLTZMANN_K_SI * t_eq_k) / (mu_bg * g_surf)

        # Binary diffusion coefficient: b(T) ~ 1e21 (T/300)^0.75 m^-1 s^-1
        b_diff = 1.0e21 * ((t_eq_k / 300.0) ** 0.75)

        # Number flux: phi = (b / H) * X_i
        phi_number = (b_diff / h_scale) * mixing_ratio
        mass_per_particle = species_molar_amu * PROTON_MASS_SI

        # Total mass loss over planetary surface: 4 pi R_p^2 * phi * m
        total_kg_s = 4.0 * math.pi * (r_p_m ** 2.0) * phi_number * mass_per_particle
        return total_kg_s

    def evaluate_planetary_escape(
        self,
        planet_mass_earth: float = 1.0,
        a_orb_au: float = 1.0,
        m_star_solar: float = 1.0,
        l_star_solar: float = 1.0,
        stellar_lifetime_yr: float = 1.0e10,
        u: Optional[UniverseParameters] = None,
    ) -> HydrodynamicEscapeResult:
        """Full simulation of hydrodynamic atmospheric escape over the planet's history."""
        if u is None:
            u = UniverseParameters.standard()

        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio

        m_p_kg = planet_mass_earth * EARTH_MASS_KG
        # Scaled planetary radius from mass-radius relation
        r_p_m = (
            EARTH_RADIUS_M
            * (planet_mass_earth ** 0.27)
            * (ALPHA_STANDARD / u.alpha)
            * (BETA_STANDARD / u.beta)
            * ((ALPHA_G_STANDARD / u.alpha_G) ** 0.05)
        )

        m_star_kg = m_star_solar * SOLAR_MASS_KG * ((ALPHA_G_STANDARD / u.alpha_G) ** 1.5)
        l_bol_w = l_star_solar * SOLAR_LUMINOSITY_W
        a_orb_m = a_orb_au * ASTRONOMICAL_UNIT_M

        # Velocity and gravity
        v_esc_ms = math.sqrt(2.0 * G_u * m_p_kg / r_p_m)
        v_esc_kms = v_esc_ms / 1000.0

        r_hill_m, k_tide = self.compute_tidal_correction(r_p_m, m_p_kg, m_star_kg, a_orb_m)

        # Representative contemporary XUV flux at 1 Gyr
        t_ref_s = 1.0 * GYR_IN_SECONDS
        l_xuv_ref = self.compute_stellar_xuv_luminosity(l_bol_w, t_ref_s)
        f_xuv_ref = l_xuv_ref / (4.0 * math.pi * (a_orb_m ** 2.0))

        inst_loss_kg_s = self.compute_hydrodynamic_loss_rate(f_xuv_ref, r_p_m, m_p_kg, k_tide, u)

        # Analytical integration of M_loss over time:
        # Integral_0^tau = Integral_0^t_sat (saturated) + Integral_t_sat^tau (decay)
        tau_s = min(stellar_lifetime_yr * YEAR_IN_SECONDS, 10.0 * GYR_IN_SECONDS)
        t_sat = min(self.t_sat_s, tau_s)

        # Pre-factor for energy-limited loss per unit XUV luminosity
        prefactor = (self.epsilon_xuv * math.pi * r_p_m * ((self.r_xuv_ratio * r_p_m) ** 2.0)) / (
            4.0 * math.pi * (a_orb_m ** 2.0) * G_u * m_p_kg * k_tide
        )

        # Phase 1: Saturated flux (0 to t_sat)
        e_xuv_phase1 = self.f_sat * l_bol_w * t_sat

        # Phase 2: Power-law decay (t_sat to tau_s)
        if tau_s > t_sat:
            # Integral of (t / t_sat)^(-beta) dt = t_sat / (1 - beta) * [ (tau/t_sat)^(1-beta) - 1 ]
            exponent = 1.0 - self.beta_decay
            integral_decay = (self.t_sat_s / exponent) * (
                ((tau_s / self.t_sat_s) ** exponent) - 1.0
            )
            e_xuv_phase2 = self.f_sat * l_bol_w * integral_decay
        else:
            e_xuv_phase2 = 0.0

        cum_loss_kg = prefactor * (e_xuv_phase1 + e_xuv_phase2)
        oceans_lost = cum_loss_kg / EARTH_OCEAN_MASS_KG

        is_preserved = oceans_lost < self.initial_oceans

        # Equilibrium temperature & Jeans comparison
        albedo = 0.30
        t_eq = ((l_bol_w * (1.0 - albedo)) / (16.0 * math.pi * STEFAN_BOLTZMANN_SI * (a_orb_m ** 2.0))) ** 0.25
        v_th_h2o = math.sqrt(3.0 * BOLTZMANN_K_SI * t_eq / (18.0 * PROTON_MASS_SI))
        lambda_j = (v_esc_ms ** 2.0) / (2.0 * (v_th_h2o ** 2.0))

        # Diffusion limit
        diff_limit_kg_s = self.compute_diffusion_limit(t_eq, m_p_kg, r_p_m, u)
        is_diff_limited = inst_loss_kg_s > diff_limit_kg_s

        return HydrodynamicEscapeResult(
            planet_mass_kg=m_p_kg,
            planet_radius_m=r_p_m,
            escape_velocity_kms=v_esc_kms,
            hill_radius_m=r_hill_m,
            tidal_correction_k_tide=k_tide,
            stellar_xuv_flux_w_m2=f_xuv_ref,
            instantaneous_mass_loss_kg_s=inst_loss_kg_s,
            cumulative_mass_lost_kg=cum_loss_kg,
            oceans_equivalent_lost=oceans_lost,
            is_ocean_preserved=is_preserved,
            diffusion_limited_loss_kg_s=diff_limit_kg_s,
            is_diffusion_limited=is_diff_limited,
            jeans_lambda_parameter=lambda_j,
            effective_xuv_absorption_radius_m=self.r_xuv_ratio * r_p_m,
        )
