"""Cosmology and Big Bang Nucleosynthesis (BBN) Numerical Solver.

Simulates the early radiation-dominated cosmological expansion (Friedmann equations)
and primordial nucleosynthesis in dimensionless parameter space:
    - Friedmann radiation expansion H(T) governed by alpha_G.
    - Weak n <-> p freeze-out temperature T_f and initial abundance ratio (n/p)_f.
    - Deuterium bottleneck timescale t_BBN and free neutron beta decay.
    - Primordial mass fractions: Hydrogen X_H, Helium-4 Y_p, Deuterium D/H, and 3He/H.
    - Cosmological viability assessment (prevents the all-helium and all-neutron catastrophes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    Q_STANDARD,
    UniverseParameters,
    HBAR_SI,
    SPEED_OF_LIGHT_SI,
    PROTON_MASS_SI,
    YEAR_IN_SECONDS,
)


@dataclass(frozen=True)
class BBNResults:
    """Detailed results from the Big Bang Nucleosynthesis simulation."""
    params: UniverseParameters
    # Thermodynamic freeze-out parameters
    t_freeze_out_sec: float
    t_freeze_out_mev: float
    np_ratio_frozen: float
    # Post-bottleneck parameters
    t_bottleneck_sec: float
    t_bottleneck_mev: float
    neutron_survival_fraction: float
    np_ratio_bbn: float
    # Primordial abundances
    helium4_mass_fraction_yp: float        # Y_p = 4*n_He / n_baryon (~ 0.245 standard)
    hydrogen_mass_fraction_xh: float       # X_H = 1 - Y_p (~ 0.755 standard)
    deuterium_abundance_dh: float          # D/H ratio (~ 2.5e-5 standard)
    helium3_abundance_he3_h: float         # 3He/H ratio (~ 1.0e-5 standard)
    # Cosmological viability
    score_bbn_viability: float
    is_bbn_viable: bool
    failure_reasons: List[str] = field(default_factory=list)
    # Profile history for plotting
    temperature_mev_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    time_sec_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yp_evolution: np.ndarray = field(default_factory=lambda: np.zeros(0))
    xh_evolution: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        status = "VIABLE" if self.is_bbn_viable else "UNVIABLE"
        lines = [
            f"=== BBN Cosmological Nucleosynthesis Report [{status}] ===",
            f"Viability Score       : {self.score_bbn_viability:.4f}",
            f"Freeze-out Temp T_f   : {self.t_freeze_out_mev:.3f} MeV (t = {self.t_freeze_out_sec:.3f} s)",
            f"Frozen (n/p)_f ratio  : {self.np_ratio_frozen:.4f}",
            f"Bottleneck Temp T_BBN : {self.t_bottleneck_mev:.3f} MeV (t = {self.t_bottleneck_sec:.1f} s)",
            f"BBN (n/p)_BBN ratio   : {self.np_ratio_bbn:.4f} (Neutron survival: {self.neutron_survival_fraction * 100:.1f}%)",
            f"Primordial Helium-4 Yp: {self.helium4_mass_fraction_yp * 100:.2f}%",
            f"Primordial Hydrogen XH: {self.hydrogen_mass_fraction_xh * 100:.2f}%",
            f"Deuterium Abundance   : D/H = {self.deuterium_abundance_dh:.2e}",
        ]
        if self.failure_reasons:
            lines.append("Cosmological Violations:")
            for r in self.failure_reasons:
                lines.append(f"  * {r}")
        return "\n".join(lines)


class CosmologyBBNSolver:
    """Simulates Friedmann expansion and Big Bang Nucleosynthesis under varying physical constants."""

    # Standard reference scales (in MeV and SI units)
    M_P_MEV: float = 938.272088               # Proton mass in MeV
    B_D_0_MEV: float = 2.224575               # Standard Deuteron binding energy (MeV)
    T_FREEZE_0_MEV: float = 0.800             # Standard n-p freeze-out temperature (~ 0.8 MeV)
    TAU_N_0_SEC: float = 879.4                # Standard free neutron mean lifetime (~ 14.65 min)
    ETA_BARYON: float = 6.1e-10               # Baryon-to-photon ratio eta = n_b / n_gamma

    def __init__(
        self,
        g_star: float = 10.75,
        min_hydrogen_fraction: float = 0.15,
        max_helium_fraction: float = 0.85,
    ) -> None:
        self.g_star = g_star
        self.min_hydrogen_fraction = min_hydrogen_fraction
        self.max_helium_fraction = max_helium_fraction

    def compute_neutron_lifetime(self, u: UniverseParameters) -> float:
        """Computes the free neutron lifetime tau_n scaled by Fermi decay phase space:
        tau_n proportional to 1 / (Q - beta)^5 via Fermi beta-decay kinematics.
        """
        if u.Q <= u.beta:
            # If Q <= beta, neutron is stable (proton decays into neutron instead)
            return float("inf")
        # Relative phase-space scaling normalized to standard parameters
        phase_ratio = ((u.Q - u.beta) / (Q_STANDARD - BETA_STANDARD)) ** 5
        tau_n = self.TAU_N_0_SEC / max(1e-9, phase_ratio)
        return max(0.1, min(1e9, tau_n))

    def compute_deuteron_binding(self, u: UniverseParameters) -> float:
        """Computes Deuteron binding energy B_d(alpha_s) in MeV."""
        delta_s_rel = (u.alpha_s - ALPHA_S_STANDARD) / ALPHA_S_STANDARD
        b_d = self.B_D_0_MEV * (1.0 + 5.5 * delta_s_rel)
        return max(0.0, b_d)

    def compute_freeze_out_temperature(self, u: UniverseParameters) -> float:
        """Calculates n-p freeze-out temperature T_f where weak rate Gamma_weak ~ H_expansion.
        H proportional to alpha_G^(1/2) * T^2.
        Gamma_weak proportional to T^5.
        Therefore T_f proportional to alpha_G^(1/6).
        """
        alpha_g_ratio = u.alpha_G / ALPHA_G_STANDARD
        t_f = self.T_FREEZE_0_MEV * (alpha_g_ratio ** (1.0 / 6.0))
        return max(1e-4, min(100.0, t_f))

    def compute_bottleneck_temperature(self, b_d_mev: float) -> float:
        """Calculates temperature T_BBN where high-energy photons can no longer photo-disintegrate deuterons.
        In cosmology, because eta_baryon ~ 6e-10 << 1, there are ~ 2 billion photons per nucleon.
        Deuterium becomes stable when T drops below B_d / ln(1/eta) ~ B_d / 36.0 (~ 0.062 MeV in our universe).
        """
        if b_d_mev <= 0.0:
            return 0.0
        return b_d_mev / 36.0

    def compute_time_from_temperature(self, t_mev: float, u: UniverseParameters) -> float:
        """Computes cosmological time t in seconds at temperature T in radiation era:
        t(T) = (hbar / (2 * sqrt(alpha_G) * m_p * c^2)) * (m_p / T)^2.
        """
        if t_mev <= 0.0:
            return float("inf")
        # Calibrated to standard universe where T = 1 MeV corresponds to t ~ 1.0 s
        alpha_g_ratio = u.alpha_G / ALPHA_G_STANDARD
        t_sec = 1.0 / (math.sqrt(alpha_g_ratio) * (t_mev ** 2))
        return max(1e-6, t_sec)

    def solve_bbn(self, u: UniverseParameters) -> BBNResults:
        """Executes full cosmological BBN nucleosynthesis integration."""
        reasons: List[str] = []

        # 1. Neutron Lifetime & Binding Energy
        tau_n = self.compute_neutron_lifetime(u)
        b_d_mev = self.compute_deuteron_binding(u)
        q_mev = u.Q * self.M_P_MEV

        # 2. Freeze-out temperature and timescale
        t_f_mev = self.compute_freeze_out_temperature(u)
        t_f_sec = self.compute_time_from_temperature(t_f_mev, u)

        # Equilibrium ratio at freeze-out: (n/p)_f = exp(-Delta_m / T_f) = exp(-Q * m_p / T_f)
        exponent = -q_mev / t_f_mev
        if exponent < -50.0:
            np_f = 0.0
        elif exponent > 50.0:
            np_f = 1.0
        else:
            np_f = math.exp(exponent)

        # 3. Deuterium Bottleneck and Neutron Decay
        t_bbn_mev = self.compute_bottleneck_temperature(b_d_mev)
        t_bbn_sec = self.compute_time_from_temperature(t_bbn_mev, u) if t_bbn_mev > 0.0 else float("inf")

        # Duration of neutron decay window between freeze-out and BBN ignition
        delta_t_decay = max(0.0, t_bbn_sec - t_f_sec) if math.isfinite(t_bbn_sec) else 1e9

        if math.isinf(tau_n):
            # Neutrons do not decay (e.g. Q <= beta)
            decay_fraction = 1.0
        else:
            decay_fraction = math.exp(-delta_t_decay / tau_n)

        # Effective (n/p) ratio at the onset of nucleosynthesis
        np_bbn = np_f * decay_fraction

        # 4. Check for catastrophic diproton binding
        # If diproton binds (B_pp > 0), strong interaction burns ALL hydrogen directly
        delta_alpha_rel = (u.alpha - ALPHA_STANDARD) / ALPHA_STANDARD
        delta_alpha_s_rel = (u.alpha_s - ALPHA_S_STANDARD) / ALPHA_S_STANDARD
        b_pp_mev = -0.100 + self.B_D_0_MEV * (5.5 * delta_alpha_s_rel - 2.8 * delta_alpha_rel)
        diproton_bound = b_pp_mev > 0.0

        # 5. Mass Fractions calculation
        if b_d_mev <= 0.0:
            # Deuteron cannot bind: no nucleosynthesis, universe remains pure hydrogen (or free decaying neutrons)
            yp = 0.0
            xh = 1.0
            dh = 0.0
            he3 = 0.0
            reasons.append("Deuteron is unbound: nucleosynthesis impossible, no complex nuclei can form.")
        elif diproton_bound:
            # Catastrophic fast fusion: all protons fuse into 2He -> decays or binds into 4He
            yp = 1.0
            xh = 0.0
            dh = 0.0
            he3 = 0.0
            reasons.append("Diproton (2He) is bound: instantaneous strong fusion consumes 100% of primordial hydrogen.")
        else:
            # Standard bottleneck channel: all available neutrons are captured into 4He
            # Y_p = 2*(n/p) / (1 + (n/p))
            yp = (2.0 * np_bbn) / (1.0 + np_bbn)
            yp = max(0.0, min(1.0, yp))
            xh = max(0.0, 1.0 - yp)

            # Trace abundances D/H and 3He/H:
            # Scaled inversely with nuclear reaction efficiency and bottleneck duration
            dh_standard = 2.5e-5
            dh = dh_standard * (self.B_D_0_MEV / max(0.1, b_d_mev)) * math.sqrt(u.alpha_G / ALPHA_G_STANDARD)
            dh = max(1e-9, min(0.1, dh))
            he3 = 1.0e-5 * (self.B_D_0_MEV / max(0.1, b_d_mev))

        # 6. Check Cosmological Catastrophes
        if xh < self.min_hydrogen_fraction:
            reasons.append(
                f"All-Helium Catastrophe: Primordial hydrogen fraction X_H = {xh*100:.1f}% "
                f"< {self.min_hydrogen_fraction*100:.1f}%; insufficient for water and organic molecules."
            )
        if yp > self.max_helium_fraction:
            reasons.append(
                f"Helium Dominance: Y_p = {yp*100:.1f}% exceeds threshold {self.max_helium_fraction*100:.1f}%."
            )
        if q_mev <= u.beta * self.M_P_MEV:
            reasons.append("Protons unstable against decay into neutrons: primordial hydrogen cannot survive.")

        # 7. Smooth Viability Score S_BBN
        # Demands adequate Hydrogen (X_H in [0.2, 0.9]) and non-zero Helium (Y_p in [0.05, 0.7])
        s_hydrogen = 1.0 / (1.0 + math.exp(-20.0 * (xh - 0.20)))
        s_helium = 1.0 / (1.0 + math.exp(20.0 * (yp - 0.80)))
        s_diproton = 0.0 if diproton_bound else 1.0
        s_deuteron = 1.0 / (1.0 + math.exp(-10.0 * (b_d_mev - 0.5)))

        score = float(s_hydrogen * s_helium * s_diproton * s_deuteron)
        is_viable = (score >= 0.85) and (len(reasons) == 0)

        # 8. Generate temperature profile for plotting
        t_temps = np.logspace(math.log10(10.0), math.log10(0.01), 60) # 10 MeV down to 10 keV
        t_times = np.array([self.compute_time_from_temperature(t, u) for t in t_temps])
        
        # Track evolution of free nucleon ratios as a function of temperature
        yp_evo = []
        xh_evo = []
        for t_val, t_sec in zip(t_temps, t_times):
            if t_val > t_f_mev:
                # Equilibrium
                np_t = math.exp(-q_mev / t_val) if t_val > 0 else 0.0
                y_curr = 0.0
            elif t_val > t_bbn_mev:
                # Decaying free neutrons before bottleneck
                decay = math.exp(-(t_sec - t_f_sec) / tau_n) if math.isfinite(tau_n) else 1.0
                np_t = np_f * max(0.0, decay)
                y_curr = 0.0
            else:
                # Post-bottleneck: helium is formed
                y_curr = yp
            
            x_curr = 1.0 - y_curr
            yp_evo.append(y_curr)
            xh_evo.append(x_curr)

        return BBNResults(
            params=u,
            t_freeze_out_sec=t_f_sec,
            t_freeze_out_mev=t_f_mev,
            np_ratio_frozen=np_f,
            t_bottleneck_sec=t_bbn_sec,
            t_bottleneck_mev=t_bbn_mev,
            neutron_survival_fraction=decay_fraction,
            np_ratio_bbn=np_bbn,
            helium4_mass_fraction_yp=yp,
            hydrogen_mass_fraction_xh=xh,
            deuterium_abundance_dh=dh,
            helium3_abundance_he3_h=he3,
            score_bbn_viability=score,
            is_bbn_viable=is_viable,
            failure_reasons=reasons,
            temperature_mev_history=t_temps,
            time_sec_history=t_times,
            yp_evolution=np.array(yp_evo),
            xh_evolution=np.array(xh_evo),
        )
