"""Analytical Viability Manifolds in the Dimensionless Fundamental Constants Space.

Evaluates whether a candidate universe θ = (alpha, alpha_G, beta, alpha_s, Q)
possesses the fundamental prerequisites for:
  1. Di-nucleon & Hydrogen stability (p + e -> n + nu prohibited, bound deuteron, unbound diproton).
  2. Hoyle resonance window (fine-tuned 3-alpha triple collision for carbon/oxygen balance).
  3. Stellar lifetime scale (tau_* > 10^8 yr equivalent for planetary & structural evolution).
  4. Atomic / molecular scale separation (Bohr radius a_0 >> nuclear confinement radius r_n).

All penalties are computed via smooth, differentiable functions S_i(θ) ∈ (0, 1]
allowing gradient-based and evolutionary optimization on the manifold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    Q_STANDARD,
    CHRONO_SCALE_SI,
    YEAR_IN_SECONDS,
    UniverseParameters,
    compute_dynamic_q,
)
from multiverse.solvers.nuclear_cluster import NuclearClusterSolver


@dataclass
class ViabilityReport:
    """Detailed breakdown of viability metrics for a candidate universe."""
    params: UniverseParameters
    total_score: float
    score_hydrogen_dinucleon: float
    score_hoyle_resonance: float
    score_stellar_lifetime: float
    score_atomic_hierarchy: float
    stellar_lifetime_years: float
    bohr_to_nuclear_ratio: float
    hoyle_resonance_shift_mev: float
    deuteron_binding_energy_mev: float
    diproton_bound: bool
    is_viable: bool
    score_born_oppenheimer: float = 1.0
    score_zams_window: float = 1.0
    born_oppenheimer_param: float = 0.1528
    stellar_mass_window_ratio: float = 1500.0
    score_habitable_zone: float = 1.0
    score_atmospheric_retention: float = 1.0
    habitable_zone_width_au: float = 0.725
    water_retention_ratio: float = 10.11
    diagnostics: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "VIABLE" if self.is_viable else "UNVIABLE"
        lines = [
            f"=== Universe Viability Report [{status}] ===",
            f"Total Viability Score : {self.total_score:.5f}",
            f"  - Hydrogen & Di-nucleon : {self.score_hydrogen_dinucleon:.5f}",
            f"  - Hoyle 3-alpha Resonance: {self.score_hoyle_resonance:.5f}",
            f"  - Stellar Lifetime Scale : {self.score_stellar_lifetime:.5f} ({self.stellar_lifetime_years:.2e} yr)",
            f"  - Atomic Hierarchy (a0/rn): {self.score_atomic_hierarchy:.5f} (ratio={self.bohr_to_nuclear_ratio:.1f})",
            f"  - Born-Oppenheimer Metric: {self.score_born_oppenheimer:.5f} (kappa_BO={self.born_oppenheimer_param:.4f})",
            f"  - ZAMS Stellar Mass Window: {self.score_zams_window:.5f} (ratio={self.stellar_mass_window_ratio:.1f})",
            f"  - Habitable Zone Existence: {self.score_habitable_zone:.5f} (width={self.habitable_zone_width_au:.3f} AU)",
            f"  - Atmospheric H2O Retain : {self.score_atmospheric_retention:.5f} (v_esc/v_th={self.water_retention_ratio:.2f})",
            f"Diagnostics:",
            f"  - Hoyle shift (MeV)     : {self.hoyle_resonance_shift_mev:+.4f} MeV",
            f"  - Deuteron Binding (MeV): {self.deuteron_binding_energy_mev:.4f} MeV",
            f"  - Diproton bound state   : {self.diproton_bound}",
            f"  - Born-Oppenheimer kappa: {self.born_oppenheimer_param:.4f}",
            f"  - Stellar window ratio   : {self.stellar_mass_window_ratio:.1f}",
            f"  - CHZ width (AU)         : {self.habitable_zone_width_au:.3f} AU",
            f"  - Water retention ratio  : {self.water_retention_ratio:.2f}",
        ]
        if self.failure_reasons:
            lines.append("Violations / Bottlenecks:")
            for r in self.failure_reasons:
                lines.append(f"  * {r}")
        return "\n".join(lines)


class AnalyticalManifold:
    """Analytical evaluation of cosmological and physical viability boundaries."""

    # Reference nuclear parameters (in MeV)
    M_P_MEV: float = 938.272088
    B_D_0_MEV: float = 2.224575                # Deuteron binding energy at alpha_s_standard
    HOYLE_WINDOW_TOLERANCE_MEV: float = 0.120  # ~120 keV fine-tuning window
    MIN_STELLAR_LIFETIME_YR: float = 1.0e8     # Minimum viable stellar lifetime (100 Myr)

    def __init__(
        self,
        min_lifetime_yr: float = 1.0e8,
        hoyle_tolerance_mev: float = 0.120,
        atomic_ratio_min: float = 50.0,
        viability_threshold: float = 0.90,
        use_dynamic_q: bool = True,
        use_cluster_solver: bool = True,
        use_chiral_eft: bool = False,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.min_lifetime_yr = min_lifetime_yr
        self.hoyle_tolerance_mev = hoyle_tolerance_mev
        self.atomic_ratio_min = atomic_ratio_min
        self.viability_threshold = viability_threshold
        self.use_dynamic_q = use_dynamic_q
        self.use_cluster_solver = use_cluster_solver
        self.use_chiral_eft = use_chiral_eft
        self.cluster_solver = NuclearClusterSolver(hoyle_tolerance_mev=self.hoyle_tolerance_mev)
        if weights is not None:
            w_arr = np.array(weights, dtype=float)
            if len(w_arr) != 8:
                raise ValueError(f"weights must have 8 components, got {len(w_arr)}")
            self.weights: Optional[np.ndarray] = w_arr / np.sum(w_arr)
        else:
            self.weights = None

    @staticmethod
    def _smooth_step(val: float, threshold: float, scale: float) -> float:
        """Smooth logistic transition: 1 / (1 + exp(-(val - threshold) / scale))."""
        z = (val - threshold) / scale
        if z > 35.0:
            return 1.0
        if z < -35.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _smooth_interval(val: float, center: float, width: float) -> float:
        """Smooth Gaussian-like window around center with characteristic width."""
        diff = (val - center) / width
        return math.exp(-0.5 * (diff * diff))

    def evaluate(self, u: UniverseParameters) -> ViabilityReport:
        """Evaluates all analytical viability constraints for the given universe parameters."""
        reasons: List[str] = []

        # --------------------------------------------------------------------
        # 1. Hydrogen & Di-nucleon Stability (with Dynamic Q(alpha, alpha_s))
        # --------------------------------------------------------------------
        # Deuteron binding energy variation with strong coupling:
        # Effective field theory sensitivity: d(ln B_d) / d(ln alpha_s) ~ 5.5
        delta_alpha_s_rel = (u.alpha_s - ALPHA_S_STANDARD) / ALPHA_S_STANDARD
        b_d_mev = self.B_D_0_MEV * (1.0 + 5.5 * delta_alpha_s_rel)
        if b_d_mev < 0.0:
            b_d_mev = 0.0

        # Dynamic Cottingham / QCD-QED mass difference (respects explicit custom Q if provided)
        if self.use_dynamic_q:
            if abs(u.Q - Q_STANDARD) > 1e-12 and abs(u.Q - compute_dynamic_q(u.alpha, u.alpha_s)) > 1e-12:
                q_eff = u.Q
            else:
                q_eff = compute_dynamic_q(u.alpha, u.alpha_s)
        else:
            q_eff = u.Q
        q_mev = q_eff * self.M_P_MEV
        m_e_mev = u.beta * self.M_P_MEV

        # Barrier 1: Proton beta stability (Q > 0)
        # Prevents m_p > m_n which causes free proton beta decay p -> n + e+ + nu
        score_proton_stable = self._smooth_step(q_eff, 0.0, 0.05 * Q_STANDARD)
        if q_eff <= 0.0:
            reasons.append("Proton free beta-decay catastrophe: Q <= 0 (m_p >= m_n) causes all cosmic hydrogen to decay into neutrons.")

        # Barrier 2: Hydrogen stability condition: Q > beta (i.e. m_n - m_p > m_e)
        # Prevents spontaneous p + e -> n + nu capture in neutral atoms.
        score_h_stable = self._smooth_step(q_eff - u.beta, 0.0, 0.1 * BETA_STANDARD)
        if q_eff <= u.beta:
            reasons.append("Hydrogen atom unstable: m_n - m_p <= m_e causes spontaneous electron capture into neutrons.")

        # Deuteron stability condition: Q < B_d / m_p (i.e. m_n - m_p < B_d)
        # Prevents deuteron spontaneous disintegration into two neutrons.
        b_d_dimensionless = b_d_mev / self.M_P_MEV
        score_d_stable = self._smooth_step(b_d_dimensionless - q_eff, 0.0, 0.1 * Q_STANDARD)
        if b_d_dimensionless <= q_eff:
            reasons.append("Deuteron unstable: m_n - m_p exceeds deuteron binding energy; no bound di-nucleons.")

        # Diproton unbinding condition:
        # In our universe, 2He is unbound by ~100 keV (0.100 MeV).
        # If alpha_s increases by > ~3% or alpha decreases, diproton binds.
        delta_alpha_rel = (u.alpha - ALPHA_STANDARD) / ALPHA_STANDARD
        b_pp_mev = -0.100 + self.B_D_0_MEV * (5.5 * delta_alpha_s_rel - 2.8 * delta_alpha_rel)
        diproton_bound = b_pp_mev > 0.0
        # Penalty if diproton is bound (destroys Big Bang hydrogen abundance)
        score_diproton = self._smooth_step(-b_pp_mev, 0.0, 0.018)
        if diproton_bound:
            reasons.append("Diproton (2He) is bound: causes catastrophic primordial helium conversion, leaving no hydrogen.")

        score_dinucleon = score_proton_stable * score_h_stable * score_d_stable * score_diproton

        # --------------------------------------------------------------------
        # 2. Hoyle State / Stellar Nucleosynthesis Window (Chiral EFT / Ali-Bodmer Cluster Solver)
        # --------------------------------------------------------------------
        if self.use_chiral_eft:
            chiral_shift, _ = self.cluster_solver.compute_chiral_eft_hoyle_resonance(u.alpha, u.alpha_s)
            hoyle_shift_mev = chiral_shift
        elif self.use_cluster_solver:
            cluster_res = self.cluster_solver.compute_hoyle_resonance(u.alpha, u.alpha_s)
            hoyle_shift_mev = cluster_res.hoyle_shift_mev
        else:
            c_alpha_mev = 2.45
            c_s_mev = 2.25
            hoyle_shift_mev = c_alpha_mev * delta_alpha_rel - c_s_mev * delta_alpha_s_rel

        score_hoyle = self._smooth_interval(hoyle_shift_mev, 0.0, self.HOYLE_WINDOW_TOLERANCE_MEV)
        if abs(hoyle_shift_mev) > 2.0 * self.HOYLE_WINDOW_TOLERANCE_MEV:
            reasons.append(
                f"Hoyle state shifted by {hoyle_shift_mev:+.3f} MeV (tolerance ±{self.HOYLE_WINDOW_TOLERANCE_MEV:.3f} MeV): "
                f"destroys carbon-oxygen nucleosynthesis equilibrium."
            )

        # --------------------------------------------------------------------
        # 3. Stellar Lifetime Scale
        # --------------------------------------------------------------------
        # tau_* ~ alpha^(-2) * alpha_G^(-1) * tau_0
        tau_star_sec = (u.alpha ** -2.0) * (u.alpha_G ** -1.0) * CHRONO_SCALE_SI
        tau_star_yr = tau_star_sec / YEAR_IN_SECONDS

        # We require tau_* >= min_lifetime_yr (e.g. 10^8 yr for evolutionary complexity)
        # Logarithmic scale transition:
        log_tau = math.log10(max(tau_star_yr, 1e-10))
        log_min = math.log10(self.min_lifetime_yr)
        score_stellar = self._smooth_step(log_tau, log_min, 0.3)

        # Also upper bound: if tau_* > 10^16 yr, stars are extremely cold or non-burning
        if log_tau > 16.0:
            score_stellar *= self._smooth_step(20.0, log_tau, 1.0)

        if tau_star_yr < self.min_lifetime_yr:
            reasons.append(f"Stellar lifetime too short ({tau_star_yr:.2e} yr < {self.min_lifetime_yr:.2e} yr) for planetary evolution.")

        # --------------------------------------------------------------------
        # 4. Atomic Scale Hierarchy & Chemistry Range
        # --------------------------------------------------------------------
        # Bohr radius: a_0 / lambda_p = 1 / (alpha * beta)
        # Confinement radius: r_n / lambda_p ~ 1 / alpha_s
        # Scale separation: R_ratio = a_0 / r_n = alpha_s / (alpha * beta)
        bohr_to_nuclear = u.alpha_s / (u.alpha * u.beta)

        log_ratio = math.log10(max(bohr_to_nuclear, 1e-5))
        log_ratio_min = math.log10(self.atomic_ratio_min)
        score_hierarchy = self._smooth_step(log_ratio, log_ratio_min, 0.4)

        # Relativistic limit for bound atomic states: alpha < 0.5 (Dirac vacuum stability)
        score_rel = self._smooth_step(0.40 - u.alpha, 0.0, 0.05)
        score_atomic = score_hierarchy * score_rel

        if bohr_to_nuclear < self.atomic_ratio_min:
            reasons.append(f"Atomic Bohr radius collapses into nuclear radius (a0/rn = {bohr_to_nuclear:.1f} < {self.atomic_ratio_min:.1f}).")
        if u.alpha > 0.40:
            reasons.append(f"Electromagnetic fine-structure constant too strong (alpha = {u.alpha:.3f} > 0.40): Dirac vacuum instability.")

        # --------------------------------------------------------------------
        # 5. Born-Oppenheimer Molecular Stability Limit
        # --------------------------------------------------------------------
        # The adiabatic parameter kappa_BO = beta^(1/4) = (m_e / m_p)^(1/4)
        # must satisfy kappa_BO < 0.25 (beta < (0.25)^4 = 1/256 ~ 0.00390625)
        # to ensure electronic clouds adjust instantaneously to nuclear vibrations.
        # beta > 0.0039 destroys localized covalent bonding and molecular chemistry.
        kappa_bo = u.beta ** 0.25
        score_bo = self._smooth_step(0.00390625 - u.beta, 0.0, 0.00025)
        if u.beta > 0.00390625:
            reasons.append(
                f"Born-Oppenheimer breakdown: beta = {u.beta:.5f} > 0.0039 (kappa_BO = {kappa_bo:.3f} > 0.25) "
                f"destroys localized covalent molecular orbitals."
            )

        # --------------------------------------------------------------------
        # 6. Stellar ZAMS Mass Burning Window (M_max / M_min >= 10)
        # --------------------------------------------------------------------
        # Stable main sequence population requires a burning window spanning
        # at least an order of magnitude: M_max / M_min >= 10.
        # M_min ~ 0.080 * alpha_G^(-3/2) * beta^(3/4) * (alpha / alpha_s)^(3/2)
        # M_max ~ 120.0 * alpha_G^(-3/2) * alpha^(-2)
        scale_gravity = (ALPHA_G_STANDARD / u.alpha_G) ** 1.5
        m_min_ratio = (
            0.080
            * scale_gravity
            * ((u.beta / BETA_STANDARD) ** 0.75)
            * (((u.alpha / ALPHA_STANDARD) / (u.alpha_s / ALPHA_S_STANDARD)) ** 1.5)
        )
        m_max_ratio = 120.0 * scale_gravity * ((ALPHA_STANDARD / u.alpha) ** 2.0)
        mass_window_ratio = m_max_ratio / max(m_min_ratio, 1e-30)

        # Penalize if mass_window_ratio < 10.0 (log10 < 1.0)
        log_window = math.log10(max(mass_window_ratio, 1e-10))
        score_zams = self._smooth_step(log_window, 1.0, 0.25)
        if mass_window_ratio < 10.0:
            reasons.append(
                f"Stellar ZAMS mass burning window collapsed: M_max / M_min = {mass_window_ratio:.2f} < 10 "
                f"prohibits stable main-sequence stellar population."
            )

        # --------------------------------------------------------------------
        # 7. Circumstellar Habitable Zone (CHZ) Existence & Width
        # --------------------------------------------------------------------
        # L_* ~ M^3.5 * (alpha_G0 / alpha_G)^1.5 * (alpha / alpha0)^2
        # r_in = sqrt(L_* / 1.107), r_out = sqrt(L_* / 0.356)
        scale_L = ((ALPHA_G_STANDARD / u.alpha_G) ** 1.5) * ((u.alpha / ALPHA_STANDARD) ** 2.0)
        r_chz_in = math.sqrt(max(1e-10, scale_L) / 1.107)
        r_chz_out = math.sqrt(max(1e-10, scale_L) / 0.356)
        width_chz = max(0.0, r_chz_out - r_chz_in)
        score_chz = self._smooth_step(width_chz, 0.10, 0.05)
        if width_chz < 0.05:
            reasons.append(
                f"Circumstellar habitable zone collapsed (Delta r_CHZ = {width_chz:.3f} AU < 0.05 AU): "
                f"no orbital niche for stable liquid surface water."
            )

        # --------------------------------------------------------------------
        # 8. Atmospheric Volatiles Jeans Retention (Water Retention vs H2 Escape)
        # --------------------------------------------------------------------
        # Terrestrial radius R_p ~ (alpha0 / alpha) * (beta0 / beta) * (alpha_G0 / alpha_G)^0.05
        # v_esc ~ sqrt(G M_p / R_p), v_th(H2O) ~ sqrt(3 k_B T_exo / (18 m_p))
        scale_R_p = (ALPHA_STANDARD / u.alpha) * (BETA_STANDARD / u.beta) * ((ALPHA_G_STANDARD / u.alpha_G) ** 0.05)
        scale_v_esc = math.sqrt((u.alpha_G / ALPHA_G_STANDARD) / max(1e-10, scale_R_p))
        v_esc_approx_kms = 11.19 * scale_v_esc

        scale_T_eq = max(1e-10, scale_L) ** 0.25
        T_exo_approx = 3.5 * 254.0 * scale_T_eq
        v_th_h2o_kms = math.sqrt((3.0 * 1.380649e-23 * max(10.0, T_exo_approx)) / (18.015 * 1.6726219e-27)) / 1000.0
        v_ratio_h2o = v_esc_approx_kms / max(1e-5, v_th_h2o_kms)

        score_atmo = self._smooth_step(v_ratio_h2o - 6.0, 0.0, 0.5)
        if v_ratio_h2o < 6.0:
            reasons.append(
                f"Atmospheric water retention failure: v_esc / v_th(H2O) = {v_ratio_h2o:.2f} < 6.0 "
                f"causes catastrophic Jeans thermal desiccation of oceans."
            )

        # --------------------------------------------------------------------
        # Composite Total Viability Score
        # --------------------------------------------------------------------
        # Geometric weighting over fundamental manifolds
        sub_scores = np.array([
            max(1e-12, score_dinucleon),
            max(1e-12, score_hoyle),
            max(1e-12, score_atomic),
            max(1e-12, score_bo),
            max(1e-12, score_stellar),
            max(1e-12, score_zams),
            max(1e-12, score_chz),
            max(1e-12, score_atmo),
        ], dtype=float)

        if self.weights is not None:
            total_score = float(np.prod(sub_scores ** self.weights))
        else:
            score_core = (score_dinucleon * score_hoyle * score_stellar * score_atomic) ** 0.25
            total_score = score_core * ((score_bo * score_zams * score_chz * score_atmo) ** 0.25)

        is_viable = (total_score >= self.viability_threshold) and (len(reasons) == 0)

        diagnostics = {
            "delta_alpha_rel": delta_alpha_rel,
            "delta_alpha_s_rel": delta_alpha_s_rel,
            "hoyle_shift_mev": hoyle_shift_mev,
            "b_d_mev": b_d_mev,
            "b_pp_mev": b_pp_mev,
            "stellar_lifetime_yr": tau_star_yr,
            "bohr_to_nuclear_ratio": bohr_to_nuclear,
            "q_effective": q_eff,
            "q_minus_beta": q_eff - u.beta,
            "b_d_minus_q_mev": b_d_mev - q_mev,
            "born_oppenheimer_kappa": kappa_bo,
            "stellar_mass_window_ratio": mass_window_ratio,
            "habitable_zone_width_au": width_chz,
            "water_retention_ratio": v_ratio_h2o,
        }

        return ViabilityReport(
            params=u,
            total_score=total_score,
            score_hydrogen_dinucleon=score_dinucleon,
            score_hoyle_resonance=score_hoyle,
            score_stellar_lifetime=score_stellar,
            score_atomic_hierarchy=score_atomic,
            stellar_lifetime_years=tau_star_yr,
            bohr_to_nuclear_ratio=bohr_to_nuclear,
            hoyle_resonance_shift_mev=hoyle_shift_mev,
            deuteron_binding_energy_mev=b_d_mev,
            diproton_bound=diproton_bound,
            is_viable=is_viable,
            score_born_oppenheimer=score_bo,
            score_zams_window=score_zams,
            born_oppenheimer_param=kappa_bo,
            stellar_mass_window_ratio=mass_window_ratio,
            score_habitable_zone=score_chz,
            score_atmospheric_retention=score_atmo,
            habitable_zone_width_au=width_chz,
            water_retention_ratio=v_ratio_h2o,
            diagnostics=diagnostics,
            failure_reasons=reasons,
        )

    def evaluate_decomposed(self, u: UniverseParameters) -> Tuple[float, float, Dict[str, float]]:
        """Evaluates decomposed train (quantum/microscopic) and test (macroscopic/astro) scores.
        
        C_train: Dinucleon/Hydrogen (Q > beta, Bd > Q, Bpp < 0), Hoyle resonance, Atomic hierarchy.
        C_test:  Stellar lifetime (tau_* >= 1e8 yr), ZAMS burning window, Circumstellar Habitable Zone,
                 Atmospheric Jeans volatile retention.
        """
        rep = self.evaluate(u)
        s_train = float((rep.score_hydrogen_dinucleon * rep.score_hoyle_resonance * rep.score_atomic_hierarchy) ** (1.0 / 3.0))
        s_test = float((rep.score_stellar_lifetime * rep.score_zams_window * rep.score_habitable_zone * rep.score_atmospheric_retention) ** 0.25)

        subscores = {
            "s_dinucleon": rep.score_hydrogen_dinucleon,
            "s_hoyle": rep.score_hoyle_resonance,
            "s_atomic": rep.score_atomic_hierarchy,
            "s_bo": rep.score_born_oppenheimer,
            "s_stellar": rep.score_stellar_lifetime,
            "s_zams": rep.score_zams_window,
            "s_chz": rep.score_habitable_zone,
            "s_atmo": rep.score_atmospheric_retention,
            "s_train": s_train,
            "s_test": s_test,
        }
        return s_train, s_test, subscores


def evaluate_universe_viability(u: UniverseParameters) -> ViabilityReport:
    """Convenience helper to evaluate candidate universe viability with default manifold."""
    manifold = AnalyticalManifold()
    return manifold.evaluate(u)
