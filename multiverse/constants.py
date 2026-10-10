"""MultiverseEquilibriumSim: Fundamental Physical Constants & Universe Parameter Definitions.

Models the universe through the dimensionless parameter space:
    alpha    : Fine-structure constant (electromagnetic coupling ~ 1/137.036)
    alpha_G  : Gravitational coupling constant for protons (G * m_p^2 / (hbar * c) ~ 5.9e-39)
    beta     : Electron-to-proton mass ratio (m_e / m_p ~ 5.446e-4)
    alpha_s  : Effective strong interaction coupling constant (~ 0.118)
    Q        : Relative neutron-proton mass difference ((m_n - m_p) / m_p ~ 1.378e-3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


# ============================================================================
# STANDARD UNIVERSE EMPIRICAL DIMENSIONLESS CONSTANTS (CODATA 2022)
# ============================================================================
ALPHA_STANDARD: float = 1.0 / 137.035999084  # ~ 7.297352569e-3
ALPHA_G_STANDARD: float = 5.906103e-39       # G * m_p^2 / (hbar * c)
BETA_STANDARD: float = 5.446170215e-4        # m_e / m_p (1 / 1836.152673)
ALPHA_S_STANDARD: float = 0.11800            # Effective strong coupling at low/nuclear scale
Q_STANDARD: float = 1.3784192e-3             # (m_n - m_p) / m_p = 1.29333 MeV / 938.272 MeV

# ============================================================================
# SI REFERENCE VALUES (for dimensional conversion when analyzing timescales)
# ============================================================================
HBAR_SI: float = 1.054571817e-34             # J * s
SPEED_OF_LIGHT_SI: float = 2.99792458e8      # m / s
PROTON_MASS_SI: float = 1.67262192369e-27    # kg
ELEMENTARY_CHARGE_SI: float = 1.602176634e-19# C
GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11 # m^3 / (kg * s^2)
YEAR_IN_SECONDS: float = 3.15576e7           # 365.25 days in seconds

# Natural timescale: tau_0 = hbar / (m_p * c^2) ~ 7.016e-25 seconds
CHRONO_SCALE_SI: float = HBAR_SI / (PROTON_MASS_SI * (SPEED_OF_LIGHT_SI ** 2))

# Natural length scale: Compton wavelength of proton lambda_p = hbar / (m_p * c)
COMPTON_PROTON_SI: float = HBAR_SI / (PROTON_MASS_SI * SPEED_OF_LIGHT_SI)

BOLTZMANN_K_SI: float = 1.380649e-23         # J / K (CODATA 2018/2022)


def compute_dynamic_q(
    alpha: float = ALPHA_STANDARD,
    alpha_s: float = ALPHA_S_STANDARD,
    delta_m_qcd_0_mev: float = 2.053,
    delta_m_qed_0_mev: float = -0.760,
    m_p_0_mev: float = 938.272088,
    b0: float = 9.0,
) -> float:
    """Computes dynamic neutron-proton mass difference ratio Q = (m_n - m_p) / m_p.
    
    Derived from ab initio Lattice QCD + QED (BMW Collaboration, Borsanyi et al. 2015)
    and dimensional transmutation:
        Delta m_{n-p} = Delta m_QCD(m_d - m_u) + Delta m_QED(alpha) * Lambda_QCD(alpha_s)
        Lambda_QCD(alpha_s) = Lambda_0 * exp(-2*pi / (b0 * alpha_s))
    where Delta m_QCD_0 = +2.053 MeV and Delta m_QED_0 = -0.760 MeV yield m_n - m_p = 1.293 MeV.
    """
    inv_diff = (1.0 / max(1e-4, alpha_s)) - (1.0 / ALPHA_S_STANDARD)
    exponent = -(2.0 * math.pi / b0) * inv_diff
    exponent_clamped = max(-5.0, min(5.0, exponent))
    lambda_ratio = math.exp(exponent_clamped)

    delta_m_qed = delta_m_qed_0_mev * (alpha / ALPHA_STANDARD) * lambda_ratio
    delta_m_qcd = delta_m_qcd_0_mev

    delta_m_np = delta_m_qcd + delta_m_qed
    m_p = m_p_0_mev * lambda_ratio

    return delta_m_np / max(1e-5, m_p)


@dataclass(frozen=True)
class UniverseParameters:
    """Set of dimensionless fundamental constants defining a candidate universe.
    
    Attributes:
        alpha: Electromagnetic fine-structure constant.
        alpha_G: Gravitational coupling constant for baryons.
        beta: Mass ratio of electron to proton (m_e / m_p).
        alpha_s: Strong force effective coupling.
        Q: Dimensionless neutron-proton mass difference (m_n - m_p) / m_p.
    """
    alpha: float = ALPHA_STANDARD
    alpha_G: float = ALPHA_G_STANDARD
    beta: float = BETA_STANDARD
    alpha_s: float = ALPHA_S_STANDARD
    Q: float = Q_STANDARD

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.alpha_G <= 0:
            raise ValueError(f"alpha_G must be positive, got {self.alpha_G}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.alpha_s <= 0:
            raise ValueError(f"alpha_s must be positive, got {self.alpha_s}")
        if not math.isfinite(self.Q):
            raise ValueError(f"Q must be a finite real number, got {self.Q}")

    @property
    def log_vector(self) -> Tuple[float, float, float, float]:
        """Returns log10 of (alpha, alpha_G, beta, alpha_s)."""
        return (
            math.log10(self.alpha),
            math.log10(self.alpha_G),
            math.log10(self.beta),
            math.log10(self.alpha_s),
        )

    @property
    def dynamic_q(self) -> float:
        """Returns the dynamically evaluated Q(alpha, alpha_s) from Cottingham/QCD decomposition."""
        return compute_dynamic_q(self.alpha, self.alpha_s)

    @classmethod
    def from_log_vector(
        cls,
        log_alpha: float,
        log_alpha_G: float,
        log_beta: float,
        log_alpha_s: float,
        Q: Optional[float] = None,
        use_dynamic_q: bool = True,
    ) -> UniverseParameters:
        """Constructs a UniverseParameters instance from log10-space parameters."""
        a = 10.0 ** log_alpha
        s = 10.0 ** log_alpha_s
        if Q is None:
            if use_dynamic_q:
                q_val = compute_dynamic_q(a, s)
            else:
                q_val = Q_STANDARD
        else:
            q_val = Q

        return cls(
            alpha=a,
            alpha_G=10.0 ** log_alpha_G,
            beta=10.0 ** log_beta,
            alpha_s=s,
            Q=q_val,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "alpha_G": self.alpha_G,
            "beta": self.beta,
            "alpha_s": self.alpha_s,
            "Q": self.Q,
        }

    @classmethod
    def standard(cls) -> UniverseParameters:
        """Returns the Standard Model / Observed Universe parameters."""
        return cls()

    @property
    def bohr_radius_compton(self) -> float:
        """Bohr radius in units of proton Compton wavelength: a_0 / lambda_p = 1 / (alpha * beta)."""
        return 1.0 / (self.alpha * self.beta)

    @property
    def stellar_lifetime_seconds(self) -> float:
        """Typical stellar lifetime tau_* ~ alpha^(-2) * alpha_G^(-1) * (hbar / (m_p * c^2))."""
        return (self.alpha ** -2.0) * (self.alpha_G ** -1.0) * CHRONO_SCALE_SI

    @property
    def stellar_lifetime_years(self) -> float:
        """Stellar lifetime in equivalent Earth years."""
        return self.stellar_lifetime_seconds / YEAR_IN_SECONDS
