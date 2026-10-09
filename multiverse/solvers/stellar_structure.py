"""Astrophysical Stellar Structure and Nuclear Fusion Solver.

Solves the stellar structure equations and Lane-Emden polytropes:
    (1 / xi^2) * d/d_xi (xi^2 * d_theta / d_xi) = - theta^n
for:
    n = 1.5 : Non-relativistic degenerate gas (white dwarfs, brown dwarfs).
    n = 3.0 : Relativistic degenerate gas / radiation dominance (Chandrasekhar limit).

Integrates the full 4 differential structure equations across fundamental constants:
    1. Hydrostatic equilibrium:  dP/dr = - G M(r) rho / r^2
    2. Mass conservation:        dM/dr = 4 pi r^2 rho
    3. Energy generation:        dL/dr = 4 pi r^2 rho epsilon_nuc(rho, T)
       with quantum tunneling Gamow peak E_0(alpha, alpha_s) for p-p chain and CNO cycle.
    4. Energy transport:         dT/dr = - (G M(r) rho / r^2) * (T / P) * nabla
       where nabla = min(nabla_rad, nabla_ad) via Schwarzschild convective criterion.

Evaluates whether the candidate universe supports stable main-sequence stars:
    M_min (Gamow barrier hydrogen ignition) < M_Ch (Chandrasekhar collapse limit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import numba

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    PROTON_MASS_SI,
    SPEED_OF_LIGHT_SI,
    GRAVITATIONAL_CONSTANT_SI,
    HBAR_SI,
    YEAR_IN_SECONDS,
    UniverseParameters,
)
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Reference Astronomical & Physical Constants in SI
SOLAR_MASS_KG: float = 1.98847e30
SOLAR_RADIUS_M: float = 6.957e8
SOLAR_LUMINOSITY_W: float = 3.828e26
BOLTZMANN_K_SI: float = 1.380649e-23
RADIATION_A_SI: float = 7.5657e-16            # J / (m^3 * K^4)
STEFAN_BOLTZMANN_SI: float = 5.670374e-8      # W / (m^2 * K^4)
KEV_TO_JOULES: float = 1.602176634e-16        # 1 keV in Joules
GYR_IN_SECONDS: float = 1e9 * YEAR_IN_SECONDS # 1 Gyr in seconds

# Standard solar composition & interior calibration
SOLAR_CORE_T_K: float = 1.57e7
SOLAR_CORE_RHO_SI: float = 1.50e5             # 150 g/cm^3
SOLAR_CORE_P_SI: float = 3.19e16              # Pa
SOLAR_MU: float = 0.61                        # Mean molecular weight for ionized solar mix
SOLAR_X: float = 0.7381                       # Primordial/solar Hydrogen mass fraction


@numba.njit(fastmath=True)
def integrate_lane_emden_rk4(
    n: float,
    dxi: float = 1e-4,
    xi_max: float = 30.0,
    xi_start: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Integrates Lane-Emden differential equation using RK4 until theta <= 0.
    
    Returns:
        (xi_arr, theta_arr, xi_1, omega_n)
        where xi_1 is the surface coordinate theta(xi_1) = 0
        and omega_n = - xi_1^2 * theta'(xi_1).
    """
    max_steps = int(xi_max / dxi) + 100
    xi_out = np.zeros(max_steps, dtype=np.float64)
    th_out = np.zeros(max_steps, dtype=np.float64)

    # Initial condition near origin via Taylor expansion:
    # theta(xi) ~ 1 - xi^2 / 6 + n * xi^4 / 120
    # v(xi) = d_theta/d_xi ~ - xi / 3 + n * xi^3 / 30
    xi = xi_start
    th = 1.0 - (xi * xi) / 6.0 + n * (xi ** 4) / 120.0
    v = -xi / 3.0 + n * (xi ** 3) / 30.0

    xi_out[0] = 0.0
    th_out[0] = 1.0
    xi_out[1] = xi
    th_out[1] = th

    step = 1
    while xi < xi_max and th > 0.0:
        # System: d_theta/d_xi = v
        #         dv/d_xi = - theta^n - (2 / xi) * v
        k1_th = v
        k1_v = -(th ** n) - (2.0 / xi) * v

        xi_mid = xi + 0.5 * dxi
        th_k2 = th + 0.5 * dxi * k1_th
        v_k2 = v + 0.5 * dxi * k1_v
        if th_k2 < 0.0:
            th_k2 = 0.0
        k2_th = v_k2
        k2_v = -(th_k2 ** n) - (2.0 / xi_mid) * v_k2

        th_k3 = th + 0.5 * dxi * k2_th
        v_k3 = v + 0.5 * dxi * k2_v
        if th_k3 < 0.0:
            th_k3 = 0.0
        k3_th = v_k3
        k3_v = -(th_k3 ** n) - (2.0 / xi_mid) * v_k3

        xi_end = xi + dxi
        th_k4 = th + dxi * k3_th
        v_k4 = v + dxi * k3_v
        if th_k4 < 0.0:
            th_k4 = 0.0
        k4_th = v_k4
        k4_v = -(th_k4 ** n) - (2.0 / xi_end) * v_k4

        th_next = th + (dxi / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
        v_next = v + (dxi / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        xi += dxi
        th = th_next
        v = v_next

        step += 1
        xi_out[step] = xi
        th_out[step] = th

        if th <= 0.0:
            break

    # Linear interpolation for precise root xi_1
    if step > 2 and th_out[step] <= 0.0:
        th_prev = th_out[step - 1]
        xi_prev = xi_out[step - 1]
        th_curr = th_out[step]
        xi_curr = xi_out[step]
        frac = th_prev / (th_prev - th_curr)
        xi_1 = xi_prev + frac * (xi_curr - xi_prev)
    else:
        xi_1 = xi

    omega_n = -(xi_1 ** 2.0) * v
    return xi_out[:step + 1], th_out[:step + 1], xi_1, omega_n


def solve_lane_emden(
    n: float,
    dxi: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Backward-compatible helper function to integrate the Lane-Emden equation.
    
    Returns:
        (xi_array, theta_array, xi_surface, omega_n)
    """
    return integrate_lane_emden_rk4(n, dxi=dxi)


@dataclass
class ZAMSSolution:
    """1D Zero-Age Main-Sequence (ZAMS) stellar structure solution from RK45 shooting method."""
    converged: bool
    mass_solar: float
    mass_kg: float
    radius_m: float
    radius_solar: float
    luminosity_watts: float
    luminosity_solar: float
    central_temp_kelvin: float
    central_density_kg_m3: float
    central_pressure_pa: float
    effective_temp_kelvin: float
    r_grid: np.ndarray
    m_grid: np.ndarray
    l_grid: np.ndarray
    p_grid: np.ndarray
    t_grid: np.ndarray
    rho_grid: np.ndarray
    iterations: int = 0


@dataclass
class PolytropeSolution:
    """Solution of the Lane-Emden polytropic structure."""
    index_n: float
    xi_surface: float
    omega_n: float
    xi_array: np.ndarray
    theta_array: np.ndarray


@dataclass
class StellarModel1D:
    """One-dimensional radial interior profile of a star."""
    r_array_m: np.ndarray
    r_array_solar: np.ndarray
    enclosed_mass_kg: np.ndarray
    enclosed_mass_solar: np.ndarray
    density_kg_m3: np.ndarray
    pressure_pa: np.ndarray
    temperature_kelvin: np.ndarray
    enclosed_luminosity_watts: np.ndarray
    enclosed_luminosity_solar: np.ndarray
    radiative_gradient: np.ndarray
    actual_gradient: np.ndarray
    specific_energy_generation: np.ndarray


@dataclass
class StellarEquilibriumSolution:
    """Detailed equilibrium state of a stable star."""
    stellar_mass_kg: float
    stellar_mass_solar: float
    radius_m: float
    radius_solar: float
    luminosity_watts: float
    luminosity_solar: float
    central_temp_kelvin: float
    central_density_kg_m3: float
    central_pressure_pa: float
    effective_temp_kelvin: float
    gamow_peak_kev: float
    gamow_window_width_kev: float
    pp_luminosity_fraction: float
    cno_luminosity_fraction: float
    eddington_luminosity_solar: float
    eddington_fraction: float
    has_convective_core: bool
    main_sequence_lifetime_gyr: float
    spectral_class: str
    profile_1d: Optional[StellarModel1D] = None


@dataclass
class StellarViabilityResult:
    """Astrophysical assessment of stellar equilibrium and mass boundaries."""
    universe: UniverseParameters
    polytrope_n15: PolytropeSolution
    polytrope_n30: PolytropeSolution
    chandrasekhar_mass_solar: float      # in units of standard solar mass
    min_ignition_mass_solar: float       # in units of standard solar mass
    max_stable_mass_solar: float         # in units of standard solar mass
    mass_window_ratio: float             # M_Ch / M_min
    has_stable_nuclear_burning: bool
    solar_analog: Optional[StellarEquilibriumSolution] = None
    stellar_population: List[StellarEquilibriumSolution] = field(default_factory=list)
    zams_solution: Optional[ZAMSSolution] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)


class StellarStructureSolver:
    """High-precision stellar interior and nuclear fusion solver."""

    def __init__(self, dxi: float = 2e-4) -> None:
        self.dxi = dxi
        self._sol_n15 = self.solve_polytrope(1.5)
        self._sol_n30 = self.solve_polytrope(3.0)

    def solve_polytrope(self, n: float) -> PolytropeSolution:
        """Integrates Lane-Emden equation for polytropic index n."""
        xi_arr, th_arr, xi_1, omega_n = integrate_lane_emden_rk4(n, dxi=self.dxi)
        return PolytropeSolution(
            index_n=n,
            xi_surface=xi_1,
            omega_n=omega_n,
            xi_array=xi_arr,
            theta_array=th_arr,
        )

    def solve_lane_emden(self, n: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Backward-compatible method returning (xi_array, theta_array, xi_surface, omega_n)."""
        sol = self.solve_polytrope(n)
        return sol.xi_array, sol.theta_array, sol.xi_surface, sol.omega_n

    def compute_gamow_parameters(
        self,
        T_kelvin: float,
        u: UniverseParameters,
    ) -> Tuple[float, float, float, float]:
        """Calculates Gamow energy, Gamow peak E_0, width delta_E_0, and exponent tau_pp.
        
        Returns:
            (E_G_kev, E_0_kev, delta_E_kev, tau_pp)
        """
        # Gamow energy: E_G = 2 m_r c^2 (pi alpha)^2 = pi^2 alpha^2 m_p c^2
        E_G_joules = (math.pi * u.alpha) ** 2.0 * PROTON_MASS_SI * (SPEED_OF_LIGHT_SI ** 2.0)
        E_G_kev = E_G_joules / KEV_TO_JOULES

        k_B_T_joules = BOLTZMANN_K_SI * max(T_kelvin, 1.0)
        k_B_T_kev = k_B_T_joules / KEV_TO_JOULES

        tau_pp = 3.0 * ((E_G_joules / (4.0 * k_B_T_joules)) ** (1.0 / 3.0))
        E_0_kev = (tau_pp * k_B_T_kev) / 3.0
        delta_E_kev = 4.0 * math.sqrt(max(0.0, (E_0_kev * k_B_T_kev) / 3.0))

        return E_G_kev, E_0_kev, delta_E_kev, tau_pp

    def compute_nuclear_generation_rates(
        self,
        rho: float,
        T: float,
        u: UniverseParameters,
    ) -> Tuple[float, float, float]:
        """Computes specific nuclear generation rates (epsilon_pp, epsilon_cno, epsilon_tot) in W/kg."""
        _, _, _, tau_pp = self.compute_gamow_parameters(T, u)

        # Strong coupling enhancement: S(0) ~ alpha_s^2
        s_strong = (u.alpha_s / ALPHA_S_STANDARD) ** 2.0
        rho_ratio = rho / SOLAR_CORE_RHO_SI

        # p-p chain rate calibrated to solar core (0.017 W/kg at rho=1.5e5, T=1.57e7)
        eps_pp = 0.017 * rho_ratio * s_strong * ((tau_pp / 13.50) ** 2.0) * math.exp(
            max(-100.0, min(50.0, 13.50 - tau_pp))
        )

        # CNO cycle: Z_1=1, Z_2=7 -> E_G_cno ~ 91.47 * E_G_pp -> tau_cno ~ 4.50 * tau_pp
        tau_cno = 4.506 * tau_pp
        eps_cno = 0.00017 * rho_ratio * s_strong * ((tau_cno / 60.83) ** 2.0) * math.exp(
            max(-100.0, min(50.0, 60.83 - tau_cno))
        )

        eps_tot = eps_pp + eps_cno
        return eps_pp, eps_cno, eps_tot

    def compute_opacity(
        self,
        rho: float,
        T: float,
        u: UniverseParameters,
    ) -> Tuple[float, float, float]:
        """Computes Thomson electron scattering and Kramers opacities (m^2/kg)."""
        # Thomson opacity: kappa_es ~ (alpha / beta)^2
        kappa_es = 0.0348 * ((u.alpha / ALPHA_STANDARD) ** 2.0) * ((BETA_STANDARD / u.beta) ** 2.0)

        # Kramers free-free / bound-free opacity: kappa_Kr ~ alpha^3 beta^(-1.5) rho T^(-3.5)
        scale_kr = ((u.alpha / ALPHA_STANDARD) ** 3.0) * ((BETA_STANDARD / u.beta) ** 1.5)
        T_safe = max(T, 1e4)
        kappa_kr = 1.05e19 * scale_kr * rho * (T_safe ** -3.5)

        kappa_tot = kappa_es + kappa_kr
        return kappa_es, kappa_kr, kappa_tot

    def solve_stellar_equilibrium(
        self,
        mass_solar: float,
        u: UniverseParameters,
        compute_profile: bool = True,
    ) -> StellarEquilibriumSolution:
        """Solves self-consistent hydrostatic, energy, and transport equilibrium for a star."""
        # Scale mass to candidate universe gravitational mass scale
        mass_scale_universe = (ALPHA_G_STANDARD / u.alpha_G) ** 1.5
        M_star_kg = mass_solar * SOLAR_MASS_KG * mass_scale_universe

        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio

        # Relativistic mass ratio relative to standard solar
        m_rel = mass_solar

        def evaluate_radius(r_rel: float) -> Tuple[float, float, float, float, float, float, float]:
            r_rel_clamped = max(1e-4, min(1e4, r_rel))
            # Core temperature and density from Lane-Emden n=3 polytrope
            T_c = SOLAR_CORE_T_K * (m_rel * grav_ratio / r_rel_clamped)
            rho_c = SOLAR_CORE_RHO_SI * (m_rel / (r_rel_clamped ** 3.0))

            eps_pp, eps_cno, eps_tot = self.compute_nuclear_generation_rates(rho_c, T_c, u)
            # Profile central concentration factor: f_pp ~ 0.011324
            f_profile = 0.011324
            L_nuc = M_star_kg * (eps_pp * 0.99 + eps_cno) * f_profile

            _, _, kappa_tot = self.compute_opacity(rho_c, T_c, u)
            kappa_std_core = 0.1375
            kappa_ratio = max(1e-4, kappa_tot / kappa_std_core)

            # Radiative energy transport luminosity
            L_rad = (
                SOLAR_LUMINOSITY_W
                * (m_rel ** 3.0)
                * (grav_ratio ** 4.0)
                / kappa_ratio
                * ((T_c / SOLAR_CORE_T_K) ** 0.5)
            )
            return L_nuc, L_rad, T_c, rho_c, eps_pp, eps_cno, eps_tot

        # Monotonic bisection root search for L_nuc(R) = L_rad(R)
        r_lo = 0.001
        r_hi = 100.0
        for _ in range(60):
            r_mid = 0.5 * (r_lo + r_hi)
            l_nuc, l_rad, _, _, _, _, _ = evaluate_radius(r_mid)
            if l_nuc > l_rad:
                r_lo = r_mid
            else:
                r_hi = r_mid

        r_equilibrium = 0.5 * (r_lo + r_hi)
        l_nuc, l_rad, T_c, rho_c, eps_pp, eps_cno, eps_tot = evaluate_radius(r_equilibrium)

        R_star_m = r_equilibrium * SOLAR_RADIUS_M * (mass_scale_universe ** (1.0 / 3.0))
        L_star_w = l_rad

        # Core pressure: P_c = rho_c * k_B * T_c / (mu * m_p) + (a * T_c^4) / 3
        P_gas_c = (rho_c * BOLTZMANN_K_SI * T_c) / (SOLAR_MU * PROTON_MASS_SI)
        P_rad_c = (RADIATION_A_SI * (T_c ** 4.0)) / 3.0
        P_c = P_gas_c + P_rad_c

        # Effective surface temperature from Stefan-Boltzmann law
        T_eff = ((L_star_w / (4.0 * math.pi * STEFAN_BOLTZMANN_SI * (R_star_m ** 2.0))) ** 0.25)

        # Gamow tunneling metrics
        E_G_kev, E_0_kev, delta_E_kev, _ = self.compute_gamow_parameters(T_c, u)

        # Energy fractions
        pp_frac = eps_pp / max(eps_tot, 1e-30)
        cno_frac = eps_cno / max(eps_tot, 1e-30)

        # Convective core check: CNO dominance (steep T sensitivity) drives convective core
        has_convective_core = cno_frac > 0.5 or (m_rel > 1.3)

        # Eddington luminosity: L_Edd = 4 pi G M c / kappa_es
        kappa_es, _, _ = self.compute_opacity(rho_c, T_c, u)
        L_edd_w = (4.0 * math.pi * G_u * M_star_kg * SPEED_OF_LIGHT_SI) / max(kappa_es, 1e-5)
        L_edd_solar = L_edd_w / SOLAR_LUMINOSITY_W
        edd_fraction = L_star_w / max(L_edd_w, 1e-30)

        # Main-sequence nuclear lifetime
        # E_nuc = 0.10 * M * X * eta_fusion * c^2, where eta_fusion ~ 0.0071 * (alpha_s / alpha_s0)
        eta_fusion = 0.0071 * (u.alpha_s / ALPHA_S_STANDARD)
        total_fusion_energy = 0.10 * M_star_kg * SOLAR_X * eta_fusion * (SPEED_OF_LIGHT_SI ** 2.0)
        tau_ms_gyr = total_fusion_energy / max(L_star_w, 1e-30) / GYR_IN_SECONDS

        # Spectral classification
        spectral_class = self._classify_spectral_type(T_eff)

        # Generate 1D interior profile if requested
        profile_1d = None
        if compute_profile:
            profile_1d = self.integrate_stellar_interior(
                mass_kg=M_star_kg,
                radius_m=R_star_m,
                T_c=T_c,
                rho_c=rho_c,
                u=u,
                n_points=250,
            )

        return StellarEquilibriumSolution(
            stellar_mass_kg=M_star_kg,
            stellar_mass_solar=mass_solar,
            radius_m=R_star_m,
            radius_solar=R_star_m / SOLAR_RADIUS_M,
            luminosity_watts=L_star_w,
            luminosity_solar=L_star_w / SOLAR_LUMINOSITY_W,
            central_temp_kelvin=T_c,
            central_density_kg_m3=rho_c,
            central_pressure_pa=P_c,
            effective_temp_kelvin=T_eff,
            gamow_peak_kev=E_0_kev,
            gamow_window_width_kev=delta_E_kev,
            pp_luminosity_fraction=pp_frac,
            cno_luminosity_fraction=cno_frac,
            eddington_luminosity_solar=L_edd_solar,
            eddington_fraction=edd_fraction,
            has_convective_core=has_convective_core,
            main_sequence_lifetime_gyr=tau_ms_gyr,
            spectral_class=spectral_class,
            profile_1d=profile_1d,
        )

    def integrate_stellar_interior(
        self,
        mass_kg: float,
        radius_m: float,
        T_c: float,
        rho_c: float,
        u: UniverseParameters,
        n_points: int = 250,
    ) -> StellarModel1D:
        """Computes 1D radial interior profile mapped from the Lane-Emden polytrope."""
        xi_surface = self._sol_n30.xi_surface
        xi_arr = self._sol_n30.xi_array
        th_arr = self._sol_n30.theta_array

        r_rel = np.linspace(1e-4, 1.0, n_points)
        r_m = r_rel * radius_m
        xi_grid = r_rel * xi_surface

        # Interpolate theta(xi) from exact RK4 integration
        theta = np.interp(xi_grid, xi_arr, th_arr)
        theta = np.maximum(0.0, theta)

        density = rho_c * (theta ** 3.0)
        temperature = T_c * theta

        P_c = (rho_c * BOLTZMANN_K_SI * T_c) / (SOLAR_MU * PROTON_MASS_SI) + (RADIATION_A_SI * (T_c ** 4.0)) / 3.0
        pressure = P_c * (theta ** 4.0)

        # Enclosed mass and luminosity numerical integrals
        dr = r_m[1] - r_m[0]
        m_enclosed = np.zeros(n_points)
        l_enclosed = np.zeros(n_points)
        eps_nuc = np.zeros(n_points)

        for i in range(n_points):
            eps_pp, eps_cno, eps_tot = self.compute_nuclear_generation_rates(density[i], temperature[i], u)
            eps_nuc[i] = eps_tot

        # Trapezoidal cumulative sums
        dm_dr = 4.0 * math.pi * (r_m ** 2.0) * density
        dl_dr = 4.0 * math.pi * (r_m ** 2.0) * density * eps_nuc

        m_enclosed = np.cumsum(dm_dr) * dr
        l_enclosed = np.cumsum(dl_dr) * dr

        # Normalization to exact boundary values
        if m_enclosed[-1] > 0:
            m_enclosed = m_enclosed * (mass_kg / m_enclosed[-1])

        # Temperature gradients: Schwarzschild criterion
        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio
        rad_grad = np.zeros(n_points)
        act_grad = np.zeros(n_points)

        for i in range(n_points):
            if m_enclosed[i] > 0 and temperature[i] > 1e4:
                _, _, kappa = self.compute_opacity(density[i], temperature[i], u)
                rad_grad[i] = (
                    (3.0 * kappa * pressure[i] * l_enclosed[i])
                    / (16.0 * math.pi * RADIATION_A_SI * SPEED_OF_LIGHT_SI * G_u * m_enclosed[i] * (temperature[i] ** 4.0))
                )
            else:
                rad_grad[i] = 0.4
            # Schwarzschild criterion: convection if rad_grad > 0.4
            act_grad[i] = min(0.40, rad_grad[i])

        return StellarModel1D(
            r_array_m=r_m,
            r_array_solar=r_m / SOLAR_RADIUS_M,
            enclosed_mass_kg=m_enclosed,
            enclosed_mass_solar=m_enclosed / SOLAR_MASS_KG,
            density_kg_m3=density,
            pressure_pa=pressure,
            temperature_kelvin=temperature,
            enclosed_luminosity_watts=l_enclosed,
            enclosed_luminosity_solar=l_enclosed / SOLAR_LUMINOSITY_W,
            radiative_gradient=rad_grad,
            actual_gradient=act_grad,
            specific_energy_generation=eps_nuc,
        )

    def solve_zams(
        self,
        mass_solar: float = 1.0,
        u: Optional[UniverseParameters] = None,
    ) -> ZAMSSolution:
        """Solves 1D Zero-Age Main-Sequence (ZAMS) stellar structure via shooting method coupled with RK45.
        
        Differential equations:
            dP/dr = - G M rho / r^2
            dM/dr = 4 pi r^2 rho
            dL/dr = 4 pi r^2 rho epsilon_pp(rho, T)
            dT/dr = - 3 kappa rho L / (16 pi a c r^2 T^3)
        coupled with Schwarzschild convection criterion.
        
        Microphysics:
            kappa = kappa_es + kappa_ff
            where kappa_es scales as alpha^2 / (beta^2 m_p^2)
            and kappa_ff scales as alpha^3 beta^(-7/2) rho T^(-7/2).
            Nuclear generation epsilon_pp integrates Gamow tunneling peak E_0(alpha, alpha_s).
            
        Boundary conditions:
            Center: M(0) = 0, L(0) = 0
            Surface: P(R_*) -> 0, T(R_*)^4 = L(R_*) / (4 pi sigma_SB R_*^2)
        """
        if u is None:
            u = UniverseParameters.standard()

        scale_gravity = (ALPHA_G_STANDARD / u.alpha_G) ** 1.5
        grav_ratio = u.alpha_G / ALPHA_G_STANDARD
        G_u = GRAVITATIONAL_CONSTANT_SI * grav_ratio
        M_target = mass_solar * SOLAR_MASS_KG * scale_gravity

        # Microphysical constants
        m_p = PROTON_MASS_SI
        c = SPEED_OF_LIGHT_SI
        a = RADIATION_A_SI
        k_B = BOLTZMANN_K_SI
        mu = SOLAR_MU
        sigma_sb = STEFAN_BOLTZMANN_SI

        E_G = (math.pi * u.alpha) ** 2.0 * m_p * (c ** 2.0)
        s_strong = (u.alpha_s / ALPHA_S_STANDARD) ** 2.0
        k_es = 0.0348 * ((u.alpha / ALPHA_STANDARD) ** 2.0) * ((BETA_STANDARD / u.beta) ** 2.0)
        k_ff_coeff = 1.05e19 * ((u.alpha / ALPHA_STANDARD) ** 3.0) * ((BETA_STANDARD / u.beta) ** 3.5)

        r0 = 1e-4 * SOLAR_RADIUS_M * (scale_gravity ** (1.0 / 3.0))

        def shooting_residuals(params: np.ndarray) -> List[float]:
            log_rho, log_T = params
            rho_c = math.exp(log_rho)
            T_c = math.exp(log_T)

            P_c = (rho_c * k_B * T_c) / (mu * m_p) + (a * (T_c ** 4.0)) / 3.0

            tau_0 = 3.0 * ((E_G / (4.0 * k_B * T_c)) ** (1.0 / 3.0))
            eps_0 = 0.0035 * (rho_c / SOLAR_CORE_RHO_SI) * s_strong * ((tau_0 / 13.5) ** 2.0) * math.exp(
                max(-50.0, min(50.0, 13.5 - tau_0))
            )

            M0 = (4.0 / 3.0) * math.pi * (r0 ** 3.0) * rho_c
            L0 = (4.0 / 3.0) * math.pi * (r0 ** 3.0) * rho_c * eps_0
            P0 = P_c - (2.0 / 3.0) * math.pi * G_u * (rho_c ** 2.0) * (r0 ** 2.0)

            kappa_0 = k_es + k_ff_coeff * rho_c * (T_c ** -3.5)
            T0 = T_c - (3.0 * kappa_0 * rho_c * L0) / (16.0 * math.pi * a * c * (r0 ** 2.0) * (T_c ** 3.0)) * (r0 / 2.0)

            def odes(r: float, y: np.ndarray) -> List[float]:
                P, M, L, T = y
                if P <= 1e2 or T <= 1e3 or r <= 0:
                    return [0.0, 0.0, 0.0, 0.0]

                rho = max(1e-12, (P - (a * (T ** 4.0)) / 3.0) * (mu * m_p) / (k_B * T))

                tau = 3.0 * ((E_G / (4.0 * k_B * max(T, 1e4))) ** (1.0 / 3.0))
                eps = 0.0035 * (rho / SOLAR_CORE_RHO_SI) * s_strong * ((tau / 13.5) ** 2.0) * math.exp(
                    max(-50.0, min(50.0, 13.5 - tau))
                )

                kappa = k_es + k_ff_coeff * rho * (max(T, 1e4) ** -3.5)

                dP = - (G_u * M * rho) / (r ** 2.0)
                dM = 4.0 * math.pi * (r ** 2.0) * rho
                dL = 4.0 * math.pi * (r ** 2.0) * rho * eps

                dT_rad = - (3.0 * kappa * rho * L) / (16.0 * math.pi * a * c * (r ** 2.0) * (T ** 3.0))
                dT_ad = - 0.40 * (T / P) * (G_u * M * rho / (r ** 2.0))
                dT = max(dT_rad, dT_ad)

                return [dP, dM, dL, dT]

            def surf_event(r: float, y: np.ndarray) -> float:
                return y[0] - 1e-4 * P_c

            surf_event.terminal = True
            surf_event.direction = -1

            sol = solve_ivp(
                odes,
                (r0, 5.0 * SOLAR_RADIUS_M * (scale_gravity ** (1.0 / 3.0))),
                [P0, M0, L0, T0],
                method='RK45',
                events=surf_event,
                rtol=1e-4,
                atol=1e-4,
            )

            if len(sol.t_events[0]) == 0:
                return [10.0, 10.0]

            r_surf = sol.t_events[0][0]
            M_surf = sol.y_events[0][0][1]
            L_surf = max(1e20, sol.y_events[0][0][2])
            T_surf = sol.y_events[0][0][3]
            T_eff = (L_surf / (4.0 * math.pi * sigma_sb * (r_surf ** 2.0))) ** 0.25

            res_M = (M_surf - M_target) / M_target
            res_T = (T_surf - T_eff) / T_eff
            return [res_M, res_T]

        guess_rho = math.log(1.0e5 * mass_solar)
        guess_T = math.log(1.5e7 * (mass_solar ** 0.2))

        opt = least_squares(
            shooting_residuals,
            [guess_rho, guess_T],
            bounds=([math.log(1e3), math.log(1.2e7)], [math.log(5e6), math.log(2.0e7)]),
            ftol=1e-3,
            xtol=1e-3,
            max_nfev=25,
        )

        rho_c = math.exp(opt.x[0])
        T_c = math.exp(opt.x[1])
        P_c = (rho_c * k_B * T_c) / (mu * m_p) + (a * (T_c ** 4.0)) / 3.0

        # High-resolution reconstructed profile
        tau_0 = 3.0 * ((E_G / (4.0 * k_B * T_c)) ** (1.0 / 3.0))
        eps_0 = 0.0035 * (rho_c / SOLAR_CORE_RHO_SI) * s_strong * ((tau_0 / 13.5) ** 2.0) * math.exp(
            max(-50.0, min(50.0, 13.5 - tau_0))
        )
        M0 = (4.0 / 3.0) * math.pi * (r0 ** 3.0) * rho_c
        L0 = (4.0 / 3.0) * math.pi * (r0 ** 3.0) * rho_c * eps_0
        P0 = P_c - (2.0 / 3.0) * math.pi * G_u * (rho_c ** 2.0) * (r0 ** 2.0)
        kappa_0 = k_es + k_ff_coeff * rho_c * (T_c ** -3.5)
        T0 = T_c - (3.0 * kappa_0 * rho_c * L0) / (16.0 * math.pi * a * c * (r0 ** 2.0) * (T_c ** 3.0)) * (r0 / 2.0)

        def odes_final(r: float, y: np.ndarray) -> List[float]:
            P, M, L, T = y
            if P <= 1e2 or T <= 1e3 or r <= 0:
                return [0.0, 0.0, 0.0, 0.0]
            rho = max(1e-12, (P - (a * (T ** 4.0)) / 3.0) * (mu * m_p) / (k_B * T))
            tau = 3.0 * ((E_G / (4.0 * k_B * max(T, 1e4))) ** (1.0 / 3.0))
            eps = 0.0035 * (rho / SOLAR_CORE_RHO_SI) * s_strong * ((tau / 13.5) ** 2.0) * math.exp(
                max(-50.0, min(50.0, 13.5 - tau))
            )
            kappa = k_es + k_ff_coeff * rho * (max(T, 1e4) ** -3.5)
            dP = - (G_u * M * rho) / (r ** 2.0)
            dM = 4.0 * math.pi * (r ** 2.0) * rho
            dL = 4.0 * math.pi * (r ** 2.0) * rho * eps
            dT_rad = - (3.0 * kappa * rho * L) / (16.0 * math.pi * a * c * (r ** 2.0) * (T ** 3.0))
            dT_ad = - 0.40 * (T / P) * (G_u * M * rho / (r ** 2.0))
            return [dP, dM, dL, max(dT_rad, dT_ad)]

        def surf_event_final(r: float, y: np.ndarray) -> float:
            return y[0] - 1e-4 * P_c
        surf_event_final.terminal = True
        surf_event_final.direction = -1

        sol_final = solve_ivp(
            odes_final,
            (r0, 5.0 * SOLAR_RADIUS_M * (scale_gravity ** (1.0 / 3.0))),
            [P0, M0, L0, T0],
            method='RK45',
            events=surf_event_final,
            rtol=1e-5,
            atol=1e-5,
        )

        r_grid = sol_final.t
        p_grid = sol_final.y[0]
        m_grid = sol_final.y[1]
        l_grid = sol_final.y[2]
        t_grid = sol_final.y[3]
        rho_grid = np.maximum(1e-12, (p_grid - (a * (t_grid ** 4.0)) / 3.0) * (mu * m_p) / (k_B * np.maximum(t_grid, 1.0)))

        R_star = r_grid[-1]
        M_star = m_grid[-1]
        L_star = max(1e20, l_grid[-1])
        T_eff = (L_star / (4.0 * math.pi * sigma_sb * (R_star ** 2.0))) ** 0.25

        return ZAMSSolution(
            converged=bool(opt.success),
            mass_solar=mass_solar,
            mass_kg=M_star,
            radius_m=R_star,
            radius_solar=R_star / SOLAR_RADIUS_M,
            luminosity_watts=L_star,
            luminosity_solar=L_star / SOLAR_LUMINOSITY_W,
            central_temp_kelvin=T_c,
            central_density_kg_m3=rho_c,
            central_pressure_pa=P_c,
            effective_temp_kelvin=T_eff,
            r_grid=r_grid,
            m_grid=m_grid,
            l_grid=l_grid,
            p_grid=p_grid,
            t_grid=t_grid,
            rho_grid=rho_grid,
            iterations=int(opt.nfev),
        )

    def evaluate_stellar_viability(self, u: UniverseParameters) -> StellarViabilityResult:
        """Evaluates whether this universe supports stable hydrogen-burning stars."""
        omega_3 = self._sol_n30.omega_n

        # 1. Chandrasekhar Mass Limit:
        # M_Ch = C_Ch * alpha_G^(-3/2) * m_p
        m_ch_standard_solar = 1.44
        scale_gravity = (ALPHA_G_STANDARD / u.alpha_G) ** 1.5
        m_ch_solar = m_ch_standard_solar * scale_gravity

        # 2. Minimum Hydrogen Ignition Mass:
        # Burrows & Ostriker / Adams scaling:
        # M_min ~ C_min * alpha_G^(-3/2) * beta^(3/4) * (alpha / alpha_s)^(3/2) * m_p
        m_min_standard_solar = 0.080
        scale_electron = (u.beta / BETA_STANDARD) ** 0.75
        scale_coulomb_nuclear = ((u.alpha / ALPHA_STANDARD) / (u.alpha_s / ALPHA_S_STANDARD)) ** 1.5
        m_min_solar = m_min_standard_solar * scale_gravity * scale_electron * scale_coulomb_nuclear

        # 3. Maximum Stable Stellar Mass (Eddington radiation limit):
        # M_max ~ 120 * M_sun * (alpha_G_0 / u.alpha_G)^1.5 * (alpha_0 / u.alpha)^2
        m_max_solar = 120.0 * scale_gravity * ((ALPHA_STANDARD / u.alpha) ** 2.0)

        # 4. Viability Condition:
        # M_min must be strictly less than M_Ch and M_max to allow stable main-sequence stars
        mass_window_ratio = m_ch_solar / max(m_min_solar, 1e-30)
        has_stable_burning = (m_min_solar < m_ch_solar) and (m_min_solar < m_max_solar)

        # 5. Compute Solar-Analog Star and Population Spectrum
        solar_analog = None
        stellar_population: List[StellarEquilibriumSolution] = []
        zams_sol: Optional[ZAMSSolution] = None

        if has_stable_burning:
            try:
                solar_analog = self.solve_stellar_equilibrium(mass_solar=1.0, u=u, compute_profile=True)
                sample_masses = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
                for m_sample in sample_masses:
                    if m_min_solar / scale_gravity < m_sample < m_max_solar / scale_gravity:
                        star_sol = self.solve_stellar_equilibrium(m_sample, u=u, compute_profile=False)
                        stellar_population.append(star_sol)
            except Exception:
                pass

            try:
                zams_sol = self.solve_zams(mass_solar=1.0, u=u)
            except Exception:
                zams_sol = None

        diagnostics = {
            "m_ch_solar": m_ch_solar,
            "m_min_solar": m_min_solar,
            "m_max_solar": m_max_solar,
            "mass_window_ratio": mass_window_ratio,
            "log10_window_ratio": math.log10(max(mass_window_ratio, 1e-10)),
            "xi_1_n15": self._sol_n15.xi_surface,
            "omega_n15": self._sol_n15.omega_n,
            "xi_1_n30": self._sol_n30.xi_surface,
            "omega_n30": self._sol_n30.omega_n,
        }

        if solar_analog:
            diagnostics.update({
                "solar_core_temp_k": solar_analog.central_temp_kelvin,
                "solar_core_density_si": solar_analog.central_density_kg_m3,
                "solar_gamow_peak_kev": solar_analog.gamow_peak_kev,
                "solar_gamow_width_kev": solar_analog.gamow_window_width_kev,
                "solar_radius_solar": solar_analog.radius_solar,
                "solar_luminosity_solar": solar_analog.luminosity_solar,
                "solar_lifetime_gyr": solar_analog.main_sequence_lifetime_gyr,
            })

        if zams_sol and zams_sol.converged:
            diagnostics.update({
                "zams_core_temp_k": zams_sol.central_temp_kelvin,
                "zams_core_density_si": zams_sol.central_density_kg_m3,
                "zams_radius_solar": zams_sol.radius_solar,
                "zams_luminosity_solar": zams_sol.luminosity_solar,
            })

        return StellarViabilityResult(
            universe=u,
            polytrope_n15=self._sol_n15,
            polytrope_n30=self._sol_n30,
            chandrasekhar_mass_solar=m_ch_solar,
            min_ignition_mass_solar=m_min_solar,
            max_stable_mass_solar=m_max_solar,
            mass_window_ratio=mass_window_ratio,
            has_stable_nuclear_burning=has_stable_burning,
            solar_analog=solar_analog,
            stellar_population=stellar_population,
            zams_solution=zams_sol,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _classify_spectral_type(T_eff: float) -> str:
        """Assigns stellar spectral classification (OBAFGKM) based on effective temperature."""
        if T_eff >= 30000.0:
            return "O"
        elif T_eff >= 10000.0:
            return "B"
        elif T_eff >= 7500.0:
            return "A"
        elif T_eff >= 6000.0:
            return "F"
        elif T_eff >= 5200.0:
            return "G"
        elif T_eff >= 3700.0:
            return "K"
        else:
            return "M"
