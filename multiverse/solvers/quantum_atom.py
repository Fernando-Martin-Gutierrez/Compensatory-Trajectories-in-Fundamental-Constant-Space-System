"""Numerical Quantum Mechanical Solver for Atoms and Molecules with Modified Couplings.

Solves the radial Schrödinger equation for arbitrary electromagnetic coupling alpha
and mass ratio beta = m_e / m_p:
    [- (hbar^2 / (2 * mu)) d^2/dr^2 + V_eff(r)] u(r) = E u(r)
where:
    mu = m_p * beta / (1 + beta)
    V_eff(r) = - (alpha * hbar * c) / r + hbar^2 l(l+1) / (2 mu r^2)

Implements Level 3 Quantum Chemistry & Relativistic Corrections:
  1. Tridiagonal finite-difference Schrödinger eigenvalue solver.
  2. Numba-accelerated Numerov algorithm for wavefunction shooting and propagation.
  3. Relativistic Dirac and Breit-Pauli fine-structure splitting:
         Delta E_FS(2P_3/2 - 2P_1/2) = (alpha^4 * mu * c^2) / 32
     and Fermi contact hyperfine splitting (21 cm transition):
         Delta E_HFS = (8/3) * g_p * beta * alpha^4 * mu * c^2 / 2
  4. Born-Oppenheimer molecular bonding for H_2 and C-C covalent single bonds:
     evaluates dissociation energy D_e, zero-point energy (ZPE), and thermodynamic stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

import numpy as np
from scipy import linalg
import numba

from multiverse.constants import (
    ALPHA_STANDARD,
    BETA_STANDARD,
    PROTON_MASS_SI,
    SPEED_OF_LIGHT_SI,
    HBAR_SI,
    ELEMENTARY_CHARGE_SI,
    COMPTON_PROTON_SI,
    UniverseParameters,
)

# 1 eV in Joules
EV_TO_JOULE: float = ELEMENTARY_CHARGE_SI
JOULE_TO_EV: float = 1.0 / EV_TO_JOULE

# Proton g-factor for Fermi contact interaction
G_FACTOR_PROTON: float = 5.5856947


@numba.njit(fastmath=True)
def numerov_step(u_prev: float, u_curr: float, k2_prev: float, k2_curr: float, k2_next: float, h2: float) -> float:
    """Performs a single Numerov forward integration step with O(h^6) local error."""
    c_prev = 1.0 + (h2 / 12.0) * k2_prev
    c_curr = 2.0 * (1.0 - (5.0 * h2 / 12.0) * k2_curr)
    c_next = 1.0 + (h2 / 12.0) * k2_next
    return (c_curr * u_curr - c_prev * u_prev) / c_next


@numba.njit(fastmath=True)
def integrate_numerov_wavefunction(
    r_grid: np.ndarray,
    epsilon: float,
    l: int,
    h: float,
) -> np.ndarray:
    """Integrates the radial wave function outwards using Numerov's method."""
    n_pts = len(r_grid)
    u = np.zeros(n_pts, dtype=np.float64)
    h2 = h * h

    k2 = np.zeros(n_pts, dtype=np.float64)
    for i in range(n_pts):
        rho = r_grid[i]
        v_eff = -1.0 / rho + (l * (l + 1)) / (2.0 * rho * rho)
        k2[i] = 2.0 * (epsilon - v_eff)

    u[0] = 0.0
    u[1] = (r_grid[1] ** (l + 1))

    for i in range(1, n_pts - 1):
        u[i + 1] = numerov_step(u[i - 1], u[i], k2[i - 1], k2[i], k2[i + 1], h2)

    return u


@dataclass
class FineStructureSplitting:
    """Relativistic fine-structure and hyperfine atomic splittings."""
    delta_e_fs_ev: float
    delta_e_fs_ghz: float
    lamb_shift_ev: float
    hyperfine_21cm_ev: float
    hyperfine_freq_mhz: float


@dataclass
class MolecularBondResult:
    """Born-Oppenheimer molecular bonding and stability evaluation."""
    molecule_name: str
    dissociation_energy_de_ev: float
    zero_point_energy_ev: float
    net_binding_energy_d0_ev: float
    equilibrium_bond_length_m: float
    equilibrium_bond_length_angstrom: float
    vibrational_frequency_cm1: float
    is_chemically_stable: bool


@dataclass
class AtomicOrbitResult:
    """Quantum mechanical results for an atom and molecules in a candidate universe."""
    universe: UniverseParameters
    e1_hartree: float
    e2_hartree: float
    e1_ev: float
    e2_ev: float
    bohr_radius_si: float          # meters
    bohr_radius_compton: float     # a_* / lambda_p
    mean_radius_ground_si: float   # <r> in meters
    polarizability_si: float       # m^3 (atomic polarizability)
    wavefunction_r: np.ndarray     # radial grid (SI meters)
    wavefunction_u1: np.ndarray    # reduced radial wave function u_1(r) = r * R_1(r)
    fine_structure: Optional[FineStructureSplitting] = None
    h2_molecule: Optional[MolecularBondResult] = None
    cc_covalent_bond: Optional[MolecularBondResult] = None
    dirac_levels: Dict[str, float] = field(default_factory=dict)


class QuantumAtomSolver:
    """Solves the hydrogen atom eigenvalue problem and molecular stability."""

    def __init__(self, n_points: int = 2000, rho_max: float = 40.0) -> None:
        self.n_points = n_points
        self.rho_max = rho_max
        self.h = rho_max / (n_points + 1)
        self.rho_grid = (np.arange(n_points, dtype=np.float64) + 1.0) * self.h

    def compute_fine_structure(
        self,
        u: UniverseParameters,
        hartree_ev: float,
    ) -> Tuple[FineStructureSplitting, Dict[str, float]]:
        """Calculates relativistic Dirac fine-structure and hyperfine 21cm line."""
        rydberg_ev = 0.5 * hartree_ev

        # 1. Fine-structure splitting between 2P_3/2 and 2P_1/2:
        # Delta E_FS = (alpha^2 / 16) * Rydberg
        delta_fs_ev = (u.alpha ** 2.0 / 16.0) * rydberg_ev
        delta_fs_joules = delta_fs_ev * EV_TO_JOULE
        delta_fs_ghz = delta_fs_joules / (HBAR_SI * 2.0 * math.pi) * 1e-9

        # 2. Lamb shift radiative correction (Bethe formula leading order):
        # Delta E_Lamb ~ (alpha^5 * mu * c^2) / (6 pi) * ln(1 / (alpha^2))
        lamb_shift_ev = (u.alpha ** 3.0 / (6.0 * math.pi)) * rydberg_ev * math.log(1.0 / (u.alpha ** 2.0))

        # 3. Fermi contact hyperfine splitting for ground state 1s (21 cm hydrogen line):
        # Delta E_HFS = (8/3) * g_p * beta * alpha^2 * Rydberg
        delta_hfs_ev = (8.0 / 3.0) * G_FACTOR_PROTON * u.beta * (u.alpha ** 2.0) * rydberg_ev
        delta_hfs_joules = delta_hfs_ev * EV_TO_JOULE
        freq_21cm_mhz = delta_hfs_joules / (HBAR_SI * 2.0 * math.pi) * 1e-6

        # 4. Dirac energy levels (including relativistic kinetic and spin-orbit corrections):
        # E(n, j) ~ E_n^0 * [1 + (alpha/n)^2 * (n / (j + 1/2) - 3/4)]
        dirac_1s_12 = -rydberg_ev * (1.0 + (u.alpha ** 2.0) * (1.0 - 0.75))
        dirac_2s_12 = -(rydberg_ev / 4.0) * (1.0 + ((u.alpha / 2.0) ** 2.0) * (2.0 - 0.75))
        dirac_2p_12 = -(rydberg_ev / 4.0) * (1.0 + ((u.alpha / 2.0) ** 2.0) * (2.0 - 0.75))
        dirac_2p_32 = -(rydberg_ev / 4.0) * (1.0 + ((u.alpha / 2.0) ** 2.0) * (1.0 - 0.75))

        dirac_dict = {
            "1s_1/2": dirac_1s_12,
            "2s_1/2": dirac_2s_12,
            "2p_1/2": dirac_2p_12,
            "2p_3/2": dirac_2p_32,
        }

        fs = FineStructureSplitting(
            delta_e_fs_ev=delta_fs_ev,
            delta_e_fs_ghz=delta_fs_ghz,
            lamb_shift_ev=lamb_shift_ev,
            hyperfine_21cm_ev=delta_hfs_ev,
            hyperfine_freq_mhz=freq_21cm_mhz,
        )
        return fs, dirac_dict

    def compute_molecular_bond(
        self,
        molecule_name: str,
        u: UniverseParameters,
        a_star_si: float,
        hartree_ev: float,
    ) -> MolecularBondResult:
        """Evaluates Born-Oppenheimer covalent bond parameters."""
        if molecule_name == "H2":
            # Molecular hydrogen (H_2):
            # D_e ~ 0.1744 Hartree, R_e ~ 1.401 a_0
            de_hartree = 0.1744
            re_a0 = 1.4011
            # Nuclear reduced mass: M_nuc = m_p / 2
            # Vibrational frequency: omega_vib ~ sqrt(k / M_nuc) ~ Hartree * sqrt(beta)
            vib_coeff = 0.02006
        else:
            # Carbon-Carbon (C-C) covalent single bond:
            # D_e ~ 0.1324 Hartree, R_e ~ 2.910 a_0
            de_hartree = 0.1324
            re_a0 = 2.910
            # Carbon-12 reduced mass: M_nuc = 6 m_p -> reduced beta factor sqrt(beta / 12)
            vib_coeff = 0.0165

        de_ev = de_hartree * hartree_ev
        re_m = re_a0 * a_star_si
        re_angstrom = re_m * 1e10

        # Vibrational frequency and zero-point energy:
        # omega_vib (eV) ~ vib_coeff * sqrt(beta / beta_0) * (alpha / alpha_0)^2 * Hartree
        beta_scale = math.sqrt(u.beta / BETA_STANDARD)
        alpha_scale = u.alpha / ALPHA_STANDARD
        omega_vib_ev = vib_coeff * beta_scale * (alpha_scale ** 2.0) * (27.2114)  # in eV
        zpe_ev = 0.5 * omega_vib_ev
        d0_ev = max(0.0, de_ev - zpe_ev)

        # Convert to cm^-1: 1 eV = 8065.54 cm^-1
        vib_freq_cm1 = omega_vib_ev * 8065.54

        # Chemical stability: requires positive net binding energy D_0 > 0
        # and D_0 significantly greater than thermal noise k_B * 300K (~0.025 eV)
        is_chemically_stable = d0_ev > 0.05 and (zpe_ev < de_ev)

        return MolecularBondResult(
            molecule_name=molecule_name,
            dissociation_energy_de_ev=de_ev,
            zero_point_energy_ev=zpe_ev,
            net_binding_energy_d0_ev=d0_ev,
            equilibrium_bond_length_m=re_m,
            equilibrium_bond_length_angstrom=re_angstrom,
            vibrational_frequency_cm1=vib_freq_cm1,
            is_chemically_stable=is_chemically_stable,
        )

    def solve(self, u: UniverseParameters) -> AtomicOrbitResult:
        """Computes ground state, wave functions, relativistic corrections, and molecular stability."""
        # 1. Dimensionless atomic unit scales
        mu_over_mp = u.beta / (1.0 + u.beta)
        mu_si = mu_over_mp * PROTON_MASS_SI

        a_star_si = HBAR_SI / (u.alpha * mu_si * SPEED_OF_LIGHT_SI)
        a_star_compton = (1.0 + u.beta) / (u.alpha * u.beta)

        hartree_joule = (u.alpha ** 2.0) * mu_si * (SPEED_OF_LIGHT_SI ** 2.0)
        hartree_ev = hartree_joule * JOULE_TO_EV

        # 2. Discrete finite-difference Hamiltonian for l = 0 (s-states)
        rho = self.rho_grid
        h = self.h
        inv_h2 = 1.0 / (h * h)

        diag = inv_h2 - (1.0 / rho)
        off_diag = np.full(self.n_points - 1, -0.5 * inv_h2)

        eigenvalues, eigenvectors = linalg.eigh_tridiagonal(
            diag, off_diag, select="i", select_range=(0, 1)
        )

        eps_1 = float(eigenvalues[0])
        eps_2 = float(eigenvalues[1])

        # Normalize ground state reduced wave function
        u1_raw = eigenvectors[:, 0]
        norm_factor = np.sqrt(np.trapezoid(u1_raw ** 2, rho))
        u1 = u1_raw / norm_factor

        # 3. Mean orbital radius <r>
        mean_rho = float(np.trapezoid(rho * (u1 ** 2), rho))
        mean_radius_si = mean_rho * a_star_si

        # 4. Atomic polarizability: alpha_pol = 4.5 * a_*^3
        polarizability_si = 4.5 * (a_star_si ** 3.0)

        # Dimensional energies
        e1_ev = eps_1 * hartree_ev
        e2_ev = eps_2 * hartree_ev

        # 5. Relativistic fine structure & Dirac levels
        fine_structure, dirac_levels = self.compute_fine_structure(u, hartree_ev)

        # 6. Born-Oppenheimer molecular bonding
        h2_bond = self.compute_molecular_bond("H2", u, a_star_si, hartree_ev)
        cc_bond = self.compute_molecular_bond("C-C", u, a_star_si, hartree_ev)

        return AtomicOrbitResult(
            universe=u,
            e1_hartree=eps_1,
            e2_hartree=eps_2,
            e1_ev=e1_ev,
            e2_ev=e2_ev,
            bohr_radius_si=a_star_si,
            bohr_radius_compton=a_star_compton,
            mean_radius_ground_si=mean_radius_si,
            polarizability_si=polarizability_si,
            wavefunction_r=rho * a_star_si,
            wavefunction_u1=u1,
            fine_structure=fine_structure,
            h2_molecule=h2_bond,
            cc_covalent_bond=cc_bond,
            dirac_levels=dirac_levels,
        )

    def verify_numerov_shooting(self, epsilon: float, l: int = 0) -> float:
        """Executes Numerov shooting to evaluate boundary behavior at rho_max."""
        u_wave = integrate_numerov_wavefunction(self.rho_grid, epsilon, l, self.h)
        return float(u_wave[-1])
