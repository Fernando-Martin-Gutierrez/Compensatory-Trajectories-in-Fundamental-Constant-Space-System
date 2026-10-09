"""Publication-Quality Scientific Visualizations for MultiverseEquilibriumSim.

Generates:
  1. 2D Viability Heatmaps & Islands of Viability in (alpha, alpha_s) and (alpha, alpha_G).
  2. Covariant Equilibrium Manifold Curves with power-law scaling fits.
  3. Quantum Radial Wave Functions & Probability Densities.
  4. Lane-Emden Polytropic Density Profiles (n=1.5, n=3.0).
  5. 1PN Relativistic 3-body Orbital Trajectories & Phase-Space Precession.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List
from pathlib import Path

import matplotlib
# Use non-interactive Agg backend for robust headless rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    UniverseParameters,
)
from multiverse.core.manifolds import AnalyticalManifold, ViabilityReport
from multiverse.optimizer.equilibrium_search import EquilibriumManifoldCurve
from multiverse.optimizer.corridor import EquilibriumCorridor
from multiverse.solvers.quantum_atom import AtomicOrbitResult
from multiverse.solvers.stellar_structure import StellarViabilityResult
from multiverse.solvers.orbital_dynamics import OrbitalStabilityResult
from multiverse.solvers.cosmology_bbn import BBNResults


def apply_scientific_style() -> None:
    """Applies clean, high-contrast scientific plot aesthetics."""
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.autolayout": True,
    })


def _save_fig(fig: plt.Figure, output_path: Optional[str | Path]) -> None:
    """Safely saves figure, creating any missing parent directories."""
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")


def plot_viability_heatmap_2d(
    alpha_range: Tuple[float, float] = (0.005, 0.010),
    alpha_s_range: Tuple[float, float] = (0.08, 0.16),
    resolution: int = 120,
    output_path: Optional[str | Path] = "viability_heatmap_2d.png",
) -> plt.Figure:
    """Generates a 2D heatmap showing islands of viability in the (alpha, alpha_s) plane."""
    apply_scientific_style()
    manifold = AnalyticalManifold()

    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    alpha_s_vals = np.linspace(alpha_s_range[0], alpha_s_range[1], resolution)
    alpha_grid, as_grid = np.meshgrid(alphas, alpha_s_vals)

    scores = np.zeros_like(alpha_grid)
    hoyle_shifts = np.zeros_like(alpha_grid)

    for i in range(resolution):
        for j in range(resolution):
            u = UniverseParameters(
                alpha=alpha_grid[i, j],
                alpha_G=ALPHA_G_STANDARD,
                beta=BETA_STANDARD,
                alpha_s=as_grid[i, j],
            )
            rep = manifold.evaluate(u)
            scores[i, j] = rep.total_score
            hoyle_shifts[i, j] = rep.hoyle_resonance_shift_mev

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Viability heatmap
    pcm = ax.pcolormesh(
        alpha_grid,
        as_grid,
        scores,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(pcm, ax=ax, label="Universe Viability Score $S(\\vec{\\theta})$")

    # Contour lines for Hoyle resonance window |Delta E| <= 100 keV
    contours = ax.contour(
        alpha_grid,
        as_grid,
        hoyle_shifts,
        levels=[-0.12, 0.0, 0.12],
        colors=["#e74c3c", "#f1c40f", "#e67e22"],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 1.8, 1.2],
    )
    ax.clabel(contours, fmt="%+1.2f MeV", fontsize=8)

    # Contour line for viability threshold >= 0.90
    ax.contour(
        alpha_grid,
        as_grid,
        scores,
        levels=[0.90, 0.95],
        colors=["#ffffff", "#2ecc71"],
        linestyles=["-", "-."],
        linewidths=[1.8, 2.0],
    )

    # Standard Universe Marker
    ax.plot(
        ALPHA_STANDARD,
        ALPHA_S_STANDARD,
        marker="*",
        markersize=14,
        color="#e74c3c",
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Standard Universe $(\\alpha_0, \\alpha_{s,0})$",
        zorder=10,
    )

    ax.set_xlabel(r"Fine-Structure Constant $\alpha$ (EM coupling)")
    ax.set_ylabel(r"Strong Coupling $\alpha_s$ (low-energy nuclear)")
    ax.set_title(r"Islands of Viability: Hoyle Resonance & Di-nucleon Stability in $(\alpha, \alpha_s)$ Space")
    ax.legend(loc="upper left", framealpha=0.9)

    _save_fig(fig, output_path)
    return fig


def plot_manifold_curve(
    curve: EquilibriumManifoldCurve,
    output_path: Optional[str | Path] = "equilibrium_manifold.png",
) -> plt.Figure:
    """Plots the 1D covariant equilibrium curve gamma(alpha) embedded in viability domain V in R^4."""
    apply_scientific_style()
    valid = curve.valid_points
    if not valid:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No viable points found", ha="center")
        return fig

    alphas, log_a, log_ag, log_b, log_as = curve.extract_log_arrays()
    scores = np.array([p.viability.total_score for p in curve.points])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

    # Panel 1: log(alpha_G) vs log(alpha)
    ax1 = axes[0, 0]
    ax1.plot(log_a, log_ag, "o", color="#2980b9", markersize=6, label="Covariant Points")
    ax1.set_xlabel(r"$\log_{10}(\alpha)$")
    ax1.set_ylabel(r"$\log_{10}(\alpha_G)$")
    ax1.set_title(r"Convex Gravitational Basin: $\alpha_G(\alpha)$")

    if curve.scaling_laws and curve.scaling_laws.fit_alpha_G:
        ag_fit = curve.scaling_laws.fit_alpha_G
        x_min = min(log_a.min(), ag_fit.log_alpha_star - 0.005)
        x_max = max(log_a.max(), ag_fit.log_alpha_star + 0.005)
        x_grid = np.linspace(x_min, x_max, 150)
        y_quad = ag_fit.a2 * (x_grid ** 2.0) + ag_fit.a1 * x_grid + ag_fit.a0
        ax1.plot(x_grid, y_quad, "-", color="#e74c3c", lw=2.0, label=f"Quadratic Fit ($R^2={ag_fit.r_squared_quad:.3f}$)")
        # Marker for vertex alpha*
        ax1.axvline(ag_fit.log_alpha_star, color="#e67e22", ls="--", lw=1.5, alpha=0.8,
                    label=f"Vertex $\\alpha^*={ag_fit.alpha_star_ratio:.3f}\\alpha_0$")
        ax1.plot(ag_fit.log_alpha_star, ag_fit.y_star, "*", color="#f39c12", markersize=14, markeredgecolor="black", zorder=6)
        ax1.set_xlim(x_min - 0.005, x_max + 0.005)
    elif curve.scaling_laws:
        gamma = curve.scaling_laws.d_log_alpha_G_d_log_alpha
        ax1.plot(log_a, log_ag, "-", color="#2980b9", lw=1.5)
        ax1.text(0.05, 0.10, f"Slope $\\gamma = {gamma:+.2f}$", transform=ax1.transAxes,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax1.legend(loc="best", fontsize=8)

    # Panel 2: log(beta) vs log(alpha)
    ax2 = axes[0, 1]
    ax2.plot(log_a, log_b, "s", color="#8e44ad", markersize=6, label="Covariant Points")
    ax2.set_xlabel(r"$\log_{10}(\alpha)$")
    ax2.set_ylabel(r"$\log_{10}(\beta)$ ($m_e/m_p$)")
    ax2.set_title(r"Leptonic Mass Scaling: $\beta(\alpha)$")

    if curve.scaling_laws and curve.scaling_laws.fit_beta:
        b_fit = curve.scaling_laws.fit_beta
        x_min = min(log_a.min(), b_fit.log_alpha_star - 0.005)
        x_max = max(log_a.max(), b_fit.log_alpha_star + 0.005)
        x_grid = np.linspace(x_min, x_max, 150)
        y_quad = b_fit.a2 * (x_grid ** 2.0) + b_fit.a1 * x_grid + b_fit.a0
        ax2.plot(x_grid, y_quad, "-", color="#e74c3c", lw=2.0, label=f"Quadratic Fit ($R^2={b_fit.r_squared_quad:.3f}$)")
        ax2.axvline(b_fit.log_alpha_star, color="#e67e22", ls="--", lw=1.5, alpha=0.8,
                    label=f"Vertex $\\alpha^*={b_fit.alpha_star_ratio:.3f}\\alpha_0$")
        ax2.plot(b_fit.log_alpha_star, b_fit.y_star, "*", color="#f39c12", markersize=14, markeredgecolor="black", zorder=6)
        ax2.set_xlim(x_min - 0.005, x_max + 0.005)
    elif curve.scaling_laws:
        delta = curve.scaling_laws.d_log_beta_d_log_alpha
        ax2.plot(log_a, log_b, "-", color="#8e44ad", lw=1.5)
        ax2.text(0.05, 0.10, f"Slope $\\delta = {delta:+.2f}$", transform=ax2.transAxes,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax2.legend(loc="best", fontsize=8)

    # Panel 3: alpha_s vs alpha (Hoyle resonance balance)
    ax3 = axes[1, 0]
    ax3.plot(alphas, 10.0 ** log_as, "^-", color="#d35400", lw=1.8, label="Covariant Path")
    ax3.axhline(ALPHA_S_STANDARD, color="gray", ls="--", alpha=0.7, label=r"Standard $\alpha_{s,0}$")
    if curve.scaling_laws:
        r2_as = curve.scaling_laws.r_squared_alpha_s
        ax3.text(0.05, 0.15, f"Monotonic Scaling $R^2 = {r2_as:.3f}$", transform=ax3.transAxes,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=8)
    ax3.set_xlabel(r"$\alpha$")
    ax3.set_ylabel(r"$\alpha_s$")
    ax3.set_title(r"Strong Coupling Response: $\alpha_s(\alpha)$")
    ax3.legend(loc="best", fontsize=8)

    # Panel 4: Viability Score along curve
    ax4 = axes[1, 1]
    ax4.plot(alphas, scores, "D-", color="#27ae60", lw=2.0)
    ax4.axhline(0.95, color="#e74c3c", ls="--", label="Target Viability (0.95)")
    ax4.set_xlabel(r"$\alpha$")
    ax4.set_ylabel(r"Viability Functional $S(u)$")
    ax4.set_title(r"Cosmic Viability Persistence: $S(u)$")
    ax4.set_ylim(0.85, 1.02)
    ax4.legend(loc="lower right", fontsize=8)

    fig.suptitle(r"Multiverse Covariant Equilibrium Manifold: Convex Basins & Quadratic Scaling Laws", fontsize=13, y=1.02)
    fig.tight_layout()

    _save_fig(fig, output_path)
    return fig


def plot_transverse_corridor(
    corridor: EquilibriumCorridor,
    output_path: Optional[str | Path] = "corridor_thickness.png",
) -> plt.Figure:
    """Plots the transverse viability corridor thickness and non-zero Lebesgue measure volume (Fig 3)."""
    apply_scientific_style()
    profiles = corridor.profiles
    if not profiles:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No corridor profiles found", ha="center")
        return fig

    alpha_ratios = np.array([p.alpha_ratio for p in profiles])
    alphas = np.array([p.alpha for p in profiles])
    log_a = np.log10(alphas)

    # Tolerances
    as_nom = np.array([p.tolerances["alpha_s"].nominal_val for p in profiles])
    as_low = np.array([p.tolerances["alpha_s"].lower_val for p in profiles])
    as_high = np.array([p.tolerances["alpha_s"].upper_val for p in profiles])

    log_ag_nom = np.array([p.tolerances["alpha_G"].log_nominal_val for p in profiles])
    log_ag_low = np.array([p.tolerances["alpha_G"].log_lower_val for p in profiles])
    log_ag_high = np.array([p.tolerances["alpha_G"].log_upper_val for p in profiles])

    beta_nom = np.array([p.tolerances["beta"].nominal_val for p in profiles])
    beta_low = np.array([p.tolerances["beta"].lower_val for p in profiles])
    beta_high = np.array([p.tolerances["beta"].upper_val for p in profiles])

    vol_trans = np.array([p.transverse_volume_log for p in profiles])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

    # Panel 1: alpha_s corridor envelope
    ax1 = axes[0, 0]
    ax1.plot(alpha_ratios, as_nom, "o-", color="#d35400", lw=2.0, label=r"Center $\gamma(\alpha)$")
    ax1.fill_between(
        alpha_ratios, as_low, as_high,
        color="#e67e22", alpha=0.25,
        label=rf"Viable Corridor ($S \geq {corridor.threshold_score:.2f}$)"
    )
    ax1.plot(alpha_ratios, as_low, "--", color="#d35400", lw=1.0, alpha=0.7)
    ax1.plot(alpha_ratios, as_high, "--", color="#d35400", lw=1.0, alpha=0.7)
    ax1.axhline(ALPHA_S_STANDARD, color="gray", ls=":", alpha=0.7, label=r"$\alpha_{s,0}$")
    ax1.set_xlabel(r"$\alpha / \alpha_0$")
    ax1.set_ylabel(r"$\alpha_s$")
    ax1.set_title(r"Strong Coupling Response: $\alpha_s(\alpha)$")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: log(alpha_G) corridor envelope
    ax2 = axes[0, 1]
    ax2.plot(alpha_ratios, log_ag_nom, "o-", color="#2980b9", lw=2.0, label=r"Center $\gamma(\alpha)$")
    ax2.fill_between(
        alpha_ratios, log_ag_low, log_ag_high,
        color="#3498db", alpha=0.25,
        label=rf"Tolerance Envelope ($S \geq {corridor.threshold_score:.2f}$)"
    )
    ax2.plot(alpha_ratios, log_ag_low, "--", color="#2980b9", lw=1.0, alpha=0.7)
    ax2.plot(alpha_ratios, log_ag_high, "--", color="#2980b9", lw=1.0, alpha=0.7)
    ax2.set_xlabel(r"$\alpha / \alpha_0$")
    ax2.set_ylabel(r"$\log_{10}(\alpha_G)$")
    ax2.set_title(r"Convex Gravitational Basin: $\alpha_G(\alpha)$")
    ax2.legend(loc="lower left", fontsize=8)

    # Panel 3: beta corridor envelope
    ax3 = axes[1, 0]
    ax3.plot(alpha_ratios, beta_nom * 1e4, "s-", color="#8e44ad", lw=2.0, label=r"Center $\gamma(\alpha)$")
    ax3.fill_between(
        alpha_ratios, beta_low * 1e4, beta_high * 1e4,
        color="#9b59b6", alpha=0.25,
        label=rf"Tolerance Envelope ($S \geq {corridor.threshold_score:.2f}$)"
    )
    ax3.plot(alpha_ratios, beta_low * 1e4, "--", color="#8e44ad", lw=1.0, alpha=0.7)
    ax3.plot(alpha_ratios, beta_high * 1e4, "--", color="#8e44ad", lw=1.0, alpha=0.7)
    ax3.set_xlabel(r"$\alpha / \alpha_0$")
    ax3.set_ylabel(r"$\beta \times 10^4$ ($m_e/m_p$)")
    ax3.set_title(r"Leptonic Mass Scaling: $\beta(\alpha)$")
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Transverse corridor volume in log-space (dex^3)
    ax4 = axes[1, 1]
    ax4.bar(
        alpha_ratios, vol_trans, width=0.015, color="#16a085", alpha=0.7,
        edgecolor="#117a65", label=r"Transverse Volume $V_{\rm trans}$"
    )
    ax4.plot(alpha_ratios, vol_trans, "D-", color="#0e6251", lw=1.5)
    ax4.set_xlabel(r"$\alpha / \alpha_0$")
    ax4.set_ylabel(r"$V_{\rm trans}$ $(\mathrm{dex}^3)$")
    ax4.set_title(r"Non-Zero Lebesgue Measure: $\operatorname{Vol}_{4D} > 0$")
    text_info = (
        f"Integrated 4D Vol.: {corridor.total_integrated_volume:.4f} dex$^4$\n"
        f"Transverse Mean: {corridor.mean_transverse_volume:.4f} dex$^3$\n"
        r"$\operatorname{Vol}_{4D} > 0$ (Proven)"
    )
    ax4.text(
        0.05, 0.70, text_info, transform=ax4.transAxes,
        bbox=dict(boxstyle="round", fc="white", ec="#16a085", alpha=0.9),
        fontsize=8.5
    )
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        rf"Transverse Viability Corridor Thickness along $\gamma(\alpha)$ ($S \geq {corridor.threshold_score:.2f}$)",
        fontsize=13, y=1.02
    )
    fig.tight_layout()

    _save_fig(fig, output_path)
    return fig


def plot_quantum_wavefunction(
    atomic_result: AtomicOrbitResult,
    output_path: Optional[str | Path] = "quantum_hydrogen.png",
) -> plt.Figure:
    """Plots radial wave functions and probability densities for modified hydrogen atom."""
    apply_scientific_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Convert radius to Angstroms
    r_ang = atomic_result.wavefunction_r * 1e10
    u1 = atomic_result.wavefunction_u1
    prob = u1 ** 2.0
    a0_ang = atomic_result.bohr_radius_si * 1e10

    # Cutoff grid where probability is negligible
    mask = r_ang <= 8.0 * a0_ang
    r_sub = r_ang[mask]
    u1_sub = u1[mask]
    prob_sub = prob[mask]

    # Reduced wave function
    ax1.plot(r_sub, u1_sub, color="#2980b9", lw=2.0, label=r"$u_1(r) = r R_{1s}(r)$")
    ax1.axvline(a0_ang, color="#e74c3c", ls="--", label=f"$a_0 = {a0_ang:.3f}$ Å")
    ax1.set_xlabel(r"Radial distance $r$ [Å]")
    ax1.set_ylabel(r"Reduced Wave Function $u_1(r)$")
    ax1.set_title(f"Ground State: $E_1 = {atomic_result.e1_ev:.2f}$ eV")
    ax1.legend()

    # Radial probability distribution P(r) = |u1|^2
    ax2.plot(r_sub, prob_sub, color="#27ae60", lw=2.0, label=r"$P(r) = |u_1(r)|^2$")
    ax2.axvline(a0_ang, color="#e74c3c", ls="--", label=r"Peak $r = a_0$")
    ax2.fill_between(r_sub, prob_sub, color="#2ecc71", alpha=0.3)
    ax2.set_xlabel(r"Radial distance $r$ [Å]")
    ax2.set_ylabel(r"Radial Probability Density $P(r)$")
    ax2.set_title(r"Orbital Probability Density")
    ax2.legend()

    fig.suptitle(r"Quantum Mechanical Hydrogen-Like Atom Eigenstate", fontsize=13)
    fig.tight_layout()

    _save_fig(fig, output_path)
    return fig


def plot_stellar_polytropes(
    stellar_result: StellarViabilityResult,
    output_path: Optional[str | Path] = "stellar_polytropes.png",
) -> plt.Figure:
    """Plots comprehensive 4-panel astrophysical stellar structure and nuclear fusion profiles."""
    apply_scientific_style()

    star = stellar_result.solar_analog
    prof = star.profile_1d if star else None

    if prof is not None and len(stellar_result.stellar_population) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
        ax2 = ax3 = ax4 = None

    # Panel 1: Lane-Emden Polytropes
    p15 = stellar_result.polytrope_n15
    p30 = stellar_result.polytrope_n30
    ax1.plot(p15.xi_array, p15.theta_array, color="#2980b9", lw=2.0,
             label=f"Polytrope $n=1.5$ (Degenerate, $\\xi_1={p15.xi_surface:.2f}$)")
    ax1.plot(p30.xi_array, p30.theta_array, color="#e74c3c", lw=2.0,
             label=f"Polytrope $n=3.0$ (Radiation/$M_{{Ch}}$, $\\xi_1={p30.xi_surface:.2f}$)")
    ax1.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax1.set_xlim(0.0, 7.5)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel(r"Dimensionless Radius $\xi$")
    ax1.set_ylabel(r"Normalized Density $\theta(\xi)$")
    ax1.set_title(r"(a) Lane-Emden Hydrostatic Equilibrium")
    ax1.legend(loc="upper right", fontsize=8)

    if prof is not None and ax2 is not None and star is not None:
        # Panel 2: Temperature & Density Profiles
        r_norm = prof.r_array_m / prof.r_array_m[-1]
        t_millions = prof.temperature_kelvin * 1e-6
        rho_cgs = prof.density_kg_m3 * 1e-3  # g/cm^3

        color_t = "#d35400"
        ax2.plot(r_norm, t_millions, color=color_t, lw=2.2, label=r"Temperature $T(r)$ [$10^6$ K]")
        ax2.set_xlabel(r"Fractional Radius $r / R_*$")
        ax2.set_ylabel(r"Temperature $T$ [$10^6$ K]", color=color_t)
        ax2.tick_params(axis="y", labelcolor=color_t)

        ax2_rho = ax2.twinx()
        color_rho = "#27ae60"
        ax2_rho.plot(r_norm, rho_cgs, color=color_rho, lw=2.0, ls="--", label=r"Density $\rho(r)$ [g/cm$^3$]")
        ax2_rho.set_ylabel(r"Density $\rho$ [g/cm$^3$]", color=color_rho)
        ax2_rho.tick_params(axis="y", labelcolor=color_rho)
        ax2.set_title(f"(b) Interior Profiles: $T_c={star.central_temp_kelvin*1e-6:.1f}$ MK, $\\rho_c={star.central_density_kg_m3*1e-3:.1f}$ g/cm$^3$")

        # Panel 3: Energy Generation Rate & Enclosed Luminosity
        color_eps = "#8e44ad"
        eps_mw = prof.specific_energy_generation * 1e3  # mW/kg
        ax3.plot(r_norm, eps_mw, color=color_eps, lw=2.0, label=r"Fusion Rate $\epsilon_{\rm nuc}$ [mW/kg]")
        ax3.set_xlabel(r"Fractional Radius $r / R_*$")
        ax3.set_ylabel(r"Nuclear Generation $\epsilon_{\rm nuc}$ [mW/kg]", color=color_eps)
        ax3.tick_params(axis="y", labelcolor=color_eps)

        ax3_l = ax3.twinx()
        color_l = "#2980b9"
        l_norm = prof.enclosed_luminosity_watts / max(prof.enclosed_luminosity_watts[-1], 1e-30)
        ax3_l.plot(r_norm, l_norm, color=color_l, lw=2.2, ls="-.", label=r"Enclosed Luminosity $L(r)/L_*$")
        ax3_l.set_ylabel(r"Enclosed Luminosity $L(r) / L_*$", color=color_l)
        ax3_l.tick_params(axis="y", labelcolor=color_l)
        ax3_l.set_ylim(-0.05, 1.05)
        ax3.set_title(f"(c) Energy Generation & Transport ($L_* = {star.luminosity_solar:.2f} L_\\odot$)")

        # Panel 4: Stellar Population Spectrum & Mass Window
        pop_masses = [s.stellar_mass_solar for s in stellar_result.stellar_population]
        pop_lums = [s.luminosity_solar for s in stellar_result.stellar_population]
        ax4.loglog(pop_masses, pop_lums, "o-", color="#e67e22", lw=2.2, markersize=7, label="Main Sequence Population")
        ax4.axvline(stellar_result.min_ignition_mass_solar, color="#c0392b", ls="--", lw=1.8,
                    label=f"$M_{{\\rm min}} = {stellar_result.min_ignition_mass_solar:.3f} M_\\odot$")
        ax4.axvline(stellar_result.chandrasekhar_mass_solar, color="#7f8c8d", ls="--", lw=1.8,
                    label=f"$M_{{\\rm Ch}} = {stellar_result.chandrasekhar_mass_solar:.2f} M_\\odot$")
        ax4.plot([1.0], [star.luminosity_solar], "*", color="#f1c40f", markersize=14, markeredgecolor="black", label=r"Solar Analog ($1.0 M_\odot$)")
        ax4.set_xlabel(r"Stellar Mass $M / M_\odot$")
        ax4.set_ylabel(r"Luminosity $L / L_\odot$")
        ax4.set_title(f"(d) Mass-Luminosity Scaling (Window Ratio $= {stellar_result.mass_window_ratio:.1f}$)")
        ax4.legend(loc="lower right", fontsize=8)

    title_str = (
        f"Astrophysical Stellar Structure & Gamow Tunneling Equilibrium\n"
        f"Burning Window: $M_{{\\rm min}} = {stellar_result.min_ignition_mass_solar:.3f} M_\\odot < "
        f"M_{{\\rm Ch}} = {stellar_result.chandrasekhar_mass_solar:.2f} M_\\odot$"
    )
    if star is not None:
        title_str += f" ($E_0 = {star.gamow_peak_kev:.1f} \\pm {star.gamow_window_width_kev*0.5:.1f}$ keV)"

    fig.suptitle(title_str, fontsize=13, y=1.01)
    fig.tight_layout()

    _save_fig(fig, output_path)
    return fig


def plot_orbital_trajectories(
    orbital_result: OrbitalStabilityResult,
    output_path: Optional[str | Path] = "orbital_trajectories.png",
) -> plt.Figure:
    """Plots 3-body orbits with 1PN relativistic precession and stability metrics."""
    apply_scientific_style()
    t1 = orbital_result.trajectory_planet_1
    t2 = orbital_result.trajectory_planet_2

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)

    # Central star
    ax.plot(0.0, 0.0, marker="o", markersize=12, color="#f39c12", label="Central Star")

    # Planet 1 (Inner)
    ax.plot(t1[:, 0], t1[:, 1], color="#2980b9", lw=1.2, alpha=0.85, label="Inner Planet (1PN Precession)")

    # Planet 2 (Outer)
    ax.plot(t2[:, 0], t2[:, 1], color="#27ae60", lw=1.2, alpha=0.85, label="Outer Planet")

    ax.set_xlabel("X [Orbital Units]")
    ax.set_ylabel("Y [Orbital Units]")
    ax.set_aspect("equal")
    ax.set_title(
        f"3-Body Symplectic Orbital Integration (Yoshida 4th Order)\n"
        f"Lyapunov $\\lambda = {orbital_result.lyapunov_exponent:.2e}$, "
        f"Drift $\\Delta a/a = {orbital_result.max_semi_major_axis_drift:.2e}$"
    )
    ax.legend(loc="upper right")

    _save_fig(fig, output_path)
    return fig


def plot_bbn_abundances(
    bbn_result: BBNResults,
    output_path: Optional[str | Path] = "bbn_primordial_nucleosynthesis.png",
) -> plt.Figure:
    """Plots Big Bang Nucleosynthesis mass fractions and neutron freeze-out curves."""
    apply_scientific_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    temps = bbn_result.temperature_mev_history
    times = bbn_result.time_sec_history
    yp = bbn_result.yp_evolution
    xh = bbn_result.xh_evolution

    # Panel 1: Primordial Mass Fractions X_H and Y_p vs Temperature / Time
    ax1.plot(temps, xh, color="#2980b9", lw=2.5, label=f"Hydrogen $X_H$ ({bbn_result.hydrogen_mass_fraction_xh*100:.1f}%)")
    ax1.plot(temps, yp, color="#e74c3c", lw=2.5, label=f"Helium-4 $Y_p$ ({bbn_result.helium4_mass_fraction_yp*100:.1f}%)")
    ax1.axvline(bbn_result.t_freeze_out_mev, color="#f39c12", ls="--", lw=1.2, label=f"Freeze-out $T_f = {bbn_result.t_freeze_out_mev:.2f}$ MeV")
    ax1.axvline(bbn_result.t_bottleneck_mev, color="#8e44ad", ls=":", lw=1.5, label=f"Deuterium Bottleneck $T_{{BBN}} = {bbn_result.t_bottleneck_mev*1e3:.0f}$ keV")

    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Cosmic Temperature $T$ [MeV]")
    ax1.set_ylabel("Primordial Mass Fraction")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Primordial Nucleosynthesis (BBN) Abundance Evolution")
    ax1.legend(loc="center left", fontsize=8.5)

    # Panel 2: Neutron-to-Proton Ratio and Reaction Window
    ax2.axhspan(0.20, 0.80, color="#2ecc71", alpha=0.15, label="Chemistry-Viable Helium Window")
    ax2.bar(
        ["(n/p) Freeze-out", "(n/p) at BBN", "Deuterium (D/H) x1e4", "Helium-3 (3He/H) x1e4"],
        [
            bbn_result.np_ratio_frozen,
            bbn_result.np_ratio_bbn,
            bbn_result.deuterium_abundance_dh * 1e4,
            bbn_result.helium3_abundance_he3_h * 1e4,
        ],
        color=["#f39c12", "#e67e22", "#3498db", "#9b59b6"],
        width=0.55,
    )
    ax2.set_ylabel("Dimensionless Ratio / Abundance")
    ax2.set_title(
        f"Cosmological Yields (Viability Score: {bbn_result.score_bbn_viability:.4f})\n"
        f"Free Neutron Survival: {bbn_result.neutron_survival_fraction*100:.1f}%"
    )
    ax2.grid(True, axis="y", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    _save_fig(fig, output_path)
    return fig

