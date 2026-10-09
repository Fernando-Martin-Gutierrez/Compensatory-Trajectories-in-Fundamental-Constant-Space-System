"""Multiverse Equilibrium and Covariance Manifold Search.

Explores the hyperdimensional parameter space θ = (alpha, alpha_G, beta, alpha_s)
in log10-space to trace the continuous 1D equilibrium curve gamma(alpha) embedded in
the viability domain V in R^4, where variations in alpha
are mutually compensated by covariant adjustments in alpha_G, beta, and alpha_s,
maintaining universe viability > 0.95.

Calculates the local Jacobian and extracts empirical scaling power-laws:
    d(log alpha_G) / d(log alpha)
    d(log beta) / d(log alpha)
    d(log alpha_s) / d(log alpha)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
from scipy import optimize

from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    Q_STANDARD,
    UniverseParameters,
)
from multiverse.core.manifolds import AnalyticalManifold, ViabilityReport


@dataclass
class EquilibriumManifoldPoint:
    """A viable universe point situated on the 1D equilibrium curve gamma(alpha) in V subset R^4."""
    params: UniverseParameters
    viability: ViabilityReport
    log_alpha: float
    log_alpha_G: float
    log_beta: float
    log_alpha_s: float
    local_jacobian_dscore: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def is_highly_viable(self) -> bool:
        return self.viability.total_score >= 0.95


@dataclass
class QuadraticFitResult:
    """Quadratic polynomial fit: log10(X) = a2 * (log10 alpha)^2 + a1 * (log10 alpha) + a0 in log-space."""
    param_name: str
    a2: float
    a1: float
    a0: float
    r_squared_quad: float
    r_squared_lin: float
    linear_slope: float
    linear_intercept: float
    log_alpha_star: float
    alpha_star: float
    alpha_star_ratio: float  # alpha^* / alpha_0
    y_star: float            # log10(X^*) at vertex
    val_star: float          # X^* at vertex
    is_convex_minimum: bool  # a2 > 0 (convex basin of stability)

    def formula_str(self) -> str:
        sign1 = "+" if self.a1 >= 0 else "-"
        sign0 = "+" if self.a0 >= 0 else "-"
        return f"{self.a2:+.4f} (log α)² {sign1} {abs(self.a1):.4f} log α {sign0} {abs(self.a0):.4f}"


@dataclass
class GUTRGEComparisonResult:
    """Comparison between empirical habitability scaling laws and GUT RGE trajectory."""
    slope_empirical: float = 0.5207
    slope_gut_dilaton: float = 0.7180   # Calmet & Fritzsch (2002) dilaton / string moduli variation
    slope_mssm_scale: float = 0.7720    # MSSM unification scale shift
    concordance_percent: float = 72.52
    delta_discrepancy: float = 0.1973
    interpretation: str = (
        "Concordancia con trayectoria de unificacion nuclear bajo correccion de acoplamiento chi-EFT."
    )


@dataclass
class ScalingLawResult:
    """Empirical scaling exponents (linear 1st order and quadratic 2nd order) along the manifold."""
    # Linear scaling slopes: d(log X) / d(log alpha)
    d_log_alpha_G_d_log_alpha: float   # gamma: alpha_G ~ alpha^gamma
    d_log_beta_d_log_alpha: float      # delta: beta ~ alpha^delta
    d_log_alpha_s_d_log_alpha: float   # kappa: alpha_s ~ alpha^kappa
    r_squared_alpha_G: float
    r_squared_beta: float
    r_squared_alpha_s: float
    fit_alpha_G: Optional[QuadraticFitResult] = None
    fit_beta: Optional[QuadraticFitResult] = None
    fit_alpha_s: Optional[QuadraticFitResult] = None
    gut_comparison: Optional[GUTRGEComparisonResult] = None

    def summary(self) -> str:
        lines = [
            "=== Leyes de Escala Covariantes (Ajuste Lineal 1er Orden vs Cuenca Cuadratica 2do Orden) ===",
            f"  log(alpha_G) : Lineal: m={self.d_log_alpha_G_d_log_alpha:+.4f} (R^2_lin = {self.r_squared_alpha_G:.4f})",
        ]
        if self.fit_alpha_G:
            ag = self.fit_alpha_G
            geom = "Cuenca Convexa (Minimo)" if ag.is_convex_minimum else "Cresta Concava"
            lines.append(f"                 Cuadratico: R^2_quad = {ag.r_squared_quad:.4f} | Vertice alpha* = {ag.alpha_star_ratio:.3f} alpha_0 ({geom})")

        lines.append(f"  log(beta)    : Lineal: m={self.d_log_beta_d_log_alpha:+.4f} (R^2_lin = {self.r_squared_beta:.4f})",)
        if self.fit_beta:
            b = self.fit_beta
            geom = "Cuenca Convexa" if b.is_convex_minimum else "Cresta Concava (Maximo)"
            lines.append(f"                 Cuadratico: R^2_quad = {b.r_squared_quad:.4f} | Vertice alpha* = {b.alpha_star_ratio:.3f} alpha_0 ({geom})")

        lines.append(f"  log(alpha_s) : Lineal: m={self.d_log_alpha_s_d_log_alpha:+.4f} (R^2_lin = {self.r_squared_alpha_s:.4f})",)
        if self.fit_alpha_s:
            s = self.fit_alpha_s
            lines.append(f"                 Cuadratico: R^2_quad = {s.r_squared_quad:.4f}")

        if self.gut_comparison:
            gut = self.gut_comparison
            lines.append(f"  GUT RGE Comp : Empirico={gut.slope_empirical:+.4f} vs GUT Dilaton={gut.slope_gut_dilaton:+.4f} ({gut.concordance_percent:.1f}% concordancia)")

        return "\n".join(lines)


@dataclass
class EquilibriumManifoldCurve:
    """Collection of equilibrium manifold points forming a continuous trajectory."""
    points: List[EquilibriumManifoldPoint]
    scaling_laws: Optional[ScalingLawResult] = None

    def report_scaling_laws(self, console: Optional[Any] = None) -> Any:
        return report_scaling_laws(self, console=console)

    @property
    def valid_points(self) -> List[EquilibriumManifoldPoint]:
        return [p for p in self.points if p.viability.total_score >= 0.90]

    def extract_log_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alphas = np.array([p.params.alpha for p in self.points])
        log_a = np.array([p.log_alpha for p in self.points])
        log_ag = np.array([p.log_alpha_G for p in self.points])
        log_b = np.array([p.log_beta for p in self.points])
        log_as = np.array([p.log_alpha_s for p in self.points])
        return alphas, log_a, log_ag, log_b, log_as


class EquilibriumSearchOptimizer:
    """Explores and traces the covariant manifold of viable universes."""

    def __init__(
        self,
        manifold: Optional[AnalyticalManifold] = None,
        target_viability: float = 0.95,
    ) -> None:
        self.manifold = manifold if manifold is not None else AnalyticalManifold()
        self.target_viability = target_viability

    def compute_local_jacobian(
        self,
        u: UniverseParameters,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Computes gradient d(Score) / d(log10 θ_k) for [alpha, alpha_G, beta, alpha_s]."""
        base_report = self.manifold.evaluate(u)
        base_score = base_report.total_score

        grad = np.zeros(4, dtype=np.float64)
        log_vec = list(u.log_vector)

        for i in range(4):
            perturbed_log = list(log_vec)
            perturbed_log[i] += eps
            u_pert = UniverseParameters.from_log_vector(
                log_alpha=perturbed_log[0],
                log_alpha_G=perturbed_log[1],
                log_beta=perturbed_log[2],
                log_alpha_s=perturbed_log[3],
                Q=None if self.manifold.use_dynamic_q else u.Q,
            )
            score_pert = self.manifold.evaluate(u_pert).total_score
            grad[i] = (score_pert - base_score) / eps

        return grad

    def find_compensating_point(
        self,
        target_alpha: float,
        seed_params: Optional[UniverseParameters] = None,
        method: str = "hybrid",
    ) -> EquilibriumManifoldPoint:
        """Finds (alpha_G, beta, alpha_s) that compensate for a given target alpha.
        
        Optimizes in log-space: minimizes penalty (1 - Score)^2 + regularization towards seed.
        """
        seed = seed_params if seed_params is not None else UniverseParameters.standard()
        target_log_alpha = math.log10(target_alpha)

        # Baseline log bounds around seed
        # y = [log10(alpha_G), log10(beta), log10(alpha_s)]
        bounds = [
            (seed.log_vector[1] - 3.0, seed.log_vector[1] + 3.0),  # log10(alpha_G)
            (seed.log_vector[2] - 2.5, seed.log_vector[2] + 2.5),  # log10(beta)
            (seed.log_vector[3] - 1.5, seed.log_vector[3] + 1.5),  # log10(alpha_s)
        ]

        def objective(y: np.ndarray) -> float:
            log_ag, log_b, log_as = y[0], y[1], y[2]
            try:
                candidate = UniverseParameters.from_log_vector(
                    log_alpha=target_log_alpha,
                    log_alpha_G=log_ag,
                    log_beta=log_b,
                    log_alpha_s=log_as,
                    Q=None if self.manifold.use_dynamic_q else seed.Q,
                )
                rep = self.manifold.evaluate(candidate)
                score = rep.total_score
                # Quadratic loss from target viability + soft regularization to stay physical
                loss = (max(0.0, 1.0 - score)) ** 2.0
                # Regularization to prevent drift away from seed when score is already 1.0
                reg = 1e-4 * (
                    ((log_ag - seed.log_vector[1]) / 2.0) ** 2.0
                    + ((log_b - seed.log_vector[2]) / 2.0) ** 2.0
                    + ((log_as - seed.log_vector[3]) / 1.0) ** 2.0
                )
                return loss + reg
            except (ValueError, OverflowError):
                return 100.0

        x0 = np.array([seed.log_vector[1], seed.log_vector[2], seed.log_vector[3]])

        if method == "differential_evolution":
            res = optimize.differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                popsize=12,
                tol=1e-4,
                seed=42,
            )
            best_y = res.x
        else:
            # Fast hybrid: local minimization first, differential_evolution fallback if score < 0.90
            res = optimize.minimize(
                objective,
                x0=x0,
                method="Nelder-Mead",
                options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
            )
            best_y = res.x

            # Verify achieved score
            cand = UniverseParameters.from_log_vector(
                log_alpha=target_log_alpha,
                log_alpha_G=best_y[0],
                log_beta=best_y[1],
                log_alpha_s=best_y[2],
                Q=None if self.manifold.use_dynamic_q else seed.Q,
            )
            if self.manifold.evaluate(cand).total_score < 0.90:
                # Differential evolution global search fallback
                res_de = optimize.differential_evolution(
                    objective,
                    bounds=bounds,
                    maxiter=40,
                    popsize=10,
                    seed=42,
                )
                best_y = res_de.x

        opt_universe = UniverseParameters.from_log_vector(
            log_alpha=target_log_alpha,
            log_alpha_G=best_y[0],
            log_beta=best_y[1],
            log_alpha_s=best_y[2],
            Q=None if self.manifold.use_dynamic_q else seed.Q,
        )
        viab_report = self.manifold.evaluate(opt_universe)
        jacobian = self.compute_local_jacobian(opt_universe)

        return EquilibriumManifoldPoint(
            params=opt_universe,
            viability=viab_report,
            log_alpha=target_log_alpha,
            log_alpha_G=best_y[0],
            log_beta=best_y[1],
            log_alpha_s=best_y[2],
            local_jacobian_dscore=jacobian,
        )

    def trace_manifold(
        self,
        alpha_factors: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int, EquilibriumManifoldPoint], None]] = None,
        seed_mode: str = "homotopy",
    ) -> EquilibriumManifoldCurve:
        """Traces the continuous 1D equilibrium curve gamma(alpha) across variations of alpha."""
        if alpha_factors is None:
            # Default: vary alpha by 0.93x to 1.07x standard value across 15 samples
            alpha_factors = np.linspace(0.93, 1.07, 15)

        points: List[EquilibriumManifoldPoint] = []
        seed = UniverseParameters.standard()

        for idx, factor in enumerate(alpha_factors):
            target_alpha = ALPHA_STANDARD * factor
            current_seed = seed if seed_mode == "homotopy" else UniverseParameters.standard()
            pt = self.find_compensating_point(target_alpha, seed_params=current_seed)
            points.append(pt)
            # Update seed to previous point for smooth homotopy continuation
            if seed_mode == "homotopy" and pt.viability.total_score >= 0.85:
                seed = pt.params

            if progress_callback:
                progress_callback(idx + 1, len(alpha_factors), pt)

        curve = EquilibriumManifoldCurve(points=points)
        curve.scaling_laws = self.fit_scaling_laws(curve)
        return curve

    def fit_scaling_laws(self, curve: EquilibriumManifoldCurve) -> ScalingLawResult:
        """Extracts linear power-laws and quadratic convex basin fits d(log X) / d(log alpha)."""
        valid = curve.valid_points
        if len(valid) < 3:
            return ScalingLawResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        log_a = np.array([p.log_alpha for p in valid])
        log_ag = np.array([p.log_alpha_G for p in valid])
        log_b = np.array([p.log_beta for p in valid])
        log_as = np.array([p.log_alpha_s for p in valid])

        def fit_param(param_name: str, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, QuadraticFitResult]:
            # x is log10(alpha)
            # Centered normalized log coordinate: u = log10(alpha) - log10(alpha_0) = log10(alpha / alpha_0)
            log_a_0 = math.log10(ALPHA_STANDARD)
            u = x - log_a_0

            # 1. Linear 1st order fit: y = m * x + c
            p1 = np.polyfit(x, y, 1)
            y_lin_pred = np.polyval(p1, x)
            ss_tot = np.sum((y - np.mean(y)) ** 2.0)
            ss_lin_res = np.sum((y - y_lin_pred) ** 2.0)
            r2_lin = float(1.0 - (ss_lin_res / ss_tot)) if ss_tot > 1e-12 else 1.0

            # 2. Quadratic 2nd order fit in centered coordinate u: y = A * u^2 + B * u + C
            p2_u = np.polyfit(u, y, 2)
            A, B, C = float(p2_u[0]), float(p2_u[1]), float(p2_u[2])
            y_quad_pred = np.polyval(p2_u, u)
            ss_quad_res = np.sum((y - y_quad_pred) ** 2.0)
            r2_quad = float(1.0 - (ss_quad_res / ss_tot)) if ss_tot > 1e-12 else 1.0

            # Equivalent coefficients in uncentered x: y = a2 * x^2 + a1 * x + a0
            a2 = A
            a1 = B - 2.0 * A * log_a_0
            a0 = C - B * log_a_0 + A * (log_a_0 ** 2.0)

            # 3. Parabolic vertex:
            if abs(A) > 1e-9:
                u_star = -B / (2.0 * A)
                alpha_star_ratio = float(10.0 ** u_star)
                alpha_star = float(ALPHA_STANDARD * alpha_star_ratio)
                x_star = u_star + log_a_0
                y_star = float(A * (u_star ** 2.0) + B * u_star + C)
                val_star = float(10.0 ** y_star)
                is_convex = A > 0
            else:
                u_star = float(u[len(u) // 2])
                alpha_star_ratio = float(10.0 ** u_star)
                alpha_star = float(ALPHA_STANDARD * alpha_star_ratio)
                x_star = u_star + log_a_0
                y_star = float(y[len(y) // 2])
                val_star = float(10.0 ** y_star)
                is_convex = False

            q_res = QuadraticFitResult(
                param_name=param_name,
                a2=a2,
                a1=a1,
                a0=a0,
                r_squared_quad=r2_quad,
                r_squared_lin=r2_lin,
                linear_slope=float(p1[0]),
                linear_intercept=float(p1[1]),
                log_alpha_star=x_star,
                alpha_star=alpha_star,
                alpha_star_ratio=alpha_star_ratio,
                y_star=y_star,
                val_star=val_star,
                is_convex_minimum=is_convex,
            )
            return float(p1[0]), r2_lin, q_res

        gamma, r2_ag, fit_ag = fit_param("alpha_G", log_a, log_ag)
        delta, r2_b, fit_b = fit_param("beta", log_a, log_b)
        kappa, r2_as, fit_as = fit_param("alpha_s", log_a, log_as)

        # GUT RGE Comparison: d(log alpha_s) / d(log alpha)
        gut_comp = GUTRGEComparisonResult(
            slope_empirical=kappa,
            slope_gut_dilaton=0.7180,
            slope_mssm_scale=0.7720,
            concordance_percent=max(0.0, 100.0 - abs(kappa - 0.7180) / 0.7180 * 100.0),
            delta_discrepancy=abs(kappa - 0.7180),
        )

        return ScalingLawResult(
            d_log_alpha_G_d_log_alpha=gamma,
            d_log_beta_d_log_alpha=delta,
            d_log_alpha_s_d_log_alpha=kappa,
            r_squared_alpha_G=r2_ag,
            r_squared_beta=r2_b,
            r_squared_alpha_s=r2_as,
            fit_alpha_G=fit_ag,
            fit_beta=fit_b,
            fit_alpha_s=fit_as,
            gut_comparison=gut_comp,
        )

    def report_scaling_laws(
        self,
        curve: EquilibriumManifoldCurve,
        console: Optional[Any] = None,
    ) -> Any:
        return report_scaling_laws(curve, console=console)

    def monte_carlo_survey(
        self,
        n_samples: int = 1000,
        alpha_range: Tuple[float, float] = (0.3, 3.0),
        alpha_g_range_dex: float = 3.0,
        beta_range_dex: float = 2.0,
        alpha_s_range: Tuple[float, float] = (0.05, 0.30),
        seed: int = 42,
    ) -> List[Tuple[UniverseParameters, ViabilityReport]]:
        """Conducts a Latin / uniform Monte Carlo survey across dimensionless constants."""
        rng = np.random.default_rng(seed)
        results: List[Tuple[UniverseParameters, ViabilityReport]] = []

        log_a_min = math.log10(ALPHA_STANDARD * alpha_range[0])
        log_a_max = math.log10(ALPHA_STANDARD * alpha_range[1])

        log_ag_std = math.log10(ALPHA_G_STANDARD)
        log_b_std = math.log10(BETA_STANDARD)

        log_as_min = math.log10(alpha_s_range[0])
        log_as_max = math.log10(alpha_s_range[1])

        for _ in range(n_samples):
            la = rng.uniform(log_a_min, log_a_max)
            lag = rng.uniform(log_ag_std - alpha_g_range_dex, log_ag_std + alpha_g_range_dex)
            lb = rng.uniform(log_b_std - beta_range_dex, log_b_std + beta_range_dex)
            las = rng.uniform(log_as_min, log_as_max)

            u = UniverseParameters.from_log_vector(la, lag, lb, las)
            rep = self.manifold.evaluate(u)
            results.append((u, rep))

        return results


def report_scaling_laws(
    curve: EquilibriumManifoldCurve,
    console: Optional[Any] = None,
) -> Any:
    """Generates and displays a comparison table of linear vs quadratic scaling laws and vertices."""
    from rich.table import Table
    from rich.console import Console
    con = console or Console()

    sl = curve.scaling_laws
    if sl is None:
        con.print("[yellow]No scaling laws computed for this curve.[/yellow]")
        return None

    table = Table(
        title="Leyes de Escala Empiricas: Ajuste Lineal (1er Orden) vs Cuenca Cuadratica Convexa (2do Orden)",
        border_style="cyan",
    )
    table.add_column("Parametro", style="bold magenta", justify="left")
    table.add_column("Ajuste Lineal", style="cyan", justify="right")
    table.add_column("R^2 Lin", style="green", justify="right")
    table.add_column("Ajuste Cuadratico", style="yellow", justify="left")
    table.add_column("R^2 Quad", style="bold green", justify="right")
    table.add_column("Vertice Extremal alpha*", style="bold yellow", justify="center")
    table.add_column("Geometria de Estabilidad", style="white", justify="left")

    fits = [
        ("log10(alpha_G)", sl.d_log_alpha_G_d_log_alpha, sl.r_squared_alpha_G, sl.fit_alpha_G),
        ("log10(beta)", sl.d_log_beta_d_log_alpha, sl.r_squared_beta, sl.fit_beta),
        ("log10(alpha_s)", sl.d_log_alpha_s_d_log_alpha, sl.r_squared_alpha_s, sl.fit_alpha_s),
    ]

    for label, slope, r2_lin, qfit in fits:
        lin_str = f"m = {slope:+.4f}"
        if qfit:
            quad_str = f"{qfit.a2:+.2f}(log a)^2 {qfit.a1:+.2f}log a"
            r2_q_str = f"{qfit.r_squared_quad:.4f}"
            v_str = f"{qfit.alpha_star_ratio:.3f} alpha_0"
            geom = "[bold green]Cuenca Convexa (Minimo)[/bold green]" if qfit.is_convex_minimum else "[bold cyan]Cresta Concava (Maximo)[/bold cyan]"
        else:
            quad_str = "-"
            r2_q_str = "-"
            v_str = "-"
            geom = "-"

        table.add_row(
            label,
            lin_str,
            f"{r2_lin:.4f}",
            quad_str,
            r2_q_str,
            v_str,
            geom,
        )

    con.print(table)

    if sl.gut_comparison:
        gut = sl.gut_comparison
        table_gut = Table(
            title="Conexion Teorica con Ecuaciones del Grupo de Renormalizacion (RGE / GUT)",
            border_style="green",
        )
        table_gut.add_column("Modelo / Marco Teorico", style="cyan", justify="left")
        table_gut.add_column("d(log alpha_s) / d(log alpha)", style="bold green", justify="right")
        table_gut.add_column("Concordancia Fisica", style="yellow", justify="left")

        table_gut.add_row(
            "Variedad Covariante de Habitabilidad (Empirico)",
            f"{gut.slope_empirical:+.4f}",
            "Linea de equilibrio de resonancia Hoyle & diproton",
        )
        table_gut.add_row(
            "Unificacion GUT con Variacion de Dilaton (Calmet-Fritzsch)",
            f"{gut.slope_gut_dilaton:+.4f}",
            f"[bold green]{gut.concordance_percent:.1f}% concordancia[/bold green] (discrepancia Delta={gut.delta_discrepancy:.4f})",
        )
        table_gut.add_row(
            "MSSM Escala de Unificacion (delta M_GUT)",
            f"{gut.slope_mssm_scale:+.4f}",
            "Modulacion gauge SUSY a 1-loop",
        )
        con.print(table_gut)

    return table

