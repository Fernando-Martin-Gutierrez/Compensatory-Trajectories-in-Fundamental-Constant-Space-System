"""Unit tests for Multiverse Equilibrium Search, Quadratic Scaling Laws, and Robust CLI Parsing."""

import pytest
import numpy as np

from multiverse.constants import ALPHA_STANDARD, UniverseParameters
from multiverse.optimizer.equilibrium_search import (
    EquilibriumSearchOptimizer,
    EquilibriumManifoldPoint,
    EquilibriumManifoldCurve,
    ScalingLawResult,
    QuadraticFitResult,
    report_scaling_laws,
)
from multiverse.cli import parse_scientific_input


def test_quadratic_scaling_laws_fit_quality():
    """Verify that quadratic polynomial fitting for alpha_G and beta yields R^2_quad >= 0.90."""
    opt = EquilibriumSearchOptimizer()
    alpha_factors = np.linspace(0.93, 1.07, 15)
    curve = opt.trace_manifold(alpha_factors)

    assert curve.scaling_laws is not None
    sl = curve.scaling_laws

    # Alpha_G quadratic fit
    assert sl.fit_alpha_G is not None
    assert sl.fit_alpha_G.r_squared_quad >= 0.90
    # Convex basin of stability (parabolic curvature a2 > 0)
    assert sl.fit_alpha_G.is_convex_minimum
    assert sl.fit_alpha_G.a2 > 0.0

    # Vertex location of alpha_G near standard universe (~ 0.70x to 1.20x alpha_0)
    assert 0.70 <= sl.fit_alpha_G.alpha_star_ratio <= 1.20

    # Beta quadratic fit
    assert sl.fit_beta is not None
    assert sl.fit_beta.r_squared_quad >= 0.90
    assert 0.70 <= sl.fit_beta.alpha_star_ratio <= 1.20

    # Strong coupling alpha_s maintains high monotonic linear fit
    assert sl.r_squared_alpha_s >= 0.95


def test_report_scaling_laws_rich_output():
    """Verify report_scaling_laws generates a valid table and informative summary."""
    opt = EquilibriumSearchOptimizer()
    curve = opt.trace_manifold(np.linspace(0.95, 1.05, 7))

    table = report_scaling_laws(curve)
    assert table is not None

    summary_str = curve.scaling_laws.summary()
    assert "Lineal" in summary_str
    assert "Cuadratico" in summary_str
    assert "Vertice" in summary_str


def test_parse_scientific_input_formats():
    """Verify that parse_scientific_input robustly accepts all user-entered scientific notations."""
    # Standard float and scientific E
    assert parse_scientific_input("0.118", default=1.0) == pytest.approx(0.118)
    assert parse_scientific_input("5.676e-39", default=1.0) == pytest.approx(5.676e-39)
    assert parse_scientific_input("5.676E-39", default=1.0) == pytest.approx(5.676e-39)

    # Multiplication and caret exponent formats
    assert parse_scientific_input("5.676*10^-39", default=1.0) == pytest.approx(5.676e-39)
    assert parse_scientific_input("5.676 x 10^-39", default=1.0) == pytest.approx(5.676e-39)
    assert parse_scientific_input("5.676 * 10^-39", default=1.0) == pytest.approx(5.676e-39)
    assert parse_scientific_input("5.676*10**-39", default=1.0) == pytest.approx(5.676e-39)
    assert parse_scientific_input("5.676 x 10**-39", default=1.0) == pytest.approx(5.676e-39)

    # Positive exponents
    assert parse_scientific_input("1.5 * 10^4", default=1.0) == pytest.approx(15000.0)

    # Default fallbacks
    assert parse_scientific_input("", default=42.0) == 42.0
    assert parse_scientific_input(None, default=42.0) == 42.0
    assert parse_scientific_input("   ", default=137.0) == 137.0


def test_compensating_point_continuity_and_viability():
    """Verify that compensating search produces a highly viable universe near standard."""
    opt = EquilibriumSearchOptimizer()
    pt = opt.find_compensating_point(ALPHA_STANDARD * 0.96)

    assert pt.is_highly_viable
    assert pt.viability.total_score >= 0.95
    assert abs(pt.params.alpha - ALPHA_STANDARD * 0.96) < 1e-8
