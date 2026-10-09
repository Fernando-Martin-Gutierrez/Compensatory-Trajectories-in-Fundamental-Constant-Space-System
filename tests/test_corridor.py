"""Unit tests for transverse corridor thickness and Lebesgue measure verification."""

import pytest
import math
import numpy as np

from multiverse.constants import UniverseParameters, ALPHA_STANDARD
from multiverse.core.manifolds import AnalyticalManifold
from multiverse.optimizer.equilibrium_search import EquilibriumSearchOptimizer
from multiverse.optimizer.corridor import (
    ParameterTolerance,
    PointCorridorProfile,
    EquilibriumCorridor,
    compute_transverse_tolerances,
    measure_equilibrium_corridor,
    format_corridor_report,
)


def test_compute_transverse_tolerances_standard():
    """Verifies that the standard universe has finite, strictly positive tolerance margins."""
    u = UniverseParameters.standard()
    manifold = AnalyticalManifold()
    tols = compute_transverse_tolerances(u, manifold, threshold_score=0.85)

    assert "alpha_s" in tols
    assert "alpha_G" in tols
    assert "beta" in tols

    # Each parameter must have lower < nominal < upper
    for name, tol in tols.items():
        assert tol.lower_val < tol.nominal_val < tol.upper_val
        assert tol.log_width > 0.0
        assert tol.fractional_span > 0.0

    # Nuclear tolerance in log10 space should be non-trivial but narrow due to Hoyle resonance
    assert tols["alpha_s"].log_width >= 0.01
    # Gravitational tolerance
    assert tols["alpha_G"].log_width >= 0.05


def test_measure_equilibrium_corridor():
    """Verifies the integration of transverse corridor volume along an equilibrium curve."""
    opt = EquilibriumSearchOptimizer()
    # Sample a short curve of 5 points around standard
    alpha_factors = np.array([0.96, 0.98, 1.00, 1.02, 1.04])
    curve = opt.trace_manifold(alpha_factors, seed_mode="homotopy")

    corridor = measure_equilibrium_corridor(curve, threshold_score=0.85)

    assert len(corridor.profiles) == len(curve.points)
    assert corridor.threshold_score == 0.85
    # The mean transverse volume must be strictly positive
    assert corridor.mean_transverse_volume > 0.0
    assert corridor.min_transverse_volume > 0.0
    assert corridor.total_integrated_volume > 0.0

    # Format report check
    report = format_corridor_report(corridor)
    assert "CUANTIFICACION DEL CORREDOR TRANSVERSAL" in report
    assert "DEMOSTRADO" in report
