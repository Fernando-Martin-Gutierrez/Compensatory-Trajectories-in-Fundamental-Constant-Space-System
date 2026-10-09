import pytest
import numpy as np
from multiverse.constants import ALPHA_STANDARD
from multiverse.optimizer.equilibrium_search import EquilibriumSearchOptimizer


def test_equilibrium_compensation_search():
    opt = EquilibriumSearchOptimizer()
    # Vary alpha by +2%
    target_alpha = ALPHA_STANDARD * 1.02
    point = opt.find_compensating_point(target_alpha)

    # Viability should remain > 0.95
    assert point.viability.total_score >= 0.95
    assert point.is_highly_viable is True
    # Local Jacobian should be computable
    assert len(point.local_jacobian_dscore) == 4


def test_manifold_curve_and_scaling_laws():
    opt = EquilibriumSearchOptimizer()
    factors = np.linspace(0.97, 1.03, 5)
    curve = opt.trace_manifold(factors)

    assert len(curve.points) == 5
    assert len(curve.valid_points) >= 4
    assert curve.scaling_laws is not None
    # alpha_s should have positive slope to compensate Hoyle shift
    assert curve.scaling_laws.d_log_alpha_s_d_log_alpha > 0.0
