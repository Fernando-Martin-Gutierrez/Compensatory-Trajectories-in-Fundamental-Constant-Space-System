import pytest
import math
from multiverse.constants import (
    ALPHA_STANDARD,
    ALPHA_G_STANDARD,
    BETA_STANDARD,
    ALPHA_S_STANDARD,
    Q_STANDARD,
    UniverseParameters,
)


def test_standard_universe_values():
    u = UniverseParameters.standard()
    assert math.isclose(u.alpha, 1.0 / 137.035999, rel_tol=1e-5)
    assert u.alpha_G > 0
    assert u.beta < 1e-3
    assert math.isclose(u.alpha_s, 0.118, rel_tol=1e-3)
    assert u.Q > u.beta  # Hydrogen stability condition


def test_universe_log_roundtrip():
    u = UniverseParameters.standard()
    log_vec = u.log_vector
    u2 = UniverseParameters.from_log_vector(*log_vec, Q=u.Q)
    assert math.isclose(u.alpha, u2.alpha, rel_tol=1e-8)
    assert math.isclose(u.alpha_G, u2.alpha_G, rel_tol=1e-8)
    assert math.isclose(u.beta, u2.beta, rel_tol=1e-8)
    assert math.isclose(u.alpha_s, u2.alpha_s, rel_tol=1e-8)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        UniverseParameters(alpha=-0.01)
    with pytest.raises(ValueError):
        UniverseParameters(alpha_G=0.0)
