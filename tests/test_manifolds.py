import pytest
from multiverse.constants import UniverseParameters, ALPHA_STANDARD, ALPHA_S_STANDARD
from multiverse.core.manifolds import AnalyticalManifold, evaluate_universe_viability


def test_standard_universe_viability():
    u = UniverseParameters.standard()
    rep = evaluate_universe_viability(u)
    assert rep.total_score >= 0.95
    assert rep.is_viable is True
    assert rep.diproton_bound is False
    assert abs(rep.hoyle_resonance_shift_mev) < 0.01
    assert rep.stellar_lifetime_years > 1e10


def test_hydrogen_instability_when_q_smaller_than_beta():
    # If Q < beta, proton captures electron spontaneously: p + e -> n + nu
    u = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=UniverseParameters.standard().alpha_G,
        beta=1e-3,
        alpha_s=ALPHA_S_STANDARD,
        Q=5e-4,  # Q < beta
    )
    manifold = AnalyticalManifold()
    rep = manifold.evaluate(u)
    assert rep.score_hydrogen_dinucleon < 0.05
    assert rep.total_score < 0.50
    assert any("Hydrogen atom unstable" in r for r in rep.failure_reasons)


def test_hoyle_resonance_violation():
    # Shift alpha by 30% without changing alpha_s -> breaks triple alpha resonance
    u = UniverseParameters(
        alpha=ALPHA_STANDARD * 1.30,
        alpha_G=UniverseParameters.standard().alpha_G,
        beta=UniverseParameters.standard().beta,
        alpha_s=ALPHA_S_STANDARD,
    )
    rep = evaluate_universe_viability(u)
    assert rep.score_hoyle_resonance < 0.05
    assert rep.is_viable is False
    assert abs(rep.hoyle_resonance_shift_mev) > 0.50


def test_diproton_bound_state():
    # Increase strong coupling by 10% -> diproton becomes bound
    u = UniverseParameters(
        alpha=ALPHA_STANDARD,
        alpha_G=UniverseParameters.standard().alpha_G,
        beta=UniverseParameters.standard().beta,
        alpha_s=ALPHA_S_STANDARD * 1.10,
    )
    rep = evaluate_universe_viability(u)
    assert rep.diproton_bound is True
    assert any("Diproton" in r for r in rep.failure_reasons)
