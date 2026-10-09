import pytest
import math
from multiverse.constants import UniverseParameters
from multiverse.solvers.orbital_dynamics import SymplecticOrbitalIntegrator


def test_standard_orbits_stability():
    u = UniverseParameters.standard()
    integrator = SymplecticOrbitalIntegrator(dt=0.01, total_orbits=50.0)
    res = integrator.run_simulation(u)

    # Planetary orbits should be strictly stable in standard universe
    assert res.is_orbitally_stable is True
    assert res.max_semi_major_axis_drift < 0.01  # Very low energy/semi-major drift
    assert abs(res.lyapunov_exponent) < 0.15


def test_trajectory_recording():
    u = UniverseParameters.standard()
    integrator = SymplecticOrbitalIntegrator(dt=0.01, total_orbits=20.0, sample_stride=20)
    res = integrator.run_simulation(u)

    assert len(res.trajectory_planet_1) > 10
    assert len(res.trajectory_planet_2) > 10
    # Planet 1 starts at distance 1.0
    r0 = math.sqrt(res.trajectory_planet_1[0, 0]**2 + res.trajectory_planet_1[0, 1]**2)
    assert math.isclose(r0, 1.0, rel_tol=1e-3)
