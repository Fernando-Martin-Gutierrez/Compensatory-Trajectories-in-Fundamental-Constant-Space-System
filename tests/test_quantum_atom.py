import pytest
import math
from multiverse.constants import UniverseParameters
from multiverse.solvers.quantum_atom import QuantumAtomSolver


def test_standard_hydrogen_eigenvalues():
    solver = QuantumAtomSolver(n_points=2000, rho_max=40.0)
    u = UniverseParameters.standard()
    res = solver.solve(u)

    # In standard universe: E1 ~ -13.60 eV, E2 ~ -3.40 eV
    assert math.isclose(res.e1_ev, -13.60, abs_tol=0.10)
    assert math.isclose(res.e2_ev, -3.40, abs_tol=0.10)

    # In atomic units: eps_1 = -0.50, eps_2 = -0.125
    assert math.isclose(res.e1_hartree, -0.50, abs_tol=0.01)
    assert math.isclose(res.e2_hartree, -0.125, abs_tol=0.01)


def test_bohr_radius_and_mean_radius():
    solver = QuantumAtomSolver()
    u = UniverseParameters.standard()
    res = solver.solve(u)

    # Bohr radius a_0 ~ 0.529 Angstroms
    a0_ang = res.bohr_radius_si * 1e10
    assert math.isclose(a0_ang, 0.529, abs_tol=0.01)

    # Quantum mean radius for 1s state: <r> = 1.5 * a_0
    mean_ang = res.mean_radius_ground_si * 1e10
    assert math.isclose(mean_ang, 1.5 * a0_ang, rel_tol=0.02)


def test_atomic_scaling_with_varying_alpha():
    solver = QuantumAtomSolver()
    u_std = UniverseParameters.standard()
    u_2x = UniverseParameters(
        alpha=u_std.alpha * 2.0,
        alpha_G=u_std.alpha_G,
        beta=u_std.beta,
        alpha_s=u_std.alpha_s,
    )

    res_std = solver.solve(u_std)
    res_2x = solver.solve(u_2x)

    # E1 scales as alpha^2 -> factor of 4
    ratio_e = res_2x.e1_ev / res_std.e1_ev
    assert math.isclose(ratio_e, 4.0, rel_tol=0.05)

    # a0 scales as 1 / alpha -> factor of 0.5
    ratio_a = res_2x.bohr_radius_si / res_std.bohr_radius_si
    assert math.isclose(ratio_a, 0.5, rel_tol=0.05)


def test_fine_structure_and_21cm_hyperfine():
    solver = QuantumAtomSolver()
    u = UniverseParameters.standard()
    res = solver.solve(u)

    assert res.fine_structure is not None
    fs = res.fine_structure

    # Fine-structure splitting (2P_3/2 - 2P_1/2): ~ 45.3 ueV (~ 10.9 GHz)
    assert math.isclose(fs.delta_e_fs_ev * 1e6, 45.26, rel_tol=0.02)
    assert math.isclose(fs.delta_e_fs_ghz, 10.94, rel_tol=0.02)

    # 21cm hydrogen line: ~ 5.87 ueV (~ 1420 MHz)
    assert math.isclose(fs.hyperfine_21cm_ev * 1e6, 5.87, rel_tol=0.03)
    assert math.isclose(fs.hyperfine_freq_mhz, 1420.4, rel_tol=0.03)

    # Dirac energy levels: 2P_3/2 slightly higher than 2P_1/2
    assert "2p_3/2" in res.dirac_levels
    assert "2p_1/2" in res.dirac_levels
    assert res.dirac_levels["2p_3/2"] > res.dirac_levels["2p_1/2"]


def test_molecular_hydrogen_bonding():
    solver = QuantumAtomSolver()
    u = UniverseParameters.standard()
    res = solver.solve(u)

    assert res.h2_molecule is not None
    h2 = res.h2_molecule

    # Experimental H2 values: D_e ~ 4.74 eV, R_e ~ 0.74 Angstroms
    assert math.isclose(h2.dissociation_energy_de_ev, 4.74, rel_tol=0.05)
    assert math.isclose(h2.equilibrium_bond_length_angstrom, 0.74, rel_tol=0.05)
    assert h2.net_binding_energy_d0_ev > 4.0
    assert h2.is_chemically_stable is True


def test_carbon_covalent_bonding():
    solver = QuantumAtomSolver()
    u = UniverseParameters.standard()
    res = solver.solve(u)

    assert res.cc_covalent_bond is not None
    cc = res.cc_covalent_bond

    # C-C single bond: D_e ~ 3.60 eV, R_e ~ 1.54 Angstroms
    assert math.isclose(cc.dissociation_energy_de_ev, 3.60, rel_tol=0.05)
    assert math.isclose(cc.equilibrium_bond_length_angstrom, 1.54, rel_tol=0.05)
    assert cc.net_binding_energy_d0_ev > 3.0
    assert cc.is_chemically_stable is True
