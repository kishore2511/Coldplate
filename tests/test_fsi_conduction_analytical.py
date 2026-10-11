import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.dirichlet_p1 import apply_dirichlet_elimination
from coldplate_studio.fr.thermal_p1 import (
    assemble_region_stiffness,
    interface_flux_imbalance,
    interface_heat_flux,
)
from coldplate_studio.mesh.two_material_bar import build_two_material_tet_bar


@pytest.mark.parametrize("nx", [8, 16])
def test_two_material_fsi_conduction_interface_flux_matches_series_solution(nx: int):
    """1D composite bar: T(0)=300 K, T(L)=400 K, k1≠k2; ∫ q·n dA matches series formula."""
    k1, k2 = 1.0, 5.0
    height, width = 0.1, 0.1
    model = build_two_material_tet_bar(nx=nx, height=height, width=width)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    k = assemble_region_stiffness(mesh, mesh.fluid_tets, k1) + assemble_region_stiffness(
        mesh, mesh.solid_tets, k2
    )
    x = mesh.nodes[:, 0]
    pairs = [(int(i), 300.0) for i in np.where(x < 1e-12)[0]] + [
        (int(i), 400.0) for i in np.where(x > 1.0 - 1e-12)[0]
    ]
    b = np.zeros(n)
    a, b = apply_dirichlet_elimination(k.tocsr(), b, pairs)
    t = spsolve(a, b)
    qf, qs = interface_heat_flux(mesh, t, k1, k2)
    imb = interface_flux_imbalance(qf, qs)
    q_theory = 100.0 * height * width / (0.5 / k1 + 0.5 / k2)
    assert abs(qs - q_theory) / q_theory < 0.08
    assert imb < 0.12
