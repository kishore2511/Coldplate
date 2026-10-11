import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.dirichlet_p1 import apply_dirichlet_elimination
from coldplate_studio.fr.thermal_interface_flux import galerkin_interface_conduction_rates, interface_flux_report
from coldplate_studio.fr.thermal_p1 import assemble_region_stiffness
from coldplate_studio.mesh.two_material_bar import build_two_material_tet_bar


@pytest.mark.parametrize("nx", [8, 16])
def test_galerkin_interface_flux_balanced_on_analytical_bar(nx: int):
    k1, k2 = 1.0, 5.0
    model = build_two_material_tet_bar(nx=nx, height=0.1, width=0.1)
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
    rep = interface_flux_report(mesh, t, k1, k2)
    assert rep["galerkin_imbalance"] < 0.05
    qf_g, qs_g = galerkin_interface_conduction_rates(mesh, t, k1, k2)
    assert abs(qf_g + qs_g) / max(abs(qf_g), 1e-12) < 0.05
