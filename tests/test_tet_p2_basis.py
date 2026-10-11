import numpy as np

from coldplate_studio.fr.tet_p2_basis import p2_shape_gradients, p2_shape_values


def test_p2_shape_partition_of_unity_and_vertex_interpolation():
    L = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(np.sum(p2_shape_values(L)) - 1.0) < 1e-12
    assert abs(p2_shape_values(L)[0] - 1.0) < 1e-12
    Lmid = np.array([0.5, 0.5, 0.0, 0.0])
    vals = p2_shape_values(Lmid)
    assert abs(vals[4] - 1.0) < 1e-12
    assert abs(vals[0] + vals[1] - 0.0) < 1e-12


def test_p2_gradients_sum_to_zero_for_constant():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    _, vgrads = _tet_gradients(verts)
    L = np.array([0.25, 0.25, 0.25, 0.25])
    gphi = p2_shape_gradients(L, vgrads)
    grad_const = np.sum(gphi, axis=0)
    assert np.linalg.norm(grad_const) < 1e-10
