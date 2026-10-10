import numpy as np

from coldplate_studio.fr.elements import lagrange_derivative_matrix, legendre_gauss_lobatto_nodes


def test_derivative_matrix_reproduces_polynomial():
    degree = 4
    nodes = legendre_gauss_lobatto_nodes(degree)
    d = lagrange_derivative_matrix(nodes)
    for p in range(degree + 1):
        values = nodes**p
        deriv = d @ values
        expected = p * nodes ** (p - 1) if p > 0 else np.zeros_like(nodes)
        np.testing.assert_allclose(deriv, expected, rtol=1e-10, atol=1e-10)
