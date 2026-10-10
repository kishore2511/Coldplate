import numpy as np

from coldplate_studio.fr.scalar_solver import ScalarFR1DConfig, ScalarFR1DSolver


def test_constant_solution_zero_rhs():
    cfg = ScalarFR1DConfig(n_elements=2, degree=3, advection_speed=1.2)
    solver = ScalarFR1DSolver(cfg)
    solver.set_solution(np.ones_like(solver.state.u))
    rhs = solver.rhs(solver.state.u)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_polynomial_advection_exact_steady_residual():
    cfg = ScalarFR1DConfig(n_elements=1, degree=3, advection_speed=1.0)
    solver = ScalarFR1DSolver(cfg)
    x = solver.state.x
    u = x**3
    solver.set_solution(u)
    rhs = solver.rhs(u)
    expected = -3.0 * x**2
    np.testing.assert_allclose(rhs, expected, rtol=1e-9, atol=1e-9)


def test_advection_convergence_order():
    def l2_error(n_elem: int) -> float:
        cfg = ScalarFR1DConfig(n_elements=n_elem, degree=3, advection_speed=1.0)
        solver = ScalarFR1DSolver(cfg)
        x = solver.state.x
        u0 = np.sin(2.0 * np.pi * x)
        solver.set_solution(u0)
        dt = 0.25 / n_elem
        for _ in range(20):
            solver.step_rk4(dt)
        exact = np.sin(2.0 * np.pi * (x - 0.25))
        return float(np.sqrt(np.mean((solver.state.u - exact) ** 2)))

    e1 = l2_error(4)
    e2 = l2_error(8)
    assert e2 < e1
    order = np.log(e1 / e2) / np.log(2.0)
    assert order > 2.0
