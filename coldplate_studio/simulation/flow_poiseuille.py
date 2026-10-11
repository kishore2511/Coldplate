"""Steady Poiseuille-style benchmark on integral fluid duct (P1 penalty Stokes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.connectivity.ports import channel_ports_from_user_confirmation
from coldplate_studio.fr.flow_diagnostics import analyze_mass_continuity
from coldplate_studio.fr.flow_p1_penalty import solve_steady_channel_flow
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralModel


def poiseuille_parabolic_ux(y: np.ndarray, height: float, grad_p: float, nu: float) -> np.ndarray:
    """Fully developed x-velocity between parallel plates y∈[0,H] with ∂p/∂x = grad_p < 0."""
    g = -grad_p  # body-force equivalent magnitude
    return (g / (2.0 * nu)) * y * (height - y)


@dataclass
class PoiseuilleBenchmarkResult:
    mesh_size: float
    n_nodes: int
    n_fluid_tets: int
    ux_l2_error: float
    ux_linf_error: float
    analytical_flow_rate: float
    numerical_flow_rate: float
    flow_rate_relative_error: float
    divergence_l2: float
    mass_report: object
    mean_velocity_x: float


def _duct_port_faces(model: IntegralModel, length: float, tol: float = 0.02):
    report = classify_boundary_faces(model.mesh)
    inlet, outlet = [], []
    for f in report.faces:
        if f.face_type != "exterior_fluid":
            continue
        if abs(f.centroid[0] - 0.0) <= tol:
            inlet.append(f.nodes)
        if abs(f.centroid[0] - length) <= tol:
            outlet.append(f.nodes)
    return inlet, outlet


def run_poiseuille_benchmark(
    model: IntegralModel,
    *,
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.2,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    mesh_size: float,
) -> PoiseuilleBenchmarkResult:
    inlet, outlet = _duct_port_faces(model, length)
    ports = channel_ports_from_user_confirmation(model.mesh, inlet, outlet)
    if not ports.ok_for_flow_solve:
        raise RuntimeError(ports.notes)
    port_nodes = {i for f in inlet + outlet for i in f}
    nodes = model.mesh.nodes
    # Quasi-2D Poiseuille: no-slip on y=0,H only; z faces remain natural (uniform in z).
    ytol = 0.02 * height
    wall_nodes = np.array(
        [
            i
            for i in model.mesh.boundary_fluid_nodes
            if int(i) not in port_nodes
            and (nodes[i, 1] <= ytol or nodes[i, 1] >= height - ytol)
        ],
        dtype=int,
    )
    flow = solve_steady_channel_flow(
        model.mesh,
        nu=nu,
        beta=1e3,
        dp_dx=grad_p,
        no_slip_nodes=wall_nodes,
        inlet_faces=inlet,
        outlet_faces=outlet,
    )
    mass = analyze_mass_continuity(model.mesh, flow.velocity, inlet, outlet)
    y = nodes[:, 1]
    # Interior band away from x-ends and z-walls (thin duct: parabolic in y).
    z = nodes[:, 2]
    interior = (
        (nodes[:, 0] > 0.15 * length)
        & (nodes[:, 0] < 0.85 * length)
        & (y > 0.05 * height)
        & (y < 0.95 * height)
        & (z > 0.25 * width)
        & (z < 0.75 * width)
    )
    u_exact = poiseuille_parabolic_ux(y[interior], height, grad_p, nu)
    u_num = flow.velocity[interior, 0]
    diff = u_num - u_exact
    l2 = float(np.sqrt(np.mean(diff**2)))
    linf = float(np.max(np.abs(diff)))
    g = -grad_p
    q_analytical = g * width * (height**3) / (12.0 * nu)
    q_numerical = abs(flow.outlet_flux_m3_s)
    mass_report = mass
    return PoiseuilleBenchmarkResult(
        mesh_size=mesh_size,
        n_nodes=model.mesh.nodes.shape[0],
        n_fluid_tets=model.mesh.fluid_tets.shape[0],
        ux_l2_error=l2,
        ux_linf_error=linf,
        analytical_flow_rate=q_analytical,
        numerical_flow_rate=q_numerical,
        flow_rate_relative_error=abs(q_numerical - q_analytical) / max(abs(q_analytical), 1e-12),
        divergence_l2=flow.divergence_l2,
        mass_report=mass_report,
        mean_velocity_x=flow.mean_velocity_x,
    )
