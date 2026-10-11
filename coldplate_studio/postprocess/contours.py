"""Matplotlib contour plots for temperature and VOF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_midplane_contours(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    temperature: np.ndarray,
    vof: np.ndarray,
    output_dir: Path,
    z_index: int | None = None,
) -> tuple[Path, Path]:
    """Save temperature and VOF contour PNGs on a z-plane slice (structured grid)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if z_index is None:
        z_index = len(z) // 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    T_slice = temperature[:, :, z_index]
    A_slice = vof[:, :, z_index]
    t_path = output_dir / "temperature_contour.png"
    a_path = output_dir / "vof_contour.png"
    for data, title, path, cmap in [
        (T_slice, "Temperature (K)", t_path, "inferno"),
        (A_slice, "Volume of fluid", a_path, "Blues"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 3))
        cf = ax.contourf(Y, X, data, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=title)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"{title} at z={z[z_index]:.4f} m")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return t_path, a_path


def write_integral_midplane_contours(
    nodes: np.ndarray,
    temperature: np.ndarray,
    vof: np.ndarray,
    output_dir: Path,
    z_plane: float | None = None,
    tol: float = 0.015,
) -> tuple[Path, Path]:
    """Contour temperature and VOF on nodes near a z plane (integral tet mesh)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if z_plane is None:
        z_plane = float(0.5 * (nodes[:, 2].min() + nodes[:, 2].max()))
    mask = np.abs(nodes[:, 2] - z_plane) <= tol
    if mask.sum() < 10:
        z_plane = float(np.median(nodes[:, 2]))
        mask = np.abs(nodes[:, 2] - z_plane) <= tol
    pts = nodes[mask]
    t_path = output_dir / "temperature_contour.png"
    a_path = output_dir / "vof_contour.png"
    for values, title, path, cmap in [
        (temperature[mask], "Temperature (K)", t_path, "inferno"),
        (vof[mask], "Volume of fluid", a_path, "Blues"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 3))
        sc = ax.tricontourf(pts[:, 1], pts[:, 0], values, levels=20, cmap=cmap)
        fig.colorbar(sc, ax=ax, label=title)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"{title} near z={z_plane:.4f} m (integral mesh)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return t_path, a_path
