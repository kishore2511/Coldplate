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
    """Save temperature and VOF contour PNGs on a z-plane slice."""
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
