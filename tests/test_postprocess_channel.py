from pathlib import Path

import numpy as np

from coldplate_studio.postprocess.contours import write_midplane_contours
from coldplate_studio.postprocess.vtk_io import write_vtk_structured
from coldplate_studio.simulation.channel_3d import run_channel_3d_case


def test_vtk_and_contour_exports(tmp_path: Path):
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 0.2, 3)
    z = np.linspace(0, 0.1, 3)
    T = np.ones((5, 3, 3)) * 300.0
    A = np.linspace(0, 1, 5)[:, None, None] * np.ones((5, 3, 3))
    vtk_path = tmp_path / "t.vtk"
    write_vtk_structured(vtk_path, x, y, z, {"temperature": T, "vof": A})
    assert vtk_path.exists()
    assert "temperature" in vtk_path.read_text()
    t_png, a_png = write_midplane_contours(x, y, z, T, A, tmp_path)
    assert t_png.exists() and a_png.stat().st_size > 1000


def test_channel_3d_end_to_end(tmp_path: Path):
    result = run_channel_3d_case(tmp_path / "case")
    assert result.connectivity_ok
    assert result.interface_triangles > 0
    assert result.vtk_path.exists()
    assert result.fr_element_family == "p1_tetrahedron"
    assert result.temperature_contour.exists()
    assert result.vof_contour.exists()
    assert 0.0 <= result.conservation_final["alpha_min"] <= 1.0
    assert 0.0 <= result.conservation_final["alpha_max"] <= 1.0
