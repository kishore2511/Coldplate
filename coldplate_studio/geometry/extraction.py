"""Automatic coolant-domain extraction via CAD Booleans (spec B2–B4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import gmsh

from coldplate_studio.geometry.validation import GeometryValidationReport, validate_gmsh_model_stats


RegionKind = Literal["fluid", "solid", "void_external", "trapped_cavity"]


@dataclass
class ExtractedRegion:
    tag: int
    kind: RegionKind
    volume_m3: float
    component_id: int


@dataclass
class FluidExtractionResult:
    ok: bool
    fluid_tags: list[int]
    solid_tags: list[int]
    regions: list[ExtractedRegion]
    validation: GeometryValidationReport
    interface_surface_tags: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def extract_channel_fluid_domain(
    length: float,
    width: float,
    height: float,
    fin_thickness: float,
    fin_height: float,
    mesh_size: float = 0.05,
) -> FluidExtractionResult:
    """
    Procedural cold-plate channel: fluid envelope minus central fin solid.

    Demonstrates Boolean subtraction, fluid/solid labeling, and shared interface surfaces.
    """
    gmsh.initialize()
    gmsh.model.add("coldplate_channel")
    try:
        env = gmsh.model.occ.addBox(0, 0, 0, length, width, height)
        fin_x0 = 0.5 * (length - fin_thickness)
        fin = gmsh.model.occ.addBox(fin_x0, 0, 0, fin_thickness, width, fin_height)
        gmsh.model.occ.synchronize()
        fluid_frag, _ = gmsh.model.occ.fragment([(3, env)], [(3, fin)])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        fluid_tags: list[int] = []
        solid_tags: list[int] = []
        regions: list[ExtractedRegion] = []
        for dim, tag in volumes:
            if dim != 3:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            # Fin solid occupies elevated z and narrow x band.
            is_fin = (
                abs((xmax + xmin) - (2 * fin_x0 + fin_thickness)) < 1e-6
                and abs(zmax - fin_height) < 1e-6
                and abs(zmin) < 1e-6
            )
            if is_fin:
                solid_tags.append(tag)
                kind: RegionKind = "solid"
            else:
                fluid_tags.append(tag)
                kind = "fluid"
            regions.append(ExtractedRegion(tag=tag, kind=kind, volume_m3=vol, component_id=tag))
        gmsh.model.addPhysicalGroup(3, fluid_tags, name="coolant")
        gmsh.model.addPhysicalGroup(3, solid_tags, name="fin_solid")
        interface_tags: list[int] = []
        fluid_set = set(fluid_tags)
        solid_set = set(solid_tags)
        for _, stag in gmsh.model.getEntities(2):
            vol_adj, _ = gmsh.model.getAdjacencies(2, stag)
            vol_adj_set = set(vol_adj)
            if vol_adj_set & fluid_set and vol_adj_set & solid_set:
                interface_tags.append(stag)
        gmsh.model.addPhysicalGroup(2, interface_tags, name="fluid_solid_interface")
        stats = {
            "volumes": len(volumes),
            "surfaces": len(gmsh.model.getEntities(2)),
            "bbox_diagonal": float((length**2 + width**2 + height**2) ** 0.5),
        }
        validation = validate_gmsh_model_stats(stats, min_feature_length=mesh_size * 0.25)
        if not fluid_tags:
            validation.add("no_fluid", "Boolean extraction produced no fluid volumes.", "error")
        if not interface_tags:
            validation.add("no_interface", "No shared fluid-solid interface surfaces detected.", "error")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        if mesh_size > 0:
            gmsh.model.mesh.generate(3)
        return FluidExtractionResult(
            ok=validation.ok,
            fluid_tags=fluid_tags,
            solid_tags=solid_tags,
            regions=regions,
            validation=validation,
            interface_surface_tags=interface_tags,
            notes=["Procedural channel: fragment(envelope, fin) with physical groups."],
        )
    finally:
        gmsh.finalize()
