"""Command-line entry points for diagnostics and smoke tests."""

from __future__ import annotations

import argparse
import json

from coldplate_studio.capabilities import CAPABILITY_MATRIX
from coldplate_studio.connectivity.diagnostics import run_connectivity_diagnostics
from coldplate_studio.connectivity.graph import build_connectivity_from_extraction
from coldplate_studio.geometry.extraction import extract_channel_fluid_domain
from coldplate_studio.gpu.backend import ComputeBackend


def main() -> None:
    parser = argparse.ArgumentParser(prog="coldplate")
    parser.add_argument("command", choices=["capabilities", "smoke-geometry", "backend-info"])
    args = parser.parse_args()
    if args.command == "capabilities":
        print(json.dumps({k: v.value for k, v in CAPABILITY_MATRIX.items()}, indent=2))
    elif args.command == "backend-info":
        backend = ComputeBackend(gpu_only=False)
        print(json.dumps(backend.diagnostics().to_dict(), indent=2))
    elif args.command == "smoke-geometry":
        result = extract_channel_fluid_domain(1.0, 0.2, 0.1, 0.02, 0.05)
        graph = build_connectivity_from_extraction(result)
        diag = run_connectivity_diagnostics(graph)
        print(
            json.dumps(
                {
                    "extraction_ok": result.ok,
                    "fluid_tags": result.fluid_tags,
                    "solid_tags": result.solid_tags,
                    "interface_surfaces": len(result.interface_surface_tags),
                    "connectivity_ok": diag.ok,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
