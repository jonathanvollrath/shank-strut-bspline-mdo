from pathlib import Path
import yaml
import gmsh
from dataclasses import dataclass
from src.settings import load_config

@dataclass(frozen=True)
class MeshConfig:
    default_size: float
    min_size: float
    max_size: float
    surface_algorithm: int
    optimize: bool
    curvature_based: bool

def load_mesh_config(config=load_config()):

    mesh = config['analysis']['mesh']
    size = mesh["size"]
    algorithm = mesh.get("algorithm", {})
    options = mesh.get("options", {})

    default_size = float(size["default"])
    minimum_size = float(size.get("minimum", default_size))
    maximum_size = float(size.get("maximum", default_size))

    if minimum_size <= 0:
        raise ValueError("mesh.size.minimum must be greater than zero")

    if maximum_size < minimum_size:
        raise ValueError(
            "mesh.size.maximum must be greater than or equal to "
            "mesh.size.minimum"
        )

    if not minimum_size <= default_size <= maximum_size:
        raise ValueError(
            "mesh.size.default must be between minimum and maximum"
        )

    return MeshConfig(
        default_size=default_size,
        min_size=minimum_size,
        max_size=maximum_size,
        surface_algorithm=int(algorithm.get("surface", 6)),
        optimize=bool(options.get("optimize", True)),
        curvature_based=bool(
            options.get("curvature_based", False)
        ),
    )

def configure_gmsh_mesh(mesh_config: MeshConfig) -> None:
    gmsh.option.setNumber(
        "Mesh.MeshSizeMin", mesh_config.min_size
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax", mesh_config.max_size
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        int(mesh_config.curvature_based),
    )
    gmsh.option.setNumber(
        "Mesh.Optimize", int(mesh_config.optimize)
    )
    gmsh.option.setNumber(
        "Mesh.Algorithm", mesh_config.surface_algorithm
    )