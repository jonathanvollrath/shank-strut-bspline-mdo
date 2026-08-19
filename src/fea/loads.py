import re
from pathlib import Path
from dataclasses import dataclass
import gmsh
import numpy as np

from src.fea.physical_grouping import HolePhysicalGroup, PhysicalGroup
from src.mesh.hole_mesh import DrilledHoleLoop
from src.settings import ANALYSIS_DEF_INP_FILE, MESH_INP_FILE, load_config

@dataclass(frozen=True)
class HoleLoad:
    load_name: str
    magnitude: float
    direction: np.ndarray
    position: np.ndarray
    node_groups: list[str]
    ref_node: int | None = None

def normalize_vector(vector: tuple[float, float, float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)

    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        raise ValueError("Direction vector cannot be zero.")

    return vector / magnitude

def configure_hole_loads(holes: dict[str, HolePhysicalGroup], config=load_config()) -> list[HoleLoad]:

    hole_loads: list[HoleLoad] = []
    loads_config = config["analysis"]["loads"]
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    max_node_tag = int(np.max(node_tags))
    ref_node = max_node_tag + 1

    for load_id, load_data in loads_config.items():

        node_groups = load_data["node_groups"]
        missing_groups = [
            group
            for group in node_groups
            if group not in holes.keys()
        ]

        if missing_groups:
            raise ValueError(f"Node groups {missing_groups} specified for load '{load_id}' do not exist in the model.")

        hole_load_config = loads_config[load_id]
        magnitude = hole_load_config["magnitude"]
        direction = normalize_vector(hole_load_config["direction"])
        position = hole_load_config["position"]
        node_groups = hole_load_config["node_groups"]

        hole_load = HoleLoad(
            load_name=load_id,
            magnitude=magnitude,
            direction=direction,
            position=position,
            node_groups=node_groups,
            ref_node=ref_node,
        )
        hole_loads.append(hole_load)
        ref_node += 1

    return hole_loads

def get_physical_group_edges(physical_group: PhysicalGroup) -> set[frozenset[int]]:

    edges: set[frozenset[int]] = set()

    for curve_tag in physical_group.entity_tags:

        element_types, _, node_tags_by_type = (
            gmsh.model.mesh.getElements(dim=1, tag=curve_tag)
        )

        for element_type, node_tags in zip(element_types, node_tags_by_type):

            properties = gmsh.model.mesh.getElementProperties(element_type)

            num_nodes = properties[3]

            if num_nodes != 2:
                raise ValueError(
                    f"Unexpected number of nodes for element type {element_type} "
                    f"on curve {curve_tag}. Expected 2, got {num_nodes}."
                )

            connectivity = np.asarray(
                node_tags, dtype=int
            ).reshape(-1, 2)

            for n1, n2 in connectivity:
                edges.add(
                    frozenset((int(n1), int(n2))
                    )
                )
    return edges

def get_calculix_surface_faces(
    boundary_edges: set[frozenset[int]],
    surface_tag: int,
) -> list[tuple[int, str]]:

    faces: list[tuple[int, str]] = []

    element_types, element_tags_by_type, node_tags_by_type = (
        gmsh.model.mesh.getElements(
            dim=2,
            tag=surface_tag,
        )
    )

    for element_type, element_tags, node_tags in zip(
        element_types,
        element_tags_by_type,
        node_tags_by_type,
    ):
        properties = gmsh.model.mesh.getElementProperties(
            element_type
        )

        num_nodes = properties[3]

        if num_nodes != 3:
            continue

        connectivity = np.asarray(
            node_tags,
            dtype=int,
        ).reshape(-1, 3)

        for element_tag, nodes in zip(element_tags, connectivity):

            n1, n2, n3 = map(int, nodes)
            element_edges = (
                ("S3", frozenset((n1, n2))),
                ("S4", frozenset((n2, n3))),
                ("S5", frozenset((n3, n1))),
            )

            for face_name, edge in element_edges:
                if edge in boundary_edges:
                    faces.append(
                        (int(element_tag), face_name)
                    )

    if len(faces) != len(boundary_edges):
        raise ValueError(
            f"Mismatch between number of boundary edges ({len(boundary_edges)}) "
            f"and number of faces found ({len(faces)})."
        )

    return faces

def write_loads(
    hole_loads: list[HoleLoad],
    analysis_def_inp_path: str | Path = ANALYSIS_DEF_INP_FILE,
    mesh_inp_path: str | Path = MESH_INP_FILE,
) -> None:

    analysis_def_inp_path = Path(analysis_def_inp_path)
    mesh_inp_path = Path(mesh_inp_path)
    analysis_text = analysis_def_inp_path.read_text()
    mesh_text = mesh_inp_path.read_text()

    mesh_text = (
        mesh_text.rstrip() + "\n"
        "*NODE" + "\n"
    )

    analysis_block_l = f"""\
** --------------------------------------------------
** BEGIN GENERATED LOADS
** --------------------------------------------------
** HOLE LOADS
*CLOAD
"""

    for load in hole_loads:
        mesh_text += f"{load.ref_node}, {load.position[0]}, {load.position[1]}, {load.position[2]}\n"
        analysis_block_l += f"""\
{load.ref_node}, 1, {load.magnitude * load.direction[0]}
{load.ref_node}, 2, {load.magnitude * load.direction[1]}
{load.ref_node}, 3, {load.magnitude * load.direction[2]}
""" + "\n"

    mesh_inp_path.write_text(mesh_text.rstrip() + "\n", encoding="utf-8")


    analysis_block_r = f"""\
** --------------------------------------------------
** END GENERATED LOADS
** --------------------------------------------------
"""

    analysis_block = analysis_block_l.rstrip() + "\n" + analysis_block_r.lstrip()

    block_pattern = re.compile(
        r"\*\* --------------------------------------------------\s*"
        r"\*\* BEGIN GENERATED LOADS.*?"
        r"\*\* END GENERATED LOADS\s*"
        r"\*\* --------------------------------------------------\s*",
        flags=re.DOTALL,
    )

    if block_pattern.search(analysis_text):
        analysis_text = block_pattern.sub(
            analysis_block + "\n",
            analysis_text,
            count=1
        )
    else:
        step_match = re.search(
            r"(?im)^\s*\*STEP\b",
            analysis_text,
        )

        if step_match:
            insert_index = step_match.start()

            analysis_text = (
                analysis_text[:insert_index].rstrip()
                + "\n\n"
                + analysis_block
                + "\n"
                + analysis_text[insert_index:].lstrip()
            )
        else:
            analysis_text = (
                analysis_text.rstrip()
                + "\n\n"
                + analysis_block
            )
    analysis_def_inp_path.write_text(analysis_text.rstrip() + "\n", encoding="utf-8")