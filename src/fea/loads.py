import re
from pathlib import Path
from dataclasses import dataclass
import gmsh
import numpy as np

from src.fea.physical_grouping import HolePhysicalGroup
from src.mesh.hole_mesh import DrilledHoleLoop
from src.settings import ANALYSIS_DEF_INP_FILE, MESH_INP_FILE, load_config

@dataclass(frozen=True)
class HoleLoad:
    load_name: str
    magnitude: float
    direction: np.ndarray
    position: np.ndarray
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

        hole_load = HoleLoad(
            load_name=load_id,
            magnitude=magnitude,
            direction=direction,
            position=position,
            ref_node=ref_node,
        )
        hole_loads.append(hole_load)
        ref_node += 1

    return hole_loads

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