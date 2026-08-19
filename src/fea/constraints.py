import re
from pathlib import Path
from dataclasses import dataclass
from src.settings import ANALYSIS_DEF_INP_FILE, MESH_INP_FILE, load_config

def write_constraints(
    symmetry_group_id: str,
    pin_constraint_id: str,
    inp_path: str | Path = ANALYSIS_DEF_INP_FILE,
    config=load_config(),
) -> None:

    inp_path = Path(inp_path)
    text = inp_path.read_text()

    generated_block = f"""\
** --------------------------------------------------
** BEGIN GENERATED CONSTRAINTS
** --------------------------------------------------

** SYMMETRY CONSTRAINTS
*BOUNDARY
{symmetry_group_id}, 1, 1, 0.0 # TODO: Update to auto generate symmetry constraints from config file
{symmetry_group_id}, 5, 6, 0.0

** PIN CONSTRAINTS

** --------------------------------------------------
** END GENERATED CONSTRAINTS
** --------------------------------------------------
    """

    block_pattern = re.compile(
        r"\*\* BEGIN GENERATED CONSTRAINTS.*?"
        r"\*\* END GENERATED CONSTRAINTS\s*",
        flags=re.DOTALL,
    )

    if block_pattern.search(text):
        text = block_pattern.sub(
            generated_block + "\n",
            text,
            count=1
        )
    else:
        step_match = re.search(
            r"(?im)^\s*\*STEP\b",
            text,
        )

        if step_match:
            insert_index = step_match.start()

            text = (
                text[:insert_index].rstrip()
                + "\n\n"
                + generated_block
                + "\n"
                + text[insert_index:].lstrip()
            )
        else:
            text = (
                text.rstrip()
                + "\n\n"
                + generated_block
            )
    inp_path.write_text(text.rstrip() + "\n", encoding="utf-8")