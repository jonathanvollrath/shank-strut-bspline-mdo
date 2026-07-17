from dataclasses import dataclass
import re
import gmsh

@dataclass(frozen=True)
class GmshHoleRegion:
    hole_id: str
    curve_tags: tuple[int, ...]

@dataclass(frozen=True)
class PhysicalGroup:
    dimension: int
    physical_tag: int
    name: str
    entity_tags: tuple[int, ...]

@dataclass(frozen=True)
class GmshPhysicalGroups:
    structure: PhysicalGroup
    symmetry: PhysicalGroup
    holes_by_id: dict[str, PhysicalGroup]
    all_holes: PhysicalGroup

def sanitize_physical_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    return sanitized.strip("_").lower()

def add_named_physical_group(
    dimension: int,
    entity_tags: list[int] | tuple[int, ...],
    name: str
) -> PhysicalGroup:

    unique_tags = tuple(
        sorted({abs(int(tag)) for tag in entity_tags})
    )

    if not unique_tags:
        raise ValueError(f"No valid entity tags provided for physical group '{name}'.")

    existing_entities = {
        tag for dim, tag in gmsh.model.getEntities(dimension)
        if dim == dimension
    }

    missing_tags = [
        tag for tag in unique_tags
        if tag not in existing_entities
    ]

    if missing_tags:
        raise ValueError(
            f"Some entity tags for physical group '{name}' do not exist in the model: {missing_tags}"
        )

    physical_tag = gmsh.model.addPhysicalGroup(dimension, list(unique_tags),)

    gmsh.model.setPhysicalName(dimension, physical_tag, name)

    return PhysicalGroup(
        dimension=dimension,
        physical_tag=physical_tag,
        name=name,
        entity_tags=unique_tags
    )

def get_surface_boundary_curve_tags(surface_tag: int) -> list[int]:

    boundary_curve_tags = gmsh.model.getBoundary([(2, surface_tag)], oriented=False, recursive=False, combined=False)

    return sorted({
        abs(int(tag))
        for dim, tag in boundary_curve_tags
        if dim == 1
    })

def curve_lies_on_x_plane(curve_tag: int, x_value: float=0.0, tolerance: float=1e-6) -> bool:

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, curve_tag)

    return (
        abs(xmin - x_value) <= tolerance and
        abs(xmax - x_value) <= tolerance
    )

def find_symmetry_curve_tags(
    surface_tag: int,
    hole_curve_tags: set[int],
    x_value: float=0.0,
    tolerance: float=1e-6,
    ) -> list[int]:

    boundary_curve_tags = get_surface_boundary_curve_tags(surface_tag)

    symmetry_curve_tags = [
        curve_tag
        for curve_tag in boundary_curve_tags
        if curve_tag not in hole_curve_tags and
        curve_lies_on_x_plane(curve_tag, x_value=x_value, tolerance=tolerance)
    ]

    if not symmetry_curve_tags:
        raise ValueError(
            f"No symmetry curve tags found for surface {surface_tag} "
            f"with hole curve tags {hole_curve_tags} on x={x_value} plane."
        )

    return symmetry_curve_tags

def add_fea_physical_groups(
    surface_tag: int,
    hole_regions: list[GmshHoleRegion],
    x_value: float=0.0,
    tolerance: float=1e-6,
) -> GmshPhysicalGroups:

    gmsh.model.occ.synchronize()

    surface_entities = {
        tag for dim, tag in gmsh.model.getEntities(2) if dim == 2
    }

    if surface_tag not in surface_entities:
        raise ValueError(
            f"Final surface tag {surface_tag} does not exist. "
            "Physical groups must be created after trimming and "
            "after OCC synchronization."
        )

    structure_physical_group = add_named_physical_group(
        dimension=2,
        entity_tags=[surface_tag],
        name="structure"
    )

    holes_by_id: dict[str, GmshHoleRegion] = {}
    all_hole_curve_tags: set[int] = set()

    for hole_region in hole_regions:
        curve_tags = tuple(
            abs(int(tag)) for tag in hole_region.curve_tags
        )

        if not curve_tags:
            raise ValueError(
                f"Hole region '{hole_region.hole_id}' has no valid curve tags."
            )

        physical_name = (f"hole_{sanitize_physical_name(hole_region.hole_id)}")

        holes_by_id[hole_region.hole_id] = (add_named_physical_group(
                dimension=1,
                entity_tags=curve_tags,
                name=physical_name
            )
        )

        all_hole_curve_tags.update(curve_tags)

    if not all_hole_curve_tags:
        raise ValueError(
            "No valid hole curve tags found. "
            "Ensure that the hole regions have valid curve tags."
        )

    all_holes_physical_group = add_named_physical_group(
        dimension=1,
        entity_tags=list(all_hole_curve_tags),
        name="all_holes"
    )

    symmetry_curve_tags = find_symmetry_curve_tags(
        surface_tag=surface_tag,
        hole_curve_tags=all_hole_curve_tags,
        x_value=x_value,
        tolerance=tolerance
    )

    symmetry_physical_group = add_named_physical_group(
        dimension=1,
        entity_tags=symmetry_curve_tags,
        name="symmetry_curve"
    )

    return GmshPhysicalGroups(
        structure=structure_physical_group,
        symmetry=symmetry_physical_group,
        holes_by_id=holes_by_id,
        all_holes=all_holes_physical_group
    )

def print_physical_groups(
    groups: GmshPhysicalGroups,
) -> None:
    print(
        f"Structure: surface tags "
        f"{groups.structure.entity_tags}"
    )

    print(
        f"Symmetry: curve tags "
        f"{groups.symmetry.entity_tags}"
    )

    for hole_id, group in groups.holes_by_id.items():
        print(
            f"Hole {hole_id}: curve tags "
            f"{group.entity_tags}"
        )

    print(
        f"All holes: curve tags "
        f"{groups.all_holes.entity_tags}"
    )

def color_fea_physical_groups(
    groups: GmshPhysicalGroups,
) -> None:
    # Structure
    gmsh.model.setColor(
        [(2, tag) for tag in groups.structure.entity_tags],
        210, 210, 210, 255,
    )

    # Hole curves
    hole_colors = [
        (50, 120, 230),
        (40, 180, 100),
        (220, 140, 30),
        (160, 80, 200),
        (30, 180, 180),
    ]

    for index, group in enumerate(groups.holes_by_id.values()):
        red, green, blue = hole_colors[index % len(hole_colors)]

        gmsh.model.setColor(
            [(1, tag) for tag in group.entity_tags],
            red, green, blue, 255,
        )

    # Symmetry last so it is red if any overlap exists.
    gmsh.model.setColor(
        [(1, tag) for tag in groups.symmetry.entity_tags],
        230, 50, 50, 255,
    )

    # Hide CAD geometry.
    gmsh.option.setNumber("Geometry.Points", 0)
    gmsh.option.setNumber("Geometry.Curves", 0)
    gmsh.option.setNumber("Geometry.Surfaces", 0)

    # Show the colored mesh.
    gmsh.option.setNumber("Mesh.Points", 0)
    gmsh.option.setNumber("Mesh.Lines", 1)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
    gmsh.option.setNumber("Mesh.LineWidth", 4)

def show_only_physical_group(
    group: PhysicalGroup,
) -> None:
    # Hide all geometric entities.
    gmsh.model.setVisibility(
        gmsh.model.getEntities(),
        0,
        recursive=True,
    )

    # Show only the group's entities.
    entities = [
        (group.dimension, tag)
        for tag in group.entity_tags
    ]

    gmsh.model.setVisibility(
        entities,
        1,
        recursive=True,
    )