import numpy as np
import pyvista as pv

import numpy as np
import pyvista as pv
from src.geometry.holes import HoleDefinition
from src.geometry.holes import ProjectedHole


def add_holes_to_plotter(
    plotter: pv.Plotter,
    hole_definitions: list[HoleDefinition] | None = None,
    projected_holes: list[ProjectedHole] | None = None,
    *,
    point_size: float = 12.0,
    axis_length: float = 30.0,
    show_labels: bool = True,
) -> None:
    """
    Visualize imported hole centers, projected hole centers, projection axes,
    and center-to-surface projection lines.

    This function works when either input list is None or empty.

    Parameters
    ----------
    plotter
        Existing PyVista plotter.

    hole_definitions
        Original imported hole definitions.

    projected_holes
        Holes projected onto the B-spline surface.

    point_size
        Size of the rendered hole-center points.

    axis_length
        Total displayed length of each hole axis.

    show_labels
        Whether to show hole ID labels.
    """

    hole_definitions = hole_definitions or []
    projected_holes = projected_holes or []

    imported_by_id = {
        hole.hole_id: hole
        for hole in hole_definitions
    }

    projected_by_id = {
        hole.definition.hole_id: hole
        for hole in projected_holes
    }

    # Include IDs appearing in either input list.
    hole_ids = set(imported_by_id) | set(projected_by_id)

    imported_points: list[np.ndarray] = []
    imported_labels: list[str] = []

    projected_points: list[np.ndarray] = []
    projected_labels: list[str] = []

    for hole_id in sorted(hole_ids):
        imported_definition = imported_by_id.get(hole_id)
        projected_hole = projected_by_id.get(hole_id)

        # ProjectedHole already stores its original HoleDefinition.
        definition = imported_definition

        if definition is None and projected_hole is not None:
            definition = projected_hole.definition

        if definition is None:
            continue

        center = np.asarray(
            definition.center_xyz,
            dtype=float,
        ).reshape(3)

        axis = np.asarray(
            definition.axis_xyz,
            dtype=float,
        ).reshape(3)

        axis_norm = np.linalg.norm(axis)

        if axis_norm <= 1e-12:
            raise ValueError(
                f"Hole {hole_id!r} has a zero-length projection axis."
            )

        axis_unit = axis / axis_norm

        # Show the imported center only when the HoleDefinition list
        # explicitly contains this hole.
        if imported_definition is not None:
            imported_points.append(center)
            imported_labels.append(hole_id)

        # Show the projection axis whenever a definition is available.
        half_axis_length = 0.5 * axis_length

        axis_arrow = pv.Arrow(
            start=center,
            direction=axis_unit,
            scale=axis_length,
            tip_length=0.20,
            tip_radius=0.08,
            shaft_radius=0.025,
        )

        plotter.add_mesh(
            axis_arrow,
            color="orange",
        )

        if projected_hole is not None:
            surface_point = np.asarray(
                projected_hole.surface_xyz,
                dtype=float,
            ).reshape(3)

            projected_points.append(surface_point)
            projected_labels.append(hole_id)

            # Actual center-to-surface projection path.
            plotter.add_mesh(
                pv.Line(center, surface_point),
                color="yellow",
                line_width=4,
                name=f"projection-line-{hole_id}",
            )

    if imported_points:
        imported_points_array = np.asarray(
            imported_points,
            dtype=float,
        )

        plotter.add_points(
            imported_points_array,
            color="red",
            point_size=point_size,
            render_points_as_spheres=True,
            label="Imported hole centers",
            name="imported-hole-points",
        )

        if show_labels:
            plotter.add_point_labels(
                imported_points_array,
                imported_labels,
                point_size=0,
                font_size=10,
                shape_opacity=0.5,
                name="imported-hole-labels",
            )

    if projected_points:
        projected_points_array = np.asarray(
            projected_points,
            dtype=float,
        )

        plotter.add_points(
            projected_points_array,
            color="lime",
            point_size=point_size,
            render_points_as_spheres=True,
            label="Projected hole centers",
            name="projected-hole-points",
        )

        if show_labels:
            plotter.add_point_labels(
                projected_points_array,
                projected_labels,
                point_size=0,
                font_size=10,
                shape_opacity=0.5,
                name="projected-hole-labels",
            )

    if imported_points or projected_points:
        plotter.add_legend()

def visualize_bspline_surface(plotter: pv.Plotter, surface_pts: np.ndarray, control_pts: np.ndarray):
    x = surface_pts[:, :, 0]
    y = surface_pts[:, :, 1]
    z = surface_pts[:, :, 2]

    surface_grid = pv.StructuredGrid(x, y, z)

    plotter.add_mesh(surface_grid, show_edges=True, opacity=0.75)

    plotter.add_points(control_pts.reshape(-1, 3), point_size=10, render_points_as_spheres=True)

    for i in range(control_pts.shape[0]):
        line = pv.lines_from_points(control_pts[i, :, :])
        plotter.add_mesh(line, line_width=3)

    for j in range(control_pts.shape[1]):
        line = pv.lines_from_points(control_pts[:, j, :])
        plotter.add_mesh(line, line_width=3)

def visualize_shank_strut(
    surface_pts: np.ndarray,
    control_pts: np.ndarray,
    hole_definitions: list[HoleDefinition] | None = None,
    projected_holes: list[ProjectedHole] | None = None,
) -> None:
    plotter = pv.Plotter()

    visualize_bspline_surface(
        plotter,
        surface_pts,
        control_pts,
    )

    add_holes_to_plotter(
        plotter,
        hole_definitions=hole_definitions,
        projected_holes=projected_holes,
    )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()