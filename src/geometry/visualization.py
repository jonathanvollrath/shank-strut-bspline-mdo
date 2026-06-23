import numpy as np
import pyvista as pv

def visualize_bspline_surface(surface_pts: np.ndarray, control_pts: np.ndarray):
    x = surface_pts[:, :, 0]
    y = surface_pts[:, :, 1]
    z = surface_pts[:, :, 2]

    surface_grid = pv.StructuredGrid(x, y, z)
    plotter = pv.Plotter()

    plotter.add_mesh(surface_grid, show_edges=True, opacity=0.75)

    plotter.add_points(control_pts.reshape(-1, 3), point_size=10, render_points_as_spheres=True)

    for i in range(control_pts.shape[0]):
        line = pv.lines_from_points(control_pts[i, :, :])
        plotter.add_mesh(line, line_width=3)

    for j in range(control_pts.shape[1]):
        line = pv.lines_from_points(control_pts[:, j, :])
        plotter.add_mesh(line, line_width=3)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()