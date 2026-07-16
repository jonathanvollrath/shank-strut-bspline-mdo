import gmsh
from src.mesh.mesh_config import configure_gmsh_mesh, load_mesh_config
from src.settings import CONTROL_POINTS_CSV, MESH_FILE
from src.geometry.bspline_surface import import_control_points_from_csv

from src.geometry.bspline_surface import import_knot_vectors_from_config, import_degrees_from_config, import_samples_from_config, BSplineSurface

from src.geometry.holes import load_holes_from_csv, HoleDefinition, ProjectedHole, project_holes

from src.geometry.visualization import visualize_shank_strut

from src.mesh.surface_mesh import knot_to_unique_multiplicities, add_bspline_surface_to_gmsh

from src.mesh.hole_mesh import projected_holes_to_loops, add_hole_wires_to_surface


def main():
    # Load mesh configuration from config.yaml
    mesh_config = load_mesh_config()
    # Load control points from CSV
    ctrl_net = import_control_points_from_csv(CONTROL_POINTS_CSV)

    # Import knot vectors and degrees from config
    knot_vector_u, knot_vector_v = import_knot_vectors_from_config()
    degree_u, degree_v = import_degrees_from_config()

    prim_surface = BSplineSurface(
        ctrl_net=ctrl_net,
        knot_u=knot_vector_u,
        knot_v=knot_vector_v,
        deg_u=degree_u,
        deg_v=degree_v
    )

    # Quick visualization of BSpline surface
    u_samples, v_samples = import_samples_from_config()
    surface_eval_pts = prim_surface.evaluate(u_samples, v_samples)

    hole_defs = load_holes_from_csv()
    projected_holes = project_holes(prim_surface, hole_defs)

    visualize_shank_strut(surface_eval_pts, ctrl_net, hole_definitions=hole_defs, projected_holes=projected_holes)

    # Recreate BSplineSurface using Gmsh
    gmsh.initialize()
    gmsh.model.add("shank_strut")

    try:
        drilled_hole_loops = projected_holes_to_loops(prim_surface, projected_holes)

        configure_gmsh_mesh(mesh_config)

        surface_tag = add_bspline_surface_to_gmsh(prim_surface)
        surface_tag, hole_curves, hole_wires = add_hole_wires_to_surface(
            surface_tag=surface_tag,
            hole_loops=drilled_hole_loops,
            knot_u=knot_vector_u,
            knot_v=knot_vector_v,
            degree_u=degree_u,
            degree_v=degree_v,
        )

        gmsh.model.addPhysicalGroup(2, [surface_tag], name="primary_surface")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(MESH_FILE))
        gmsh.open(str(MESH_FILE))
        gmsh.fltk.run()

    finally:
        gmsh.finalize()

if __name__ == "__main__":
    main()