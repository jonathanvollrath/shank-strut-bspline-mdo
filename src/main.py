import gmsh
from src.settings import CONTROL_POINTS_CSV
from src.settings import MESH_FILE
from src.geometry.bspline_surface import import_control_points_from_csv

from src.geometry.bspline_surface import import_knot_vectors_from_config
from src.geometry.bspline_surface import import_degrees_from_config
from src.geometry.bspline_surface import import_samples_from_config
from src.geometry.bspline_surface import BSplineSurface

from src.geometry.visualization import visualize_bspline_surface

from src.mesh.surface_mesh import knot_to_unique_multiplicities
from src.mesh.surface_mesh import add_bspline_surface_to_gmsh


def main():
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

    # Recreate BSplineSurface using Gmsh
    gmsh.initialize()
    gmsh.model.add("shank_strut")

    surface_tag = add_bspline_surface_to_gmsh(prim_surface)

    gmsh.model.addPhysicalGroup(2, [surface_tag], name="primary_surface")

    gmsh.model.mesh.generate(2)
    gmsh.write(str(MESH_FILE))
    gmsh.open(str(MESH_FILE))
    gmsh.fltk.run()
    gmsh.finalize()

    # Visualize the BSpline surface
    # visualize_bspline_surface(surface_eval_pts, ctrl_net)

if __name__ == "__main__":
    main()