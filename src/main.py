from src.settings import CONTROL_POINTS_CSV
from src.geometry.bspline_surface import import_control_points_from_csv

from src.geometry.bspline_surface import import_knot_vectors_from_config
from src.geometry.bspline_surface import import_degrees_from_config
from src.geometry.bspline_surface import import_samples_from_config
from src.geometry.bspline_surface import BSplineSurface

from src.geometry.visualization import visualize_bspline_surface


def main():
    # Load control points from CSV
    ctrl_net = import_control_points_from_csv(CONTROL_POINTS_CSV)

    # Test print control net shape and values
    # print("Control Net Shape:", ctrl_net.shape)
    # print("Control Net:")
    # print(ctrl_net)

    # Import knot vectors and degrees from config
    knot_vector_u, knot_vector_v = import_knot_vectors_from_config()
    degree_u, degree_v = import_degrees_from_config()

    # Test import knot vectors and degrees from config
    # print("Knot Vector U:", knot_vector_u)
    # print("Type of Knot Vector U:", type(knot_vector_u))
    # print("Knot Vector V:", knot_vector_v)
    # print("Type of Knot Vector V:", type(knot_vector_v))
    # print("Degree U:", degree_u)
    # print("Type of Degree U:", type(degree_u))
    # print("Degree V:", degree_v)
    # print("Type of Degree V:", type(degree_v))

    # Create BSplineSurface
    surface = BSplineSurface(ctrl_net, knot_vector_u, knot_vector_v, degree_u, degree_v)
    u_samples, v_samples = import_samples_from_config()
    surface_eval_pts = surface.evaluate(u_samples, v_samples)

    # Test surface evaluation
    # print("Surface Evaluation Points Shape:", surface_eval_pts.shape)

    # Visualize the BSpline surface
    visualize_bspline_surface(surface_eval_pts, ctrl_net)

if __name__ == "__main__":
    main()