import gmsh
import numpy as np

from src.fea.loads import configure_hole_loads, write_loads
from src.settings import CONTROL_POINTS_CSV, MESH_MSH_FILE, MESH_INP_FILE
from src.geometry.bspline_surface import import_control_points_from_csv
from src.geometry.bspline_surface import import_knot_vectors_from_config, import_degrees_from_config, import_samples_from_config, BSplineSurface
from src.geometry.holes import load_holes_from_csv, HoleDefinition, ProjectedHole, project_holes
from src.geometry.visualization import visualize_shank_strut
from src.mesh.mesh_config import configure_gmsh_mesh, load_mesh_config
from src.mesh.surface_mesh import knot_to_unique_multiplicities, add_bspline_surface_to_gmsh
from src.mesh.hole_mesh import projected_holes_to_loops, add_hole_wires_to_surface
from src.fea.physical_grouping import PhysicalGroup, GmshPhysicalGroups, GmshHoleRegion, add_fea_physical_groups, print_physical_groups, color_fea_physical_groups, show_only_physical_group
from src.fea.analysis_setup import assign_material, convert_surface_elements_to_shells, create_analysis_deck


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
        surface_tag, hole_curves_tags, hole_curves_by_wires_tags = add_hole_wires_to_surface(
            surface_tag=surface_tag,
            hole_loops=drilled_hole_loops,
            knot_u=knot_vector_u,
            knot_v=knot_vector_v,
            degree_u=degree_u,
            degree_v=degree_v,
        )

        gmsh.model.occ.synchronize()

        hole_regions = [
            GmshHoleRegion(
                hole_id=loop.hole_id,
                curve_tags=curve_tag,
                xyz_center=loop.center_xyz
            )
            for loop, curve_tag in zip(
                drilled_hole_loops,
                hole_curves_tags,
                strict=True
            )
        ]

        # print("Hole regions created:")
        # for hole_region in hole_regions:
        #     print(f"Hole ID: {hole_region.hole_id}, Curve Tags: {hole_region.curve_tags}")

        physical_groups = add_fea_physical_groups(
            surface_tag=surface_tag,
            hole_regions=hole_regions,
            x_value=0.0,
            tolerance=1e-5
        )

        # print("Physical groups created:")
        # print_physical_groups(physical_groups)
        print("Physical hole groups created:")
        for hole_id, hole_group in physical_groups.holes_by_id.items():
            print(f"Hole ID: {hole_id}, Physical Group Name: {hole_group.name}, Physical Tag: {hole_group.physical_tag}, Entity Tags: {hole_group.entity_tags}, Center: {hole_group.xyz_center}\n")

        gmsh.model.mesh.generate(2)

        gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -100)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -10)

        # gmsh.write(str(MESH_MSH_FILE))
        gmsh.write(str(MESH_INP_FILE))

        # show_only_physical_group(physical_groups.symmetry)
        # color_fea_physical_groups(physical_groups)

        convert_surface_elements_to_shells()
        create_analysis_deck()
        assign_material(elset_name=physical_groups.structure.name)

        hole_loads = configure_hole_loads(holes=physical_groups.holes_by_id)
        print("Hole loads configured:")
        for load in hole_loads:
            print(f"Load Name: {load.load_name}, Magnitude: {load.magnitude}, Direction: {load.direction}, Position: {load.position}\n")

        write_loads(hole_loads)

        gmsh.fltk.run()

    finally:
        gmsh.finalize()

if __name__ == "__main__":
    main()