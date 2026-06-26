import numpy as np
import gmsh
from src.geometry.bspline_surface import BSplineSurface

def knot_to_unique_multiplicities(knot_vec):
    knots = np.asarray(knot_vec, dtype=float)
    unique_knots = []
    multiplicities = []

    for knot in knots:
        if knot in unique_knots:
            multiplicities[unique_knots.index(knot)] += 1
        else:
            unique_knots.append(knot)
            multiplicities.append(1)

    return unique_knots, multiplicities

def add_bspline_surface_to_gmsh(primary_surface: BSplineSurface, mesh_size=None):

    n_u, n_v, _ = primary_surface.ctrl_net.shape

    point_tags = []

    for j in range(n_v):
        for i in range(n_u):
            x, y, z = primary_surface.ctrl_net[i, j]
            point_tags.append(
                gmsh.model.occ.addPoint(float(x), float(y), float(z), mesh_size or 0.0)
            )

    if primary_surface.knot_u is not None:
        knots_u, mult_u = knot_to_unique_multiplicities(primary_surface.knot_u)
    else:
        knots_u, mult_u = [], []

    if primary_surface.knot_v is not None:
        knots_v, mult_v = knot_to_unique_multiplicities(primary_surface.knot_v)
    else:
        knots_v, mult_v = [], []

    surface_tag = gmsh.model.occ.addBSplineSurface(
        point_tags,
        numPointsU = n_u,
        degreeU = primary_surface.deg_u,
        degreeV = primary_surface.deg_v,
        knotsU=knots_u,
        knotsV=knots_v,
        multiplicitiesU=mult_u,
        multiplicitiesV=mult_v,
    )

    gmsh.model.occ.synchronize()

    return surface_tag