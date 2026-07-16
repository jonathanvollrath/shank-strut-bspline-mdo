import numpy as np
import gmsh
from dataclasses import dataclass
from scipy.optimize import least_squares
from collections.abc import Sequence
from src.geometry.holes import HoleDefinition, ProjectedHole

from src.geometry.bspline_surface import BSplineSurface

@dataclass (frozen=True)
class DrilledHoleLoop:
    hole_id: str
    uv_points: np.ndarray
    xyz_points: np.ndarray
    center_uv: np.ndarray
    center_xyz: np.ndarray
    axis_xyz: np.ndarray
    radius: float
    residuals: np.ndarray

def normalize_axis(axis: np.ndarray, eps: float=1e-14) -> np.ndarray:
    v = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("Axis vector cannot be zero.")
    return v / norm

def cylinder_frame(axis: np.ndarray, preferred: np.ndarray | None = None):

    a = normalize_axis(axis)

    if preferred is None:
        preferred = np.asarray(preferred, dtype=float)
        e1 = preferred - np.dot(preferred, a) * a

        if np.linalg.norm(e1) > 1e-12:
            e1 = normalize_axis(e1)
            e2 = np.cross(a, e1)
            return e1, e2

    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    ref = min(candidates, key=lambda x: abs(np.dot(x, a)))

    e1 = ref - np.dot(ref, a) * a
    e1 = normalize_axis(e1)
    e2 = normalize_axis(np.cross(a, e1))

    return e1, e2

def make_drilled_loop_hole(
        surface,
        projected_hole,
        n_samples: int=64,
        uv_bounds=((0.0, 1.0), (0.0, 1.0)),
        close_loop: bool=True,
        residual_tol: float=1e-6,
) -> DrilledHoleLoop:

    hole = projected_hole.definition
    hole_id = hole.hole_id

    C = np.asarray(hole.center_xyz)
    a = normalize_axis(hole.axis_xyz)
    r = 0.5 * float(hole.diameter)

    u0 = float(projected_hole.u)
    v0 = float(projected_hole.v)

    S0 = surface.evaluate_single(u0, v0)
    Su0, Sv0 = surface.derivatives(u0, v0)

    n0 = normalize_axis(np.cross(Su0, Sv0))

    axis_normal_alignment = abs(np.dot(n0, a))
    if axis_normal_alignment < 0.05:
        raise ValueError(
            f"Hole axis is nearly perpendicular to surface normal at projected point. "
            f"Alignment: {axis_normal_alignment:.4f}"
        )

    e1, e2 = cylinder_frame(a, preferred=Su0)

    u_min, u_max = uv_bounds[0]
    v_min, v_max = uv_bounds[1]

    uv_points = []
    xyz_points = []
    residuals = []

    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False, dtype=float)

    previous_solution = None

    for theta in thetas:
        radial = r * (np.cos(theta) * e1 + np.sin(theta) * e2)
        Ptheta = C + radial

        def residual(x):
            u, v, t = x
            S = surface.evaluate_single(u, v)
            return S - (Ptheta + t * a)

        def jacobian(x):
            u, v, t = x
            Su, Sv = surface.derivatives(u, v)
            return np.column_stack([Su, Sv, -a])

        A = np.column_stack([Su0, Sv0, -a])
        b = Ptheta - S0

        try:
            du, dv, dt_guess = np.linalg.lstsq(A, b, rcond=None)[0]
            x0_linear = np.array([u0 + du, v0 + dv, dt_guess], dtype=float)
        except np.linalg.LinAlgError:
            x0_linear = np.array([u0, v0, 0.0])

        if previous_solution is not None:
            x0 = previous_solution.copy()
            x0[2] = np.dot(surface.evaluate_single(x0[0], x0[1]) - Ptheta, a)
        else:
            x0 = x0_linear

        lower = np.array([u_min, v_min, -np.inf])
        upper = np.array([u_max, v_max, np.inf])

        result = least_squares(
            residual,
            x0,
            jac=jacobian,
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=50,
        )

        u, v, t = result.x
        S = surface.evaluate_single(u, v)
        err = np.linalg.norm(residual(result.x))

        if err > residual_tol:
            raise RuntimeError(
                f"Failed to intersect drill cylinder for hole {hole_id} "
                f"at theta={theta:.4f}. Residual={err:.6e}"
            )

        uv_points.append([u, v])
        xyz_points.append(S)
        residuals.append(err)

        previous_solution = result.x

    uv_points = np.asarray(uv_points)
    xyz_points = np.asarray(xyz_points)
    residuals = np.asarray(residuals)


    if close_loop:
        uv_points = np.vstack([uv_points, uv_points[0]])
        xyz_points = np.vstack([xyz_points, xyz_points[0]])
        residuals = np.append(residuals, residuals[0])

    return DrilledHoleLoop(
        hole_id=hole_id,
        uv_points=uv_points,
        xyz_points=xyz_points,
        center_uv=np.array([u0, v0]),
        center_xyz=S0,
        axis_xyz=a,
        radius=r,
        residuals=residuals,
    )

def projected_holes_to_loops(
    surface,
    projected_holes: list[ProjectedHole],
    n_samples: int=64,
    uv_bounds=((0.0, 1.0), (0.0, 1.0)),
    close_loop: bool=True,
    residual_tol: float=1e-6,
) -> list[DrilledHoleLoop]:

    loops = []

    for ph in projected_holes:
        loop = make_drilled_loop_hole(
            surface,
            ph,
            n_samples=n_samples,
            uv_bounds=uv_bounds,
            close_loop=close_loop,
            residual_tol=residual_tol,
        )
        loops.append(loop)

    return loops

def make_uv_wire(
    hole_loop: DrilledHoleLoop,
    residual_tol: float=1e-5,
) -> tuple[int, int]:

    uv = np.asarray(hole_loop.uv_points, dtype=float)
    residuals = np.asarray(hole_loop.residuals, dtype=float)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(
            f"{hole_loop.hole_id}: uv_points must have shape (n, 2), "
            f"got {uv.shape}"
        )

    if len(uv) < 3:
        raise ValueError(
            f"{hole_loop.hole_id}: at least three UV points are required"
        )

    if not np.all(np.isfinite(uv)):
        raise ValueError(
            f"{hole_loop.hole_id}: uv_points contain non-finite values"
        )

    if residuals.size and not np.all(np.isfinite(residuals)):
        raise ValueError(
            f"{hole_loop.hole_id}: residuals contain non-finite values"
        )

    if residuals.size and np.max(np.abs(residuals)) > residual_tol:
        raise ValueError(
            f"{hole_loop.hole_id}: maximum projection residual "
            f"{np.max(np.abs(residuals)):.3e} exceeds tolerance "
            f"{residual_tol:.3e}"
        )

    if np.allclose(uv[0], uv[-1]):
        uv = uv[:-1]

    point_tags = [
        gmsh.model.occ.addPoint(
            float(u),
            float(v),
            0.0,
        )
        for u, v in uv
    ]

    curve_tag = gmsh.model.occ.addSpline(
        point_tags + [point_tags[0]]
    )

    wire_tag = gmsh.model.occ.addWire(
        [curve_tag],
        checkClosed=True,
    )

    return curve_tag, wire_tag

def make_outer_uv_wire(
    knot_u: np.ndarray,
    knot_v: np.ndarray,
    degree_u: int,
    degree_v: int,
) -> tuple[list[int], int]:

    u_min = float(knot_u[degree_u])
    u_max = float(knot_u[-degree_u - 1])

    v_min = float(knot_v[degree_v])
    v_max = float(knot_v[-degree_v - 1])

    p00 = gmsh.model.occ.addPoint(u_min, v_min, 0.0)
    p10 = gmsh.model.occ.addPoint(u_max, v_min, 0.0)
    p11 = gmsh.model.occ.addPoint(u_max, v_max, 0.0)
    p01 = gmsh.model.occ.addPoint(u_min, v_max, 0.0)

    curve_tags = [
        gmsh.model.occ.addLine(p00, p10),
        gmsh.model.occ.addLine(p10, p11),
        gmsh.model.occ.addLine(p11, p01),
        gmsh.model.occ.addLine(p01, p00),
    ]

    wire_tag = gmsh.model.occ.addWire(
        curve_tags,
        checkClosed=True,
    )

    return curve_tags, wire_tag

def add_hole_wires_to_surface(
    surface_tag: int,
    hole_loops: Sequence[DrilledHoleLoop],
    knot_u: np.ndarray,
    knot_v: np.ndarray,
    degree_u: int,
    degree_v: int,
    residual_tol: float = 1e-5,
    remove_original_surface: bool = True,
) -> tuple[int, list[int], list[int]]:

    if not hole_loops:
        return surface_tag, [], []

    _, outer_uv_wire_tag = make_outer_uv_wire(
        knot_u=knot_u,
        knot_v=knot_v,
        degree_u=degree_u,
        degree_v=degree_v,
    )

    gmsh.model.occ.synchronize()

    hole_curve_tags: list[int] = []
    hole_wire_tags: list[int] = []

    for hole_loop in hole_loops:
        curve_tag, wire_tag = make_uv_wire(
            hole_loop,
            residual_tol=residual_tol,
        )
        hole_curve_tags.append(int(curve_tag))
        hole_wire_tags.append(int(wire_tag))

    trimming_wire_tags = [
        int(outer_uv_wire_tag),
        *hole_wire_tags,
    ]

    print("UV outer wire:", outer_uv_wire_tag)
    print("UV hole wires:", hole_wire_tags)
    print("All trimming wires:", trimming_wire_tags)

    trimmed_surface_tag = gmsh.model.occ.addTrimmedSurface(
        surfaceTag=int(surface_tag),
        wireTags=trimming_wire_tags,
        wire3D=False,
    )

    if remove_original_surface:
        gmsh.model.occ.remove([(2, int(surface_tag))], recursive=False)

    gmsh.model.occ.synchronize()

    wire_tags, curves_by_wire = gmsh.model.occ.getCurveLoops(
        trimmed_surface_tag
    )

    for i, (wire_tag, curve_tags) in enumerate(
        zip(wire_tags, curves_by_wire)
    ):
        loop_type = "outer boundary" if i == 0 else f"hole {i}"

        print(
            f"  Loop {i}: {loop_type}, "
            f"wire tag={wire_tag}, "
            f"curve tags={list(curve_tags)}"
        )

    expected_loop_count = 1 + len(hole_loops)

    if len(wire_tags) != expected_loop_count:
        raise RuntimeError(
            f"Expected {expected_loop_count} trimming loops "
            f"(1 outer + {len(hole_loops)} holes), but surface "
            f"{trimmed_surface_tag} has {len(wire_tags)}."
        )

    return (
        int(trimmed_surface_tag),
        hole_curve_tags,
        hole_wire_tags,
    )
