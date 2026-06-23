from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from src.geometry.bspline_surface import BSplineSurface
import src.settings as settings

@dataclass(frozen=True)
class HoleDefinition:
    hole_id: str
    center_xyz: np.ndarray
    axis_xyz: np.ndarray
    diameter: float
    is_design_var: bool
    tag: str | None = None


@dataclass(frozen=True)
class ProjectedHole:
    definition: HoleDefinition
    u: float
    v: float
    axis_distance: float
    surface_xyz: np.ndarray
    projection_error: float

def load_holes_from_csv(file_path: str | Path | None = None) -> list[HoleDefinition]:
    if file_path is None:
        file_path = settings.HOLES_CSV_PATH

    df = pd.read_csv(file_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    required_columns = {
        "hole_id",
        "x",
        "y",
        "z",
        "is_design_var",
        "diameter",
        "axis_mode",
        "axis_x",
        "axis_y",
        "axis_z",
        "tag",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required hole columns: {sorted(missing_columns)}")

    df["hole_id"] = df["hole_id"].str.strip()
    df["tag"] = df["tag"].str.strip()

    df["is_design_var"] = (
        df["is_design_var"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["is_design_var"].isna().any():
        bad_values = df.loc[df["is_design_var"].isna(), "hole_id"].tolist()
        raise ValueError(f"Invalid is_design_var values for holes: {bad_values}")

    centers = df[["x", "y", "z"]].to_numpy(dtype=float)
    axes = df[["axis_x", "axis_y", "axis_z"]].to_numpy(dtype=float)
    diameters = df["diameter"].to_numpy(dtype=float)

    axis_norms = np.linalg.norm(axes, axis=1)

    if np.any(axis_norms == 0.0):
        bad_holes = df.loc[axis_norms == 0.0, "hole_id"].tolist()
        raise ValueError(f"Hole axes cannot be zero for holes: {bad_holes}")

    if np.any(diameters <= 0.0):
        bad_holes = df.loc[diameters <= 0.0, "hole_id"].tolist()
        raise ValueError(f"Hole diameters must be positive for holes: {bad_holes}")

    axes = axes / axis_norms[:, None]

    holes = []

    for row, center_xyz, axis_xyz, diameter in zip(
        df.itertuples(index=False),
        centers,
        axes,
        diameters,
    ):
        holes.append(
            HoleDefinition(
                hole_id=row.hole_id,
                center_xyz=center_xyz,
                axis_xyz=axis_xyz,
                diameter=float(diameter),
                is_design_var=bool(row.is_design_var),
                tag=row.tag if isinstance(row.tag, str) and row.tag else None,
            )
        )

    return holes

def project_hole_along_axis(surface: BSplineSurface, hole: HoleDefinition, course_samples=15, tol=1e-6):

    center = np.asarray(hole.center_xyz)
    axis = np.asarray(hole.axis_xyz)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Hole axis cannot be zero")
    axis = axis / axis_norm

    u_min = surface.knot_vector_u[surface.degree_u]
    u_max = surface.knot_vector_u[-surface.degree_u - 1]
    v_min = surface.knot_vector_v[surface.degree_v]
    v_max = surface.knot_vector_v[-surface.degree_v - 1]

    best_dis = np.inf
    init_guess = None

    for u in np.linspace(u_min, u_max, course_samples):
        for v in np.linspace(v_min, v_max, course_samples):
            surface_point = np.asarray(surface.evaluate(u, v))

            offset = surface_point - center
            t = np.dot(offset, axis)
            closest_axis_point = center + t * axis

            perp_dis = np.linalg.norm(surface_point - closest_axis_point)

            if perp_dis < best_dis:
                best_dis = perp_dis
                init_guess = [u, v, t]

    def residual(parameters):
        u, v, t = parameters
        surface_point = np.asarray(surface.evaluate(u, v))
        axis_point = center + t * axis
        return surface_point - axis_point

    result = least_squares(
        residual,
        x0=init_guess,
        bounds=(
            [u_min, v_min, -np.inf],
            [u_max, v_max, np.inf],
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    u, v, t = result.x
    error = np.linalg.norm(result.fun)

    if not result.success or error > tol:
        raise ValueError("Hole axis projection failed: "f"residual error = {error:.6g}")

    return ProjectedHole(
        definition=hole,
        u=float(u),
        v=float(v),
        axis_distance=float(t),
        surface_xyz=np.asarray(surface.evaluate(u, v)),
        projection_error=float(error),
    )

def project_holes(surface: BSplineSurface, holes: list[HoleDefinition]):
    projected = []

    for hole in holes:
        result = project_hole_along_axis(
            surface,
            hole,
        )

        projected.append(
            ProjectedHole(
                definition=hole,
                u=result.u,
                v=result.v,
                axis_distance=result.axis_distance,
                surface_xyz=result.surface_xyz,
                projection_error=result.projection_error,
            )
        )

    return projected


