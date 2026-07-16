import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

import src.settings as settings

class BSplineSurface:
    def __init__(self, ctrl_net, knot_u, knot_v, deg_u=3, deg_v=3):
        self.ctrl_net = np.asarray(ctrl_net)
        self.knot_u = np.asarray(knot_u)
        self.knot_v = np.asarray(knot_v)
        self.deg_u = deg_u
        self.deg_v = deg_v

    def basis_u(self, u_samples):
        return BSpline.design_matrix(u_samples, self.knot_u, self.deg_u)

    def basis_v(self, v_samples):
        return BSpline.design_matrix(v_samples, self.knot_v, self.deg_v)

    def basis_derivative(self, samples, knots, degree):
        samples = np.atleast_1d(samples).astype(float)
        knots = np.asarray(knots, dtype=float)

        n_basis = len(knots) - degree - 1

        if degree == 0:
            return np.zeros((len(samples), n_basis))

        lower_basis = BSpline.design_matrix(
            samples,
            knots,
            degree - 1,
        ).toarray()

        dB = np.zeros((len(samples), n_basis))

        for i in range(n_basis):
            left_denom = knots[i + degree] - knots[i]
            right_denom = knots[i + degree + 1] - knots[i + 1]

            if left_denom != 0.0:
                dB[:, i] += degree / left_denom * lower_basis[:, i]

            if right_denom != 0.0:
                dB[:, i] -= degree / right_denom * lower_basis[:, i + 1]

        return dB

    def basis_u_derivative(self, u_samples):
        return self.basis_derivative(u_samples, self.knot_u, self.deg_u)

    def basis_v_derivative(self, v_samples):
        return self.basis_derivative(v_samples, self.knot_v, self.deg_v)

    def evaluate(self, u_samples, v_samples):
        Bu = self.basis_u(u_samples).toarray()
        Bv = self.basis_v(v_samples).toarray()
        return np.einsum("ai,bj,ijc->abc", Bu, Bv, self.ctrl_net)

    def evaluate_single(self, u, v):
        return np.asarray(self.evaluate([u], [v])[0, 0], dtype=float)

    def derivatives(self, u, v):
        u_is_scalar = np.ndim(u) == 0
        v_is_scalar = np.ndim(v) == 0

        u_samples = np.atleast_1d(u).astype(float)
        v_samples = np.atleast_1d(v).astype(float)

        Bu = self.basis_u(u_samples).toarray()
        Bv = self.basis_v(v_samples).toarray()

        dBu = self.basis_u_derivative(u_samples)
        dBv = self.basis_v_derivative(v_samples)

        Su = np.einsum("ai,bj,ijc->abc", dBu, Bv, self.ctrl_net)
        Sv = np.einsum("ai,bj,ijc->abc", Bu, dBv, self.ctrl_net)

        if u_is_scalar and v_is_scalar:
            return Su[0, 0], Sv[0, 0]

        return Su, Sv

def import_control_points_from_csv(file_path: str):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Seperate out the u and v indices and the control points
    u_index = df['u_index'].to_numpy()
    v_index = df['v_index'].to_numpy()
    ctrl_pts = df[['x', 'y', 'z']].to_numpy()

    # Shape X, Y, and Z matrices according to size of u and v indices
    size_n = len(np.unique(u_index))
    size_m = len(np.unique(v_index))
    ctrl_pts_x = np.zeros((size_n, size_m))
    ctrl_pts_y = np.zeros((size_n, size_m))
    ctrl_pts_z = np.zeros((size_n, size_m))

    # Fill control point matrices - access control points P[i,j, x=0, y=1, z=2] using u_index and v_index
    ctrl_net = np.full((size_n, size_m, 3), np.nan)
    ctrl_net[u_index, v_index, :] = ctrl_pts

    return ctrl_net

def import_knot_vectors_from_config(config=settings.load_config()):
    knot_vector_u = np.array(config['geometry']['surface_definition']['knot_vector_u'])
    knot_vector_v = np.array(config['geometry']['surface_definition']['knot_vector_v'])
    return knot_vector_u, knot_vector_v

def import_degrees_from_config(config=settings.load_config()):
    degree_u = config['geometry']['surface_definition']['u_degree']
    degree_v = config['geometry']['surface_definition']['v_degree']
    return degree_u, degree_v

def import_samples_from_config(config=settings.load_config()):
    u_num_samples = np.array(config['analysis']['u_samples'])
    v_num_samples = np.array(config['analysis']['v_samples'])
    u_samples = np.linspace(0, 1, u_num_samples)
    v_samples = np.linspace(0, 1, v_num_samples)
    return u_samples, v_samples


