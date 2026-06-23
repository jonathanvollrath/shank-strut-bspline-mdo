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

    def evaluate(self, u_samples, v_samples):
        Bu = self.basis_u(u_samples).toarray()
        Bv = self.basis_v(v_samples).toarray()
        return np.einsum('ai,bj,ijc->abc', Bu, Bv, self.ctrl_net)

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


