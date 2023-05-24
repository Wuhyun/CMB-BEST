import numpy as np
import os
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from scipy.special import legendre
from scipy.interpolate import RegularGridInterpolator
from numpy.random import default_rng
from scipy.sparse.linalg import cg as conjugate_gradient
import pandas as pd
import h5py
import itertools

CMB_T0 = 2.72548
PLANCK_F_SKY_T = 0.77941
BASE_A_S = 2.100549E-9
BASE_N_SCALAR = 0.9649

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__)) 
CMBBEST_DATA_FILE_PATH =  os.path.join(CURRENT_PATH, "data/cmbbest_data.hdf5")

def set_data_path(path):
    CMBBEST_DATA_FILE_PATH = path


# For numpy PyArray_* API in Cython
np.import_array()

# A trick to find out which integer dtype to use
def GET_SIGNED_NUMPY_INT_TYPE():
    cdef int tmp
    return np.asarray(<int[:1]>(&tmp)).dtype
# If the below fails, there might be problems using int arrays in Cython
assert GET_SIGNED_NUMPY_INT_TYPE() == np.dtype("i")


cdef extern from "tetrapyd.h":
    void monte_carlo_tetrapyd_weights(double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    double *k_grid, double *k_weights, int k_npts, int n_samples)

    void compute_mode_bispectra_covariance(double *bispectra_covariance, 
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts)

    void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts)



def tensor_prod_coeffs(coeffs_1, coeffs_2, pmax_1, pmax_2):
    # Given S(k1,k2,k3) = S1(k1,k2,k3) * S2(k1,k2,k3)
    #                   = (x_abc q_a q_b q_c) * (y_def r_d r_e r_f),
    #                   = z_{(ad)(be)(cf)} (q_a r_d) (q_b r_e) (q_c r_f)
    # Compute z, the tensor product coefficient.
    # Assumes that coeffs_1 and coeffs_2 have the last axis length of pmax_1**3, pmax_2**3.

    tot_pmax = pmax_1 * pmax_2
    tot_coeffs = np.ones((coeffs_1.shape[0], tot_pmax**3), dtype=float)

    inds = np.arange(tot_coeffs.shape[1])

    ad_inds = inds // tot_pmax // tot_pmax
    a_inds = ad_inds // pmax_2
    d_inds = ad_inds % pmax_2

    be_inds = (inds // tot_pmax) % tot_pmax
    b_inds = be_inds // pmax_2
    e_inds = be_inds % pmax_2

    cf_inds = inds % tot_pmax
    c_inds = cf_inds // pmax_2
    f_inds = cf_inds % pmax_2

    abc_inds = a_inds * pmax_1**2 + b_inds * pmax_1 + c_inds
    def_inds = d_inds * pmax_2**2 + e_inds * pmax_2 + f_inds

    tot_coeffs[:,:] = coeffs_1[:,abc_inds] * coeffs_2[:,def_inds]

    return tot_coeffs



class Basis:
    ''' Class for CMBBEST's basis set'''

    def __init__(self, basis_type="Legendre", mode_p_max=10, polarization_on=True, **kwargs):
        self.basis_type = basis_type
        self.mode_p_max = mode_p_max
        self.polarization_on = polarization_on

        if basis_type == "Monomial" and not polarization_on:
            mode_p_max = 4 
            self.data_path = "/trio/T"

        elif basis_type == "Monomial" and polarization_on:
            mode_p_max = 4 
            self.data_path = "/trio/TP"

        elif basis_type == "Legendre" and mode_p_max == 10 and not polarization_on:
            self.data_path = "legendre/base/T"

        elif basis_type == "Legendre" and mode_p_max == 10 and polarization_on:
            self.data_path = "legendre/base/TP"

        elif basis_type == "Legendre" and mode_p_max == 30 and polarization_on:
            self.data_path = "legendre/hires/TP"

        elif basis_type == "SineLegendre1000" and mode_p_max == 10 and polarization_on:
            self.data_path = "legendre/sinleg/TP"

        else:
            print("Basis type '{}' with p_max={} and polarization_on={} is currently not supported".format(basis_type, mode_p_max, polarization_on)) 


        # Load gamma and beta data from files
        use_precomputed_QQ = kwargs.get("use_precomputed_QQ", True)
        self.load_data(use_precomputed_QQ=use_precomputed_QQ)
        #print("Gamma and beta data loaded")

        if not use_precomputed_QQ:
            # When not using the precomputed bispectra covariance (QQ),
            # we need to set up the 1D k grid and the 3D tetrapyd grid

            self.k_grid_size = kwargs.get("k_grid_size", 50) 
            self.k_grid_type = kwargs.get("k_grid_type", "uniform") 
            self.k_grid, self.k_weights = self.create_k_grid(grid_type=self.k_grid_type)

            precomputed_weights_path = kwargs.get("precomputed_weights_path", None)
            if precomputed_weights_path is None:
                self.tetrapyd_indices, self.tetrapyd_grid = self.create_tetrapyd_grid()
                self.tetrapyd_grid_size = self.tetrapyd_grid.shape[1]
                self.tetrapyd_grid_weights = self.compute_tetrapyd_grid_weights()
                #print("Tetrapyd weights computed")
            else:
                #print("Loading precomputed tetrapyd weights from", precomputed_weights_path)
                #self.tetrapyd_indices, self.tetrapyd_grid, self.tetrapyd_grid_weights = self.load_tetraquad(precomputed_weights_path)
                df = pd.read_csv(precomputed_weights_path)
                inds = np.stack([df["i1"], df["i2"], df["i3"]], axis=0)
                grid = np.stack([df["k1"], df["k2"], df["k3"]], axis=0)
                weights = df["weights"].to_numpy()
                self.tetrapyd_indices, self.tetrapyd_grid, self.tetrapyd_grid_weights = inds, grid, weights
                self.tetrapyd_grid_size = self.tetrapyd_grid.shape[1]
        
        # Evaluate mode functions on 1D k grid
        self.mode_function_evaluations = self.mode_functions(self.k_grid)
        #print("1D mode functions evaluated")

        # 3D mode indices
        self.mode_indices, self.mode_symmetry_factor = self.create_mode_indices()
        self.mode_bispectra_evaluations = None

        if use_precomputed_QQ:
            self.mode_bispectra_norms = np.sqrt(np.diag(self.mode_bispectra_covariance))

        else:
            # When not using the precomputed bispectra covariance (QQ), compute it

            # Do not evaluate 3D basis functions on the 3D grid directly,
            # since this can often be too large to store in memory
            self.precomputed_QQ_path = kwargs.get("precomputed_QQ_path", None)

            if self.precomputed_QQ_path is not None:
                #print("Loading precomputed bispectra covariance from", self.precomputed_QQ_path)
                self.mode_bispectra_covariance = np.load(self.precomputed_QQ_path)
            else:
                self.mode_bispectra_covariance = self.compute_mode_bispectra_covariance_C()
            self.mode_bispectra_norms = np.sqrt(np.diag(self.mode_bispectra_covariance))

        #print("{} Basis is now ready".format(self.basis_type))


    def load_data(self, use_precomputed_QQ=True):
        # Load beta and gamma data for the basis
        
        path = self.data_path

        # Conversion factor from B_\zeta to B_\Phi
        # Data is precomputed for B_\zeta, need to convert to B_\Phi
        # \Phi = (3/5) \zeta at superhorizon scales
        # Alphas obtain a factor of (3/5)**3,
        # while the late-time B^{th}_{l1,l2,l3} need be fixed,
        # so beta_\Phi = (5/3)**3 * beta_\zeta
        # and gamma_\Phi = (5/3)**6 * beta_\gamma
        zeta_conv = (5 / 3) ** 3

        with h5py.File(CMBBEST_DATA_FILE_PATH, "r") as hf:
            dg = hf[path]   # h5py data group

            self.beta_cubic = zeta_conv * np.array(dg["beta_cubic"])
            self.beta_linear = zeta_conv * np.array(dg["beta_linear"])
            self.beta_LISW = zeta_conv * np.array(dg["beta_LISW"])
            self.gamma = (zeta_conv ** 2) * np.array(dg["gamma"])
            self.beta = self.beta_cubic - 3 * self.beta_linear

            self.parameter_n_scalar = dg.attrs["parameter_n_scalar"]
            self.parameter_A_scalar = dg.attrs["parameter_A_scalar"]
            self.parameter_k_pivot = dg.attrs["parameter_k_pivot"]
            self.parameter_f_sky = dg.attrs["parameter_f_sky"]

            # Parameter 'A' in Eq (1) of the Planck 2018 PNG paper (1905.05697)
            self.parameter_delta_phi =  (2 * (np.pi ** 2) * ((3 / 5) ** 2)
                                           * (self.parameter_k_pivot ** (1 - self.parameter_n_scalar))
                                           * self.parameter_A_scalar)

            if dg.attrs["mode_orthogonalised"]:
                self.orthogonalisation_coefficients = np.array(dg["orthogonalisation_coefficients"])
                #print("Using orthog. coefficients", self.orthogonalisation_coefficients)
                #print("Using mode orthogonalisation coefficients")
            else:
                self.orthogonalisation_coefficients = None

            if dg.attrs["mode_normalised"]:
                self.mode_normalisations = np.array(dg["mode_normalisations"])
                #print("Using normalisation", self.mode_normalisations)
                #print("Using mode normalisation")
            else:
                self.mode_normalisations = None
            
            if use_precomputed_QQ:
                # load bispectra covariance and tetrapyd quadrature from file
                ds = dg["precomputes/mode_bispectra_covariance"]
                self.mode_bispectra_covariance = np.array(ds)

                self.k_grid_size = int(ds.attrs["tetrapyd_N_k"])
                self.mode_k_min = dg.attrs["mode_k_min"]
                self.mode_k_max = dg.attrs["mode_k_max"]
                assert ds.attrs["quadrature"][-7:] == "uniform"        # TODO!: Relax this restriction
                self.k_grid, self.k_grid_weights = self.create_k_grid(grid_type="uniform")

                quad_group = hf[ds.attrs["quadrature"]]
                self.tetrapyd_indices = np.array(quad_group["indices"])
                self.tetrapyd_grid = np.array(quad_group["grid"])
                self.tetrapyd_grid_weights = np.array(quad_group["weights"])
                self.tetrapyd_grid_size = self.tetrapyd_grid.shape[1]

            if self.basis_type == "Monomial":
                self.mode_k_min = 2e-5
                self.mode_k_max = 2e-1
                self.mode_functions = self.monomial_basis()

            elif self.basis_type == "Legendre":
                self.mode_k_min = dg.attrs["mode_k_min"]
                self.mode_k_max = dg.attrs["mode_k_max"]
                self.mode_functions = self.legendre_basis(self.orthogonalisation_coefficients, self.mode_normalisations)

            elif self.basis_type == "SineLegendre1000":
                self.mode_k_min = dg.attrs["mode_k_min"]
                self.mode_k_max = dg.attrs["mode_k_max"]
                self.mode_omega = dg.attrs["mode_omega"]
                self.mode_functions = self.legendre_basis(self.orthogonalisation_coefficients, self.mode_normalisations)


    def monomial_basis(self):
        # Monomials from 1/k to k^2, with appropriate n_s scaling
        # mode_p_max needs to be exactly 4

        assert self.mode_p_max == 4
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot

        def basis_function(k):
            k_rescaled = k_pivot * ((k / k_pivot) ** ((4-n_s)/3))
            mode_evals = np.zeros((4, len(k)))

            mode_evals[0,:] = k * k / (k_rescaled ** 3)
            mode_evals[1,:] = k * k / (k_rescaled ** 2)
            mode_evals[2,:] = k * k / k_rescaled
            mode_evals[3,:] = k * k

            return mode_evals

        return basis_function
    

    def legendre_basis(self, orthog_coeffs=None, mode_norms=None):
        # Legendre polynomials + one 1/k mode which is orthogonalised
        # Note that the coefficients of orthogonalisation is fixed
        # for each basis set, regardless of the grid specifications

        n_s = self.parameter_n_scalar
        k_max = self.mode_k_max
        k_min = self.mode_k_min
        p_max = self.mode_p_max
        
        def basis_function(k):
            # Rescale k to lie in [-1, 1]
            fact = 2 / (k_max - k_min)
            k_bar = -1 + fact * (k - k_min)
            mode_evals = np.zeros((p_max, len(k)))

            # Modes number 1 to p_max are Legendre polynomials
            for p in range(1, p_max):
                mode_evals[p,:] = legendre(p-1)(k_bar)
            
            # Mode number 0 is 1/k
            mode_evals[0,:] = k_min / np.power(k, 2 - n_s)

            if orthog_coeffs is not None:
                # Orthogonalise the 1/k mode with respect to others
                for p in range(1, p_max):
                    mode_evals[0,:] -= orthog_coeffs[p-1] * mode_evals[p,:]
            
            if mode_norms is not None:
                # Normalise all modes
                mode_evals /= mode_norms[:,np.newaxis]
            
            return mode_evals

        return basis_function
    

    def create_k_grid(self, grid_type="uniform"):
        # Creates a 1D grid of k's from mode_k_min to mode_k_max

        if grid_type == "uniform":
            # Create a uniform one-dimensional k grid 
            k_grid = np.linspace(self.mode_k_min, self.mode_k_max, self.k_grid_size)
            k_weights = np.ones_like(k_grid) * (self.mode_k_max - self.mode_k_min) / (self.k_grid_size - 1)
            k_weights[0] /= 2
            k_weights[-1] /= 2
        
        elif grid_type == "GL":
            gl_nodes, gl_weights = np.polynomial.legendre.leggauss(self.k_grid_size)
            k_grid = self.mode_k_min + ((self.mode_k_max - self.mode_k_min) / 2) * (gl_nodes + 1)
            k_weights = ((self.mode_k_max - self.mode_k_min) / 2) * gl_weights
        
        else:
            print("Grid type {} is currently unsupported".format(grid_type))

        return k_grid, k_weights
    

    def create_tetrapyd_grid(self, include_borderline=False):
        # Creates a 3D grid of k's confined in a 'tetrapyd', satisfying
        # k_max >= k_1 >= k_2 >= k_3 >= k_min  and  k_2 + k_3 >= k_1 
        # if include_borderline is True, keep points that are outside the tetrapyd but
        # its voxel intersects the tetrapyd.

        Nk = self.k_grid_size
        k_grid = self.k_grid
        k_weights = self.k_weights

        tuples = [[i1, i2, i3] for i1 in range(Nk)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)]
        i1, i2, i3 = np.array(tuples).T

        in_tetrapyd = (k_grid[i2] + k_grid[i3] >= k_grid[i1])

        if include_borderline:
            # 1D bounds for k grid points
            interval_bounds = self.mode_k_min
            interval_bounds[0] =  k_grid[0]
            interval_bounds[1:-1] = (k_grid[:-1] + k_grid[1:]) / 2
            interval_bounds[-1] = self.mode_k_max

            # Corners specifying the grid volume (voxel)
            lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
            lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
            lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
            borderline = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                        | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                        | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
            keep = (in_tetrapyd | borderline)

        else:
            keep = in_tetrapyd

        i1, i2, i3 = i1[keep], i2[keep], i3[keep]

        tetrapyd_indices = np.stack([i1, i2, i3], axis=0)
        k1 = k_grid[i1]
        k2 = k_grid[i2]
        k3 = k_grid[i3]
        tetrapyd_grid = np.stack([k1, k2, k3], axis=0)

        return tetrapyd_indices, tetrapyd_grid


    def compute_tetrapyd_grid_weights(self, MC_N_SAMPLES=100000):
        # Use Monte Carlo approach to compute tetrapyd weights

        assert MC_N_SAMPLES > 0
        i1, i2, i3 = self.tetrapyd_indices

        cdef double[:] k_grid = self.k_grid
        cdef double[:] k_weights = self.k_weights
        cdef int k_npts = self.k_grid_size
        cdef int[:] tetra_i1 = np.ascontiguousarray(i1, dtype=np.dtype("i"))
        cdef int[:] tetra_i2 = np.ascontiguousarray(i2, dtype=np.dtype("i"))
        cdef int[:] tetra_i3 = np.ascontiguousarray(i3, dtype=np.dtype("i"))
        cdef int tetra_npts = self.tetrapyd_grid_size
        cdef int n_samples = MC_N_SAMPLES

        # Call C routine
        tetra_weights = np.zeros(self.tetrapyd_grid_size, dtype=np.dtype("d"))
        cdef double[:] tetra_weights_view = tetra_weights
        monte_carlo_tetrapyd_weights(&tetra_weights_view[0], 
                    <int *> &tetra_i1[0], <int *> &tetra_i2[0], <int *> &tetra_i3[0], tetra_npts,
                    &k_grid[0], &k_weights[0], k_npts, n_samples)
        
        return tetra_weights
    

    def compute_tetrapyd_grid_weights_python(self, MC_N_SAMPLES=5000):
        # Slower python routine
        # Use Monte Carlo approach to compute tetrapyd weights

        Nk = self.k_grid_size
        k_grid = self.k_grid
        i1, i2, i3 = self.tetrapyd_indices

        # 1D bounds and weights for k grid points
        interval_bounds = np.zeros(Nk + 1)
        interval_bounds[0] = self.mode_k_min
        interval_bounds[1:-1] = (k_grid[:-1] + k_grid[1:]) / 2
        interval_bounds[-1] = self.mode_k_max
        k_weights = np.diff(interval_bounds)

        # Initialise weights based on symmetry and grid intervals
        tetrapyd_weights = k_weights[i1] * k_weights[i2] * k_weights[i3]
        tetrapyd_weights[(i1 != i2) & (i2 != i3)] *= 6   # Distinct indices
        tetrapyd_weights[(i1 != i2) & (i2 == i3)] *= 3   # Two identical indices
        tetrapyd_weights[(i1 == i2) & (i2 != i3)] *= 3   # Two identical indices

        # Further weights for points of the surface of the tetrapyd
        lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
        lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k1
        lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k1
        need_MC = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                    | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                    | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
        MC_count = np.sum(need_MC)

        # Draw samples uniformly within each cubic cell
        rng = default_rng(seed=0)
        r1 = rng.uniform(lb1[need_MC], ub1[need_MC], size=(MC_N_SAMPLES, MC_count))
        r2 = rng.uniform(lb2[need_MC], ub2[need_MC], size=(MC_N_SAMPLES, MC_count))
        r3 = rng.uniform(lb3[need_MC], ub3[need_MC], size=(MC_N_SAMPLES, MC_count))

        # Count the ratio of samples that lies inside the tetrapyd
        MC_weights = np.sum(((r1 + r2 >= r3) & (r2 + r3 >= r1) & (r3 + r1 >= r2)), axis=0) / MC_N_SAMPLES
        
        tetrapyd_weights[need_MC] *= MC_weights

        return tetrapyd_weights


    def create_mode_indices(self, p_max=None):
        # Creates 3D indices for mode bispectra
        # If p_max=None, use self.mode_p_max

        # Vectorise the 1D mode indices to 3D
        if p_max is None:
            p_max = self.mode_p_max
        p_inds = np.arange(p_max, dtype=np.dtype("i"))
        p1, p2, p3 = np.meshgrid(p_inds, p_inds, p_inds, indexing="ij")
        ordered = (p1 >= p2) & (p2 >= p3)
        p1, p2, p3 = p1[ordered], p2[ordered], p3[ordered]
        mode_indices = np.stack([p1, p2, p3], axis=0)

        # Compute the symmetry factor: 6 if distinct, 3 if two are identical, 1 otherwise
        mode_symmetry_factor = np.ones(mode_indices.shape[1])
        mode_symmetry_factor[(p1 != p2) & (p2 != p3)] = 6
        mode_symmetry_factor[(p1 == p2) & (p2 != p3)] = 3
        mode_symmetry_factor[(p1 != p2) & (p2 == p3)] = 3

        return mode_indices, mode_symmetry_factor


    def evaluate_mode_bispectra(self):
        # Evaluate mode bispectra ('Q') on a 3D tetrapyd grid

        p1, p2, p3 = self.mode_indices
        
        # Compute the 3D mode function
        func_evals  = self.mode_function_evaluations
        t1, t2, t3 = self.tetrapyd_indices
        func_evals_1 = func_evals[p1,:]
        func_evals_2 = func_evals[p2,:]
        func_evals_3 = func_evals[p3,:]
        # Symmetrising over mode indices is equivalent to symetrising over tetrapyd indices
        bisp_sum = (func_evals_1[:,t1] * func_evals_2[:,t2] * func_evals_3[:,t3]
                    + func_evals_1[:,t1] * func_evals_2[:,t3] * func_evals_3[:,t2]
                    + func_evals_1[:,t2] * func_evals_2[:,t1] * func_evals_3[:,t3]
                    + func_evals_1[:,t2] * func_evals_2[:,t3] * func_evals_3[:,t1]
                    + func_evals_1[:,t3] * func_evals_2[:,t1] * func_evals_3[:,t2]
                    + func_evals_1[:,t3] * func_evals_2[:,t2] * func_evals_3[:,t1])

        mode_bispectra_evaluations = bisp_sum / 6.0

        return mode_bispectra_evaluations


    def compute_mode_bispectra_covariance_C(self):
        # Evaluate the covariance matrix ('QQ') between mode bispectra
        # (QQ)_{mn} := <Q_m, Q_n>

        i1, i2, i3 = self.tetrapyd_indices
        p1, p2, p3 = self.mode_indices

        cdef double [::1] tetra_weights = self.tetrapyd_grid_weights
        cdef int [::1] tetra_i1 = i1.astype("i")
        cdef int [::1] tetra_i2 = i2.astype("i")
        cdef int [::1] tetra_i3 = i3.astype("i")
        cdef int tetra_npts = self.tetrapyd_grid_size

        cdef int [::1] mode_p1 = p1.astype("i")
        cdef int [::1] mode_p2 = p2.astype("i")
        cdef int [::1] mode_p3 = p3.astype("i")
        cdef int n_modes = self.mode_indices.shape[1]

        cdef double [:,::1] mode_evals = self.mode_function_evaluations
        cdef int mode_p_max = self.mode_p_max
        cdef int k_npts = self.k_grid_size

        mode_bispectra_covariance = np.zeros((n_modes, n_modes), dtype=np.dtype("d"))
        cdef double [:,::1] mode_bispectra_covariance_view = mode_bispectra_covariance

        # Call the wrapped C function
        compute_mode_bispectra_covariance(&mode_bispectra_covariance_view[0,0],
                        &tetra_weights[0], <int *> &tetra_i1[0], <int *> &tetra_i2[0], <int *> &tetra_i3[0], tetra_npts,
                        <int *> &mode_p1[0], <int *> &mode_p2[0], <int *> &mode_p3[0], n_modes,
                        &mode_evals[0,0], mode_p_max, k_npts)

        return mode_bispectra_covariance 
    

    def basis_expansion(self, model_list, check_convergence=True, silent=False):
        # Expand given model shape functions with respect to separable basis

        N_models = len(model_list)
        N_modes = self.mode_indices.shape[1]
        Q = self.mode_bispectra_evaluations     # (N_modes, N_tetrapyd_points)
        QQ = self.mode_bispectra_covariance     # (N_modes, N_modes)
        norms = self.mode_bispectra_norms       # (N_modes)
        sym_fact = self.mode_symmetry_factor    # (N_modes)
        p1, p2, p3 = self.mode_indices          # (N_modes)
        w = self.tetrapyd_grid_weights          # (N_tetrapyd_points)
        k1, k2, k3 = self.tetrapyd_grid         # (N_tetrapyd_points)
        p_max = self.mode_p_max

        # Evaluate given shape functions and their covariance on a tetrapyd
        # 'S' is a matrix of size (N_models, N_tetrapyd_points)
        # S is assumed to be symmetric under permutations of k1, k2, k3
        S = np.stack([model.shape_function(k1, k2 ,k3) for model in model_list])
        shape_covariance = np.matmul(S * w[np.newaxis,:], S.T)

        # Find the inner product between shapes and mode bispectra
        # 'QS' is a matrix of size (N_models, N_modes)

        if Q is None:
            QS = self.compute_QS_C(S)      # Doesn't require having computed Q
        else:
            QS = np.matmul(S * w[np.newaxis,:], Q.T)       # Requires having computed Q

        # Use conjugate gradient algorithm to solve (alpha @ QQ = QS)
        # 'alpha' is a matrix of size (N_models, N_modes)
        QQ_tilde = QQ / norms[:,np.newaxis] / norms[np.newaxis,:]     # Normalise modes
        alpha = np.zeros((N_models, N_modes))
        for model_no in range(N_models):
            QS_tilde = QS[model_no,:] / norms   # Normalise mode
            alpha_tilde, exit_code = conjugate_gradient(QQ_tilde, QS_tilde, tol=1e-8, atol=0, maxiter=min([10*N_modes, 10000]))
            #print("Shape #{}/{} expanded using CG with exit code {}".format(model_no+1, N_models, exit_code))
            if not silent:
                print("Shape #{}/{} expanded".format(model_no+1, N_models))
            alpha[model_no,:] = alpha_tilde / norms          # Reintroduce normalisation factor

        expansion_coefficients = alpha
        if not silent:
            print("Expansion complete.")

        if check_convergence:
            # Optional check on convergence of the mode expansion
            sum_SS = np.diag(shape_covariance)                      # <S, S>
            sum_SR = np.sum(alpha * QS, axis=1)                     # <S, S_rec>    
            sum_RR = np.sum(alpha * np.matmul(alpha, QQ), axis=1)   # <S_rec, S_rec>    

            convergence_correlation = sum_SR / np.sqrt(sum_SS * sum_RR)
            convergence_correlation = 1.0 - np.abs(1 - convergence_correlation)     # For when corr > 1 due to numerical errors
            # convergence_epsilon = np.sqrt(2 - 2 * convergence_correlation)
            convergence_MSE = np.abs(sum_SS + sum_RR - 2 * sum_SR) / sum_SS

            return expansion_coefficients, shape_covariance, convergence_correlation, convergence_MSE
        else:
            return expansion_coefficients, shape_covariance


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def compute_QS_cython(self, S):
        # Evaluate the inner product('QS') between mode bispectra and shape functions
        # (QS)_{in} := <S_i, Q_n}

        #print("QS Function start!", flush=True)

        i1, i2, i3 = self.tetrapyd_indices
        p1, p2, p3 = self.mode_indices

        mode_evals = self.mode_function_evaluations

        cdef double [::1] tetra_weights = self.tetrapyd_grid_weights
        cdef double [:,:] mode_evals_i1 = mode_evals[:,i1]
        cdef double [:,:] mode_evals_i2 = mode_evals[:,i2]
        cdef double [:,:] mode_evals_i3 = mode_evals[:,i3]

        cdef int tetra_npts = self.tetrapyd_grid_size

        #print("mode evals declared", flush=True)

        cdef int [::1] mode_p1 = p1.astype("i")
        cdef int [::1] mode_p2 = p2.astype("i")
        cdef int [::1] mode_p3 = p3.astype("i")
        cdef int n_modes = self.mode_indices.shape[1]

        #print("mode p1,2,3s declared", flush=True)

        cdef double [:,:] S_view = S
        cdef int n_shapes = S.shape[0]

        QS = np.zeros((n_shapes, n_modes), dtype=np.dtype("d"))
        cdef double [:,::1] QS_view = QS

        cdef double sum = 0
        cdef double * tmp
        cdef int n_p1, n_p2, n_p3
        cdef Py_ssize_t n, k, i

        #print("Loop starts", flush=True)

        with nogil, parallel():

            tmp = <double *> malloc(sizeof(double) * tetra_npts)

            for n in prange(n_modes):
                n_p1 = mode_p1[n]
                n_p2 = mode_p2[n]
                n_p3 = mode_p3[n]

                # Compute and store Q_n(k)
                for k in range(tetra_npts):
                    tmp[k] = tetra_weights[k] * mode_evals_i1[n_p1,k] * mode_evals_i2[n_p2,k] * mode_evals_i3[n_p3,k]
                    #TODO: need to update this to include symmetry factors

                for i in range(n_shapes):

                    # Compute <S_i, Q_n>
                    sum = 0
                    for k in range(tetra_npts):
                        sum = sum + tmp[k] * S_view[i,k]
                    
                    QS_view[i,n] = sum
            
            free(tmp)

        return QS


    def compute_QS_C(self, S):
        # Evaluate the inner product('QS') between mode bispectra and shape functions
        # (QS)_{in} := <S_i, Q_n}

        #print("QS Function start!", flush=True)

        i1, i2, i3 = self.tetrapyd_indices
        p1, p2, p3 = self.mode_indices

        cdef int mode_p_max = self.mode_p_max
        cdef int k_npts = self.k_grid_size

        cdef double [::1] tetra_weights = self.tetrapyd_grid_weights
        cdef int [::1] tetra_i1 = i1.astype("i")
        cdef int [::1] tetra_i2 = i2.astype("i")
        cdef int [::1] tetra_i3 = i3.astype("i")

        cdef int tetra_npts = self.tetrapyd_grid_size

        cdef int [::1] mode_p1 = p1.astype("i")
        cdef int [::1] mode_p2 = p2.astype("i")
        cdef int [::1] mode_p3 = p3.astype("i")
        cdef int n_modes = self.mode_indices.shape[1]

        cdef double [::1] S_view = S.flatten()
        cdef int n_shapes = S.shape[0]

        cdef double [:,::1] mode_evals = self.mode_function_evaluations

        QS = np.zeros((n_shapes, n_modes), dtype=np.dtype("d"))
        QS = QS.flatten()
        cdef double [::1] QS_view = QS
        S = S.flatten()

        #print("QS start!")
        #print(n_modes, mode_p_max, k_npts, tetra_npts, n_shapes)

        # Call the wrapped C function

        compute_QS(&QS_view[0], &S_view[0], n_shapes,
                    &tetra_weights[0], <int *> &tetra_i1[0], <int *> &tetra_i2[0], <int *> &tetra_i3[0], tetra_npts,
                    <int *> &mode_p1[0], <int *> &mode_p2[0], <int *> &mode_p3[0], n_modes,
                    &mode_evals[0,0], mode_p_max, k_npts)

        QS = QS.reshape((n_shapes, n_modes))
        #print("done!")

        return QS


    def constrain_models(self, model_list, expansion_coefficients=None, convergence_correlation=None, convergence_MSE=None, silent=False):
        # Main function for constraining different models!
        # 'model_list' is a list of cambbest.Model instances
        # Returns a pandas DataFrame containing the results

        if expansion_coefficients is None:
            coeff, shape_cov, conv_corr, conv_MSE = self.basis_expansion(model_list, check_convergence=True, silent=silent)
            expansion_coefficients = coeff
            shape_covariance = shape_cov
            convergence_correlation = conv_corr
            convergence_MSE = conv_MSE
        else:
            coeff = expansion_coefficients


        if self.basis_type == "SineLegendre1000":
            # For the targeted basis, 'coeff' is the expansion coeff of the envelope
            # with respect to Legendre basis. For the full constraints, need the coefficents
            # multiplied by the phase under consideration. (sin(omega * k + phi))

            # First, extend the coefficients to shape (p_max**3)
            p1, p2, p3 = self.mode_indices
            sym_fact = self.mode_symmetry_factor
            p_max = self.mode_p_max

            full_coeffs = np.zeros((coeff.shape[0], p_max ** 3))
            for pp1, pp2, pp3 in itertools.permutations([p1, p2, p3]):
                full_coeffs[:,(pp1 * p_max ** 2 + pp2 * p_max + pp3)] = coeff[:,:] / sym_fact[np.newaxis,:]

            # Take tensor product with sinusoidal coeffs (p_max = 2)
            # S(k1,k2,k3) = A(k1,k2,k3) * sin(w(k1+k2+k3) + phase)
            def sine_coeff(phase):
                phase *= np.pi / 180.0  # degrees to radians
                sinp, cosp = np.sin(phase), np.cos(phase)
                # Im[exp(i(k1+k2+k3+phase))] = sinp * Re[...] + cosp * Im[...]
                # sss ssc scs scc css csc ccs ccc
                return np.array([-cosp, -sinp, -sinp, cosp, -sinp, cosp, cosp, sinp])

            # Create two (sin and cos) coefficients per each row
            full_coeffs = np.repeat(full_coeffs, 2, axis=0)
            sin_coeffs = np.tile(np.array([sine_coeff(0), sine_coeff(90)]), (len(model_list), 1))

            # Tensor product on coefficients
            sinleg_coeffs = tensor_prod_coeffs(full_coeffs, sin_coeffs, p_max, 2)

            # Reduce the coefficients to shape (n_modes) corresponding to 2*p_max
            new_mode_inds, new_sym_fact = self.create_mode_indices(p_max=2*self.mode_p_max)
            q1, q2, q3 = new_mode_inds
            
            coeff = np.zeros((sinleg_coeffs.shape[0], q1.shape[0]))
            for qq1, qq2, qq3 in itertools.permutations([q1, q2, q3]):
                coeff = coeff + sinleg_coeffs[:,(qq1 * p_max**2 + qq2 * p_max + qq3)]
            coeff[:,:] = coeff[:,:] * new_sym_fact[np.newaxis,:] / 6

            print("NOTE: coefficients converted for sine-Legendre basis")


        # Constraints
        beta = self.beta                # (N_sims, n_modes)
        beta_LISW = self.beta_LISW      # (n_modes)
        gamma = self.gamma              # (n_modes, n_modes)
        f_sky = self.parameter_f_sky

        fisher_matrix = np.matmul(coeff, np.matmul(coeff, gamma).T) * f_sky / 6.
        # Note that f_NL constraints account for the lensing-ISW bias
        coeff_dot_beta = np.matmul(coeff, (beta - f_sky * beta_LISW[np.newaxis,:]).T)   # (N_models, N_sims)

        # Treat each model independently
        single_f_NL = (1/6) * coeff_dot_beta / np.diag(fisher_matrix)[:,np.newaxis] # (N_models, N_sims)
        single_fisher_sigma = np.sqrt(1 / np.diag(fisher_matrix))                   # (N_models)
        single_sample_sigma = np.std(single_f_NL[:,1:], axis=1)                     # (N_models)
        single_LISW_bias = ((1/6) * np.matmul(coeff, f_sky * beta_LISW)
                                / np.diag(fisher_matrix))                           # (N_models)

        # Marginalise over other models
        try:
            inverse_fisher_matrix = np.linalg.inv(fisher_matrix)
            marginal_f_NL = (1/6) * np.matmul(inverse_fisher_matrix, coeff_dot_beta)    # (N_models, N_sims)
            marginal_fisher_sigma = np.sqrt(np.diag(inverse_fisher_matrix))             # (N_models)
            marginal_sample_sigma = np.std(marginal_f_NL[:,1:], axis=1)                 # (N_models)
            marginal_LISW_bias = ((1/6) * np.matmul(inverse_fisher_matrix,
                                                np.matmul(coeff, f_sky * beta_LISW)))   # (N_models)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print("Skipping marginal constraints due to colinearity in the models considered.")
                inverse_fisher_matrix = None
                marginal_f_NL = None
                marginal_fisher_sigma = None
                marginal_sample_sigma = None
                marginal_LISW_bias = None
            else:
                raise


        # Save the results
        constraints = Constraints(basis=self,
                                  model_list=model_list,
                                  expansion_coefficients=expansion_coefficients,
                                  convergence_correlation=convergence_correlation,
                                  convergence_MSE=convergence_MSE,
                                  fisher_matrix=fisher_matrix,
                                  single_f_NL=single_f_NL,
                                  single_fisher_sigma=single_fisher_sigma,
                                  single_sample_sigma=single_sample_sigma,
                                  single_LISW_bias=single_LISW_bias,
                                  marginal_f_NL=marginal_f_NL,
                                  marginal_fisher_sigma=marginal_fisher_sigma,
                                  marginal_sample_sigma=marginal_sample_sigma,
                                  marginal_LISW_bias=marginal_LISW_bias,
                                 )

        return constraints



class Constraints:
    ''' Class for storing cmbbest's computation results (constraints).
        Contains references (pointers) to the basis and model list '''

    def __init__(self, **kwargs):
        #  Core info: basis, model, and coefficients
        self.basis = kwargs.get("basis")
        self.model_list = kwargs.get("model_list")
        self.expansion_coefficients = kwargs.get("expansion_coefficients")

        # Deep copies of some basis and model information
        self.basis_type = self.basis.basis_type
        self.mode_p_max = self.basis.mode_p_max
        self.polarization_on = self.basis.polarization_on
        self.n_models = len(self.model_list)
        self.shape_name_list = [model.shape_name for model in self.model_list]

        # Basis expansion related data
        self.convergence_correlation = kwargs.get("convergence_correlation")
        self.convergence_epsilon = kwargs.get("convergence_epsilon")
        self.convergence_MSE = kwargs.get("convergence_MSE")

        # Single shape analysis results
        self.fisher_matrix = kwargs.get("fisher_matrix")
        self.single_f_NL = kwargs.get("single_f_NL")
        self.single_fisher_sigma = kwargs.get("single_fisher_sigma")
        self.single_sample_sigma = kwargs.get("single_sample_sigma")
        self.single_LISW_bias = kwargs.get("single_LISW_bias")

        # Multi shape analysis results
        self.marginal_f_NL = kwargs.get("marginal_f_NL")
        self.marginal_fisher_sigma = kwargs.get("marginal_fisher_sigma")
        self.marginal_sample_sigma = kwargs.get("marginal_sample_sigma")
        self.marginal_LISW_bias = kwargs.get("marginal_LISW_bias")
    

    def to_dataframe(self, full_result=False):
        # Return the results as a pandas dataframe 
        # If full_result=True, also show results from simulations 

        df = pd.DataFrame()

        if self.basis_type == "SineLegendre1000":
            df["shape_name"] = [name + phase for name in self.shape_name_list for phase in [" sin", " cos"]]

            if self.convergence_correlation is not None:
                df["convergence_correlation"] = np.repeat(self.convergence_correlation, 2)
                df["convergence_epsilon"] = np.repeat(np.sqrt(2 - 2 * self.convergence_correlation), 2)
            if self.convergence_MSE is not None:
                df["convergence_MSE"] = np.repeat(self.convergence_MSE, 2)

        else:
            df["shape_name"] = self.shape_name_list

            if self.convergence_correlation is not None:
                df["convergence_correlation"] = self.convergence_correlation
                df["convergence_epsilon"] = np.sqrt(2 - 2 * self.convergence_correlation)
            if self.convergence_MSE is not None:
                df["convergence_MSE"] = self.convergence_MSE
        
        if self.marginal_f_NL is None:
            # Independent constraints only
            df = df.join(pd.DataFrame(
                                        {
                                            "single_f_NL": self.single_f_NL[:,0],      # The 0th map is the observed map
                                            "single_fisher_sigma": self.single_fisher_sigma,
                                            "single_sample_sigma": self.single_sample_sigma,
                                            "single_LISW_bias": self.single_LISW_bias,
                                        }))
        else:
            # All constraints
            df = df.join(pd.DataFrame(
                                        {
                                            "single_f_NL": self.single_f_NL[:,0],      # The 0th map is the observed map
                                            "single_fisher_sigma": self.single_fisher_sigma,
                                            "single_sample_sigma": self.single_sample_sigma,
                                            "single_LISW_bias": self.single_LISW_bias,
                                            "marginal_f_NL": self.marginal_f_NL[:,0],
                                            "marginal_fisher_sigma": self.marginal_fisher_sigma,
                                            "marginal_sample_sigma": self.marginal_sample_sigma,
                                            "marginal_LISW_bias": self.marginal_LISW_bias
                                        }))
        
        if full_result:
            # Include all simulation map constraints
            n_maps = self.single_f_NL.shape[1]
            n_models = self.n_models

            # Duplicate rows (n_maps) times
            df = df.loc[df.index.repeat(n_maps)].reset_index()

            # Insert map number values 
            map_no_index = df.columns.get_loc("single_f_NL") + 1
            map_no_values = np.tile(np.arange(n_maps), n_models) 
            df.insert(map_no_index, "map_number", map_no_values)

            # Add correct f_NL values for each map
            df["single_f_NL"] = self.single_f_NL.flatten()
            if self.marginal_f_NL is not None:
                df["marginal_f_NL"] = self.marginal_f_NL.flatten()
        
        return df

    
    def to_csv(self, filename, full_result=True):
        # Save the results to a csv file
        # If full_result=True, also save results from simulations 

        df = self.to_dataframe(full_result=full_result)
        df.to_csv(filename, float_format=".18e")
    

    def to_file(self, filename):
        # Save the full constraints to a hdf5 file so that
        # the computationally expensive contents can be loaded easily in the future
        # TODO
        pass
    

    def summarize(self, constraint_type="single"):
        # Summarize the results as a dataframe
        # "single" for independent shape analysis, and
        # "joint" for joint (marginalised) shape analysis.

        df = self.to_dataframe(full_result=False)

        if constraint_type == "single":
            df = df[["shape_name", "single_f_NL", "single_sample_sigma"]]
            df["signal_to_noise"] = df["single_f_NL"] / df["single_sample_sigma"]

        elif constraint_type == "joint":
            df = df[["shape_name", "marginal_f_NL", "marginal_sample_sigma"]]
            df["signal_to_noise"] = df["marginal_f_NL"] / df["marginal_sample_sigma"]
        
        return df


    def summarize_latex(self, constraint_type="single", formatter=None):
        # Summarize the results as a latex table
        # "single" for independent shape analysis, and
        # "joint" for joint (marginalised) shape analysis.

        df = self.summarize(constraint_type=constraint_type)

        if formatter is None:
            if constraint_type == "single":
                formatter = lambda row: "${:.1e} \pm {:.1e}$".format(row["single_f_NL"], row["single_sample_sigma"])

            elif constraint_type == "joint":
                formatter = lambda row: "${:.1e} \pm {:.1e}$".format(row["marginal_f_NL"], row["marginal_sample_sigma"])

        df["Shape"] = df["shape_name"]
        df["Constraint"] = df.apply(formatter, axis=1)
        
        if constraint_type == "single":
            df["S/N"] = df["single_f_NL"] / df["single_sample_sigma"]

        elif constraint_type == "joint":
            df["S/N"] = df["marginal_f_NL"] / df["marginal_sample_sigma"]

        df["$f_\mathrm{NL}/\sigma(f_\mathrm{NL})$"] = df.apply(lambda row: "{:.2f}".format(row["S/N"]), axis=1)
        
        df = df[["Shape", "Constraint", "$f_\mathrm{NL}/\sigma(f_\mathrm{NL})$"]]
        
        return df.to_latex(index=False, escape=False)
    

    def shape_correlation_matrix(self):
        # The correlation matrix of the models under study

        fisher = self.fisher_matrix
        fisher_diag = np.diag(fisher)
        corr = fisher / np.sqrt(fisher_diag[:,np.newaxis] * fisher_diag[np.newaxis,:])

        return corr


    def from_file(filename):
        # Load the full constraints from a hdf5 file,
        # saved using the to_file() function
        pass


    def triangle_plot(constraints_list, constraints_labels=None, shape_names=None, shape_labels=None, fig_width_inch=6., plot_kwargs={}):
        # A triangle plot for fNL constraints using fisher error and GetDist 

        import getdist
        from getdist import plots
        from getdist.gaussian_mixtures import GaussianND

        N_c = len(constraints_list)
        model_list = constraints_list[0].model_list
        N_m = len(model_list)

        if constraints_labels is None:
            constraints_labels = [f"#{i}" for i in range(N_c)]
        if shape_names is None:
            shape_names = [m.shape_name for m in model_list]
        if shape_labels is None:
            shape_labels = shape_names

        sub_inds = np.array([i for i in range(N_m) if model_list[i].shape_name in shape_names])

        # Likelihoods
        fisher_list = [GaussianND(c.marginal_f_NL[sub_inds,0],
                                  #np.linalg.inv((c.fisher_matrix)[sub_inds,:][:,sub_inds]),
                                  (c.fisher_matrix)[sub_inds,:][:,sub_inds], is_inv_cov=True,
                                  names=shape_names,
                                  labels=shape_labels,
                                  label=l)
                        for c, l in zip(constraints_list, constraints_labels)]

        g = plots.get_subplot_plotter(width_inch=fig_width_inch)
        g.settings.figure_legend_frame = False
        g.triangle_plot(fisher_list, **plot_kwargs)

        return g


class Model:
    ''' Class for the bispectrum template or model of interest '''

    def __init__(self, shape_type, **kwargs):
        self.shape_type = shape_type.lower()

        if shape_type == "custom_shape_evals":
            # Custom shape function specified by the k grid and evalutations
            self.grid_k_1 = kwargs["grid_k_1"]
            self.grid_k_2 = kwargs["grid_k_2"]
            self.grid_k_3 = kwargs["grid_k_3"]
            self.shape_name = kwargs.get("shape_name", "custom")
            self.shape_function_values = kwargs["shape_function_values"]
            self.shape_function = self.custom_shape_function_from_evals()
        
        elif shape_type == "custom":
            # Custom shape function specified by the given function
            # Takes 'shape_function' defined as
            # S(k_1, k_2, k_3) := (k_1 k_2 k_3)^2 B_\Phi (k_1, k_2, k_3),
            # where <\Phi \Phi \Phi> = B_\Phi(k_1, k_2, k_3) (2\pi)^3 \delta^{(3)}(\mathbf{K})
            self.shape_name = kwargs.get("shape_name", "custom")
            self.shape_function = kwargs["shape_function"]
        
        else:
            # Preset shapes
            preset_shapes_list = ["local", "equilateral", "orthogonal"]

            if shape_type == "local":
                self.shape_name = kwargs.get("shape_name", "local")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)            # k value where P(k) = A_s

                self.shape_function = self.local_shape_function()

            elif shape_type == "equilateral" or shape_type == "equil":
                self.shape_name = kwargs.get("shape_name", "equilateral")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)    # k value where P(k) = A_s

                self.shape_function = self.equilateral_shape_function()
                
            elif shape_type == "orthogonal" or shape_type == "ortho":
                self.shape_name = kwargs.get("shape_name", "orthogonal")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)    # k value where P(k) = A_s

                self.shape_function = self.orthogonal_shape_function()

            else:
                print("Shape type preset '{}' is currently not supported".format(shape_type)) 
                print("Supported shapes:", str(preset_shapes_list))
                return
        

    def custom_shape_function_from_evals(self):
        # Performs a 3D linear interporlation to evaluate the shape function

        interp = RegularGridInterpolator((self.grid_k_1, self.grid_k_2, self.grid_k_3), self.shape_function_values)

        def shape_function(k_1, k_2, k_3):
            ks = np.column_stack([k_1, k_2, k_3])
            return interp(ks)
        
        return shape_function


    def local_shape_function(self):
        # Local template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 2 * (delta_phi ** 2)

            S_1 = k_1 * k_1 * np.power(k_2 * k_3, n_s - 2)
            S_2 = k_2 * k_2 * np.power(k_3 * k_1, n_s - 2)
            S_3 = k_3 * k_3 * np.power(k_1 * k_2, n_s - 2)

            S = pref * (S_1 + S_2 + S_3)
            return S
        
        return shape_function


    def equilateral_shape_function(self):
        # Equilateral template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 6 * (delta_phi ** 2)

            # Precompute different powers of k's: square, linear, constant, inverse
            k_1_sq = k_1 * k_1   
            k_1_li = np.power(k_1, (n_s + 2) / 3)
            k_1_co = np.power(k_1, 2 * (n_s - 1) / 3)
            k_1_in = np.power(k_1, n_s - 2)

            k_2_sq = k_2 * k_2
            k_2_li = np.power(k_2, (n_s + 2) / 3)
            k_2_co = np.power(k_2, 2 * (n_s - 1) / 3)
            k_2_in = np.power(k_2, n_s - 2)

            k_3_sq = k_3 * k_3
            k_3_li = np.power(k_3, (n_s + 2) / 3)
            k_3_co = np.power(k_3, 2 * (n_s - 1) / 3)
            k_3_in = np.power(k_3, n_s - 2)

            S_1 = (k_1_sq * k_2_in * k_3_in
                    + k_2_sq * k_3_in * k_1_in
                    + k_3_sq * k_1_in * k_2_in)

            S_2 = k_1_co * k_2_co * k_3_co

            S_3 = (k_1_in * k_2_co * k_3_li 
                    + k_1_in * k_3_co * k_2_li
                    + k_2_in * k_1_co * k_3_li
                    + k_2_in * k_3_co * k_1_li
                    + k_3_in * k_1_co * k_2_li
                    + k_3_in * k_2_co * k_1_li)

            S = pref * ((-1) * S_1 + (-2) * S_2 + 1 * S_3)
            return S
        
        return shape_function


    def orthogonal_shape_function(self):
        # Orthogonal template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 6 * (delta_phi ** 2)

            # Precompute different powers of k's: square, linear, constant, inverse
            k_1_sq = k_1 * k_1   
            k_1_li = np.power(k_1, (n_s + 2) / 3)
            k_1_co = np.power(k_1, 2 * (n_s - 1) / 3)
            k_1_in = np.power(k_1, n_s - 2)

            k_2_sq = k_2 * k_2
            k_2_li = np.power(k_2, (n_s + 2) / 3)
            k_2_co = np.power(k_2, 2 * (n_s - 1) / 3)
            k_2_in = np.power(k_2, n_s - 2)

            k_3_sq = k_3 * k_3
            k_3_li = np.power(k_3, (n_s + 2) / 3)
            k_3_co = np.power(k_3, 2 * (n_s - 1) / 3)
            k_3_in = np.power(k_3, n_s - 2)

            S_1 = (k_1_sq * k_2_in * k_3_in
                    + k_2_sq * k_3_in * k_1_in
                    + k_3_sq * k_1_in * k_2_in)

            S_2 = k_1_co * k_2_co * k_3_co

            S_3 = (k_1_in * k_2_co * k_3_li 
                    + k_1_in * k_3_co * k_2_li
                    + k_2_in * k_1_co * k_3_li
                    + k_2_in * k_3_co * k_1_li
                    + k_3_in * k_1_co * k_2_li
                    + k_3_in * k_2_co * k_1_li)

            S = pref * ((-3) * S_1 + (-8) * S_2 + 3 * S_3)
            return S
        
        return shape_function
