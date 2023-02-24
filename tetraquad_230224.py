''' tetraquad.py
Module for computing quadrature rules for numerical integration
over a "tetrapyd" volume in three dimensions,
which consists of points (k1, k2, k3) satisfying:
k_min <= k1, k2, k3 <= k_max
k1 + k2 >= k3
k2 + k3 >= k1
k3 + k1 >= k2
'''

import numpy as np
from scipy.special import gamma, beta, betainc, hyp2f1
from numpy.random import default_rng
import scipy
import cvxopt
import pandas as pd
import mpmath


def quad(func, k_min=None, k_max=None, N=10, grid=None, weights=None):
    ''' Integrate the given function over a tetrapyd volume.
    May provide a precomputed quadrature rule (grid, weights) to
    speed up the calculations.
    '''

    if grid is None or weights is None:
        grid, weights = quadrature(k_min, k_max, N)

    k1, k2, k3 = grid
    integrand = func(k1, k2, k3)
    eval = np.dot(weights, integrand)

    return eval


def quadrature(k_min, k_max, N, grid_type="Uniform", include_endpoints=True):
    ''' Returns a quadrature rule that guarantees the integral of
    symmetric polynomials of order <= N over a tetrapyd is almost exact.
    '''

    alpha = k_min / k_max
    grid, weights = unit_quadrature_nnls(alpha, N, grid_type, include_endpoints)
    grid *= k_max
    weights *= k_max ** 3

    return grid, weights


def save_quadrature(save_path, k_min, k_max, N, grid_type="Uniform", include_endpoints=True, negative_power=None):
    ''' Computes and saves tetraquad rule in a Dataframe format
    '''

    alpha = k_min / k_max
    grid, weights, i1, i2, i3, grid_1d = unit_quadrature_nnls(alpha, N, grid_type, include_endpoints, get_grid_indices=True, negative_power=negative_power, save_path=save_path)
    grid *= k_max
    weights *= k_max ** 3

    df = pd.DataFrame({"i1": i1, "i2": i2, "i3": i3, "k1": grid[0], "k2": grid[1], "k3": grid[2],
                            "weight": weights})

    df.to_csv(save_path+".csv", float_format="%.18e")


def load_quadrature(filename):
    ''' Loads tetraquad rule from a csv file
    '''

    df = pd.read_csv(filename)
    grid = np.stack([df["k1"], df["k2"], df["k3"]], axis=0)
    weights = df["weight"].to_numpy()

    return grid, weights


def unit_quadrature_nnls(alpha, N, grid_type="Uniform", include_endpoints=True, get_grid_indices=False, negative_power=None, save_path=None):
    ''' Returns a quadrature rule which has N grid points on each dimension.
    Minimises the integration error of symmetric polynomials 
    order <= f(N) over a tetrapyd specified by the triangle conditions and
    alpha <= k1, k2, k3 <= 1
    If negative_power is not None, additionally include symmetric polynomials with one power equal to that value.
    E.g. negative_power = ns - 2
    '''

    # Set up grid points
    if grid_type == "Uniform":
        #i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints)
        i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints, keep_borderline=False)
    elif grid_type == "GL":
        i1, i2, i3, grid_1d = gl_tetra_triplets(alpha, N)
    else:
        print("Grid name {} currently unsupported.".format(grid_type))

    if save_path is not None:
        np.savetxt(save_path+"_k_grid.txt", grid_1d)

    num_weights = i1.shape[0]
    grid = np.array([grid_1d[i1], grid_1d[i2], grid_1d[i3]])

    # Prepare orthogonal polynomials
    M = 2 * N
    '''
    while True:
        # List of polynomial orders (p,q,r)
        ps, qs, rs = poly_triplets_total_degree(M, negative_power)
        #ps, qs, rs = poly_triplets_individual_degree(M)
        #ps, qs, rs = poly_triplets_total_degree_ns(M)
        num_polys = ps.shape[0]

        if num_polys > num_weights:
            M -= 1
            ps, qs, rs = poly_triplets_total_degree(M, negative_power)
            #ps, qs, rs = poly_triplets_individual_degree(M)
            #ps, qs, rs = poly_triplets_total_degree_ns(M)
            num_polys = ps.shape[0]
            break
        else:
            M += 1
    '''
    # TEST!!
    ps, qs, rs = poly_triplets_total_degree(M, negative_power)
    num_polys = ps.shape[0]

    print("M =", M, ", N =", N)

    # Obtain orthonormalisation coefficients for the polynomials
    ortho_L = orthonormal_polynomials(ps, qs, rs, alpha)
    if save_path is not None:
        np.save(save_path+"_ortho_L.npy", ortho_L)

    poly_ampl = np.copy(np.diag(ortho_L))
    ortho_L /= np.sqrt(poly_ampl)[:,np.newaxis]

    # Evaluations of the polynomials at grid points
    grid_evals = grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3)
    grid_evals = np.matmul(ortho_L, grid_evals)

    # We are now ready to compute the quadrature weights.
    # Non-Negative Least Squares:
    # minimise ||A x - b||^2 for non-negative x

    print(num_polys, num_weights)

    A = grid_evals
    tetra_volume = 0.5 - 3 * alpha ** 2 + 3 * alpha ** 3
    b = np.concatenate([[ortho_L[0,0]*tetra_volume], np.zeros(num_polys-1)])

    x, rnorm = scipy.optimize.nnls(A, b)
    print("NNLS complete, rnorm {}".format(rnorm))
    weights = x.flatten()

    #nonzero_weights = (weights > 1e-6 * (tetra_volume / num_weights))
    nonzero_weights = np.nonzero(weights)[0]
    print("Out of {} weights, {} of them are nonzero".format(len(weights), len(nonzero_weights)))
    grid = grid[:, nonzero_weights]
    weights = weights[nonzero_weights]

    if get_grid_indices:
        i1, i2, i3 = i1[nonzero_weights], i2[nonzero_weights], i3[nonzero_weights]
        return grid, weights, i1, i2, i3, grid_1d
    else:
        return grid, weights



def unit_quadrature_qp(alpha, N, grid_type="Uniform", include_endpoints=True):
    ''' Returns a quadrature rule which has N grid points on each dimension.
    This should guarantee that the integral of
    symmetric polynomials of order <= f(N) over a tetrapyd is exact.
    Tetrapyd specified by the triangle conditions and
    alpha <= k1, k2, k3 <= 1
    '''

    # Set up grid points
    if grid_type == "Uniform":
        i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints)
    elif grid_type == "GL":
        i1, i2, i3, grid_1d = gl_tetra_triplets(alpha, N)
    else:
        print("Grid name {} currently unsupported.".format(grid_type))

    num_weights = i1.shape[0]
    grid = np.array([grid_1d[i1], grid_1d[i2], grid_1d[i3]])

    # Prepare orthogonal polynomials
    M = N // 2
    while True:
        # List of polynomial orders (p,q,r)
        ps, qs, rs = poly_triplets_total_degree(M)
        #ps, qs, rs = poly_triplets_individual_degree(M)
        num_polys = ps.shape[0]

        if num_polys > num_weights:
            M -= 1
            ps, qs, rs = poly_triplets_total_degree(M)
            #ps, qs, rs = poly_triplets_individual_degree(M)
            num_polys = ps.shape[0]
            break
        else:
            M += 1

    # Obtain orthonormalisation coefficients for the polynomials
    ortho_L = orthonormal_polynomials(ps, qs, rs, alpha)
    poly_ampl = np.copy(np.diag(ortho_L))
    ortho_L /= np.sqrt(poly_ampl)[:,np.newaxis]

    # Evaluations of the polynomials at grid points
    grid_evals = grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3)
    grid_evals = np.matmul(ortho_L, grid_evals)

    # Base value for the weights
    tetra_volume = (1 - alpha ** 3) / 2.
    tetra_volume = 0.5 - 3 * alpha ** 2 + 3 * alpha ** 3
    base_weights = np.ones(num_weights) * tetra_volume / num_weights
    
    # Analytic values for the integrals
    analytic = analytic_poly_integrals_alpha(ps, qs, rs, alpha)
    analytic = np.matmul(ortho_L, analytic)

    num_constraints = np.linalg.matrix_rank(grid_evals)


    # We are now ready to compute the quadrature weights.
    # Quadratic Programming:
    # minimise  (1/2) x^T P x + q^T x  subject to Gx <= h  and  Ax = b

    n_c = num_constraints
    print(len(ps), num_constraints, len(base_weights))

    #'''
    P = np.matmul(grid_evals[n_c:,:].T, grid_evals[n_c:,:])
    #P += np.max(np.abs(P)) * 1e-10 * np.identity(P.shape[0])
    qp_P = cvxopt.matrix(P)
    qp_q = cvxopt.matrix(-np.matmul(grid_evals[n_c:,:].T, analytic[n_c:]))
    qp_G = cvxopt.matrix(-np.identity(len(base_weights)))
    qp_h = cvxopt.matrix(np.zeros_like(base_weights))
    #qp_A = cvxopt.matrix(grid_evals[1:n_c,:])
    #qp_b = cvxopt.matrix(np.zeros(n_c-1))
    qp_A = cvxopt.matrix(grid_evals[:n_c,:])
    qp_b = cvxopt.matrix(np.concatenate([[1], np.zeros(n_c-1)]))
    qp_i = cvxopt.matrix(base_weights)
    #qp_i = cvxopt.matrix(np.ones_like(base_weights)/len(base_weights))

    qp_result = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b, initvals=qp_i)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q, A=qp_A, b=qp_b, initvals=qp_i)
    #'''

    '''
    qp_P = cvxopt.matrix(np.matmul(grid_evals[1:,:].T, grid_evals[1:,:]))
    qp_q = cvxopt.matrix(-np.matmul(grid_evals[1:,:].T, analytic[1:]))
    '''

    #qp_result = cvxopt.solvers.qp(qp_P, qp_q,initvals=qp_i)
    #qp_result = cvxopt.solvers.qp(qp_P, qp_q)

    weights = np.array(qp_result["x"])

    return grid, weights


def uniform_tetrapyd_weights(alpha, N, MC_N_SAMPLES=5000, include_endpoints=True):
    ''' Returns a uniform tetrapyd grid with weights proportional to 
    the volume of the grid point (voxel).
    '''

    # Set up grid points
    i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N, include_endpoints)
    num_weights = i1.shape[0]

    grid = np.zeros((3, num_weights))
    grid[0,:] = grid_1d[i1]
    grid[1,:] = grid_1d[i2]
    grid[2,:] = grid_1d[i3]

    # 1D bounds and weights for k grid points
    interval_bounds = np.zeros(N + 1)
    interval_bounds[0] = alpha
    interval_bounds[1:-1] = (grid_1d[:-1] + grid_1d[1:]) / 2
    interval_bounds[-1] = 1
    k_weights = np.diff(interval_bounds)

    # Initialise weights based on symmetry and grid intervals
    tetrapyd_weights = k_weights[i1] * k_weights[i2] * k_weights[i3]
    tetrapyd_weights[(i1 != i2) & (i2 != i3)] *= 6   # Distinct indices
    tetrapyd_weights[(i1 != i2) & (i2 == i3)] *= 3   # Two identical indices
    tetrapyd_weights[(i1 == i2) & (i2 != i3)] *= 3   # Two identical indices

    # Further weights for points of the surface of the tetrapyd
    lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
    lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
    lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
    need_MC = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
    MC_count = np.sum(need_MC)

    # Draw samples uniformly within each cubic cell
    rng = default_rng(seed=0)
    r1 = rng.uniform(lb1[need_MC], ub1[need_MC], size=(MC_N_SAMPLES, MC_count))
    r2 = rng.uniform(lb2[need_MC], ub2[need_MC], size=(MC_N_SAMPLES, MC_count))
    r3 = rng.uniform(lb3[need_MC], ub3[need_MC], size=(MC_N_SAMPLES, MC_count))

    # Estimate the fraction of samples that lie inside the tetrapyd
    in_tetra = ((r1 + r2 >= r3) & (r2 + r3 >= r1) & (r3 + r1 >= r2))
    MC_weights = np.sum(in_tetra, axis=0) / MC_N_SAMPLES
    
    tetrapyd_weights[need_MC] *= MC_weights

    # Make sure the overall volume is exact
    #tetra_volume = 0.5 - 3 * alpha ** 2 + 3 * alpha ** 3
    #tetrapyd_weights *= tetra_volume / np.sum(tetrapyd_weights)
    
    return grid, tetrapyd_weights


def save_uniform_quadrature(filename, k_min, k_max, N, MC_N_SAMPLES=1000000):
    ''' Computes and saves the uniform quadrature where the weights are proportinal
    to the grid (voxel) volume within tetrapyd.
    '''

    alpha = k_min / k_max
    grid, weights = uniform_tetrapyd_weights(alpha, N, MC_N_SAMPLES)
    grid *= k_max
    weights *= k_max ** 3

    i1, i2, i3, grid_1d = uni_tetra_triplets(alpha, N)

    df = pd.DataFrame({"i1": i1, "i2": i2, "i3": i3, "k1": grid[0], "k2": grid[1], "k3": grid[2],
                            "weight": weights})

    df.to_csv(filename, float_format="%.18e")


def poly_pairs_inidividual_degree(N):
    ''' List of pairs (p,q) such taht
    N >= p >= q >= 0
    '''
    pairs = [[p,q] for p in range(N+1)
                for q in range(p+1)]
    
    return np.array(pairs).T


def poly_pairs_total_degree(N):
    ''' List of pairs (p,q) such taht
    N >= p >= q >= 0 and p + q <= N
    '''
    pairs = [[p, n-p] for n in range(N+1)
                for p in range(n, (n-1)//2, -1)]
    
    return np.array(pairs).T


def poly_triplets_individual_degree(N, negative_power=None):
    ''' List of triplets (p,q,r) such that
    N >= p >= q >= r >= 0 
    If negative_power is not None,
    p, q, r are drawn from {negative_power, 0, 1, 2, ..., N-1}, 
    and p >= q >= r.
    '''

    if negative_power is None:
        tuples = [[p, q, r] for p in range(N+1)
                    for q in range(p+1)
                        for r in range(q+1)]
        
        return np.array(tuples).T

    else:
        res = poly_triplets_individual_degree(N, negative_power=None)
        res = res - 1.
        res[res < -0.5] = negative_power

        return res


def poly_triplets_individual_degree_next_order(N):
    ''' List of triplets (p,q,r) such that
    N+1 = p >= q >= r >= 0 
    '''
    tuples = [[N+1, q, r] for q in range(N+2)
                    for r in range(q+1)]
    
    return np.array(tuples).T


def poly_triplets_total_degree(N, negative_power=None):
    ''' List of triplets (p,q,r) such that
    p >= q >= r >= 0 and  p + q + r <= N
    If negative_power is not None, additionally include (p,q,r)s with
    p >= q >= 0, r = negative_power, and p + q <= N.
    '''

    if negative_power is None:
        # Increasing p, q within each n = p + q + r 
        #tuples = [[p, q, n-p-q] for n in range(N+1)
        #                 for p in range((n+2)//3, n+1)
        #                    for q in range((n-p+1)//2, min(p+1, n-p+1))]

        # Decreasing p, q within each n = p + q + r
        tuples = [[p, q, n-p-q] for n in range(N+1)
                        for p in range(n, (n+2)//3-1, -1)
                            for q in range(min(p, n-p), (n-p+1)//2-1, -1)]

    else: 
        tuples = [[0, 0, 0]]    # (0,0,0) always comes first
        tuples = tuples + [[p, n-p, negative_power] for n in range(N+1)
                                        for p in range(n, (n-1)//2, -1)]
        tuples = tuples + [[p, q, n-p-q] for n in range(1, N+1)
                                for p in range(n, (n+2)//3-1, -1)
                                    for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def poly_triplets_total_degree_next_order(N):
    ''' List of triplets (p,q,r) such that
    p >= q >= r >= 0 and  p + q + r = N+1
    '''
    n = N + 1
    # Increasing p, q 
    # tuples = [[p, q, n-p-q] for p in range((n+2)//3, n+1)
    #                    for q in range((n-p+1)//2, min(p+1, n-p+1))]

    # Deacreasing p, q 
    tuples = [[p, q, n-p-q] for p in range(n, (n+2)//3-1, -1)
                        for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def poly_triplets_total_degree_ns(N, ns=0.9660499):
    ''' List of triplets (p,q,r) such that
    (p >= q >= r >= 0 and  p + q + r <= N) OR
    (p >= q >= 0 and r = ns-2 and p + q <= N).
    '''

    tuples = [[0, 0, 0]]

    # r = ns-2
    ps, qs = poly_pairs_total_degree(N)
    tuples = tuples + [[p, q, ns-2] for p, q in zip(ps, qs) if p + q >= 2] 

    # Decreasing p, q within each n = p + q + r
    tuples = tuples + [[p, q, n-p-q] for n in range(1, N+1)
                        for p in range(n, (n+2)//3-1, -1)
                            for q in range(min(p, n-p), (n-p+1)//2-1, -1)]
    
    return np.array(tuples).T


def gl_quadrature(alpha, N):
    # Gauss-Legendre quadrature on the interval [alpha, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(N)
    k_grid = alpha + (1 - alpha) / 2 * (gl_nodes + 1)
    k_weights = (1 - alpha) / 2 * gl_weights

    return k_grid, k_weights


def uni_tetra_triplets(alpha, N, include_endpoints=True, keep_borderline=True):
    ''' Indices for a uniform grid inside the tetrapyd
    satisfying 1 >= k1 >= k2 >= k3 >= alpha and k2 + k3 >= k1.
    When keep_borderline is True, keep all the grid points that lie outside
    the tetrapyd but has their voxel overlap with it.
    '''

    if include_endpoints:
        k_grid = np.linspace(alpha, 1, N)
    else:
        dk = (1 - alpha) / N
        k_grid = np.linspace(alpha+dk/2, 1-dk/2, N)
    

    if not keep_borderline:
        # The following can miss out some volume elements
        tuples = [[i1, i2, i3] for i1 in range(N)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)
                            if k_grid[i2] + k_grid[i3] >= k_grid[i1]]
        i1, i2, i3 = np.array(tuples).T
    
    else:
        # 1D bounds for k grid points
        interval_bounds = np.zeros(N + 1)
        interval_bounds[0] = alpha
        interval_bounds[1:-1] = (k_grid[:-1] + k_grid[1:]) / 2
        interval_bounds[-1] = 1

        tuples = [[i1, i2, i3] for i1 in range(N)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)]
        i1, i2, i3 = np.array(tuples).T

        # Corners specifying the grid volume (voxel)
        lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
        lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
        lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
        inside = (k_grid[i2] + k_grid[i3] >= k_grid[i1])
        borderline = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                    | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                    | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
        keep = (inside | borderline)

        i1, i2, i3 = i1[keep], i2[keep], i3[keep]
    
    return i1, i2, i3, k_grid


def gl_tetra_triplets(alpha, N):
    ''' Indices for a grid inside the tetrapyd where the grid points
    come from (one-dimensional) Gauss-Legendre quadrature points.
    Satisfies 1 >= k1 >= k2 >= k3 >= alpha and k2 + k3 >= k1.
    '''

    k_grid, k_weights = gl_quadrature(alpha, N)

    tuples = [[i1, i2, i3] for i1 in range(N)
                for i2 in range(i1+1)
                    for i3 in range(i2+1)
                        if k_grid[i2] + k_grid[i3] >= k_grid[i1]]
    i1, i2, i3 = np.array(tuples).T
    
    return i1, i2, i3, k_grid


def incomplete_beta(x, a, b):
    ''' The incomplete beta function B(x; a, b) as defined in https://dlmf.nist.gov/8.17
    Inputs a and b can be array-like, whereas x needs be a scalar.
    '''
    if np.abs(x) <= 1:
        # Used scipy's regularised incomplete beta function 'betainc'
        return beta(a, b) * betainc(a, b, x)
    else:
        # Use hyptergeometric function to compute the analytically-continued beta function
        # See Equation (8.17.7) of https://dlmf.nist.gov/8.17
        #return (x ** a) / a * hyp2f1(a, 1-b, a+1, x)
        #print("F({},{};{};{}) = {}".format(a+b, 1, a+1, x, hyp2f1(a+b,1,a+1,x)))
        
        #return ((0.+x) ** a) * ((1.-x) ** b) / a * hyp2f1(a+b, 1., a+1., x)
        return np.power(x, a) * np.power(1.-x, b) / a * hyp2f1(a+b, 1., a+1., x)



def generalised_incomplete_beta(x1, x2, a, b):
    ''' The generalised incomplete beta function 
    B(x1, x2; a, b) = B(x2; a, b) - B(x1; a, b)
    '''
    a, b = np.array(a), np.array(b)
    if x1 == 2:
        x1 = x1 + 1e-10     # Ad-hoc, avoid badness at x1=b=2

    int_ab = (a % 1 == 0) & (b % 1 == 0)
    if np.sum(int_ab) == a.size:
        # Powers a, b are integers
        return incomplete_beta(x2, a, b) - incomplete_beta(x1, a, b)
    else:
        result = np.zeros(a.size, dtype=complex) 

        # Normal method for integer powers
        if np.sum(int_ab) > 0:
            result[int_ab] = incomplete_beta(x2, a[int_ab], b[int_ab]) - incomplete_beta(x1, a[int_ab], b[int_ab])

        # Use mpmath for non-integer powers
        nint_ab = np.invert(int_ab)
        result[nint_ab] = np.array([complex(mpmath.betainc(av, bv, x1, x2)) for av, bv in zip(a[nint_ab], b[nint_ab])])

        return result

    #b1 = incomplete_beta(x1, a, b)
    #b2 = incomplete_beta(x2, a, b)
    #print("B({};{},{}) = {}".format(x1, a, b, b1))
    #print("B({};{},{}) = {}".format(x2, a, b, b2))
    #return b2-b1


def analytic_poly_integrals(ps, qs, rs):
    ''' Analytic values for the integrals of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    (or equivalently, x^p * y^q * z^r)
    over the tetrapyd with alpha = 0 (bounded below by zero).
    '''

    evals = 1 / ((1 + ps) * (1 + qs) * (1 + rs))
    evals -= gamma(1 + qs) * gamma(1 + rs) / ((3 + ps + qs + rs) * gamma(3 + qs + rs))
    evals -= gamma(1 + rs) * gamma(1 + ps) / ((3 + ps + qs + rs) * gamma(3 + rs + ps))
    evals -= gamma(1 + ps) * gamma(1 + qs) / ((3 + ps + qs + rs) * gamma(3 + ps + qs))

    return evals


def analytic_poly_integrals_alpha(ps, qs, rs, alpha):
    ''' Analytic values for the integrals of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    (or equivalently, x^p * y^q * z^r)
    over the tetrapyd, bounded below by alpha, assumed to be less than 1/2.
    See Wuhyun's notes for the derivation of this formula.
    '''

    if alpha == 0:
        return analytic_poly_integrals(ps, qs, rs)

    a = alpha
    b = 1 - a

    evals = np.zeros(len(ps))

    # Prevent overflow errors from large p+q+r
    #ignore_alpha = (alpha ** (ps+qs+rs) < 1e-200)
    ignore_alpha = ((ps+qs+rs) > np.log(1e-200) / np.log(alpha))
    evals[ignore_alpha] = analytic_poly_integrals(ps[ignore_alpha], qs[ignore_alpha], rs[ignore_alpha])

    do_alpha = np.invert(ignore_alpha)
    p, q, r = ps[do_alpha], qs[do_alpha], rs[do_alpha]

    evals_alpha = np.zeros(len(do_alpha))

    # alpha^(p+1)
    ap1, aq1, ar1 = a ** (p+1), a ** (q+1), a ** (r+1)
    # beta^(p+1)
    bp1, bq1, br1 = b ** (p+1), b ** (q+1), b ** (r+1)

    # B(x1, x2; a, b)
    B = generalised_incomplete_beta

    # Integral over the cube [alpha,1]^3
    I_cube = (1-ap1) * (1-aq1) * (1-ar1) / ((p+1.) * (q+1.) * (r+1.))

    # Integral over the volume inside the cube but outside the tetrapyd,
    # where x >= y+z
    I_x = B(a, b, q+1, r+1) / (p+q+r+3.) / (q+r+2.)
    I_x -= (((r+1.) * ar1 * bq1 / (q+r+2.)) + ((q+1.) * aq1 * br1 / (q+r+2.)) - aq1 * ar1) / ((p+1.) * (q+1.) * (r+1.))
    #I_x += a**(p+q+r+3) * ((-1)**q * B(2, 1./a, p+2, q+1) + (-1)**r * B(2, 1./a, p+2, r+1)) / ((p+1.) * (p+q+r+3.))
    I_x += a**(p+q+r+3) * np.real((-1.+0j)**q * B(2, 1/a, p+2, q+1) + (-1.+0j)**r * B(2, 1/a, p+2, r+1)) / ((p+1.) * (p+q+r+3.))

    # Same for y >= z+x
    I_y = B(a, b, r+1, p+1) / (p+q+r+3.) / (r+p+2.)
    I_y -= (((p+1.) * ap1 * br1 / (r+p+2.)) + ((r+1.) * ar1 * bp1 / (r+p+2.)) - ar1 * ap1) / ((p+1.) * (q+1.) * (r+1.))
    #I_y += a**(p+q+r+3) * ((-1)**r * B(2, 1./a, q+2, r+1) + (-1)**p * B(2, 1./a, q+2, p+1)) / ((q+1.) * (p+q+r+3.))
    I_y += a**(p+q+r+3) * np.real((-1.+0j)**r * B(2, 1/a, q+2, r+1) + (-1.+0j)**p * B(2, 1/a, q+2, p+1)) / ((q+1.) * (p+q+r+3.))

    # Same for z >= x_y
    I_z = B(a, b, p+1, q+1) / (p+q+r+3.) / (p+q+2.)
    I_z -= (((q+1.) * aq1 * bp1 / (p+q+2.)) + ((p+1.) * ap1 * bq1 / (p+q+2.)) - ap1 * aq1) / ((p+1.) * (q+1.) * (r+1.))
    #I_z += a**(p+q+r+3) * ((-1)**p * B(2, 1./a, r+2, p+1) + (-1)**q * B(2, 1./a, r+2, q+1)) / ((r+1.) * (p+q+r+3.))
    I_z += a**(p+q+r+3) * np.real((-1.+0j)**p * B(2, 1/a, r+2, p+1) + (-1.+0j)**q * B(2, 1/a, r+2, q+1)) / ((r+1.) * (p+q+r+3.))

    evals_alpha = I_cube - I_x - I_y - I_z

    evals[do_alpha] = np.real(evals_alpha[:])

    return evals


def analytic_poly_cross_product_alpha(ps, qs, rs, alpha):
    ''' Returns a matrix containing cross inner products of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    over the tetrapyd domain bounded below by alpha (<0.5).
    Optimised for this special purpose.
    '''

    # Arrays need be int for indexing purposes
    ps = np.round(ps).astype(int)
    qs = np.round(qs).astype(int)
    rs = np.round(rs).astype(int)

    # Shorthand definitions
    a = alpha
    b = 1 - a
    p_max = np.max(ps)
    max_power = 2 * p_max + 2

    # Precompute B(a, b; p, q) for 1 <= p, q <= max_power
    # Since a + b = 1, B(a, b; p, q) is symmteric in p and q
    pre_beta_a_b = np.zeros((max_power+1, max_power+1))
    for p in range(1, max_power+1):
        # Scipy's normalised incomplete beta function
        q = np.arange(1, p+1)
        pre_beta_a_b[p,1:p+1] = beta(p, q) * (betainc(p, q, b) - betainc(p, q, a))
    # Symmetrise matrix
    pre_beta_a_b += pre_beta_a_b.T - np.diag(np.diag(pre_beta_a_b))

    # Precompute (a**(p+q) * (-1)**q) * B(2, 1/a; p, q)
    cut = int(np.ceil(np.log(1e-200) / np.log(a)))   # Ignore this term for p+q > cut
    pre_beta_2_ainv = np.zeros((max_power+1, max_power+1))
    for p in range(1, min(max_power+1, cut)):
        f2, fainv = np.power(2.*a, p) / p, 1. / p
        #B(x;p,q) = np.power(x, p) * np.power(1.-x, q) / p * hyp2f1(p+q, 1., p+1., x)
        # Scipy's hypergeometric function
        q = np.arange(1, min(max_power+1, cut+1-p))
        pre_beta_2_ainv[p,q] = (fainv * np.power(1.-a, q) * hyp2f1(p+q, 1, p+1, 1/a)
                                    - f2 * np.power(a, q) * hyp2f1(p+q, 1, p+1, 2))
        '''
        mpmath.mp.dps = 30
        for q in range(1, max_power+1):
            print("!")
            pre_beta_2_ainv[p,q] = (fainv * np.power(1.-a, q) * float(mpmath.hyp2f1(p+q, 1, p+1, 1/a).real)
                                        - f2 * np.power(a, q) * float(mpmath.hyp2f1(p+q, 1, p+1, 2)).real)
            # Note that F(a,b;c;z) = (1-z)**(-a) F(a,c-b;c;z/(z-1))
            #pre_beta_2_ainv[p,q] = (a ** (p+q)) * ((-1) ** q) * float(mpmath.betainc(p, q, x1=2, x2=1/a).real)
        '''

    #print("B(a,b)=", pre_beta_a_b)
    #print("B(2,1/a)=", pre_beta_2_ainv)

    print("Betas precomputed")

    num_polys = len(ps)
    cross_prod = np.zeros((num_polys, num_polys))

    for n in range(num_polys):
        p1, q1, r1 = ps[n], qs[n], rs[n]
        p2, q2, r2 = ps[:n+1], qs[:n+1], rs[:n+1]
        perms = [(p1+p2, q1+q2, r1+r2), (p1+p2, q1+r2, r1+q2),
                 (p1+q2, q1+p2, r1+r2), (p1+q2, q1+r2, r1+p2),
                 (p1+r2, q1+p2, r1+q2), (p1+r2, q1+q2, r1+p2)]

        for p, q, r in perms:
            # alpha^(p+1)
            ap1, aq1, ar1 = a ** (p+1), a ** (q+1), a ** (r+1)
            # beta^(p+1)
            bp1, bq1, br1 = b ** (p+1), b ** (q+1), b ** (r+1)

            # Integral over the cube [alpha,1]^3
            I_cube = (1-ap1) * (1-aq1) * (1-ar1) / ((p+1.) * (q+1.) * (r+1.))

            # Integral over the volume inside the cube but outside the tetrapyd,
            # where x >= y+z
            I_x = pre_beta_a_b[q+1,r+1] / (p+q+r+3.) / (q+r+2.)
            I_x -= (((r+1.) * ar1 * bq1 / (q+r+2.)) + ((q+1.) * aq1 * br1 / (q+r+2.)) - aq1 * ar1) / ((p+1.) * (q+1.) * (r+1.))
            #I_x += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[p+2,q+1] + (-1) * pre_beta_2_ainv[p+2,r+1]) / ((p+1.) * (p+q+r+3.))
            I_x -= (ar1 * pre_beta_2_ainv[p+2,q+1] + aq1 * pre_beta_2_ainv[p+2,r+1]) / (a * (p+1.) * (p+q+r+3.))

            # Same for y >= z+x
            I_y = pre_beta_a_b[r+1,p+1] / (p+q+r+3.) / (r+p+2.)
            I_y -= (((p+1.) * ap1 * br1 / (r+p+2.)) + ((r+1.) * ar1 * bp1 / (r+p+2.)) - ar1 * ap1) / ((p+1.) * (q+1.) * (r+1.))
            #I_y += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[q+2,r+1] + (-1) * pre_beta_2_ainv[q+2,p+1]) / ((q+1.) * (p+q+r+3.))
            I_y -= (ap1 * pre_beta_2_ainv[q+2,r+1] + ar1 * pre_beta_2_ainv[q+2,p+1]) / (a * (q+1.) * (p+q+r+3.))

            # Same for z >= x_y
            I_z = pre_beta_a_b[p+1,q+1] / (p+q+r+3.) / (p+q+2.)
            I_z -= (((q+1.) * aq1 * bp1 / (p+q+2.)) + ((p+1.) * ap1 * bq1 / (p+q+2.)) - ap1 * aq1) / ((p+1.) * (q+1.) * (r+1.))
            #I_z += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[r+2,p+1] + (-1) * pre_beta_2_ainv[r+2,q+1]) / ((r+1.) * (p+q+r+3.))
            I_z -= (aq1 * pre_beta_2_ainv[r+2,p+1] + ap1 * pre_beta_2_ainv[r+2,q+1]) / (a * (r+1.) * (p+q+r+3.))

            cross_prod[n,:n+1] += (I_cube - I_x - I_y - I_z) / 6

    # Symmetrise
    cross_prod += cross_prod.T - np.diag(np.diag(cross_prod))

    return cross_prod


def analytic_poly_cross_product_alpha_ns(ps, qs, rs, alpha, negative_power):
    ''' Returns a matrix containing cross inner products of
    (x^p * y^q + z^r  + (5 syms)) / 6.0
    over the tetrapyd domain bounded below by alpha (<0.5).
    ps, qs, rs are either non-negative integers or 'negative_power', often ns-2.
    Optimised for this special purpose.
    '''

    p_max = int(np.round(np.max(ps)))
    max_power = 2 * p_max + 2
    M = max_power + 1
    mpmath.mp.dps = 15

    # For the convenience of coding, we replace all negative powers with
    # a sufficiently large integer M.
    # Arrays need be int for indexing purposes
    def replace_negative(arr, new_value):
        replaced = np.copy(arr)
        replaced[replaced < 0] = new_value
        return np.round(replaced).astype(int)

    pinds = replace_negative(ps, M)
    qinds = replace_negative(qs, M)
    rinds = replace_negative(rs, M)

    # Shorthand definitions
    a = alpha
    b = 1 - alpha
    v = negative_power

    # Precompute B(a, b; p, q) for 1 <= p, q <= max_power
    # Since a + b = 1, B(a, b; p, q) is symmteric in p and q
    pre_beta_a_b = np.zeros((2 * M + 3, 2 * M + 3))
    for p in range(1, max_power+1):
        # Scipy's normalised incomplete beta function
        q = np.arange(1, p+1)
        pre_beta_a_b[p,1:p+1] = beta(p, q) * (betainc(p, q, b) - betainc(p, q, a))

    # Symmetrise matrix
    pre_beta_a_b += pre_beta_a_b.T - np.diag(np.diag(pre_beta_a_b))

    # Precompute betas with negative powers in 
    for p in range(1, max_power+1):
        for q in range(1, max_power+1):
            # B(a, b; v+p, q)
            pre_beta_a_b[M+p,q] = float(mpmath.betainc(v+p, q, x1=a, x2=b).real)
            # B(a, b; p, v+q)
            pre_beta_a_b[p,M+q] = float(mpmath.betainc(p, v+q, x1=a, x2=b).real)
            # B(a, b; v+p, v+q)
            pre_beta_a_b[M+p,M+q] = float(mpmath.betainc(v+p, v+q, x1=a, x2=b).real)
        for q in range(1, 3):
            # B(a, b; p, 2*v+q)
            pre_beta_a_b[p,2*M+q] = float(mpmath.betainc(p, 2*v+q, x1=a, x2=b).real)
            # B(a, b; v+p, 2*v+q)
            pre_beta_a_b[M+p,2*M+q] = float(mpmath.betainc(v+p, 2*v+q, x1=a, x2=b).real)
    for p in range(1, 3):
        for q in range(1, max_power+1):
            # B(a, b; 2*v+p, q)
            pre_beta_a_b[2*M+p,q] = float(mpmath.betainc(2*v+p, q, x1=a, x2=b).real)
            # B(a, b; 2*v+p, v+q)
            pre_beta_a_b[2*M+p,M+q] = float(mpmath.betainc(2*v+p, v+q, x1=a, x2=b).real)
        for q in range(1, 3):
            # B(a, b; 2*v+p, 2*v+q)
            pre_beta_a_b[2*M+p,2*M+q] = float(mpmath.betainc(2*v+p, 2*v+q, x1=a, x2=b).real)
    
    # Precompute (a**(p+q) * (-1)**q) * B(2, 1/a; p, q)
    cut = int(np.ceil(np.log(1e-200) / np.log(a)))   # Ignore this term for p+q > cut
    pre_beta_2_ainv = np.zeros((2 * M + 3, 2 * M + 3))
    '''
    for p in range(1, min(max_power+1, cut)):
        f2, fainv = np.power(2.*a, p) / p, 1. / p
        #B(x;p,q) = np.power(x, p) * np.power(1.-x, q) / p * hyp2f1(p+q, 1., p+1., x)
        # Scipy's hypergeometric function
        q = np.arange(1, min(max_power+1, cut+1-p))
        pre_beta_2_ainv[p,q] = (fainv * np.power(1.-a, q) * hyp2f1(p+q, 1, p+1, 1/a)
                                    - f2 * np.power(a, q) * hyp2f1(p+q, 1, p+1, 2))
    '''

    def beta_2_ainv(p, q):
        f2, fainv = np.power(2.*a, p) / p, 1. / p
        return float((fainv * np.power(b, q) * mpmath.hyp2f1(p+q, 1, p+1, 1/a)
                            - f2 * np.power(a, q) * mpmath.hyp2f1(p+q, 1, p+1, 2)).real)

    # Same betas with negative powers in
    for p in range(1, min(max_power+1, cut)):
        for q in range(1, min(max_power+1, cut+1-p)):
            # B(a, b; p, q)
            #pre_beta_2_ainv[p,q] = beta_2_ainv(p, q)
            f2, fainv = np.power(2.*a, p) / p, 1. / p
            pre_beta_2_ainv[p,q] = float((fainv * np.power(b, q) * mpmath.hyp2f1(p+q, 1, p+1, 1/a)
                                           - f2 * np.power(a, q) * mpmath.hyp2f1(p+q, 1, p+1, 2+1e-7)).real)
            # B(a, b; v+p, q)
            pre_beta_2_ainv[M+p,q] = beta_2_ainv(v+p, q)
            # B(a, b; p, v+q)
            pre_beta_2_ainv[p,M+q] = beta_2_ainv(p, v+q)
            # B(a, b; v+p, v+q)
            pre_beta_2_ainv[M+p,M+q] = beta_2_ainv(v+p, v+q)
        for q in range(1, 3):
            # B(a, b; p, 2*v+q)
            pre_beta_2_ainv[p,2*M+q] = beta_2_ainv(p, 2*v+q)
            # B(a, b; v+p, 2*v+q)
            pre_beta_2_ainv[M+p,2*M+q] = beta_2_ainv(v+p, 2*v+q)
    for p in range(1, 3):
        for q in range(1, min(max_power+1, cut+1-p)):
            # B(a, b; 2*v+p, q)
            pre_beta_2_ainv[2*M+p,q] = beta_2_ainv(2*v+p, q)
            # B(a, b; 2*v+p, v+q)
            pre_beta_2_ainv[2*M+p,M+q] = beta_2_ainv(2*v+p, v+q)
        for q in range(1, 3):
            # B(a, b; 2*v+p, 2*v+q)
            pre_beta_2_ainv[2*M+p,2*M+q] = beta_2_ainv(2*v+p, 2*v+q)

    '''
    # Same betas with negative powers in
    for p in range(1, min(max_power+1, cut)):
        for q in range(1, min(max_power+1, cut+1-p)):
            # B(a, b; p, q)
            pre_beta_2_ainv[p,q] = float((np.power(a, p+q) * np.power(-1+0j, q) * mpmath.betainc(p, q, x1=2+1e-6, x2=1/a)).real)
            # B(a, b; v+p, q)
            pre_beta_2_ainv[M+p,q] = float((np.power(a, v+p+q) * np.power(-1+0j, q) * mpmath.betainc(v+p, q, x1=2, x2=1/a)).real)
            # B(a, b; p, v+q)
            pre_beta_2_ainv[p,M+q] = float((np.power(a, p+v+q) * np.power(-1+0j, v+q) * mpmath.betainc(p, v+q, x1=2, x2=1/a)).real)
            # B(a, b; v+p, v+q)
            pre_beta_2_ainv[M+p,M+q] = float((np.power(a, v+p+v+q) * np.power(-1+0j, v+q) * mpmath.betainc(v+p, v+q, x1=2, x2=1/a)).real)
        for q in range(1, 3):
            # B(a, b; p, 2*v+q)
            pre_beta_2_ainv[p,2*M+q] = float((np.power(a, p+2*v+q) * np.power(-1+0j, 2*v+q) * mpmath.betainc(p, 2*v+q, x1=2, x2=1/a)).real)
            # B(a, b; v+p, 2*v+q)
            pre_beta_2_ainv[M+p,2*M+q] = float((np.power(a, v+p+2*v+q) * np.power(-1+0j, 2*v+q) * mpmath.betainc(v+p, 2*v+q, x1=2, x2=1/a)).real)
    for p in range(1, 3):
        for q in range(1, min(max_power+1, cut+1-p)):
            # B(a, b; 2*v+p, q)
            pre_beta_2_ainv[2*M+p,q] = float((np.power(a, 2*v+p+q) * np.power(-1+0j, q) * mpmath.betainc(2*v+p, q, x1=2, x2=1/a)).real)
            # B(a, b; 2*v+p, v+q)
            pre_beta_2_ainv[2*M+p,M+q] = float((np.power(a, 2*v+p+v+q) * np.power(-1+0j, v+q) * mpmath.betainc(2*v+p, v+q, x1=2, x2=1/a)).real)
        for q in range(1, 3):
            # B(a, b; 2*v+p, 2*v+q)
            pre_beta_2_ainv[2*M+p,2*M+q] = float((np.power(a, 2*v+p+2*v+q) * np.power(-1+0j, 2*v+q) * mpmath.betainc(2*v+p, 2*v+q, x1=2, x2=1/a)).real)
    '''

    #print("B(a,b)=", pre_beta_a_b)
    #print("B(2,1/a)=", pre_beta_2_ainv)
    #print(pre_beta_2_ainv.shape)

    print("Betas precomputed")

    num_polys = len(ps)
    cross_prod = np.zeros((num_polys, num_polys))

    for n in range(num_polys):
        p1, q1, r1 = ps[n], qs[n], rs[n]
        p2, q2, r2 = ps[:n+1], qs[:n+1], rs[:n+1]
        perms = [(p1+p2, q1+q2, r1+r2), (p1+p2, q1+r2, r1+q2),
                 (p1+q2, q1+p2, r1+r2), (p1+q2, q1+r2, r1+p2),
                 (p1+r2, q1+p2, r1+q2), (p1+r2, q1+q2, r1+p2)]
                    
        # Indices array are used for referencing precomputed betas
        pi1, qi1, ri1 = pinds[n], qinds[n], rinds[n]
        pi2, qi2, ri2 = pinds[:n+1], qinds[:n+1], rinds[:n+1]
        iperms = [(pi1+pi2, qi1+qi2, ri1+ri2), (pi1+pi2, qi1+ri2, ri1+qi2),
                 (pi1+qi2, qi1+pi2, ri1+ri2), (pi1+qi2, qi1+ri2, ri1+pi2),
                 (pi1+ri2, qi1+pi2, ri1+qi2), (pi1+ri2, qi1+qi2, ri1+pi2)]

        for perm in range(len(perms)):
            p, q, r = perms[perm]
            pi, qi, ri = iperms[perm]

            # alpha^(p+1)
            ap1, aq1, ar1 = a ** (p+1), a ** (q+1), a ** (r+1)
            # beta^(p+1)
            bp1, bq1, br1 = b ** (p+1), b ** (q+1), b ** (r+1)

            # Integral over the cube [alpha,1]^3
            I_cube = (1-ap1) * (1-aq1) * (1-ar1) / ((p+1.) * (q+1.) * (r+1.))

            # Integral over the volume inside the cube but outside the tetrapyd,
            # where x >= y+z
            I_x = pre_beta_a_b[qi+1,ri+1] / (p+q+r+3.) / (q+r+2.)
            I_x -= (((r+1.) * ar1 * bq1 / (q+r+2.)) + ((q+1.) * aq1 * br1 / (q+r+2.)) - aq1 * ar1) / ((p+1.) * (q+1.) * (r+1.))
            #I_x += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[p+2,q+1] + (-1) * pre_beta_2_ainv[p+2,r+1]) / ((p+1.) * (p+q+r+3.))
            I_x -= (ar1 * pre_beta_2_ainv[pi+2,qi+1] + aq1 * pre_beta_2_ainv[pi+2,ri+1]) / (a * (p+1.) * (p+q+r+3.))

            # Same for y >= z+x
            I_y = pre_beta_a_b[ri+1,pi+1] / (p+q+r+3.) / (r+p+2.)
            I_y -= (((p+1.) * ap1 * br1 / (r+p+2.)) + ((r+1.) * ar1 * bp1 / (r+p+2.)) - ar1 * ap1) / ((p+1.) * (q+1.) * (r+1.))
            #I_y += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[q+2,r+1] + (-1) * pre_beta_2_ainv[q+2,p+1]) / ((q+1.) * (p+q+r+3.))
            I_y -= (ap1 * pre_beta_2_ainv[qi+2,ri+1] + ar1 * pre_beta_2_ainv[qi+2,pi+1]) / (a * (q+1.) * (p+q+r+3.))

            # Same for z >= x_y
            I_z = pre_beta_a_b[pi+1,qi+1] / (p+q+r+3.) / (p+q+2.)
            I_z -= (((q+1.) * aq1 * bp1 / (p+q+2.)) + ((p+1.) * ap1 * bq1 / (p+q+2.)) - ap1 * aq1) / ((p+1.) * (q+1.) * (r+1.))
            #I_z += a**(p+q+r+3) * ((-1) * pre_beta_2_ainv[r+2,p+1] + (-1) * pre_beta_2_ainv[r+2,q+1]) / ((r+1.) * (p+q+r+3.))
            I_z -= (aq1 * pre_beta_2_ainv[ri+2,pi+1] + ap1 * pre_beta_2_ainv[ri+2,qi+1]) / (a * (r+1.) * (p+q+r+3.))
            
            #print(f"I_cube({pi},{qi},{ri})={I_cube}")
            #print(f"I_x({pi},{qi},{ri})={I_x}")

            cross_prod[n,:n+1] += (I_cube - I_x - I_y - I_z) / 6

    # Symmetrise
    cross_prod += cross_prod.T - np.diag(np.diag(cross_prod))

    return cross_prod


def grid_poly_evaluations(grid_1d, ps, qs, rs, i1, i2, i3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = (x^p * y^q + z^r  + (5 syms)) / 6.0
    at the grid points.
    '''

    eval_p = np.power(grid_1d[np.newaxis,:], ps[:,np.newaxis])
    eval_q = np.power(grid_1d[np.newaxis,:], qs[:,np.newaxis])
    eval_r = np.power(grid_1d[np.newaxis,:], rs[:,np.newaxis])

    grid_evals = (eval_p[:,i1] * eval_q[:,i2] * eval_r[:,i3]
                    + eval_p[:,i1] * eval_q[:,i3] * eval_r[:,i2]
                    + eval_p[:,i2] * eval_q[:,i1] * eval_r[:,i3]
                    + eval_p[:,i2] * eval_q[:,i3] * eval_r[:,i1]
                    + eval_p[:,i3] * eval_q[:,i1] * eval_r[:,i2]
                    + eval_p[:,i3] * eval_q[:,i2] * eval_r[:,i1]) / 6.0

    return grid_evals

def poly_evaluations(p, q, r, k1, k2, k3):
    ''' Computes the matrix containing the evaluations of
    f(k1,k2,k3) = (k1^p * k2^q + k3^r  + (5 syms)) / 6.0
    '''

    evals = (k1 ** p * k2 ** q * k3 ** r
           + k1 ** p * k2 ** r * k3 ** q
           + k1 ** q * k2 ** p * k3 ** r
           + k1 ** q * k2 ** r * k3 ** p
           + k1 ** r * k2 ** p * k3 ** q
           + k1 ** r * k2 ** q * k3 ** p) / 6

    return evals


def analytic_sine_integrals(omegas, phis, alpha):
    ''' Analytic values for the integrals of
    f(x,y,z) = sin(omega * (x + y + z) + phi)
    over the tetrapyd.
    '''

    fact =  1 / (4 * omegas ** 3)
    evals = (9 * np.cos(phis + 2 * omegas) + 4 * np.cos(phis + 3 * omegas)
             -4 * np.cos(phis + 3 * alpha * omegas) + 3 * np.cos(phis + 4 * alpha * omegas)
             -12 * np.cos(phis + (2 + alpha) * omegas) + 6 * omegas * np.sin(phis + 2 * omegas)
             -12 * alpha * omegas * np.sin(phis + 2 * omegas))
    evals *= fact

    return evals


def grid_sine_evaluations(grid_1d, omegas, phis, i1, i2, i3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = sin(omega * (x + y + z) + phi)
    at the grid points.
    '''

    eval_K = grid_1d[i1] + grid_1d[i2] + grid_1d[i3]
    grid_evals = np.sin(omegas[:,np.newaxis] * eval_K[np.newaxis,:] + phis[:,np.newaxis])

    return grid_evals


def sine_evaluations(omega, phi, k1, k2, k3):
    ''' Computes the matrix containing the evaluations of
    f(x,y,z) = sin(omega * (k1 * k2 * k3) + phi)
    '''

    evals = np.sin(omega * (k1 + k2 + k3) + phi)

    return evals


def orthonormal_polynomials(ps, qs, rs, alpha, negative_power=None):
    ''' Orthonormalise the polynomials x^p y^q r^z on the unit tetrapyd (with alpha < 1).
        Uses modified Gram-Schmidt orthogonalisation based on the analytic integral values.
        Returns a lower-triangluer matrix L such that the nth row specifies
        the orthgonal coefficients for the nth polynomial with (p,q,r) = (ps[n], qs[n], rs[n]).
    '''

    num_polys = ps.shape[0]

    # Cross inner product between polynomials
    if True:
        # New routine, should be more efficient
        if negative_power is None:
            I = analytic_poly_cross_product_alpha(ps, qs, rs, alpha)
        else:
            I = analytic_poly_cross_product_alpha_ns(ps, qs, rs, alpha, negative_power)


    else:
        # Legacy routine, not used for now.
        I = np.zeros((num_polys, num_polys))

        for n in range(num_polys):
            p1, q1, r1 = ps[n], qs[n], rs[n]
            p2, q2, r2 = ps[:n+1], qs[:n+1], rs[:n+1]
            I[n,:n+1] = (analytic_poly_integrals_alpha(p1+p2, q1+q2, r1+r2, alpha)
                            + analytic_poly_integrals_alpha(p1+p2, q1+r2, r1+q2, alpha)
                            + analytic_poly_integrals_alpha(p1+q2, q1+p2, r1+r2, alpha)
                            + analytic_poly_integrals_alpha(p1+q2, q1+r2, r1+p2, alpha)
                            + analytic_poly_integrals_alpha(p1+r2, q1+p2, r1+q2, alpha)
                            + analytic_poly_integrals_alpha(p1+r2, q1+q2, r1+p2, alpha)
                                ) / 6
        
        print("I=", I)
        print("ps=", ps, "qs=", qs, "rs=", rs)
        
        I += I.T - np.diag(np.diag(I))   # Symmetrise
    print("I=", I)

    C = np.zeros((num_polys, num_polys))
    N = np.zeros(num_polys)
    C[0,0] = 1
    N[0] = I[0,0] ** (-0.5)

    for n in range(num_polys):
        v = N[:n] ** 2 * np.matmul(C[:n,:n], I[n,:n])
        C[n,:n] = -np.matmul(v[:], C[:n,:n])
        C[n,n] = 1

        #N[n] = np.dot(I[n,:n+1], C[n,:n+1]) ** (-0.5)      # Faster but less accurate
        N[n] = np.dot(C[n,:n+1], np.matmul(I[:n+1,:n+1], C[n,:n+1])) ** (-0.5)      # Slower but more accurate

    L = C[:,:] * N[:,np.newaxis]

    return L
    
    
