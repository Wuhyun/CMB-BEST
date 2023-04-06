/* James' routine "calculate_volume.c" for computing weights of grid points in tetrapyd */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
// #include "util.h"
#include "arrays.h"

/* generate a random floating point number from min to max */
double rand_from(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void monte_carlo_tetrapyd_weights(double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts, double *k_grid, double *k_weights, int k_npts, int n_samples) {
    /* Compute tetrapyd weights using Monte Carlo method.
     * First, creates a 3D cube grid and calculate corresponding weights.
     * This may contain an overall 1/(x1+x2+x3) weight, suitable for bispectrum shapes.
     * Then, check if the 3D block at each given points in the grid falls into a tetrapyd.
     * If it is partially included, then use MC with n_samples to compute its weight.
	 * Point is specified by the indices tetra_i1, tetra_i2, tetra_i3
     * Returns an array of length tetra_npts containing the tetrapyd weights for the */

    /* Initialise RNG */
    srand((unsigned int) time(NULL));

	/* 1D bounds and weights for k grid points */
    double *int_bd = create_array(k_npts + 1);
    int_bd[0] = k_grid[0];
    for (int i = 1; i < k_npts; i++) {
        int_bd[i] = (k_grid[i-1] + k_grid[i]) / 2;
    }
    int_bd[k_npts] = k_grid[k_npts-1];

    /* Initialise weights */
    #pragma omp parallel for
    for (int t = 0; t < tetra_npts; t++) {
        int i1 = tetra_i1[t];
        int i2 = tetra_i2[t];
        int i3 = tetra_i3[t];
		tetra_weights[t] = k_weights[i1] * k_weights[i2] * k_weights[i3];
		// tetra_weights[t] = k_weights[i1] * k_weights[i2] * k_weights[i3] / (k_grid[i1] * k_grid[i2] * k_grid[i3]);

        /* Symmetry factor */
        double sym;
        // if ((i1 == i2) && (i2 == i3)) sym = 1;
        // else if ((i1 == i2) || (i2 == i3) || (i3 == i1)) sym = 3;
        // else sym = 6;
		if ((i1 != i2) && (i2 != i3)) {
			sym = 6;
		} else if ((i1 != i2) && (i2 == i3)) {
			sym = 3;
		} else if ((i1 == i2) && (i2 != i3)) {
			sym = 3;
		} else {
			sym = 1;
		}
//		printf("(%d,%d,%d)\n", i1, i2, i3);


        tetra_weights[t] *= sym;
    }

    /* Monte Carlo computation */
    #pragma omp parallel for
    for (int t = 0; t < tetra_npts; t++) {
        int i1 = tetra_i1[t];
        int i2 = tetra_i2[t];
        int i3 = tetra_i3[t];

        double x_l = int_bd[i1];     // lower bound for x
        double x_u = int_bd[i1+1];   // upper bound for x
        double y_l = int_bd[i2];
        double y_u = int_bd[i2+1];
        double z_l = int_bd[i3];
        double z_u = int_bd[i3+1];


        if (((x_l + y_l - z_u < 0) && (x_u + y_u - z_l > 0))		// The plane x+y-z = 0 intersects
			|| ((y_l + z_l - x_u < 0) && (y_u + z_u - x_l > 0))		// The plane y+z-x = 0 intersects
			|| ((z_l + x_l - y_u < 0) && (z_u + x_u - y_l > 0))) {	// The plane y+z-x = 0 intersects
			// Perform MC
            int cnt = 0;
            for (int n = 0; n < n_samples; n++) {
                double x_r = rand_from(x_l, x_u);
                double y_r = rand_from(y_l, y_u);
                double z_r = rand_from(z_l, z_u);

                if ((x_r + y_r >= z_r) && (y_r + z_r >= x_r) && (z_r + x_r >= y_r)) {
                    // (x_r, y_r, z_r) is in the tetrapyd
                    cnt += 1;
                }
            }
            tetra_weights[t] *= 1.0 * cnt / n_samples;
		}
    }

	#pragma omp parallel
	{
		#pragma omp single
		printf("Number of threads: %d\n", omp_get_num_threads());
	}

    free_array(int_bd);
}


void compute_mode_bispectra_covariance(double *mode_bispectra_covariance, 
							double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
							int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
							double *mode_evals, int mode_p_max, int k_npts) {
	/* Dimensions:
	 * bispectra_covariance: (n_modes, n_modes) flattened
	 * tetra_weights: 		 (tetra_npts)
	 * tetra_i1, 2 3:		 (tetra_npts)
	 * mode_evaluations:	 (mode_p_max, k_npts) flattened
	 * Note that the flattened arrays are assumed to have contiguous memory allocations.
	 * */
	const int N_K = k_npts;
	const int N_MODES = n_modes;
	
	#pragma omp parallel
	{
		double *Q_vec = create_array(tetra_npts);

		#pragma omp for
		for (int n = 0; n < n_modes; n++) {
			int p1 = mode_p1[n];
			int p2 = mode_p2[n];
			int p3 = mode_p3[n];

			/* Evaluate one of the Q(k1,k2,k3)s first */
			for (int t = 0; t < tetra_npts; t++) {
				int i1 = tetra_i1[t];
				int i2 = tetra_i2[t];
				int i3 = tetra_i3[t];

				double mode_11 = mode_evals[p1 * N_K + i1];
				double mode_12 = mode_evals[p1 * N_K + i2];
				double mode_13 = mode_evals[p1 * N_K + i3];

				double mode_21 = mode_evals[p2 * N_K + i1];
				double mode_22 = mode_evals[p2 * N_K + i2];
				double mode_23 = mode_evals[p2 * N_K + i3];

				double mode_31 = mode_evals[p3 * N_K + i1];
				double mode_32 = mode_evals[p3 * N_K + i2];
				double mode_33 = mode_evals[p3 * N_K + i3];

				double Q_eval = ( mode_11 * mode_22 * mode_33 + mode_11 * mode_23 * mode_32 
								+ mode_12 * mode_21 * mode_33 + mode_12 * mode_23 * mode_31
								+ mode_13 * mode_21 * mode_32 + mode_13 * mode_22 * mode_31) / 6.0;
				
				Q_vec[t] = tetra_weights[t] * Q_eval;
			}

			/* Now loop over the second Q and compute the inner product */
			for (int m = 0; m < n_modes; m++) {
				int q1 = mode_p1[m];
				int q2 = mode_p2[m];
				int q3 = mode_p3[m];

				double sum = 0.0;
				for (int t = 0; t < tetra_npts; t++) {
					int j1 = tetra_i1[t];
					int j2 = tetra_i2[t];
					int j3 = tetra_i3[t];

					double q_11 = mode_evals[q1 * N_K + j1];
					double q_12 = mode_evals[q1 * N_K + j2];
					double q_13 = mode_evals[q1 * N_K + j3];

					double q_21 = mode_evals[q2 * N_K + j1];
					double q_22 = mode_evals[q2 * N_K + j2];
					double q_23 = mode_evals[q2 * N_K + j3];

					double q_31 = mode_evals[q3 * N_K + j1];
					double q_32 = mode_evals[q3 * N_K + j2];
					double q_33 = mode_evals[q3 * N_K + j3];

					double Q2_eval = (q_11 * q_22 * q_33 + q_11 * q_23 * q_32 
									+ q_12 * q_21 * q_33 + q_12 * q_23 * q_31
									+ q_13 * q_21 * q_32 + q_13 * q_22 * q_31) / 6.0;
					
					sum += Q2_eval * Q_vec[t];
				}
				// Below should be thread-safe since n is distinct across threads
				mode_bispectra_covariance[n * N_MODES + m] = sum;
			}
		}

		free_array(Q_vec);
	}

	/*
	for (int n = 0; n < 10; n++) {
		for (int m = 0; m < 10; m++) {
			printf("%.10e ", mode_bispectra_covariance[n * N_MODES + m]);
		}
		printf("\n");
	}
	*/

	return;
}

void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts) {

	/* Dimensions:
	 * QS: (n_shapes, n_modes) flattened
	 # S: (n_shapes, tetra_npts) flattened
	 * tetra_weights: 		 (tetra_npts)
	 * tetra_i1, 2 3:		 (tetra_npts)
	 * mode_evaluations:	 (mode_p_max, k_npts) flattened
	 * Note that the flattened arrays are assumed to have contiguous memory allocations.
	 * */


	const int N_T = tetra_npts;
	const int N_K = k_npts;
	const int N_MODES = n_modes;
	
	#pragma omp parallel
	{
		double *Q_vec = create_array(tetra_npts);

		#pragma omp for
		for (int n = 0; n < n_modes; n++) {
			int p1 = mode_p1[n];
			int p2 = mode_p2[n];
			int p3 = mode_p3[n];

			/* Evaluate the mode bispectra Q_n(k1,k2,k3) */
			for (int t = 0; t < tetra_npts; t++) {
				int i1 = tetra_i1[t];
				int i2 = tetra_i2[t];
				int i3 = tetra_i3[t];

				double mode_11 = mode_evals[p1 * N_K + i1];
				double mode_12 = mode_evals[p1 * N_K + i2];
				double mode_13 = mode_evals[p1 * N_K + i3];

				double mode_21 = mode_evals[p2 * N_K + i1];
				double mode_22 = mode_evals[p2 * N_K + i2];
				double mode_23 = mode_evals[p2 * N_K + i3];

				double mode_31 = mode_evals[p3 * N_K + i1];
				double mode_32 = mode_evals[p3 * N_K + i2];
				double mode_33 = mode_evals[p3 * N_K + i3];

				double Q_eval = ( mode_11 * mode_22 * mode_33 + mode_11 * mode_23 * mode_32 
								+ mode_12 * mode_21 * mode_33 + mode_12 * mode_23 * mode_31
								+ mode_13 * mode_21 * mode_32 + mode_13 * mode_22 * mode_31) / 6.0;
				
				Q_vec[t] = tetra_weights[t] * Q_eval;
			}

			/* Now loop over S and compute the inner product */
			for (int s = 0; s < n_shapes; s++) {

				double sum = 0.0;
				for (int t = 0; t < tetra_npts; t++) {
					sum += S[s * N_T + t] * Q_vec[t];
				}
				// Below should be thread-safe since n is distinct across threads
				QS[s * N_MODES + n] = sum;
			}
		}

		free_array(Q_vec);
	}

	return;

}

#if 0
void inverse_cholesky_WS(double *result, double *M, int n) {
    /* Compute cholesky decomposition of the inverse of symmetric, positive-definite matrix M
    using a Modified Gram-Schmidt process.
    Computes a lower triangular matrix L such that C^{-1} = L^T L
	*/

	const int N = n;

	for (int i = 0; i < n; i++) {

	}
}
#endif

#if 0

int **vectorise_tetrapyd_k_indices(double *k_grid, int npts_k, int *npts_tetra) {
    /* Returns (npts_tetra) X 3 array containing discrete inidicies
     * (i, j, k), npts_k > i >= j >= k >= 0, where
     * (k(i), k(j), k(k)) of each point within tetrapyd. I.e.,
     * k_max >= k(i) >= k(j) >= k(k) >= k_min, and (k(i), k(j), k(k)) form a triangle. */

    npts_tetra[0] = (npts_k + 2) * (npts_k + 1) * npts_k / 6;    // Rough bound for now
    int **tetra_inds = create_2D_int_array(npts_tetra[0], 3);
    int cnt = 0;

    for (int x = 0; x < npts_k; x++) {
        for (int y = 0; y <= x; y++) {
            for (int z = 0; z <= y; z++) {
                if (k_grid[x] <= k_grid[y] + k_grid[z]) {
                    tetra_inds[cnt][0] = x;
                    tetra_inds[cnt][1] = y;
                    tetra_inds[cnt][2] = z;
                    cnt++;
                }
            }
        }
    }

    npts_tetra[0] = cnt;
    return tetra_inds;
}

int **vectorise_tetrapyd_coordinates(int min, int max, int step, int *npts_tetra) {
    /* Returns (npts_tetra) X 3 array containing discrete coordinates
     * (x, y, z) of each point within tetrapyd.
     * max >= x >= y >= z >= min, and (x, y, z) form a triangle.
     * NOTE: ASSUMES that the k grid points are proportional to (x, y, z) */

    int npts = (max - min) / step;      // Number of points on each side
    npts_tetra[0] = (npts + 2) * (npts + 1) * npts / 6;    // Rough bound for now
    int **coords = create_2D_int_array(npts * npts * npts, 3);
    int cnt = 0;

    for (int i = min; i <= max; i += step) {
        for (int j = min; j <= i; j += step) {
            for (int k = min; k <= j; k += step) {
                if (i <= j + k) {
                    coords[cnt][0] = i;
                    coords[cnt][1] = j;
                    coords[cnt][2] = k;
                    cnt++;
                }
            }
        }
    }

    npts_tetra[0] = cnt;
    return coords;
}


double general_tetrapyd_weight(int min, int max, int i, int j, int k, int step, int do_sym) {
    /* Weight of the given grid point for tetrapyd integration.
     * i, j, k denote discrete coordinates for grid coordinates in x, y, z, respectively.
     * Assumes i >= j >= k and scales weights accordingly.
     * Equiavelent to tetrapyd_weight() if step = 1.
     * If do_sym = 0, do not multiply symmetry factors. */

	int l, m, n, sum;
	int i0, j0, k0;
	int grid[3][3][3];
	double pt[3][3][3];
	double weight = 0e0;

	sum = 0;
	for (l=0; l<3; l++) {
		i0 = i + step * (l - 1);
		if (i0 < min || i0 > max) {
			for (m = 0; m < 3; m++) {
				for (n = 0; n < 3; n++) {
					grid[l][m][n] = -8;
				}
			}
		} else {
			for (m = 0; m < 3; m++) {
				j0 = j + step * (m - 1);
				if (j0 < min || j0 > max) {
					for (n = 0; n < 3; n++) {
						grid[l][m][n] = -8;
					}
				} else {
					for (n = 0; n < 3; n++) {
						k0 = k + step * (n - 1);
						if (k0 < min || k0 > max) {
							grid[l][m][n] = -8;
						}else if (i0 > (j0 + k0) || j0 > (i0 + k0) || k0 > (i0 + j0)) {
							grid[l][m][n] = 0;
						} else {
							grid[l][m][n] = 1;
							sum++;
						}
					}
				}
			}
		}
	}
	
	if (sum == 27) {
		weight = 1e0;
	} else {
		
		int s1, s2, s3;
		int a, b, c, d, e, f, g, h;

		for (l = 0; l < 3; l++) {
			for (m = 0; m < 3; m++) {
				for (n = 0; n < 3; n++) {
					pt[l][m][n] = 0e0;
				}
			}
		}
		pt[1][1][1]=1e0;
		
		for (s1 = 0; s1 < 2; s1++) {
			for (s2 = 0; s2 < 2; s2++) {
				for (s3 = 0; s3 < 2; s3++) {
					
					sum=0;
					for(l=0; l<2; l++) {
						for(m=0; m<2; m++) {
							for(n=0; n<2; n++) {
								sum += grid[l+s1][m+s2][n+s3];
							}
						}
					}
					a = pt[s1][s2][s3];
					b = pt[s1+1][s2][s3];
					c = pt[s1][s2+1][s3];
					d = pt[s1+1][s2+1][s3];
					e = pt[s1][s2][s3+1];
					f = pt[s1+1][s2][s3+1];
					g = pt[s1][s2+1][s3+1];
					h = pt[s1+1][s2+1][s3+1];
					
					if (sum==4) {
		
						if (grid[1+s1][0+s2][1+s3] == 1) weight += cell2(f,e,h,g,b,a,d,c);
						if (grid[1+s1][1+s2][0+s3] == 1) weight += cell2(d,c,b,a,h,g,f,e);
						if (grid[0+s1][1+s2][1+s3] == 1) weight += cell2(g,h,e,f,c,d,a,b);
		
					} else if (sum==5) {
		
						weight += cell1(a,b,c,d,e,f,g,h);
		
					} else if (sum==6) {
		
						if (grid[0+s1][0+s2][1+s3] == 1) weight += cell3(g,h,e,f,c,d,a,b);
						if (grid[1+s1][0+s2][0+s3] == 1) weight += cell3(f,h,b,d,e,g,a,c);
						if (grid[0+s1][1+s2][0+s3] == 1) weight += cell3(d,h,c,g,b,f,a,e);
		
					} else if (sum==7) {
		
						if (grid[0+s1][1+s2][0+s3] == 0) weight += cell4(f,e,h,g,b,a,d,c);
						if (grid[0+s1][0+s2][1+s3] == 0) weight += cell4(d,c,b,a,h,g,f,e);
						if (grid[1+s1][0+s2][0+s3] == 0) weight += cell4(g,h,e,f,c,d,a,b);
		
					} else if (sum==8) {
	
						weight += 1e0/8e0;
		
					}
					
				}
			}
		}
	}
	
    if (do_sym) {
        if(i!=k){
            if(i==j || j==k){
                weight *= 3e0;
            }else{
                weight *= 6e0;
            }
        }
    }

    weight *= step * step * step;
 	return weight;

}

double tetrapyd_weight(int min, int max, int i, int j, int k) {

	int l,m,n,sum;
	int i0,j0,k0;
	int grid[3][3][3];
	double pt[3][3][3];
	double weight = 0e0;
	
	sum = 0;
	for(l=0; l<3; l++) {
		i0 = i+l-1;
		if(i0<min || i0>max){
			for(m=0; m<3; m++) {
				for(n=0; n<3; n++) {
					grid[l][m][n] = -8;
				}
			}
		}else{
			for(m=0; m<3; m++) {
				j0 = j+m-1;
				if(j0<min || j0>max){
					for(n=0; n<3; n++) {
						grid[l][m][n] = -8;
					}
				}else{
					for(n=0; n<3; n++) {
						k0 = k+n-1;
						if(k0<min || k0>max){
							grid[l][m][n] = -8;
						}else if(i0>(j0+k0) || j0>(i0+k0) || k0>(i0+j0)){
							grid[l][m][n] = 0;
						} else {
							grid[l][m][n] = 1;
							sum++;
						}
					}
				}
			}
		}
	}
	
	if(sum==27){
		weight = 1e0;
	}else{
		
		int s1,s2,s3;
		int a,b,c,d,e,f,g,h;

		for(l=0; l<3; l++) {
			for(m=0; m<3; m++) {
				for(n=0; n<3; n++) {
					pt[l][m][n] = 0e0;
				}
			}
		}
		pt[1][1][1]=1e0;
		
		for(s1=0; s1<2; s1++) {
			for(s2=0; s2<2; s2++) {
				for(s3=0; s3<2; s3++) {
					
					sum=0;
					for(l=0; l<2; l++) {
						for(m=0; m<2; m++) {
							for(n=0; n<2; n++) {
								sum += grid[l+s1][m+s2][n+s3];
							}
						}
					}
					a = pt[s1][s2][s3];
					b = pt[s1+1][s2][s3];
					c = pt[s1][s2+1][s3];
					d = pt[s1+1][s2+1][s3];
					e = pt[s1][s2][s3+1];
					f = pt[s1+1][s2][s3+1];
					g = pt[s1][s2+1][s3+1];
					h = pt[s1+1][s2+1][s3+1];
					
					if (sum==4) {
		
						if (grid[1+s1][0+s2][1+s3] == 1) weight += cell2(f,e,h,g,b,a,d,c);
						if (grid[1+s1][1+s2][0+s3] == 1) weight += cell2(d,c,b,a,h,g,f,e);
						if (grid[0+s1][1+s2][1+s3] == 1) weight += cell2(g,h,e,f,c,d,a,b);
		
					} else if (sum==5) {
		
						weight += cell1(a,b,c,d,e,f,g,h);
		
					} else if (sum==6) {
		
						if (grid[0+s1][0+s2][1+s3] == 1) weight += cell3(g,h,e,f,c,d,a,b);
						if (grid[1+s1][0+s2][0+s3] == 1) weight += cell3(f,h,b,d,e,g,a,c);
						if (grid[0+s1][1+s2][0+s3] == 1) weight += cell3(d,h,c,g,b,f,a,e);
		
					} else if (sum==7) {
		
						if (grid[0+s1][1+s2][0+s3] == 0) weight += cell4(f,e,h,g,b,a,d,c);
						if (grid[0+s1][0+s2][1+s3] == 0) weight += cell4(d,c,b,a,h,g,f,e);
						if (grid[1+s1][0+s2][0+s3] == 0) weight += cell4(g,h,e,f,c,d,a,b);
		
					} else if (sum==8) {
	
						weight += 1e0/8e0;
		
					}
					
				}
			}
		}
	}
	
	if(i!=k){
		if(i==j || j==k){
			weight *= 3e0;
		}else{
			weight *= 6e0;
		}
	}
 	return weight;

}

double tetrapyd_volume(int i, int j, int k, double ***points) {

	double a = points[0][0][0];
	double b = points[1][0][0];
	double c = points[0][1][0];
	double d = points[1][1][0];
	double e = points[0][0][1];
	double f = points[1][0][1];
	double g = points[0][1][1];
	double h = points[1][1][1];

	int l,m,n,sum=0;
	int grid[2][2][2];
	double volume;
	
	for(l=0; l<2; l++) {
		for(m=0; m<2; m++) {
			for(n=0; n<2; n++) {
				if ( (i+l)>(j+k+m+n) || (j+m)>(i+k+l+n) || (k+n)>(i+j+l+m) ){
					grid[l][m][n] = 0;
				} else {
					grid[l][m][n] = 1;
					sum++;
				}
			}
		}
	}
	
	if (sum==4) {
		
		if (grid[1][0][1] == 1) volume = cell2(f,e,h,g,b,a,d,c);
		if (grid[1][1][0] == 1) volume = cell2(d,c,b,a,h,g,f,e);
		if (grid[0][1][1] == 1) volume = cell2(g,h,e,f,c,d,a,b);
		
	} else if (sum==5) {
		
		volume = cell1(a,b,c,d,e,f,g,h);
		
	} else if (sum==6) {
		
		if (grid[0][0][1] == 1) volume = cell3(g,h,e,f,c,d,a,b);
		if (grid[1][0][0] == 1) volume = cell3(f,h,b,d,e,g,a,c);
		if (grid[0][1][0] == 1) volume = cell3(d,h,c,g,b,f,a,e);
		
	} else if (sum==7) {
		
		if (grid[0][1][0] == 0) volume = cell4(f,e,h,g,b,a,d,c);
		if (grid[0][0][1] == 0) volume = cell4(d,c,b,a,h,g,f,e);
		if (grid[1][0][0] == 0) volume = cell4(g,h,e,f,c,d,a,b);
		
	} else if (sum==8) {
	
		volume = cell5(a,b,c,d,e,f,g,h);
		
	} else {
		volume = 0;
	}

 	return volume;

}

double cell1(double a, double b, double c, double d, double e, double f, double g, double h){

// | a   |  |   f |
// |   d |  | g h |

	double result = (11*(a+b+c+e)+17*(d+g+f)+25*h)/240.0;
	return result;

}

double cell2(double a, double b, double c, double d, double e, double f, double g, double h){

// | a b |  | e   |
// | c   |  |     |

	double result = (47*a+19*(b+c+e)+5*(d+f+g)+h)/720.0;
	return result;

}
double cell3(double a, double b, double c, double d, double e, double f, double g, double h){

// | a b |  |   f |
// | c d |  | g   |

	double result = (40*(b+c)+35*(a+d)+26*(f+g)+19*(e+h))/360.0;
	return result;

}
double cell4(double a, double b, double c, double d, double e, double f, double g, double h){

// | a b |  | e f |
// | c d |  | g   |

	double result = (89*a+85*(b+c+e)+71*(d+f+g)+43*h)/720.0;
	return result;

}
double cell5(double a, double b, double c, double d, double e, double f, double g, double h){

// | a b |  | e f |
// | c d |  | g h |

	double result = (a+b+c+d+e+f+g+h)/8.0;
	return result;
	
}

#endif
