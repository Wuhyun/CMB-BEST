/* Routine for computing weights of grid points in tetrapyd */

#ifndef TETRAPYD_H
#define TETRAPYD_H

//int **vectorise_tetrapyd_k_indices(double *k_grid, int npts_k, int *npts_tetra);
//int **vectorise_tetrapyd_coordinates(int min, int max, int step, int *npts_tetra);

void monte_carlo_tetrapyd_weights(double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts, double *k_grid, double *k_weights, int k_npts, int n_samples);
void compute_mode_bispectra_covariance(double *mode_bispectra_covariance, 
							double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
							int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
							double *mode_evals, int mode_p_max, int k_npts);
void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts);


#endif /* TETRAPYD_H */
