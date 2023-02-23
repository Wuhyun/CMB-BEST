#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

const int PADDING_SIZE = 8;

double *create_array(int d1){
	double *arr = (double *) malloc(d1 * sizeof(double));
	if (arr == NULL){
		printf("Insuffcient memory for creating a %d array\n", d1);
		return NULL;
	}
	return arr;
}

double **create_2D_array(int d1, int d2){
	double **arr = (double **) malloc(d1 * sizeof(double *));
	arr[0] = (double *) malloc(d1 * d2 * sizeof(double));
	if (arr[0] == NULL){
		printf("Insuffcient memory for creating a %d X %d array\n", d1, d2);
		return NULL;
	}
	for (int i = 1; i < d1; i++) {
		arr[i] = arr[i-1] + d2;
	}
	return arr;
}

double ***create_3D_array(int d1, int d2, int d3){
	double ***arr = (double ***) malloc(d1 * sizeof(double **));
	arr[0] = (double **) malloc(d1 * d2 * sizeof(double *));
	arr[0][0] = (double *) malloc(d1 * d2 * d3 * sizeof(double));
	if (arr[0][0] == NULL){
		printf("Insufficient memory for creating a %d X %d X %d array\n", d1, d2, d3);
		return NULL;
	}
	for (int i = 1; i < d1; i++) {
		arr[i] = arr[i-1] + d2;
	}
	int n = 0;
	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d2; j++){
			arr[i][j] = arr[0][0] + d3 * n;
			n++;
		}
	}
	return arr;
}

double **create_2D_array_from_data(double *data, int d1, int d2){
	// Create a 2D array from existing data, assumed to have at least d1 * d2 elements
	double **arr = (double **) malloc(d1 * sizeof(double *));
	arr[0] = data;

	for (int i = 1; i < d1; i++) {
		arr[i] = arr[i-1] + d2;
	}
	return arr;
}

double ***create_3D_array_from_data(double *data, int d1, int d2, int d3){
	// Create a 3D array from existing data, assumed to have at least d1 * d2 * d3 elements
	double ***arr = (double ***) malloc(d1 * sizeof(double **));
	arr[0] = (double **) malloc(d1 * d2 * sizeof(double *));
	arr[0][0] = data;

	for (int i = 1; i < d1; i++) {
		arr[i] = arr[i-1] + d2;
	}
	int n = 0;
	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d2; j++){
			arr[i][j] = arr[0][0] + d3 * n;
			n++;
		}
	}
	return arr;
}


void free_array(double *arr){
	free(arr);
}

void free_2D_array(double **arr){
	free(arr[0]);
	free(arr);
}

void free_3D_array(double ***arr){
    free(arr[0][0]);
	free(arr[0]);
	free(arr);
}

void free_2D_array_from_data(double **arr){
	free(arr);
}

void free_3D_array_from_data(double ***arr){
	free(arr[0]);
	free(arr);
}

#if 0

void free_aligned_array(double *arr){
    /* Aligned methods from ICC have different methods */
#if defined(__INTEL_COMPILER)
    _mm_free(arr);
#elif defined(__GNUC__)    
    free(arr);
#else
#endif /* Compilers */
}

void free_aligned_2D_array(double **arr){

	free_aligned_array(arr[0]);

#if defined(__INTEL_COMPILER)
    _mm_free(arr);
#elif defined(__GNUC__)    
    free(arr);
#else
#endif /* Compilers */

}

void free_aligned_3D_array(double ***arr){

	free_aligned_2D_array(arr[0]);

#if defined(__INTEL_COMPILER)
    _mm_free(arr);
#elif defined(__GNUC__)    
    free(arr);
#else
#endif /* Compilers */

}

void free_pointer_array(double **arr){
    free(arr);
}

void free_2D_pointer_array(double ***arr){
    free(arr[0]);
    free(arr);
}

void free_aligned_2D_complex_array(complex double **arr){

#if defined(__INTEL_COMPILER)
    _mm_free(arr[0]);
    _mm_free(arr);
#elif defined(__GNUC__)    
    free(arr[0]);
    free(arr);
#else
#endif /* Compilers */

}


int *create_int_array(int d1){
    int *arr = (int *) malloc(d1 * sizeof(int));
    if(arr == NULL){
		printf("Insuffcient memory for creating a %d array\n", d1);
		return;
    }
    return arr;
}


int **create_2D_int_array(int d1, int d2){
	int **arr = (int **) malloc(d1 * sizeof(int *));
	arr[0] = (int *) malloc(d1 * d2 * sizeof(int));
	if (arr[0] == NULL){
		printf("Insuffcient memory for creating a %d X %d array\n", d1, d2);
		return;
	}
	int i;
	for(i=1; i<d1; i++) arr[i] = arr[i-1] + d2;
	return arr;
}

void free_int_array(int *arr){
	free(arr);
}

void free_2D_int_array(int **arr){
	free(arr[0]);
	free(arr);
}

int **vectorise_two_indices(int size1, int size2){
	int **vec_ind = (int **) malloc(size1 * size2 * sizeof(int *));
	vec_ind[0] = (int *) malloc(2 * size1 * size2 * sizeof(int));
	int i, j, n = 0;
	for(i=0; i<size1; i++){
		for(j=0; j<size2; j++){
			vec_ind[n] = vec_ind[0] + 2*n;
			vec_ind[n][0] = i;
			vec_ind[n][1] = j;
			n++;
		}
	}
	return vec_ind;
}

int **vectorise_three_indices(int size1, int size2, int size3){
	int **vec_ind = (int **) malloc(size1 * size2 * size3 * sizeof(int *));
	vec_ind[0] = (int *) malloc(3 * size1 * size2 * size3 * sizeof(int));
	int i, j, k, n = 0;
	for(i=0; i<size1; i++){
		for(j=0; j<size2; j++){
			for(k=0; k<size3; k++){
				vec_ind[n] = vec_ind[0] + 3*n;
				vec_ind[n][0] = i;
				vec_ind[n][1] = j;
				vec_ind[n][2] = k;
				n++;
			}
		}
	}
	return vec_ind;
}

int **vectorise_three_ordered_indicies(int size) {
    /* Vectorise indicies so that size > i >= j >= k >= 0 */
    int length = (size + 2) * (size + 1) * size / 6;
	int **vec_ind = (int **) malloc(length * sizeof(int *));
	vec_ind[0] = (int *) malloc(3 * length * sizeof(int));
	int i, j, k, n = 0;
	for(i=0; i<size; i++){
		for(j=0; j<=i; j++){
			for(k=0; k<=j; k++){
				vec_ind[n] = vec_ind[0] + 3*n;
				vec_ind[n][0] = i;
				vec_ind[n][1] = j;
				vec_ind[n][2] = k;
				n++;
			}
		}
	}
	return vec_ind;
}

int vectorised_beta_indices_length(int N){
    /* Returns length of the vectorised array for tuples (a,b,c), where 0 <= a,b,c < N and a >= b */
    return N * N * (N + 1 ) / 2;
}

int vectorised_gamma_indices_length(int N){
    /* Returns length of the vectorised array for tuples (a,b,c), where 0 <= a,b,c < N and a >= b */
    return (N + 2) * (N + 1) * N / 6;
}

int **vectorised_beta_indices(int N, int *n_tuples){
    /* Returns a vectorised array for tuples (a,b,c), where 0 <= a,b,c < N and a >= b */

    *n_tuples = vectorised_beta_indices_length(N);
    int **ind = create_2D_int_array(*n_tuples, 3);

    int a, b, c;
    int n = 0;
    for(a=0; a<N; a++){
        for(b=0; b<=a; b++){
            for(c=0; c<N; c++){
                ind[n][0] = a;
                ind[n][1] = b;
                ind[n][2] = c;
                n++;
            }
        }
    }
    
    return ind;

}

int **vectorised_gamma_indices(int N, int *n_tuples){
    /* Returns a vectorised array for tuples (a,b,c), where N > a >= b >= c >= 0 */

    *n_tuples = vectorised_gamma_indices_length(N);
    int **ind = create_2D_int_array(*n_tuples, 3);

    int a, b, c;
    int n = 0;
    for(a=0; a<N; a++){
        for(b=0; b<=a; b++){
            for(c=0; c<=b; c++){
                ind[n][0] = a;
                ind[n][1] = b;
                ind[n][2] = c;
                n++;
            }
        }
    }
    
    return ind;

}

int *vectorised_beta_sym_factors(int N, int *n_tuples){
    /* Returns an array containing symmetric factors for vectorised beta indices.
     * For each tuple (a,b,c) with 0 <= a,b,c < N and a >= b, the symmetry factor equals
     * 2 if a != b
     * 1 if a == b. */

    *n_tuples = vectorised_beta_indices_length(N);
    int *sym = create_int_array(*n_tuples);

    int a, b, c;
    int n = 0;
    for(a=0; a<N; a++){
        for(b=0; b<a; b++){
            for(c=0; c<N; c++){
                sym[n] = 2;
                n++;
            }
        }
        for(c=0; c<N; c++){
            sym[n] = 1;
            n++;
        }
    }
    
    return sym;

}

int *vectorised_gamma_sym_factors(int N, int *n_tuples){
    /* Returns an array containing symmetric factors for vectorised gamma indices.
     * For each tuple (a,b,c) with N > a >= b >= c >= 0, the symmetry factor equals
     * 6 if a, b, c distinct
     * 3 if two of a, b, c identical
     * 1 if a == b == c */

    *n_tuples = vectorised_beta_indices_length(N);
    int *sym = create_int_array(*n_tuples);

    int a, b, c;
    int n = 0;
    for(a=0; a<N; a++){
        for(b=0; b<a; b++){
            for(c=0; c<b; c++){
                /* a, b, c distinct */
                sym[n] = 6;
                n++;
            }
            /* a != b == c */
            sym[n] = 3;
            n++;
        }
        for(c=0; c<b; c++){
            /* a == b != c*/
            sym[n] = 3;
            n++;
        }
        /* a == b == c */
        sym[n] = 1;
        n++;
    }

    return sym;

}
#endif
